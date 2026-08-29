// SRLC32E golden model + testbench.
//
// Xilinx 32-bit shift register LUT, per the Zenith PS section C:
//   Inputs : D (1b data), CE (1b clock enable), A[4:0] (5b dynamic read addr)
//   Edge   : on the single global rising edge, if CE==1 the 32-bit state
//            shifts left (LSB -> MSB) and D is loaded into index 0.
//   Outputs: Q   = state[A]  (combinational, dynamic address)
//            Q31 = state[31] (combinational, cascade port)
//   Init   : state = 0 (no INIT string parsing, per PS note 3).
//
// Cycle semantics matter for GEM integration. Both outputs are combinational
// reads of the CURRENT (pre-edge) state, and the shift is the only clocked
// action. So one simulated cycle is strictly two phases:
//     phase R (read)   : Q/Q31 <- srlc32e_read(state, A)      [old state]
//     phase U (update) : state  <- srlc32e_tick(state, D, CE)  [new state]
// This is exactly GEM's DFF discipline (`EndpointGroup::DFF` reads Q from the
// previous cycle's state word and commits D at the end), so an SRLC32E drops
// into the existing state-array machinery as a 32-bit-wide register endpoint.
//
// Consequence for scheduling: a Q31 -> D cascade is a REGISTER-to-register
// edge, not a combinational one. Every SRLC32E in a chain can be updated in
// the same step from the previous cycle's state; SRL chains therefore need no
// in-tree macro collapsing (unlike CARRY4, whose CO[3] -> CIN is genuinely
// combinational).
//
// Build:
//   CPU  : g++  -O2 -x c++ srlc32e_test.cu -o srlc32e_test
//   CUDA : nvcc -O2       srlc32e_test.cu -o srlc32e_test   (runs CPU tests
//                                                            then GPU x-check)

#include <iostream>
#include <cstdint>
#include <cassert>
#include <deque>
#include <vector>
#include <random>

#ifndef __CUDACC__
#define __host__
#define __device__
#define __global__
#endif

// ---------------------------------------------------------------------------
// 1. PACKED I/O ENCODING
//
// The boomerang tree hands macros loose bits; the host formatter gathers them
// into one 32-bit control word per instance so the kernel does a single
// coalesced load instead of seven scattered bit-gathers.
//
//   bit  0     : D
//   bit  1     : CE
//   bits 2..6  : A[4:0]
//
// Output word: bit 0 = Q, bit 1 = Q31.
// ---------------------------------------------------------------------------

#define SRL_IN_D_BIT    0u
#define SRL_IN_CE_BIT   1u
#define SRL_IN_A_SHIFT  2u
#define SRL_IN_A_MASK   0x1fu

#define SRL_OUT_Q_BIT   0u
#define SRL_OUT_Q31_BIT 1u

__host__ __device__ inline uint32_t srl_pack_in(bool D, bool CE, uint8_t A) {
    return ((uint32_t)(D  & 1) << SRL_IN_D_BIT)  |
           ((uint32_t)(CE & 1) << SRL_IN_CE_BIT) |
           (((uint32_t)A & SRL_IN_A_MASK) << SRL_IN_A_SHIFT);
}

struct SRLC32EOut {
    uint8_t Q   : 1;  // state[A]
    uint8_t Q31 : 1;  // state[31], cascade
};

// ---------------------------------------------------------------------------
// 2. CORE MODEL (host + device)
// ---------------------------------------------------------------------------

// Combinational read port. Pure function of the current state; no clock.
__host__ __device__ inline SRLC32EOut srlc32e_read(uint32_t sr, uint8_t A) {
    SRLC32EOut o;
    o.Q   = (uint8_t)((sr >> (A & SRL_IN_A_MASK)) & 1u);
    o.Q31 = (uint8_t)((sr >> 31) & 1u);
    return o;
}

// Clocked update. Shift LSB -> MSB, D into index 0. Held when CE == 0.
// Bit 31 falls off the end (Q31 is the cascade tap, not a wrap).
__host__ __device__ inline uint32_t srlc32e_tick(uint32_t sr, bool D, bool CE) {
    if (!CE) return sr;
    return (uint32_t)((sr << 1) | (uint32_t)(D & 1));
}

// Fused evaluate over the packed encoding: returns the packed output word and
// writes the post-edge state back through `sr`. Read happens before the shift,
// so the caller cannot accidentally observe the new state.
__host__ __device__ inline uint32_t srlc32e_eval_packed(uint32_t *sr, uint32_t in) {
    uint32_t s = *sr;

    uint8_t A  = (uint8_t)((in >> SRL_IN_A_SHIFT) & SRL_IN_A_MASK);
    bool    D  = (in >> SRL_IN_D_BIT)  & 1u;
    bool    CE = (in >> SRL_IN_CE_BIT) & 1u;

    SRLC32EOut o = srlc32e_read(s, A);
    *sr = srlc32e_tick(s, D, CE);

    return ((uint32_t)o.Q   << SRL_OUT_Q_BIT) |
           ((uint32_t)o.Q31 << SRL_OUT_Q31_BIT);
}

// ---------------------------------------------------------------------------
// 3. GPU MACRO KERNEL
//
// One macro instance per thread, grid-stride. All three arrays are indexed by
// the same dense instance id, so a warp touches 32 consecutive u32s = one
// 128-byte transaction per array. Note we deliberately keep the state array as
// packed u32 rather than padding each instance to 64 bits: the SRL state is
// exactly 32 bits, and padding would halve the achieved bandwidth on this
// array for no alignment benefit (the base pointer is already 256B-aligned
// from cudaMalloc, so 64-bit vector loads over instance PAIRS remain
// available if we later want them).
//
// The body is fully branch-free apart from the grid-stride bound, so a warp
// evaluating 32 SRLs has zero divergence.
// ---------------------------------------------------------------------------

#ifdef __CUDACC__
__global__ void srlc32e_macro_kernel(
    const uint32_t *__restrict__ in_packed,
    uint32_t *__restrict__ state,
    uint32_t *__restrict__ out_packed,
    int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x)
    {
        uint32_t s = state[i];
        out_packed[i] = srlc32e_eval_packed(&s, in_packed[i]);
        state[i] = s;
    }
}
#endif

// ---------------------------------------------------------------------------
// 4. INDEPENDENT REFERENCE MODEL
//
// Deliberately NOT a reimplementation of the shift. This models the primitive
// as what it architecturally is -- a CE-gated delay line -- so a bug in the
// bit twiddling above cannot be mirrored here. hist[k] is the value shifted in
// k+1 enabled edges ago, which is by definition state[k].
// ---------------------------------------------------------------------------

struct SRLReference {
    std::deque<uint8_t> hist;

    SRLReference() : hist(32, 0) {}

    uint8_t read(uint8_t A) const { return hist[A & 31]; }
    uint8_t q31()          const { return hist[31]; }

    void tick(bool D, bool CE) {
        if (!CE) return;
        hist.push_front(D ? 1 : 0);
        hist.pop_back();
    }
};

// ---------------------------------------------------------------------------
// 5. CPU TESTBENCH
// ---------------------------------------------------------------------------

static int g_tests = 0;
static int g_fails = 0;

static void check(bool cond, const char *what) {
    ++g_tests;
    if (!cond) {
        ++g_fails;
        std::cout << "  FAIL: " << what << "\n";
    }
}

// Test 1: every dynamic address, CE tied high, random data stream.
// Verifies read-before-shift ordering and the A -> delay(A+1) relationship.
static void test_all_addresses() {
    std::cout << "[T1] all 32 read addresses vs delay-line reference\n";
    std::mt19937 rng(0xC0FFEE);

    for (uint8_t A = 0; A < 32; ++A) {
        uint32_t sr = 0;
        SRLReference ref;
        std::vector<uint8_t> stream;

        for (int cyc = 0; cyc < 256; ++cyc) {
            bool D = (rng() & 1) != 0;
            stream.push_back(D ? 1 : 0);

            uint32_t out = srlc32e_eval_packed(&sr, srl_pack_in(D, true, A));
            uint8_t q = (out >> SRL_OUT_Q_BIT) & 1;

            check(q == ref.read(A), "Q != reference");

            // Independent third check: Q at cycle t must equal the datum
            // driven at cycle t-(A+1), or 0 before the pipe has filled.
            int src = cyc - (int)A - 1;
            uint8_t expect = (src < 0) ? 0 : stream[src];
            check(q == expect, "Q != stream delayed by A+1");

            ref.tick(D, true);
        }
        // After the loop the shifted-in history must equal the state word.
        for (int k = 0; k < 32; ++k) {
            check(((sr >> k) & 1) == ref.hist[k], "state word != reference");
        }
    }
}

// Test 2: CE gating. State must freeze exactly on the cycles where CE==0,
// and the address read must stay live regardless of CE.
static void test_clock_enable() {
    std::cout << "[T2] CE gating (state holds, read stays combinational)\n";
    std::mt19937 rng(0xBEEF);

    uint32_t sr = 0;
    SRLReference ref;

    for (int cyc = 0; cyc < 4096; ++cyc) {
        bool    D  = (rng() & 1) != 0;
        bool    CE = (rng() % 4) != 0;          // 75% enabled
        uint8_t A  = (uint8_t)(rng() & 31);

        uint32_t before = sr;
        uint32_t out = srlc32e_eval_packed(&sr, srl_pack_in(D, CE, A));

        check(((out >> SRL_OUT_Q_BIT)   & 1) == ref.read(A), "Q != reference");
        check(((out >> SRL_OUT_Q31_BIT) & 1) == ref.q31(),   "Q31 != reference");

        if (!CE) check(sr == before, "state moved while CE==0");
        else     check(sr == ((before << 1) | (uint32_t)D), "bad shift on CE==1");

        ref.tick(D, CE);
    }
}

// Test 3: Q31 is a plain state[31] tap and is unaffected by A.
static void test_q31_independent_of_addr() {
    std::cout << "[T3] Q31 == state[31], independent of A\n";
    std::mt19937 rng(0x51D0);

    uint32_t sr = 0;
    for (int cyc = 0; cyc < 512; ++cyc) {
        bool D = (rng() & 1) != 0;

        uint32_t probe = sr;
        uint8_t seen = 0xff;
        for (uint8_t A = 0; A < 32; ++A) {
            uint32_t s = probe;                 // read-only probe per address
            uint32_t out = srlc32e_eval_packed(&s, srl_pack_in(D, false, A));
            uint8_t q31 = (out >> SRL_OUT_Q31_BIT) & 1;
            if (seen == 0xff) seen = q31;
            check(q31 == seen, "Q31 varied with A");
            check(q31 == ((probe >> 31) & 1), "Q31 != state[31]");
        }
        sr = srlc32e_tick(sr, D, true);
    }
}

// Test 4: cascade. srl1.D <- srl0.Q31. Because Q31 is a read of the CURRENT
// state, both instances update from the same pre-edge snapshot -- this test
// fails loudly if the cascade is ever evaluated with a combinational
// (same-cycle) dependency, which is the trap we must not fall into on the GPU.
static void test_cascade_64_deep() {
    std::cout << "[T4] Q31 cascade: two SRLs form a 64-cycle delay line\n";

    uint32_t sr0 = 0, sr1 = 0;
    const int kMarker = 3;
    std::vector<uint8_t> stream;

    for (int cyc = 0; cyc < 256; ++cyc) {
        bool D = (cyc == kMarker);
        stream.push_back(D ? 1 : 0);

        // Read phase for BOTH instances off the pre-edge state.
        uint32_t out0 = srlc32e_eval_packed(&sr0, srl_pack_in(D, true, 31));
        bool cascade  = ((out0 >> SRL_OUT_Q31_BIT) & 1) != 0;
        uint32_t out1 = srlc32e_eval_packed(&sr1, srl_pack_in(cascade, true, 31));

        uint8_t q_tail = (out1 >> SRL_OUT_Q_BIT) & 1;

        // srl0 Q(A=31) delays by 32, srl1 by another 32 -> total 64.
        int src = cyc - 64;
        uint8_t expect = (src < 0) ? 0 : stream[src];
        check(q_tail == expect, "cascade tail != 64-cycle delay");
    }
}

// Test 5: zero-init contract. Q must read 0 for the first A+1 cycles.
static void test_zero_init() {
    std::cout << "[T5] zero init: Q reads 0 until the pipe fills\n";

    for (uint8_t A = 0; A < 32; ++A) {
        uint32_t sr = 0;
        for (int cyc = 0; cyc < 40; ++cyc) {
            uint32_t out = srlc32e_eval_packed(&sr, srl_pack_in(true, true, A));
            uint8_t q = (out >> SRL_OUT_Q_BIT) & 1;
            uint8_t expect = (cyc >= (int)A + 1) ? 1 : 0;
            check(q == expect, "zero-init fill sequence wrong");
        }
    }
}

// ---------------------------------------------------------------------------
// 6. GPU CROSS-CHECK
// ---------------------------------------------------------------------------

#ifdef __CUDACC__
static void test_gpu_matches_cpu() {
    std::cout << "[T6] GPU kernel vs CPU model, 4096 instances x 128 cycles\n";

    const int N = 4096;
    const int CYCLES = 128;

    std::mt19937 rng(0xDEADBEEF);
    std::vector<uint32_t> h_state(N, 0), h_in(N), h_out(N);
    std::vector<uint32_t> cpu_state(N, 0);

    uint32_t *d_in = nullptr, *d_state = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in,    N * sizeof(uint32_t));
    cudaMalloc(&d_state, N * sizeof(uint32_t));
    cudaMalloc(&d_out,   N * sizeof(uint32_t));
    cudaMemcpy(d_state, h_state.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    for (int cyc = 0; cyc < CYCLES; ++cyc) {
        for (int i = 0; i < N; ++i) {
            h_in[i] = srl_pack_in((rng() & 1) != 0,
                                  (rng() % 4) != 0,
                                  (uint8_t)(rng() & 31));
        }
        cudaMemcpy(d_in, h_in.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);

        srlc32e_macro_kernel<<<64, 256>>>(d_in, d_state, d_out, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "  CUDA launch error: " << cudaGetErrorString(err) << "\n";
            ++g_fails;
            break;
        }
        cudaDeviceSynchronize();
        cudaMemcpy(h_out.data(), d_out, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        for (int i = 0; i < N; ++i) {
            uint32_t s = cpu_state[i];
            uint32_t ref_out = srlc32e_eval_packed(&s, h_in[i]);
            cpu_state[i] = s;
            check(h_out[i] == ref_out, "GPU output != CPU model");
        }
    }

    cudaMemcpy(h_state.data(), d_state, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        check(h_state[i] == cpu_state[i], "GPU state != CPU state");
    }

    cudaFree(d_in);
    cudaFree(d_state);
    cudaFree(d_out);
}
#endif

// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== SRLC32E golden model verification ===\n";

    test_all_addresses();
    test_clock_enable();
    test_q31_independent_of_addr();
    test_cascade_64_deep();
    test_zero_init();

#ifdef __CUDACC__
    test_gpu_matches_cpu();
#else
    std::cout << "[T6] skipped (built without nvcc)\n";
#endif

    std::cout << "-----------------------------------------\n";
    std::cout << "Checks passed: " << (g_tests - g_fails) << " / " << g_tests << "\n";
    std::cout << (g_fails == 0 ? "SRLC32E MODEL OK\n" : "SRLC32E MODEL FAILED\n");

    return g_fails == 0 ? 0 : 1;
}
