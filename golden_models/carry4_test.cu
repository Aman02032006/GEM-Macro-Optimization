// CARRY4 golden model + testbench.
//
// Xilinx 4-bit carry chain, per the Zenith PS section B:
//   Inputs : S[3:0] (sum/propagate select), DI[3:0] (data/generate),
//            CIN (cascade carry-in), CYINIT (carry initialisation)
//   Logic  : C[0]   = CYINIT | CIN          (only one is active in valid RTL)
//            C[i+1] = (S[i] & C[i]) | (~S[i] & DI[i])   for i in 0..3
//            O[i]   = S[i] ^ C[i]
//            CO[i]  = C[i+1]
//   State  : none. This macro is purely combinational.
//
// Cycle semantics matter for GEM integration. Unlike SRLC32E (a clocked
// endpoint that reads its own pre-edge state), CARRY4 has zero registers, so
// a CO[3] -> CIN cascade is a genuine COMBINATIONAL edge inside one simulated
// cycle. That is precisely why these blocks cannot be deferred to the
// post-tree SRAM/write-out phase the way SRLC32E can: a 32-bit adder is 8
// chained CARRY4s, and one macro per major stage would burn 8 of the
// simulator's 32-major-stage budget for a single adder. They must be collapsed
// in-tree by the boomerang scheduler (Option B).
//
// GPU note: the whole 4-bit Manchester chain is evaluated with ONE native
// integer add (see carry4_eval_packed). The sequential recurrence never
// appears in the emitted SASS, so a warp of 32 CARRY4s retires the chain in a
// single ALU op with no divergence and no serial dependency.
//
// Build:
//   CPU  : g++  -O2 -x c++ carry4_test.cu -o carry4_test
//   CUDA : nvcc -O2 -arch=sm_86 carry4_test.cu -o carry4_test
//          (native SASS is required; the installed driver will not JIT
//           CUDA 13.3 PTX. Runs CPU tests then the GPU cross-check.)

#include <iostream>
#include <cstdint>
#include <cassert>
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
// coalesced load instead of ten scattered bit-gathers.
//
//   bits 0..3 : S[3:0]
//   bits 4..7 : DI[3:0]
//   bit  8    : CIN
//   bit  9    : CYINIT
//
// Output word: bits 0..3 = O[3:0], bits 4..7 = CO[3:0].
//
// CIN and CYINIT stay in separate bits even though the model only ever uses
// their OR. The formatter routes a cascaded CO[3] into CIN and a constant or
// LUT-driven init into CYINIT, so keeping them distinct is what lets the
// scheduler recognise a chain structurally (CIN driven by another macro) and
// fuse it, rather than having to pattern-match on an already-merged bit.
// ---------------------------------------------------------------------------

#define C4_IN_S_SHIFT      0u
#define C4_IN_S_MASK       0xfu
#define C4_IN_DI_SHIFT     4u
#define C4_IN_DI_MASK      0xfu
#define C4_IN_CIN_BIT      8u
#define C4_IN_CYINIT_BIT   9u

#define C4_OUT_O_SHIFT     0u
#define C4_OUT_O_MASK      0xfu
#define C4_OUT_CO_SHIFT    4u
#define C4_OUT_CO_MASK     0xfu

__host__ __device__ inline uint32_t c4_pack_in(uint8_t S, uint8_t DI,
                                               bool CIN, bool CYINIT) {
    return (((uint32_t)S  & C4_IN_S_MASK)  << C4_IN_S_SHIFT)  |
           (((uint32_t)DI & C4_IN_DI_MASK) << C4_IN_DI_SHIFT) |
           ((uint32_t)(CIN    & 1) << C4_IN_CIN_BIT)          |
           ((uint32_t)(CYINIT & 1) << C4_IN_CYINIT_BIT);
}

__host__ __device__ inline uint8_t c4_unpack_o(uint32_t out) {
    return (uint8_t)((out >> C4_OUT_O_SHIFT) & C4_OUT_O_MASK);
}

__host__ __device__ inline uint8_t c4_unpack_co(uint32_t out) {
    return (uint8_t)((out >> C4_OUT_CO_SHIFT) & C4_OUT_CO_MASK);
}

struct Carry4Out {
    uint8_t O  : 4;  // O[3:0]
    uint8_t CO : 4;  // CO[3:0]
};

// ---------------------------------------------------------------------------
// 2. CORE MODEL (host + device)
//
// The Manchester chain collapsed onto the ALU. With
//     p[i] = S[i]                    (propagate C[i])
//     g[i] = DI[i] & ~S[i]           (inject DI[i]);  note g & p == 0
// the spec recurrence C[i+1] = g[i] | (p[i] & C[i]) is exactly the carry
// recurrence of a binary adder whose generate is g and propagate is p. Feeding
//     a = g       = DI & ~S
//     b = g | p   = DI |  S
// gives a & b == g and a ^ b == p, so the hardware adder t = a + b + C[0]
// produces the entire carry vector at once:
//     C[0..3] = (t ^ a ^ b) & 0xF        (sum[i] = a[i]^b[i]^C[i])
//     C[4]    = (t >> 4) & 1
// Verified exhaustively against the literal per-bit recurrence in [T1].
// ---------------------------------------------------------------------------

__host__ __device__ inline uint32_t carry4_eval_packed(uint32_t in) {
    uint32_t S      = (in >> C4_IN_S_SHIFT)  & C4_IN_S_MASK;
    uint32_t DI     = (in >> C4_IN_DI_SHIFT) & C4_IN_DI_MASK;
    uint32_t CIN    = (in >> C4_IN_CIN_BIT)    & 1u;
    uint32_t CYINIT = (in >> C4_IN_CYINIT_BIT) & 1u;

    uint32_t c0 = CYINIT | CIN;          // C[0]

    uint32_t a = DI & ~S & C4_IN_S_MASK; // generate
    uint32_t b = DI |  S;                // generate | propagate
    uint32_t t = a + b + c0;             // one native ALU add == whole chain

    uint32_t cvec = (t ^ a ^ b) & 0xfu;  // C[3:0]
    uint32_t c4   = (t >> 4) & 1u;       // C[4]

    uint32_t O  = (S ^ cvec) & 0xfu;                 // O[i]  = S[i] ^ C[i]
    uint32_t CO = ((cvec >> 1) | (c4 << 3)) & 0xfu;  // CO[i] = C[i+1]

    return (O << C4_OUT_O_SHIFT) | (CO << C4_OUT_CO_SHIFT);
}

// Convenience wrapper preserving the original struct-returning interface.
__host__ __device__ inline Carry4Out evaluate_carry4(uint8_t S, uint8_t DI,
                                                     bool CIN, bool CYINIT) {
    uint32_t out = carry4_eval_packed(c4_pack_in(S, DI, CIN, CYINIT));
    Carry4Out r;
    r.O  = c4_unpack_o(out);
    r.CO = c4_unpack_co(out);
    return r;
}

// ---------------------------------------------------------------------------
// 3. GPU MACRO KERNEL
//
// One CARRY4 instance per thread, grid-stride. Both arrays are indexed by the
// same dense instance id, so a warp touches 32 consecutive u32s = one 128-byte
// transaction per array. There is no state array: the macro is combinational,
// which is what makes it cheap in VRAM but expensive in schedule depth.
//
// The body is fully branch-free apart from the grid-stride bound, so a warp
// evaluating 32 CARRY4s has zero divergence.
//
// Cascades are deliberately NOT resolved here. CO[3] -> CIN is a scheduling
// obligation owned by the boomerang DAG (Option B), not something a flat
// macro kernel may serialise on its own.
// ---------------------------------------------------------------------------

#ifdef __CUDACC__
__global__ void carry4_macro_kernel(
    const uint32_t *__restrict__ in_packed,
    uint32_t *__restrict__ out_packed,
    int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x)
    {
        out_packed[i] = carry4_eval_packed(in_packed[i]);
    }
}
#endif

// ---------------------------------------------------------------------------
// 4. INDEPENDENT REFERENCE MODEL
//
// The literal PS recurrence, one bit at a time, with an explicit multiplexer
// rather than the boolean form -- so a sign/mask error in the fast path cannot
// be mirrored here. This is the arbiter of correctness; carry4_eval_packed is
// only an optimisation of it.
// ---------------------------------------------------------------------------

struct Carry4Reference {
    uint8_t O;
    uint8_t CO;
    uint8_t C[5];   // C[0..4], exposed so tests can inspect the chain itself

    void eval(uint8_t S, uint8_t DI, bool CIN, bool CYINIT) {
        O = 0;
        CO = 0;
        C[0] = (uint8_t)((CIN | CYINIT) & 1);

        for (int i = 0; i < 4; ++i) {
            uint8_t s_bit  = (S  >> i) & 1;
            uint8_t di_bit = (DI >> i) & 1;

            // strict multiplexer form: S selects between the carry and DI
            uint8_t c_next = s_bit ? C[i] : di_bit;

            O  |= (uint8_t)((s_bit ^ C[i]) << i);
            CO |= (uint8_t)(c_next << i);
            C[i + 1] = c_next;
        }
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
        if (g_fails <= 10) std::cout << "  FAIL: " << what << "\n";
    }
}

// Test 1: exhaustive. All 2^10 = 1024 input combinations against the literal
// recurrence. This is the complete truth table of the primitive, so the fast
// arithmetic path is proven, not sampled.
static void test_exhaustive() {
    std::cout << "[T1] exhaustive 1024-vector sweep vs literal recurrence\n";

    Carry4Reference ref;
    for (uint32_t s = 0; s < 16; ++s) {
        for (uint32_t di = 0; di < 16; ++di) {
            for (uint32_t cin = 0; cin < 2; ++cin) {
                for (uint32_t cyinit = 0; cyinit < 2; ++cyinit) {
                    uint32_t out = carry4_eval_packed(
                        c4_pack_in((uint8_t)s, (uint8_t)di, cin != 0, cyinit != 0));
                    ref.eval((uint8_t)s, (uint8_t)di, cin != 0, cyinit != 0);

                    check(c4_unpack_o(out)  == ref.O,  "O != reference");
                    check(c4_unpack_co(out) == ref.CO, "CO != reference");

                    // CO[i] must literally be C[i+1], not a recomputation.
                    for (int i = 0; i < 4; ++i) {
                        check(((c4_unpack_co(out) >> i) & 1) == ref.C[i + 1],
                              "CO[i] != C[i+1]");
                    }
                    // Nothing may be set outside the 8 defined output bits.
                    check((out & ~0xffu) == 0, "output word has stray bits");
                }
            }
        }
    }
}

// Test 2: the CIN vs CYINIT constraint. The PS states only one is active in
// valid RTL, but silicon ORs them. We verify (a) each source alone drives the
// chain identically, (b) the illegal both-high case still ORs rather than
// doing anything surprising, and (c) with both low the chain starts at 0.
static void test_cin_vs_cyinit() {
    std::cout << "[T2] CIN / CYINIT carry-init sourcing\n";

    Carry4Reference ref;
    for (uint32_t s = 0; s < 16; ++s) {
        for (uint32_t di = 0; di < 16; ++di) {
            uint32_t only_cin    = carry4_eval_packed(c4_pack_in(s, di, true,  false));
            uint32_t only_cyinit = carry4_eval_packed(c4_pack_in(s, di, false, true));
            uint32_t both        = carry4_eval_packed(c4_pack_in(s, di, true,  true));
            uint32_t neither     = carry4_eval_packed(c4_pack_in(s, di, false, false));

            // (a) the two legal sources are interchangeable
            check(only_cin == only_cyinit, "CIN and CYINIT paths differ");

            // (b) the illegal overlap degenerates to the same OR result
            check(both == only_cin, "both-high != OR semantics");

            // (c) neither high must start the chain at C[0] = 0
            ref.eval((uint8_t)s, (uint8_t)di, false, false);
            check(ref.C[0] == 0, "C[0] != 0 with both inits low");
            check(c4_unpack_o(neither)  == ref.O,  "O != reference (no init)");
            check(c4_unpack_co(neither) == ref.CO, "CO != reference (no init)");

            // C[0] must be the ONLY thing the init bits can influence: with
            // S[0]=0 the chain kills the carry immediately, so CO must not
            // depend on the init at all.
            if ((s & 1) == 0) {
                check(c4_unpack_co(only_cin) == c4_unpack_co(neither),
                      "init leaked past a killed carry");
            }
        }
    }
}

// Test 3: structural cascade. Eight CARRY4s chained CO[3] -> CIN implement a
// 32-bit adder (S = A^B propagate, DI = A generate, CYINIT = carry-in on the
// first block only). Verified against native uint32_t arithmetic. This is the
// exact dependency pattern the boomerang scheduler must collapse in-tree.
static void test_cascade_32bit_adder() {
    std::cout << "[T3] 8x CARRY4 cascade == 32-bit adder\n";

    struct { uint32_t a, b; uint32_t cin; } edges[] = {
        {0x00000000u, 0x00000000u, 0}, {0xffffffffu, 0x00000001u, 0},
        {0xffffffffu, 0xffffffffu, 1}, {0x7fffffffu, 0x00000001u, 0},
        {0x80000000u, 0x80000000u, 0}, {0xdeadbeefu, 0x12345678u, 1},
        {0x0000ffffu, 0x00010000u, 0}, {0xaaaaaaaau, 0x55555555u, 1},
    };

    std::mt19937 rng(0xCA44);
    const int kRandom = 20000;

    for (int t = 0; t < (int)(sizeof(edges) / sizeof(edges[0])) + kRandom; ++t) {
        uint32_t A, B, cin;
        if (t < (int)(sizeof(edges) / sizeof(edges[0]))) {
            A = edges[t].a; B = edges[t].b; cin = edges[t].cin;
        } else {
            A = rng(); B = rng(); cin = rng() & 1;
        }

        uint32_t sum = 0;
        bool carry = (cin != 0);

        for (int blk = 0; blk < 8; ++blk) {
            uint8_t a4 = (uint8_t)((A >> (blk * 4)) & 0xf);
            uint8_t b4 = (uint8_t)((B >> (blk * 4)) & 0xf);
            uint8_t S  = (uint8_t)(a4 ^ b4);   // propagate where A != B
            uint8_t DI = a4;                   // generate  where A == B == 1

            // Block 0 takes the adder carry-in on CYINIT; every later block
            // takes it on CIN from the previous CO[3]. Exactly one is active.
            uint32_t out = carry4_eval_packed(
                (blk == 0) ? c4_pack_in(S, DI, false, carry)
                           : c4_pack_in(S, DI, carry, false));

            sum |= (uint32_t)c4_unpack_o(out) << (blk * 4);
            carry = ((c4_unpack_co(out) >> 3) & 1) != 0;
        }

        uint64_t expect = (uint64_t)A + (uint64_t)B + (uint64_t)cin;
        check(sum == (uint32_t)expect, "cascade sum != A+B+cin");
        check(carry == (((expect >> 32) & 1) != 0), "cascade carry-out wrong");
    }
}

// Test 4: packing round-trip. Guards the encoding the Rust formatter will
// emit -- a field overlap here would silently corrupt every macro in the
// netlist rather than fail loudly.
static void test_packing_roundtrip() {
    std::cout << "[T4] packed-word field isolation\n";

    for (uint32_t s = 0; s < 16; ++s) {
        for (uint32_t di = 0; di < 16; ++di) {
            for (uint32_t cin = 0; cin < 2; ++cin) {
                for (uint32_t cyinit = 0; cyinit < 2; ++cyinit) {
                    uint32_t w = c4_pack_in(s, di, cin != 0, cyinit != 0);

                    check(((w >> C4_IN_S_SHIFT)  & C4_IN_S_MASK)  == s,  "S field");
                    check(((w >> C4_IN_DI_SHIFT) & C4_IN_DI_MASK) == di, "DI field");
                    check(((w >> C4_IN_CIN_BIT)    & 1) == cin,    "CIN field");
                    check(((w >> C4_IN_CYINIT_BIT) & 1) == cyinit, "CYINIT field");
                    check((w >> 10) == 0, "packed word exceeds 10 bits");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. GPU CROSS-CHECK
// ---------------------------------------------------------------------------

#ifdef __CUDACC__
static void test_gpu_matches_cpu() {
    std::cout << "[T5] GPU kernel vs CPU reference, 65536 instances\n";

    const int N = 65536;

    std::mt19937 rng(0xDEADC4);
    std::vector<uint32_t> h_in(N), h_out(N);

    // First 1024 instances are the exhaustive truth table, so the GPU path is
    // checked against every legal vector, not just random ones.
    for (int i = 0; i < N; ++i) {
        if (i < 1024) {
            h_in[i] = c4_pack_in((uint8_t)(i & 0xf), (uint8_t)((i >> 4) & 0xf),
                                 ((i >> 8) & 1) != 0, ((i >> 9) & 1) != 0);
        } else {
            h_in[i] = c4_pack_in((uint8_t)(rng() & 0xf), (uint8_t)(rng() & 0xf),
                                 (rng() & 1) != 0, (rng() & 1) != 0);
        }
    }

    uint32_t *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in,  N * sizeof(uint32_t));
    cudaMalloc(&d_out, N * sizeof(uint32_t));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    carry4_macro_kernel<<<256, 256>>>(d_in, d_out, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "  CUDA launch error: " << cudaGetErrorString(err) << "\n";
        ++g_fails;
    } else {
        cudaDeviceSynchronize();
        cudaMemcpy(h_out.data(), d_out, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        Carry4Reference ref;
        for (int i = 0; i < N; ++i) {
            uint32_t w = h_in[i];
            ref.eval((uint8_t)((w >> C4_IN_S_SHIFT)  & C4_IN_S_MASK),
                     (uint8_t)((w >> C4_IN_DI_SHIFT) & C4_IN_DI_MASK),
                     ((w >> C4_IN_CIN_BIT)    & 1) != 0,
                     ((w >> C4_IN_CYINIT_BIT) & 1) != 0);

            check(c4_unpack_o(h_out[i])  == ref.O,  "GPU O != reference");
            check(c4_unpack_co(h_out[i]) == ref.CO, "GPU CO != reference");
        }
    }

    cudaFree(d_in);
    cudaFree(d_out);
}
#endif

// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== CARRY4 golden model verification ===\n";

    test_exhaustive();
    test_cin_vs_cyinit();
    test_cascade_32bit_adder();
    test_packing_roundtrip();

#ifdef __CUDACC__
    test_gpu_matches_cpu();
#else
    std::cout << "[T5] skipped (built without nvcc)\n";
#endif

    std::cout << "----------------------------------------\n";
    std::cout << "Checks passed: " << (g_tests - g_fails) << " / " << g_tests << "\n";
    std::cout << (g_fails == 0 ? "CARRY4 MODEL OK\n" : "CARRY4 MODEL FAILED\n");

    return g_fails == 0 ? 0 : 1;
}
