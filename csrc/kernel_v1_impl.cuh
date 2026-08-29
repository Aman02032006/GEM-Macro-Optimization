// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <crates/ulib/includes.hpp>
#include <cstdio>
#include <cooperative_groups.h>

struct alignas(8) VectorRead2 {
  u32 c1, c2;

  __device__ __forceinline__ void read(const VectorRead2 *t) {
    *this = *t;
  }
};

struct alignas(16) VectorRead4 {
  u32 c1, c2, c3, c4;

  __device__ __forceinline__ void read(const VectorRead4 *t) {
    *this = *t;
  }
};

// ---------------------------------------------------------------------------
// Mid-partition word-level macro phase.
//
// Transliterated from run_macro_phase() in src/bin/flatten_test.rs, which the
// CPU oracle proves against naive_sim. The one deliberate difference is the
// carry chain: the oracle walks it sequentially, while here a whole chain
// occupies consecutive lanes of ONE warp and the carry is propagated with a
// Kogge-Stone __shfl_up_sync scan. Both must agree bit for bit.
//
// Layout constants mirror src/macros.rs.
// ---------------------------------------------------------------------------
#define GEM_MACRO_LANE_WORDS   12
#define GEM_MACRO_MAX_LANES    32
#define GEM_MACRO_IN_SLOTS     10
#define GEM_MACRO_OUT_SLOTS     8
#define GEM_MACRO_PHASE_WORDS  (GEM_MACRO_MAX_LANES * GEM_MACRO_LANE_WORDS)

#define GEM_MACRO_POS_NONE     0xffffu
#define GEM_PERM_POS_MASK      0x1fffu
#define GEM_PERM_INV_BIT       13
#define GEM_PERM_CONST_BIT     14

#define GEM_DESC_KIND_MASK     0x3u
#define GEM_DESC_CHAIN_BIT     2
#define GEM_DESC_VALID_BIT     3
#define GEM_DESC_STATE_SHIFT   16

#define GEM_KIND_CARRY4        0u
#define GEM_KIND_SRLC32E       1u

// CARRY4 slots
#define GEM_C4_IN_CI      0
#define GEM_C4_IN_CYINIT  1
#define GEM_C4_IN_DI      2
#define GEM_C4_IN_S       6
#define GEM_C4_OUT_O      0
#define GEM_C4_OUT_CO     4
// SRLC32E slots
#define GEM_SRL_IN_D      0
#define GEM_SRL_IN_CE     1
#define GEM_SRL_IN_A      2
#define GEM_SRL_OUT_Q     0
#define GEM_SRL_OUT_Q31   1

/// Sign-extend the low `w` bits of `v`.
__device__ __forceinline__ long long gem_sext(unsigned long long v, int w) {
  unsigned long long m = 1ull << (w - 1);
  return (long long)(((v & ((m << 1) - 1)) ^ m) - m);
}

__device__ __forceinline__ u32 gem_macro_read_bit(
  const u32 *__restrict__ shared_state, u32 pos)
{
  return (shared_state[pos >> 5] >> (pos & 31)) & 1u;
}

/// Decode one input code: position + invert, or a constant.
__device__ __forceinline__ u32 gem_macro_decode(
  const u32 *__restrict__ shared_state, u32 code)
{
  u32 inv = (code >> GEM_PERM_INV_BIT) & 1u;
  if ((code >> GEM_PERM_CONST_BIT) & 1u) return inv;
  return gem_macro_read_bit(shared_state, code & GEM_PERM_POS_MASK) ^ inv;
}

__device__ void gem_macro_phase(
  const u32 *__restrict__ phase,
  u32 *__restrict__ shared_state,
  u32 *__restrict__ sram_data)
{
  // One phase is one warp by construction (pe.rs caps a wave at 32 lanes), so
  // the carry scan never crosses a shuffle boundary.
  u32 lane = threadIdx.x;
  u32 desc = 0, in_code[GEM_MACRO_IN_SLOTS], out_pos[GEM_MACRO_OUT_SLOTS];
  u32 valid = 0, kind = 0, chain_start = 0, state_off = 0;
  u32 inp[GEM_MACRO_IN_SLOTS];

  if (lane < GEM_MACRO_MAX_LANES) {
    const u32 *w = phase + lane * GEM_MACRO_LANE_WORDS;
    desc = w[0];
    valid = (desc >> GEM_DESC_VALID_BIT) & 1u;
    kind = desc & GEM_DESC_KIND_MASK;
    chain_start = (desc >> GEM_DESC_CHAIN_BIT) & 1u;
    state_off = desc >> GEM_DESC_STATE_SHIFT;
#pragma unroll
    for (int k = 0; k < GEM_MACRO_IN_SLOTS / 2; ++k) {
      u32 t = w[1 + k];
      in_code[k * 2]     = t & 0xffffu;
      in_code[k * 2 + 1] = t >> 16;
    }
#pragma unroll
    for (int k = 0; k < GEM_MACRO_OUT_SLOTS / 2; ++k) {
      u32 t = w[1 + GEM_MACRO_IN_SLOTS / 2 + k];
      out_pos[k * 2]     = t & 0xffffu;
      out_pos[k * 2 + 1] = t >> 16;
    }
#pragma unroll
    for (int k = 0; k < GEM_MACRO_IN_SLOTS; ++k) {
      inp[k] = valid ? gem_macro_decode(shared_state, in_code[k]) : 0u;
    }
  }
  __syncwarp();

  // ---- CARRY4: segmented Kogge-Stone carry scan -------------------------
  // Per lane, reduce the 4-bit Manchester chain to a block (generate,
  // propagate) pair, then scan across lanes. A chain head clears its
  // propagate so the scan cannot leak a carry in from the segment above it.
  u32 is_c4 = (valid && kind == GEM_KIND_CARRY4) ? 1u : 0u;
  u32 S = 0, DI = 0, seed = 0;
  if (is_c4) {
    S  = (inp[GEM_C4_IN_S + 0]) | (inp[GEM_C4_IN_S + 1] << 1)
       | (inp[GEM_C4_IN_S + 2] << 2) | (inp[GEM_C4_IN_S + 3] << 3);
    DI = (inp[GEM_C4_IN_DI + 0]) | (inp[GEM_C4_IN_DI + 1] << 1)
       | (inp[GEM_C4_IN_DI + 2] << 2) | (inp[GEM_C4_IN_DI + 3] << 3);
    // A chain head injects its own CI; a link takes CI from the scan. CYINIT
    // is OR-ed in either way, matching the reference model.
    seed = (chain_start ? (inp[GEM_C4_IN_CI] | inp[GEM_C4_IN_CYINIT])
                        : inp[GEM_C4_IN_CYINIT]) & 1u;
  }
  // block generate with carry-in 0, and block propagate
  u32 blk_g = 0, blk_p = 0;
  if (is_c4) {
    u32 c = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      u32 sb = (S >> i) & 1u, db = (DI >> i) & 1u;
      c = sb ? c : db;
    }
    blk_g = c;
    blk_p = (S == 0xfu) ? 1u : 0u;
  }
  // g' = G | (P & seed) ; p' = P & is_link  (head clears propagate)
  u32 g = is_c4 ? (blk_g | (blk_p & seed)) : 0u;
  u32 p = (is_c4 && !chain_start) ? blk_p : 0u;
#pragma unroll
  for (int d = 1; d < GEM_MACRO_MAX_LANES; d <<= 1) {
    u32 gu = __shfl_up_sync(0xffffffffu, g, d);
    u32 pu = __shfl_up_sync(0xffffffffu, p, d);
    if (lane >= (u32)d) { g = g | (p & gu); p = p & pu; }
  }
  // g is now the carry OUT of this lane; the carry IN of a link is the
  // previous lane's carry out.
  u32 carry_prev = __shfl_up_sync(0xffffffffu, g, 1);
  u32 c_in = is_c4 ? (seed | ((!chain_start && lane > 0) ? carry_prev : 0u)) : 0u;

  // ---- evaluate and scatter --------------------------------------------
  u32 o_bits[GEM_MACRO_OUT_SLOTS];
#pragma unroll
  for (int k = 0; k < GEM_MACRO_OUT_SLOTS; ++k) o_bits[k] = 0;

  if (valid && kind == GEM_KIND_CARRY4) {
    u32 c = c_in & 1u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      u32 sb = (S >> i) & 1u, db = (DI >> i) & 1u;
      u32 cn = sb ? c : db;
      o_bits[GEM_C4_OUT_O + i]  = sb ^ c;
      o_bits[GEM_C4_OUT_CO + i] = cn;
      c = cn;
    }
  }
  else if (valid && kind == GEM_KIND_SRLC32E) {
    // Read strictly before the shift: both taps observe the pre-edge
    // register, so the committed value is only visible next cycle.
    u32 sr = sram_data[state_off];
    u32 a = inp[GEM_SRL_IN_A + 0] | (inp[GEM_SRL_IN_A + 1] << 1)
          | (inp[GEM_SRL_IN_A + 2] << 2) | (inp[GEM_SRL_IN_A + 3] << 3)
          | (inp[GEM_SRL_IN_A + 4] << 4);
    o_bits[GEM_SRL_OUT_Q]   = (sr >> a) & 1u;
    o_bits[GEM_SRL_OUT_Q31] = (sr >> 31) & 1u;
    if (inp[GEM_SRL_IN_CE] & 1u) {
      sram_data[state_off] = (sr << 1) | (inp[GEM_SRL_IN_D] & 1u);
    }
  }
  __syncthreads();

  // Different lanes may target different bits of the same shared word, so the
  // scatter is done with atomics over disjoint bit sets: clear-then-set is
  // safe in any interleaving because each lane only touches its own bits.
  if (valid) {
#pragma unroll
    for (int k = 0; k < GEM_MACRO_OUT_SLOTS; ++k) {
      u32 pos = out_pos[k];
      if (pos == GEM_MACRO_POS_NONE) continue;
      u32 w = pos >> 5, b = pos & 31;
      atomicAnd(&shared_state[w], ~(1u << b));
      atomicOr(&shared_state[w], (o_bits[k] & 1u) << b);
    }
  }
  __syncthreads();
}

__device__ void simulate_block_v1(
  const u32 *__restrict__ script,
  usize script_size,
  const u32 *__restrict__ input_state,
  u32 *__restrict__ output_state,
  u32 *__restrict__ sram_data,
  u32 *__restrict__ shared_metadata,
  u32 *__restrict__ shared_writeouts,
  u32 *__restrict__ shared_state
  )
{
  int script_pi = 0;
  while(true) {
    VectorRead2 t2_1, t2_2;
    VectorRead4 t4_1, t4_2, t4_3, t4_4, t4_5;
    shared_metadata[threadIdx.x] = script[script_pi + threadIdx.x];
    script_pi += 256;
    t2_1.read(((const VectorRead2 *)(script + script_pi)) + threadIdx.x);
    __syncthreads();
    int num_stages = shared_metadata[0];
    if(!num_stages) {
      break;
    }
    int is_last_part = shared_metadata[1];
    int num_ios = shared_metadata[2];
    int io_offset = shared_metadata[3];
    int num_srams = shared_metadata[4];
    int sram_offset = shared_metadata[5];
    int num_global_read_rounds = shared_metadata[6];
    int num_output_duplicates = shared_metadata[7];
    int num_macro_phases = shared_metadata[8];
    int num_dsps = shared_metadata[9];
    int dsp_state_base = shared_metadata[10];
    int dsp_words = num_dsps * 2;
    // Phase headers live at [11..]; each is (after_stage << 16) | num_lanes.
    // Phases scheduled before stage 0 sit between the global read section and
    // stage 0's data, so the very first prefetch has to step over them.
    int ph_cursor = 0;
    int ph0_words = 0;
    while (ph_cursor + (ph0_words / GEM_MACRO_PHASE_WORDS) < num_macro_phases &&
           (shared_metadata[11 + ph_cursor + ph0_words / GEM_MACRO_PHASE_WORDS] >> 16) == 0) {
      ph0_words += GEM_MACRO_PHASE_WORDS;
    }
    u32 writeout_hook_i = shared_metadata[128 + threadIdx.x / 2];
    if(threadIdx.x % 2 == 0) {
      writeout_hook_i = writeout_hook_i & ((1 << 16) - 1);
    }
    else {
      writeout_hook_i = writeout_hook_i >> 16;
    }

    t4_1.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + ph0_words) + threadIdx.x);
    t4_2.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + ph0_words + 256 * 4) + threadIdx.x);
    t4_3.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + ph0_words + 256 * 4 * 2) + threadIdx.x);
    t4_4.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + ph0_words + 256 * 4 * 3) + threadIdx.x);
    t4_5.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + ph0_words + 256 * 4 * 4) + threadIdx.x);
    u32 t_global_rd_state = 0;
    for(int gr_i = 0; gr_i < num_global_read_rounds; gr_i += 2) {
      u32 idx = t2_1.c1;
      u32 mask = t2_1.c2;
      script_pi += 256 * 2;
      t2_2.read(((const VectorRead2 *)(script + script_pi)) + threadIdx.x);
      if(mask) {
        const u32 *real_input_array;
        if(idx >> 31) real_input_array = output_state - (1 << 31);
        else real_input_array = input_state;
        u32 value = real_input_array[idx];
        while(mask) {
          t_global_rd_state <<= 1;
          u32 lowbit = mask & -mask;
          if(value & lowbit) t_global_rd_state |= 1;
          mask ^= lowbit;
        }
      }

      if(gr_i + 1 >= num_global_read_rounds) break;
      idx = t2_2.c1;
      mask = t2_2.c2;
      script_pi += 256 * 2;
      t2_1.read(((const VectorRead2 *)(script + script_pi)) + threadIdx.x);
      if(mask) {
        const u32 *real_input_array;
        if(idx >> 31) real_input_array = output_state - (1 << 31);
        else real_input_array = input_state;
        u32 value = real_input_array[idx];
        while(mask) {
          t_global_rd_state <<= 1;
          u32 lowbit = mask & -mask;
          if(value & lowbit) t_global_rd_state |= 1;
          mask ^= lowbit;
        }
      }
    }
    shared_state[threadIdx.x] = t_global_rd_state;
    __syncthreads();

    // phases scheduled before the first boomerang stage
    {
      usize ph_pi = script_pi + 256 * 2 * num_global_read_rounds;
      while (ph_cursor < num_macro_phases &&
             (shared_metadata[11 + ph_cursor] >> 16) == 0) {
        gem_macro_phase(script + ph_pi, shared_state, sram_data);
        ph_pi += GEM_MACRO_PHASE_WORDS;
        ph_cursor += 1;
      }
    }
    script_pi += ph0_words;

    for(int bs_i = 0; bs_i < num_stages; ++bs_i) {
      // How many phase words sit between this stage and the next one. The
      // prefetch below has to step over them, and the phases themselves run
      // at the end of this iteration -- i.e. after stage bs_i, before stage
      // bs_i+1, which is exactly their scheduled slot.
      int next_ph = 0;
      while (ph_cursor + next_ph < num_macro_phases &&
             (int)(shared_metadata[11 + ph_cursor + next_ph] >> 16) == bs_i + 1) {
        next_ph += 1;
      }
      int next_ph_words = next_ph * GEM_MACRO_PHASE_WORDS;
      u32 hier_input = 0, hier_flag_xora = 0, hier_flag_xorb = 0, hier_flag_orb = 0;
#define GEMV1_SHUF_INPUT_K(k_outer, k_inner, t_shuffle) {           \
        u32 k = k_outer * 4 + k_inner;                              \
        u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);          \
        u32 t_shuffle_2_idx = t_shuffle >> 16;                      \
                                                                    \
        hier_input |= (shared_state[t_shuffle_1_idx >> 5] >>        \
                       (t_shuffle_1_idx & 31) & 1) << (k * 2);      \
        hier_input |= (shared_state[t_shuffle_2_idx >> 5] >>        \
                       (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);  \
      }
#define GEMV1_SHUF_INPUT_K_4(k_outer, t_shuffle) {    \
        GEMV1_SHUF_INPUT_K(k_outer, 0, t_shuffle.c1); \
        GEMV1_SHUF_INPUT_K(k_outer, 1, t_shuffle.c2); \
        GEMV1_SHUF_INPUT_K(k_outer, 2, t_shuffle.c3); \
        GEMV1_SHUF_INPUT_K(k_outer, 3, t_shuffle.c4); \
      }
      script_pi += 256 * 4 * 5 + next_ph_words;
      GEMV1_SHUF_INPUT_K_4(0, t4_1);
      t4_1.read(((const VectorRead4 *)(script + script_pi)) + threadIdx.x);
      GEMV1_SHUF_INPUT_K_4(1, t4_2);
      t4_2.read(((const VectorRead4 *)(script + script_pi + 256 * 4)) + threadIdx.x);
      GEMV1_SHUF_INPUT_K_4(2, t4_3);
      t4_3.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 2)) + threadIdx.x);
      GEMV1_SHUF_INPUT_K_4(3, t4_4);
      t4_4.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 3)) + threadIdx.x);
#undef GEMV1_SHUF_INPUT_K
#undef GEMV1_SHUF_INPUT_K_4
      hier_flag_xora = t4_5.c1;
      hier_flag_xorb = t4_5.c2;
      hier_flag_orb = t4_5.c3;
      t4_5.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 4)) + threadIdx.x);

      __syncthreads();
      shared_state[threadIdx.x] = hier_input;
      __syncthreads();

      // hier[0]
      if(threadIdx.x >= 128) {
        u32 hier_input_a = shared_state[threadIdx.x - 128];
        u32 hier_input_b = hier_input;
        u32 ret = (hier_input_a ^ hier_flag_xora) & ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
        shared_state[threadIdx.x] = ret;
      }
      __syncthreads();
      // hier[1..3]
      u32 tmp_cur_hi;
      for(int hi = 1; hi <= 3; ++hi) {
        int hier_width = 1 << (7 - hi);
        if(threadIdx.x >= hier_width && threadIdx.x < hier_width * 2) {
          u32 hier_input_a = shared_state[threadIdx.x + hier_width];
          u32 hier_input_b = shared_state[threadIdx.x + hier_width * 2];
          u32 ret = (hier_input_a ^ hier_flag_xora) & ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
          tmp_cur_hi = ret;
          shared_state[threadIdx.x] = ret;
        }
        __syncthreads();
      }
      // hier[4..7], within the first warp.
      if(threadIdx.x < 32) {
        for(int hi = 4; hi <= 7; ++hi) {
          int hier_width = 1 << (7 - hi);
          u32 hier_input_a = __shfl_down_sync(0xffffffff, tmp_cur_hi, hier_width);
          u32 hier_input_b = __shfl_down_sync(0xffffffff, tmp_cur_hi, hier_width * 2);
          if(threadIdx.x >= hier_width && threadIdx.x < hier_width * 2) {
            tmp_cur_hi = (hier_input_a ^ hier_flag_xora) & ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
          }
        }
        u32 v1 = __shfl_down_sync(0xffffffff, tmp_cur_hi, 1);
        // hier[8..12]
        if(threadIdx.x == 0) {
          u32 r8 = ((v1 << 16) ^ hier_flag_xora) & ((v1 ^ hier_flag_xorb) | hier_flag_orb) & 0xffff0000;
          u32 r9 = ((r8 >> 8) ^ hier_flag_xora) & (((r8 >> 16) ^ hier_flag_xorb) | hier_flag_orb) & 0xff00;
          u32 r10 = ((r9 >> 4) ^ hier_flag_xora) & (((r9 >> 8) ^ hier_flag_xorb) | hier_flag_orb) & 0xf0;
          u32 r11 = ((r10 >> 2) ^ hier_flag_xora) & (((r10 >> 4) ^ hier_flag_xorb) | hier_flag_orb) & 12 /* 0b1100 */;
          u32 r12 = ((r11 >> 1) ^ hier_flag_xora) & (((r11 >> 2) ^ hier_flag_xorb) | hier_flag_orb) & 2 /* 0b10 */;
          tmp_cur_hi = r8 | r9 | r10 | r11 | r12;
        }
        shared_state[threadIdx.x] = tmp_cur_hi;
      }
      __syncthreads();

      // write out
      if((writeout_hook_i >> 8) == bs_i) {
        shared_writeouts[threadIdx.x] = shared_state[writeout_hook_i & 255];
      }

      // Macro phases scheduled before stage bs_i+1 run here. They sit just
      // ahead of that stage's data, which script_pi already points past.
      if(next_ph) {
        usize ph_pi = script_pi - next_ph_words;
        for(int q = 0; q < next_ph; ++q) {
          gem_macro_phase(script + ph_pi, shared_state, sram_data);
          ph_pi += GEM_MACRO_PHASE_WORDS;
        }
        ph_cursor += next_ph;
      }
    }
    __syncthreads();

    // phases scheduled after the final boomerang stage
    while(ph_cursor < num_macro_phases) {
      gem_macro_phase(script + script_pi, shared_state, sram_data);
      script_pi += GEM_MACRO_PHASE_WORDS;
      ph_cursor += 1;
    }

    // sram & duplicate permutation
    u32 sram_duplicate_t = 0;
#define GEMV1_SHUF_SRAM_DUPL_K(k_outer, k_inner, t_shuffle) { \
      u32 k = k_outer * 4 + k_inner;                          \
      u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);      \
      u32 t_shuffle_2_idx = t_shuffle >> 16;                  \
                                                              \
      sram_duplicate_t |=                                     \
        (shared_writeouts[t_shuffle_1_idx >> 5] >>            \
         (t_shuffle_1_idx & 31) & 1) << (k * 2);              \
      sram_duplicate_t |=                                     \
        (shared_writeouts[t_shuffle_2_idx >> 5] >>            \
         (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);          \
    }
#define GEMV1_SHUF_SRAM_DUPL_K_4(k_outer, t_shuffle) {  \
      GEMV1_SHUF_SRAM_DUPL_K(k_outer, 0, t_shuffle.c1); \
      GEMV1_SHUF_SRAM_DUPL_K(k_outer, 1, t_shuffle.c2); \
      GEMV1_SHUF_SRAM_DUPL_K(k_outer, 2, t_shuffle.c3); \
      GEMV1_SHUF_SRAM_DUPL_K(k_outer, 3, t_shuffle.c4); \
    }
    script_pi += 256 * 4 * 5;
    GEMV1_SHUF_SRAM_DUPL_K_4(0, t4_1);
    t4_1.read(((const VectorRead4 *)(script + script_pi)) + threadIdx.x);
    GEMV1_SHUF_SRAM_DUPL_K_4(1, t4_2);
    t4_2.read(((const VectorRead4 *)(script + script_pi + 256 * 4)) + threadIdx.x);
    GEMV1_SHUF_SRAM_DUPL_K_4(2, t4_3);
    t4_3.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 2)) + threadIdx.x);
    GEMV1_SHUF_SRAM_DUPL_K_4(3, t4_4);
    t4_4.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 3)) + threadIdx.x);
#undef GEMV1_SHUF_SRAM_DUPL_K_4
#undef GEMV1_SHUF_SRAM_DUPL_K
    sram_duplicate_t = (sram_duplicate_t & ~t4_5.c2) ^ t4_5.c1;
    t4_5.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 4)) + threadIdx.x);

    // sram read fires here.
    u32 *ram = nullptr;
    u32 r, w0;
    u32 port_w_addr_iv, port_w_wr_en, port_w_wr_data_iv;
    if(threadIdx.x < num_srams * 4) {
      u32 addrs = sram_duplicate_t;
      u32 last_tid = 32 + threadIdx.x / 32 * 32;
      u32 mask = (last_tid <= num_srams * 4)
        ? 0xffffffff : (0xffffffff >> (last_tid - num_srams * 4));
      port_w_wr_en = __shfl_down_sync(mask, sram_duplicate_t, 1);
      port_w_wr_data_iv = __shfl_down_sync(mask, sram_duplicate_t, 2);

      if(threadIdx.x % 4 == 0) {
        u32 sram_i = threadIdx.x / 4;
        u32 sram_st = sram_offset + sram_i * (1 << 13);
        // u32 sram_ed = sram_st + (1 << 13);
        u32 port_r_addr_iv = addrs & 0xffff;
        port_w_addr_iv = addrs >> 16;

        ram = sram_data + sram_st;
        r = ram[port_r_addr_iv];
        w0 = ram[port_w_addr_iv];
      }
    }
    // __syncthreads();

    // clock enable permutation
    u32 clken_perm = 0;
#define GEMV1_SHUF_CLKEN_K(k_outer, k_inner, t_shuffle) { \
      u32 k = k_outer * 4 + k_inner;                      \
      u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);  \
      u32 t_shuffle_2_idx = t_shuffle >> 16;              \
                                                          \
      clken_perm |=                                       \
        (shared_writeouts[t_shuffle_1_idx >> 5] >>        \
         (t_shuffle_1_idx & 31) & 1) << (k * 2);          \
      clken_perm |=                                       \
        (shared_writeouts[t_shuffle_2_idx >> 5] >>        \
         (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);      \
    }
#define GEMV1_SHUF_CLKEN_K_4(k_outer, t_shuffle) {  \
      GEMV1_SHUF_CLKEN_K(k_outer, 0, t_shuffle.c1); \
      GEMV1_SHUF_CLKEN_K(k_outer, 1, t_shuffle.c2); \
      GEMV1_SHUF_CLKEN_K(k_outer, 2, t_shuffle.c3); \
      GEMV1_SHUF_CLKEN_K(k_outer, 3, t_shuffle.c4); \
    }
    script_pi += 256 * 4 * 5;
    GEMV1_SHUF_CLKEN_K_4(0, t4_1);
    GEMV1_SHUF_CLKEN_K_4(1, t4_2);
    GEMV1_SHUF_CLKEN_K_4(2, t4_3);
    GEMV1_SHUF_CLKEN_K_4(3, t4_4);
#undef GEMV1_SHUF_CLKEN_K
#undef GEMV1_SHUF_CLKEN_K_4

    // DSP48E2 gather. Done with the whole warp active so the shuffles are
    // well-formed; the guarded block below just picks the group leaders.
    u32 dsp_g1 = __shfl_down_sync(0xffffffffu, sram_duplicate_t, 1);
    u32 dsp_g2 = __shfl_down_sync(0xffffffffu, sram_duplicate_t, 2);
    u32 dsp_g3 = __shfl_down_sync(0xffffffffu, sram_duplicate_t, 3);

    // sram commit
    if(threadIdx.x < num_srams * 4) {
      if(threadIdx.x % 4 == 0) {
        u32 sram_i = threadIdx.x / 4;
        shared_writeouts[num_ios - num_srams + sram_i] = r;
        ram[port_w_addr_iv] = (w0 & ~port_w_wr_en) | (port_w_wr_data_iv & port_w_wr_en);
      }
    }
    else if(threadIdx.x < num_srams * 4 + num_dsps * 4) {
      // A DSP occupies 4 gather lanes: 121 input bits, the clock enable, and
      // the ALU configuration carried as constants in the spare bits.
      u32 rel = threadIdx.x - num_srams * 4;
      if(rel % 4 == 0) {
        int dsp_i = rel / 4;
        u32 w[4] = { sram_duplicate_t, dsp_g1, dsp_g2, dsp_g3 };
#define GEM_GB(i) ((w[(i) >> 5] >> ((i) & 31)) & 1u)
        unsigned long long ua = 0, ub = 0, uc = 0, ud = 0;
#pragma unroll
        for(int i = 0; i < 27; ++i) ua |= (unsigned long long)GEM_GB(0 + i) << i;
#pragma unroll
        for(int i = 0; i < 18; ++i) ub |= (unsigned long long)GEM_GB(27 + i) << i;
#pragma unroll
        for(int i = 0; i < 48; ++i) uc |= (unsigned long long)GEM_GB(45 + i) << i;
#pragma unroll
        for(int i = 0; i < 27; ++i) ud |= (unsigned long long)GEM_GB(93 + i) << i;
        u32 cep     = GEM_GB(120);
        u32 clken   = GEM_GB(121);
        u32 st_code = GEM_GB(122) | (GEM_GB(123) << 1);
        u32 preadd  = GEM_GB(124);
#undef GEM_GB
        long long a = gem_sext(ua, 27);
        long long b = gem_sext(ub, 18);
        long long c = gem_sext(uc, 48);
        long long d = gem_sext(ud, 27);

        const unsigned long long p_mask = (1ull << 48) - 1;
        int off = dsp_state_base + dsp_i * 2;
        unsigned long long p_cur =
          ((unsigned long long)sram_data[off])
          | (((unsigned long long)sram_data[off + 1]) << 32);
        p_cur &= p_mask;

        unsigned long long p_next = p_cur;
        if(clken && cep) {
          // AD wraps at 27 bits, M is exact at 45, P wraps at 48.
          long long ad = gem_sext(
            (unsigned long long)(preadd ? (a + d) : a) & ((1ull << 27) - 1), 27);
          long long m = gem_sext(
            (unsigned long long)(ad * b) & ((1ull << 45) - 1), 45);
          long long v;
          if(st_code == 0) v = c;
          else if(st_code == 1) v = m;
          else if(st_code == 2) v = gem_sext(p_cur, 48) + m;
          else v = gem_sext(p_cur, 48);
          p_next = (unsigned long long)v & p_mask;
        }
        sram_data[off] = (u32)p_next;
        sram_data[off + 1] = (u32)((p_next >> 32) & 0xffffu);

        int wb = num_ios - num_srams - num_output_duplicates
               - dsp_words + dsp_i * 2;
        shared_writeouts[wb] = (u32)p_next;
        shared_writeouts[wb + 1] = (u32)((p_next >> 32) & 0xffffu);
      }
    }
    else if(threadIdx.x < num_srams * 4 + num_dsps * 4 + num_output_duplicates) {
      shared_writeouts[num_ios - num_srams - num_output_duplicates
        + (threadIdx.x - num_srams * 4 - num_dsps * 4)] = sram_duplicate_t;
    }

    __syncthreads();
    u32 writeout_inv = shared_writeouts[threadIdx.x];

    clken_perm = (clken_perm & ~t4_5.c2) ^ t4_5.c1;
    writeout_inv ^= t4_5.c3;

    if(threadIdx.x < num_ios) {
      u32 old_wo = input_state[io_offset + threadIdx.x];
      u32 wo = (old_wo & ~clken_perm) | (writeout_inv & clken_perm);
      output_state[io_offset + threadIdx.x] = wo;
    }

    if(is_last_part) break;
  }
  assert(script_size == script_pi);
}

__global__ void simulate_v1_noninteractive_simple_scan(
  usize num_blocks,
  usize num_major_stages,
  const usize *__restrict__ blocks_start,
  const u32 *__restrict__ blocks_data,
  u32 *__restrict__ sram_data,
  usize num_cycles,
  usize state_size,
  u32 *__restrict__ states_noninteractive
  )
{
  assert(num_blocks == gridDim.x);
  assert(256 == blockDim.x);
  __shared__ u32 shared_metadata[256];
  __shared__ u32 shared_writeouts[256];
  __shared__ u32 shared_state[256];
  __shared__ u32 script_starts[32], script_sizes[32];
  assert(num_major_stages <= 32);
  if(threadIdx.x < num_major_stages) {
    script_starts[threadIdx.x] = blocks_start[threadIdx.x * num_blocks + blockIdx.x];
    script_sizes[threadIdx.x] = blocks_start[threadIdx.x * num_blocks + blockIdx.x + 1] - script_starts[threadIdx.x];
  }
  __syncthreads();
  for(usize cycle_i = 0; cycle_i < num_cycles; ++cycle_i) {
    for(usize stage_i = 0; stage_i < num_major_stages; ++stage_i) {
      simulate_block_v1(
        blocks_data + script_starts[stage_i],
        script_sizes[stage_i],
        states_noninteractive + cycle_i * state_size,
        states_noninteractive + (cycle_i + 1) * state_size,
        sram_data,
        shared_metadata, shared_writeouts, shared_state
        );
      cooperative_groups::this_grid().sync();
    }
  }
}
