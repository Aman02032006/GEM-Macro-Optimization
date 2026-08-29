// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "kernel_v1_impl.cuh"

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

extern "C"
void simulate_v1_noninteractive_simple_scan_cuda(
  usize num_blocks,
  usize num_major_stages,
  const usize *blocks_start,
  const u32 *blocks_data,
  u32 *sram_data,
  usize num_cycles,
  usize state_size,
  u32 *states_noninteractive
  )
{
  // Cooperative launch requires the ENTIRE grid to be co-resident: every
  // block must fit on the device simultaneously, or the launch fails outright
  // rather than time-slicing. Register pressure is what decides this, and the
  // margin is thin by construction -- with -maxrregcount=128 and 256 threads,
  // a block claims 256*128 = 32768 of an SM's 65536 registers, i.e. exactly
  // 2 blocks/SM, matching the 2x-SM grid size documented in usage.md.
  //
  // So any register growth in the macro phase shows up here as a hard launch
  // failure with an opaque message. Check it up front and say what happened.
  {
    int dev = 0;
    checkCudaErrors(cudaGetDevice(&dev));
    int num_sms = 0;
    checkCudaErrors(cudaDeviceGetAttribute(
      &num_sms, cudaDevAttrMultiProcessorCount, dev));

    int coop = 0;
    checkCudaErrors(cudaDeviceGetAttribute(
      &coop, cudaDevAttrCooperativeLaunch, dev));
    if (!coop) {
      printf("FATAL: device %d does not support cooperative kernel launch, "
             "which GEM's major-stage grid sync requires.\n", dev);
      exit(EXIT_FAILURE);
    }

    int blocks_per_sm = 0;
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, (void *)simulate_v1_noninteractive_simple_scan,
      256, 0));

    usize max_coop_grid = (usize)blocks_per_sm * (usize)num_sms;
    if (blocks_per_sm < 2) {
      printf("WARNING: occupancy dropped to %d block/SM (was 2). The macro "
             "evaluation phase has pushed register usage past the budget; "
             "ptxas is either spilling or the block no longer fits twice. "
             "Max cooperative grid is now %zu blocks.\n",
             blocks_per_sm, (size_t)max_coop_grid);
    }
    if (num_blocks > max_coop_grid) {
      printf("FATAL: requested %zu blocks but only %zu can be co-resident "
             "(%d blocks/SM x %d SMs). A cooperative launch of this size "
             "will fail. Re-run with NUM_BLOCKS <= %zu, or reduce register "
             "pressure in simulate_block_v1.\n",
             (size_t)num_blocks, (size_t)max_coop_grid,
             blocks_per_sm, num_sms, (size_t)max_coop_grid);
      exit(EXIT_FAILURE);
    }
  }

  void *arg_ptrs[8] = {
    (void *)&num_blocks, (void *)&num_major_stages,
    (void *)&blocks_start, (void *)&blocks_data,
    (void *)&sram_data, (void *)&num_cycles, (void *)&state_size,
    (void *)&states_noninteractive
  };
  checkCudaErrors(cudaLaunchCooperativeKernel(
    (void *)simulate_v1_noninteractive_simple_scan, num_blocks, 256,
    arg_ptrs, 0, (cudaStream_t)0
    ));
}
