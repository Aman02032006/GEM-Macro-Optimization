# Welcome to GEM
GEM is an open-source RTL logic simulator with CUDA acceleration, developed and maintained by NVIDIA Research.
GEM can deliver up to 5--40X speed-up compared to CPU-based leading RTL simulators.
A summary of the work with paper can be found [here](https://research.nvidia.com/publication/2025-06_gem-gpu-accelerated-emulator-inspired-rtl-simulation).

## Compile and Run Your Design with GEM
GEM works in a way similar to an FPGA-based RTL emulator.
It first synthesizes your design with a special and-inverter graph (AIG) process, and then map the synthesized gate-level netlist to a virtual manycore Boolean processor which can be emulated with CUDA-compatible GPUs.

The synthesis and mapping is slower than the compiling/elaboration process of CPU-based simulators. But it is a one-time cost and your design can be simulated under different testbenches without re-running the synthesis or mapping.

**See [usage.md](./usage.md) for usage documentation.**

---

# Fork: Heterogeneous Word-Level Macro Evaluation

This fork extends GEM to evaluate three Xilinx hardware macros natively on the GPU ALU instead of shredding them into the And-Inverter Graph: **DSP48E2** (simplified MAC subset), **CARRY4** (4-bit carry chain), and **SRLC32E** (32-bit shift register LUT).

Preventing the frontend from flattening these primitives keeps the compiled execution script dramatically smaller. Because GEM re-reads that script from global memory on every simulated cycle, script size is a per-cycle bandwidth cost rather than a one-time compile cost. Profiling of the unmodified simulator on the RISC-V Rocket core measured **89.31 % of peak DRAM bandwidth**, confirming that GEM at scale is memory-bound rather than compute-bound.

**New to GEM and just want to simulate your own Verilog?** Start with [docs/SIMULATION_GUIDE.md](./docs/SIMULATION_GUIDE.md), a step-by-step manual covering dependencies, synthesis, mapping, stimulus and troubleshooting.

See [docs/TECHNICAL_REPORT.md](./docs/TECHNICAL_REPORT.md) for the full architecture, verification methodology and measured results.

## Modified Repository Structure

### Added directories

| Path | Purpose |
|---|---|
| `docs/` | Documentation. [`SIMULATION_GUIDE.md`](./docs/SIMULATION_GUIDE.md) is a step-by-step manual for simulating your own RTL from scratch, including every dependency and a troubleshooting section. `TECHNICAL_REPORT.md` covers the modified scheduling model, memory layout, CUDA architecture and numerical analysis. |
| `golden_models/` | Cycle-accurate C++/CUDA reference models for the three primitives (`carry4_test.cu`, `dsp_test.cu`, `srlc32e_test.cu`). Each compiles standalone for CPU or GPU and carries its own testbench: CARRY4 is verified exhaustively over all 1,024 input vectors; DSP48E2 and SRLC32E use boundary-biased stimulus targeting the 27-bit pre-adder wrap and 48-bit accumulator wrap. |
| `scripts/` | Yosys interception scripts and benchmark automation. `macro.ys` preserves the macros; `baseline.ys` shreds them for the control comparison; `macros_bb.v` declares the blackboxes; `macros_behav.v` provides synthesizable behavioural models used to build the baseline; `bench.sh` drives Nsight Compute and emits CSV. |
| `tests/` | Netlist-level regression cases and `macro_gold_check.py`, an independent Python reference written from the problem statement (not transcribed from the Rust or CUDA sources, so a shared misreading cannot cancel out). |
| `test_data/` | The macro-rich micro-benchmark (`macrobench/macro_bench.v`, 16 MAC lanes yielding 104 CARRY4, 16 DSP48E2 and 32 SRLC32E instances). |

### Added source module

| Path | Purpose |
|---|---|
| `src/macros.rs` | Single source of truth for the intercepted primitives: pin names, bit widths, slot ordering, per-output combinational fan-in, and the script encoding for macro phases. Consumed by both `aigpdk.rs` (netlist pin typing) and `aig.rs` (graph construction) so the two cannot drift apart. Also defines the cell-type mangling contract (`DSP48E2_MAC_PREADD` and similar), required because cell parameters do not survive `netlistdb`. |

### Heavily modified upstream files

| Path | Nature of change |
|---|---|
| `src/pe.rs` | Partition scheduler. Adds mid-partition macro phases, carry-chain grouping for the warp scan, per-output dependency modelling, and macro-aware write-out accounting. Largest single change in the fork. |
| `csrc/kernel_v1_impl.cuh` | CUDA kernel. Adds the macro evaluation phase between boomerang stages, including the segmented Kogge-Stone `__shfl_up_sync` carry scan and the SRLC32E read-then-shift path, plus the DSP48E2 commit alongside the existing SRAM commit. |
| `src/flatten.rs` | Memory formatter. Emits the macro script sections, allocates macro outputs into spare `hier[1]` shared-memory slots, and lays out 64-bit aligned macro state in block-affinity order for coalesced access. |
| `src/aig.rs` | Graph construction. Adds the `Macro` driver and endpoint kinds, per-output fan-in traversal, and macro clock tracing. |
| `src/bin/naive_sim.rs` | CPU reference simulator, extended to evaluate all three macros at netlist level. Provides the ground-truth values for the verification ladder. |
| `src/bin/flatten_test.rs` | CPU script emulator, extended to execute macro phases. Validates the emitted script without requiring CUDA. |
| `csrc/kernel_v1.cu` | Adds a cooperative-launch occupancy check that fails with a diagnostic message when the requested grid cannot be co-resident. |
| `src/aigpdk.rs` | Netlist pin typing for the macro cell types, delegating to `src/macros.rs`. |
| `build.rs`, `.cargo/config.toml` | Toolchain configuration for CUDA 13.x, GCC 13+ and modern Rust. See comments in `scripts/gxx-cstdint` for the rationale behind each workaround. |

## Verification

Correctness is established through a layered ladder, each rung isolating one component:

```
Independent Python model  ->  naive_sim (netlist)  ->  flatten_test (script)  ->  cuda_test (GPU)
```

The strongest result is a same-RTL equivalence check: the identical design compiled twice — once with macros shredded to AIG, once with them evaluated natively — produces **bit-identical GPU waveforms**.

```
cargo build --release --features=cuda
yosys -s scripts/macrobench.ys          # macros preserved
yosys -s scripts/macrobench_base.ys     # macros shredded (baseline)
./target/release/cut_map_interactive build/macrobench.gv build/macrobench.gemparts
python3 tests/macro_gold_check.py ./target/release/naive_sim
```

## Known Limitations

- **`.gemparts` binary incompatibility.** Two fields were added to the `Partition` struct, which `serde_bare` serialises positionally. Partition files produced by upstream GEM cannot be read by this fork, and vice versa. Regenerate with `cut_map_interactive`.
- **MIPS flatten ordering.** A small MIPS processor used during development maps successfully but fails during script generation on some partitionings. `mt-kahypar` is multi-threaded and non-deterministic, so reproduction requires a fixed seed. That design is not bundled here, so `scripts/macro.ys` and `scripts/baseline.ys` are included as references rather than runnable scripts.
- **Benchmark coverage.** The macro-versus-baseline speedup is demonstrated on the macro-rich micro-benchmark. The supplied industrial designs (Rocket, Gemmini, NVDLA) ship as post-AIG netlists containing no macro cells and no RTL, so macro interception cannot be applied to them; they provide baseline measurements only.

---

## Citation
Please cite our paper if you find GEM useful.

``` bibtex
@inproceedings{gem,
 author = {Guo, Zizheng and Zhang, Yanqing and Wang, Runsheng and Lin, Yibo and Ren, Haoxing},
 booktitle = {Proceedings of the 62nd Annual Design Automation Conference 2025},
 organization = {IEEE},
 title = {{GEM}: {GPU}-Accelerated Emulator-Inspired {RTL} Simulation},
 year = {2025}
}
```
