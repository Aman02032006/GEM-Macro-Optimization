# Simulating Your Own RTL with GEM — A Step-by-Step Manual

This manual takes you from *"I have some Verilog files"* to *"I have an output waveform produced by a GPU."* It assumes you know Verilog and can use a Linux terminal. It does **not** assume you know anything about synthesis, graph partitioning, or CUDA.

Every command below has been run on the machine this repository was developed on. Where a command needs a value that differs per machine, the manual tells you how to find that value rather than guessing it for you.

---

## 0. First, understand what GEM is (2 minutes, worth it)

A normal simulator like Icarus Verilog or Verilator reads your RTL and *interprets* it, event by event, on a CPU. GEM does something quite different, and knowing this up front explains every restriction later in this manual.

GEM behaves like an **FPGA emulator**. It first converts your design into a giant network of two-input AND gates and inverters, then packs that network onto a "virtual many-core Boolean processor" that runs on your GPU. Thousands of gates are evaluated simultaneously, one clock cycle at a time.

Three consequences follow directly, and they are the source of almost every problem beginners hit:

**1. Compiling is slow; simulating is fast.** The synthesis and mapping steps (Steps 3 and 4 below) can take minutes. That cost is paid **once**. Afterwards you can re-simulate the same design under different stimulus as many times as you like without redoing it.

**2. Your testbench cannot be Verilog.** GEM is *non-interactive*. It cannot execute `initial` blocks, `$display`, or procedural stimulus. Instead you hand it a **VCD file** containing a pre-recorded waveform for the design's primary inputs, and it replays that waveform. Anything your testbench used to compute on the fly must be computed in advance and written into that VCD.

**3. Your design must be fully synchronous.** Everything must be flip-flops clocked by an edge. Latches, asynchronous resets on the data path, and other asynchronous sequential logic are **not supported**. Combinational loops are not supported either. If your design has clock gating written directly in RTL, you must replace each gate with an instantiation of the `CKLNQD` module found in [aigpdk/aigpdk.v](../aigpdk/aigpdk.v).

If your design violates rule 3, stop here and fix the RTL first. No amount of tool flags will work around it.

---

## 1. Dependencies

Everything below was verified on **Ubuntu 25.10 under WSL2**. Any recent Ubuntu or Debian works. The versions in the right-hand column are what this repository was tested with — you do not need these exact versions, but if something breaks, comparing against them is the fastest way to find out why.

| What | Why you need it | Tested version |
|---|---|---|
| NVIDIA GPU, compute capability ≥ 7.5 | Runs the simulation. Must support *cooperative kernel launch* (all Turing and newer do). | RTX 3050, cc 8.6 |
| NVIDIA driver | Talks to the GPU. | — |
| CUDA Toolkit ≥ 12.0 | Provides `nvcc`, which compiles GEM's GPU kernel. | 13.3 |
| Rust toolchain (via rustup) | GEM itself is written in Rust. | 1.98.0 |
| Yosys | Turns your Verilog into a gate-level netlist. | 0.68 |
| GCC / G++ and `build-essential` | Compiles GEM's C++ and CUDA parts. | 15.2.0 |
| CMake | Builds the bundled `mt-kahypar` partitioner. | 4.2.3 |
| Python 3 | Generates stimulus and runs the checking scripts. | 3.14 |
| Git | Cloning, and fetching the submodule. | — |
| Icarus Verilog *(optional)* | Produces a golden reference waveform to compare against. Strongly recommended. | — |

### Install commands

Run these in order. The first block covers everything except CUDA and Rust.

```bash
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-pip yosys iverilog
```

**Rust** — install through rustup, not apt. Ubuntu's packaged `rustc` has a compiler bug that crashes partway through GEM's build.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version        # confirm it works
```

**CUDA Toolkit** — follow NVIDIA's official installer for your distribution at <https://developer.nvidia.com/cuda-downloads>.

> **If you are on WSL2:** install the NVIDIA driver on **Windows only**, and inside WSL install *only* the toolkit, choosing the `WSL-Ubuntu` target. Installing a Linux driver inside WSL will break your GPU access.

Then make `nvcc` visible to your shell, adding this line to `~/.bashrc` so it persists:

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

### Confirm everything is present

```bash
nvidia-smi                     # should print your GPU and driver version
nvcc --version                 # should print a CUDA release
yosys -V                       # should print a Yosys version
cargo --version                # should print a cargo version
```

If `nvidia-smi` works but `nvcc` is "command not found", your toolkit is installed but not on `PATH` — revisit the `export` line above.

> **A note on Yosys versions.** The version from `apt` is fine for the standard flow in this manual. You only need a newer Yosys built from source if your RTL uses recent SystemVerilog constructs that the packaged parser rejects, or if you want the macro flow in Appendix A, which needs Yosys ≥ 0.40 for the `xilinx_srl` pass.

---

## 2. Get and build GEM

```bash
git clone <your-gem-repository-url> GEM
cd GEM
git submodule update --init --recursive
```

The submodule step is **not optional** — GEM will not compile without it.

Before building, tell the CUDA compiler which GPU to target. Find your compute capability:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

That prints something like `8.6`. Drop the dot to get `86`, then build:

```bash
UCC_CUDA_PTX=86 cargo build --release --features=cuda
```

Substitute your own number. Common values: Turing `75`, Ampere `80` or `86`, Ada `89`, Hopper `90`, Blackwell `120`.

The first build compiles the partitioner from C++ source and takes roughly **5–15 minutes**. Later builds take seconds. When it finishes you will have four programs in `target/release/`:

| Program | What it does |
|---|---|
| `cut_map_interactive` | **Compiles**: maps your netlist onto the virtual Boolean processor. |
| `cuda_test` | **Simulates**: runs the mapped design on the GPU. |
| `naive_sim` | A slow, simple CPU simulator. Used for checking correctness. |
| `flatten_test` | A CPU emulator of the GPU program. Used for debugging. |

> If the build fails, jump to [Troubleshooting](#8-troubleshooting) — the two or three failures people actually hit are listed there with fixes.

---

## 3. Prepare your RTL

Collect your Verilog into one place and identify your **top module name**. You will need it repeatedly.

Go through this checklist before continuing. Each item corresponds to a real failure mode:

- [ ] No latches — every storage element is an edge-triggered flip-flop.
- [ ] No combinational feedback loops.
- [ ] No `initial` blocks used to establish reset state. Drive reset through a port instead.
- [ ] Any RTL clock gating replaced with a `CKLNQD` instance.
- [ ] Memories are written in a recognisable style — a simple `reg [W-1:0] mem [0:N-1]` array with a **synchronous** read. Asynchronous reads are handled, but expensively.
- [ ] You know your top module's exact input port list and each port's bit width. Write it down; Step 5 needs it.

---

## 4. Synthesis — turn Verilog into a gate-level netlist

This step uses Yosys to convert your RTL into a netlist built only from cells GEM understands. These cells are defined in the `aigpdk/` directory, which ships with this repository.

Create a file called `synth.ys` in the GEM directory. Copy the template below and **edit the two marked lines** at the top.

```tcl
# ===== EDIT THESE TWO LINES =========================================
read_verilog -I rtl rtl/alu.v rtl/regfile.v rtl/top.v
hierarchy -check -top my_top_module
# ====================================================================

# --- simplify the design before mapping ---
proc;;
opt_expr; opt_dff; opt_clean

# --- recognise and map memories ---
memory -nomap
memory_libmap -lib aigpdk/memlib_yosys.txt -logic-cost-rom 100 -logic-cost-ram 100

# --- map all logic onto the GEM cell library ---
synth -flatten
delete t:$print
dfflibmap -liberty aigpdk/aigpdk_nomem.lib
opt_clean -purge
abc -liberty aigpdk/aigpdk_nomem.lib
opt_clean -purge
techmap
abc -liberty aigpdk/aigpdk_nomem.lib
opt_clean -purge

# --- write the result ---
write_verilog -noattr -noexpr build/design.gv
stat
```

Notes on the two lines you edit: add `-sv` to `read_verilog` if you are using SystemVerilog, `-I <dir>` for each include directory, and `-D NAME=value` for each define. List every source file. The `-top` name must match your top module exactly.

Run it:

```bash
mkdir -p build
yosys -s synth.ys
```

### Reading the memory report

Partway through, `memory_libmap` prints how it mapped each memory array. This is the single most important output of the whole step, so read it:

| What you see | What it means | What to do |
|---|---|---|
| `$__RAMGEM_SYNC_` | Mapped correctly to a GEM RAM. | Nothing — this is the goal. |
| `$__RAMGEM_ASYNC_` | Yosys thinks the read port is asynchronous. | If it should be synchronous, fix the RTL. A common cause is using different clocks for read and write. |
| `using FF mapping for memory` | Memory not recognised, so it becomes flip-flops and multiplexer trees. | Fine for small arrays. Very expensive for large ones — consider rewriting the memory in a more standard style. |

If your design uses asynchronous-read memories and you accept the cost of turning them into registers, add `memory_map` and `opt -full` immediately after the `memory_libmap` line.

### Check that synthesis did not change your design

Do not skip this. Synthesis bugs are far easier to find now than after mapping.

```bash
iverilog -o check.vvp build/design.gv aigpdk/aigpdk.v your_testbench.v
vvp check.vvp
```

Your existing Verilog testbench should still pass against `build/design.gv`. If it does not, the problem is in synthesis, not in GEM.

---

## 5. Mapping — compile the netlist for the GPU

```bash
./target/release/cut_map_interactive build/design.gv build/design.gemparts
```

This partitions your circuit across GPU thread blocks and writes a binary "program" to `build/design.gemparts`. Expect anywhere from seconds to several minutes depending on design size.

If it fails complaining about partitioning a circuit with 0 or 1 endpoints, your circuit is too deep to fit in one stage. Force a split:

```bash
./target/release/cut_map_interactive build/design.gv build/design.gemparts --level-split 30
```

You can pass several thresholds, e.g. `--level-split 20,40`. **Remember whichever value you used** — you must pass the identical `--level-split` argument in Step 7, or the simulation will be wrong.

Add `--top-module my_top_module` if GEM guesses the top module incorrectly.

---

## 6. Create the input waveform (VCD)

GEM replays a waveform on your design's primary inputs. You must produce that waveform yourself. There are two ways.

### Option A — record it from your existing testbench (easiest if you have one)

Add a VCD dump to your Verilog testbench:

```verilog
initial begin
    $dumpfile("build/input.vcd");
    $dumpvars(0, tb.dut);      // scope containing your design's ports
end
```

Run it with Icarus Verilog, and you now have a waveform GEM can replay. Note the scope name you dumped — Step 7 needs it.

### Option B — generate it with a script

Write the VCD directly. The format is simple. Below is a complete, working generator; change only the `INPUTS` list and the loop that sets values.

```python
#!/usr/bin/env python3
"""Generate an input VCD for GEM. Usage: python3 gen_input.py build/input.vcd"""
import sys

# ===== EDIT: your top module's input ports and their widths =====
INPUTS = [("reset", 1), ("enable", 1), ("data_in", 32)]
CYCLES = 100
# ===============================================================

def bits(v, w):
    return "".join("1" if (v >> i) & 1 else "0" for i in range(w - 1, -1, -1))

out = sys.argv[1] if len(sys.argv) > 1 else "build/input.vcd"
ids, nxt = {}, 34                       # '!' (33) is reserved for clk
lines = ["$timescale 1ns $end", "$scope module top $end",
         "$var wire 1 ! clk $end"]
for nm, w in INPUTS:
    ids[nm] = chr(nxt); nxt += 1
    lines.append(f"$var wire {w} {ids[nm]} {nm} $end")
lines += ["$upscope $end", "$enddefinitions $end"]

state = {nm: 0 for nm, _ in INPUTS}
t = 0
lines += [f"#{t}", "0!"]

def clock_cycle():
    """Emit one full clock period: settle inputs, rise, fall."""
    global t
    lines.append(f"#{t}")
    for nm, w in INPUTS:
        lines.append(f"b{bits(state[nm], w)} {ids[nm]}" if w > 1
                     else f"{state[nm]}{ids[nm]}")
    t += 10; lines.extend([f"#{t}", "1!"])   # rising edge -> one simulated cycle
    t += 10; lines.extend([f"#{t}", "0!"])
    t += 10

# ===== EDIT: your stimulus =====
state["reset"] = 1
for _ in range(2):
    clock_cycle()
state["reset"] = 0
state["enable"] = 1
for i in range(CYCLES):
    state["data_in"] = i * 7
    clock_cycle()
# ===============================

open(out, "w").write("\n".join(lines) + "\n")
print(f"wrote {out}")
```

```bash
python3 gen_input.py build/input.vcd
```

Three rules govern this file, and breaking any of them produces silently wrong results:

1. **The clock must be in the VCD and must toggle.** Each rising edge advances the simulation by exactly one cycle. A VCD without clock transitions simulates zero cycles.
2. **Drive every primary input.** An input you never assign is undefined.
3. **Change inputs while the clock is low**, as the generator above does. Changing them on the edge itself makes the result depend on ordering you do not control.

---

## 7. Simulate on the GPU

One value is still missing: `NUM_BLOCKS`, which should be **twice your GPU's streaming-multiprocessor count**. Find it with:

```bash
python3 -c "
import ctypes
cu = ctypes.CDLL('libcuda.so.1'); cu.cuInit(0)
d = ctypes.c_int(); cu.cuDeviceGet(ctypes.byref(d), 0)
n = ctypes.c_int(); cu.cuDeviceGetAttribute(ctypes.byref(n), 16, d)
print('SMs =', n.value, '-> NUM_BLOCKS =', n.value * 2)"
```

On the development machine this prints `SMs = 20 -> NUM_BLOCKS = 40`.

Now run the simulation:

```bash
./target/release/cuda_test \
    build/design.gv \
    build/design.gemparts \
    build/input.vcd \
    build/output.vcd \
    40 \
    --input-vcd-scope top
```

The arguments are **positional and order matters**: netlist, mapped program, input VCD, output VCD, number of blocks.

`--input-vcd-scope` must match the scope your input VCD actually uses. In the generator above that is `top`. If you recorded from a testbench with `$dumpvars(0, tb.dut)`, use `--input-vcd-scope tb.dut`. Getting this wrong is the most common cause of "all outputs are zero or X".

Add `--output-vcd-scope my_top` if you want a specific scope name in the output; otherwise GEM writes `gem_top_module`.

Your results are now in `build/output.vcd`. Open it with GTKWave:

```bash
gtkwave build/output.vcd
```

> **On timing:** GEM prints its GPU runtime separately from total wall-clock time. Most of the wall-clock time is spent parsing the input VCD, not simulating. Do not mistake one for the other when benchmarking.

---

## 8. Troubleshooting

### Build problems

| Symptom | Cause | Fix |
|---|---|---|
| `nvcc fatal: Unsupported gpu architecture 'compute_70'` | CUDA 12+ dropped older architectures; GEM's default targets are stale. | Set `UCC_CUDA_PTX` to your own compute capability, as in Step 2. |
| `error: 'uint8_t' has not been declared` in `mt-kahypar` | GCC 13+ no longer includes `<cstdint>` transitively. | Handled automatically by [.cargo/config.toml](../.cargo/config.toml). Ensure you build from inside the GEM directory and have not overridden `CXX` in your shell. |
| Internal compiler error mentioning `mismatched_lifetime_syntaxes` | Ubuntu's packaged `rustc`. | Install Rust through rustup, as in Step 1. |
| `cannot find -lmtkahypar` or missing submodule files | Submodule never fetched. | `git submodule update --init --recursive` |

### Mapping problems

| Symptom | Cause | Fix |
|---|---|---|
| "partitioning a circuit with 0/1 endpoints" | Circuit too deep for a single stage. | Add `--level-split 30` (and use the same value when simulating). |
| Guessed the wrong top module | Multiple candidate top modules in the netlist. | Pass `--top-module <name>` explicitly. |
| Mapping runs for a very long time | Large design; the partitioner is genuinely slow. | Expected. It is a one-time cost — reuse the `.gemparts` file. |

### Simulation problems

| Symptom | Cause | Fix |
|---|---|---|
| All outputs zero or `x` | Input VCD scope wrong, so no stimulus is being applied. | Fix `--input-vcd-scope` to match your VCD's `$scope`. |
| `cudaErrorCooperativeLaunchTooLarge`, or a message about co-residency | `NUM_BLOCKS` exceeds what fits on your GPU simultaneously. | Lower it. GEM prints the maximum it can accept. |
| Output waveform is shorter than expected | Not enough clock edges in the input VCD. | Add more cycles to your stimulus. |
| Results differ from your Verilog testbench | Usually a `--level-split` mismatch, or asynchronous logic in the design. | Ensure the mapping and simulation `--level-split` match. Then re-check the Step 3 checklist. |

### When results are wrong and you cannot tell why

GEM ships two CPU tools that let you find *which stage* introduced the error. Run them in this order — the first one that disagrees with the previous is where your problem lives.

```bash
# 1. Simple CPU simulation directly from the netlist.
./target/release/naive_sim build/design.gv build/input.vcd build/out_naive.vcd \
    --input-vcd-scope top

# 2. CPU emulation of the compiled GPU program.
./target/release/flatten_test build/design.gv build/design.gemparts \
    build/input.vcd build/out_flat.vcd --input-vcd-scope top

# 3. The real GPU run (Step 7) -> build/output.vcd
```

Interpreting the result:

- **`naive_sim` already disagrees with your Verilog testbench** → the problem is in synthesis (Step 4), not in GEM.
- **`naive_sim` is right but `flatten_test` is wrong** → the problem is in mapping or code generation (Step 5).
- **`flatten_test` is right but the GPU is wrong** → a GPU-side issue; report it with your netlist.

---

## 9. Quick reference

The whole flow, once you know it:

```bash
# One-time setup
git submodule update --init --recursive
UCC_CUDA_PTX=86 cargo build --release --features=cuda

# Per design — compile once
yosys -s synth.ys                                                  # RTL   -> build/design.gv
./target/release/cut_map_interactive build/design.gv build/design.gemparts

# Per test — simulate as often as you like
python3 gen_input.py build/input.vcd
./target/release/cuda_test build/design.gv build/design.gemparts \
    build/input.vcd build/output.vcd 40 --input-vcd-scope top
gtkwave build/output.vcd
```

---

## Appendix A — Optional: keeping DSP, carry-chain and shift-register macros intact

This is a feature of **this fork**, not of upstream GEM. It is entirely optional and the flow above works without it.

Standard GEM shreds every arithmetic operator into AND gates and inverters. A 48-bit multiply-accumulate becomes thousands of gates. This fork can instead recognise three Xilinx primitives and evaluate them directly on the GPU's arithmetic units, which makes the compiled program dramatically smaller — and since GEM re-reads that program from memory every single cycle, smaller means faster.

The three supported primitives are declared in [scripts/macros_bb.v](../scripts/macros_bb.v):

| Macro | What it is |
|---|---|
| `CARRY4` | 4-bit carry chain, the building block of fast adders. |
| `DSP48E2_MAC_PREADD` | Multiply-accumulate with a pre-adder: `P += (A + D) * B`. |
| `SRLC32E` | 32-bit addressable shift register. |

You benefit if your design contains wide adders, multipliers, MAC units, or long shift-register delay lines. To use it, add three things to your `synth.ys`.

**First**, read the blackbox declarations before anything else, so Yosys knows these modules exist and never tries to synthesize into them:

```tcl
read_verilog -lib scripts/macros_bb.v
```

**Second**, after `opt_clean` and before the general `techmap`, add the extraction passes:

```tcl
# Gather +, - and comparators into $alu / $lcu cells. Without this the adders
# are already broken apart and there is nothing left to extract.
alumacc
opt -full

# $alu / $lcu -> CARRY4. Counter-intuitively LUT_SIZE=6 selects the CARRY4
# path; LUT_SIZE=4 takes an older, different branch that GEM does not accept.
techmap -D LUT_SIZE=6 -map +/xilinx/arith_map.v
opt_clean

# Shift-register chains -> SRLC32E.
xilinx_srl -variable -minlen 3
opt_clean
```

**Third**, confirm it worked. The `stat` line at the end of your script prints a cell inventory; you should see the macros listed:

```
   104   CARRY4
    16   DSP48E2_MAC_PREADD
    32   SRLC32E
```

Mapping then reports how many it took over:

```
[INFO] intercepted 152 word-level macros: 104 CARRY4, 16 DSP48E2, 32 SRLC32E
```

Everything from Step 5 onward is unchanged. For a complete working example that you can run as-is, see [scripts/macrobench.ys](../scripts/macrobench.ys). [scripts/macro.ys](../scripts/macro.ys) shows the same idea applied to a small MIPS processor and is useful to read, but note that the MIPS RTL it references is not bundled in this repository, so it will not run without supplying that design yourself.

DSP48E2 instances are not inferred automatically — instantiate `DSP48E2_MAC_PREADD` directly in your RTL where you want one. The architecture and the verification evidence behind all of this are documented in [TECHNICAL_REPORT.md](./TECHNICAL_REPORT.md).

---

## Appendix B — Where to look next

| File | Contents |
|---|---|
| [usage.md](../usage.md) | Upstream NVIDIA documentation, including the Synopsys DC flow if you have a commercial synthesis licence. |
| [TECHNICAL_REPORT.md](./TECHNICAL_REPORT.md) | How the macro extension works internally, and how it was verified. |
| [scripts/](../scripts/) | Working Yosys scripts you can copy from. |
| [aigpdk/](../aigpdk/) | The cell library your netlist is synthesized against. |
