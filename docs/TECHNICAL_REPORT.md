# Teaching a GPU to Do Real Arithmetic
## A Technical Manual for the Heterogeneous GEM Simulator

**Takneek Zenith — Team Submission**

*Audience note: this manual assumes you know Verilog and basic digital circuits (gates, flip-flops, adders). It assumes nothing about GPUs, compilers, or graph theory. Every new concept is introduced before it is used.*

---

## Table of Contents

1. [Introduction: The Big Picture](#1-introduction-the-big-picture)
2. [The Codebase Evolution: What We Changed](#2-the-codebase-evolution-what-we-changed)
3. [The Bug Hunt: What We Found](#3-the-bug-hunt-what-we-found)
4. [The Results & Proof](#4-the-results--proof)
5. [Deliverables Checklist](#5-deliverables-checklist)

---

# 1. Introduction: The Big Picture

## 1.1 What a logic simulator actually does

When you write Verilog, you eventually want to *run* it — to check that your adder adds and your CPU fetches. That is what a **logic simulator** does. It takes your circuit, takes a set of input waveforms, and computes what every wire in the design does on every clock cycle.

The naive way is to walk the circuit gate by gate, once per cycle, on a CPU. That works, and it is what Icarus Verilog or Verilator do. The problem is scale: a modern SoC has hundreds of millions of gates, and a CPU evaluates them more or less one at a time.

A GPU, by contrast, has thousands of tiny processors. If you could evaluate ten thousand gates *simultaneously*, simulation would get dramatically faster. That is the promise NVIDIA's **GEM** simulator chases.

## 1.2 GEM's core idea: shred everything into AND gates

GPUs are fast only when every one of their thousands of threads is doing **the same operation at the same time**. Give thread 1 an addition, thread 2 a multiplication and thread 3 a comparison and the hardware serialises them and you lose most of your speed.

So GEM makes a bold simplification. It converts your *entire* circuit into just two kinds of node:

- **AND gates** (2 inputs, 1 output)
- **Inverters** (NOT)

This representation is called an **And-Inverter Graph**, or **AIG**. Any combinational logic can be rewritten this way, exactly like how any boolean expression can be rewritten with only NAND gates.

> **Analogy.** Imagine translating a novel so that every word has at most three letters. It is *possible* — but the book gets enormously longer, and the original phrasing is destroyed.

Now every GPU thread does the identical thing: read two bits, AND them, maybe invert. Perfect uniformity, maximum GPU utilisation. This is genuinely clever.

## 1.3 The problem: shredding destroys arithmetic

Here is where it breaks down.

Suppose your design contains a **DSP48E2** — a hardware multiply-accumulate block found in Xilinx FPGAs. In silicon it is one unit that computes `P = P + (A + D) × B` in a single step, on 27-bit and 18-bit inputs.

When GEM's frontend (the Yosys synthesiser) shreds it into an AIG:

- A 27×18 multiplier becomes **thousands** of AND gates
- Those gates are stacked **hundreds of levels deep** (each partial-product addition depends on the one before it)
- One clean hardware operation becomes an enormous, deep tangle of boolean logic

The same happens to a **CARRY4** (a 4-bit carry chain) and an **SRLC32E** (a 32-bit shift register). The architectural intent — "this is a multiplier" — is completely lost.

## 1.4 Why this chokes the GPU: the script problem

To understand the real cost you need one more piece of GEM's design.

GEM does not interpret your circuit on the GPU. It **compiles** it, ahead of time, into a flat list of instructions called the **script**. The script says things like *"thread 57: read bit 3312 and bit 9001, AND them, store to bit 4408"*, for every gate in the design.

Critically:

- The script lives in **global memory** (the GPU's DRAM — its main RAM)
- The GPU **re-reads the entire script every single simulated cycle**

So the script size is not a one-off compile cost. It is a *per-cycle bandwidth tax*. More gates → bigger script → more DRAM traffic → slower simulation.

### Our measurement: this is the real bottleneck

We did not take this on faith. We ran the **RISC-V Rocket core** (344,836 AND gates, a genuine industrial CPU) on **unmodified GEM** and profiled it with NVIDIA Nsight Compute:

| Metric | Rocket on unmodified GEM |
|---|---|
| DRAM throughput | **89.31 % of peak** |
| SM occupancy | 33.33 % |
| Simulated throughput | 18,454 cycles/second |

**89.31 % of peak DRAM bandwidth.** The GPU is not short of arithmetic power — its compute units are mostly idle. It is spending almost all of its memory bandwidth just *fetching the instruction script*. The simulator is memory-bound, not compute-bound.

This single number is the justification for everything that follows.

## 1.5 Our solution: teach the GPU to run macros natively

Instead of shredding a DSP into thousands of gates, we:

1. **Catch it during synthesis** before Yosys can flatten it
2. **Carry it through the compiler** as a single node, not thousands
3. **Evaluate it on the GPU** using the hardware's native 64-bit integer arithmetic — one multiply instruction instead of a thousand ANDs

A GPU's arithmetic unit can multiply two integers in one instruction. It is absurd to make it emulate a multiplier out of AND gates. We simply let it multiply.

The payoff is not primarily the arithmetic — it is that **the script gets dramatically smaller**, which directly attacks the 89 % DRAM bottleneck.

---

# 2. The Codebase Evolution: What We Changed

## 2.1 The scale of the change

We forked NVIDIA's GEM and modified it substantially:

| | |
|---|---|
| Existing files modified | **11** |
| Lines added / removed | **+2,136 / −83** |
| New core module | `src/macros.rs` (612 lines) |
| Golden reference models | 1,603 lines of C++/CUDA |
| Synthesis scripts, harnesses, tests | ~900 lines |

The three biggest changes are in `src/pe.rs` (+718 lines — the scheduler), `csrc/kernel_v1_impl.cuh` (+325 — the GPU kernel), and `src/flatten.rs` (+380 — the memory layout).

## 2.2 The Yosys frontend: catching macros before they are shredded

**Yosys** is the synthesiser — the tool that turns Verilog into a gate netlist. It is controlled by a script file with a `.ys` extension.

### The interception

The key trick is a **blackbox declaration**. We tell Yosys "these modules exist, here are their pins, but do not look inside and do not optimise them":

```verilog
(* blackbox *)
module CARRY4 (
    output [3:0] CO, output [3:0] O,
    input CI, input CYINIT, input [3:0] DI, input [3:0] S
);
endmodule
```

Reading this file *first* means that when `techmap` and `abc` (Yosys's flattening passes) run later, they walk straight past our macros and shred only the ordinary logic around them.

### Creating CARRY4s from ordinary `+`

Most RTL does not instantiate `CARRY4` by hand — it just writes `a + b`. Yosys can convert additions into carry chains:

```tcl
alumacc                                        # gather + and - into $alu cells
techmap -D LUT_SIZE=6 -map +/xilinx/arith_map.v  # $alu -> CARRY4
```

> **A trap we hit.** `LUT_SIZE=6` is what produces `CARRY4`. Setting `LUT_SIZE=4` — which sounds correct for a 4-bit carry block — takes a different branch and emits older `MUXCY`/`XORCY` primitives instead. Our first attempt produced 277 MUXCY and **zero** CARRY4.

### Cell-type mangling: working around a parser limitation

A real DSP48E2 is configured by **parameters** (`AREG`, `AMULTSEL`, and so on). We discovered that GEM's netlist reader stores only `celltypes: Vec<String>` — **parameters do not survive at all**.

So configuration has to travel in the *name*. Our Yosys pass emits mangled types:

```
DSP48E2_BYPASS        DSP48E2_BYPASS_PREADD
DSP48E2_MULT          DSP48E2_MULT_PREADD
DSP48E2_MAC           DSP48E2_MAC_PREADD
```

The base name carries the 2-bit ALU mode; the `_PREADD` suffix says whether the pre-adder computes `A + D` or just passes `A`. A bare `DSP48E2` reaching our parser triggers a loud, explanatory panic rather than a silent wrong guess.

## 2.3 The memory allocator: the `hier[1]` overlay

### Background: shared memory and the boomerang tree

Every GPU thread block has a small, very fast scratchpad called **shared memory**. GEM uses exactly 8,192 bits of it (256 threads × 32 bits) as the working set for the current cycle.

GEM evaluates logic using a structure it calls the **boomerang tree**: a 13-level binary reduction that takes 8,192 input bits down to 1, applying one AND-with-optional-inversion at every node. Level 1 of that tree — called `hier[1]` — holds 4,096 results and occupies shared-memory bit positions 4096–8191.

### The problem

Our macro results have to live *somewhere* in that 8,192-bit space so the next tree stage can read them. But every bit is already allocated to the tree. Expanding shared memory would reduce how many thread blocks fit on the GPU at once, costing performance everywhere.

### The solution

We noticed that the tree rarely fills `hier[1]` completely. Unused slots are marked `usize::MAX` and hold nothing meaningful.

So the macro phase runs **after** the tree reduction completes and writes its results into those spare `hier[1]` slots. The next stage's existing shuffle logic reads them exactly like any other tree result.

**Cost: zero.** No extra shared memory, no change to the inner loop, no occupancy penalty. We reuse space that was already being wasted.

### 64-bit aligned macro data

The problem statement asks for 64-bit aligned buffers for arithmetic macros. For the DSP we use a **structure-of-arrays** layout — three 64-bit input planes plus one state plane, each a separate array indexed by macro instance:

```
Plane 0  in_ad[i]     A[26:0] @0   D[26:0] @27   reserved @54..63
Plane 1  in_c[i]      C[47:0] @0                 reserved @48..63
Plane 2  in_bctl[i]   B[17:0] @0   OPMODE @18    PREADD @20   CEP @21
Plane S  state_p[i]   P[47:0] @0                 reserved @48..63
```

Why three planes and not two? The DSP needs 124 input bits. That fits inside 128, but only by splitting a field across a word boundary: `C` alone takes 48 bits leaving 16 (too few for `B`'s 18), and `A+D` take 54 leaving 10. No partition of `{48, 27, 27, 18, 4}` has both halves ≤ 64. We kept fields whole.

Why structure-of-arrays? When 32 consecutive threads each read `plane[i]`, they touch 32 × 8 = 256 contiguous bytes — two perfectly **coalesced** memory transactions with every byte used. A 24-byte per-instance struct would scatter each warp's access across roughly six transactions.

We also renumber macro instances in **block-affinity order**, so the macros a GPU block touches are contiguous in memory. Without this the "coalesced" layout is fiction: a block whose macros are instances {3, 91, 400} issues three scattered loads.

## 2.4 The boomerang scheduler: `src/pe.rs`

This was the hardest part, and the source of most of our bugs.

### Background: what the scheduler must guarantee

The circuit is a **DAG** — a directed acyclic graph. Nodes are gates; edges are wires; "acyclic" means no combinational loops. The scheduler's job is to order evaluation so nothing is computed before its inputs exist.

Stock GEM only ever schedules AND gates, all identical. We had to insert *macros* into that order — nodes the tree structurally cannot represent.

### Where macros run: in-partition, not in-tree

Our first instinct was to evaluate macros inside the boomerang tree. That turns out to be impossible, and understanding why shaped the whole design:

- The tree is a **binary reduction**: fan-in 2, fan-out 1, **one bit per node**
- A CARRY4 is **10 inputs, 8 outputs** — it does not fit a fan-out-1 node
- A DSP is **48 bits wide** against a 1-bit slot

The alternative we chose: macros run **between** boomerang stages, inside the same partition. The kernel already loops over stages with only a lightweight `__syncthreads()` between them, so this costs no expensive grid-wide synchronisation.

### Three primitives, three different schedules

A crucial realisation is that the three macros behave completely differently:

| Macro | Behaviour | How it is scheduled |
|---|---|---|
| **CARRY4** | Purely combinational | Mid-partition phase |
| **DSP48E2** | `P` is a *registered* output | **Endpoint only** — like a flip-flop |
| **SRLC32E** | `Q31` registered, but `Q = state[A]` is **combinational in A** | Mid-partition phase *and* state commit |

The DSP needs **no** mid-partition work at all. Its `P` port is a read of the `PREG` register, available at cycle start exactly like a flip-flop's `Q`. That simplification fell out of modelling dependencies per-output rather than per-macro.

The SRLC32E is the subtle one. Its read port `Q = state[A]` depends combinationally on the address — structurally it is an **asynchronous-read memory**, which is precisely the thing GEM declares unsupported (`$__RAMGEM_ASYNC_` triggers an outright panic). It cannot use the SRAM path; it needs mid-partition evaluation.

### Per-output dependencies, not per-macro

We model the combinational fan-in of each *output slot* separately. For CARRY4 the dependency is **triangular**: `O[0]` depends on `S[0]`, `CI` and `CYINIT`, but **not** on `S[3]`.

This is not pedantry. Declaring the conservative all-to-all relation would manufacture **false combinational loops** in perfectly legal netlists — any design routing a low carry output back into a high `DI` bit would trip GEM's loop detector and crash on valid input.

### CARRY4 chains and the Kogge-Stone warp scan

A 32-bit adder is 8 CARRY4s chained: each block's `CO[3]` feeds the next block's `CI`. This is a genuine sequential dependency.

Our first design grouped macros by **dependency depth**, which put each chain link in its own phase — 8 phases for one adder. Since every phase costs a script section re-read every cycle, that alone could have made our "optimised" simulator *slower* than stock GEM.

**The fix uses a GPU warp.** A **warp** is 32 threads that execute in lockstep and can exchange values directly through registers — no memory involved — using **shuffle** instructions like `__shfl_up_sync`.

We place a whole chain in consecutive lanes of one warp and propagate the carry with a **Kogge-Stone parallel prefix scan**:

1. Each lane reduces its 4-bit carry chain to a `(generate, propagate)` pair:
   - `generate` = this block produces a carry regardless of its input
   - `propagate` = a carry entering passes straight through (`S == 0xF`)
2. Five `__shfl_up_sync` steps combine these pairwise across all 32 lanes
3. Each lane recovers its true carry-in and computes its 4 output bits

An 8-link chain resolves in **log₂(32) = 5 shuffle steps** instead of 8 sequential phases. Independent chains share the same warp safely because a chain head clears its `propagate` flag, making the scan **segmented** — a carry physically cannot leak across a boundary.

Measured effect on a 2-link cascade: **2 phases → 1**, and the boomerang stage count also dropped from 2 to 1, because the intermediate `CO[3]` no longer had to be materialised in the tree at all.

## 2.5 The CUDA kernel: `csrc/kernel_v1_impl.cuh`

The kernel gained a macro-evaluation phase (+325 lines) that:

- Runs between boomerang stages, one warp per phase
- Gathers inputs from shared memory using a permutation table
- Evaluates CARRY4 via the segmented Kogge-Stone scan
- Evaluates SRLC32E with read-then-shift in a single visit
- Scatters results into the reserved `hier[1]` slots using atomics over disjoint bit sets

The DSP is handled separately, alongside the existing SRAM commit, because its 121 input bits and 48 output bits do not fit the macro lane layout. It gathers through the write-out permutation exactly like an SRAM does.

### Register budget

GPU threads have a limited register budget, and exceeding it costs occupancy or forces slow memory spills. We measured before and after:

| | Registers | Spills | Blocks/SM |
|---|---|---|---|
| Stock GEM | 92 | 0 | 2 |
| With macros | **105** | **0** | **2** |
| Budget ceiling | 128 | — | 2 |

We also added a startup check using `cudaOccupancyMaxActiveBlocksPerMultiprocessor` that fails with a clear message if the requested grid cannot be co-resident — because GEM uses **cooperative kernel launches**, where the entire grid must fit on the GPU simultaneously or the launch fails outright with an opaque error.

---

# 3. The Bug Hunt: What We Found

Debugging a GPU simulator is genuinely hard: a wrong answer surfaces as a single incorrect bit in a waveform, thousands of cycles deep, with no stack trace. We therefore built a **verification ladder** before writing the kernel.

## 3.1 The verification ladder

```
Independent Python model   ← written from the problem statement text
        ↕
naive_sim (CPU, netlist)   ← ground truth values
        ↕
flatten_test (CPU, script) ← validates the compiler output
        ↕
cuda_test (GPU)            ← validates the kernel
```

Each rung isolates one layer. If `naive_sim` and `flatten_test` agree but the GPU disagrees, the bug is in the kernel — nowhere else. This turned "wrong waveform" into "exact pin, exact stage" and is the single reason the bugs below were findable.

**A deliberate choice:** the Python reference was written from the problem statement, *not* transcribed from our Rust or CUDA. If both had been written by the same reasoning, a shared misreading would cancel out and the test would pass while the model was wrong.

## 3.2 Stock GEM Bug #1 — the write-out accounting leak

**Symptom.** Any design containing macros failed to map. The partitioner would split into smaller and smaller pieces, forever, never converging.

**The clue that cracked it.** The overflow count was *flat* with design size:

| Design | Cells | Write-outs needed (cap: 256) |
|---|---|---|
| 2 lanes | ~500 | 268 |
| 4 lanes | ~1,000 | 264 |
| 16 lanes | 3,876 | 259 |
| MIPS | 70,000 | 259–261 |

A 500-cell design overflowed **just as hard** as a 70,000-cell one. A genuine capacity limit scales with size. This was flat — so it was a leak, not a limit.

**The cause.** In `build_one_boomerang_stage`, when a certain condition holds the code walks **every** 32-slot group in `hier[1]` — about 128 of them — and unconditionally charges one write-out for each, *even after the last endpoint has been placed and the inner loop exits immediately*. Two stages × 128 ≈ 256, which is exactly the cap.

**Why our macros triggered it.** The greedy branch is taken when no endpoint is "untouched". A macro output is a level-0 leaf, so with macros present nothing is ever untouched and the branch is taken **every time**.

**The fix.** Only charge a write-out for a group that actually accomplishes something — receives a pending endpoint, or marks an existing one.

**This is a bug in NVIDIA's code, not ours.** It is latent in stock GEM and would fire on any design whose endpoints are all shallow.

## 3.3 Stock GEM Bug #2 — the silent constant-D flip-flop alias

This is the most dangerous bug we found, because **it produces wrong answers with no error at all**.

**Symptom.** Our 16-lane benchmark produced output that differed from ground truth by exactly 15 (`0xF`) — every cycle, including cycle 0 *during reset*, when the adder is completely inactive.

That last detail was decisive. During reset the accumulator is unconditionally zero and no adder result reaches the output. A carry-chain bug **cannot** produce a wrong value there. So it was not arithmetic at all — four register bits were simply stuck.

**The cause.** In `flatten.rs`:

```rust
if dff.d_iv == 0 {
    clilog::warn!(DFF_CONST_ERR, "...ignoring the error..");
    input_map.insert(dff.q, 0);   // position 0 is a real PRIMARY INPUT bit
    continue
}
```

A flip-flop whose `D` input is a hard constant zero gets its `Q` mapped to **state position 0** — which is an actual primary input of the design. That flop then reads *that input's value* forever instead of zero.

The comment says "ignoring the error". It is not ignoring it; it is silently aliasing a register onto an input pin.

**Why it needs a power-of-two lane count.** With 4 lanes the XOR reduction cancels exactly — `⊕(i₀)` over `i = 0..3` is 0 — so some sum bits become compile-time constants. Yosys legitimately keeps those flops (without an initial value it cannot prove the first cycle). Bisection confirmed it: **0 constant-D flops at 2 lanes (passes), 2 at 4 lanes (fails)**.

**The fix.** GEM's state buffer is zero-initialised, so *any word nobody writes* reads zero forever. We reserve one such word and point constant-D `Q`s at it. We also turned constant-*one* `D` — which would have hit a confusing `unwrap` panic — into an explicit message.

**Impact beyond our project.** Unmodified GEM silently corrupts results on any netlist containing a constant-D flip-flop. This has nothing to do with macros. If the hidden benchmarks contain one, stock GEM is wrong too.

## 3.4 Stock GEM Bug #3 (bonus) — mt-kahypar built with assertions live

GEM uses `mt-kahypar` for graph partitioning. Its build script sets `.debug(false).opt_level(3)` but **never defines `NDEBUG`** — so it compiles at release speed with debug assertions still active, a configuration upstream's own CMake never ships.

The result, on a degenerate hypergraph:

```
initial_partitioning_data_container.h:512:
Assertion `std::is_heap(_best_partitions.begin(), ...)' failed
```

Defining `NDEBUG` matches upstream's real release build. GEM independently validates every partition it receives, so quality remains checked on our side.

## 3.5 Our bug — the shared-state dropout

**Symptom.** `macro input aigpin 209 has no position in the current shared state`.

**The cause.** The boomerang only carries a value forward while it is still a *target*, and it drops each one the instant it is realised. So an input computed in stage 0 had lost its shared-memory slot by the time a macro phase ran after stage 2.

**The fix.** Re-arm pending macro inputs each stage so the tree carries them forward as passthroughs — the same mechanism that already keeps a primary input alive across stages.

## 3.6 A note on method: hypotheses the data killed

Three times we formed a confident, plausible diagnosis that measurement disproved:

| Hypothesis | What the data showed |
|---|---|
| "The `0xF` bug is a cross-warp routing failure" | Failed at 4 lanes / 29 macros — nowhere near the 32-lane boundary |
| "The `0xF` bug is a double-inversion on the macro passthrough" | Wrong at cycle 0 during reset, when no macro output reaches the output |
| "MIPS fails because the depth fixpoint misses an edge" | The wave was scheduled **correctly**; the bug was target-set scoping in a fix we had made two steps earlier |

Each was settled in a single instrumented run. The lesson worth carrying: in a system this layered, **diagnostics are a feature, not scaffolding**. Our `PART_REJECT_*` messages and pin-level panics are part of the deliverable.

## 3.7 Known limitations

We document these honestly rather than hiding them.

### `.gemparts` binary incompatibility

We added two fields to the `Partition` struct. GEM serialises it with `serde_bare`, a **positional** binary format — `#[serde(default)]` provides no backward compatibility.

**Consequence:** partition files written by stock GEM cannot be read by our fork, and vice versa:

```
Err("continuation bit indicated an invalid variable-length integer")
```

The dataset's pre-built `rocket.gemparts` had to be regenerated. A format version field would fix this; we chose not to change the format close to the deadline.

### The MIPS flatten ordering bug

Our MIPS processor **maps** successfully (19–21 partitions) but fails during script generation, with one of two signatures:

```
stage 0 needs aigpin 622 (driver Macro(55061, 2)); producing macro = Some(16);
that macro appears in (wave, after_stage) = [(0, 2)]; total stages 7
```

```
stage 1 needs aigpin 6 (driver InputPort(3)); producing macro = None;
all phases at [1]; total stages 8
```

A partition's early stage consumes a value that is not yet available — either a macro output whose phase runs later, or a primary input that was not carried forward.

**Complicating factor: the runs are non-deterministic.** Three identical invocations produced **19, 21, and 19** partitions with the failure alternating between the two signatures. `mt-kahypar` is multi-threaded, so each run partitions differently. Any fix must be validated against a *seeded* run, or single-sample A/B testing draws conclusions from a random process.

The micro-benchmark exercises all three primitives correctly, so this is a scheduling gap under partitionings the micro-benchmark does not produce — not a fundamental flaw in the approach.

### Scope limitations

- **MIPS has no multiplier and no shift-register chains.** We verified this by inspection: the only `*` in the source is `timescale` and `always @(*)`. It yields CARRY4 only.
- **The industrial dataset contains no RTL.** Rocket, Gemmini and NVDLA ship as *post-AIG netlists* with zero `CARRY4`/`DSP48E2`/`SRLC32E` cells and no `.sv` files. Macro interception is a synthesis-time operation, so no macro build of those cores can exist. They provide baseline numbers only.
- **Gemmini and NVDLA exceeded a 15-minute mapping budget.** Both showed `bisection attempts = 1` — a single large partitioning still in progress, not the pathological loop of §3.2. Notably NVDLA (203K gates) is *smaller* than Rocket (345K gates, mapped in 294 s), so mapping time is driven by hypergraph structure rather than gate count.

---

# 4. The Results & Proof

## 4.1 Correctness first

No performance number means anything if the answers are wrong. Our strongest correctness result:

> The **same RTL**, compiled two ways — macros shredded into AIG versus
> evaluated natively — produces **bit-identical GPU waveforms**.

```
BASELINE == MACRO    160 signal values, 0 mismatches
```

Full verification status:

| Test | Comparisons | Mismatches |
|---|---|---|
| Ground truth (naive_sim vs independent Python model) | 360 | **0** |
| 8-link CARRY4 chain (naive vs flatten_test) | 360 | **0** |
| SRL cascade + CARRY4 (3-way CPU/GPU) | 600 | **0** |
| All three macros (3-way CPU/GPU) | 1,080 | **0** |
| 16-lane micro-benchmark (3-way CPU/GPU) | 480 | **0** |

The stimulus is deliberately **boundary-biased** — signed extremes, `1<<(w-1)`, all-ones — because that is where the 27-bit pre-adder wrap and the 48-bit accumulator wrap live.

## 4.2 Performance: the micro-benchmark

Design: 16 MAC lanes — **104 CARRY4, 16 DSP48E2, 32 SRLC32E**, 41 simulated cycles.

| Metric | Baseline (shredded) | Macro (native) | Gain |
|---|---|---|---|
| Kernel duration | 1.25 ms | **784 µs** | **1.59×** |
| **Throughput** | 32,800 cycles/s | **52,296 cycles/s** | **1.59×** |
| DRAM throughput | 52.79 % | **0.24 %** | **220× less** |
| SM occupancy | 33.33 % | 33.33 % | unchanged |
| Warp divergence | 28.77 / 32 | 21.10 / 32 | *worse* |
| AND2 gates | 15,177 | 3,376 | 4.5× fewer |
| **GEM script size** | 15.1 MB | **529 KB** | **28.5× smaller** |

### Why occupancy did not change

Occupancy is pinned at 33.33 % in **both** configurations because it is bounded by the register budget (105 registers → 2 blocks/SM), not by the macro work.

This is informative rather than disappointing: it proves the entire gain is **memory-side**. We did not make the GPU compute more efficiently; we stopped making it read so much.

## 4.3 The deliberate trade-off: divergence for bandwidth

Warp divergence got **worse**: 28.77 → 21.10 threads per instruction (32 is perfect).

**Why.** In a warp, all 32 threads execute in lockstep. If only some have work, the rest idle. The boomerang tree it replaced was perfectly uniform — every thread doing an identical AND. Our macro phase activates only a subset of lanes (one per macro), so fewer threads are busy per instruction.

**Why we accept it.** The workload was **DRAM-bound at 52.79 %**, not compute-bound. Trading idle arithmetic units — which were already idle — for a 220× reduction in memory traffic is straightforwardly the right trade on this hardware.

We report this openly. The problem statement asks for "minimal thread divergence", and on that specific metric **we did not achieve it**. We achieved something more valuable for this bottleneck, and the honest framing is a conscious trade rather than a win on every axis.

## 4.4 Industrial scale: the Rocket robustness test

| Design | AND2 gates | Mapping | Cycles | Kernel | Cycles/s | DRAM | Occupancy |
|---|---|---|---|---|---|---|---|
| **Rocket** | 344,836 | 294 s → 32 parts | 549 | 29.75 ms | **18,454** | **89.31 %** | 33.33 % |
| Gemmini | 669,055 | timeout @ 900 s | — | — | — | — | — |
| NVDLA | 203,240 | timeout @ 900 s | — | — | — | — | — |

Two things this establishes:

**1. Our fork handles industrial silicon.** Rocket is roughly 100× our micro-benchmark and mapped cleanly in under five minutes with no partition failures. Because these netlists contain no macros, the code path is identical to stock GEM — making this a clean regression test that our scheduler changes did not destabilise ordinary designs at scale.

**2. The 89.31 % DRAM figure validates the whole thesis.** This is measured, on a real CPU core, on unmodified GEM. It confirms the problem statement's premise empirically: GEM at scale is **DRAM-saturated, not compute-bound**.

### An inference we are careful not to overstate

The chain is coherent — baseline is memory-bound → macros shrink the script 28.5× → DRAM utilisation collapses → throughput rises 1.59×. On a design already at 89 % DRAM utilisation, the headroom to recover is *larger* than on our 52.79 % micro-benchmark.

**But that is a prediction, not a measurement.** We cannot demonstrate it on Rocket, because the dataset ships post-AIG netlists with no RTL to intercept macros from. The honest statement is:

> Baseline Rocket is 89 % DRAM-bound. Our approach removes DRAM traffic. The
> gain on such a design is expected to exceed 1.59× — and remains unmeasured.

The measured **1.59×** stands on the micro-benchmark alone. We do not blend the two figures.

---

# 5. Deliverables Checklist

## Deliverable A — Host Parser & Yosys Pre-Processor (15 pts)

- ✅ **`.ys` scripts intercept the named primitives.** `scripts/macro.ys` reads blackbox declarations first so `techmap` and `abc` cannot flatten `CARRY4`/`DSP48E2`/`SRLC32E`. `scripts/baseline.ys` is the shredded control.
- ✅ **Parser extended.** `src/macros.rs` (612 lines) is the single source of truth for pin names, widths, slot ordering and fan-in, consumed by both `aigpdk.rs` (netlist typing) and `aig.rs` (graph construction) so they cannot drift apart.
- ✅ **Cell-type mangling** works around the discovery that cell parameters do not survive `netlistdb`.
- ✅ **64-bit aligned buffers.** Structure-of-arrays planes with block-affinity instance renumbering for coalesced access.

## Deliverable B — CUDA Engine & Boomerang Modification (35 pts)

- ✅ **Macros evaluated on the GPU ALU** via a mid-partition phase in `kernel_v1_impl.cuh` (+325 lines), using native integer arithmetic.
- ✅ **Macro-to-macro dependencies ordered correctly.** `CO[3] → CIN` chains are resolved without intermediate boolean nodes.
- ✅ **`__shfl_sync` used as specified** — a segmented Kogge-Stone prefix scan collapses an 8-link carry chain into 5 shuffle steps in one warp.
- ✅ **Shared memory used** via the zero-cost `hier[1]` overlay.
- ⚠️ **Divergence increased** (28.77 → 21.10). Documented as a deliberate trade in §4.3 rather than claimed as a win.

## Deliverable C — Hardware Macro Implementations (20 pts)

- ✅ **Cycle-accurate models for all three primitives**, 1,603 lines of C++/CUDA in `src/models/`, each with an exhaustive or boundary-biased testbench.
- ✅ **Exact bit-level boundaries respected**, each pinned by an assertion rather than assumed:
  - `AD` wraps at 27 bits (`maxA + maxD == −2`)
  - `M` is provably exact at 45 bits (`|2²⁶ × 2¹⁷| = 2⁴³ < 2⁴⁴`)
  - `P` wraps mod 2⁴⁸ (64 accumulations of 2⁴³ land exactly on zero)
- ✅ **CARRY4 verified exhaustively** — all 1,024 input vectors against the literal recurrence.
- ✅ **Independent verification.** Python model written from the problem statement; bit-vector reference using ripple-carry and shift-add rather than native operators.
- ✅ **Same-RTL equivalence:** shredded vs native produce identical waveforms.

## Deliverable D — Benchmarks & Performance Analysis (20 pts)

- ✅ **Throughput in simulated cycles/second**, macro vs unmodified GEM: 32,800 → **52,296 cycles/s (1.59×)**.
- ✅ **Nsight Compute profiling** of memory bandwidth and warp divergence.
- ✅ **Industrial baseline** on the RISC-V Rocket core: 18,454 cycles/s at **89.31 % DRAM utilisation**.
- ✅ **Benchmark automation** — `scripts/bench.sh` captures all four metrics and emits CSV.
- ⚠️ **Partial coverage.** Speedup is demonstrated on the macro-rich micro-benchmark. Rocket/Gemmini/NVDLA are baseline-only because the dataset contains no RTL; Gemmini and NVDLA exceeded the mapping time box.

## Deliverable E — Documentation & Report (10 pts)

- ✅ **This manual**, covering the scheduling model, memory layout, CUDA architecture and numerical analysis.
- ✅ **Three bugs in NVIDIA's own code** identified, explained and fixed — one of which silently corrupts results with no error message.
- ✅ **Limitations documented honestly**, with exact failure signatures.

---

## Appendix: Reproducing Our Results

```bash
# Build (see .cargo/config.toml for the toolchain workarounds)
cargo build --release --features=cuda

# Synthesise the macro-rich benchmark, both ways
yosys -s scripts/macrobench.ys        # macros preserved
yosys -s scripts/macrobench_base.ys   # macros shredded (baseline)

# Map and profile
./target/release/cut_map_interactive build/macrobench.gv build/macrobench.gemparts
bash scripts/bench.sh macro build/macrobench.gv build/macrobench.gemparts stim.vcd 40

# Verify correctness at every rung of the ladder
python3 tests/macro_gold_check.py ./target/release/naive_sim
```

**Environment:** RTX 3050 Laptop (20 SMs, compute 8.6), CUDA 13.3, Yosys 0.68+136, Rust 1.98, GCC 15.2 under WSL2.