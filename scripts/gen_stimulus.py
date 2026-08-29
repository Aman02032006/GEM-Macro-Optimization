#!/usr/bin/env python3
"""Generate an input VCD for the MIPS Computer netlist.

GEM is a non-interactive simulator: it replays a fixed waveform on the design's
PRIMARY INPUT ports and evolves everything else inside the netlist. So the
stimulus only has to drive Computer's inputs -- no RTL simulator is required to
produce it, which matters because iverilog/verilator are not installed here.

Use tb.v + iverilog instead if you want a golden reference waveform to diff
against; this script covers the GEM input side.

  python3 scripts/gen_stimulus.py build/mips_in.vcd [cycles]
"""
import sys

# Computer's input ports, with widths.
INPUTS = [
    ("reset", 1), ("ins_addr", 8), ("ins", 32),
    ("done_storing", 1), ("done_copying_io_regs", 1),
    ("input_value", 32), ("input_value_valid", 1),
]

# A short MIPS program: immediates, the full R-type ALU set, a store, a load,
# then a self-branch. Chosen to exercise the adders, which is where the CARRY4
# macros live.
PROGRAM = [
    0x2001000a,  # addi $1, $0, 10
    0x20020014,  # addi $2, $0, 20
    0x00221820,  # add  $3, $1, $2
    0x00412022,  # sub  $4, $2, $1
    0x00222824,  # and  $5, $1, $2
    0x00223025,  # or   $6, $1, $2
    0x00223826,  # xor  $7, $1, $2
    0x0022402a,  # slt  $8, $1, $2
    0x20080064,  # addi $8, $0, 100
    0x01094820,  # add  $9, $8, $9
    0xac030000,  # sw   $3, 0($0)
    0x8c0a0000,  # lw   $10, 0($0)
    0x214a0001,  # addi $10, $10, 1
    0x1000fffe,  # beq  $0, $0, -2
    0x00000000,
    0x00000000,
]


def bits(v, w):
    return "".join("1" if (v >> i) & 1 else "0" for i in range(w - 1, -1, -1))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "build/mips_in.vcd"
    run_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    ids, nxt = {}, 34            # '!' is reserved for clk
    lines = ["$timescale 1ns $end", "$scope module top $end",
             "$var wire 1 ! clk $end"]
    for nm, w in INPUTS:
        ids[nm] = chr(nxt); nxt += 1
        lines.append(f"$var wire {w} {ids[nm]} {nm} $end")
    lines += ["$upscope $end", "$enddefinitions $end"]

    st = {nm: 0 for nm, _ in INPUTS}
    st["reset"] = 1
    t = 0

    def emit(first=False):
        lines.append(f"#{t}")
        for nm, w in INPUTS:
            lines.append(f"b{bits(st[nm], w)} {ids[nm]}" if w > 1
                         else f"{st[nm]}{ids[nm]}")

    def edge():
        """One clock period: inputs settle, rise, fall."""
        nonlocal t
        emit(); t += 10
        lines.append(f"#{t}"); lines.append("1!"); t += 10
        lines.append(f"#{t}"); lines.append("0!"); t += 10

    lines.append(f"#{t}"); lines.append("0!")
    emit(); t += 10

    # hold reset briefly
    for _ in range(2):
        edge()
    st["reset"] = 0

    # program load: done_storing low, drive ins_addr/ins one word per cycle
    for i, word in enumerate(PROGRAM):
        st["ins_addr"] = i & 0xff
        st["ins"] = word
        edge()

    # run: done_storing high releases the processor
    st["done_storing"] = 1
    st["ins_addr"] = 0
    st["ins"] = 0
    for _ in range(run_cycles):
        edge()

    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}: {len(PROGRAM)} program words + {run_cycles} run cycles")


if __name__ == "__main__":
    main()
