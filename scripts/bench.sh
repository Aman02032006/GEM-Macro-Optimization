#!/bin/bash
# Deliverable D benchmark harness.
#
#   scripts/bench.sh <name> <netlist.gv> <parts.gemparts> <input.vcd> <blocks>
#
# Runs cuda_test twice: once bare to capture GEM's own kernel timing (which
# excludes VCD parsing), and once under Nsight Compute for the hardware
# counters. Emits one CSV row.
#
# Counters chosen to match the problem statement's Deliverable D:
#   gpu__time_duration.sum                  raw kernel duration
#   sm__warps_active...pct_of_peak          achieved SM occupancy
#   gpu__dram_throughput...pct_of_peak      memory bandwidth utilisation
#   smsp__thread_inst_executed_per_inst...  warp divergence (32 = no divergence)
set -u
NAME=$1; GV=$2; PARTS=$3; VCD=$4; BLOCKS=$5
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NCU=/usr/local/cuda/bin/ncu
cd "$ROOT" || exit 1

OUT=$(mktemp -d)
BARE="$OUT/bare.log"
timeout 1800 ./target/release/cuda_test "$GV" "$PARTS" "$VCD" "$OUT/o.vcd" "$BLOCKS" \
    --input-vcd-scope top > "$BARE" 2>&1
if [ $? -ne 0 ]; then echo "$NAME,RUN_FAILED,,,,"; tail -3 "$BARE" >&2; exit 1; fi

# GEM prints its own GPU-only runtime; that is the number to quote, not
# wall-clock, because VCD parsing dominates the process time.
GEMTIME=$(grep -oiE "(simulat|gpu|kernel)[^0-9]*[0-9.]+ *(ms|s|us)" "$BARE" | tail -1)

METRICS="gpu__time_duration.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__thread_inst_executed_per_inst_executed.ratio"

NCULOG="$OUT/ncu.log"
timeout 3600 $NCU --metrics "$METRICS" --target-processes all --kernel-name-base function \
    ./target/release/cuda_test "$GV" "$PARTS" "$VCD" "$OUT/o2.vcd" "$BLOCKS" \
    --input-vcd-scope top > "$NCULOG" 2>&1

# ncu prints "<metric> <unit> <value>"; keep the unit, otherwise a msecond
# and a usecond row look like the same magnitude.
val () { grep -E "^ +$1" "$NCULOG" | tail -1 | awk '{print $(NF-1)":"$(NF)}'; }
DUR=$(val "gpu__time_duration.sum")
OCC=$(val "sm__warps_active.avg.pct_of_peak_sustained_active")
DRAM=$(val "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")
DIV=$(val "smsp__thread_inst_executed_per_inst_executed.ratio")

echo "$NAME,$DUR,$OCC,$DRAM,$DIV,${GEMTIME:-n/a}"
