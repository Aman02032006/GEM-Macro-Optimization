#!/usr/bin/env python3
"""Ground-truth harness for the word-level macro models in naive_sim.

Generates a stimulus VCD for tests/macro_gold.gv, runs naive_sim over it, then
checks every output against an INDEPENDENT Python model of the three
primitives -- written from the Zenith problem statement text, not transcribed
from the Rust or CUDA code, so a shared misreading cannot cancel out.

Usage:  python3 tests/macro_gold_check.py <path-to-naive_sim>
"""
import subprocess
import sys
import os
import random
import tempfile

# --------------------------------------------------------------------------
# Independent reference model (from the PS text)
# --------------------------------------------------------------------------

def carry4(s, di, ci, cyinit):
    """C[0]=CYINIT|CIN ; C[i+1]=(S[i]&C[i])|(~S[i]&DI[i]) ; O[i]=S[i]^C[i]."""
    c = (ci | cyinit) & 1
    o = co = 0
    for i in range(4):
        sb = (s >> i) & 1
        db = (di >> i) & 1
        cn = (sb & c) | ((sb ^ 1) & db)
        o |= (sb ^ c) << i
        co |= cn << i
        c = cn
    return o, co


def sext(v, w):
    v &= (1 << w) - 1
    return v - (1 << w) if v & (1 << (w - 1)) else v


def dsp_mac_preadd(a, b, c_unused, d, p_cur):
    """AD = A+D wrapped to 27 bits; M = AD*B (45b); P_next = P + M mod 2^48."""
    ad = sext(a + d, 27)
    m = sext(ad * b, 45)
    return (sext(p_cur, 48) + m) & ((1 << 48) - 1)


class Srl:
    def __init__(self):
        self.sr = 0

    def tick(self, d, ce):
        if ce:
            self.sr = ((self.sr << 1) | (d & 1)) & 0xFFFFFFFF

    def q(self, a):
        return (self.sr >> (a & 31)) & 1

    def q31(self):
        return (self.sr >> 31) & 1


# --------------------------------------------------------------------------
# Stimulus
# --------------------------------------------------------------------------

INPUTS = [
    ("cyinit", 1), ("s0", 4), ("di0", 4), ("s1", 4), ("di1", 4),
    ("srl_d", 1), ("srl_ce", 1), ("srl_a", 5),
    ("dsp_a", 27), ("dsp_b", 18), ("dsp_c", 48), ("dsp_d", 27),
]
OUTPUTS = [
    ("o0", 4), ("o1", 4), ("cout", 1),
    ("srl_q", 1), ("srl_q31", 1), ("p", 48),
]

NCYCLES = 60


def gen_vectors(seed=0xC0FFEE):
    rng = random.Random(seed)
    vecs = []
    for cyc in range(NCYCLES):
        v = {}
        # bias hard toward the signed boundaries -- that is where the
        # 27-bit pre-adder wrap and the 48-bit accumulator wrap live
        def pick(w):
            r = rng.randrange(8)
            if r == 0: return 0
            if r == 1: return (1 << w) - 1
            if r == 2: return 1 << (w - 1)
            if r == 3: return (1 << (w - 1)) - 1
            return rng.randrange(1 << w)
        for name, w in INPUTS:
            v[name] = pick(w)
        vecs.append(v)
    return vecs


def bits(val, w):
    return "".join("1" if (val >> i) & 1 else "0" for i in range(w - 1, -1, -1))


def write_vcd(path, vecs):
    ids = {}
    nxt = 34   # '!' (33) is taken by clk below; a collision here silently
               # aliases two signals and the sim sees no clock edges
    with open(path, "w") as f:
        f.write("$timescale 1ns $end\n$scope module top $end\n")
        f.write("$var wire 1 ! clk $end\n")
        for name, w in INPUTS:
            ids[name] = chr(nxt)
            nxt += 1
            f.write(f"$var wire {w} {ids[name]} {name} $end\n")
        f.write("$upscope $end\n$enddefinitions $end\n")

        t = 0
        f.write(f"#{t}\n0!\n")
        for name, w in INPUTS:
            f.write((f"b{bits(0, w)} {ids[name]}\n") if w > 1 else f"0{ids[name]}\n")
        for v in vecs:
            t += 10
            f.write(f"#{t}\n")
            for name, w in INPUTS:
                f.write((f"b{bits(v[name], w)} {ids[name]}\n") if w > 1
                        else f"{v[name]}{ids[name]}\n")
            t += 10
            f.write(f"#{t}\n1!\n")   # rising edge
            t += 10
            f.write(f"#{t}\n0!\n")
    return ids


def parse_vcd(path):
    """Return {time: {signal: int}} from a VCD.

    naive_sim emits bit-blasted scalars (`p[47]`, `o0[3]`, ...) rather than
    vectors, so buses are reassembled here from their per-bit vars.
    """
    id2sig = {}            # vcd id -> (base_name, bit_index or None)
    bitvals = {}           # (base_name, bit) -> 0/1
    scal = {}              # base_name -> value
    out = {}
    cur = None

    def snapshot():
        d = dict(scal)
        acc = {}
        for (nm, b), v in bitvals.items():
            acc[nm] = acc.get(nm, 0) | (v << b)
        d.update(acc)
        return d

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("$var"):
                parts = line.split()
                sid, nm = parts[3], parts[4]
                if nm.endswith("]") and "[" in nm:
                    base, idx = nm[:-1].split("[")
                    id2sig[sid] = (base, int(idx))
                else:
                    id2sig[sid] = (nm, None)
            elif line.startswith("#"):
                if cur is not None:
                    out[cur] = snapshot()
                cur = int(line[1:])
            elif line and line[0] == "b":
                sp = line.split()
                if len(sp) > 1 and sp[1] in id2sig:
                    raw = sp[0][1:].replace("x", "0").replace("z", "0")
                    v = int(raw, 2) if raw else 0
                    nm, bit = id2sig[sp[1]]
                    if bit is None:
                        scal[nm] = v
                    else:
                        bitvals[(nm, bit)] = v & 1
            elif line and line[0] in "01xz" and len(line) > 1:
                sid = line[1:]
                if sid in id2sig:
                    v = int(line[0]) if line[0] in "01" else 0
                    nm, bit = id2sig[sid]
                    if bit is None:
                        scal[nm] = v
                    else:
                        bitvals[(nm, bit)] = v
    if cur is not None:
        out[cur] = snapshot()
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: macro_gold_check.py <naive_sim binary>")
        return 2
    sim = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    gv = os.path.join(here, "macro_gold.gv")

    tmp = tempfile.mkdtemp(prefix="macrogold")
    in_vcd = os.path.join(tmp, "in.vcd")
    out_vcd = os.path.join(tmp, "out.vcd")

    vecs = gen_vectors()
    write_vcd(in_vcd, vecs)

    r = subprocess.run(
        [sim, gv, in_vcd, out_vcd, "--input-vcd-scope", "top"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("naive_sim FAILED")
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        return 1

    got = parse_vcd(out_vcd)
    times = sorted(got)

    def reference(state_lag):
        """Run the stimulus through the independent model.

        `state_lag` selects the cycle convention for the clocked elements:
        naive_sim latches BEFORE it propagates, so at edge k the registers
        sample whatever the previous propagate left on their inputs, i.e.
        vecs[k-1]. Combinational reads always use vecs[k]. Purely
        combinational macros (CARRY4) are unaffected either way.
        """
        srl = Srl()
        p_reg = 0
        prev = {k: 0 for k, _ in INPUTS}
        exp = []
        for v in vecs:
            sv = prev if state_lag else v
            srl.tick(sv["srl_d"], sv["srl_ce"])
            p_reg = dsp_mac_preadd(sext(sv["dsp_a"], 27), sext(sv["dsp_b"], 18),
                                   sv["dsp_c"], sext(sv["dsp_d"], 27), p_reg)
            o0, co0 = carry4(v["s0"], v["di0"], 0, v["cyinit"])
            o1, co1 = carry4(v["s1"], v["di1"], (co0 >> 3) & 1, 0)
            exp.append({
                "o0": o0, "o1": o1, "cout": (co1 >> 3) & 1,
                "srl_q": srl.q(v["srl_a"]), "srl_q31": srl.q31(),
                "p": p_reg,
            })
            prev = v
        return exp

    def score(exp):
        per = {}
        n = min(len(exp), len(times))
        for i in range(n):
            g, e = got[times[i]], exp[i]
            for name, _w in OUTPUTS:
                if name in g and g[name] != e[name]:
                    per[name] = per.get(name, 0) + 1
        return per

    cand = {"latch-before-propagate (lagged)": reference(True),
            "propagate-before-latch": reference(False)}
    print("convention probe (mismatches per signal):")
    for label, exp in cand.items():
        print(f"  {label}: {score(exp) or 'ALL MATCH'}")

    expected = min(cand.values(), key=lambda e: sum(score(e).values()))

    # naive_sim emits one timestamp per simulated rising edge.
    fails = 0
    checks = 0
    n = min(len(expected), len(times))
    if n == 0:
        print("no timestamps in naive_sim output -- nothing was simulated")
        return 1
    for i in range(n):
        g = got[times[i]]
        e = expected[i]
        for name, _w in OUTPUTS:
            if name not in g:
                continue
            checks += 1
            if g[name] != e[name]:
                fails += 1
                if fails <= 12:
                    print(f"  MISMATCH cycle {i} t={times[i]} {name}: "
                          f"got {g[name]} expected {e[name]}")
    print(f"compared {n} cycles, {checks} signal checks, {fails} mismatches")
    print("GROUND TRUTH OK" if fails == 0 and checks > 0 else "GROUND TRUTH FAILED")
    return 0 if (fails == 0 and checks > 0) else 1


if __name__ == "__main__":
    sys.exit(main())
