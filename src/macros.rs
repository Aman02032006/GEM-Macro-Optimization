// Word-level hardware macros natively evaluated by GEM.
//
// Single source of truth for the three Xilinx primitives evaluated natively
// rather than shredded into the AIG: CARRY4, DSP48E2 (simplified subset per
// the problem statement) and SRLC32E. Pin names, bit widths, slot ordering
// and combinational fan-in all live here so that `aigpdk.rs` (netlist pin
// typing) and `aig.rs` (graph construction) cannot drift apart.
//
// # Cell type mangling
//
// Neither `sverilogparse` nor `netlistdb` carries cell parameters: `NetlistDB`
// stores only `celltypes: Vec<CompactString>`. The cell type name is therefore
// the ONLY per-instance attribute that survives to the AIG, so any DSP48E2
// configuration that changes evaluation semantics must be encoded in the name.
// The Yosys pass is expected to emit mangled types:
//
//     DSP48E2_BYPASS        DSP48E2_BYPASS_PREADD
//     DSP48E2_MULT          DSP48E2_MULT_PREADD
//     DSP48E2_MAC           DSP48E2_MAC_PREADD
//
// where the base name selects the 2-bit ALU state and the `_PREADD` suffix
// selects `AD = A + D` over `AD = A`. CARRY4 and SRLC32E have no semantics-
// affecting parameters under the PS constraints (all registers fixed, INIT
// parsing not required), so they keep their plain names.
//
// # Slot ordering
//
// Every macro's inputs and outputs are addressed by a dense slot index. The
// orderings below are the contract between the parser, the memory formatter
// and the CUDA kernel; changing one means changing all three.

use netlistdb::Direction;
use sverilogparse::SVerilogRange;

/// The 2-bit ALU state of a DSP48E2, as extracted by the Yosys pass.
///
/// Mirrors the `DSPState` enum in the CUDA model. State 3 is reserved and is
/// never emitted by the parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DspState {
    /// `P_next = C`
    Bypass = 0,
    /// `P_next = M`
    Mult = 1,
    /// `P_next = P_current + M`
    Mac = 2,
}

/// A natively-evaluated hardware macro kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroKind {
    /// 4-bit carry chain. Purely combinational.
    Carry4,
    /// DSP48E2 simplified subset. Only `PREG` is clocked.
    Dsp48e2 {
        state: DspState,
        use_preadder: bool,
    },
    /// 32-bit shift register LUT with a dynamic combinational read port.
    Srlc32e,
}

// ---------------------------------------------------------------------------
// Slot layout constants
// ---------------------------------------------------------------------------

// CARRY4 inputs: CI, CYINIT, DI[3:0], S[3:0]
pub const C4_IN_CI: usize = 0;
pub const C4_IN_CYINIT: usize = 1;
pub const C4_IN_DI: usize = 2; // .. +4
pub const C4_IN_S: usize = 6; // .. +4
pub const C4_NUM_INPUTS: usize = 10;
// CARRY4 outputs: O[3:0], CO[3:0]
pub const C4_OUT_O: usize = 0; // .. +4
pub const C4_OUT_CO: usize = 4; // .. +4
pub const C4_NUM_OUTPUTS: usize = 8;

// DSP48E2 inputs: A[26:0], B[17:0], C[47:0], D[26:0], CEP
pub const DSP_IN_A: usize = 0; // .. +27
pub const DSP_IN_B: usize = 27; // .. +18
pub const DSP_IN_C: usize = 45; // .. +48
pub const DSP_IN_D: usize = 93; // .. +27
pub const DSP_IN_CEP: usize = 120;
pub const DSP_NUM_INPUTS: usize = 121;
// DSP48E2 outputs: P[47:0]
pub const DSP_OUT_P: usize = 0; // .. +48
pub const DSP_NUM_OUTPUTS: usize = 48;

// SRLC32E inputs: D, CE, A[4:0]
pub const SRL_IN_D: usize = 0;
pub const SRL_IN_CE: usize = 1;
pub const SRL_IN_A: usize = 2; // .. +5
pub const SRL_NUM_INPUTS: usize = 7;
// SRLC32E outputs: Q, Q31
pub const SRL_OUT_Q: usize = 0;
pub const SRL_OUT_Q31: usize = 1;
pub const SRL_NUM_OUTPUTS: usize = 2;

impl MacroKind {
    /// Recognise a (possibly mangled) cell type name.
    ///
    /// Returns `None` for any cell type that is not an intercepted macro,
    /// allowing callers to fall through to ordinary AIGPDK cell handling.
    pub fn from_celltype(celltype: &str) -> Option<MacroKind> {
        match celltype {
            "CARRY4" => Some(MacroKind::Carry4),
            "SRLC32E" => Some(MacroKind::Srlc32e),

            "DSP48E2_BYPASS" => Some(MacroKind::Dsp48e2 {
                state: DspState::Bypass,
                use_preadder: false,
            }),
            "DSP48E2_BYPASS_PREADD" => Some(MacroKind::Dsp48e2 {
                state: DspState::Bypass,
                use_preadder: true,
            }),
            "DSP48E2_MULT" => Some(MacroKind::Dsp48e2 {
                state: DspState::Mult,
                use_preadder: false,
            }),
            "DSP48E2_MULT_PREADD" => Some(MacroKind::Dsp48e2 {
                state: DspState::Mult,
                use_preadder: true,
            }),
            "DSP48E2_MAC" => Some(MacroKind::Dsp48e2 {
                state: DspState::Mac,
                use_preadder: false,
            }),
            "DSP48E2_MAC_PREADD" => Some(MacroKind::Dsp48e2 {
                state: DspState::Mac,
                use_preadder: true,
            }),

            // A bare DSP48E2 carries its configuration in parameters, which do
            // not survive the parser. Fail loudly rather than guessing a state.
            "DSP48E2" => panic!(
                "Cell type `DSP48E2` reached the AIG builder unmangled. \
                 Cell parameters are not preserved by netlistdb, so the ALU \
                 state and pre-adder select must be encoded in the cell type \
                 name by the Yosys pass (e.g. DSP48E2_MAC_PREADD). See \
                 src/macros.rs."
            ),

            _ => None,
        }
    }

    pub fn num_inputs(&self) -> usize {
        match self {
            MacroKind::Carry4 => C4_NUM_INPUTS,
            MacroKind::Dsp48e2 { .. } => DSP_NUM_INPUTS,
            MacroKind::Srlc32e => SRL_NUM_INPUTS,
        }
    }

    pub fn num_outputs(&self) -> usize {
        match self {
            MacroKind::Carry4 => C4_NUM_OUTPUTS,
            MacroKind::Dsp48e2 { .. } => DSP_NUM_OUTPUTS,
            MacroKind::Srlc32e => SRL_NUM_OUTPUTS,
        }
    }

    /// Does this macro have a `CLK` pin that must be traced?
    ///
    /// CARRY4 is purely combinational and has none.
    pub fn has_clock(&self) -> bool {
        !matches!(self, MacroKind::Carry4)
    }

    /// The AIG pin value an input slot takes when the pin is absent from the
    /// instantiation.
    ///
    /// Enable-like pins default to constant 1 (aigpin index 0 with the invert
    /// bit set, i.e. `!tie0`); data pins default to constant 0. Getting this
    /// wrong silently disables every register in the design, so it is explicit.
    pub fn default_input(&self, slot: usize) -> usize {
        match (self, slot) {
            (MacroKind::Dsp48e2 { .. }, DSP_IN_CEP) => 1,
            (MacroKind::Srlc32e, SRL_IN_CE) => 1,
            _ => 0,
        }
    }

    /// Map a netlist input pin (name, bit) to its dense slot index.
    ///
    /// `CLK` is deliberately not mapped: it is consumed by clock tracing, not
    /// stored as a data input.
    pub fn input_slot(&self, pin: &str, bit: Option<usize>) -> Option<usize> {
        match self {
            MacroKind::Carry4 => match (pin, bit) {
                // Xilinx names the cascade input `CI`; the Zenith PS calls it
                // `CIN`. Accept both so either spelling links up.
                ("CI" | "CIN", None) => Some(C4_IN_CI),
                ("CYINIT", None) => Some(C4_IN_CYINIT),
                ("DI", Some(i)) if i < 4 => Some(C4_IN_DI + i),
                ("S", Some(i)) if i < 4 => Some(C4_IN_S + i),
                _ => None,
            },
            MacroKind::Dsp48e2 { .. } => match (pin, bit) {
                ("A", Some(i)) if i < 27 => Some(DSP_IN_A + i),
                ("B", Some(i)) if i < 18 => Some(DSP_IN_B + i),
                ("C", Some(i)) if i < 48 => Some(DSP_IN_C + i),
                ("D", Some(i)) if i < 27 => Some(DSP_IN_D + i),
                ("CEP", None) => Some(DSP_IN_CEP),
                _ => None,
            },
            MacroKind::Srlc32e => match (pin, bit) {
                ("D", None) => Some(SRL_IN_D),
                ("CE", None) => Some(SRL_IN_CE),
                ("A", Some(i)) if i < 5 => Some(SRL_IN_A + i),
                _ => None,
            },
        }
    }

    /// Map a netlist output pin (name, bit) to its dense slot index.
    pub fn output_slot(&self, pin: &str, bit: Option<usize>) -> Option<usize> {
        match self {
            MacroKind::Carry4 => match (pin, bit) {
                ("O", Some(i)) if i < 4 => Some(C4_OUT_O + i),
                ("CO", Some(i)) if i < 4 => Some(C4_OUT_CO + i),
                _ => None,
            },
            MacroKind::Dsp48e2 { .. } => match (pin, bit) {
                ("P", Some(i)) if i < 48 => Some(DSP_OUT_P + i),
                _ => None,
            },
            MacroKind::Srlc32e => match (pin, bit) {
                ("Q", None) => Some(SRL_OUT_Q),
                ("Q31", None) => Some(SRL_OUT_Q31),
                _ => None,
            },
        }
    }

    /// The input slots an output slot depends on **combinationally**, within
    /// the same simulated cycle.
    ///
    /// This is what makes macros schedulable, and it is deliberately precise
    /// rather than "every output depends on every input":
    ///
    /// - **CARRY4** is fully combinational, but the dependency is triangular.
    ///   `O[i]` cannot see `S[3]`. Declaring the conservative all-to-all
    ///   relation would manufacture false combinational cycles in legal
    ///   netlists that route a low carry output back into a high `DI` bit, and
    ///   the loop detector in `dfs_netlistdb_build_aig` would panic on a valid
    ///   design.
    /// - **DSP48E2** exports nothing combinationally: `P` is a read of the
    ///   clocked `PREG`, so it is a graph leaf exactly like a DFF `Q`.
    /// - **SRLC32E** is the subtle one. `Q31` is a plain register read with no
    ///   fan-in, but the read port `Q = state[A]` IS combinational in `A[4:0]`
    ///   -- structurally an asynchronous-read memory. Missing this would
    ///   schedule `Q` before its address is resolved.
    pub fn comb_fanin_of_output(&self, out_slot: usize) -> Vec<usize> {
        match self {
            MacroKind::Carry4 => {
                let mut deps = vec![C4_IN_CI, C4_IN_CYINIT];
                if out_slot < C4_OUT_CO {
                    // O[i] = S[i] ^ C[i]; C[i] needs S[0..i-1], DI[0..i-1]
                    let i = out_slot - C4_OUT_O;
                    for j in 0..i {
                        deps.push(C4_IN_S + j);
                        deps.push(C4_IN_DI + j);
                    }
                    deps.push(C4_IN_S + i);
                } else {
                    // CO[i] = C[i+1] needs S[0..i], DI[0..i]
                    let i = out_slot - C4_OUT_CO;
                    for j in 0..=i {
                        deps.push(C4_IN_S + j);
                        deps.push(C4_IN_DI + j);
                    }
                }
                deps
            }
            // P is registered: no same-cycle fan-in.
            MacroKind::Dsp48e2 { .. } => Vec::new(),
            MacroKind::Srlc32e => {
                if out_slot == SRL_OUT_Q {
                    (0..5).map(|i| SRL_IN_A + i).collect()
                } else {
                    // Q31 is a fixed tap on the register: no fan-in.
                    Vec::new()
                }
            }
        }
    }

    /// Is this output a read of clocked state rather than a combinational
    /// function of the current inputs?
    ///
    /// Used by the scheduler to decide whether the macro must also be realised
    /// as an endpoint (something has to compute and commit the next state).
    pub fn is_stateful(&self) -> bool { self.has_state() }

    /// Does this macro carry persistent state in the global macro-state array?
    pub fn has_state(&self) -> bool {
        matches!(self, MacroKind::Dsp48e2 { .. } | MacroKind::Srlc32e)
    }

    /// Does the mid-partition phase itself commit the next state?
    ///
    /// True for SRLC32E: the phase already reads the shift register to serve
    /// `Q` and `Q31`, so it can read-then-shift in one visit. That is
    /// cycle-correct because a macro appears in exactly one phase, and the
    /// read strictly precedes the write, so the new value is only observable
    /// on the next cycle -- the same relationship a DFF has between D and Q.
    ///
    /// False for DSP48E2: 121 input bits and 48 output bits do not fit the
    /// 10-in / 8-out lane layout, so its commit goes through the write-out
    /// path instead, exactly like an SRAM.
    pub fn commits_state_in_phase(&self) -> bool {
        matches!(self, MacroKind::Srlc32e)
    }

    /// Does this macro need an endpoint group to commit its state?
    pub fn is_endpoint(&self) -> bool {
        self.has_state() && !self.commits_state_in_phase()
    }

    /// The input slots that feed the **clocked next-state** computation.
    ///
    /// This is deliberately distinct from [Self::comb_fanin_of_output]. For an
    /// SRLC32E the address `A` drives the combinational read port but has no
    /// influence whatsoever on the next state, so including it here would
    /// impose a false ordering constraint on the state commit.
    pub fn state_fanin(&self) -> Vec<usize> {
        match self {
            MacroKind::Carry4 => Vec::new(),
            // Everything upstream of PREG: pre-adder, multiplier, ALU mux.
            MacroKind::Dsp48e2 { .. } => (0..DSP_NUM_INPUTS).collect(),
            // Only the serial input and its enable; NOT the read address.
            MacroKind::Srlc32e => vec![SRL_IN_D, SRL_IN_CE],
        }
    }

    /// Does this output have to be produced part-way through a partition,
    /// rather than being available from state memory at cycle start?
    ///
    /// This is the property that decides how the macro is scheduled:
    ///
    /// - `false` -- the output is a pure state read (DSP `P`, SRL `Q31`). It
    ///   behaves exactly like a DFF `Q`: loaded during the partition's global
    ///   read phase and treated as a graph leaf. No mid-partition work.
    /// - `true` -- the output is combinational in this cycle's values (CARRY4
    ///   `O`/`CO`, SRL `Q`). It must be evaluated between boomerang stages,
    ///   after its inputs are realised and before its consumers.
    pub fn output_needs_mid_partition_eval(&self, out_slot: usize) -> bool {
        // SRLC32E `Q31` has no combinational fan-in, but it is a tap on state
        // that only exists inside the macro-state array -- there are no AIG
        // pins for the shift register. So it cannot be loaded by the global
        // read like a DFF Q; the phase has to produce it.
        if matches!(self, MacroKind::Srlc32e) { return true }
        !self.comb_fanin_of_output(out_slot).is_empty()
    }

    /// Does any output of this macro need mid-partition evaluation?
    pub fn needs_mid_partition_eval(&self) -> bool {
        (0..self.num_outputs()).any(|s| self.output_needs_mid_partition_eval(s))
    }

    /// Width in bits of the clocked state this macro carries.
    pub fn state_bits(&self) -> usize {
        match self {
            MacroKind::Carry4 => 0,
            // PREG
            MacroKind::Dsp48e2 { .. } => 48,
            // the 32-bit shift register
            MacroKind::Srlc32e => 32,
        }
    }

    /// State size in `u32` words, as laid out in the global macro-state array.
    ///
    /// This mirrors how `sram_data` is sized and offset: one flat global
    /// array, each instance owning a contiguous aligned run.
    pub fn state_words(&self) -> usize {
        (self.state_bits() + 31) / 32
    }

    /// Number of `u32` permute words needed to gather this macro's inputs out
    /// of the boomerang write-out space, analogous to the 4 words an SRAM
    /// takes for its address/enable/data ports.
    pub fn input_words(&self) -> usize {
        (self.num_inputs() + 31) / 32
    }

    // -----------------------------------------------------------------------
    // Netlist pin typing, consumed by `AIGPDKLeafPins`
    // -----------------------------------------------------------------------

    /// Direction of a pin, or `None` if the macro has no such pin.
    pub fn direction_of(&self, pin: &str, bit: Option<isize>) -> Option<Direction> {
        let ubit = match bit {
            Some(b) if b >= 0 => Some(b as usize),
            Some(_) => return None,
            None => None,
        };
        if self.has_clock() && pin == "CLK" && ubit.is_none() {
            return Some(Direction::I);
        }
        if self.input_slot(pin, ubit).is_some() {
            return Some(Direction::I);
        }
        if self.output_slot(pin, ubit).is_some() {
            return Some(Direction::O);
        }
        None
    }

    /// Bus range of a pin, as `SVerilogRange(msb, lsb)`.
    ///
    /// `None` means the pin is scalar (or unknown -- unknown pins are rejected
    /// by `direction_of`, matching how the stock AIGPDK cells behave).
    pub fn width_of(&self, pin: &str) -> Option<SVerilogRange> {
        match (self, pin) {
            (MacroKind::Carry4, "DI" | "S" | "O" | "CO") => Some(SVerilogRange(3, 0)),

            (MacroKind::Dsp48e2 { .. }, "A" | "D") => Some(SVerilogRange(26, 0)),
            (MacroKind::Dsp48e2 { .. }, "B") => Some(SVerilogRange(17, 0)),
            (MacroKind::Dsp48e2 { .. }, "C" | "P") => Some(SVerilogRange(47, 0)),

            (MacroKind::Srlc32e, "A") => Some(SVerilogRange(4, 0)),

            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Script format for mid-partition macro phases
//
// A phase occupies `ceil(lanes/32) * 32 * MACRO_LANE_WORDS` u32. One lane is
// one macro instance, and a phase never spans more than one warp so the carry
// scan stays inside `__shfl_*_sync`.
//
// Per-lane layout (12 words):
//   w0        descriptor  (see MACRO_DESC_* below)
//   w1 .. w5  10 input codes, 2 per word (low half first)
//   w6 .. w9   8 output positions, 2 per word (low half first)
//   w10, w11  reserved, must be zero
//
// An input code is NOT a bare position: it carries the AIG inversion bit and a
// constant flag, mirroring `FlatteningPart::query_permute_with_pin_iv`.
//   bits 0..12  after-array position (0..8191)
//   bit  13     invert
//   bit  14     constant: the value is the invert bit, position ignored
// ---------------------------------------------------------------------------

/// u32 words per macro lane.
pub const MACRO_LANE_WORDS: usize = 12;
/// Input codes stored per lane. CARRY4 uses all 10; SRLC32E uses 7.
pub const MACRO_LANE_IN_SLOTS: usize = 10;
/// Output positions stored per lane. CARRY4 uses all 8; SRLC32E uses 2.
pub const MACRO_LANE_OUT_SLOTS: usize = 8;
/// Lanes per phase. One warp, so the carry scan never crosses a shuffle
/// boundary.
pub const MACRO_MAX_LANES: usize = 32;

pub const MACRO_PERM_POS_MASK: u16 = 0x1fff;
pub const MACRO_PERM_INV_BIT: u16 = 13;
pub const MACRO_PERM_CONST_BIT: u16 = 14;
/// Marks an output slot this macro does not drive.
pub const MACRO_POS_NONE: u16 = 0xffff;

// descriptor word fields
pub const MACRO_DESC_KIND_MASK: u32 = 0x3;
pub const MACRO_DESC_CHAIN_START_BIT: u32 = 2;
pub const MACRO_DESC_VALID_BIT: u32 = 3;
pub const MACRO_DESC_STATE_SHIFT: u32 = 16;

/// Script kind codes. DSP48E2 never appears in a macro phase -- its `P` port
/// is a pure state read, so it is scheduled as an endpoint instead.
pub const MACRO_KIND_CARRY4: u32 = 0;
pub const MACRO_KIND_SRLC32E: u32 = 1;

impl MacroKind {
    /// The code written into a macro lane descriptor.
    pub fn script_kind_code(&self) -> u32 {
        match self {
            MacroKind::Carry4 => MACRO_KIND_CARRY4,
            MacroKind::Srlc32e => MACRO_KIND_SRLC32E,
            MacroKind::Dsp48e2 { .. } => panic!(
                "DSP48E2 has no mid-partition phase: P is a read of PREG and \
                 is scheduled as an endpoint. This is a scheduler bug."
            ),
        }
    }
}

/// Encode one macro input as a script permutation code.
///
/// `pin_iv` is an AIG pin index with its inversion bit, as stored in
/// [MacroInst::inputs]. Values 0 and 1 are the tie-low / tie-high constants.
pub fn macro_encode_input(pin_iv: usize, pos_of: impl Fn(usize) -> Option<u16>) -> u16 {
    if pin_iv <= 1 {
        return (1 << MACRO_PERM_CONST_BIT) | ((pin_iv as u16) << MACRO_PERM_INV_BIT)
    }
    let pos = pos_of(pin_iv >> 1).unwrap_or_else(|| panic!(
        "macro input aigpin {} has no position in the current shared state; \
         the scheduler should have realised it before this phase",
        pin_iv >> 1
    ));
    (pos & MACRO_PERM_POS_MASK) | (((pin_iv & 1) as u16) << MACRO_PERM_INV_BIT)
}

/// Decode a macro input code against a bit-addressable state array.
///
/// Shared by the flattener's self-check, the CPU oracle in flatten_test and
/// (transliterated) the CUDA kernel, so all three agree by construction.
pub fn macro_decode_input(code: u16, read_bit: impl Fn(usize) -> u8) -> u8 {
    let inv = ((code >> MACRO_PERM_INV_BIT) & 1) as u8;
    if (code >> MACRO_PERM_CONST_BIT) & 1 != 0 {
        return inv
    }
    read_bit((code & MACRO_PERM_POS_MASK) as usize) ^ inv
}

/// One instantiated macro, resolved against the AIG.
#[derive(Debug, Clone)]
pub struct MacroInst {
    pub kind: MacroKind,
    /// Input AIG pins **with** invert bit, indexed by input slot.
    pub inputs: Vec<usize>,
    /// Output AIG pins **without** invert bit, indexed by output slot.
    ///
    /// A zero entry means that output is unused in this design.
    pub outputs: Vec<usize>,
    /// Clock enable traced from `CLK`, with invert bit.
    ///
    /// Constant 1 for combinational macros. This is the same representation
    /// `DFF::en_iv` uses, so a gated clock feeding a DSP or SRL maps onto the
    /// existing enable machinery unchanged.
    pub clk_en_iv: usize,
}

impl MacroInst {
    pub fn new(kind: MacroKind) -> MacroInst {
        MacroInst {
            inputs: (0..kind.num_inputs()).map(|s| kind.default_input(s)).collect(),
            outputs: vec![0; kind.num_outputs()],
            clk_en_iv: 1,
            kind,
        }
    }

    /// Iterate the combinational fan-in AIG pins (without invert) of a given
    /// output slot, skipping constants.
    pub fn for_each_comb_fanin(&self, out_slot: usize, mut f: impl FnMut(usize)) {
        for slot in self.kind.comb_fanin_of_output(out_slot) {
            let iv = self.inputs[slot];
            if (iv >> 1) != 0 {
                f(iv >> 1);
            }
        }
    }

    /// Iterate every AIG pin (without invert) that must be realised before
    /// this macro's mid-partition evaluation can run.
    ///
    /// This is the union over outputs that need mid-partition evaluation, so
    /// a macro with no such outputs (a DSP) yields nothing.
    pub fn for_each_eval_fanin(&self, mut f: impl FnMut(usize)) {
        let mut seen = std::collections::BTreeSet::new();
        for out_slot in 0..self.kind.num_outputs() {
            if !self.kind.output_needs_mid_partition_eval(out_slot) {
                continue
            }
            for slot in self.kind.comb_fanin_of_output(out_slot) {
                let iv = self.inputs[slot];
                if (iv >> 1) != 0 && seen.insert(iv >> 1) {
                    f(iv >> 1);
                }
            }
        }
    }

    /// Iterate the AIG pins (without invert) feeding the clocked next-state
    /// computation, including the clock enable.
    pub fn for_each_state_fanin(&self, mut f: impl FnMut(usize)) {
        if (self.clk_en_iv >> 1) != 0 {
            f(self.clk_en_iv >> 1);
        }
        for slot in self.kind.state_fanin() {
            let iv = self.inputs[slot];
            if (iv >> 1) != 0 {
                f(iv >> 1);
            }
        }
    }

    /// The output AIG pins that are produced by mid-partition evaluation.
    pub fn mid_partition_outputs(&self) -> Vec<usize> {
        (0..self.kind.num_outputs())
            .filter(|&s| self.kind.output_needs_mid_partition_eval(s))
            .map(|s| self.outputs[s])
            .filter(|&p| p != 0)
            .collect()
    }

    /// The output AIG pins that are pure state reads, available at cycle start
    /// like a DFF `Q`.
    pub fn cycle_start_outputs(&self) -> Vec<usize> {
        (0..self.kind.num_outputs())
            .filter(|&s| !self.kind.output_needs_mid_partition_eval(s))
            .map(|s| self.outputs[s])
            .filter(|&p| p != 0)
            .collect()
    }
}
