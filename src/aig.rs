// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! And-inverter graph format
//!
//! An AIG is derived from netlistdb synthesized in AIGPDK.

use netlistdb::{NetlistDB, GeneralPinName, Direction};
use indexmap::{IndexMap, IndexSet};
use crate::aigpdk::AIGPDK_SRAM_ADDR_WIDTH;
use crate::macros::{MacroInst, MacroKind};

/// A DFF.
#[derive(Debug, Default, Clone)]
pub struct DFF {
    /// The D input pin with invert (last bit)
    pub d_iv: usize,
    /// If the DFF is enabled, i.e., if the clock, S, or R is active.
    pub en_iv: usize,
    /// The Q pin output with invert.
    pub q: usize,
}

/// A ram block resembling the interface of `$__RAMGEM_SYNC_`.
#[derive(Debug, Default, Clone)]
pub struct RAMBlock {
    pub port_r_addr_iv: [usize; AIGPDK_SRAM_ADDR_WIDTH],

    /// controls whether r_rd_data should update. (from read clock)
    pub port_r_en_iv: usize,
    pub port_r_rd_data: [usize; 32],

    pub port_w_addr_iv: [usize; AIGPDK_SRAM_ADDR_WIDTH],
    /// controls whether memory should be updated.
    ///
    /// this is a combination of write enable and write clock.
    pub port_w_wr_en_iv: [usize; 32],
    pub port_w_wr_data_iv: [usize; 32],
}

/// A type of endpoint group. can be a primary output-related pin,
/// a D flip-flop, or a ram block.
///
/// A group means a task for the partition to complete.
/// For primary output pins, the task is just to store.
/// For DFFs, the task is to store only when the clock is enable.
/// For RAMBlocks, the task is to simulate a sync SRAM.
/// A StagedIOPin indicates a temporary live pin between different
/// major stages but reside in the same simulated cycle.
#[derive(Debug, Copy, Clone)]
pub enum EndpointGroup<'i> {
    PrimaryOutput(usize),
    DFF(&'i DFF),
    RAMBlock(&'i RAMBlock),
    /// A stateful word-level macro (DSP48E2 `PREG`, SRLC32E shift register).
    ///
    /// The task is to compute the next state and commit it under the macro's
    /// clock enable, exactly like a DFF -- the difference is only that the
    /// state is 32 or 48 bits wide and lives in the global macro-state array
    /// rather than in a write-out slot.
    ///
    /// Purely combinational macros (CARRY4) are NOT endpoints; they are
    /// evaluated mid-partition, see [crate::pe::MacroStage].
    Macro(&'i MacroInst),
    StagedIOPin(usize),
}

impl EndpointGroup<'_> {
    /// Enumerate all related aigpin inputs for this endpoint group.
    ///
    /// The enumerated inputs may have duplicates.
    pub fn for_each_input(self, mut f_nz: impl FnMut(usize)) {
        let mut f = |i| {
            if i >= 1 { f_nz(i); }
        };
        match self {
            Self::PrimaryOutput(idx) => f(idx >> 1),
            Self::DFF(dff) => {
                f(dff.en_iv >> 1);
                f(dff.d_iv >> 1);
            },
            Self::RAMBlock(ram) => {
                f(ram.port_r_en_iv >> 1);
                for i in 0..13 {
                    f(ram.port_r_addr_iv[i] >> 1);
                    f(ram.port_w_addr_iv[i] >> 1);
                }
                for i in 0..32 {
                    f(ram.port_w_wr_en_iv[i] >> 1);
                    f(ram.port_w_wr_data_iv[i] >> 1);
                }
            },
            Self::Macro(m) => {
                // Only the next-state cone, plus the clock enable. The
                // SRLC32E read address is deliberately excluded: it drives the
                // combinational read port, not the shift, so requiring it here
                // would impose a false ordering constraint on the commit.
                m.for_each_state_fanin(|i| f_nz(i));
            },
            Self::StagedIOPin(idx) => f(idx),
        }
    }
}

/// The driver type of an AIG pin.
#[derive(Debug, Clone)]
pub enum DriverType {
    /// Driven by an and gate.
    ///
    /// The inversion bit is stored as the last bits in
    /// two input indices.
    ///
    /// Only this type has combinational fan-in.
    AndGate(usize, usize),
    /// Driven by a primary input port (with its netlistdb id).
    InputPort(usize),
    /// Driven by a clock flag (with clock port netlistdb id, and pos/negedge)
    InputClockFlag(usize, u8),
    /// Driven by a DFF (with its index)
    DFF(usize),
    /// Driven by a 13-bit by 32-bit RAM block (with its index)
    SRAM(usize),
    /// Driven by an output of a word-level macro (cell id, output slot).
    ///
    /// Unlike [DriverType::AndGate] the fan-in is not stored inline: look the
    /// instance up in [AIG::macros] and consult
    /// [crate::macros::MacroKind::comb_fanin_of_output] for the slot, because
    /// the combinational dependency is per-output, not per-cell. A CARRY4
    /// `O[0]` does not depend on `S[3]`, and a DSP `P` bit depends on nothing
    /// at all in the current cycle.
    Macro(usize, usize),
    /// Tie0: tied to zero. Only the 0-th aig pin is allowed to have this.
    Tie0
}

/// An AIG associated with a netlistdb.
#[derive(Debug, Default)]
pub struct AIG {
    /// The number of AIG pins.
    ///
    /// This number might be smaller than num_pins in netlistdb,
    /// because inverters and buffers are merged when possible.
    /// It might also be larger because we may add mux circuits.
    ///
    /// AIG pins are numbered from 1 to num_aigpins inclusive.
    /// The AIG pin id zero (0) is tied to 0.
    ///
    /// AIG pins are guaranteed to have topological order.
    pub num_aigpins: usize,
    /// The mapping from a netlistdb pin to an AIG pin.
    ///
    /// The inversion bit is stored as the last bit.
    /// E.g., `pin2aigpin_iv[pin_id] = aigpin_id << 1 | invert`.
    pub pin2aigpin_iv: Vec<usize>,
    /// The clock pins map. Every clock pin has a pair of flag pins
    /// showing if they are posedge/negedge.
    ///
    /// The flag pin can be empty which means the circuit is not
    /// active with that edge.
    pub clock_pin2aigpins: IndexMap<usize, (usize, usize)>,
    /// The driver types of AIG pins.
    pub drivers: Vec<DriverType>,
    /// A cache for identical and gates.
    pub and_gate_cache: IndexMap<(usize, usize), usize>,
    /// Unique primary output aigpin indices
    pub primary_outputs: IndexSet<usize>,
    /// The D flip-flops (DFFs), indexed by cell id
    pub dffs: IndexMap<usize, DFF>,
    /// The SRAMs, indexed by cell id
    pub srams: IndexMap<usize, RAMBlock>,
    /// The natively-evaluated word-level macros, indexed by cell id.
    pub macros: IndexMap<usize, MacroInst>,
    /// Positions within [Self::macros] of the macros that carry clocked state
    /// and therefore appear as endpoint groups. Sorted ascending.
    pub stateful_macros: Vec<usize>,
    /// Reverse map: AIG pin -> position within [Self::macros] of the macro
    /// that drives it. Only populated for macro output pins.
    pub aigpin2macro: IndexMap<usize, usize>,
    /// The fanout CSR start array.
    pub fanouts_start: Vec<usize>,
    /// The fanout CSR array.
    pub fanouts: Vec<usize>,
}

impl AIG {
    fn add_aigpin(&mut self, driver: DriverType) -> usize {
        self.num_aigpins += 1;
        self.drivers.push(driver);
        self.num_aigpins
    }

    fn add_and_gate(&mut self, a: usize, b: usize) -> usize {
        assert_ne!(a | 1, usize::MAX);
        assert_ne!(b | 1, usize::MAX);
        if a == 0 || b == 0 {
            return 0
        }
        if a == 1 {
            return b
        }
        if b == 1 {
            return a
        }
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        if let Some(o) = self.and_gate_cache.get(&(a, b)) {
            return o << 1;
        }
        let aigpin = self.add_aigpin(DriverType::AndGate(a, b));
        self.and_gate_cache.insert((a, b), aigpin);
        aigpin << 1
    }

    /// given a clock pin, trace back to clock root and return its
    /// enable signal (with invert bit).
    ///
    /// if result is 0, that means the pin is dangled.
    /// if an error occurs because of a undecipherable multi-input cell,
    /// we will return in error the last output pin index of that cell.
    fn trace_clock_pin(
        &mut self,
        netlistdb: &NetlistDB,
        pinid: usize, is_negedge: bool,
        // should we ignore cklnqd in this tracing.
        // if set to true, we will treat cklnqd as a simple buffer.
        // otherwise, we assert that cklnqd/en is already built in
        // our aig mapping (pin2aigpin_iv).
        ignore_cklnqd: bool,
    ) -> Result<usize, usize> {
        if netlistdb.pindirect[pinid] == Direction::I {
            let netid = netlistdb.pin2net[pinid];
            if Some(netid) == netlistdb.net_zero || Some(netid) == netlistdb.net_one {
                return Ok(0)
            }
            let root = netlistdb.net2pin.items[
                netlistdb.net2pin.start[netid]
            ];
            return self.trace_clock_pin(
                netlistdb, root, is_negedge,
                ignore_cklnqd
            )
        }
        let cellid = netlistdb.pin2cell[pinid];
        if cellid == 0 {
            let clkentry = self.clock_pin2aigpins.entry(pinid)
                .or_insert((usize::MAX, usize::MAX));
            let clksignal = match is_negedge {
                false => clkentry.0,
                true => clkentry.1
            };
            if clksignal != usize::MAX {
                return Ok(clksignal << 1)
            }
            let aigpin = self.add_aigpin(DriverType::InputClockFlag(pinid, is_negedge as u8));
            let clkentry = self.clock_pin2aigpins.get_mut(&pinid).unwrap();
            let clksignal = match is_negedge {
                false => &mut clkentry.0,
                true => &mut clkentry.1
            };
            *clksignal = aigpin;
            return Ok(aigpin << 1)
        }
        let mut pin_a = usize::MAX;
        let mut pin_cp = usize::MAX;
        let mut pin_en = usize::MAX;
        let celltype = netlistdb.celltypes[cellid].as_str();
        if !matches!(celltype, "INV" | "BUF" | "CKLNQD") {
            clilog::error!("cell type {} supported on clock path. expecting only INV, BUF, or CKLNQD", celltype);
            return Err(pinid)
        }
        for ipin in netlistdb.cell2pin.iter_set(cellid) {
            if netlistdb.pindirect[ipin] == Direction::I {
                match netlistdb.pinnames[ipin].1.as_str() {
                    "A" => pin_a = ipin,
                    "CP" => pin_cp = ipin,
                    "E" => pin_en = ipin,
                    i @ _ => {
                        clilog::error!("input pin {} unexpected for ck element {}", i, celltype);
                        return Err(ipin)
                    }
                }
            }
        }
        match celltype {
            "INV" => {
                assert_ne!(pin_a, usize::MAX);
                self.trace_clock_pin(
                    netlistdb, pin_a, !is_negedge,
                    ignore_cklnqd
                )
            },
            "BUF" => {
                assert_ne!(pin_a, usize::MAX);
                self.trace_clock_pin(
                    netlistdb, pin_a, is_negedge,
                    ignore_cklnqd
                )
            },
            "CKLNQD" => {
                assert_ne!(pin_cp, usize::MAX);
                assert_ne!(pin_en, usize::MAX);
                let ck_iv = self.trace_clock_pin(
                    netlistdb, pin_cp, is_negedge,
                    ignore_cklnqd
                )?;
                if ignore_cklnqd {
                    return Ok(ck_iv)
                }
                let en_iv = self.pin2aigpin_iv[pin_en];
                assert_ne!(en_iv, usize::MAX, "clken not built");
                Ok(self.add_and_gate(ck_iv, en_iv))
            },
            _ => unreachable!()
        }
    }

    /// recursively add aig pins for netlistdb pins
    ///
    /// for sequential logics like DFF and RAM,
    /// 1. their netlist pin inputs are not patched,
    /// 2. their aig pin inputs (in dffs and srams arrays) will be
    ///    patched to include mux -- but not inside this function.
    /// 3. their netlist/aig outputs are directly built here,
    ///    with possible patches for asynchronous DFFSR polyfill.
    fn dfs_netlistdb_build_aig(
        &mut self,
        netlistdb: &NetlistDB,
        topo_vis: &mut Vec<bool>,
        topo_instack: &mut Vec<bool>,
        pinid: usize
    ) {
        if topo_instack[pinid] {
            panic!("circuit has a loop around pin {}",
                   netlistdb.pinnames[pinid].dbg_fmt_pin());
        }
        if topo_vis[pinid] {
            return
        }
        topo_vis[pinid] = true;
        topo_instack[pinid] = true;
        let netid = netlistdb.pin2net[pinid];
        let cellid = netlistdb.pin2cell[pinid];
        let celltype = netlistdb.celltypes[cellid].as_str();
        if netlistdb.pindirect[pinid] == Direction::I {
            if Some(netid) == netlistdb.net_zero {
                self.pin2aigpin_iv[pinid] = 0;
            }
            else if Some(netid) == netlistdb.net_one {
                self.pin2aigpin_iv[pinid] = 1;
            }
            else {
                let root = netlistdb.net2pin.items[
                    netlistdb.net2pin.start[netid]
                ];
                self.dfs_netlistdb_build_aig(
                    netlistdb, topo_vis, topo_instack,
                    root
                );
                self.pin2aigpin_iv[pinid] = self.pin2aigpin_iv[root];
                if cellid == 0 {
                    self.primary_outputs.insert(self.pin2aigpin_iv[pinid]);
                }
            }
        }
        else if cellid == 0 {
            let aigpin = self.add_aigpin(
                DriverType::InputPort(pinid)
            );
            self.pin2aigpin_iv[pinid] = aigpin << 1;
        }
        else if matches!(celltype, "DFF" | "DFFSR") {
            let q = self.add_aigpin(DriverType::DFF(cellid));
            let dff = self.dffs.entry(cellid).or_default();
            dff.q = q;
            let mut ap_s_iv = 1;
            let mut ap_r_iv = 1;
            let mut q_out = q << 1;
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                if !matches!(netlistdb.pinnames[pinid].1.as_str(), "S" | "R") {
                    continue
                }
                self.dfs_netlistdb_build_aig(
                    netlistdb, topo_vis, topo_instack, pinid
                );
                let prev = self.pin2aigpin_iv[pinid];
                match netlistdb.pinnames[pinid].1.as_str() {
                    "S" => ap_s_iv = prev,
                    "R" => ap_r_iv = prev,
                    _ => unreachable!()
                }
            }
            q_out = self.add_and_gate(q_out ^ 1, ap_s_iv) ^ 1;
            q_out = self.add_and_gate(q_out, ap_r_iv);
            self.pin2aigpin_iv[pinid] = q_out;
        }
        else if celltype == "LATCH" {
            panic!("latches are intentionally UNSUPPORTED by GEM, \
                    except in identified gated clocks. \n\
                    you can link a FF&MUX-based LATCH module, \
                    but most likely that is NOT the right solution. \n\
                    check all your assignments inside always@(*) block \
                    to make sure they cover all scenarios.");
        }
        else if celltype == "$__RAMGEM_SYNC_" {
            let o = self.add_aigpin(DriverType::SRAM(cellid));
            self.pin2aigpin_iv[pinid] = o << 1;
            assert_eq!(netlistdb.pinnames[pinid].1.as_str(),
                       "PORT_R_RD_DATA");
            let sram = self.srams.entry(cellid).or_default();
            sram.port_r_rd_data[netlistdb.pinnames[pinid].2.unwrap() as usize] = o;
        }
        else if let Some(kind) = MacroKind::from_celltype(celltype) {
            // Word-level macro output pin. Mirrors the $__RAMGEM_SYNC_ arm:
            // an AIG pin is minted for the output here, while macro *inputs*
            // are resolved in the second pass of from_netlistdb, once every
            // pin in the design has an AIG mapping.
            //
            // Without this arm the pin falls through to the INV/BUF/AND2 arm
            // below and hits its `unreachable!()`.
            let pinname = netlistdb.pinnames[pinid].1.as_str();
            let pinbit = netlistdb.pinnames[pinid].2.map(|i| i as usize);
            let slot = match kind.output_slot(pinname, pinbit) {
                Some(slot) => slot,
                None => panic!(
                    "Macro cell {} ({:?}) drives pin {}, which is not one of \
                     its declared outputs. See src/macros.rs.",
                    netlistdb.celltypes[cellid], kind,
                    netlistdb.pinnames[pinid].dbg_fmt_pin()
                )
            };
            let o = self.add_aigpin(DriverType::Macro(cellid, slot));
            self.pin2aigpin_iv[pinid] = o << 1;
            self.macros.entry(cellid)
                .or_insert_with(|| MacroInst::new(kind))
                .outputs[slot] = o;
        }
        else if celltype == "CKLNQD" {
            let mut prev_cp = usize::MAX;
            let mut prev_en = usize::MAX;
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                match netlistdb.pinnames[pinid].1.as_str() {
                    "CP" => prev_cp = pinid,
                    "E" => prev_en = pinid,
                    _ => {}
                }
            }
            assert_ne!(prev_cp, usize::MAX);
            assert_ne!(prev_en, usize::MAX);
            for prev in [prev_cp, prev_en] {
                self.dfs_netlistdb_build_aig(
                    netlistdb, topo_vis, topo_instack,
                    prev
                );
            }
            // do not define pin2aigpin_iv[pinid] which is CKLNQD/Q and unused in logic.
        }
        else {
            let mut prev_a = usize::MAX;
            let mut prev_b = usize::MAX;
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                match netlistdb.pinnames[pinid].1.as_str() {
                    "A" => prev_a = pinid,
                    "B" => prev_b = pinid,
                    _ => {}
                }
            }
            for prev in [prev_a, prev_b] {
                if prev != usize::MAX {
                    self.dfs_netlistdb_build_aig(
                        netlistdb, topo_vis, topo_instack,
                        prev
                    );
                }
            }
            match celltype {
                "AND2_00_0" | "AND2_01_0" | "AND2_10_0" | "AND2_11_0" | "AND2_11_1" => {
                    assert_ne!(prev_a, usize::MAX);
                    assert_ne!(prev_b, usize::MAX);
                    let name = netlistdb.celltypes[cellid].as_bytes();
                    let iv_a = name[5] - b'0';
                    let iv_b = name[6] - b'0';
                    let iv_y = name[8] - b'0';
                    let apid = self.add_and_gate(
                        self.pin2aigpin_iv[prev_a] ^ (iv_a as usize),
                        self.pin2aigpin_iv[prev_b] ^ (iv_b as usize),
                    ) ^ (iv_y as usize);
                    self.pin2aigpin_iv[pinid] = apid;
                },
                "INV" => {
                    assert_ne!(prev_a, usize::MAX);
                    self.pin2aigpin_iv[pinid] = self.pin2aigpin_iv[prev_a] ^ 1;
                },
                "BUF" => {
                    assert_ne!(prev_a, usize::MAX);
                    self.pin2aigpin_iv[pinid] = self.pin2aigpin_iv[prev_a];
                },
                _ => unreachable!()
            }
        }
        topo_instack[pinid] = false;
    }

    pub fn from_netlistdb(netlistdb: &NetlistDB) -> AIG {
        let mut aig = AIG {
            num_aigpins: 0,
            pin2aigpin_iv: vec![usize::MAX; netlistdb.num_pins],
            drivers: vec![DriverType::Tie0],
            ..Default::default()
        };

        for cellid in 1..netlistdb.num_cells {
            let celltype = netlistdb.celltypes[cellid].as_str();
            // Clocked word-level macros (DSP48E2_* via PREG, SRLC32E via the
            // shift register) participate in the same global clock domain as
            // DFFs, so their CLK pins must be pre-traced here too -- otherwise
            // their clock flag AIG pins are never created and the second pass
            // below asserts. CARRY4 is combinational and has no CLK.
            let is_clocked_macro = MacroKind::from_celltype(celltype)
                .map(|k| k.has_clock()) == Some(true);
            if !matches!(celltype, "DFF" | "DFFSR" | "$__RAMGEM_SYNC_") &&
                !is_clocked_macro
            {
                continue
            }
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                if !matches!(netlistdb.pinnames[pinid].1.as_str(),
                            "CLK" | "PORT_R_CLK" | "PORT_W_CLK") {
                    continue
                }
                if let Err(pinid) = aig.trace_clock_pin(
                    netlistdb, pinid, false,
                    true
                ) {
                    use netlistdb::GeneralHierName;
                    panic!("Tracing clock pin of cell {} error: \
                            there is a multi-input cell driving {} \
                            that clocks this sequential element. \
                            Clock gating need to be manually patched atm.",
                           netlistdb.cellnames[cellid].dbg_fmt_hier(),
                           netlistdb.pinnames[pinid].dbg_fmt_pin());
                }
            }
        }
        for (&clk, &(flagr, flagf)) in &aig.clock_pin2aigpins {
            clilog::info!(
                "inferred clock port {} ({})",
                netlistdb.pinnames[clk].dbg_fmt_pin(),
                match (flagr, flagf) {
                    (_, usize::MAX) => "posedge",
                    (usize::MAX, _) => "negedge",
                    _ => "posedge & negedge"
                }
            );
        }

        let mut topo_vis = vec![false; netlistdb.num_pins];
        let mut topo_instack = vec![false; netlistdb.num_pins];

        for pinid in 0..netlistdb.num_pins {
            aig.dfs_netlistdb_build_aig(
                netlistdb, &mut topo_vis, &mut topo_instack,
                pinid
            );
        }

        for cellid in 0..netlistdb.num_cells {
            if matches!(netlistdb.celltypes[cellid].as_str(), "DFF" | "DFFSR") {
                let mut ap_s_iv = 1;
                let mut ap_r_iv = 1;
                let mut ap_d_iv = 0;
                let mut ap_clken_iv = 0;
                for pinid in netlistdb.cell2pin.iter_set(cellid) {
                    let pin_iv = aig.pin2aigpin_iv[pinid];
                    match netlistdb.pinnames[pinid].1.as_str() {
                        "D" => ap_d_iv = pin_iv,
                        "S" => ap_s_iv = pin_iv,
                        "R" => ap_r_iv = pin_iv,
                        "CLK" => ap_clken_iv = aig.trace_clock_pin(
                            netlistdb, pinid, false,
                            false
                        ).unwrap(),
                        _ => {}
                    }
                }
                let mut d_in = ap_d_iv;

                d_in = aig.add_and_gate(d_in ^ 1, ap_s_iv) ^ 1;
                ap_clken_iv = aig.add_and_gate(ap_clken_iv ^ 1, ap_s_iv) ^ 1;
                d_in = aig.add_and_gate(d_in, ap_r_iv);
                ap_clken_iv = aig.add_and_gate(ap_clken_iv ^ 1, ap_r_iv) ^ 1;
                let dff = aig.dffs.entry(cellid).or_default();
                dff.en_iv = ap_clken_iv;
                dff.d_iv = d_in;
                assert_ne!(dff.q, 0);
            }
            else if netlistdb.celltypes[cellid].as_str() == "$__RAMGEM_SYNC_" {
                let mut sram = aig.srams.entry(cellid).or_default().clone();
                let mut write_clken_iv = 0;
                for pinid in netlistdb.cell2pin.iter_set(cellid) {
                    let bit = netlistdb.pinnames[pinid].2.map(|i| i as usize);
                    let pin_iv = aig.pin2aigpin_iv[pinid];
                    match netlistdb.pinnames[pinid].1.as_str() {
                        "PORT_R_ADDR" => {
                            sram.port_r_addr_iv[bit.unwrap()] = pin_iv;
                        },
                        "PORT_R_CLK" => {
                            sram.port_r_en_iv = aig.trace_clock_pin(
                                netlistdb, pinid, false,
                                false
                            ).unwrap();
                        },
                        "PORT_W_ADDR" => {
                            sram.port_w_addr_iv[bit.unwrap()] = pin_iv;
                        }
                        "PORT_W_CLK" => {
                            write_clken_iv = aig.trace_clock_pin(
                                netlistdb, pinid, false,
                                false
                            ).unwrap();
                        },
                        "PORT_W_WR_DATA" => {
                            sram.port_w_wr_data_iv[bit.unwrap()] = pin_iv;
                        },
                        "PORT_W_WR_EN" => {
                            sram.port_w_wr_en_iv[bit.unwrap()] = pin_iv;
                        },
                        _ => {}
                    }
                }
                for i in 0..32 {
                    let or_en = sram.port_w_wr_en_iv[i];
                    let or_en = aig.add_and_gate(
                        or_en, write_clken_iv
                    );
                    sram.port_w_wr_en_iv[i] = or_en;
                }
                *aig.srams.get_mut(&cellid).unwrap() = sram;
            }
            else if let Some(kind) = MacroKind::from_celltype(
                netlistdb.celltypes[cellid].as_str()
            ) {
                // Resolve macro inputs now that every pin has an AIG mapping.
                // A macro whose outputs are all unused was never reached by the
                // DFS and has no entry; it is dead logic, so skip it.
                if !aig.macros.contains_key(&cellid) {
                    use netlistdb::GeneralHierName;
                    clilog::debug!(
                        "macro cell {} ({:?}) has no used outputs, skipping",
                        netlistdb.cellnames[cellid].dbg_fmt_hier(), kind
                    );
                    continue
                }
                let mut inputs = (0..kind.num_inputs())
                    .map(|s| kind.default_input(s))
                    .collect::<Vec<_>>();
                let mut clk_en_iv = 1;
                for pinid in netlistdb.cell2pin.iter_set(cellid) {
                    let pinname = netlistdb.pinnames[pinid].1.as_str();
                    let pinbit = netlistdb.pinnames[pinid].2.map(|i| i as usize);
                    if kind.has_clock() && pinname == "CLK" {
                        clk_en_iv = aig.trace_clock_pin(
                            netlistdb, pinid, false, false
                        ).unwrap();
                        continue
                    }
                    if let Some(slot) = kind.input_slot(pinname, pinbit) {
                        inputs[slot] = aig.pin2aigpin_iv[pinid];
                    }
                }
                let m = aig.macros.get_mut(&cellid).unwrap();
                m.inputs = inputs;
                m.clk_en_iv = clk_en_iv;
            }
        }

        // Index the macros: which are endpoints, and which AIG pin each output
        // belongs to. Both are needed before the fanout CSR is built.
        for (macro_i, (_cellid, m)) in aig.macros.iter().enumerate() {
            if m.kind.is_endpoint() {
                aig.stateful_macros.push(macro_i);
            }
            for &o in &m.outputs {
                if o != 0 {
                    aig.aigpin2macro.insert(o, macro_i);
                }
            }
        }

        // Fanout CSR. Historically only AndGate contributed edges; macro
        // outputs now do too, otherwise anything walking fanouts (staging's
        // liveness counting, repcut) sees a macro's inputs as dead and prunes
        // nodes that are still needed.
        //
        // The edge set is per-output, matching topo_traverse_generic: a DSP
        // `P` bit contributes nothing (it is a state read), while a CARRY4
        // `O[i]` contributes only its triangular slice of `S`/`DI`.
        fn for_each_fanin(
            drivers: &Vec<DriverType>,
            macros: &IndexMap<usize, MacroInst>,
            i: usize,
            mut f: impl FnMut(usize)
        ) {
            match drivers[i] {
                DriverType::AndGate(a, b) => {
                    if (a >> 1) != 0 { f(a >> 1) }
                    if (b >> 1) != 0 { f(b >> 1) }
                }
                DriverType::Macro(cellid, slot) => {
                    if let Some(m) = macros.get(&cellid) {
                        m.for_each_comb_fanin(slot, f);
                    }
                }
                _ => {}
            }
        }

        let mut fanouts_start = vec![0usize; aig.num_aigpins + 2];
        for i in 0..aig.drivers.len() {
            for_each_fanin(&aig.drivers, &aig.macros, i, |src| {
                fanouts_start[src] += 1;
            });
        }
        for i in 1..aig.num_aigpins + 2 {
            fanouts_start[i] += fanouts_start[i - 1];
        }
        let mut fanouts = vec![0usize; fanouts_start[aig.num_aigpins + 1]];
        for i in 0..aig.drivers.len() {
            for_each_fanin(&aig.drivers, &aig.macros, i, |src| {
                let st = fanouts_start[src] - 1;
                fanouts_start[src] = st;
                fanouts[st] = i;
            });
        }
        aig.fanouts_start = fanouts_start;
        aig.fanouts = fanouts;

        if !aig.macros.is_empty() {
            let (mut n_c4, mut n_dsp, mut n_srl) = (0, 0, 0);
            for m in aig.macros.values() {
                match m.kind {
                    MacroKind::Carry4 => n_c4 += 1,
                    MacroKind::Dsp48e2 { .. } => n_dsp += 1,
                    MacroKind::Srlc32e => n_srl += 1,
                }
            }
            clilog::info!(
                "intercepted {} word-level macros: {} CARRY4, {} DSP48E2, {} SRLC32E",
                aig.macros.len(), n_c4, n_dsp, n_srl
            );
            clilog::info!(
                "  {} stateful macro endpoints, {} need mid-partition evaluation",
                aig.stateful_macros.len(),
                aig.macros.values()
                    .filter(|m| m.kind.needs_mid_partition_eval()).count()
            );
        }

        aig
    }

    /// Do any intercepted macros need scheduler support that does not exist
    /// yet?
    ///
    /// Call this from a simulation entry point to fail loudly rather than
    /// silently producing wrong waveforms.
    pub fn has_unscheduled_macros(&self) -> bool {
        !self.macros.is_empty()
    }

    pub fn topo_traverse_generic(
        &self,
        endpoints: Option<&Vec<usize>>,
        is_primary_input: Option<&IndexSet<usize>>,
    ) -> Vec<usize> {
        let mut vis = IndexSet::new();
        let mut ret = Vec::new();
        fn dfs_topo(aig: &AIG, vis: &mut IndexSet<usize>, ret: &mut Vec<usize>, is_primary_input: Option<&IndexSet<usize>>, u: usize) {
            if vis.contains(&u) {
                return
            }
            vis.insert(u);
            if is_primary_input.map(|s| s.contains(&u)) != Some(true) {
                match aig.drivers[u] {
                    DriverType::AndGate(a, b) => {
                        if (a >> 1) != 0 {
                            dfs_topo(aig, vis, ret, is_primary_input, a >> 1);
                        }
                        if (b >> 1) != 0 {
                            dfs_topo(aig, vis, ret, is_primary_input, b >> 1);
                        }
                    }
                    // Macro outputs carry per-slot combinational fan-in. This
                    // must be traversed or the scheduler will happily emit a
                    // partition that reads a CARRY4 `S` bit, or an SRLC32E
                    // address bit, that nothing computed -- a silent
                    // mis-schedule rather than a crash. Stateful outputs
                    // (DSP `P`, SRL `Q31`) return an empty fan-in and behave
                    // as graph leaves, exactly like a DFF `Q`.
                    DriverType::Macro(cellid, slot) => {
                        let inst = aig.macros.get(&cellid)
                            .expect("macro driver without an instance");
                        let mut fanin = Vec::new();
                        inst.for_each_comb_fanin(slot, |i| fanin.push(i));
                        for i in fanin {
                            dfs_topo(aig, vis, ret, is_primary_input, i);
                        }
                    }
                    _ => {}
                }
            }
            ret.push(u);
        }
        if let Some(endpoints) = endpoints {
            for &endpoint in endpoints {
                dfs_topo(self, &mut vis, &mut ret, is_primary_input, endpoint);
            }
        }
        else {
            for i in 1..self.num_aigpins + 1 {
                dfs_topo(self, &mut vis, &mut ret, is_primary_input, i);
            }
        }
        ret
    }

    /// Endpoint groups are laid out as four consecutive ranges:
    ///
    /// ```text
    ///   [ primary outputs | DFFs | SRAMs | stateful macros ]
    /// ```
    ///
    /// Stateful macros are appended last so that every pre-existing endpoint
    /// id keeps its meaning and previously serialized `.gemparts` files stay
    /// interpretable for macro-free designs.
    pub fn num_endpoint_groups(&self) -> usize {
        self.primary_outputs.len() + self.dffs.len() + self.srams.len()
            + self.stateful_macros.len()
    }

    pub fn get_endpoint_group(&self, endpt_id: usize) -> EndpointGroup<'_> {
        let n_po = self.primary_outputs.len();
        let n_dff = self.dffs.len();
        let n_sram = self.srams.len();
        if endpt_id < n_po {
            EndpointGroup::PrimaryOutput(*self.primary_outputs.get_index(endpt_id).unwrap())
        }
        else if endpt_id < n_po + n_dff {
            EndpointGroup::DFF(&self.dffs[endpt_id - n_po])
        }
        else if endpt_id < n_po + n_dff + n_sram {
            EndpointGroup::RAMBlock(&self.srams[endpt_id - n_po - n_dff])
        }
        else {
            let macro_i = self.stateful_macros[endpt_id - n_po - n_dff - n_sram];
            EndpointGroup::Macro(&self.macros[macro_i])
        }
    }

    /// The endpoint group id of a stateful macro, given its position in
    /// [Self::stateful_macros].
    pub fn macro_endpoint_id(&self, stateful_idx: usize) -> usize {
        self.primary_outputs.len() + self.dffs.len() + self.srams.len() + stateful_idx
    }
}
