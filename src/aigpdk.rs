// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! AIGPDK is a special artificial cell library used in GEM.

use netlistdb::{Direction, LeafPinProvider};
use compact_str::CompactString;
use sverilogparse::SVerilogRange;
use crate::macros::MacroKind;

/// This implements direction and width providers for
/// AIG PDK cells.
///
/// You can use it in netlistdb construction.
pub struct AIGPDKLeafPins();

/// The addr width of an SRAM.
///
/// The word width is always 32.
/// If you change this, make sure to change all other occurences in this
/// project as well as the definitions in PDK libraries.
pub const AIGPDK_SRAM_ADDR_WIDTH: usize = 13;

pub const AIGPDK_SRAM_SIZE: usize = 1 << 13;

impl LeafPinProvider for AIGPDKLeafPins {
    fn direction_of(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString, pin_idx: Option<isize>
    ) -> Direction {
        match (macro_name.as_str(), pin_name.as_str(), pin_idx) {
            ("INV" | "BUF", "A", None) => Direction::I,
            ("INV" | "BUF", "Y", None) => Direction::O,

            ("AND2_00_0" | "AND2_01_0" | "AND2_10_0" | "AND2_11_0" |
             "AND2_11_1", "A" | "B", None) => Direction::I,
            ("AND2_00_0" | "AND2_01_0" | "AND2_10_0" | "AND2_11_0" |
             "AND2_11_1", "Y", None) => Direction::O,

            ("DFF" | "LATCH", "CLK" | "D", None) => Direction::I,
            ("DFFSR", "CLK" | "D" | "S" | "R", None) => Direction::I,
            ("DFF" | "DFFSR" | "LATCH", "Q", None) => Direction::O,

            ("CKLNQD", "CP" | "E", None) => Direction::I,
            ("CKLNQD", "Q", None) => Direction::O,

            ("$__RAMGEM_ASYNC_", _, _) => {
                panic!("Async RAM (lib cell {}) not supported yet in GEM.", macro_name);
            },

            ("$__RAMGEM_SYNC_",
             "PORT_R_CLK" | "PORT_W_CLK",
             None) => Direction::I,
            ("$__RAMGEM_SYNC_",
             "PORT_R_ADDR" | "PORT_W_ADDR",
             Some(0..=12)) => Direction::I,
            ("$__RAMGEM_SYNC_",
             "PORT_W_WR_EN" | "PORT_W_WR_DATA",
             Some(0..=31)) => Direction::I,
            ("$__RAMGEM_SYNC_",
             "PORT_R_RD_DATA",
             Some(0..=31)) => Direction::O,

            _ => {
                // Natively-evaluated word-level macros (CARRY4, DSP48E2_*,
                // SRLC32E). Their pin tables live in crate::macros so that
                // netlist typing and AIG construction share one definition.
                if let Some(kind) = MacroKind::from_celltype(macro_name.as_str()) {
                    if let Some(dir) = kind.direction_of(pin_name.as_str(), pin_idx) {
                        return dir
                    }
                    use netlistdb::{GeneralPinName, HierName};
                    panic!("Macro cell {:?} has no pin {}. Check that the Yosys \
                            macro-interception pass emits the pin set declared \
                            in src/macros.rs.",
                           kind,
                           (HierName::single(macro_name.clone()),
                            pin_name, pin_idx).dbg_fmt_pin());
                }

                use netlistdb::{GeneralPinName, HierName};
                panic!("Cannot recognize pin type {}, please make sure the verilog netlist is synthesized in GEM's aigpdk.",
                       (HierName::single(macro_name.clone()),
                        pin_name, pin_idx).dbg_fmt_pin());
            }
        }
    }

    fn width_of(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString
    ) -> Option<SVerilogRange> {
        match (macro_name.as_str(), pin_name.as_str()) {
            ("INV" | "BUF", "A" | "Y") => None,
            ("AND2_00_0" | "AND2_01_0" | "AND2_10_0" | "AND2_11_0" |
             "AND2_11_1", "A" | "B" | "Y") => None,
            ("DFF" | "DFFSR" | "LATCH", "CLK" | "D" | "Q" | "S" | "R") => None,
            ("CKLNQD", "CP" | "E" | "Q") => None,
            ("$__RAMGEM_SYNC_",
             "PORT_R_CLK" | "PORT_W_CLK") => None,
            ("$__RAMGEM_SYNC_",
             "PORT_R_ADDR" | "PORT_W_ADDR")
                => Some(SVerilogRange(12, 0)),
            ("$__RAMGEM_SYNC_",
             "PORT_W_WR_EN" | "PORT_W_WR_DATA" | "PORT_R_RD_DATA")
                => Some(SVerilogRange(31, 0)),
            _ => {
                // Word-level macros. Note SVerilogRange is (msb, lsb): the
                // stock entries above declare a 13-bit bus as (12, 0), so a
                // 48-bit P port is (47, 0), not (0, 47).
                if let Some(kind) = MacroKind::from_celltype(macro_name.as_str()) {
                    return kind.width_of(pin_name.as_str())
                }
                None
            }
        }
    }
}
