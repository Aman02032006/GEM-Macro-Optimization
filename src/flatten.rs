// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Partition scheduler and flattener

use crate::aig::{AIG, EndpointGroup, DriverType};
use crate::aigpdk::AIGPDK_SRAM_ADDR_WIDTH;
use crate::pe::{Partition, BOOMERANG_NUM_STAGES, MacroStage};
use crate::macros::*;
use crate::staging::StagedAIG;
use indexmap::IndexMap;
use std::collections::BTreeMap;
use ulib::UVec;

pub const NUM_THREADS_V1: usize = 1 << (BOOMERANG_NUM_STAGES - 5);

/// A flattened script, for partition executor version 1.
/// See [FlattenedScriptV1::blocks_data] for the format details.
///
/// Generally, a script contains a number of major stages.
/// Each stage consists of the same number of blocks.
/// Each block contains a list of flattened partitions.
pub struct FlattenedScriptV1 {
    /// the number of blocks
    pub num_blocks: usize,
    /// the number of major stages
    pub num_major_stages: usize,
    /// the CSR start indices of stages and blocks.
    ///
    /// this length is num_blocks * num_major_stages + 1
    pub blocks_start: UVec<usize>,
    /// the partition instructions.
    ///
    /// the instructions follow a special format.
    /// it consists of zero or more partitions.
    /// 1. metadata section [1x256]:
    ///    the number of boomerang stages.
    ///      32-bit
    ///      if this is zero, the stage will not run. only happens
    ///      when the block has no partition mapped onto it.
    ///    is this the last boomerang stage?
    ///      32-bit, only 0 or 1
    ///    the number of valid write-outs.
    ///      32-bit
    ///    the write-out destination offset.
    ///      32-bit
    ///      (at this offset, we put all FFs and other outputs.)
    ///    the srams count and offsets.
    ///      32-bit count
    ///      32-bit memory offset for the first mem.
    ///    the number of global read rounds
    ///      32-bit count
    ///    the number of output-duplicate writeouts
    ///      32-bit count
    ///      this is used when one output pin is used by either
    ///      <both output and FFs>, or <multiple FFs with different
    ///      enabling conditions>.
    ///    padding towards 128
    ///    the location of early write-outs, excl. mem.
    ///      256 * [#boomstage & 0~256 id], compressed to 128
    /// 2. initial read-global permutation [2x]*rounds
    ///    32-bit indices*1 for each of the 256 threads.
    ///    32-bit valid mask for each of the 256 threads.
    ///    if the valid mask is zero, the memory is not read.
    ///    the result should be stored "like pext instruction",
    ///    but "reversed", and then appended to the low bits
    ///    in each round.
    ///    the index is encoded with a type bit at the highest bit:
    ///      if it is 0: it means it is offset from previous iteration.
    ///      if it is 1: it is offset from current iteration
    ///        which means it is an intermediate value coming from
    ///        the same cycle but a previous major stage.
    /// 3. boomerang sections, repeat below N*16
    ///    1. local shuffle permutation
    ///       32-bit indices * 16 for each of the 256 threads.
    ///    2. input (with inv) * 8192bits * (3+1padding)
    ///       32-bit * 256 threads * (3+1): xora, xorb, orb, [0 padding]
    ///       0xy: and gate, out = (a^x)&(b^y).
    ///       100: passthrough, out = a.
    ///       111: invalid, can be skipped.
    ///    -. write out, according to rest
    /// 4. global write-outs.
    ///    1. sram & additional endpoint copy permutations, inv. [16x].
    ///       only the inputs within sram and endpoint copy
    ///       range will be considered.
    ///       followed by [4x] invert, set0, and two 0 paddings.
    ///    2. permutation for the write-out enabler pins, inv. [16]
    ///       include itself inv and data inv.
    ///       followed by [3(+1padding)x]
    ///         clock invert, clock set0, data invert, and 0 padding
    ///    -. commit the write-out
    pub blocks_data: UVec<u32>,
    /// the state size including DFF and I/O states only.
    ///
    /// the inputs are always at front.
    pub reg_io_state_size: u32,
    /// the u32 array length for storing SRAMs.
    pub sram_storage_size: u32,
    /// the u32 array length for storing persistent word-level macro state
    /// (SRLC32E shift registers, DSP48E2 PREG).
    ///
    /// Laid out in block-affinity order like [Self::sram_storage_size], so a
    /// block's macros occupy one contiguous run.
    pub macro_storage_size: u32,
    /// expected input AIG pins layout
    pub input_layout: Vec<usize>,
    /// maps from primary outputs, FF:D and SRAM:PORT_R_RD_DATA AIG pins
    /// to state offset, index with invert.
    pub output_map: IndexMap<usize, u32>,
    /// maps from primary inputs, FF:Q/SRAM:* input AIG pins to state offset,
    /// index WITHOUT invert.
    pub input_map: IndexMap<usize, u32>,
    /// (for debug purpose) the relation between major stage, block and
    /// part indices as given in construction.
    pub stages_blocks_parts: Vec<Vec<Vec<usize>>>,
}

fn map_global_read_to_rounds(
    inputs_taken: &BTreeMap<u32, u32>
) -> Vec<Vec<(u32, u32)>> {
    let inputs_taken = inputs_taken.iter()
        .map(|(&a, &b)| (a, b)).collect::<Vec<_>>();
    // the larger the sorting chunk size, the better the successful chance,
    // but the less efficient due to worse cache coherency.
    let mut chunk_size = inputs_taken.len();
    while chunk_size >= 1 {
        let mut slices = inputs_taken.chunks(chunk_size).collect::<Vec<_>>();
        slices.sort_by_cached_key(|&slice| {
            u32::MAX - slice.iter()
                .map(|(_, mask)| mask.count_ones()).sum::<u32>()
        });
        let mut rounds_idx_masks: Vec<Vec<(u32, u32)>> = vec![vec![]; NUM_THREADS_V1];
        let mut round_map_j = 0;
        let mut fail = false;
        for slice in slices {
            for &(offset, mask) in slice {
                let wrap_fail_j = round_map_j;
                while rounds_idx_masks[round_map_j].iter().map(|(_, mask)| mask.count_ones()).sum::<u32>() + mask.count_ones() > 32 {
                    round_map_j += 1;
                    if round_map_j == NUM_THREADS_V1 {
                        round_map_j = 0;
                    }
                    if round_map_j == wrap_fail_j {
                        // panic!("failed to map at part {} mem offset {}", i, offset);
                        fail = true;
                        break
                    }
                }
                if fail { break }
                rounds_idx_masks[round_map_j].push((offset, mask));
                round_map_j += 1;
                if round_map_j == NUM_THREADS_V1 {
                    round_map_j = 0;
                }
            }
            if fail { break }
        }
        if !fail {
            // let max_rounds = rounds_idx_masks.iter().map(|v| v.len()).max().unwrap();
            // println!("max_rounds: {}, round_map_j: {}, inputs_taken len {}", max_rounds, round_map_j, inputs_taken.len());
            return rounds_idx_masks
        }
        chunk_size /= 2;
    }
    panic!("cannot map global init to any multiples of rounds.");
}

/// Claim the next free bit position in the shared-state image.
///
/// Scanning starts in the hier[1] range (4096..8191). That range is where the
/// tree writes its level-1 results, and `build_one_boomerang_stage` leaves the
/// slots it did not need as `usize::MAX`. The macro phase runs *after* the
/// tree reduction, so overwriting a spare slot there is free: nothing has read
/// it and the next stage's shuffle addresses it like any other position.
fn alloc_free_pos(occupancy: &mut Vec<bool>, cursor: &mut usize) -> u16 {
    let hi = 1usize << BOOMERANG_NUM_STAGES;
    while *cursor < hi && occupancy[*cursor] { *cursor += 1 }
    assert!(*cursor < hi,
            "no free shared-state slot left for a macro output; the partition              needs to be split");
    occupancy[*cursor] = true;
    let p = *cursor as u16;
    *cursor += 1;
    p
}

/// temporaries for a part being flattened. will be discarded after built.
#[derive(Debug, Clone, Default)]
struct FlatteningPart {
    /// for each boomerang stage, the result bits layout.
    afters: Vec<Vec<usize>>,
    /// for each partition, the output bits layout not containing sram outputs yet.
    parts_after_writeouts: Vec<usize>,
    /// mapping from aig pin index to writeout position (0~8192)
    after_writeout_pin2pos: IndexMap<usize, u16>,
    /// the number of SRAMs to simulate in this part.
    num_srams: u32,
    /// number of normal writeouts
    num_normal_writeouts: u32,
    /// number of writeout slots for output duplication
    num_duplicate_writeouts: u32,
    /// number of total writeouts
    num_writeouts: u32,
    /// the outputs categorized into activations
    comb_outputs_activations: IndexMap<usize, IndexMap<usize, Option<u16>>>,
    /// the current (placed) count of duplicate permutes
    cnt_placed_duplicate_permute: u32,

    /// bit position of a state word that is never written by anyone.
    ///
    /// The state buffer is zero-initialised and this word has no writer, so it
    /// reads 0 for the whole simulation. It gives a home to flip-flops whose D
    /// is a hard constant zero -- see make_inputs_outputs.
    zero_bit_pos: u32,
    /// the starting offset for FFs, outputs, and SRAM read results.
    state_start: u32,
    /// the starting offset of SRAM storage.
    sram_start: u32,

    /// the number of DSP48E2 endpoints committed by this part.
    ///
    /// Only DSPs reach the write-out path: CARRY4 is combinational and
    /// SRLC32E commits inside its macro phase.
    num_dsps: u32,
    /// write-out words used by DSP P ports (2 each, 48 bits).
    dsp_state_words: u32,
    /// total u32 words of persistent macro state owned by this part.
    macro_state_words: u32,
    /// the starting offset of this part's run in the global macro-state array.
    ///
    /// Allocated in block-affinity order, exactly like [Self::sram_start], so
    /// that the macros a block touches occupy one contiguous run. Without this
    /// the "coalesced SoA" property is fiction: a block whose macros are
    /// instances {3, 91, 400} issues three scattered loads instead of one.
    macro_state_start: u32,
    /// dense local id assigned to each macro (position in AIG::macros ->
    /// local index within this part). This is the renumbering that makes the
    /// per-plane arrays coalesce.
    macro_local_ids: IndexMap<usize, u32>,

    /// the partial permutation instructions for
    /// 1. sram inputs
    /// 2. duplicated output pins due to difference in polarity/clock en.
    ///
    /// len: 8192
    sram_duplicate_permute: Vec<u16>,
    /// invert bit for sram_duplicate.
    ///
    /// len: 256
    sram_duplicate_inv: Vec<u32>,
    /// set-0 bit for sram_duplicate.
    ///
    /// len: 256
    sram_duplicate_set0: Vec<u32>,
    /// the permutation for clock enable pins.
    ///
    /// len: 8192
    clken_permute: Vec<u16>,
    /// invert bit for clken
    ///
    /// len: 256
    clken_inv: Vec<u32>,
    /// set-0 bit for clken
    ///
    /// len: 256
    clken_set0: Vec<u32>,
    /// invert bit for data corresponding to clken
    ///
    /// len: 256
    data_inv: Vec<u32>,
}

fn set_bit_in_u32(v: &mut u32, pos: u32, bit: u8) {
    if bit != 0 {
        *v |= 1 << pos;
    }
    else {
        *v &= !(1 << pos);
    }
}

impl FlatteningPart {
    fn init_afters_writeouts(
        &mut self, aig: &AIG, staged: &StagedAIG, part: &Partition
    ) {
        let afters = part.stages.iter().map(|s| {
            let mut after = Vec::with_capacity(1 << BOOMERANG_NUM_STAGES);
            after.push(usize::MAX);
            for i in (1..=BOOMERANG_NUM_STAGES).rev() {
                after.extend(s.hier[i].iter().copied());
            }
            after
        }).collect::<Vec<_>>();
        let wos = part.stages.iter().zip(afters.iter()).map(|(s, after)| {
            s.write_outs.iter().map(|&woi| {
                after[woi * 32..(woi + 1) * 32].iter().copied()
            }).flatten()
        }).flatten().collect::<Vec<_>>();

        // println!("test wos: {:?}", wos);

        self.afters = afters;
        self.parts_after_writeouts = wos;
        self.num_normal_writeouts = part.stages.iter()
            .map(|s| s.write_outs.len()).sum::<usize>() as u32;
        self.num_srams = 0;

        // map: output aig pin id -> ((clken, data iv) -> pos)
        let mut comb_outputs_activations =
            IndexMap::<usize, IndexMap<usize, Option<u16>>>::new();
        for &endpt_i in &part.endpoints {
            match staged.get_endpoint_group(aig, endpt_i) {
                EndpointGroup::RAMBlock(_) => {
                    self.num_srams += 1;
                },
                EndpointGroup::PrimaryOutput(idx) => {
                    comb_outputs_activations.entry(idx >> 1)
                        .or_default().insert(2 | (idx & 1), None);
                },
                EndpointGroup::StagedIOPin(idx) => {
                    comb_outputs_activations.entry(idx)
                        .or_default().insert(2, None);
                },
                EndpointGroup::DFF(dff) => {
                    comb_outputs_activations.entry(dff.d_iv >> 1)
                        .or_default().insert(
                            dff.en_iv << 1 | (dff.d_iv & 1),
                            None);
                },
                EndpointGroup::Macro(m) => {
                    self.num_dsps += 1;
                    self.dsp_state_words += 2;
                    self.macro_state_words += m.kind.state_words() as u32;
                },
            }
        }
        self.num_duplicate_writeouts = ((
            comb_outputs_activations.values()
                .map(|v| v.len() - 1).sum::<usize>()
                + 31) / 32) as u32;
        self.comb_outputs_activations = comb_outputs_activations;

        // Write-out word layout: [normal][dsp][duplicates][sram]. DSP words go
        // before the duplicates so the existing sram/duplicate offsets, which
        // are expressed relative to the end, stay exactly as they were.
        self.num_writeouts = self.num_normal_writeouts + self.dsp_state_words
            + self.num_srams + self.num_duplicate_writeouts;

        self.after_writeout_pin2pos = self.parts_after_writeouts.iter().enumerate()
            .filter_map(|(i, &pin)| {
                if pin == usize::MAX { None }
                else { Some((pin, i as u16)) }
            })
            .collect::<IndexMap<_, _>>();
    }

    /// returns permutation id, invert bit, and setzero bit
    fn query_permute_with_pin_iv(&self, pin_iv: usize) -> (u16, u8, u8) {
        if pin_iv <= 1 {
            return (0, pin_iv as u8, 1)
        }
        let pos = self.after_writeout_pin2pos.get(&(pin_iv >> 1)).unwrap();
        (*pos, (pin_iv & 1) as u8, 0)
    }

    /// places a sram_duplicate bit.
    fn place_sram_duplicate(&mut self, pos: usize, (perm, inv, set0): (u16, u8, u8)) {
        self.sram_duplicate_permute[pos] = perm;
        set_bit_in_u32(&mut self.sram_duplicate_inv[pos >> 5],
                       (pos & 31) as u32, inv);
        set_bit_in_u32(&mut self.sram_duplicate_set0[pos >> 5],
                       (pos & 31) as u32, set0);
    }

    /// places a writeout bit's clock enable and data invert.
    fn place_clken_datainv(
        &mut self, pos: usize,
        clken_iv_perm: u16, clken_iv_inv: u8, clken_iv_set0: u8, data_inv: u8
    ) {
        self.clken_permute[pos] = clken_iv_perm;
        set_bit_in_u32(&mut self.clken_inv[pos >> 5],
                       (pos & 31) as u32, clken_iv_inv);
        set_bit_in_u32(&mut self.clken_set0[pos >> 5],
                       (pos & 31) as u32, clken_iv_set0);
        set_bit_in_u32(&mut self.data_inv[pos >> 5],
                       (pos & 31) as u32, data_inv);
    }

    /// returns a final local position for a data output bit with given pin_iv and clken_iv.
    ///
    /// if is not already placed, we will place it as well as place
    /// the clock enable bit, duplication bit, and bitflags for clock and data.
    fn get_or_place_output_with_activation(&mut self, pin_iv: usize, clken_iv: usize) -> u16 {
        let (activ_idx, _, pos) = self.comb_outputs_activations
            .get(&(pin_iv >> 1)).unwrap()
            .get_full(&(clken_iv << 1 | (pin_iv & 1))).unwrap();
        if let Some(pos) = *pos {
            return pos
        }
        let (clken_iv_perm, clken_iv_inv, clken_iv_set0) = self.query_permute_with_pin_iv(clken_iv);
        let origpos = match self.after_writeout_pin2pos.get(&(pin_iv >> 1)) {
            Some(origpos) => *origpos,
            None => {
                panic!("position of pin_iv {} (clken_iv {}) not found.. buggy boomerang, check if netlist and gemparts mismatch.", pin_iv, clken_iv)
            }
        } as usize;
        let r_pos = if activ_idx == 0 {
            self.place_clken_datainv(
                origpos, clken_iv_perm, clken_iv_inv, clken_iv_set0, (pin_iv & 1) as u8
            );
            origpos as u16
        }
        else {
            self.cnt_placed_duplicate_permute += 1;
            let dup_pos = ((self.num_writeouts - self.num_srams) * 32 - self.cnt_placed_duplicate_permute) as usize;
            // Gather-lane layout: [sram: 4 each][dsp: 4 each][duplicates].
            let dup_perm_pos = ((self.num_srams * 4 + self.num_dsps * 4
                + self.num_duplicate_writeouts) * 32
                - self.cnt_placed_duplicate_permute) as usize;
            if dup_perm_pos >= 8192 {
                panic!("sram duplicate bit larger than expected..")
                // dup_perm_pos = 8191;
            }
            self.place_sram_duplicate(
                dup_perm_pos, (origpos as u16, 0, 0)
            );
            self.place_clken_datainv(
                dup_pos, clken_iv_perm, clken_iv_inv, clken_iv_set0, (pin_iv & 1) as u8
            );
            dup_pos as u16
        };
        *self.comb_outputs_activations.get_mut(&(pin_iv >> 1)).unwrap()
            .get_mut(&(clken_iv << 1 | (pin_iv & 1))).unwrap() = Some(r_pos);
        r_pos
    }

    fn make_inputs_outputs(
        &mut self,
        aig: &AIG,
        staged: &StagedAIG,
        part: &Partition,
        input_map: &mut IndexMap<usize, u32>,
        staged_io_map: &mut IndexMap<usize, u32>,
        output_map: &mut IndexMap<usize, u32>,
    ) {
        self.sram_duplicate_permute = vec![0; 1 << BOOMERANG_NUM_STAGES];
        self.sram_duplicate_inv = vec![0u32; NUM_THREADS_V1];
        self.sram_duplicate_set0 = vec![u32::MAX; NUM_THREADS_V1];
        self.clken_permute = vec![0; 1 << BOOMERANG_NUM_STAGES];
        self.clken_inv = vec![0u32; NUM_THREADS_V1];
        self.clken_set0 = vec![u32::MAX; NUM_THREADS_V1];
        self.data_inv = vec![0u32; NUM_THREADS_V1];
        self.cnt_placed_duplicate_permute = 0;

        let mut cur_sram_id = 0;
        let mut cur_dsp_id = 0u32;
        for &endpt_i in &part.endpoints {
            match staged.get_endpoint_group(aig, endpt_i) {
                EndpointGroup::RAMBlock(ram) => {
                    let sram_rd_data_local_offset = self.num_writeouts as usize - self.num_srams as usize + cur_sram_id as usize;
                    let sram_rd_data_global_start = self.state_start + self.num_writeouts - self.num_srams + cur_sram_id;
                    let (perm_r_en_iv, perm_r_en_iv_inv, perm_r_en_iv_set0) = self.query_permute_with_pin_iv(ram.port_r_en_iv);
                    for k in 0..32 {
                        let d = ram.port_r_rd_data[k];
                        if d == usize::MAX { continue }
                        input_map.insert(d, sram_rd_data_global_start * 32 + k as u32);
                        output_map.insert(d << 1, sram_rd_data_global_start * 32 + k as u32);
                        self.place_clken_datainv(
                            sram_rd_data_local_offset * 32 + k,
                            perm_r_en_iv, perm_r_en_iv_inv, perm_r_en_iv_set0, 0
                        );
                    }
                    let sram_input_perm_st = (cur_sram_id * 32 * 4) as usize;
                    for k in 0..13 {
                        self.place_sram_duplicate(
                            sram_input_perm_st + k,
                            self.query_permute_with_pin_iv(ram.port_r_addr_iv[k])
                        );
                        self.place_sram_duplicate(
                            sram_input_perm_st + 16 + k,
                            self.query_permute_with_pin_iv(ram.port_w_addr_iv[k])
                        );
                    }
                    for k in 0..32 {
                        self.place_sram_duplicate(
                            sram_input_perm_st + 32 + k,
                            self.query_permute_with_pin_iv(ram.port_w_wr_en_iv[k])
                        );
                        self.place_sram_duplicate(
                            sram_input_perm_st + 64 + k,
                            self.query_permute_with_pin_iv(ram.port_w_wr_data_iv[k])
                        );
                    }
                    cur_sram_id += 1;
                },
                EndpointGroup::PrimaryOutput(idx_iv) => {
                    if idx_iv == 0 {
                        panic!("primary output has zero..??")
                    }
                    let pos = self.state_start * 32 + self.get_or_place_output_with_activation(
                        idx_iv, 1
                    ) as u32;
                    output_map.insert(idx_iv, pos);
                },
                EndpointGroup::StagedIOPin(idx) => {
                    if idx == 0 {
                        panic!("staged IO pin has zero..??")
                    }
                    let pos = self.state_start * 32 + self.get_or_place_output_with_activation(
                        idx << 1, 1
                    ) as u32;
                    staged_io_map.insert(idx, pos);
                },
                EndpointGroup::Macro(m) => {
                    // DSP48E2 is handled exactly like an SRAM: its inputs are
                    // gathered through the write-out permutation, the macro
                    // unit computes P_next, and the 48 result bits land in
                    // write-out slots that input_map points at, so next
                    // cycle's global read serves P[k] like any DFF Q.
                    let dsp_local = (self.num_writeouts - self.num_srams
                        - self.num_duplicate_writeouts - self.dsp_state_words
                        + cur_dsp_id * 2) as usize;
                    let dsp_global = self.state_start + self.num_writeouts
                        - self.num_srams - self.num_duplicate_writeouts
                        - self.dsp_state_words + cur_dsp_id * 2;

                    // The macro unit applies CEP and the traced clock enable
                    // itself, so the write-out gate is unconditional.
                    let one = self.query_permute_with_pin_iv(1);
                    for k in 0..48usize {
                        let pin = m.outputs[DSP_OUT_P + k];
                        if pin == 0 { continue }
                        // input_map only. Unlike a DFF, whose D (commit) and
                        // Q (read) are distinct AIG pins, a DSP has just P --
                        // so registering it as an output too would claim that
                        // the freshly committed value is what the current
                        // cycle's logic observed, which it is not. The
                        // primary-output endpoint gives P its own write-out
                        // slot when the design exposes it.
                        input_map.insert(pin, dsp_global * 32 + k as u32);
                        self.place_clken_datainv(
                            dsp_local * 32 + k, one.0, one.1, one.2, 0);
                    }

                    // 128 gather bits: 121 inputs, the clock enable, and the
                    // ALU configuration as constants. Encoding the config here
                    // rather than in metadata keeps it per-instance without a
                    // second lookup, since a constant is just a permute entry
                    // with set0 set.
                    let st = (self.num_srams * 4 * 32 + cur_dsp_id * 128) as usize;
                    for slot in 0..DSP_NUM_INPUTS {
                        self.place_sram_duplicate(
                            st + slot,
                            self.query_permute_with_pin_iv(m.inputs[slot]));
                    }
                    self.place_sram_duplicate(
                        st + DSP_NUM_INPUTS,
                        self.query_permute_with_pin_iv(m.clk_en_iv));
                    let (st_code, preadd) = match m.kind {
                        MacroKind::Dsp48e2 { state, use_preadder } =>
                            (state as usize, use_preadder as usize),
                        _ => panic!("non-DSP macro reached the write-out path"),
                    };
                    self.place_sram_duplicate(st + DSP_NUM_INPUTS + 1,
                        self.query_permute_with_pin_iv(st_code & 1));
                    self.place_sram_duplicate(st + DSP_NUM_INPUTS + 2,
                        self.query_permute_with_pin_iv((st_code >> 1) & 1));
                    self.place_sram_duplicate(st + DSP_NUM_INPUTS + 3,
                        self.query_permute_with_pin_iv(preadd));
                    cur_dsp_id += 1;
                },
                EndpointGroup::DFF(dff) => {
                    if dff.d_iv == 0 {
                        // D is a hard constant zero, so Q is zero from the
                        // first cycle (all state is zero-initialised). Point Q
                        // at the reserved never-written word.
                        //
                        // The stock code mapped it to position 0 instead, which
                        // is a real PRIMARY INPUT bit -- so such a flop read
                        // that input's value forever. It only shows up when the
                        // synthesiser leaves a constant-D flop behind, which it
                        // legitimately does when it cannot prove the first
                        // cycle without an init value.
                        clilog::warn!(DFF_CONST_ERR,
                            "dff d_iv is constant zero (unoptimized netlist);                              tying Q to the reserved zero word");
                        input_map.insert(dff.q, self.zero_bit_pos);
                        continue
                    }
                    if dff.d_iv == 1 {
                        panic!("dff d_iv is constant ONE, which has no                                 always-one state position. Re-synthesize with                                 stronger opt, or extend flatten.rs with a                                 reserved ones-word.");
                    }
                    let pos = self.state_start * 32 + self.get_or_place_output_with_activation(
                        dff.d_iv, dff.en_iv
                    ) as u32;
                    output_map.insert(dff.d_iv, pos);
                    input_map.insert(dff.q, pos);
                },
            }
        }
        assert_eq!(cur_sram_id, self.num_srams);
        assert_eq!((self.cnt_placed_duplicate_permute + 31) / 32, self.num_duplicate_writeouts);

        // println!("test clken_permute: {:?}, wos (w/o sram or dup): {:?}", self.clken_permute, self.parts_after_writeouts);
    }

    /// Emit one mid-partition macro phase.
    ///
    /// The section is a fixed `MACRO_MAX_LANES * MACRO_LANE_WORDS` words, so
    /// the script pointer advances by a constant per phase and one warp covers
    /// exactly one phase. Unused lanes have the VALID bit clear.
    ///
    /// Output positions are allocated here rather than in pe.rs because only
    /// the flattener knows bit positions; pe.rs knows pins.
    fn emit_macro_phase(
        &self,
        script: &mut Vec<u32>,
        aig: &AIG,
        ms: &MacroStage,
        last_pin2localpos: &mut IndexMap<usize, u16>,
        occupancy: &mut Vec<bool>,
        macro_state_off: &IndexMap<usize, u32>,
    ) {
        let start = script.len();
        let mut cursor = 1usize << (BOOMERANG_NUM_STAGES - 1);
        let mut newly = Vec::<(usize, u16)>::new();

        for lane in 0..MACRO_MAX_LANES {
            if lane >= ms.num_lanes() {
                for _ in 0..MACRO_LANE_WORDS { script.push(0) }
                continue
            }
            let ml = &ms.lanes[lane];
            let m = &aig.macros[ml.macro_i];
            let kind = m.kind;

            // w0: descriptor
            let state_off = *macro_state_off.get(&ml.macro_i).unwrap_or(&0);
            script.push(
                kind.script_kind_code()
                | ((ml.chain_start as u32) << MACRO_DESC_CHAIN_START_BIT)
                | (1u32 << MACRO_DESC_VALID_BIT)
                | (state_off << MACRO_DESC_STATE_SHIFT)
            );

            // w1..w5: input codes, defaulting to constant zero
            let mut codes = [1u16 << MACRO_PERM_CONST_BIT; MACRO_LANE_IN_SLOTS];
            for slot in 0..kind.num_inputs().min(MACRO_LANE_IN_SLOTS) {
                // A scanned carry-in is produced by the warp scan and must NOT
                // be gathered: its driver was never materialised in the tree,
                // which is the whole point of chaining.
                if !ml.chain_start && matches!(kind, MacroKind::Carry4)
                    && slot == C4_IN_CI
                {
                    continue
                }
                codes[slot] = macro_encode_input(
                    m.inputs[slot],
                    |pin| last_pin2localpos.get(&pin).copied());
            }
            for w in 0..(MACRO_LANE_IN_SLOTS / 2) {
                script.push((codes[w * 2] as u32) | ((codes[w * 2 + 1] as u32) << 16));
            }

            // w6..w9: output positions
            let mut outs = [MACRO_POS_NONE; MACRO_LANE_OUT_SLOTS];
            for slot in 0..kind.num_outputs().min(MACRO_LANE_OUT_SLOTS) {
                let pin = m.outputs[slot];
                if pin == 0 { continue }
                if !kind.output_needs_mid_partition_eval(slot) { continue }
                let pos = alloc_free_pos(occupancy, &mut cursor);
                outs[slot] = pos;
                newly.push((pin, pos));
            }
            for w in 0..(MACRO_LANE_OUT_SLOTS / 2) {
                script.push((outs[w * 2] as u32) | ((outs[w * 2 + 1] as u32) << 16));
            }

            // w10, w11: reserved
            script.push(0);
            script.push(0);
        }

        assert_eq!(script.len() - start, MACRO_MAX_LANES * MACRO_LANE_WORDS,
                   "macro phase section size drifted from the declared layout");

        // Publish the produced pins so the following stage's shuffle finds
        // them exactly like any tree result.
        for (pin, pos) in newly {
            last_pin2localpos.insert(pin, pos);
        }
    }

    fn build_script(
        &self, aig: &AIG, part: &Partition,
        input_map: &IndexMap<usize, u32>,
        staged_io_map: &IndexMap<usize, u32>,
        macro_state_off: &IndexMap<usize, u32>,
    ) -> Vec<u32> {
        let mut script = Vec::<u32>::new();

        // metadata
        script.push(part.stages.len() as u32);
        script.push(0);
        script.push(self.num_writeouts);
        script.push(self.state_start);
        script.push(self.num_srams);
        script.push(self.sram_start);
        script.push(0);   // [6]=num global read rounds, assigned later
        script.push(self.num_duplicate_writeouts);
        // [8] number of mid-partition macro phases, [9..] one header each.
        // Slots 8..127 were zero padding in the stock format, so macro-free
        // designs stay bit-identical to before.
        script.push(part.macro_stages.len() as u32);
        // [9] DSP endpoint count, [10] base of this part's DSP state run in
        // the global macro-state array. build_one records state_macros in
        // endpoint order and the allocator assigns them contiguously, so a
        // single base plus 2 words per DSP addresses them all.
        script.push(self.num_dsps);
        script.push(part.state_macros.first()
            .and_then(|mi| macro_state_off.get(mi)).copied().unwrap_or(0));
        for ms in &part.macro_stages {
            assert!(ms.num_lanes() <= MACRO_MAX_LANES,
                    "macro phase has {} lanes, one warp is the cap",
                    ms.num_lanes());
            script.push(((ms.after_stage as u32) << 16) | (ms.num_lanes() as u32));
        }
        assert!(script.len() <= 128,
                "{} macro phases overflow the metadata block",
                part.macro_stages.len());
        // padding
        while script.len() < 128 {
            script.push(0);
        }
        // final 128: write-out locations
        // compressed 2-1
        let mut last_wo = u32::MAX;
        for (j, bs) in part.stages.iter().enumerate() {
            for &wo in &bs.write_outs {
                let cur_wo = (j as u32) << 8 | (wo as u32);
                if last_wo == u32::MAX {
                    last_wo = cur_wo;
                }
                else {
                    script.push(last_wo | (cur_wo << 16));
                    last_wo = u32::MAX;
                }
            }
        }
        if last_wo != u32::MAX {
            script.push(last_wo | (((1 << 16) - 1) << 16));
        }
        while script.len() < 256 {
            script.push(u32::MAX);
        }
        // Is this pin produced by a mid-partition macro phase rather than
        // loaded from global memory? A DSP `P` or SRL `Q31` is a cycle-start
        // state read and DOES come from the global read like a DFF Q; only
        // combinational macro outputs (CARRY4 O/CO, SRL Q) are supplied by a
        // phase and must be skipped here.
        let is_phase_output = |pin: usize| -> bool {
            match aig.aigpin2macro.get(&pin) {
                Some(&mi) => {
                    let m = &aig.macros[mi];
                    m.outputs.iter().position(|&p| p == pin)
                        .map(|slot| m.kind.output_needs_mid_partition_eval(slot))
                        .unwrap_or(false)
                },
                None => false,
            }
        };

        // read global (256x32)
        let mut inputs_taken = BTreeMap::<u32, u32>::new();
        for &inp in &part.stages[0].hier[0] {
            if inp == usize::MAX { continue }
            if is_phase_output(inp) { continue }
            match input_map.get(&inp) {
                Some(&pos) => {
                    *inputs_taken.entry(pos >> 5).or_default() |=
                        1 << (pos & 31);
                }
                None => {
                    match staged_io_map.get(&inp) {
                        Some(&pos) => {
                            *inputs_taken.entry((pos >> 5) | (1u32 << 31))
                                .or_default() |= 1 << (pos & 31);
                        }
                        None => {
                            panic!("cannot find input pin {}, driver: {:?}, in either primary inputs or staged IOs", inp, aig.drivers[inp]);
                        }
                    }
                }
            }
        }
        // clilog::debug!(
        //     "part (?) inputs_taken len {}: {:?}",
        //     inputs_taken.len(),
        //     inputs_taken.iter().map(|(id, val)| format!("{}[{}]", id, val.count_ones())).collect::<Vec<_>>()
        // );
        let rounds_idx_masks = map_global_read_to_rounds(
            &inputs_taken
        );
        let num_global_stages = rounds_idx_masks.iter()
            .map(|v| v.len()).max().unwrap() as u32;
        script[6] = num_global_stages;
        assert_eq!(script.len(), NUM_THREADS_V1);
        let global_perm_start = script.len();
        script.extend((0..(2 * num_global_stages as usize * NUM_THREADS_V1)).map(|_| 0));
        for (i, v) in rounds_idx_masks.iter().enumerate() {
            for (round, &(idx, mask)) in v.iter().enumerate() {
                script[global_perm_start + NUM_THREADS_V1 * 2 * round + (i * 2)] = idx;
                script[global_perm_start + NUM_THREADS_V1 * 2 * round + (i * 2 + 1)] = mask;
                // println!("test: round {} i {} idx {} mask {}",
                //          round, i, idx, mask);
            }
        }

        let outputpos2localpos = rounds_idx_masks.iter().enumerate().map(|(local_i, v)| {
            let mut local_op2lp = Vec::with_capacity(32);
            let mut bit_id = 0;
            for &(idx, mask) in v.iter().rev() {
                let is_staged_io = (idx >> 31) != 0;
                for k in (0..32).rev() {
                    if (mask >> k & 1) != 0 {
                        local_op2lp.push(((is_staged_io, idx << 5 | k), (local_i * 32 + bit_id) as u16));
                        bit_id += 1;
                    }
                }
            }
            assert!(bit_id <= 32);
            local_op2lp.into_iter()
        }).flatten().collect::<IndexMap<_, _>>();
        // println!("output2localpos: {:?}", outputpos2localpos);

        let mut last_pin2localpos = IndexMap::new();
        for &inp in &part.stages[0].hier[0] {
            if inp == usize::MAX { continue }
            if is_phase_output(inp) { continue }
            let pos = match input_map.get(&inp) {
                Some(&pos) => (false, pos),
                None => (true, *staged_io_map.get(&inp).unwrap())
            };
            last_pin2localpos.insert(inp, *outputpos2localpos.get(&pos).unwrap());
        }

        // Which shared-state bit positions currently hold a live value. The
        // macro output allocator claims free slots out of this; before the
        // first stage the live set is whatever the global read phase loaded.
        let mut occupancy = vec![false; 1 << BOOMERANG_NUM_STAGES];
        for (_, &pos) in &outputpos2localpos {
            occupancy[pos as usize] = true;
        }
        let mut ph_cursor = 0usize;

        // boomerang sections start
        for (bs_i, bs) in part.stages.iter().enumerate() {
            // A phase with after_stage == bs_i runs BEFORE stage bs_i, so it
            // is emitted here, while last_pin2localpos still describes the
            // previous stage's output.
            while ph_cursor < part.macro_stages.len() &&
                part.macro_stages[ph_cursor].after_stage == bs_i
            {
                self.emit_macro_phase(
                    &mut script, aig, &part.macro_stages[ph_cursor],
                    &mut last_pin2localpos, &mut occupancy, macro_state_off);
                ph_cursor += 1;
            }
            let bs_perm = bs.hier[0].iter().map(|&pin| {
                if pin == usize::MAX { 0 }
                else { match last_pin2localpos.get(&pin) {
                    Some(&pos) => pos,
                    None => {
                        let prod = aig.aigpin2macro.get(&pin).copied();
                        let sched: Vec<(usize, usize)> = part.macro_stages.iter()
                            .enumerate()
                            .filter_map(|(w, ms)| {
                                if ms.lanes.iter().any(|l| Some(l.macro_i) == prod) {
                                    Some((w, ms.after_stage))
                                } else { None }
                            })
                            .collect();
                        panic!(
                            "stage {} needs aigpin {} (driver {:?}); producing                              macro = {:?}; that macro appears in                              (wave, after_stage) = {:?}; all phases at {:?};                              total stages {}",
                            bs_i, pin, aig.drivers[pin], prod, sched,
                            part.macro_stages.iter()
                                .map(|m| m.after_stage).collect::<Vec<_>>(),
                            part.stages.len())
                    }
                } }
            }).collect::<Vec<_>>();

            let mut bs_xora = vec![0u32; NUM_THREADS_V1];
            let mut bs_xorb = vec![0u32; NUM_THREADS_V1];
            let mut bs_orb = vec![0u32; NUM_THREADS_V1];
            for hi in 1..bs.hier.len() {
                let hi_len = bs.hier[hi].len();
                for j in 0..hi_len {
                    let out = bs.hier[hi][j];
                    let a = bs.hier[hi - 1][j];
                    let b = bs.hier[hi - 1][j + hi_len];
                    if out == usize::MAX {
                        continue
                    }
                    if out == a {
                        bs_orb[(hi_len + j) >> 5] |= 1 << ((hi_len + j) & 31);
                        continue
                    }
                    let (a_iv, b_iv) = match aig.drivers[out] {
                        DriverType::AndGate(a_iv, b_iv) => (a_iv, b_iv),
                        _ => unreachable!()
                    };
                    assert_eq!(a_iv >> 1, a);
                    assert_eq!(b_iv >> 1, b);
                    if (a_iv & 1) != 0 {
                        bs_xora[(hi_len + j) >> 5] |= 1 << ((hi_len + j) & 31);
                    }
                    if (b_iv & 1) != 0 {
                        bs_xorb[(hi_len + j) >> 5] |= 1 << ((hi_len + j) & 31);
                    }
                }
            }

            for k in 0..4 {
                for i in ((k * 8)..bs_perm.len()).step_by(32) {
                    script.push(((bs_perm[i] as u32)) |
                                (bs_perm[i + 1] as u32) << 16);
                    script.push(((bs_perm[i + 2] as u32)) |
                                (bs_perm[i + 3] as u32) << 16);
                    script.push(((bs_perm[i + 4] as u32)) |
                                (bs_perm[i + 5] as u32) << 16);
                    script.push(((bs_perm[i + 6] as u32)) |
                                (bs_perm[i + 7] as u32) << 16);
                }
            }
            for i in 0..NUM_THREADS_V1 {
                script.push(bs_xora[i]);
                script.push(bs_xorb[i]);
                script.push(bs_orb[i]);
                script.push(0);
            }

            last_pin2localpos = self.afters[bs_i].iter().enumerate().filter_map(|(i, &pin)| {
                if pin == usize::MAX { None }
                else { Some((pin, i as u16)) }
            }).collect::<IndexMap<_, _>>();
            // The tree rewrites every bit of the hier[1] range each stage, so
            // the previous phase's macro outputs are gone by now; occupancy is
            // rebuilt from the new after array rather than accumulated.
            occupancy = vec![false; 1 << BOOMERANG_NUM_STAGES];
            for (i, &pin) in self.afters[bs_i].iter().enumerate() {
                if pin != usize::MAX { occupancy[i] = true; }
            }
        }

        // Phases scheduled after the final boomerang stage.
        while ph_cursor < part.macro_stages.len() {
            self.emit_macro_phase(
                &mut script, aig, &part.macro_stages[ph_cursor],
                &mut last_pin2localpos, &mut occupancy, macro_state_off);
            ph_cursor += 1;
        }

        // sram worker
        for k in 0..4 {
            for i in ((k * 8)..self.sram_duplicate_permute.len()).step_by(32) {
                script.push(((self.sram_duplicate_permute[i] as u32)) |
                            (self.sram_duplicate_permute[i + 1] as u32) << 16);
                script.push(((self.sram_duplicate_permute[i + 2] as u32)) |
                            (self.sram_duplicate_permute[i + 3] as u32) << 16);
                script.push(((self.sram_duplicate_permute[i + 4] as u32)) |
                            (self.sram_duplicate_permute[i + 5] as u32) << 16);
                script.push(((self.sram_duplicate_permute[i + 6] as u32)) |
                            (self.sram_duplicate_permute[i + 7] as u32) << 16);
            }
        }
        for i in 0..NUM_THREADS_V1 {
            script.push(self.sram_duplicate_inv[i]);
            script.push(self.sram_duplicate_set0[i]);
            script.push(0);
            script.push(0);
        }
        // clock enable signal
        for k in 0..4 {
            for i in ((k * 8)..self.clken_permute.len()).step_by(32) {
                script.push(((self.clken_permute[i] as u32)) |
                            (self.clken_permute[i + 1] as u32) << 16);
                script.push(((self.clken_permute[i + 2] as u32)) |
                            (self.clken_permute[i + 3] as u32) << 16);
                script.push(((self.clken_permute[i + 4] as u32)) |
                            (self.clken_permute[i + 5] as u32) << 16);
                script.push(((self.clken_permute[i + 6] as u32)) |
                            (self.clken_permute[i + 7] as u32) << 16);
            }
        }
        for i in 0..NUM_THREADS_V1 {
            script.push(self.clken_inv[i]);
            script.push(self.clken_set0[i]);
            script.push(self.data_inv[i]);
            script.push(0);
        }

        script
    }
}

fn build_flattened_script_v1(
    aig: &AIG, stageds: &[&StagedAIG],
    parts_in_stages: &[&[Partition]],
    num_blocks: usize,
    input_layout: Vec<usize>
) -> FlattenedScriptV1 {
    // determine the output position.
    // this is the prerequisite for generating the read
    // permutations and more.
    // input map:
    // locate input pins and FF/SRAM Q's - for partition input
    // output map:
    // locate primary outputs - for circuit outs
    // staged io map:
    // store intermediate nodes between major stages
    let mut input_map = IndexMap::new();
    let mut output_map = IndexMap::new();
    let mut staged_io_map = IndexMap::new();
    for (i, &input) in input_layout.iter().enumerate() {
        if input == usize::MAX { continue }
        input_map.insert(input, i as u32);
    }

    let num_major_stages = parts_in_stages.len();

    let states_start = ((input_layout.len() + 31) / 32) as u32;
    // Reserve one word that no partition ever writes. Zero-initialised and
    // never committed to, it reads 0 forever, which is exactly what a
    // constant-zero-D flip-flop's Q needs.
    let zero_word = states_start;
    let mut sum_state_start = states_start + 1;
    let mut sum_srams_start = 0;
    let mut sum_macro_state = 0;

    // enumerate all major stages and build them one by one.

    // #[derive(Debug, Clone, Default)]
    // struct FlatteningStage {
    //     blocks_parts: Vec<Vec<usize>>,
    //     flattening_parts: Vec<FlatteningPart>,
    //     parts_data_split: Vec<Vec<u32>>,
    // }
    // let mut flattening_stages =
    //     Vec::<FlatteningStage>::with_capacity(num_major_stages);

    // assemble script per block.
    let mut blocks_data = Vec::new();
    let mut blocks_start = Vec::<usize>::with_capacity(num_blocks * num_major_stages + 1);
    let mut stages_blocks_parts = Vec::new();
    let mut stages_flattening_parts = Vec::new();

    for (i, (init_parts, &staged)) in parts_in_stages.into_iter().copied().zip(
        stageds.into_iter()
    ).enumerate() {
        // first arrange parts onto blocks.
        let mut blocks_parts = vec![vec![]; num_blocks];
        let mut tot_nstages_blocks = vec![0; num_blocks];
        // below models the fixed pre&post-cost for each executor
        let executor_fixed_cost = 3;
        // masonry layout of blocks. assume parts are sorted with
        // decreasing order of #stages.
        for i in 0..init_parts.len().min(num_blocks) {
            blocks_parts[i].push(i);
            tot_nstages_blocks[i] = init_parts[i].stages.len() + executor_fixed_cost;
        }
        for i in num_blocks..init_parts.len() {
            let put = tot_nstages_blocks.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.cmp(b))
                .unwrap().0;
            blocks_parts[put].push(i);
            tot_nstages_blocks[put] += init_parts[i].stages.len() + executor_fixed_cost;
        }
        // clilog::debug!("blocks_parts: {:?}", blocks_parts);
        clilog::debug!("major stage {}: max total boomerang depth (w/ cost) {}",
                       i, tot_nstages_blocks.iter().copied().max().unwrap());

        // the intermediates for parts being flattened
        let mut flattening_parts: Vec<FlatteningPart> =
            vec![Default::default(); init_parts.len()];

        // basic index preprocessing for stages
        for i in 0..init_parts.len() {
            flattening_parts[i].init_afters_writeouts(
                aig, staged, &init_parts[i]);
        }

        // allocate output state positions for all srams,
        // in the order of block affinity.
        // Allocation walks blocks_parts, not part ids, so every array is laid
        // out in block-affinity order. This is what makes a warp's accesses
        // contiguous; allocating in part-id order would scatter them.
        for block in &blocks_parts {
            for &part_id in block {
                flattening_parts[part_id].zero_bit_pos = zero_word * 32;
                flattening_parts[part_id].state_start = sum_state_start;
                sum_state_start += flattening_parts[part_id].num_writeouts;
                flattening_parts[part_id].sram_start = sum_srams_start;
                sum_srams_start += flattening_parts[part_id].num_srams * (1 << AIGPDK_SRAM_ADDR_WIDTH);
                flattening_parts[part_id].macro_state_start = sum_macro_state;
                sum_macro_state += flattening_parts[part_id].macro_state_words;
            }
        }

        // besides input ports, we also have outputs from partitions.
        // they include original-placed comb output pins,
        // copied pins for different FF activation,
        // and SRAM read outputs.
        for part_id in 0..init_parts.len() {
            // clilog::debug!("initializing output for part {}", part_id);
            flattening_parts[part_id].make_inputs_outputs(
                aig, staged, &init_parts[part_id],
                &mut input_map, &mut staged_io_map, &mut output_map
            );
        }
        stages_blocks_parts.push(blocks_parts);
        stages_flattening_parts.push(flattening_parts);
    }

    // Global macro-state allocation. An SRLC32E's shift register has no AIG
    // pin representation, so it lives only here; and its read port may be
    // scheduled into a different partition than its state commit, which is why
    // offsets are global rather than partition-relative. Allocation walks
    // block affinity first so a block's macros stay contiguous.
    // Macro state is appended to the SRAM buffer rather than living in its
    // own allocation, so the CUDA kernel needs no extra parameter and the FFI
    // signature is untouched. Offsets are therefore absolute within that
    // combined buffer from the start.
    let mut macro_state_off = IndexMap::<usize, u32>::new();
    let macro_state_origin = sum_srams_start;
    let mut macro_state_cursor = macro_state_origin;
    for (blocks_parts, init_parts) in stages_blocks_parts.iter().zip(
        parts_in_stages.into_iter().copied()
    ) {
        for block in blocks_parts {
            for &part_id in block {
                for &macro_i in &init_parts[part_id].state_macros {
                    if !macro_state_off.contains_key(&macro_i) {
                        macro_state_off.insert(macro_i, macro_state_cursor);
                        macro_state_cursor += aig.macros[macro_i].kind.state_words() as u32;
                    }
                }
            }
        }
    }
    for (macro_i, m) in aig.macros.values().enumerate() {
        if m.kind.has_state() && !macro_state_off.contains_key(&macro_i) {
            macro_state_off.insert(macro_i, macro_state_cursor);
            macro_state_cursor += m.kind.state_words() as u32;
        }
    }

    for ((blocks_parts, flattening_parts), init_parts) in stages_blocks_parts.iter().zip(
        stages_flattening_parts.iter_mut()
    ).zip(
        parts_in_stages.into_iter().copied()
    ) {
        // build script per part. we will later assemble them to blocks.
        let mut parts_data_split = vec![vec![]; init_parts.len()];
        for part_id in 0..init_parts.len() {
            // clilog::debug!("building script for part {}", part_id);
            parts_data_split[part_id] = flattening_parts[part_id].build_script(
                aig, &init_parts[part_id], &input_map, &staged_io_map,
                &macro_state_off
            );
        }

        for block_id in 0..num_blocks {
            blocks_start.push(blocks_data.len());
            if blocks_parts[block_id].is_empty() {
                let mut dummy = vec![0; NUM_THREADS_V1];
                dummy[1] = 1;
                blocks_data.extend(dummy.into_iter());
            }
            else {
                let num_parts = blocks_parts[block_id].len();
                let mut last_part_st = usize::MAX;
                for (i, &part_id) in blocks_parts[block_id].iter().enumerate() {
                    if i == num_parts - 1 {
                        last_part_st = blocks_data.len();
                    }
                    blocks_data.extend(parts_data_split[part_id].iter().copied());
                }
                assert_ne!(last_part_st, usize::MAX);
                blocks_data[last_part_st + 1] = 1;
            }
        }
    }
    blocks_start.push(blocks_data.len());
    blocks_data.extend((0..NUM_THREADS_V1 * 8).map(|_| 0)); // padding

    clilog::info!("Built script for {} blocks, reg/io state size {}, sram size {}, script size {}",
                  num_blocks, sum_state_start, sum_srams_start, blocks_data.len());

    FlattenedScriptV1 {
        num_blocks,
        num_major_stages,
        blocks_start: blocks_start.into(),
        blocks_data: blocks_data.into(),
        reg_io_state_size: sum_state_start,
        sram_storage_size: sum_srams_start,
        macro_storage_size: macro_state_cursor - macro_state_origin,
        input_layout,
        input_map,
        output_map,
        stages_blocks_parts,
    }
}

impl FlattenedScriptV1 {
    /// build a flattened script.
    ///
    /// `init_parts` give the partitions to flatten.
    /// it is better sorted in advance in descending order
    /// of #layers for better duty cycling.
    ///
    /// `num_blocks` should be set to the hardware allowances,
    /// i.e. the number of SMs in your GPU.
    /// for example, A100 should set it to 108.
    ///
    /// `input_layout` should give the expected primary input
    /// memory layout, each one is an AIG bit index.
    /// padding bits should be set to usize::MAX.
    pub fn from(
        aig: &AIG, stageds: &[&StagedAIG],
        parts_in_stages: &[&[Partition]],
        num_blocks: usize,
        input_layout: Vec<usize>
    ) -> FlattenedScriptV1 {
        build_flattened_script_v1(
            aig, stageds, parts_in_stages, num_blocks, input_layout)
    }
}
