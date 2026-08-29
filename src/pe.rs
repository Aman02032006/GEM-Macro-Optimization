// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Partition executor

use crate::aig::{DriverType, AIG, EndpointGroup};
use crate::macros::MacroKind;
use crate::staging::StagedAIG;
use indexmap::{IndexMap, IndexSet};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use rayon::prelude::*;

/// The number of boomerang stages.
///
/// This determines the shuffle width, i.e., kernel width.
/// `kernel width = (1 << BOOMERANG_NUM_STAGES)`.
pub const BOOMERANG_NUM_STAGES: usize = 13;

const BOOMERANG_MAX_WRITEOUTS: usize = 1 << (BOOMERANG_NUM_STAGES - 5);

/// One Boomerang stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoomerangStage {
    /// the boomerang hierarchy, 8192 -> 4096 -> ... -> 1.
    ///
    /// each element is an aigpin index (without iv).
    /// its parent indices should either be a passthrough or an
    /// and gate mapping.
    pub hier: Vec<Vec<usize>>,
    /// the 32-packed elements in the hierarchy where there should be
    /// a pass-through.
    pub write_outs: Vec<usize>,
}

/// One lane of a macro phase: a single macro instance assigned to one GPU
/// lane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroLane {
    /// position within [crate::aig::AIG::macros].
    pub macro_i: usize,
    /// Is this lane the head of a carry chain?
    ///
    /// A chain head takes its carry-in from ordinary logic (`CYINIT`, or a
    /// `CI` driven by something that is not another CARRY4). Every following
    /// lane takes its carry from the warp scan instead, so its `CI` pin is not
    /// gathered from `shared_state` at all.
    pub chain_start: bool,
}

/// One mid-partition macro evaluation phase.
///
/// Sits between two boomerang stages: every macro listed here has all of its
/// combinational inputs realised by boomerang stage `after_stage - 1`, and its
/// outputs become realised inputs for stage `after_stage` onwards.
///
/// Only macros with combinationally-driven outputs appear here. A DSP48E2 has
/// none -- its `P` port is a read of `PREG`, loaded during the global read
/// phase like a DFF `Q` -- so DSPs are scheduled purely as endpoints.
///
/// # Why chains, not depth levels
///
/// An earlier revision grouped macros into depth waves, which is the natural
/// dependency order but is badly wrong for carry chains: a 32-bit adder is 8
/// CARRY4s at 8 successive depths, so it produced 8 phases of one macro each.
/// Since every script section costs global bandwidth re-read every cycle, that
/// alone could make the "optimised" simulator slower than stock GEM.
///
/// A whole chain now occupies consecutive lanes of ONE phase and the carry is
/// propagated with a Kogge-Stone `__shfl_up_sync` scan, so the 8-link adder is
/// a single phase. Phases are ordered by the depth of their *external* inputs,
/// which is what actually constrains them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroStage {
    /// index into [Partition::stages]: this phase runs immediately before
    /// that boomerang stage. Equal to `stages.len()` if it runs last.
    pub after_stage: usize,
    /// lanes, laid out so that each carry chain is contiguous and ascending.
    pub lanes: Vec<MacroLane>,
}

impl MacroStage {
    pub fn num_lanes(&self) -> usize { self.lanes.len() }
}

/// One partitioned block: a basic execution unit on GPU.
///
/// A block is mapped to a GPU block with the following resource
/// constraints:
/// 1. the number of unique inputs should not exceed 8191.
/// 2. the number of unique outputs should not exceed 8191.
///    for srams and dffs, outputs include all enable pins and bus pins.
///    there might be unusable holes but the effective capacity is at least
///    4095.
/// 3. the number of intermediate pins alive at each stage should not
 ///    exceed 4095.
/// 4. the number of SRAM output groups should not exceed 64.
///    64 = 8192 / (32 * 4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition {
    /// the endpoints that are realized by this partition.
    pub endpoints: Vec<usize>,
    /// the boomerang stages.
    ///
    /// between stages there will automatically be shuffles.
    pub stages: Vec<BoomerangStage>,
    /// mid-partition macro evaluation phases, ordered by `after_stage`.
    ///
    /// Kept as a parallel list rather than folding into `stages` so that every
    /// existing consumer of `stages` (flatten.rs, flatten_test.rs, the
    /// merge heuristics below) keeps working unchanged on macro-free designs.
    #[serde(default)]
    pub macro_stages: Vec<MacroStage>,
    /// positions within [crate::aig::AIG::macros] of the stateful macros this
    /// partition commits state for, in endpoint order.
    #[serde(default)]
    pub state_macros: Vec<usize>,
}

/// build a single boomerang stage given the current inputs and
/// outputs.
fn build_one_boomerang_stage(
    aig: &AIG,
    unrealized_comb_outputs: &mut IndexSet<usize>,
    realized_inputs: &mut IndexSet<usize>,
    total_write_outs: &mut usize,
    num_reserved_writeouts: usize,
) -> Option<BoomerangStage> {
    let mut hier = Vec::new();
    for i in 0..=BOOMERANG_NUM_STAGES {
        hier.push(vec![usize::MAX; 1 << (BOOMERANG_NUM_STAGES - i)]);
    }

    // first discover the (remaining) subgraph to implement.
    let order = aig.topo_traverse_generic(
        Some(
            &unrealized_comb_outputs.iter().copied().collect()
        ),
        Some(&realized_inputs)
    );
    let id2order: IndexMap<_, _> = order.iter().copied().enumerate()
        .map(|(order_i, i)| (i, order_i))
        .collect();
    let mut level = vec![0; order.len()];
    for (order_i, i) in order.iter().copied().enumerate() {
        if realized_inputs.contains(&i) { continue }
        let mut lvli: usize = 0;
        if let DriverType::AndGate(a, b) = aig.drivers[i] {
            if a >= 2 {
                lvli = lvli.max(level[*id2order.get(&(a >> 1)).unwrap()] + 1);
            }
            if b >= 2 {
                lvli = lvli.max(level[*id2order.get(&(b >> 1)).unwrap()] + 1);
            }
        }
        level[order_i] = lvli;
    }
    let max_level = level.iter().copied().max().unwrap();
    clilog::trace!("boomerang current max level: {}", max_level);

    fn place_bit(
        aig: &AIG,
        hier: &mut Vec<Vec<usize>>,
        hier_visited_nodes_count: &mut IndexMap<usize, usize>,
        level: &Vec<usize>,
        id2order: &IndexMap<usize, usize>,
        hi: usize, j: usize, nd: usize
    ) {
        hier[hi][j] = nd;
        if hi == 0 { return }
        *hier_visited_nodes_count.entry(nd).or_default() += 1;
        let lvlnd = level[*id2order.get(&nd).unwrap()];
        assert!(lvlnd <= hi);
        if lvlnd != hi {
            place_bit(aig, hier, hier_visited_nodes_count,
                      level, id2order,
                      hi - 1, j, nd);
        }
        else {
            let (a, b) = match aig.drivers[nd] {
                DriverType::AndGate(a, b) => (a, b),
                _ => panic!()
            };
            let hier_hi_len = hier[hi].len();
            place_bit(aig, hier, hier_visited_nodes_count,
                      level, id2order,
                      hi - 1, j, a >> 1);
            place_bit(aig, hier, hier_visited_nodes_count,
                      level, id2order,
                      hi - 1, j + hier_hi_len, b >> 1);
        }
    }

    fn purge_bit(
        aig: &AIG,
        hier: &mut Vec<Vec<usize>>,
        hier_visited_nodes_count: &mut IndexMap<usize, usize>,
        level: &Vec<usize>,
        id2order: &IndexMap<usize, usize>,
        hi: usize, j: usize
    ) {
        if hier[hi][j] == usize::MAX { return }
        let nd = hier[hi][j];
        hier[hi][j] = usize::MAX;
        if hi == 0 { return }
        let hvc = hier_visited_nodes_count.get_mut(&nd).unwrap();
        *hvc -= 1;
        if *hvc == 0 {
            hier_visited_nodes_count.swap_remove(&nd);
        }
        let hier_hi_len = hier[hi].len();
        purge_bit(aig, hier, hier_visited_nodes_count,
                  level, id2order,
                  hi - 1, j);
        purge_bit(aig, hier, hier_visited_nodes_count,
                  level, id2order,
                  hi - 1, j + hier_hi_len);
    }

    // the nodes that are implemented in the hierarchy.
    // we only count for hierarchy[1 and more], [0] is not counted.
    let mut hier_visited_nodes_count: IndexMap<usize, usize> = IndexMap::new();
    let mut selected_level = max_level.min(BOOMERANG_NUM_STAGES);

    /// compute the maximum number of steps needed from this node
    /// to reach an endpoint node.
    ///
    /// during this path, except the starting point, no node should
    /// already be inside the boomerang hierarchy.
    fn compute_reverse_level(
        order: &Vec<usize>,
        id2order: &IndexMap<usize, usize>,
        unrealized_comb_outputs: &IndexSet<usize>,
        realized_inputs: &IndexSet<usize>,
        hier_visited_nodes_count: &IndexMap<usize, usize>,
        aig: &AIG
    ) -> Vec<usize> {
        let mut reverse_level = vec![usize::MAX; order.len()];
        for &i in unrealized_comb_outputs.iter() {
            reverse_level[*id2order.get(&i).unwrap()] = 0;
        }
        for (order_i, i) in order.iter().copied().enumerate().rev() {
            if realized_inputs.contains(&i) ||
                hier_visited_nodes_count.contains_key(&i)
            {
                continue
            }
            let rlvli = reverse_level[order_i];
            if let DriverType::AndGate(a, b) = aig.drivers[i] {
                if a >= 2 {
                    let a = *id2order.get(&(a >> 1)).unwrap();
                    let rlvla = &mut reverse_level[a];
                    if *rlvla == usize::MAX || *rlvla < rlvli + 1 {
                        *rlvla = rlvli + 1;
                    }
                }
                if b >= 2 {
                    let b = *id2order.get(&(b >> 1)).unwrap();
                    let rlvlb = &mut reverse_level[b];
                    if *rlvlb == usize::MAX || *rlvlb < rlvli + 1 {
                        *rlvlb = rlvli + 1;
                    }
                }
            }
        }
        reverse_level
    }

    /// compute the set of nodes that must be implemented in level 1
    /// in addition to the current hierarchy.
    ///
    /// the necessary_level1 nodes can only come from level 0 or
    /// level 1.
    /// a level 1 node is necessary if it is not already
    /// implemented, and it still drives a downstream endpoint.
    /// a level 0 node is necessary if it is not already implemented,
    /// and it either (1) is needed by a level>=2 node, or (2) is
    /// itself an unrealized endpoint.
    fn compute_lvl1_necessary_nodes(
        order: &Vec<usize>,
        id2order: &IndexMap<usize, usize>,
        level: &Vec<usize>,
        reverse_level: &Vec<usize>,
        aig: &AIG,
        unrealized_comb_outputs: &IndexSet<usize>,
        hier_visited_nodes_count: &IndexMap<usize, usize>,
    ) -> IndexSet<usize> {
        let mut lvl1_necessary_nodes = IndexSet::new();
        for order_i in 0..order.len() {
            if hier_visited_nodes_count.contains_key(&order[order_i]) {
                continue
            }
            if reverse_level[order_i] == usize::MAX { continue }
            if level[order_i] == 0 {
                if unrealized_comb_outputs.contains(&order[order_i]) {
                    lvl1_necessary_nodes.insert(order[order_i]);
                }
                continue
            }
            if level[order_i] == 1 {
                lvl1_necessary_nodes.insert(order[order_i]);
            }
            else {
                let (a, b) = match &aig.drivers[order[order_i]] {
                    DriverType::AndGate(a, b) => (*a, *b),
                    // A macro output can never be placed in the boomerang
                    // hierarchy: the tree evaluates a fixed 2-input boolean op
                    // per node, and a CARRY4 is 10-in/8-out. Reaching here
                    // means the macro was not seeded as a realized input by
                    // Partition::build_one, so say so rather than panicking
                    // with no context.
                    DriverType::Macro(cellid, slot) => panic!(
                        "macro cell {} output slot {} reached boomerang level \
                         assignment at level {}. Mid-partition macros must be \
                         evaluated in a MacroStage and their outputs seeded \
                         into realized_inputs before any stage consumes them.",
                        cellid, slot, level[order_i]
                    ),
                    d @ _ => panic!(
                        "unexpected driver {:?} at boomerang level {}",
                        d, level[order_i]
                    )
                };
                if a >= 2 &&
                    level[*id2order.get(&(a >> 1)).unwrap()] == 0 &&
                    !hier_visited_nodes_count.contains_key(&(a >> 1))
                {
                    lvl1_necessary_nodes.insert(a >> 1);
                }
                if b >= 2 &&
                    level[*id2order.get(&(b >> 1)).unwrap()] == 0 &&
                    !hier_visited_nodes_count.contains_key(&(b >> 1))
                {
                    lvl1_necessary_nodes.insert(b >> 1);
                }
            }
        }
        lvl1_necessary_nodes
    }

    let mut reverse_level = compute_reverse_level(
        &order, &id2order,
        unrealized_comb_outputs, realized_inputs,
        &hier_visited_nodes_count, aig
    );

    let mut last_lvl1_necessary_nodes = IndexSet::new();

    while selected_level >= 2 {
        // find a valid slot to place a high level bit
        let mut slot_at_level = usize::MAX;
        for i in 0..hier[selected_level].len() {
            if hier[selected_level][i] == usize::MAX {
                slot_at_level = i;
                break
            }
        }
        if slot_at_level == usize::MAX {
            clilog::trace!("no space at level {}", selected_level);
            selected_level -= 1;
            continue
        }

        // find a valuable node to put into the above slot
        let mut selected_node_ord = usize::MAX;
        for order_i in 0..order.len() {
            if level[order_i] != selected_level { continue }
            if hier_visited_nodes_count.contains_key(&order[order_i]) || reverse_level[order_i] == usize::MAX {
                continue
            }
            if selected_node_ord == usize::MAX ||
                reverse_level[selected_node_ord] < reverse_level[order_i]
            {
                selected_node_ord = order_i;
            }
        }
        if selected_node_ord == usize::MAX {
            clilog::trace!("no node at level {}", selected_level);
            selected_level -= 1;
            continue
        }
        let selected_node = order[selected_node_ord];

        place_bit(
            aig, &mut hier, &mut hier_visited_nodes_count,
            &level, &id2order,
            selected_level, slot_at_level, selected_node
        );

        let reverse_level_upd = compute_reverse_level(
            &order, &id2order,
            unrealized_comb_outputs, realized_inputs,
            &hier_visited_nodes_count, aig
        );

        // store the nodes that need to be put on the 1-level
        // (simple ands).
        // they are periodically checked to ensure they have space.
        let lvl1_necessary_nodes = compute_lvl1_necessary_nodes(
            &order, &id2order, &level,
            &reverse_level_upd, aig, &unrealized_comb_outputs,
            &hier_visited_nodes_count
        );

        let num_lvl1_hier_taken =
            hier[1].iter().filter(|i| **i != usize::MAX).count();

        clilog::trace!(
            "taken one node at level {}, used 1-level space {}, hier visited unique {}, num nodes necessary in lvl1 {}",
            selected_level, num_lvl1_hier_taken,
            hier_visited_nodes_count.len(), lvl1_necessary_nodes.len()
        );

        if lvl1_necessary_nodes.len() +
            num_lvl1_hier_taken.max(hier_visited_nodes_count.len())
            >= (1 << (BOOMERANG_NUM_STAGES - 1))
        {
            clilog::trace!("REVERSED the plan due to overflow");
            purge_bit(
                aig, &mut hier, &mut hier_visited_nodes_count,
                &level, &id2order,
                selected_level, slot_at_level
            );
            selected_level -= 1;
            continue
        }

        reverse_level = reverse_level_upd;
        last_lvl1_necessary_nodes = lvl1_necessary_nodes;
    }

    if last_lvl1_necessary_nodes.is_empty() {
        last_lvl1_necessary_nodes = compute_lvl1_necessary_nodes(
            &order, &id2order, &level,
            &reverse_level, aig, &unrealized_comb_outputs,
            &hier_visited_nodes_count
        );
    }

    // the hierarchy is now constructed except all 1-level nodes.
    // it's time to place them. during this process, we heuristically collect
    // endpoint nodes into consecutive space for early write-out.
    //
    // we first try to finalize all endpoints that have to appear in
    // level 1.
    // after that, we will try if we can write out all others scattered.
    let mut endpoints_lvl1 = Vec::new();
    let mut endpoints_untouched = Vec::new();
    let mut endpoints_hier = IndexSet::new();
    for &endpt in unrealized_comb_outputs.iter() {
        if hier_visited_nodes_count.contains_key(&endpt) {
            endpoints_hier.insert(endpt);
        }
        else if last_lvl1_necessary_nodes.contains(&endpt) {
            endpoints_lvl1.push(endpt);
        }
        else {
            endpoints_untouched.push(endpt);
        }
    }

    // collect all 32-consecutive level 1 spaces.
    // (num occupied, i), will be sorted later.
    let mut spaces = Vec::new();
    for i in 0..hier[1].len() / 32 {
        let mut num_occupied = 0u8;
        for j in i * 32..(i + 1) * 32 {
            if hier[1][j] != usize::MAX {
                num_occupied += 1;
            }
        }
        if num_occupied < 10 {
            spaces.push((num_occupied, i * 32))
        }
    }
    spaces.sort();
    let mut spaces_j = 0;
    let mut endpt_lvl1_i = 0;
    let mut realized_endpoints = IndexSet::new();
    let mut write_outs = Vec::new();
    // heuristically push level 1 endpoints.
    while spaces_j < spaces.len() &&
        (endpoints_untouched.is_empty() || // if we can try all
         endpoints_lvl1.len() - endpt_lvl1_i >= (32 - spaces[spaces_j].0) as usize)
    {
        let i = spaces[spaces_j].1;

        // Only spend a write-out on a group that actually realizes something.
        //
        // The loop condition takes the `endpoints_untouched.is_empty()` branch
        // whenever every endpoint is level-0 or level-1, and then walks EVERY
        // 32-slot group in hier[1] -- about 128 of them. Without this check it
        // pushes a write-out per group even after the last endpoint has been
        // placed, so two stages burn ~256 write-outs on empty space and the
        // partition is rejected at the cap no matter how small the design is.
        //
        // Macro designs hit this constantly: a macro output is a level-0 leaf,
        // so nothing is ever "untouched" and the greedy branch is always taken.
        // A group is worth a write-out if it can receive a pending level-1
        // endpoint, or if it already holds one that is still unrealized.
        {
            let mut free_slots = 0usize;
            let mut holds_unrealized = false;
            for j in i..i + 32 {
                if hier[1][j] == usize::MAX {
                    free_slots += 1;
                }
                else if unrealized_comb_outputs.contains(&hier[1][j])
                    && !realized_endpoints.contains(&hier[1][j])
                {
                    holds_unrealized = true;
                }
            }
            // The placement loop below bails on its first iteration once
            // endpoints_lvl1 is exhausted, so with nothing pending a group can
            // neither receive nor mark anything -- the leftovers are picked up
            // by the endpoints_hier sweep further down instead.
            let pending = endpt_lvl1_i < endpoints_lvl1.len();
            if !pending || (free_slots == 0 && !holds_unrealized) {
                spaces_j += 1;
                continue
            }
        }

        for j in i..i + 32 {
            if endpt_lvl1_i >= endpoints_lvl1.len() { break }
            if hier[1][j] == usize::MAX {
                let endpt_i = endpoints_lvl1[endpt_lvl1_i];
                place_bit(
                    aig, &mut hier, &mut hier_visited_nodes_count,
                    &level, &id2order,
                    1, j, endpt_i
                );
                realized_endpoints.insert(endpt_i);
                endpt_lvl1_i += 1;
            }
            else if unrealized_comb_outputs.contains(&hier[1][j]) {
                realized_endpoints.insert(hier[1][j]);
            }
        }
        *total_write_outs += 1;
        write_outs.push((i + hier[1].len()) / 32);
        spaces_j += 1;
    }

    if *total_write_outs > BOOMERANG_MAX_WRITEOUTS - num_reserved_writeouts {
        clilog::warn!(
            PART_REJECT_WO_COUNT,
            "partition rejected: boomerang write-outs exhausted              ({} used, {} reserved, cap {})",
            *total_write_outs, num_reserved_writeouts, BOOMERANG_MAX_WRITEOUTS);
        return None
    }

    // then place all remaining lvl1 nodes in any order.
    // clilog::debug!("last_lvl1_necessary: {}, hier visited: {}, realized endpts: {}", last_lvl1_necessary_nodes.len(), hier_visited_nodes_count.len(), realized_endpoints.len());
    let mut hier1_j = 0;
    for &nd in &last_lvl1_necessary_nodes {
        if hier_visited_nodes_count.contains_key(&nd) ||
            realized_endpoints.contains(&nd)
        {
            continue
        }
        while hier[1][hier1_j] != usize::MAX {
            hier1_j += 1;
            if hier1_j >= hier[1].len() {
                clilog::warn!(
                    PART_REJECT_LVL1,
                    "partition rejected: boomerang level-1 row full ({} slots)                      while placing necessary nodes -- the logic cone is wider                      than one partition can hold",
                    hier[1].len());
                return None
            }
        }
        place_bit(
            aig, &mut hier, &mut hier_visited_nodes_count,
            &level, &id2order,
            1, hier1_j, nd
        );
    }
    while hier[1][hier1_j] != usize::MAX {
        hier1_j += 1;
        if hier1_j >= hier[1].len() {
            clilog::warn!(
                PART_REJECT_LVL1_ZERO,
                "partition rejected: boomerang level-1 row full ({} slots),                  no free slot even for a constant pin",
                hier[1].len());
            return None
        }
    }

    // check if we can make this the last stage.
    if endpoints_untouched.is_empty() {
        let mut add_write_outs = IndexSet::new();
        for hi in 1..=BOOMERANG_NUM_STAGES {
            for j in 0..hier[hi].len() {
                let nd = hier[hi][j];
                if endpoints_hier.contains(&nd) && !realized_endpoints.contains(&nd) {
                    add_write_outs.insert((j + hier[hi].len()) / 32);
                    if add_write_outs.len() + *total_write_outs > BOOMERANG_MAX_WRITEOUTS - num_reserved_writeouts {
                        break
                    }
                }
            }
        }
        if add_write_outs.len() + *total_write_outs <= BOOMERANG_MAX_WRITEOUTS - num_reserved_writeouts {
            for wo in add_write_outs {
                write_outs.push(wo);
                *total_write_outs += 1;
            }
            for endpt in endpoints_hier {
                realized_endpoints.insert(endpt);
            }
        }
    }

    for (&i, _) in &hier_visited_nodes_count {
        realized_inputs.insert(i);
    }
    for &i in &realized_endpoints {
        assert!(unrealized_comb_outputs.swap_remove(&i));
    }

    Some(BoomerangStage {
        hier,
        write_outs
    })
}

/// Which targets cannot be built yet because their cone reads a macro output
/// whose phase has not run?
///
/// A boomerang stage works on the whole outstanding target set, not just the
/// pins the current wave needs. Without this filter an early stage happily
/// picks up an endpoint whose cone passes through a not-yet-evaluated macro
/// output, places it as a level-0 passthrough, and then asks flatten.rs for a
/// shared-state slot that no phase has written -- which is exactly how MIPS
/// failed with "stage 0 needs aigpin 622" while that macro's phase sat at
/// after_stage 2.
///
/// Taint is propagated forward over one topological order, so this costs a
/// single traversal per wave rather than a reachability query per target.
fn targets_blocked_by_pending(
    aig: &AIG,
    targets: &IndexSet<usize>,
    realized_inputs: &IndexSet<usize>,
    pending: &IndexSet<usize>,
) -> IndexSet<usize> {
    if pending.is_empty() { return IndexSet::new() }
    let order = aig.topo_traverse_generic(
        Some(&targets.iter().copied().collect()),
        Some(realized_inputs)
    );
    let mut taint = IndexSet::<usize>::new();
    for &pin in &order {
        // pending macro outputs sit in realized_inputs, so test them first
        if pending.contains(&pin) { taint.insert(pin); continue }
        if realized_inputs.contains(&pin) { continue }
        let mut t = false;
        match aig.drivers[pin] {
            DriverType::AndGate(a, b) => {
                if (a >> 1) != 0 && taint.contains(&(a >> 1)) { t = true }
                if (b >> 1) != 0 && taint.contains(&(b >> 1)) { t = true }
            }
            DriverType::Macro(cellid, slot) => {
                if let Some(m) = aig.macros.get(&cellid) {
                    m.for_each_comb_fanin(slot, |i| {
                        if taint.contains(&i) { t = true }
                    });
                }
            }
            _ => {}
        }
        if t { taint.insert(pin); }
    }
    targets.iter().copied().filter(|p| taint.contains(p)).collect()
}

/// The already-realized leaves that a deferred target's cone reads.
///
/// Withholding a target stops the boomerang from carrying its leaves forward,
/// so a primary input that only a deferred target needs loses its shared-state
/// slot -- MIPS failed as "stage 1 needs aigpin 6 (InputPort(3))". Re-arming
/// these as keep-live pins makes each stage place them as passthroughs, which
/// is the same mechanism that keeps macro inputs addressable.
fn realized_leaves_of(
    aig: &AIG,
    targets: &IndexSet<usize>,
    realized_inputs: &IndexSet<usize>,
) -> IndexSet<usize> {
    if targets.is_empty() { return IndexSet::new() }
    let order = aig.topo_traverse_generic(
        Some(&targets.iter().copied().collect()),
        Some(realized_inputs)
    );
    order.into_iter().filter(|p| realized_inputs.contains(p)).collect()
}

/// Is lane `l` of a phase the head of its carry chain?
///
/// True unless the immediately preceding lane is a CARRY4 whose `CO[3]` drives
/// this lane's `CI`. Chains are laid out contiguously and ascending by
/// `order_macro_waves`, so checking the previous lane is sufficient.
fn is_chain_start(aig: &AIG, wave: &[usize], l: usize) -> bool {
    if l == 0 { return true }
    // a warp scan never crosses a 32-lane boundary
    if l % 32 == 0 { return true }
    let me = wave[l];
    let prev = wave[l - 1];
    if me == usize::MAX || prev == usize::MAX { return true }
    let m = &aig.macros[me];
    let p = &aig.macros[prev];
    if !matches!(m.kind, MacroKind::Carry4) || !matches!(p.kind, MacroKind::Carry4) {
        return true
    }
    let ci_iv = m.inputs[crate::macros::C4_IN_CI];
    // Linked to the previous lane means this is NOT a chain head: its carry
    // arrives from the warp scan rather than from shared state.
    !((ci_iv >> 1) != 0 &&
      p.outputs[crate::macros::C4_OUT_CO + 3] == (ci_iv >> 1))
}

/// Drive boomerang stages until every pin in `targets` is realized.
///
/// The upstream loop (`while !unrealized.is_empty() { build_stage()? }`) spins
/// forever when a stage cannot make progress -- the failure mode a macro output
/// introduces, since `place_bit` represents only AND gates and can never
/// realize one. A no-progress round is detected here and fails the partition,
/// allowing `process_partitions` to fall back to a different clustering rather
/// than hanging the mapper.
fn run_boomerang_stages_until_realized(
    aig: &AIG,
    targets: &mut IndexSet<usize>,
    required: Option<&IndexSet<usize>>,
    keep_live: &IndexSet<usize>,
    realized_inputs: &mut IndexSet<usize>,
    stages: &mut Vec<BoomerangStage>,
    total_write_outs: &mut usize,
    num_reserved_writeouts: usize,
) -> Option<()> {
    // `targets` is always the FULL set of pins this partition still owes,
    // even when only `required` has to be finished before the caller can
    // continue. That matters: build_one_boomerang_stage decides which level-0
    // leaves to carry forward from the target set it is given, so handing it a
    // narrow per-wave set makes it drop leaves that a later round still needs
    // -- a DFF Q wanted by stage 1 simply vanishes from stage 0's output.
    loop {
        let done = match required {
            Some(req) => req.iter().all(|i| realized_inputs.contains(i)),
            None => targets.is_empty(),
        };
        if done { return Some(()) }
        let before_targets = targets.len();
        let before_realized = realized_inputs.len();
        let before_stages = stages.len();

        let stage = build_one_boomerang_stage(
            aig, targets, realized_inputs,
            total_write_outs, num_reserved_writeouts
        )?;
        stages.push(stage);

        // Re-arm every macro input whose phase has not run yet. The boomerang
        // only carries a pin forward while it is still a target, and it drops
        // each one the moment it is realized -- so an input realized in stage 0
        // has lost its shared-state slot by the time a phase runs after stage
        // 2. Re-inserting makes the next stage place it as a passthrough, which
        // is the same mechanism that keeps a primary input alive across stages.
        for &i in keep_live {
            if realized_inputs.contains(&i) {
                targets.insert(i);
            }
        }

        if targets.len() == before_targets &&
            realized_inputs.len() == before_realized &&
            stages.len() == before_stages + 1
        {
            clilog::error!(
                "boomerang made no progress with {} pins still unrealized; \
                 the leading one is aigpin {} driven by {:?}. This usually \
                 means a node the tree cannot represent (a macro output) was \
                 not seeded as a realized input.",
                targets.len(),
                targets[0], aig.drivers[targets[0]]
            );
            return None
        }
    }
}

/// Group the mid-partition macros reachable from `targets` into dependency
/// waves.
///
/// Returns a list of waves; every macro in wave `k` depends only on ordinary
/// logic and on macros in waves `< k`. Macros whose outputs are pure state
/// reads never appear, because they need no mid-partition evaluation.
fn order_macro_waves(
    aig: &AIG,
    targets: &IndexSet<usize>,
    realized_inputs: &IndexSet<usize>,
) -> Vec<Vec<usize>> {
    // Collect the macros in the cone of `targets`.
    let order = aig.topo_traverse_generic(
        Some(&targets.iter().copied().collect()),
        Some(realized_inputs)
    );
    let mut in_cone = IndexSet::<usize>::new();
    for &pin in &order {
        if realized_inputs.contains(&pin) { continue }
        if let Some(&macro_i) = aig.aigpin2macro.get(&pin) {
            let m = &aig.macros[macro_i];
            if m.kind.needs_mid_partition_eval() {
                in_cone.insert(macro_i);
            }
        }
    }
    if in_cone.is_empty() {
        return Vec::new()
    }

    // depth[m] = 1 + max depth of any macro reachable through m's eval fan-in.
    // `order` is topological, so a single forward sweep suffices.
    let mut pin_depth = IndexMap::<usize, usize>::new();
    let mut macro_depth = IndexMap::<usize, usize>::new();
    for &pin in &order {
        if realized_inputs.contains(&pin) {
            pin_depth.insert(pin, 0);
            continue
        }
        let mut d = 0usize;
        match aig.drivers[pin] {
            DriverType::AndGate(a, b) => {
                if (a >> 1) != 0 {
                    d = d.max(*pin_depth.get(&(a >> 1)).unwrap_or(&0));
                }
                if (b >> 1) != 0 {
                    d = d.max(*pin_depth.get(&(b >> 1)).unwrap_or(&0));
                }
            }
            DriverType::Macro(cellid, slot) => {
                if let Some(m) = aig.macros.get(&cellid) {
                    let mut fanin_depth = 0usize;
                    m.for_each_comb_fanin(slot, |i| {
                        fanin_depth = fanin_depth
                            .max(*pin_depth.get(&i).unwrap_or(&0));
                    });
                    // A macro that commits state inside the phase also has to
                    // wait for its next-state cone. This matters for an SRL
                    // cascade: `Q31 -> D` is register-to-register, but both
                    // ends live in a macro phase, and lanes of one phase run
                    // in parallel on the GPU. Without this the consumer would
                    // race its producer's write to shared state.
                    if m.kind.commits_state_in_phase() {
                        m.for_each_state_fanin(|i| {
                            fanin_depth = fanin_depth
                                .max(*pin_depth.get(&i).unwrap_or(&0));
                        });
                    }
                    if m.kind.output_needs_mid_partition_eval(slot) {
                        // crossing this macro costs one wave
                        d = fanin_depth + 1;
                        if let Some(&macro_i) = aig.aigpin2macro.get(&pin) {
                            let e = macro_depth.entry(macro_i).or_insert(0);
                            *e = (*e).max(d);
                        }
                    } else {
                        d = 0;
                    }
                }
            }
            _ => {}
        }
        pin_depth.insert(pin, d);
    }

    // Link CARRY4s into chains: an instance whose CI is driven by another
    // in-cone CARRY4's CO[3] is that instance's successor.
    let mut succ = IndexMap::<usize, usize>::new();   // macro_i -> next macro_i
    let mut has_pred = IndexSet::<usize>::new();
    for &macro_i in &in_cone {
        let m = &aig.macros[macro_i];
        if !matches!(m.kind, MacroKind::Carry4) { continue }
        let ci_iv = m.inputs[crate::macros::C4_IN_CI];
        if (ci_iv >> 1) == 0 { continue }
        let Some(&pred_i) = aig.aigpin2macro.get(&(ci_iv >> 1)) else { continue };
        if !in_cone.contains(&pred_i) { continue }
        let pred = &aig.macros[pred_i];
        if !matches!(pred.kind, MacroKind::Carry4) { continue }
        // only a CO[3] tap continues a chain; any other output is ordinary
        // combinational fan-out and does not merge the two into one scan.
        if pred.outputs[crate::macros::C4_OUT_CO + 3] != (ci_iv >> 1) { continue }
        // a fan-out of CO[3] to two different CARRY4s cannot be one linear
        // scan; leave the second one to start its own chain.
        if succ.contains_key(&pred_i) { continue }
        succ.insert(pred_i, macro_i);
        has_pred.insert(macro_i);
    }

    // Fixpoint over macro-to-macro dependencies the pin traversal cannot see.
    // An SRLC32E `Q31` feeding another SRL's `D` never shows up in the
    // comb-fanin walk, because `D` is next-state rather than a read port -- so
    // without this the producer and consumer land in the same phase and race,
    // since one phase's lanes execute in parallel on the GPU.
    //
    // Carry chains are deliberately exempt: a chained `CI` is supplied by the
    // warp scan, so a chain stays in one phase by design.
    for &macro_i in &in_cone { macro_depth.entry(macro_i).or_insert(1); }
    loop {
        let mut changed = false;
        for &macro_i in &in_cone {
            let m = &aig.macros[macro_i];
            let mut deps = Vec::<usize>::new();
            m.for_each_eval_fanin(|pin| deps.push(pin));
            if m.kind.commits_state_in_phase() {
                m.for_each_state_fanin(|pin| deps.push(pin));
            }
            let chained = has_pred.contains(&macro_i);
            let ci_pin = m.inputs[crate::macros::C4_IN_CI] >> 1;
            let mut need = *macro_depth.get(&macro_i).unwrap_or(&1);
            for pin in deps {
                if chained && matches!(m.kind, MacroKind::Carry4) && pin == ci_pin {
                    continue
                }
                if let Some(&prod) = aig.aigpin2macro.get(&pin) {
                    if prod != macro_i && in_cone.contains(&prod) {
                        need = need.max(*macro_depth.get(&prod).unwrap_or(&1) + 1);
                    }
                }
            }
            if need > *macro_depth.get(&macro_i).unwrap_or(&1) {
                macro_depth.insert(macro_i, need);
                changed = true;
            }
        }
        if !changed { break }
    }

    // Walk each chain from its head, and key the chain by the depth of its
    // external inputs -- the head's depth, since every later link's carry
    // comes from the scan rather than from the graph.
    let mut chains: Vec<(usize, Vec<usize>)> = Vec::new();
    for &macro_i in &in_cone {
        if has_pred.contains(&macro_i) { continue }
        let mut chain = vec![macro_i];
        let mut cur = macro_i;
        while let Some(&next) = succ.get(&cur) {
            // A warp scan is 32 lanes wide; split longer chains.
            if chain.len() == crate::macros::MACRO_MAX_LANES { break }
            chain.push(next);
            cur = next;
        }
        let depth = *macro_depth.get(&macro_i).unwrap_or(&1);
        chains.push((depth, chain));
    }
    // Any chain link beyond a 32-lane split still needs scheduling; pick it up
    // as its own chain head on a later pass.
    let mut placed = IndexSet::<usize>::new();
    for (_, c) in &chains { for &m in c { placed.insert(m); } }
    for &macro_i in &in_cone {
        if placed.contains(&macro_i) { continue }
        let mut chain = vec![macro_i];
        let mut cur = macro_i;
        while let Some(&next) = succ.get(&cur) {
            if chain.len() == crate::macros::MACRO_MAX_LANES || placed.contains(&next) { break }
            chain.push(next);
            cur = next;
        }
        for &m in &chain { placed.insert(m); }
        let depth = *macro_depth.get(&macro_i).unwrap_or(&1);
        chains.push((depth, chain));
    }

    // Group chains into phases by external-input depth, so independent chains
    // at the same depth share one phase and one script section.
    // A phase is capped at one warp, so a chain never straddles a shuffle
    // boundary and the section size is simply lanes * MACRO_LANE_WORDS.
    // Chains are themselves capped at MACRO_MAX_LANES above, so every chain
    // fits in a fresh wave.
    chains.sort_by_key(|(d, c)| (*d, c[0]));
    let mut waves: Vec<Vec<usize>> = Vec::new();
    let mut cur_depth = usize::MAX;
    for (d, chain) in chains {
        let need_new = match waves.last() {
            None => true,
            Some(w) => d != cur_depth ||
                w.len() + chain.len() > crate::macros::MACRO_MAX_LANES,
        };
        if need_new {
            waves.push(Vec::new());
            cur_depth = d;
        }
        waves.last_mut().unwrap().extend(chain);
    }
    waves.retain(|w| !w.is_empty());
    waves
}

impl Partition {
    /// build one partition given a set of endpoints to realize.
    ///
    /// if the resource is overflowed, None will be returned.
    /// see [Partition] for resource constraints.
    pub fn build_one(
        aig: &AIG,
        staged: &StagedAIG,
        endpoints: &Vec<usize>
    ) -> Option<Partition> {
        let mut unrealized_comb_outputs = IndexSet::new();
        let mut realized_inputs = staged.primary_inputs.as_ref()
            .cloned().unwrap_or_default();
        let mut num_srams = 0;
        let mut state_macros = Vec::new();
        let mut macro_input_words = 0usize;
        let mut macro_state_words = 0usize;
        let mut comb_outputs_activations = IndexMap::<usize, IndexSet<usize>>::new();
        for &endpt_i in endpoints {
            let edg = staged.get_endpoint_group(aig, endpt_i);
            edg.for_each_input(|i| {
                unrealized_comb_outputs.insert(i);
            });
            match edg {
                EndpointGroup::DFF(dff) => {
                    comb_outputs_activations.entry(dff.d_iv >> 1).or_default().insert(dff.en_iv << 1 | (dff.d_iv & 1));
                },
                EndpointGroup::PrimaryOutput(pin) => {
                    comb_outputs_activations.entry(pin >> 1).or_default().insert(2 | (pin & 1));
                },
                EndpointGroup::RAMBlock(_) => {
                    num_srams += 1;
                },
                EndpointGroup::Macro(m) => {
                    // A stateful macro reserves permute words to gather its
                    // next-state inputs and words to commit its wide state,
                    // exactly as an SRAM reserves 4 permute words.
                    macro_input_words += m.kind.input_words();
                    macro_state_words += m.kind.state_words();
                    if let Some(&pin) = m.outputs.iter().find(|&&p| p != 0) {
                        if let Some(&macro_i) = aig.aigpin2macro.get(&pin) {
                            state_macros.push(macro_i);
                        }
                    }
                },
                EndpointGroup::StagedIOPin(pin) => {
                    comb_outputs_activations.entry(pin).or_default().insert(2);
                },
            }
        }
        let num_output_dups = comb_outputs_activations.iter()
            .map(|(_, ckens)| ckens.len() - 1)
            .sum::<usize>();
        let num_reserved_writeouts =
            num_srams + macro_state_words + (num_output_dups + 31) / 32;
        if num_reserved_writeouts >= BOOMERANG_MAX_WRITEOUTS ||
            num_srams * 4 + macro_input_words + num_output_dups
                > BOOMERANG_MAX_WRITEOUTS
        {
            clilog::warn!(
                PART_REJECT_WRITEOUT,
                "partition rejected on the write-out budget (cap {}):                  reserved={} (srams {} + macro_state {} + dup_words {}),                  gather_lanes={} (srams*4 {} + macro_inputs {} + dups {}),                  {} endpoints",
                BOOMERANG_MAX_WRITEOUTS,
                num_reserved_writeouts, num_srams, macro_state_words,
                (num_output_dups + 31) / 32,
                num_srams * 4 + macro_input_words + num_output_dups,
                num_srams * 4, macro_input_words, num_output_dups,
                endpoints.len()
            );
            return None
        }

        // Every macro output that is a pure state read (DSP `P`, SRL `Q31`) is
        // available from the global macro-state array at cycle start, exactly
        // like a DFF `Q`. Seeding them as realised inputs is what stops the
        // boomerang traversal from trying to descend into a macro, which
        // place_bit cannot represent.
        for m in aig.macros.values() {
            for pin in m.cycle_start_outputs() {
                realized_inputs.insert(pin);
            }
        }

        // Order the mid-partition macros into waves. Wave 0 macros depend only
        // on ordinary logic; wave k macros consume a wave k-1 macro output.
        // A CARRY4 carry chain of n links yields n waves, each collapsing to a
        // single macro phase rather than a whole major stage -- which is the
        // entire point of doing this in-partition.
        let macro_waves = order_macro_waves(aig, &unrealized_comb_outputs, &realized_inputs);

        let mut stages = Vec::<BoomerangStage>::new();
        let mut macro_stages = Vec::<MacroStage>::new();
        let mut total_write_outs = 0;

        // Realise each macro wave's inputs, then evaluate the wave, then move
        // on. The endpoint cone is realised last.
        // Every mid-partition macro output is a graph LEAF in every round:
        // the boomerang tree can never compute one, so it must not appear as a
        // target before its phase has run. Without this, round 0 happily tries
        // to pass through a CARRY4 CO[3] that nothing has produced yet.
        // Mid-partition macro outputs are seeded into realized_inputs up
        // front. This is load-bearing and NOT the cause of the MIPS ordering
        // bug: without it the boomerang traversal descends into a macro's
        // combinational fan-in, the macro node lands at level 0 anyway (its
        // driver is not an AndGate), and compute_lvl1_necessary_nodes places it
        // as a passthrough in whatever stage first reaches it -- which then
        // asks for a shared-state slot no phase has written. Removing the seed
        // was measured to break every previously-green netlist, not just MIPS.
        //
        // They are still withheld from all_targets until their phase runs, so
        // the tree never tries to *compute* one.
        let mut pending_outputs = IndexSet::<usize>::new();
        for wave in &macro_waves {
            for &macro_i in wave {
                if macro_i == usize::MAX { continue }
                for pin in aig.macros[macro_i].mid_partition_outputs() {
                    realized_inputs.insert(pin);
                    pending_outputs.insert(pin);
                }
            }
        }
        let endpoint_targets = unrealized_comb_outputs.clone();

        // One shared target set for the whole partition; waves only choose
        // WHEN a subset must be finished, never what is in flight.
        let mut all_targets = unrealized_comb_outputs.iter().copied()
            .filter(|i| !pending_outputs.contains(i))
            .collect::<IndexSet<_>>();
        let mut wave_reqs: Vec<IndexSet<usize>> = Vec::new();
        for wave in &macro_waves {
            let mut wave_inputs = IndexSet::new();
            for (l, &macro_i) in wave.iter().enumerate() {
                if macro_i == usize::MAX { continue }
                let m = &aig.macros[macro_i];
                // The carry-in of a non-head chain link is supplied by the
                // warp scan, not gathered from shared_state, so it must not be
                // demanded as a realised input -- doing so would force the
                // predecessor's CO[3] to be materialised in the tree and
                // defeat the whole point of scanning.
                let scanned_ci = !is_chain_start(aig, wave, l);
                // A macro that commits its own state in the phase needs its
                // next-state cone realised by then as well, not just the
                // combinational read cone.
                let state_slots = if m.kind.commits_state_in_phase() {
                    m.kind.state_fanin()
                } else { Vec::new() };
                for slot in 0..m.kind.num_inputs() {
                    if scanned_ci && slot == crate::macros::C4_IN_CI { continue }
                    if !state_slots.contains(&slot) &&
                        !(0..m.kind.num_outputs()).any(|o|
                            m.kind.output_needs_mid_partition_eval(o) &&
                            m.kind.comb_fanin_of_output(o).contains(&slot))
                    {
                        continue
                    }
                    let iv = m.inputs[slot];
                    if (iv >> 1) != 0 && !realized_inputs.contains(&(iv >> 1)) {
                        wave_inputs.insert(iv >> 1);
                    }
                }
            }
            for &i in &wave_inputs { all_targets.insert(i); }
            wave_reqs.push(wave_inputs);
        }

        for (wi, wave) in macro_waves.iter().enumerate() {
            // Inputs of this wave and every wave still to come must stay
            // addressable until their phase consumes them.
            let keep_live = wave_reqs[wi..].iter()
                .flat_map(|r| r.iter().copied())
                .collect::<IndexSet<_>>();
            let req = wave_reqs[wi].clone();
            for &i in &keep_live {
                if !realized_inputs.contains(&i) { all_targets.insert(i); }
            }
            // Hold back anything that would read a macro output this wave (or
            // a later one) has not produced yet; it is restored below.
            let blocked = targets_blocked_by_pending(
                aig, &all_targets, &realized_inputs, &pending_outputs);
            // Their leaves must stay addressable even though the targets
            // themselves are on hold.
            let mut keep_live = keep_live;
            for leaf in realized_leaves_of(aig, &blocked, &realized_inputs) {
                // Pending macro outputs also reside in realized_inputs and so
                // are returned as "leaves" of the blocked cone. Re-arming them
                // would reinstate the pins this pass deliberately withheld.
                if pending_outputs.contains(&leaf) { continue }
                keep_live.insert(leaf);
            }
            for b in &blocked { all_targets.swap_remove(b); }
            let r = run_boomerang_stages_until_realized(
                aig, &mut all_targets, Some(&req), &keep_live,
                &mut realized_inputs,
                &mut stages, &mut total_write_outs, num_reserved_writeouts
            );
            for &b in &blocked {
                if !realized_inputs.contains(&b) { all_targets.insert(b); }
            }
            r?;
            macro_stages.push(MacroStage {
                after_stage: stages.len(),
                lanes: wave.iter().enumerate()
                    .filter(|(_, &m)| m != usize::MAX)
                    .map(|(l, &macro_i)| MacroLane {
                        macro_i,
                        chain_start: is_chain_start(aig, wave, l),
                    })
                    .collect(),
            });
            // The phase has now produced these, so an output that is itself
            // an endpoint input becomes a legal target: the final round will
            // place it as a passthrough so it gets a write-out slot.
            for &macro_i in wave {
                if macro_i == usize::MAX { continue }
                for pin in aig.macros[macro_i].mid_partition_outputs() {
                    pending_outputs.swap_remove(&pin);
                    if endpoint_targets.contains(&pin) {
                        all_targets.insert(pin);
                    }
                }
            }
        }

        // Finish everything still outstanding. Macro outputs are deliberately
        // still in here: one that directly drives an endpoint (a CARRY4 `O`
        // bit wired to a primary output) has a value but no write-out slot, so
        // the boomerang places it as a level-0 passthrough -- the same
        // mechanism that already handles a primary input feeding an output.
        run_boomerang_stages_until_realized(
            aig, &mut all_targets, None, &IndexSet::new(), &mut realized_inputs,
            &mut stages, &mut total_write_outs, num_reserved_writeouts
        )?;

        Some(Partition {
            endpoints: endpoints.clone(),
            stages,
            macro_stages,
            state_macros,
        })
    }
}

/// Given an initial clustering solution of endpoints, generate and map a
/// refined solution.
///
/// The refined solution will have smaller number of partitions
/// as we aggressively merge the partitions when possible.
pub fn process_partitions(
    aig: &AIG,
    staged: &StagedAIG,
    mut parts: Vec<Vec<usize>>,
    max_stage_degrad: usize,
) -> Option<Vec<Partition>> {
    let cnt_nodes = parts.par_iter().map(|v| {
        let mut comb_outputs = Vec::new();
        for &endpt_i in v {
            staged.get_endpoint_group(aig, endpt_i).for_each_input(|i| {
                comb_outputs.push(i);
            });
        }
        let order = aig.topo_traverse_generic(
            Some(&comb_outputs),
            staged.primary_inputs.as_ref(),
        );
        order.len()
    }).collect::<Vec<_>>();

    let all_original_parts = parts.par_iter().enumerate().map(|(i, v)| {
        let part = Partition::build_one(aig, staged, v);
        if part.is_none() {
            clilog::error!("Partition {} exceeds resource constraint.", i);
        }
        part
    }).collect::<Vec<_>>();
    let all_original_parts = all_original_parts.into_iter().collect::<Option<Vec<_>>>()?;
    let max_original_nstages = all_original_parts.iter()
        .map(|p| p.stages.len()).max().unwrap();

    let mut effective_parts = Vec::<Partition>::new();
    let max_trials = (all_original_parts.len() / 8).max(20);
    for (i, mut partition_self) in all_original_parts.into_iter().enumerate() {
        if parts[i].is_empty() {
            continue
        }
        let mut merge_blacklist = HashSet::<usize>::new();
        let mut cnt_node_i = cnt_nodes[i];
        loop {
            let mut comb_outputs = Vec::new();
            for &endpt_i in &parts[i] {
                staged.get_endpoint_group(aig, endpt_i).for_each_input(|i| {
                    comb_outputs.push(i);
                });
            }

            let mut merge_choices = parts[i + 1..parts.len()].par_iter().enumerate().filter_map(|(j, v)| {
                if v.is_empty() { return None }
                if merge_blacklist.contains(&(i + j + 1)) {
                    return None
                }
                let mut comb_outputs = comb_outputs.clone();
                for &endpt_i in v {
                    staged.get_endpoint_group(aig, endpt_i).for_each_input(|i| {
                        comb_outputs.push(i);
                    });
                }
                let order = aig.topo_traverse_generic(
                    Some(&comb_outputs),
                    staged.primary_inputs.as_ref(),
                );
                Some((order.len() - cnt_nodes[i + j + 1].max(cnt_node_i),
                      order.len(),
                      i + j + 1))
            }).collect::<Vec<_>>();
            merge_choices.sort();
            let mut merged = false;

            #[derive(Clone)]
            struct PartsPartitions {
                parts_ij: Vec<usize>,
                partition_ij: Option<Partition>,
            }
            let mut merge_trials: Vec<Option<PartsPartitions>> =
                vec![None; merge_choices.len()];
            let mut parallel_trial_stride = 4;

            for (merge_i, &(_cnt_diff, cnt_new, j)) in merge_choices.iter().enumerate() {
                if merge_trials[merge_i].is_none() {
                    if merge_i > max_trials {
                        break   // do not try too more
                    }
                    let rhs = merge_trials.len().min(
                        merge_i + parallel_trial_stride);
                    merge_trials[merge_i..rhs].par_iter_mut().enumerate().for_each(|(merge_j, trial)| {
                        let j = merge_choices[merge_i + merge_j].2;
                        let parts_ij = parts[i].iter().chain(parts[j].iter()).copied().collect();
                        let partition_ij = Partition::build_one(aig, staged, &parts_ij);
                        *trial = Some(PartsPartitions {
                            parts_ij, partition_ij
                        });
                    });
                    parallel_trial_stride *= 2;
                }

                let PartsPartitions {
                    parts_ij, partition_ij
                } = merge_trials[merge_i].take().unwrap();

                match partition_ij {
                    None => {
                        merge_blacklist.insert(j);
                    }
                    Some(partition) if partition.stages.len() >
                        max_original_nstages + max_stage_degrad =>
                    {
                        clilog::debug!("skipped merging {} with {} due to nstage degradation: \
                                        {} > {}", i, j, partition.stages.len(),
                                       max_original_nstages + max_stage_degrad);
                        merge_blacklist.insert(j);
                    }
                    Some(partition) => {
                        clilog::info!("merged partition {} with {}", i, j);
                        parts[i] = parts_ij;
                        parts[j] = vec![];
                        partition_self = partition;
                        merged = true;
                        cnt_node_i = cnt_new;
                        break
                    },
                }
            }
            if !merged { break }
        }

        clilog::info!("part {}: #stages {}, #macro-waves {} ({} macros) at {:?}, #state-macros {}",
                      i, partition_self.stages.len(),
                      partition_self.macro_stages.len(),
                      partition_self.macro_stages.iter()
                          .map(|ms| ms.num_lanes()).sum::<usize>(),
                      partition_self.macro_stages.iter()
                          .map(|ms| ms.after_stage).collect::<Vec<_>>(),
                      partition_self.state_macros.len());
        effective_parts.push(partition_self);
    }
    effective_parts.sort_by_key(|p| usize::MAX - p.stages.len());
    Some(effective_parts)
}

/// Read a cluster solution from hgr.part.xx file.
/// Then call [process_partitions].
pub fn process_partitions_from_hgr_parts_file(
    aig: &AIG,
    staged: &StagedAIG,
    hgr_parts_file: &PathBuf,
    max_stage_degrad: usize,
) -> Option<Vec<Partition>> {
    use std::io::{BufRead, BufReader};
    use std::fs::File;

    let mut parts = Vec::<Vec<usize>>::new();
    let f_parts = File::open(&hgr_parts_file).unwrap();
    let f_parts = BufReader::new(f_parts);
    for (i, line) in f_parts.lines().enumerate() {
        let line = line.unwrap();
        if line.is_empty() { continue }
        let part_id = line.parse::<usize>().unwrap();
        while parts.len() <= part_id {
            parts.push(vec![]);
        }
        parts[part_id].push(i);
    }
    clilog::info!("read parts file {} with {} parts",
                  hgr_parts_file.display(), parts.len());

    process_partitions(aig, staged, parts, max_stage_degrad)
}
