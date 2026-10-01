use alloc::{
    collections::{BTreeMap, VecDeque},
    vec::Vec,
};
use core::{cell::Cell, cmp::Ordering};

use geometric_traits::traits::{SparseMatrix2D, SparseValuedMatrix2DRef, SparseValuedMatrixRef};
use hashbrown::{HashMap, HashSet};

use super::{
    RingMembership, Smiles, SmilesAtomPolicy, SymmSssrResult, SymmSssrStatus, canonicalize_cycle,
    cycle_edges,
};
use crate::bond::bond_edge::{BondEdge, bond_edge_other};

const WHITE: u8 = 0;
const GRAY: u8 = 1;
const BLACK: u8 = 2;

#[cfg(test)]
pub(crate) fn symmetrize_sssr_with_status<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
) -> SymmSssrResult {
    RingSearchState::new(smiles).symmetrize_sssr_with_status()
}

pub(crate) fn symmetrize_sssr_with_ring_membership(
    smiles: &Smiles<impl SmilesAtomPolicy>,
    ring_membership: &RingMembership,
) -> SymmSssrResult {
    if let Some(cycles) =
        try_simple_cycle_blocks_from_ring_membership(ring_membership, smiles.nodes().len())
    {
        return SymmSssrResult { cycles, status: SymmSssrStatus::default() };
    }

    RingSearchState::new(smiles).symmetrize_sssr_with_status()
}

fn try_simple_cycle_blocks_from_ring_membership(
    ring_membership: &RingMembership,
    atom_count: usize,
) -> Option<Vec<Vec<usize>>> {
    if ring_membership.atom_ids().is_empty() {
        return Some(Vec::new());
    }

    let mut ring_neighbors = vec![Vec::<usize>::new(); atom_count];
    for &[left, right] in ring_membership.bond_edges() {
        ring_neighbors[left].push(right);
        ring_neighbors[right].push(left);
    }
    if ring_membership
        .bond_edges()
        .iter()
        .any(|&[left, right]| ring_neighbors[left].len() > 2 && ring_neighbors[right].len() > 2)
    {
        return None;
    }

    let mut search = SimpleCycleBlockSearch {
        ring_neighbors: &ring_neighbors,
        discovery_order: vec![0_usize; atom_count],
        lowlink: vec![0_usize; atom_count],
        parent: vec![None::<usize>; atom_count],
        edge_stack: Vec::<[usize; 2]>::with_capacity(ring_membership.bond_edges().len()),
        time: 0,
        cycles: Vec::<Vec<usize>>::new(),
    };

    for &start_atom_id in ring_membership.atom_ids() {
        if search.discovery_order[start_atom_id] != 0 {
            continue;
        }
        search.collect_depth_first(start_atom_id)?;
    }

    search.cycles.sort_unstable();
    search.cycles.dedup();
    Some(search.cycles)
}

struct SimpleCycleBlockSearch<'a> {
    ring_neighbors: &'a [Vec<usize>],
    discovery_order: Vec<usize>,
    lowlink: Vec<usize>,
    parent: Vec<Option<usize>>,
    edge_stack: Vec<[usize; 2]>,
    time: usize,
    cycles: Vec<Vec<usize>>,
}

impl SimpleCycleBlockSearch<'_> {
    fn collect_depth_first(&mut self, atom_id: usize) -> Option<()> {
        self.time += 1;
        self.discovery_order[atom_id] = self.time;
        self.lowlink[atom_id] = self.time;

        for &neighbor_atom_id in &self.ring_neighbors[atom_id] {
            if self.discovery_order[neighbor_atom_id] == 0 {
                self.parent[neighbor_atom_id] = Some(atom_id);
                self.edge_stack.push(edge_key(atom_id, neighbor_atom_id));
                self.collect_depth_first(neighbor_atom_id)?;
                self.lowlink[atom_id] = self.lowlink[atom_id].min(self.lowlink[neighbor_atom_id]);
                if self.lowlink[neighbor_atom_id] >= self.discovery_order[atom_id] {
                    let mut component_edges = Vec::<[usize; 2]>::new();
                    while let Some(edge) = self.edge_stack.pop() {
                        component_edges.push(edge);
                        if edge == edge_key(atom_id, neighbor_atom_id) {
                            break;
                        }
                    }
                    self.cycles.push(simple_cycle_from_block_edges(&component_edges)?);
                }
            } else if self.parent[atom_id] != Some(neighbor_atom_id)
                && self.discovery_order[neighbor_atom_id] < self.discovery_order[atom_id]
            {
                self.lowlink[atom_id] =
                    self.lowlink[atom_id].min(self.discovery_order[neighbor_atom_id]);
                self.edge_stack.push(edge_key(atom_id, neighbor_atom_id));
            }
        }

        Some(())
    }
}

fn simple_cycle_from_block_edges(component_edges: &[[usize; 2]]) -> Option<Vec<usize>> {
    let mut component_neighbors = HashMap::<usize, Vec<usize>>::new();
    for &[left, right] in component_edges {
        component_neighbors.entry(left).or_default().push(right);
        component_neighbors.entry(right).or_default().push(left);
    }

    if component_neighbors.len() < 3 || component_edges.len() != component_neighbors.len() {
        return None;
    }
    if component_neighbors.values().any(|neighbors| neighbors.len() != 2) {
        return None;
    }

    let start_atom_id = *component_neighbors.keys().min()?;
    let mut cycle = Vec::<usize>::with_capacity(component_edges.len());
    let mut previous_atom_id = None;
    let mut current_atom_id = start_atom_id;

    loop {
        if cycle.contains(&current_atom_id) {
            return None;
        }
        cycle.push(current_atom_id);

        let neighbor_atom_ids = component_neighbors.get(&current_atom_id)?;
        let next_atom_id = match previous_atom_id {
            None => neighbor_atom_ids[0],
            Some(previous_atom_id) if neighbor_atom_ids[0] == previous_atom_id => {
                neighbor_atom_ids[1]
            }
            Some(previous_atom_id) if neighbor_atom_ids[1] == previous_atom_id => {
                neighbor_atom_ids[0]
            }
            Some(_) => return None,
        };

        previous_atom_id = Some(current_atom_id);
        current_atom_id = next_atom_id;
        if current_atom_id == start_atom_id {
            break;
        }
    }

    if cycle.len() != component_edges.len() {
        return None;
    }

    Some(canonicalize_cycle(&cycle))
}

#[cfg(test)]
fn try_disjoint_simple_cycles_from_ring_membership(
    ring_membership: &RingMembership,
    atom_count: usize,
) -> Option<Vec<Vec<usize>>> {
    if ring_membership.atom_ids().is_empty() {
        return Some(Vec::new());
    }
    if ring_membership.bond_edges().len() != ring_membership.atom_ids().len() {
        return None;
    }

    let mut ring_neighbors = vec![Vec::<usize>::new(); atom_count];
    for &[left, right] in ring_membership.bond_edges() {
        ring_neighbors[left].push(right);
        ring_neighbors[right].push(left);
    }
    for &atom_id in ring_membership.atom_ids() {
        if ring_neighbors[atom_id].len() != 2 {
            return None;
        }
    }

    let mut seen = vec![false; atom_count];
    let mut cycles = Vec::<Vec<usize>>::new();
    for &start_atom_id in ring_membership.atom_ids() {
        if seen[start_atom_id] {
            continue;
        }

        let mut cycle = Vec::<usize>::new();
        let mut previous_atom_id = None;
        let mut current_atom_id = start_atom_id;
        loop {
            if current_atom_id == start_atom_id && !cycle.is_empty() {
                break;
            }
            if seen[current_atom_id] {
                return None;
            }
            seen[current_atom_id] = true;
            cycle.push(current_atom_id);

            let next_atom_id = match previous_atom_id {
                None => ring_neighbors[current_atom_id][0],
                Some(previous_atom_id) => {
                    let neighbor_atom_ids = &ring_neighbors[current_atom_id];
                    if neighbor_atom_ids[0] == previous_atom_id {
                        neighbor_atom_ids[1]
                    } else {
                        neighbor_atom_ids[0]
                    }
                }
            };
            previous_atom_id = Some(current_atom_id);
            current_atom_id = next_atom_id;
        }

        if cycle.len() < 3 {
            return None;
        }
        cycles.push(canonicalize_cycle(&cycle));
    }

    cycles.sort_unstable();
    cycles.dedup();
    Some(cycles)
}

#[derive(Debug, Clone, Default)]
struct RingSearchResult {
    rings: Vec<Vec<usize>>,
    extras: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
struct SmallestRingsBfsScratch {
    done: Vec<u8>,
    parents: Vec<Option<usize>>,
    depths: Vec<usize>,
    queue: VecDeque<usize>,
}

impl SmallestRingsBfsScratch {
    fn new(atom_count: usize) -> Self {
        Self {
            done: vec![WHITE; atom_count],
            parents: vec![None; atom_count],
            depths: vec![0; atom_count],
            queue: VecDeque::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct RingSearchState<'a, AtomPolicy = crate::smiles::ConcreteAtoms> {
    smiles: &'a Smiles<AtomPolicy>,
    fragments: Vec<Vec<usize>>,
    ordered_incident_bonds: Vec<Vec<BondEdge>>,
    bfs_scratch: SmallestRingsBfsScratch,
    d2_pick_marks: Vec<u32>,
    d2_pick_generation: u32,
    active_edges: HashSet<[usize; 2]>,
    rdkit_bridge_scan_order: RdkitBridgeScanOrder,
    bfs_budget: RdkitBfsBudget,
    atom_degrees: Vec<usize>,
    ring_atoms: Vec<bool>,
    ring_bonds: HashSet<[usize; 2]>,
    seen_invariants: HashSet<RdkitRingInvariant>,
    used_fallback: Cell<bool>,
    hit_queue_cutoff: Cell<bool>,
}

impl<'a, AtomPolicy: SmilesAtomPolicy> RingSearchState<'a, AtomPolicy> {
    fn new(smiles: &'a Smiles<AtomPolicy>) -> Self {
        Self::with_bfs_budget(smiles, RdkitBfsBudget::default())
    }

    #[cfg(test)]
    fn new_with_bfs_budget(smiles: &'a Smiles<AtomPolicy>, bfs_budget: RdkitBfsBudget) -> Self {
        Self::with_bfs_budget(smiles, bfs_budget)
    }

    fn with_bfs_budget(smiles: &'a Smiles<AtomPolicy>, bfs_budget: RdkitBfsBudget) -> Self {
        let atom_count = smiles.nodes().len();
        let ordered_incident_bonds = (0..atom_count)
            .map(|atom_id| RdkitIncidentBondOrdering::for_node(smiles, atom_id))
            .collect::<Vec<_>>();
        let mut active_edges = HashSet::<[usize; 2]>::with_capacity(smiles.number_of_bonds());
        let mut atom_degrees = vec![0_usize; atom_count];

        for incident_bonds in ordered_incident_bonds.iter().take(atom_count) {
            for edge in incident_bonds {
                let key = edge_key(edge.source(), edge.target());
                if active_edges.insert(key) {
                    atom_degrees[edge.source()] += 1;
                    atom_degrees[edge.target()] += 1;
                }
            }
        }

        Self {
            smiles,
            fragments: connected_fragments(smiles),
            ordered_incident_bonds,
            // The search revisits the same graph many times, so keep the BFS
            // buffers and the degree-2 mark array on the state instead of
            // rebuilding them for each probe.
            bfs_scratch: SmallestRingsBfsScratch::new(atom_count),
            d2_pick_marks: vec![0; atom_count],
            d2_pick_generation: 1,
            active_edges,
            rdkit_bridge_scan_order: RdkitBridgeScanOrder::new(smiles),
            bfs_budget,
            atom_degrees,
            ring_atoms: vec![false; atom_count],
            ring_bonds: HashSet::new(),
            seen_invariants: HashSet::new(),
            used_fallback: Cell::new(false),
            hit_queue_cutoff: Cell::new(false),
        }
    }

    fn status(&self) -> SymmSssrStatus {
        SymmSssrStatus {
            used_fallback: self.used_fallback.get(),
            hit_queue_cutoff: self.hit_queue_cutoff.get(),
        }
    }

    fn find_sssr(&mut self) -> RingSearchResult {
        let mut result = RingSearchResult::default();

        for fragment in self.fragments.clone() {
            if let Some(fast_rings) =
                self.find_fragment_sssr(&fragment, &mut result.rings, &mut result.extras)
            {
                result.rings = fast_rings;
                result.extras.clear();
                return result;
            }
        }

        result.rings.sort_unstable();
        result.rings.dedup();
        result.extras.sort_unstable();
        result.extras.dedup();
        result
    }

    fn symmetrize_sssr_with_status(mut self) -> SymmSssrResult {
        let result = self.find_sssr();
        if result.extras.is_empty() {
            return SymmSssrResult { cycles: result.rings, status: self.status() };
        }

        let mut rings = result.rings;
        RdkitExtraRingResolution::promote_extras(&mut rings, result.extras);
        rings.sort_unstable();
        rings.dedup();
        SymmSssrResult { cycles: rings, status: self.status() }
    }

    fn find_fragment_sssr(
        &mut self,
        fragment: &[usize],
        rings: &mut Vec<Vec<usize>>,
        extras: &mut Vec<Vec<usize>>,
    ) -> Option<Vec<Vec<usize>>> {
        if fragment.len() < 3 {
            return None;
        }

        let mut changed = alloc::collections::VecDeque::<usize>::new();
        for &atom_id in fragment {
            if self.atom_degrees[atom_id] < 2 {
                changed.push_back(atom_id);
            }
        }

        let mut done_atoms = vec![false; self.smiles.nodes().len()];
        let mut atoms_done = 0_usize;
        let mut fragment_rings = Vec::<Vec<usize>>::new();

        while atoms_done + 2 < fragment.len() {
            while let Some(cand) = changed.pop_front() {
                if done_atoms[cand] {
                    continue;
                }
                done_atoms[cand] = true;
                atoms_done += 1;
                self.trim_bonds(cand, &mut changed);
            }

            let mut d2nodes = Vec::<usize>::new();
            self.pick_d2_nodes(fragment, &mut d2nodes);
            if d2nodes.is_empty() {
                let Some(cand) =
                    fragment.iter().copied().find(|&atom_id| self.atom_degrees[atom_id] == 3)
                else {
                    break;
                };

                self.find_rings_d3node(cand, &mut fragment_rings);
                done_atoms[cand] = true;
                atoms_done += 1;
                self.trim_bonds(cand, &mut changed);
            } else {
                self.find_rings_d2nodes(&d2nodes, &mut fragment_rings);
                for &d2node in &d2nodes {
                    if !done_atoms[d2node] {
                        done_atoms[d2node] = true;
                        atoms_done += 1;
                        self.trim_bonds(d2node, &mut changed);
                    }
                }
            }
        }

        let ring_count = fragment_rings.len();
        let bond_count = fragment
            .iter()
            .flat_map(|&atom_id| self.ordered_incident_bonds[atom_id].iter())
            .map(|edge| edge_key(edge.source(), edge.target()))
            .collect::<HashSet<_>>()
            .len();
        let expected_rings = cyclomatic_ring_count(bond_count, fragment.len());

        if ring_count < expected_rings
            && let Some(first_possible) = self.find_first_possible_bridge(bond_count)
        {
            let mut dead_bonds = HashSet::<[usize; 2]>::new();
            let mut possible = vec![first_possible];
            while let Some(edge) = possible.pop() {
                if !self.find_ring_connecting_atoms(edge, &mut fragment_rings) {
                    dead_bonds.insert(edge);
                }
                possible.clear();
                if let Some(next) = self.find_next_possible_bridge(bond_count, &dead_bonds) {
                    possible.push(next);
                }
            }
        }

        if fragment_rings.len() < expected_rings {
            self.used_fallback.set(true);
            // This matches RDKit's own last-resort path in MolOps::findSSSR():
            // after the exact stage and bridge recovery still undercount rings,
            // it warns and switches to fastFindRings(). Whole-PubChem contains
            // real molecules, such as CID 58882885, that trigger this path in
            // RDKit itself.
            let all_atoms = (0..self.smiles.nodes().len()).collect::<Vec<_>>();
            let mut fast_rings = fast_find_rings(&self.ordered_incident_bonds, &all_atoms);
            fast_rings.sort_unstable();
            return Some(fast_rings);
        }

        for ring in &mut fragment_rings {
            *ring = canonicalize_cycle(ring);
        }

        if fragment_rings.len() > expected_rings {
            extras.extend(RdkitExtraRingResolution::split_extras(&mut fragment_rings));
        }

        rings.extend(fragment_rings);
        None
    }
    fn trim_bonds(&mut self, cand: usize, changed: &mut VecDeque<usize>) {
        for edge in &self.ordered_incident_bonds[cand] {
            let key = edge_key(edge.source(), edge.target());
            if !self.active_edges.remove(&key) {
                continue;
            }
            if let Some(other) = bond_edge_other(*edge, cand) {
                if self.atom_degrees[other] <= 2 {
                    changed.push_back(other);
                }
                self.atom_degrees[other] = self.atom_degrees[other].saturating_sub(1);
            }
            self.atom_degrees[cand] = self.atom_degrees[cand].saturating_sub(1);
        }
    }

    fn next_d2_pick_generation(&mut self) -> u32 {
        if self.d2_pick_generation == u32::MAX {
            self.d2_pick_marks.fill(0);
            self.d2_pick_generation = 1;
        } else {
            self.d2_pick_generation += 1;
        }
        self.d2_pick_generation
    }

    fn mark_useless_d2s(
        ordered_incident_bonds: &[Vec<BondEdge>],
        atom_degrees: &[usize],
        root: usize,
        forb: &mut [u32],
        generation: u32,
        active_edges: &HashSet<[usize; 2]>,
    ) {
        for edge in &ordered_incident_bonds[root] {
            let key = edge_key(edge.source(), edge.target());
            if !active_edges.contains(&key) {
                continue;
            }
            let Some(other) = bond_edge_other(*edge, root) else {
                continue;
            };
            if forb[other] != generation && atom_degrees[other] == 2 {
                forb[other] = generation;
                Self::mark_useless_d2s(
                    ordered_incident_bonds,
                    atom_degrees,
                    other,
                    forb,
                    generation,
                    active_edges,
                );
            }
        }
    }

    fn pick_d2_nodes(&mut self, fragment: &[usize], d2nodes: &mut Vec<usize>) {
        d2nodes.clear();

        let generation = self.next_d2_pick_generation();
        loop {
            let mut root = None;
            for &atom_id in fragment {
                if self.atom_degrees[atom_id] == 2 && self.d2_pick_marks[atom_id] != generation {
                    root = Some(atom_id);
                    d2nodes.push(atom_id);
                    self.d2_pick_marks[atom_id] = generation;
                    break;
                }
            }
            let Some(root) = root else {
                break;
            };
            Self::mark_useless_d2s(
                &self.ordered_incident_bonds,
                &self.atom_degrees,
                root,
                &mut self.d2_pick_marks,
                generation,
                &self.active_edges,
            );
        }
    }

    fn find_rings_d2nodes(&mut self, d2nodes: &[usize], fragment_rings: &mut Vec<Vec<usize>>) {
        let mut dup_d2_cands = BTreeMap::<RdkitRingInvariant, Vec<usize>>::new();
        let mut dup_map = BTreeMap::<usize, Vec<usize>>::new();

        for &cand in d2nodes {
            let srings = self.smallest_rings_bfs(cand, None);
            if srings.is_empty() {
                let mut local_changed = VecDeque::from([cand]);
                while let Some(local_cand) = local_changed.pop_front() {
                    self.trim_bonds(local_cand, &mut local_changed);
                }
                continue;
            }

            for ring in srings {
                let invariant = compute_ring_invariant(&ring, self.smiles.nodes().len());
                let duplicate_cands = dup_d2_cands.entry(invariant.clone()).or_default();
                if self.seen_invariants.insert(invariant) {
                    fragment_rings.push(canonicalize_cycle(&ring));
                    self.mark_ring_membership(&ring);
                } else {
                    for &other_cand in duplicate_cands.iter() {
                        dup_map.entry(cand).or_default().push(other_cand);
                        dup_map.entry(other_cand).or_default().push(cand);
                    }
                }
                duplicate_cands.push(cand);
            }
        }

        if !dup_map.is_empty() {
            self.find_sssr_for_dup_cands(fragment_rings, &dup_map, dup_d2_cands);
        }
    }

    fn find_sssr_for_dup_cands(
        &mut self,
        fragment_rings: &mut Vec<Vec<usize>>,
        dup_map: &BTreeMap<usize, Vec<usize>>,
        dup_d2_cands: BTreeMap<RdkitRingInvariant, Vec<usize>>,
    ) {
        for (_, dup_cands) in dup_d2_cands {
            if dup_cands.len() <= 1 {
                continue;
            }

            let mut candidate_rings = Vec::<Vec<usize>>::new();
            let mut min_size = usize::MAX;
            for dup_cand in dup_cands {
                let mut local_degrees = self.atom_degrees.clone();
                let mut local_edges = self.active_edges.clone();
                let mut local_changed = VecDeque::<usize>::new();
                if let Some(blockers) = dup_map.get(&dup_cand) {
                    for &blocker in blockers {
                        self.trim_bonds_on_copy(
                            blocker,
                            &mut local_changed,
                            &mut local_degrees,
                            &mut local_edges,
                        );
                    }
                }
                let rings = self.smallest_rings_bfs_with_state(dup_cand, None, &local_edges);
                for ring in rings {
                    min_size = min_size.min(ring.len());
                    candidate_rings.push(ring);
                }
            }
            for ring in candidate_rings {
                if ring.len() == min_size {
                    let invariant = compute_ring_invariant(&ring, self.smiles.nodes().len());
                    if self.seen_invariants.insert(invariant) {
                        fragment_rings.push(canonicalize_cycle(&ring));
                    }
                }
            }
        }
    }

    fn trim_bonds_on_copy(
        &self,
        cand: usize,
        changed: &mut VecDeque<usize>,
        atom_degrees: &mut [usize],
        active_edges: &mut HashSet<[usize; 2]>,
    ) {
        for edge in &self.ordered_incident_bonds[cand] {
            let key = edge_key(edge.source(), edge.target());
            if !active_edges.remove(&key) {
                continue;
            }
            if let Some(other) = bond_edge_other(*edge, cand) {
                if atom_degrees[other] <= 2 {
                    changed.push_back(other);
                }
                atom_degrees[other] = atom_degrees[other].saturating_sub(1);
            }
            atom_degrees[cand] = atom_degrees[cand].saturating_sub(1);
        }
    }

    fn find_rings_d3node(&mut self, cand: usize, fragment_rings: &mut Vec<Vec<usize>>) {
        let srings = self.smallest_rings_bfs(cand, None);
        for ring in &srings {
            let invariant = compute_ring_invariant(ring, self.smiles.nodes().len());
            if self.seen_invariants.insert(invariant) {
                fragment_rings.push(canonicalize_cycle(ring));
            }
        }

        if srings.len() >= 3 {
            return;
        }

        let neighbors = self.active_neighbors(cand);
        if neighbors.len() < 3 {
            return;
        }

        if srings.len() == 2 {
            let common = find_common_neighbor(&srings[0], &srings[1], &neighbors);
            if let Some(common_neighbor) = common {
                let trings = self.smallest_rings_bfs(cand, Some(&[common_neighbor]));
                for ring in trings {
                    let invariant = compute_ring_invariant(&ring, self.smiles.nodes().len());
                    if self.seen_invariants.insert(invariant) {
                        fragment_rings.push(canonicalize_cycle(&ring));
                    }
                }
            }
        } else if srings.len() == 1 {
            let [n1, n2, n3] = [neighbors[0], neighbors[1], neighbors[2]];
            let (f1, f2) = if !srings[0].contains(&n1) {
                (n2, n3)
            } else if !srings[0].contains(&n2) {
                (n1, n3)
            } else if !srings[0].contains(&n3) {
                (n1, n2)
            } else {
                return;
            };

            let trings = self.smallest_rings_bfs(cand, Some(&[f2]));
            for ring in trings {
                let invariant = compute_ring_invariant(&ring, self.smiles.nodes().len());
                if self.seen_invariants.insert(invariant) {
                    fragment_rings.push(canonicalize_cycle(&ring));
                }
            }

            let trings = self.smallest_rings_bfs(cand, Some(&[f1]));
            for ring in trings {
                let invariant = compute_ring_invariant(&ring, self.smiles.nodes().len());
                if self.seen_invariants.insert(invariant) {
                    fragment_rings.push(canonicalize_cycle(&ring));
                }
            }
        }
    }

    fn active_neighbors(&self, atom_id: usize) -> Vec<usize> {
        self.ordered_incident_bonds[atom_id]
            .iter()
            .copied()
            .filter_map(|edge| {
                let key = edge_key(edge.source(), edge.target());
                self.active_edges.contains(&key).then(|| bond_edge_other(edge, atom_id)).flatten()
            })
            .collect()
    }

    fn find_ring_connecting_atoms(
        &mut self,
        bridge: [usize; 2],
        fragment_rings: &mut Vec<Vec<usize>>,
    ) -> bool {
        let mut res = Vec::<usize>::new();
        if self.atom_search_bfs(bridge[0], bridge[1], &mut res) {
            let invariant = compute_ring_invariant(&res, self.smiles.nodes().len());
            if self.seen_invariants.insert(invariant) {
                fragment_rings.push(canonicalize_cycle(&res));
                self.mark_ring_membership(&res);
            }
            return true;
        }
        false
    }

    fn atom_search_bfs(&self, start: usize, end: usize, res: &mut Vec<usize>) -> bool {
        let mut bfsq = VecDeque::<Vec<usize>>::new();
        bfsq.push_back(vec![start]);
        let mut total_ops: usize = 0;
        while let Some(path) = bfsq.pop_front() {
            total_ops = total_ops.saturating_add(1);
            if total_ops > self.bfs_budget.max_atom_search_bfs_ops
                || bfsq.len() >= self.bfs_budget.max_queue_size
            {
                self.hit_queue_cutoff.set(true);
                return false;
            }
            let curr = *path.last().unwrap_or(&start);
            for edge in &self.ordered_incident_bonds[curr] {
                let Some(nbr) = bond_edge_other(*edge, curr) else {
                    continue;
                };
                if nbr == end {
                    if curr != start {
                        let mut ring = path.clone();
                        ring.push(end);
                        let invariant = compute_ring_invariant(&ring, self.smiles.nodes().len());
                        if !self.seen_invariants.contains(&invariant) {
                            res.clear();
                            res.extend(ring);
                            return true;
                        }
                    }
                } else if self.ring_atoms[nbr] && !path.contains(&nbr) {
                    let mut next = path.clone();
                    next.push(nbr);
                    bfsq.push_back(next);
                }
            }
        }
        false
    }

    fn smallest_rings_bfs(&mut self, root: usize, forbidden: Option<&[usize]>) -> Vec<Vec<usize>> {
        Self::smallest_rings_bfs_with_scratch(
            &self.ordered_incident_bonds,
            &self.hit_queue_cutoff,
            self.bfs_budget,
            &mut self.bfs_scratch,
            root,
            forbidden,
            &self.active_edges,
        )
    }

    fn smallest_rings_bfs_with_state(
        &mut self,
        root: usize,
        forbidden: Option<&[usize]>,
        active_edges: &HashSet<[usize; 2]>,
    ) -> Vec<Vec<usize>> {
        Self::smallest_rings_bfs_with_scratch(
            &self.ordered_incident_bonds,
            &self.hit_queue_cutoff,
            self.bfs_budget,
            &mut self.bfs_scratch,
            root,
            forbidden,
            active_edges,
        )
    }

    fn smallest_rings_bfs_with_scratch(
        ordered_incident_bonds: &[Vec<BondEdge>],
        hit_queue_cutoff: &Cell<bool>,
        bfs_budget: RdkitBfsBudget,
        scratch: &mut SmallestRingsBfsScratch,
        root: usize,
        forbidden: Option<&[usize]>,
        active_edges: &HashSet<[usize; 2]>,
    ) -> Vec<Vec<usize>> {
        let done = &mut scratch.done;
        let parents = &mut scratch.parents;
        let depths = &mut scratch.depths;
        let bfsq = &mut scratch.queue;

        // Reuse the scratch storage across BFS calls; this path is hot during
        // exact SSSR search on larger ring systems.
        done.fill(WHITE);
        parents.fill(None);
        depths.fill(0);
        if let Some(forbidden) = forbidden {
            for &idx in forbidden {
                done[idx] = BLACK;
            }
        }

        bfsq.clear();
        bfsq.push_back(root);
        let mut rings = Vec::<Vec<usize>>::new();
        let mut cur_size = usize::MAX;

        while let Some(curr) = bfsq.pop_front() {
            if bfsq.len() >= bfs_budget.max_queue_size {
                hit_queue_cutoff.set(true);
                break;
            }
            done[curr] = BLACK;
            let depth = depths[curr].saturating_add(1);
            if depth > cur_size {
                break;
            }

            for edge in &ordered_incident_bonds[curr] {
                let key = edge_key(edge.source(), edge.target());
                if !active_edges.contains(&key) {
                    continue;
                }
                let Some(nbr) = bond_edge_other(*edge, curr) else {
                    continue;
                };
                if done[nbr] == BLACK || parents[curr] == Some(nbr) {
                    continue;
                }
                if done[nbr] == WHITE {
                    parents[nbr] = Some(curr);
                    done[nbr] = GRAY;
                    depths[nbr] = depth;
                    bfsq.push_back(nbr);
                } else {
                    let mut ring = vec![nbr];
                    let mut parent = parents[nbr];
                    while let Some(pid) = parent {
                        if pid == root {
                            break;
                        }
                        ring.push(pid);
                        parent = parents[pid];
                    }
                    ring.insert(0, curr);
                    parent = parents[curr];
                    while let Some(pid) = parent {
                        if ring.contains(&pid) {
                            ring.clear();
                            break;
                        }
                        ring.insert(0, pid);
                        parent = parents[pid];
                    }
                    if ring.len() > 1 {
                        if ring.len() <= cur_size {
                            cur_size = ring.len();
                            rings.push(ring);
                        } else {
                            return rings;
                        }
                    }
                }
            }
        }

        rings
    }

    fn find_first_possible_bridge(&self, bond_scan_limit: usize) -> Option<[usize; 2]> {
        RdkitBridgeCandidateScan::find_first(
            &self.rdkit_bridge_scan_order,
            bond_scan_limit,
            &self.ring_atoms,
            &self.ring_bonds,
        )
    }

    fn find_next_possible_bridge(
        &self,
        bond_scan_limit: usize,
        dead_bonds: &HashSet<[usize; 2]>,
    ) -> Option<[usize; 2]> {
        RdkitBridgeCandidateScan::find_next(
            &self.rdkit_bridge_scan_order,
            bond_scan_limit,
            &self.ring_atoms,
            &self.ring_bonds,
            dead_bonds,
        )
    }

    fn mark_ring_membership(&mut self, ring: &[usize]) {
        for &atom_id in ring {
            self.ring_atoms[atom_id] = true;
        }
        for bond in cycle_edges(ring) {
            self.ring_bonds.insert(bond);
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct RdkitBfsBudget {
    // Mirrors RDKit's RingUtils::MAX_BFSQ_SIZE in Code/GraphMol/FindRings.cpp.
    max_queue_size: usize,
    // Per-call pop cap for atom_search_bfs; its queue holds full paths, so
    // cumulative pops (not concurrent depth) drive wall-clock cost.
    max_atom_search_bfs_ops: usize,
}

impl Default for RdkitBfsBudget {
    fn default() -> Self {
        Self { max_queue_size: 200_000, max_atom_search_bfs_ops: 50_000 }
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
struct RdkitExtraRingResolution;

impl RdkitExtraRingResolution {
    fn promote_extras(rings: &mut Vec<Vec<usize>>, extras: Vec<Vec<usize>>) {
        let bond_rings: Vec<Vec<[usize; 2]>> = rings.iter().map(|ring| cycle_edges(ring)).collect();
        let mut bond_counts = HashMap::<[usize; 2], usize>::new();
        for ring in &bond_rings {
            for &bond in ring {
                *bond_counts.entry(bond).or_insert(0) += 1;
            }
        }

        'extras: for extra_ring in extras {
            let extra_bonds = cycle_edges(&extra_ring);
            for ring in &bond_rings {
                if ring.len() != extra_bonds.len() {
                    continue;
                }

                let mut share_bond = false;
                let mut replaces_all_unique_bonds = true;
                for &bond in ring {
                    let bond_count = bond_counts.get(&bond).copied().unwrap_or(0);
                    if bond_count == 1 || !share_bond {
                        if extra_bonds.contains(&bond) {
                            share_bond = true;
                        } else if bond_count == 1 {
                            replaces_all_unique_bonds = false;
                        }
                    }
                }

                if share_bond && replaces_all_unique_bonds {
                    rings.push(extra_ring);
                    continue 'extras;
                }
            }
        }
    }

    fn split_extras(rings: &mut Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        RdkitLibstdcppRingSizeOrdering::sort_by_size(rings);

        let bond_rings: Vec<HashSet<[usize; 2]>> =
            rings.iter().map(|ring| cycle_edges(ring).into_iter().collect()).collect();
        let mut available = vec![true; rings.len()];
        let mut keep = vec![false; rings.len()];
        let mut union = HashSet::<[usize; 2]>::new();
        let mut extras = Vec::<Vec<usize>>::new();

        for i in 0..rings.len() {
            if bond_rings[i].is_subset(&union) {
                available[i] = false;
            }
            if !available[i] {
                continue;
            }

            union.extend(bond_rings[i].iter().copied());
            keep[i] = true;

            let mut consider = vec![false; rings.len()];
            for j in i + 1..rings.len() {
                if available[j] && bond_rings[j].len() == bond_rings[i].len() {
                    consider[j] = true;
                }
            }

            while consider.iter().any(|flag| *flag) {
                let mut best_j = None;
                let mut best_overlap = 0_usize;
                for j in i + 1..rings.len() {
                    if !consider[j] || !available[j] {
                        continue;
                    }

                    let overlap = bond_rings[j].iter().filter(|bond| union.contains(*bond)).count();
                    if best_j.is_none() || overlap > best_overlap {
                        best_j = Some(j);
                        best_overlap = overlap;
                    }
                }
                let Some(best_j) = best_j else {
                    break;
                };
                consider[best_j] = false;
                if bond_rings[best_j].is_subset(&union) {
                    available[best_j] = false;
                    continue;
                }
                keep[best_j] = true;
                available[best_j] = false;
                union.extend(bond_rings[best_j].iter().copied());
            }
        }

        let mut retained = Vec::<Vec<usize>>::new();
        for (idx, ring) in rings.iter().cloned().enumerate() {
            if keep[idx] {
                retained.push(ring);
            } else {
                extras.push(ring);
            }
        }
        *rings = retained;
        extras
    }
}

// Mirrors RDKit's bridge-candidate scan order: molecule bond order first,
// then the first `nbnds` positions are used as the bridge-search prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RdkitBridgeScanOrder {
    edges: Vec<[usize; 2]>,
}

impl RdkitBridgeScanOrder {
    fn new(smiles: &Smiles<impl SmilesAtomPolicy>) -> Self {
        let mut edges = smiles
            .bond_matrix()
            .sparse_entries()
            .filter_map(|((row, column), entry)| {
                (row < column).then_some((entry.order(), [row, column]))
            })
            .collect::<Vec<_>>();
        edges.sort_unstable_by_key(|(order_key, edge)| (*order_key, *edge));
        Self { edges: edges.into_iter().map(|(_, edge)| edge).collect() }
    }

    fn prefix(&self, bond_scan_limit: usize) -> impl Iterator<Item = [usize; 2]> + '_ {
        self.edges.iter().take(bond_scan_limit).copied()
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
struct RdkitBridgeCandidateScan;

impl RdkitBridgeCandidateScan {
    fn find_first(
        bond_scan_order: &RdkitBridgeScanOrder,
        bond_scan_limit: usize,
        ring_atoms: &[bool],
        ring_bonds: &HashSet<[usize; 2]>,
    ) -> Option<[usize; 2]> {
        // RDKit scans the first `nbnds` bond indices from the molecule when
        // looking for bridge candidates. `BondEntry::order()` mirrors that
        // post-reassignment bond order, so we preserve the same prefix here.
        bond_scan_order
            .prefix(bond_scan_limit)
            .find(|&edge| ring_atoms[edge[0]] && ring_atoms[edge[1]] && !ring_bonds.contains(&edge))
    }

    fn find_next(
        bond_scan_order: &RdkitBridgeScanOrder,
        bond_scan_limit: usize,
        ring_atoms: &[bool],
        ring_bonds: &HashSet<[usize; 2]>,
        dead_bonds: &HashSet<[usize; 2]>,
    ) -> Option<[usize; 2]> {
        for edge in bond_scan_order.prefix(bond_scan_limit) {
            if dead_bonds.contains(&edge) {
                continue;
            }
            if ring_atoms[edge[0]] && ring_atoms[edge[1]] && !ring_bonds.contains(&edge) {
                return Some(edge);
            }
        }
        None
    }
}

// Mirrors RDKit's `computeRingInvariant()`: duplicate suppression keys rings by
// the set of participating atoms, not by a particular traversal order.
type RdkitRingInvariant = Vec<u64>;

fn edge_key(node_a: usize, node_b: usize) -> [usize; 2] {
    if node_a < node_b { [node_a, node_b] } else { [node_b, node_a] }
}

fn cyclomatic_ring_count(bond_count: usize, atom_count: usize) -> usize {
    bond_count.checked_sub(atom_count).and_then(|delta| delta.checked_add(1)).unwrap_or(0)
}

fn compute_ring_invariant(ring: &[usize], atom_count: usize) -> RdkitRingInvariant {
    let mut words = vec![0_u64; atom_count.saturating_add(63) / 64];
    for &atom_id in ring {
        let word = atom_id / 64;
        let bit = atom_id % 64;
        words[word] |= 1_u64 << bit;
    }
    words
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
struct RdkitIncidentBondOrdering;

impl RdkitIncidentBondOrdering {
    fn for_node(smiles: &Smiles<impl SmilesAtomPolicy>, atom_id: usize) -> Vec<BondEdge> {
        let mut edges = smiles
            .bond_matrix()
            .sparse_row(atom_id)
            .zip(smiles.bond_matrix().sparse_row_values_ref(atom_id))
            .map(|(other, entry)| (entry.order(), entry.to_bond_edge(atom_id, other)))
            .collect::<Vec<_>>();
        edges.sort_unstable_by_key(|(order_key, edge)| {
            (*order_key, bond_edge_other(*edge, atom_id).unwrap_or(atom_id))
        });
        edges.into_iter().map(|(_, edge)| edge).collect()
    }
}

#[cfg(test)]
fn ordered_edges_for_node(smiles: &Smiles<impl SmilesAtomPolicy>, atom_id: usize) -> Vec<BondEdge> {
    RdkitIncidentBondOrdering::for_node(smiles, atom_id)
}

// Mirrors libstdc++'s equal-size ring ordering used by RDKit's `std::sort`
// path.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
struct RdkitLibstdcppRingSizeOrdering;

impl RdkitLibstdcppRingSizeOrdering {
    const SORT_THRESHOLD: usize = 16;

    fn sort_by_size(rings: &mut [Vec<usize>]) {
        if rings.len() <= 1 {
            return;
        }

        let depth_limit = 2 * (usize::BITS as usize - 1 - rings.len().leading_zeros() as usize);
        Self::introsort_loop(rings, 0, rings.len(), depth_limit);
        Self::final_insertion_sort(rings, 0, rings.len());
    }

    fn introsort_loop(
        rings: &mut [Vec<usize>],
        first: usize,
        mut last: usize,
        mut depth_limit: usize,
    ) {
        while last - first > Self::SORT_THRESHOLD {
            if depth_limit == 0 {
                Self::partial_sort_by_size(rings, first, last);
                return;
            }
            depth_limit -= 1;
            let cut = Self::unguarded_partition_pivot(rings, first, last);
            Self::introsort_loop(rings, cut, last, depth_limit);
            last = cut;
        }
    }

    fn partial_sort_by_size(rings: &mut [Vec<usize>], first: usize, last: usize) {
        rings[first..last].sort_by(|left, right| Self::cmp_by_size(left, right));
    }

    fn unguarded_partition_pivot(rings: &mut [Vec<usize>], first: usize, last: usize) -> usize {
        let mid = first + (last - first) / 2;
        Self::move_median_to_first(rings, first, first + 1, mid, last - 1);
        Self::unguarded_partition(rings, first + 1, last, first)
    }

    fn move_median_to_first(rings: &mut [Vec<usize>], result: usize, a: usize, b: usize, c: usize) {
        if Self::less_by_size(&rings[a], &rings[b]) {
            if Self::less_by_size(&rings[b], &rings[c]) {
                rings.swap(result, b);
            } else if Self::less_by_size(&rings[a], &rings[c]) {
                rings.swap(result, c);
            } else {
                rings.swap(result, a);
            }
        } else if Self::less_by_size(&rings[a], &rings[c]) {
            rings.swap(result, a);
        } else if Self::less_by_size(&rings[b], &rings[c]) {
            rings.swap(result, c);
        } else {
            rings.swap(result, b);
        }
    }

    fn unguarded_partition(
        rings: &mut [Vec<usize>],
        mut first: usize,
        mut last: usize,
        pivot: usize,
    ) -> usize {
        loop {
            while Self::less_by_size(&rings[first], &rings[pivot]) {
                first += 1;
            }
            last -= 1;
            while Self::less_by_size(&rings[pivot], &rings[last]) {
                last -= 1;
            }
            if first >= last {
                return first;
            }
            rings.swap(first, last);
            first += 1;
        }
    }

    fn final_insertion_sort(rings: &mut [Vec<usize>], first: usize, last: usize) {
        if last - first > Self::SORT_THRESHOLD {
            Self::insertion_sort(rings, first, first + Self::SORT_THRESHOLD);
            Self::unguarded_insertion_sort(rings, first + Self::SORT_THRESHOLD, last);
        } else {
            Self::insertion_sort(rings, first, last);
        }
    }

    fn insertion_sort(rings: &mut [Vec<usize>], first: usize, last: usize) {
        if first == last {
            return;
        }

        for index in first + 1..last {
            if Self::less_by_size(&rings[index], &rings[first]) {
                rings[first..=index].rotate_right(1);
            } else {
                Self::unguarded_linear_insert(rings, index);
            }
        }
    }

    fn unguarded_insertion_sort(rings: &mut [Vec<usize>], first: usize, last: usize) {
        for index in first..last {
            Self::unguarded_linear_insert(rings, index);
        }
    }

    fn unguarded_linear_insert(rings: &mut [Vec<usize>], last: usize) {
        let ring_size = rings[last].len();
        let mut insert_at = last;
        while ring_size < rings[insert_at - 1].len() {
            insert_at -= 1;
        }
        rings[insert_at..=last].rotate_right(1);
    }

    fn cmp_by_size(left: &[usize], right: &[usize]) -> Ordering {
        left.len().cmp(&right.len())
    }

    fn less_by_size(left: &[usize], right: &[usize]) -> bool {
        left.len() < right.len()
    }
}

fn connected_fragments(smiles: &Smiles<impl SmilesAtomPolicy>) -> Vec<Vec<usize>> {
    let mut seen = vec![false; smiles.nodes().len()];
    let mut fragments = Vec::<Vec<usize>>::new();

    for start in 0..smiles.nodes().len() {
        if seen[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut fragment = Vec::<usize>::new();
        seen[start] = true;

        while let Some(node) = stack.pop() {
            fragment.push(node);
            for edge in smiles.edges_for_node(node) {
                let Some(other) = bond_edge_other(edge, node) else {
                    continue;
                };
                if !seen[other] {
                    seen[other] = true;
                    stack.push(other);
                }
            }
        }
        fragment.sort_unstable();
        fragments.push(fragment);
    }

    fragments
}

fn find_common_neighbor(left: &[usize], right: &[usize], neighbors: &[usize]) -> Option<usize> {
    neighbors.iter().copied().find(|neighbor| left.contains(neighbor) && right.contains(neighbor))
}

fn fast_find_rings(
    ordered_incident_bonds: &[Vec<BondEdge>],
    fragment: &[usize],
) -> Vec<Vec<usize>> {
    let mut seen = HashSet::<Vec<usize>>::new();
    let mut rings = Vec::<Vec<usize>>::new();
    let mut colors = vec![0_u8; ordered_incident_bonds.len()];

    for &start in fragment {
        if colors[start] != WHITE {
            continue;
        }
        if ordered_incident_bonds[start].len() < 2 {
            colors[start] = BLACK;
            continue;
        }
        let mut traversal = Vec::<usize>::new();
        dfs_fast_find_rings(
            ordered_incident_bonds,
            start,
            None,
            &mut colors,
            &mut traversal,
            &mut rings,
            &mut seen,
        );
    }

    rings
}

fn dfs_fast_find_rings(
    ordered_incident_bonds: &[Vec<BondEdge>],
    atom: usize,
    from_atom: Option<usize>,
    colors: &mut [u8],
    traversal: &mut Vec<usize>,
    rings: &mut Vec<Vec<usize>>,
    seen: &mut HashSet<Vec<usize>>,
) {
    colors[atom] = GRAY;
    traversal.push(atom);

    for &edge in &ordered_incident_bonds[atom] {
        let Some(nbr) = bond_edge_other(edge, atom) else {
            continue;
        };
        if colors[nbr] == WHITE {
            if ordered_incident_bonds[nbr].len() < 2 {
                colors[nbr] = BLACK;
            } else {
                dfs_fast_find_rings(
                    ordered_incident_bonds,
                    nbr,
                    Some(atom),
                    colors,
                    traversal,
                    rings,
                    seen,
                );
            }
        } else if colors[nbr] == GRAY
            && from_atom != Some(nbr)
            && let Some(pos) = traversal.iter().rposition(|&node| node == nbr)
        {
            let mut ring = traversal[pos..].to_vec();
            let inv = canonicalize_cycle(&ring);
            if seen.insert(inv.clone()) {
                ring = inv;
                rings.push(ring);
            }
        }
    }

    colors[atom] = BLACK;
    traversal.pop();
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use hashbrown::HashSet;

    use super::{
        RdkitBfsBudget, RingSearchState, compute_ring_invariant, cyclomatic_ring_count,
        symmetrize_sssr_with_ring_membership, symmetrize_sssr_with_status,
        try_disjoint_simple_cycles_from_ring_membership,
        try_simple_cycle_blocks_from_ring_membership,
    };
    use crate::smiles::Smiles;

    type ExactStageState = (
        Vec<Vec<usize>>,
        Vec<bool>,
        HashSet<[usize; 2]>,
        Vec<usize>,
        usize,
        usize,
        Option<[usize; 2]>,
        usize,
    );

    const CID_58882885: &str = "CC1=C2CC3=C(C4=C5C=C3[C@@H]6C2=CC7=C1CC8=C(C9=C1C=C8[C@@H]7CCC[C@@H]2C3=C7CC8=C2C=C2C%10CCCC%11C%12=CC%13=C%14CC(=C7C)C(=C3)[C@@H]%13CCC[C@H]3C7=C%13CC%15=C3C=C3C(CCCC%16C%17=CC(=C(C4)C(=C%17CC4=C%16C=C%16[C@H](CCC6)C(=C7)C(=C%13C)CC%16=C4C)C)C5CCCC1C1=CC%10=C(CC2=C8C)C(=C1C9)C)C1=CC%11=C(CC%12=C%14C)C(=C1CC3=C%15C)C)C)C";
    const CID_101460172: &str = "C1COCCOCC2(COCCOCCO1)COC3=CC4=C(C=C3)C5=NC6=C7C=CC8=CC7=C(N6)N=C9C1=C3C=CC(=C1)OCC1(COCCOCCOCCOCCOC1)COC1=CC6=C(C=C1)C1=NC6=NC6=C7C=C(C=CC7=C(N6)N=C6C7=C(C=C(C=C7)OC2)C(=N6)NC2=NC(=N1)C1=C2C=C(C=C1)OCC1(COCCOCCOCCOCCOC1)COC1=CC2=C(C=C1)C(=NC3=N9)N=C2NC4=N5)OCC1(COCCOCCOCCOCCOC1)CO8";

    #[test]
    fn ring_invariant_depends_on_atom_set_not_cycle_start_or_direction() {
        let left = compute_ring_invariant(&[1, 2, 3, 7], 16);
        let right = compute_ring_invariant(&[3, 2, 1, 7], 16);
        let different = compute_ring_invariant(&[1, 2, 3, 8], 16);

        assert_eq!(left, right);
        assert_ne!(left, different);
    }

    #[test]
    fn symm_sssr_status_is_complete_for_benzene() {
        let smiles: Smiles = "C1=CC=CC=C1".parse().expect("valid benzene");
        let result = symmetrize_sssr_with_status(&smiles);

        assert!(result.status().is_complete());
        assert!(!result.status().used_fallback());
        assert!(!result.status().hit_queue_cutoff());
        assert_eq!(result.cycles(), &[vec![0, 1, 2, 3, 4, 5]]);
    }

    #[test]
    fn disjoint_simple_cycle_shortcut_handles_biphenyl_ring_subgraph() {
        let smiles: Smiles = "C1=CC=CC=C1C2=CC=CC=C2".parse().expect("valid biphenyl");
        let ring_membership = smiles.ring_membership();

        let cycles =
            try_disjoint_simple_cycles_from_ring_membership(&ring_membership, smiles.nodes().len())
                .expect("disjoint simple cycles should short-circuit");

        assert_eq!(cycles, vec![vec![0, 1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10, 11]]);
        let result = symmetrize_sssr_with_ring_membership(&smiles, &ring_membership);
        assert!(result.status().is_complete());
        assert_eq!(result.cycles(), cycles);
    }

    #[test]
    fn disjoint_simple_cycle_shortcut_rejects_fused_ring_subgraph() {
        let smiles: Smiles = "C1=CC2=CC=CC=C2C=C1".parse().expect("valid naphthalene");
        let ring_membership = smiles.ring_membership();

        assert!(
            try_disjoint_simple_cycles_from_ring_membership(&ring_membership, smiles.nodes().len())
                .is_none()
        );
    }

    #[test]
    fn disjoint_simple_cycle_shortcut_rejects_spiro_ring_subgraph() {
        let smiles: Smiles = "C1CCC2(CC1)CCC2".parse().expect("valid spiro system");
        let ring_membership = smiles.ring_membership();

        assert!(
            try_disjoint_simple_cycles_from_ring_membership(&ring_membership, smiles.nodes().len())
                .is_none()
        );
    }

    #[test]
    fn simple_cycle_block_shortcut_handles_spiro_ring_subgraph() {
        let smiles: Smiles = "C1CCC2(CC1)CCC2".parse().expect("valid spiro system");
        let ring_membership = smiles.ring_membership();

        let cycles =
            try_simple_cycle_blocks_from_ring_membership(&ring_membership, smiles.nodes().len())
                .expect("spiro ring blocks should short-circuit");

        assert_eq!(cycles, vec![vec![0, 1, 2, 3, 4, 5], vec![3, 6, 7, 8]]);
    }

    #[test]
    fn simple_cycle_block_shortcut_rejects_fused_ring_subgraph() {
        let smiles: Smiles = "C1=CC2=CC=CC=C2C=C1".parse().expect("valid naphthalene");
        let ring_membership = smiles.ring_membership();

        assert!(
            try_simple_cycle_blocks_from_ring_membership(&ring_membership, smiles.nodes().len())
                .is_none()
        );
    }

    #[test]
    fn symm_sssr_forced_bfs_budget_marks_benzene_incomplete_and_hits_queue_cutoff() {
        let smiles: Smiles = "C1=CC=CC=C1".parse().expect("valid benzene");
        let result = RingSearchState::new_with_bfs_budget(
            &smiles,
            RdkitBfsBudget { max_queue_size: 0, max_atom_search_bfs_ops: usize::MAX },
        )
        .symmetrize_sssr_with_status();

        assert!(!result.status().is_complete());
        assert!(result.status().used_fallback());
        assert!(result.status().hit_queue_cutoff());
        assert_eq!(result.cycles(), symmetrize_sssr_with_status(&smiles).cycles());
    }

    #[test]
    fn symm_sssr_status_marks_large_polycycle_frontier_case_as_incomplete() {
        let smiles: Smiles = CID_58882885.parse().expect("valid large-polycycle frontier case");
        let result = symmetrize_sssr_with_status(&smiles);

        assert!(!result.status().is_complete());
        assert!(result.status().used_fallback());
        assert!(!result.status().hit_queue_cutoff());
        assert_eq!(result.cycles(), symmetrize_sssr_with_status(&smiles).cycles());
    }

    #[test]
    fn symm_sssr_status_marks_cid_101460172_as_incomplete_without_queue_cutoff() {
        let smiles: Smiles = CID_101460172.parse().expect("valid fallback frontier case");
        let result = symmetrize_sssr_with_status(&smiles);

        assert!(!result.status().is_complete());
        assert!(result.status().used_fallback());
        assert!(!result.status().hit_queue_cutoff());
        assert_eq!(result.cycles(), symmetrize_sssr_with_status(&smiles).cycles());
    }

    #[test]
    fn large_polycycle_exact_stage_still_undercounts_before_fallback() {
        let smiles: Smiles = CID_58882885.parse().expect("valid large-polycycle frontier case");
        let fragment = RingSearchState::new(&smiles).fragments[0].clone();
        let (ring_count, expected_rings, first_bridge, bridge_edges) =
            post_bridge_ring_count_for_fragment(&smiles, &fragment);

        assert!(first_bridge.is_none());
        assert!(bridge_edges.is_empty());
        assert!(
            ring_count < expected_rings,
            "exact-stage search unexpectedly reached the cyclomatic target before fallback"
        );
    }

    #[test]
    fn large_polycycle_exact_stage_is_short_by_one_ring_before_fallback() {
        let smiles: Smiles = CID_58882885.parse().expect("valid large-polycycle frontier case");
        let (exact_rings, ..) = exact_stage_state(&smiles);
        let exact_result = symmetrize_sssr_with_status(&smiles);
        assert!(exact_result.status().used_fallback());
        assert_eq!(exact_result.cycles().len(), exact_rings.len() + 1);
    }

    #[test]
    fn symm_sssr_with_ring_membership_matches_full_search_for_substituted_naphthalene() {
        let smiles: Smiles =
            "Cc1cccc2ccccc12".parse().expect("valid substituted fused-ring system");
        let ring_membership = smiles.ring_membership();

        let shortcut_result = symmetrize_sssr_with_ring_membership(&smiles, &ring_membership);
        let full_result = symmetrize_sssr_with_status(&smiles);

        assert_eq!(shortcut_result.status(), full_result.status());
        assert_eq!(shortcut_result.cycles(), full_result.cycles());
    }

    #[test]
    fn symm_sssr_with_ring_membership_matches_full_search_for_large_polycycle_frontier_case() {
        let smiles: Smiles = CID_58882885.parse().expect("valid large-polycycle frontier case");
        let ring_membership = smiles.ring_membership();

        let shortcut_result = symmetrize_sssr_with_ring_membership(&smiles, &ring_membership);
        let full_result = symmetrize_sssr_with_status(&smiles);

        assert_eq!(shortcut_result.status(), full_result.status());
        assert_eq!(shortcut_result.cycles(), full_result.cycles());
    }

    #[test]
    fn symm_sssr_with_ring_membership_matches_full_search_for_spiro_system() {
        let smiles: Smiles = "C1CCC2(CC1)CCC2".parse().expect("valid spiro system");
        let ring_membership = smiles.ring_membership();

        let shortcut_result = symmetrize_sssr_with_ring_membership(&smiles, &ring_membership);
        let full_result = symmetrize_sssr_with_status(&smiles);

        assert_eq!(shortcut_result.status(), full_result.status());
        assert_eq!(shortcut_result.cycles(), full_result.cycles());
    }

    fn exact_stage_state(smiles: &Smiles) -> ExactStageState {
        let mut state = RingSearchState::new(smiles);
        let fragment = state.fragments[0].clone();
        exact_stage_state_for_fragment(smiles, &fragment, &mut state)
    }

    fn exact_stage_state_for_fragment(
        smiles: &Smiles,
        fragment: &[usize],
        state: &mut RingSearchState<'_>,
    ) -> ExactStageState {
        let mut changed = alloc::collections::VecDeque::<usize>::new();
        for &atom_id in fragment {
            if state.atom_degrees[atom_id] < 2 {
                changed.push_back(atom_id);
            }
        }

        let mut done_atoms = vec![false; state.smiles.nodes().len()];
        let mut atoms_done = 0_usize;
        let mut fragment_rings = Vec::<Vec<usize>>::new();

        while atoms_done + 2 < fragment.len() {
            while let Some(cand) = changed.pop_front() {
                if done_atoms[cand] {
                    continue;
                }
                done_atoms[cand] = true;
                atoms_done += 1;
                state.trim_bonds(cand, &mut changed);
            }

            let mut d2nodes = Vec::<usize>::new();
            state.pick_d2_nodes(fragment, &mut d2nodes);
            if d2nodes.is_empty() {
                let Some(cand) = fragment
                    .iter()
                    .copied()
                    .find(|&atom_id| state.atom_degrees[atom_id] == 3 && !done_atoms[atom_id])
                else {
                    break;
                };
                state.find_rings_d3node(cand, &mut fragment_rings);
                done_atoms[cand] = true;
                atoms_done += 1;
                state.trim_bonds(cand, &mut changed);
            } else {
                state.find_rings_d2nodes(&d2nodes, &mut fragment_rings);
                for &d2node in &d2nodes {
                    if !done_atoms[d2node] {
                        done_atoms[d2node] = true;
                        atoms_done += 1;
                        state.trim_bonds(d2node, &mut changed);
                    }
                }
            }
        }

        let bond_count = fragment
            .iter()
            .flat_map(|&atom_id| super::ordered_edges_for_node(smiles, atom_id))
            .map(|edge| super::edge_key(edge.source(), edge.target()))
            .collect::<HashSet<_>>()
            .len();
        let expected_rings = cyclomatic_ring_count(bond_count, fragment.len());
        let first_bridge = state.find_first_possible_bridge(bond_count);
        let active_edges = state.active_edges.len();

        (
            fragment_rings,
            state.ring_atoms.clone(),
            state.ring_bonds.clone(),
            fragment.to_vec(),
            expected_rings,
            bond_count,
            first_bridge,
            active_edges,
        )
    }

    fn post_bridge_ring_count_for_fragment(
        smiles: &Smiles,
        fragment: &[usize],
    ) -> (usize, usize, Option<[usize; 2]>, Vec<[usize; 2]>) {
        let mut state = RingSearchState::new(smiles);
        let mut changed = alloc::collections::VecDeque::<usize>::new();
        for &atom_id in fragment {
            if state.atom_degrees[atom_id] < 2 {
                changed.push_back(atom_id);
            }
        }

        let mut done_atoms = vec![false; state.smiles.nodes().len()];
        let mut atoms_done = 0_usize;
        let mut fragment_rings = Vec::<Vec<usize>>::new();

        while atoms_done + 2 < fragment.len() {
            while let Some(cand) = changed.pop_front() {
                if done_atoms[cand] {
                    continue;
                }
                done_atoms[cand] = true;
                atoms_done += 1;
                state.trim_bonds(cand, &mut changed);
            }

            let mut d2nodes = Vec::<usize>::new();
            state.pick_d2_nodes(fragment, &mut d2nodes);
            if d2nodes.is_empty() {
                let Some(cand) = fragment
                    .iter()
                    .copied()
                    .find(|&atom_id| state.atom_degrees[atom_id] == 3 && !done_atoms[atom_id])
                else {
                    break;
                };
                state.find_rings_d3node(cand, &mut fragment_rings);
                done_atoms[cand] = true;
                atoms_done += 1;
                state.trim_bonds(cand, &mut changed);
            } else {
                state.find_rings_d2nodes(&d2nodes, &mut fragment_rings);
                for &d2node in &d2nodes {
                    if !done_atoms[d2node] {
                        done_atoms[d2node] = true;
                        atoms_done += 1;
                        state.trim_bonds(d2node, &mut changed);
                    }
                }
            }
        }

        let bond_count = fragment
            .iter()
            .flat_map(|&atom_id| super::ordered_edges_for_node(smiles, atom_id))
            .map(|edge| super::edge_key(edge.source(), edge.target()))
            .collect::<HashSet<_>>()
            .len();
        let expected_rings = cyclomatic_ring_count(bond_count, fragment.len());
        let first_bridge = state.find_first_possible_bridge(bond_count);
        let mut bridge_edges = Vec::<[usize; 2]>::new();

        if fragment_rings.len() < expected_rings
            && let Some(first_possible) = first_bridge
        {
            let mut dead_bonds = HashSet::<[usize; 2]>::new();
            let mut possible = vec![first_possible];
            while let Some(edge) = possible.pop() {
                bridge_edges.push(edge);
                if !state.find_ring_connecting_atoms(edge, &mut fragment_rings) {
                    dead_bonds.insert(edge);
                }
                possible.clear();
                if let Some(next) = state.find_next_possible_bridge(bond_count, &dead_bonds) {
                    possible.push(next);
                }
            }
        }

        (fragment_rings.len(), expected_rings, first_bridge, bridge_edges)
    }

    #[test]
    fn libstdcpp_size_sort_matches_cid_163335646_cpp_probe_order() {
        let mut rings = vec![
            vec![7, 8, 9],
            vec![8, 9, 10, 71, 72, 73],
            vec![8, 9, 19, 20, 77],
            vec![8, 73, 74, 75, 76, 77],
            vec![9, 10, 11, 17, 18, 19],
            vec![10, 11, 12, 47, 48, 70, 71],
            vec![11, 12, 13, 16, 17],
            vec![12, 13, 14, 44, 45, 46, 47],
            vec![13, 14, 15, 16],
            vec![15, 16, 17, 18, 43, 42],
            vec![18, 19, 20, 21, 22, 43],
            vec![20, 21, 60, 61, 76, 77],
            vec![21, 22, 23, 24, 59, 60],
            vec![22, 23, 41, 42, 43],
            vec![23, 24, 25, 39, 40, 41],
            vec![24, 25, 26, 57, 58, 59],
            vec![25, 26, 27, 40, 39],
            vec![26, 27, 28, 29, 30, 56, 57],
            vec![27, 28, 36, 37, 38, 39, 40],
            vec![28, 29, 33, 34, 35, 36],
            vec![29, 30, 31, 32, 33],
            vec![30, 31, 54, 55, 56],
            vec![31, 32, 51, 52, 53, 54],
            vec![32, 33, 34, 49, 50, 51],
            vec![34, 35, 46, 47, 48, 49],
            vec![35, 36, 37, 45, 46],
            vec![37, 38, 44, 45],
            vec![48, 49, 50, 69, 70],
            vec![50, 51, 52, 67, 68, 69],
            vec![52, 53, 65, 66, 67],
            vec![53, 54, 55, 63, 64, 65],
            vec![55, 56, 57, 58, 62, 63],
            vec![58, 59, 60, 61, 62],
            vec![61, 62, 63, 64, 75, 76],
            vec![64, 65, 66, 74, 75],
            vec![66, 67, 68, 72, 73, 74],
            vec![68, 69, 70, 71, 72],
            vec![14, 15, 42, 41, 40, 39, 38, 44],
            vec![78, 79, 80, 81, 82, 83],
        ];
        super::RdkitLibstdcppRingSizeOrdering::sort_by_size(&mut rings);
        assert_eq!(
            rings,
            vec![
                vec![7, 8, 9],
                vec![37, 38, 44, 45],
                vec![13, 14, 15, 16],
                vec![25, 26, 27, 40, 39],
                vec![48, 49, 50, 69, 70],
                vec![22, 23, 41, 42, 43],
                vec![52, 53, 65, 66, 67],
                vec![35, 36, 37, 45, 46],
                vec![58, 59, 60, 61, 62],
                vec![29, 30, 31, 32, 33],
                vec![11, 12, 13, 16, 17],
                vec![64, 65, 66, 74, 75],
                vec![30, 31, 54, 55, 56],
                vec![68, 69, 70, 71, 72],
                vec![8, 9, 19, 20, 77],
                vec![34, 35, 46, 47, 48, 49],
                vec![53, 54, 55, 63, 64, 65],
                vec![50, 51, 52, 67, 68, 69],
                vec![28, 29, 33, 34, 35, 36],
                vec![55, 56, 57, 58, 62, 63],
                vec![61, 62, 63, 64, 75, 76],
                vec![66, 67, 68, 72, 73, 74],
                vec![78, 79, 80, 81, 82, 83],
                vec![32, 33, 34, 49, 50, 51],
                vec![31, 32, 51, 52, 53, 54],
                vec![24, 25, 26, 57, 58, 59],
                vec![23, 24, 25, 39, 40, 41],
                vec![21, 22, 23, 24, 59, 60],
                vec![20, 21, 60, 61, 76, 77],
                vec![18, 19, 20, 21, 22, 43],
                vec![15, 16, 17, 18, 43, 42],
                vec![9, 10, 11, 17, 18, 19],
                vec![8, 73, 74, 75, 76, 77],
                vec![8, 9, 10, 71, 72, 73],
                vec![27, 28, 36, 37, 38, 39, 40],
                vec![26, 27, 28, 29, 30, 56, 57],
                vec![12, 13, 14, 44, 45, 46, 47],
                vec![10, 11, 12, 47, 48, 70, 71],
                vec![14, 15, 42, 41, 40, 39, 38, 44],
            ]
        );
    }
}
