use alloc::{collections::VecDeque, vec::Vec};

use geometric_traits::traits::{SparseMatrix2D, SparseValuedMatrix2DRef, SparseValuedMatrixRef};

use super::chirality::stereo_chirality_normal_form;
use crate::{
    atom::bracketed::chirality::Chirality,
    bond::Bond,
    smiles::{
        Smiles, StereoNeighbor,
        double_bond_stereo::DoubleBondStereoConfig,
        stereo::{DirectionalParityConstraint, directional_override_rows_from_parity_constraints},
    },
};

#[derive(Debug, Clone)]
pub(super) struct AtomBasedDoubleBondNormalization {
    pub(super) override_rows: Vec<Vec<(usize, Bond)>>,
    pub(super) semantic_endpoints: Vec<bool>,
    pub(super) clear_chirality: Vec<bool>,
}

#[derive(Debug, Clone)]
pub(super) struct NonSemanticDirectionalNormalization {
    pub(super) override_rows: Vec<Vec<(usize, Bond)>>,
}

#[derive(Debug, Clone, Copy)]
struct AtomBasedDoubleBondSide {
    endpoint: usize,
    reference_atom: usize,
    reference_bond_is_up: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum StereoSubstituentIdentityKey {
    ExplicitHydrogen,
    Atom(AtomBasedSubstituentPriorityKey),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct AtomBasedSubstituentPriorityKey {
    atomic_number: u8,
    isotope_mass_number: Option<u16>,
    charge: i8,
    aromatic: bool,
    bond_order_to_endpoint: u8,
    rooted_class: usize,
    refined_class: usize,
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    #[allow(clippy::too_many_lines)]
    pub(super) fn atom_based_double_bond_normalization(
        &self,
        preorder_indices: &[usize],
        rooted_classes: &[usize],
        refined_classes: &[usize],
    ) -> AtomBasedDoubleBondNormalization {
        let mut records = Vec::new();
        let mut clear_chirality = vec![false; self.nodes().len()];

        for ((row, column), entry) in self.bond_matrix().sparse_entries() {
            if row >= column || !non_aromatic_double_bond(*entry) {
                continue;
            }
            if !self.atom_based_double_bond_supports_semantic_stereo(row, column) {
                continue;
            }
            let Some((side_a, side_b)) =
                self.atom_based_double_bond_sides(row, column, rooted_classes, refined_classes)
            else {
                continue;
            };
            let config = if side_a.reference_bond_is_up == side_b.reference_bond_is_up {
                DoubleBondStereoConfig::E
            } else {
                DoubleBondStereoConfig::Z
            };
            records.push((row, column, side_a, side_b, config));
            clear_chirality[row] = true;
            clear_chirality[column] = true;
        }

        let mut semantic_endpoints = vec![false; self.nodes().len()];
        if records.is_empty() {
            return AtomBasedDoubleBondNormalization {
                override_rows: vec![Vec::new(); self.nodes().len()],
                semantic_endpoints,
                clear_chirality,
            };
        }

        let constraints = records
            .iter()
            .map(|record| {
                DirectionalParityConstraint {
                    left_edge_key: crate::smiles::edge_key(
                        record.2.endpoint,
                        record.2.reference_atom,
                    ),
                    right_edge_key: crate::smiles::edge_key(
                        record.3.endpoint,
                        record.3.reference_atom,
                    ),
                    same_parity: matches!(record.4, DoubleBondStereoConfig::E),
                }
            })
            .collect::<Vec<_>>();

        for (_row, _column, side_a, side_b, _config) in &records {
            semantic_endpoints[side_a.endpoint] = true;
            semantic_endpoints[side_b.endpoint] = true;
        }

        AtomBasedDoubleBondNormalization {
            override_rows: directional_override_rows_from_parity_constraints(
                self.nodes().len(),
                &constraints,
                preorder_indices,
            ),
            semantic_endpoints,
            clear_chirality,
        }
    }

    pub(super) fn non_semantic_directional_normalization(
        &self,
        rooted_classes: &[usize],
        refined_classes: &[usize],
    ) -> NonSemanticDirectionalNormalization {
        let ring_membership = self.ring_membership();
        let mut rows = vec![Vec::new(); self.nodes().len()];

        for ((endpoint_a, endpoint_b), entry) in self.bond_matrix().sparse_entries() {
            if endpoint_a >= endpoint_b || !non_aromatic_double_bond(*entry) {
                continue;
            }
            if !non_semantic_double_bond_supports_semantic_stereo(self, endpoint_a, endpoint_b) {
                continue;
            }

            let directional_neighbors_a =
                non_semantic_double_bond_directional_neighbors(self, endpoint_a, endpoint_b);
            if directional_neighbors_a.is_empty() {
                continue;
            }

            let directional_neighbors_b =
                non_semantic_double_bond_directional_neighbors(self, endpoint_b, endpoint_a);
            if directional_neighbors_b.is_empty()
                || ring_membership.contains_edge(endpoint_a, endpoint_b)
            {
                continue;
            }

            let side_a = non_semantic_double_bond_has_unique_reference_substituent(
                self,
                endpoint_a,
                endpoint_b,
                rooted_classes,
                refined_classes,
            );
            let side_b = non_semantic_double_bond_has_unique_reference_substituent(
                self,
                endpoint_b,
                endpoint_a,
                rooted_classes,
                refined_classes,
            );

            if side_a.is_some() && side_b.is_some() {
                continue;
            }

            for directional_neighbor in directional_neighbors_a {
                let (left, right) = crate::smiles::edge_key(endpoint_a, directional_neighbor);
                rows[left].push((right, Bond::Single));
            }
            for directional_neighbor in directional_neighbors_b {
                let (left, right) = crate::smiles::edge_key(endpoint_b, directional_neighbor);
                rows[left].push((right, Bond::Single));
            }
        }

        for row in &mut rows {
            row.sort_unstable_by_key(|&(neighbor, _)| neighbor);
            row.dedup_by_key(|entry| entry.0);
        }

        NonSemanticDirectionalNormalization { override_rows: rows }
    }

    fn atom_based_double_bond_supports_semantic_stereo(
        &self,
        node_a: usize,
        node_b: usize,
    ) -> bool {
        self.atom_based_non_single_family_bond_count(node_a) == 1
            && self.atom_based_non_single_family_bond_count(node_b) == 1
            && !self.atom_based_double_bond_is_in_cycle(node_a, node_b)
            && self.atom_based_endpoint_supports_semantic_stereo(node_a, node_b)
            && self.atom_based_endpoint_supports_semantic_stereo(node_b, node_a)
    }

    fn atom_based_non_single_family_bond_count(&self, node_id: usize) -> usize {
        self.bond_matrix()
            .sparse_row_values_ref(node_id)
            .filter(|entry| non_single_family_bond(**entry))
            .count()
    }

    fn atom_based_double_bond_is_in_cycle(&self, node_a: usize, node_b: usize) -> bool {
        let mut queue = VecDeque::from([node_a]);
        let mut seen = vec![false; self.nodes().len()];
        seen[node_a] = true;

        while let Some(current) = queue.pop_front() {
            for neighbor in self.bond_matrix().sparse_row(current) {
                if (current == node_a && neighbor == node_b)
                    || (current == node_b && neighbor == node_a)
                {
                    continue;
                }
                if neighbor == node_b {
                    return true;
                }
                if !seen[neighbor] {
                    seen[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        false
    }

    fn atom_based_endpoint_supports_semantic_stereo(
        &self,
        endpoint: usize,
        opposite_endpoint: usize,
    ) -> bool {
        let atom = self.nodes()[endpoint];
        let parsed_neighbors = self.parsed_stereo_neighbors_row(endpoint);
        matches!(
            stereo_chirality_normal_form(self, endpoint, atom.chirality(), parsed_neighbors),
            Some(Chirality::At | Chirality::AtAt | Chirality::TH(1 | 2))
        ) && parsed_neighbors
            .iter()
            .filter(|neighbor| **neighbor != StereoNeighbor::Atom(opposite_endpoint))
            .count()
            == 2
            && parsed_neighbors.contains(&StereoNeighbor::Atom(opposite_endpoint))
    }

    fn atom_based_double_bond_sides(
        &self,
        endpoint_a: usize,
        endpoint_b: usize,
        rooted_classes: &[usize],
        refined_classes: &[usize],
    ) -> Option<(AtomBasedDoubleBondSide, AtomBasedDoubleBondSide)> {
        let side_a = self.atom_based_double_bond_side(
            endpoint_a,
            endpoint_b,
            rooted_classes,
            refined_classes,
        )?;
        let side_b = self.atom_based_double_bond_side(
            endpoint_b,
            endpoint_a,
            rooted_classes,
            refined_classes,
        )?;
        Some((side_a, side_b))
    }

    fn atom_based_double_bond_side(
        &self,
        endpoint: usize,
        opposite_endpoint: usize,
        rooted_classes: &[usize],
        refined_classes: &[usize],
    ) -> Option<AtomBasedDoubleBondSide> {
        let parsed_neighbors = self.parsed_stereo_neighbors_row(endpoint);
        let atom = self.nodes()[endpoint];
        let chirality =
            stereo_chirality_normal_form(self, endpoint, atom.chirality(), parsed_neighbors)?;
        if !matches!(chirality, Chirality::At | Chirality::AtAt | Chirality::TH(1 | 2)) {
            return None;
        }
        let reference_atom = self.atom_based_reference_substituent(
            endpoint,
            opposite_endpoint,
            rooted_classes,
            refined_classes,
        )?;
        let target_neighbors = atom_based_double_bond_target_neighbors(
            parsed_neighbors,
            opposite_endpoint,
            reference_atom,
        )?;
        let permutation = atom_based_permutation_from(parsed_neighbors, &target_neighbors)?;
        let reference_bond_is_up = atom_based_chirality_is_clockwise(chirality)
            ^ atom_based_permutation_is_odd(&permutation);
        Some(AtomBasedDoubleBondSide { endpoint, reference_atom, reference_bond_is_up })
    }

    fn atom_based_reference_substituent(
        &self,
        endpoint: usize,
        opposite_endpoint: usize,
        rooted_classes: &[usize],
        refined_classes: &[usize],
    ) -> Option<usize> {
        let mut neighbors =
            self.parsed_stereo_neighbors_row(endpoint).iter().filter_map(|&neighbor| {
                match neighbor {
                    StereoNeighbor::Atom(node_id) if node_id != opposite_endpoint => Some(node_id),
                    _ => None,
                }
            });
        let mut best = neighbors.next()?;
        let mut best_key = atom_based_substituent_priority_key(
            self,
            endpoint,
            best,
            rooted_classes,
            refined_classes,
        );
        let mut unique_best = true;
        for candidate in neighbors {
            let candidate_key = atom_based_substituent_priority_key(
                self,
                endpoint,
                candidate,
                rooted_classes,
                refined_classes,
            );
            match candidate_key.cmp(&best_key) {
                core::cmp::Ordering::Greater => {
                    best = candidate;
                    best_key = candidate_key;
                    unique_best = true;
                }
                core::cmp::Ordering::Equal => unique_best = false,
                core::cmp::Ordering::Less => {}
            }
        }
        unique_best.then_some(best)
    }
}

pub(super) fn atom_based_substituent_priority_key(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    endpoint: usize,
    neighbor: usize,
    rooted_classes: &[usize],
    refined_classes: &[usize],
) -> AtomBasedSubstituentPriorityKey {
    let atom = smiles.node_by_id(neighbor).unwrap_or_else(|| unreachable!());
    let atomic_number = atom.element().map_or(0, u8::from);
    let edge = smiles.edge_for_node_pair((endpoint, neighbor)).unwrap_or_else(|| unreachable!());
    let bond_order_to_endpoint = if edge.is_aromatic() {
        1
    } else {
        match edge.bond() {
            Bond::Single | Bond::Up | Bond::Down => 1,
            Bond::Double => 2,
            Bond::Triple => 3,
            Bond::Quadruple => 4,
        }
    };

    AtomBasedSubstituentPriorityKey {
        atomic_number,
        isotope_mass_number: atom.isotope_mass_number(),
        charge: atom.charge_value(),
        aromatic: atom.aromatic(),
        bond_order_to_endpoint,
        rooted_class: rooted_classes[neighbor],
        refined_class: refined_classes[neighbor],
    }
}

fn non_semantic_double_bond_supports_semantic_stereo(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    node_a: usize,
    node_b: usize,
) -> bool {
    smiles
        .bond_matrix()
        .sparse_row_values_ref(node_a)
        .filter(|entry| non_single_family_bond(**entry))
        .count()
        == 1
        && smiles
            .bond_matrix()
            .sparse_row_values_ref(node_b)
            .filter(|entry| non_single_family_bond(**entry))
            .count()
            == 1
}

#[inline]
fn non_aromatic_double_bond(entry: crate::smiles::BondEntry) -> bool {
    entry.bond() == Bond::Double && !entry.aromatic()
}

#[inline]
fn non_single_family_bond(entry: crate::smiles::BondEntry) -> bool {
    entry.aromatic() || !matches!(entry.bond(), Bond::Single | Bond::Up | Bond::Down)
}

fn non_semantic_double_bond_directional_neighbors(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    endpoint: usize,
    opposite_endpoint: usize,
) -> Vec<usize> {
    smiles
        .bond_matrix()
        .sparse_row(endpoint)
        .zip(smiles.bond_matrix().sparse_row_values_ref(endpoint))
        .filter_map(|(neighbor, entry)| {
            (neighbor != opposite_endpoint && matches!(entry.bond(), Bond::Up | Bond::Down))
                .then_some(neighbor)
        })
        .collect()
}

fn non_semantic_double_bond_has_unique_reference_substituent(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    endpoint: usize,
    opposite_endpoint: usize,
    rooted_classes: &[usize],
    refined_classes: &[usize],
) -> Option<usize> {
    let neighbors = smiles
        .bond_matrix()
        .sparse_row(endpoint)
        .filter(|&neighbor| neighbor != opposite_endpoint)
        .collect::<Vec<_>>();
    let (&first, rest) = neighbors.split_first()?;
    let mut best = first;
    let mut best_key = atom_based_substituent_priority_key(
        smiles,
        endpoint,
        best,
        rooted_classes,
        refined_classes,
    );
    let mut unique_best = true;

    for &candidate in rest {
        let candidate_key = atom_based_substituent_priority_key(
            smiles,
            endpoint,
            candidate,
            rooted_classes,
            refined_classes,
        );
        match candidate_key.cmp(&best_key) {
            core::cmp::Ordering::Greater => {
                best = candidate;
                best_key = candidate_key;
                unique_best = true;
            }
            core::cmp::Ordering::Equal => unique_best = false,
            core::cmp::Ordering::Less => {}
        }
    }

    unique_best.then_some(best)
}

fn atom_based_chirality_is_clockwise(chirality: Chirality) -> bool {
    matches!(chirality, Chirality::AtAt | Chirality::TH(2))
}

fn atom_based_double_bond_target_neighbors(
    parsed_neighbors: &[StereoNeighbor],
    opposite_endpoint: usize,
    reference_atom: usize,
) -> Option<Vec<StereoNeighbor>> {
    let reference = StereoNeighbor::Atom(reference_atom);
    let opposite = StereoNeighbor::Atom(opposite_endpoint);
    let other = parsed_neighbors
        .iter()
        .copied()
        .find(|neighbor| *neighbor != reference && *neighbor != opposite)?;
    Some(vec![reference, opposite, other])
}

fn atom_based_permutation_from(
    parsed_neighbors: &[StereoNeighbor],
    target_neighbors: &[StereoNeighbor],
) -> Option<Vec<usize>> {
    let mut used = vec![false; parsed_neighbors.len()];
    let mut permutation = Vec::with_capacity(target_neighbors.len());
    for target in target_neighbors {
        let index = parsed_neighbors
            .iter()
            .enumerate()
            .find_map(|(index, parsed)| (!used[index] && parsed == target).then_some(index))?;
        used[index] = true;
        permutation.push(index);
    }
    Some(permutation)
}

fn atom_based_permutation_is_odd(permutation: &[usize]) -> bool {
    let mut visited = vec![false; permutation.len()];
    let mut transpositions = 0;

    for start in 0..permutation.len() {
        if visited[start] {
            continue;
        }
        let mut length = 0;
        let mut cursor = start;
        while !visited[cursor] {
            visited[cursor] = true;
            cursor = permutation[cursor];
            length += 1;
        }
        if length > 0 {
            transpositions += length - 1;
        }
    }

    transpositions % 2 == 1
}

pub(super) fn atom_based_override_bond(
    rows: &[Vec<(usize, Bond)>],
    from: usize,
    to: usize,
) -> Option<Bond> {
    let (left, right) = crate::smiles::edge_key(from, to);
    let row = rows.get(left)?;
    let index = row.binary_search_by_key(&right, |&(neighbor, _)| neighbor).ok()?;
    Some(row[index].1)
}
