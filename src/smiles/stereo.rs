use alloc::vec::Vec;

use super::Smiles;
use crate::{atom::bracketed::chirality::Chirality, bond::Bond};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum StereoNeighbor {
    Atom(usize),
    ExplicitHydrogen,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct DirectionalBondOverrides {
    rows: Vec<Vec<(usize, Bond)>>,
    semantic_endpoints: Vec<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DirectionalParityConstraint {
    pub(super) left_edge_key: (usize, usize),
    pub(super) right_edge_key: (usize, usize),
    pub(super) same_parity: bool,
}

impl DirectionalBondOverrides {
    #[cfg(test)]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.rows.iter().all(Vec::is_empty)
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.rows.iter().map(Vec::len).sum()
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn contains_key(&self, edge_key: &(usize, usize)) -> bool {
        self.get(edge_key.0, edge_key.1).is_some()
    }

    #[inline]
    #[must_use]
    pub(crate) fn get(&self, from: usize, to: usize) -> Option<Bond> {
        let (left, right) = crate::smiles::edge_key(from, to);
        let row = self.rows.get(left)?;
        let index = row.binary_search_by_key(&right, |&(neighbor, _)| neighbor).ok()?;
        Some(row[index].1)
    }

    #[inline]
    #[must_use]
    pub(crate) fn has_semantic_endpoint(&self, node_id: usize) -> bool {
        self.semantic_endpoints.get(node_id).copied().unwrap_or(false)
    }
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    #[inline]
    #[must_use]
    pub(super) fn parsed_stereo_neighbors_row(&self, node_id: usize) -> &[StereoNeighbor] {
        self.parsed_stereo_neighbors.get(node_id).map_or(&[], Vec::as_slice)
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn projected_directional_bond_overrides(
        &self,
        preorder_indices: &[usize],
    ) -> DirectionalBondOverrides {
        let invariants = self.atom_invariants();
        let refined = self.refined_atom_classes_from_invariants(&invariants);
        let rooted = self.rooted_symmetry_classes_from_refined(refined.classes());
        self.projected_directional_bond_overrides_with_classes(
            preorder_indices,
            &rooted,
            refined.classes(),
        )
    }

    #[must_use]
    pub(crate) fn projected_directional_bond_overrides_with_classes(
        &self,
        preorder_indices: &[usize],
        rooted_classes: &[usize],
        refined_classes: &[usize],
    ) -> DirectionalBondOverrides {
        let records =
            self.double_bond_stereo_records_with_planning_classes(rooted_classes, refined_classes);
        if records.is_empty() {
            return DirectionalBondOverrides::default();
        }
        let node_count = self.nodes().len();
        let mut semantic_endpoints = vec![false; node_count];
        for record in &records {
            semantic_endpoints[record.side_a().endpoint()] = true;
            semantic_endpoints[record.side_b().endpoint()] = true;
        }
        let constraints = records
            .into_iter()
            .map(|record| {
                DirectionalParityConstraint {
                    left_edge_key: crate::smiles::edge_key(
                        record.side_a().endpoint(),
                        record.side_a().reference_atom(),
                    ),
                    right_edge_key: crate::smiles::edge_key(
                        record.side_b().endpoint(),
                        record.side_b().reference_atom(),
                    ),
                    same_parity: matches!(
                        record.config(),
                        crate::smiles::double_bond_stereo::DoubleBondStereoConfig::E
                    ),
                }
            })
            .collect::<Vec<_>>();

        DirectionalBondOverrides {
            rows: directional_override_rows_from_parity_constraints(
                node_count,
                &constraints,
                preorder_indices,
            ),
            semantic_endpoints,
        }
    }
}

pub(super) fn directional_override_rows_from_parity_constraints(
    node_count: usize,
    constraints: &[DirectionalParityConstraint],
    preorder_indices: &[usize],
) -> Vec<Vec<(usize, Bond)>> {
    let mut rows = vec![Vec::new(); node_count];
    if constraints.is_empty() {
        return rows;
    }

    let mut edge_keys = constraints
        .iter()
        .flat_map(|constraint| [constraint.left_edge_key, constraint.right_edge_key])
        .collect::<Vec<_>>();
    edge_keys.sort_unstable();
    edge_keys.dedup();

    let mut adjacency: Vec<Vec<(usize, bool)>> = vec![Vec::new(); edge_keys.len()];
    for constraint in constraints {
        let left_edge =
            edge_keys.binary_search(&constraint.left_edge_key).unwrap_or_else(|_| unreachable!());
        let right_edge =
            edge_keys.binary_search(&constraint.right_edge_key).unwrap_or_else(|_| unreachable!());
        adjacency[left_edge].push((right_edge, constraint.same_parity));
        adjacency[right_edge].push((left_edge, constraint.same_parity));
    }

    let mut bond_is_up: Vec<Option<bool>> = vec![None; edge_keys.len()];
    let mut component_seen = vec![false; edge_keys.len()];
    let mut stack = Vec::new();

    for start in 0..edge_keys.len() {
        if component_seen[start] {
            continue;
        }

        let mut component = Vec::new();
        stack.push(start);

        while let Some(current) = stack.pop() {
            if component_seen[current] {
                continue;
            }
            component_seen[current] = true;
            component.push(current);
            for &(neighbor, _same_parity) in &adjacency[current] {
                if !component_seen[neighbor] {
                    stack.push(neighbor);
                }
            }
        }

        let seed = component
            .iter()
            .copied()
            .min_by_key(|&edge_id| {
                canonical_directional_edge_key(edge_keys[edge_id], preorder_indices)
            })
            .unwrap_or_else(|| unreachable!());

        bond_is_up[seed] = Some(true);
        stack.push(seed);

        while let Some(current) = stack.pop() {
            let current_is_up = bond_is_up[current].unwrap_or_else(|| unreachable!());
            for &(neighbor, same_parity) in &adjacency[current] {
                let expected = if same_parity { current_is_up } else { !current_is_up };
                if let Some(existing) = bond_is_up[neighbor] {
                    debug_assert_eq!(existing, expected);
                } else {
                    bond_is_up[neighbor] = Some(expected);
                    stack.push(neighbor);
                }
            }
        }
    }

    for (edge_key, is_up) in edge_keys.into_iter().zip(bond_is_up) {
        let bond = if is_up.unwrap_or(true) { Bond::Up } else { Bond::Down };
        rows[edge_key.0].push((edge_key.1, bond));
    }
    for row in &mut rows {
        row.sort_unstable_by_key(|&(neighbor, _)| neighbor);
    }
    rows
}

fn canonical_directional_edge_key(
    edge_key: (usize, usize),
    preorder_indices: &[usize],
) -> (usize, usize) {
    let left = preorder_indices[edge_key.0];
    let right = preorder_indices[edge_key.1];
    if left <= right { (left, right) } else { (right, left) }
}

pub(crate) fn normalized_bond_for_emit(bond: Bond, from: usize, to: usize) -> Bond {
    let _ = (from, to);
    bond
}

#[must_use]
pub(crate) fn normalized_tetrahedral_chirality(
    chirality: Option<Chirality>,
    parsed_neighbors: &[StereoNeighbor],
    emitted_neighbors: &[StereoNeighbor],
) -> Option<Chirality> {
    let chirality = chirality?;
    if parsed_neighbors.len() != emitted_neighbors.len() {
        return Some(chirality);
    }
    reorder_tetrahedral_chirality(chirality, parsed_neighbors, emitted_neighbors)
        .or(Some(chirality))
}

fn permutation_from(
    parsed_neighbors: &[StereoNeighbor],
    emitted_neighbors: &[StereoNeighbor],
) -> Option<Vec<usize>> {
    let mut used = vec![false; parsed_neighbors.len()];
    let mut permutation = Vec::with_capacity(emitted_neighbors.len());

    for emitted in emitted_neighbors {
        let index = parsed_neighbors
            .iter()
            .enumerate()
            .find_map(|(index, parsed)| (!used[index] && parsed == emitted).then_some(index))?;
        used[index] = true;
        permutation.push(index);
    }

    Some(permutation)
}

fn reorder_tetrahedral_chirality(
    chirality: Chirality,
    from_neighbors: &[StereoNeighbor],
    to_neighbors: &[StereoNeighbor],
) -> Option<Chirality> {
    let permutation = permutation_from(from_neighbors, to_neighbors)?;
    Some(if permutation_is_odd(&permutation) {
        invert_tetrahedral_chirality(chirality)
    } else {
        chirality
    })
}

fn permutation_is_odd(permutation: &[usize]) -> bool {
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

fn invert_tetrahedral_chirality(chirality: Chirality) -> Chirality {
    match chirality {
        Chirality::At => Chirality::AtAt,
        Chirality::AtAt => Chirality::At,
        Chirality::TH(1) => Chirality::TH(2),
        Chirality::TH(2) => Chirality::TH(1),
        Chirality::AL(1) => Chirality::AL(2),
        Chirality::AL(2) => Chirality::AL(1),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{
        Smiles, StereoNeighbor, invert_tetrahedral_chirality, normalized_bond_for_emit,
        normalized_tetrahedral_chirality,
    };
    use crate::{atom::bracketed::chirality::Chirality, bond::Bond};

    fn identity_preorder(smiles: &Smiles) -> Vec<usize> {
        (0..smiles.nodes().len()).collect()
    }

    #[test]
    fn normalized_bond_for_emit_preserves_forward_direction() {
        assert_eq!(normalized_bond_for_emit(Bond::Up, 0, 1), Bond::Up);
        assert_eq!(normalized_bond_for_emit(Bond::Down, 0, 1), Bond::Down);
        assert_eq!(normalized_bond_for_emit(Bond::Double, 0, 1), Bond::Double);
    }

    #[test]
    fn normalized_bond_for_emit_preserves_stored_direction_for_reverse_traversal() {
        assert_eq!(normalized_bond_for_emit(Bond::Up, 1, 0), Bond::Up);
        assert_eq!(normalized_bond_for_emit(Bond::Down, 1, 0), Bond::Down);
        assert_eq!(normalized_bond_for_emit(Bond::Single, 1, 0), Bond::Single);
    }

    #[test]
    fn parsed_stereo_neighbors_follow_bond_insertion_order_and_explicit_hydrogen() {
        let smiles: Smiles = "N[C@@H](C)O".parse().unwrap();
        assert_eq!(
            smiles.parsed_stereo_neighbors_row(1),
            &[
                StereoNeighbor::Atom(0),
                StereoNeighbor::ExplicitHydrogen,
                StereoNeighbor::Atom(2),
                StereoNeighbor::Atom(3),
            ]
        );
    }

    #[test]
    fn parsed_stereo_neighbors_place_ring_digit_where_it_appears_in_source() {
        let smiles: Smiles = "C[C@H]1CCCCN1N=O".parse().unwrap();
        assert_eq!(
            smiles.parsed_stereo_neighbors_row(1),
            &[
                StereoNeighbor::Atom(0),
                StereoNeighbor::ExplicitHydrogen,
                StereoNeighbor::Atom(6),
                StereoNeighbor::Atom(2),
            ]
        );
    }

    #[test]
    fn normalized_tetrahedral_chirality_flips_on_odd_permutation() {
        let parsed = [
            StereoNeighbor::Atom(0),
            StereoNeighbor::Atom(2),
            StereoNeighbor::Atom(3),
            StereoNeighbor::ExplicitHydrogen,
        ];
        let emitted = [
            StereoNeighbor::Atom(0),
            StereoNeighbor::Atom(3),
            StereoNeighbor::Atom(2),
            StereoNeighbor::ExplicitHydrogen,
        ];

        assert_eq!(
            normalized_tetrahedral_chirality(Some(Chirality::At), &parsed, &emitted),
            Some(Chirality::AtAt)
        );
        assert_eq!(
            normalized_tetrahedral_chirality(Some(Chirality::AtAt), &parsed, &emitted),
            Some(Chirality::At)
        );
        assert_eq!(
            normalized_tetrahedral_chirality(Some(Chirality::TH(1)), &parsed, &emitted),
            Some(Chirality::TH(2))
        );
        assert_eq!(
            normalized_tetrahedral_chirality(Some(Chirality::AL(2)), &parsed, &emitted),
            Some(Chirality::AL(1))
        );
    }

    #[test]
    fn normalized_tetrahedral_chirality_preserves_even_or_unknown_permutations() {
        let parsed = [
            StereoNeighbor::Atom(0),
            StereoNeighbor::Atom(2),
            StereoNeighbor::Atom(3),
            StereoNeighbor::ExplicitHydrogen,
        ];
        let even = [
            StereoNeighbor::Atom(3),
            StereoNeighbor::Atom(0),
            StereoNeighbor::Atom(2),
            StereoNeighbor::ExplicitHydrogen,
        ];
        let mismatched = [
            StereoNeighbor::Atom(0),
            StereoNeighbor::Atom(2),
            StereoNeighbor::Atom(4),
            StereoNeighbor::ExplicitHydrogen,
        ];

        assert_eq!(
            normalized_tetrahedral_chirality(Some(Chirality::At), &parsed, &even),
            Some(Chirality::At)
        );
        assert_eq!(
            normalized_tetrahedral_chirality(Some(Chirality::SP(1)), &parsed, &even),
            Some(Chirality::SP(1))
        );
        assert_eq!(
            normalized_tetrahedral_chirality(Some(Chirality::At), &parsed, &mismatched),
            Some(Chirality::At)
        );
        assert_eq!(normalized_tetrahedral_chirality(None, &parsed, &even), None);
    }

    #[test]
    fn projected_directional_bond_overrides_are_empty_without_double_bond_stereo() {
        assert!(
            Smiles::<crate::smiles::ConcreteAtoms>::new_for_policy()
                .projected_directional_bond_overrides(&[])
                .is_empty()
        );
        let smiles: Smiles = "CCO".parse().unwrap();
        let preorder = identity_preorder(&smiles);
        assert!(smiles.projected_directional_bond_overrides(&preorder).is_empty());
    }

    #[test]
    fn projected_directional_bond_overrides_match_simple_e_constraint() {
        let smiles: Smiles = "F/C=C/F".parse().unwrap();
        let preorder = identity_preorder(&smiles);
        let overrides = smiles.projected_directional_bond_overrides(&preorder);
        assert_eq!(overrides.len(), 2);
        assert_eq!(overrides.get(0, 1), Some(Bond::Up));
        assert_eq!(overrides.get(2, 3), Some(Bond::Up));
    }

    #[test]
    fn projected_directional_bond_overrides_match_simple_z_constraint() {
        let smiles: Smiles = "F/C=C\\F".parse().unwrap();
        let preorder = identity_preorder(&smiles);
        let overrides = smiles.projected_directional_bond_overrides(&preorder);
        assert_eq!(overrides.len(), 2);
        assert_ne!(overrides.get(0, 1), overrides.get(2, 3));
    }

    #[test]
    fn projected_directional_bond_overrides_follow_reference_edges_not_raw_tokens() {
        let smiles: Smiles = "CC/C(Cl)=C(/F)C".parse().unwrap();
        let record = smiles.double_bond_stereo_records()[0];
        let preorder = identity_preorder(&smiles);
        let overrides = smiles.projected_directional_bond_overrides(&preorder);

        let left_reference =
            crate::smiles::edge_key(record.side_a().endpoint(), record.side_a().reference_atom());
        let right_reference =
            crate::smiles::edge_key(record.side_b().endpoint(), record.side_b().reference_atom());

        assert!(overrides.contains_key(&left_reference));
        assert!(overrides.contains_key(&right_reference));

        let left_raw = crate::smiles::edge_key(record.side_a().endpoint(), 1);
        if left_raw != left_reference {
            assert!(!overrides.contains_key(&left_raw));
        }
    }

    #[test]
    fn projected_directional_bond_overrides_expose_semantic_endpoints_and_missing_edges() {
        let smiles: Smiles = "F/C=C/F".parse().unwrap();
        let preorder = identity_preorder(&smiles);
        let overrides = smiles.projected_directional_bond_overrides(&preorder);

        assert!(overrides.has_semantic_endpoint(1));
        assert!(overrides.has_semantic_endpoint(2));
        assert!(!overrides.has_semantic_endpoint(0));
        assert!(!overrides.has_semantic_endpoint(3));
        assert_eq!(overrides.get(0, 3), None);
    }

    #[test]
    fn normalized_tetrahedral_chirality_preserves_on_length_mismatch_and_non_invertible_tags() {
        let parsed = [StereoNeighbor::Atom(0), StereoNeighbor::Atom(1)];
        let emitted =
            [StereoNeighbor::Atom(0), StereoNeighbor::Atom(1), StereoNeighbor::ExplicitHydrogen];
        assert_eq!(
            normalized_tetrahedral_chirality(Some(Chirality::TB(1)), &parsed, &emitted),
            Some(Chirality::TB(1))
        );

        let parsed = [
            StereoNeighbor::Atom(0),
            StereoNeighbor::Atom(2),
            StereoNeighbor::Atom(3),
            StereoNeighbor::ExplicitHydrogen,
        ];
        let odd = [
            StereoNeighbor::Atom(0),
            StereoNeighbor::Atom(3),
            StereoNeighbor::Atom(2),
            StereoNeighbor::ExplicitHydrogen,
        ];
        assert_eq!(
            normalized_tetrahedral_chirality(Some(Chirality::AL(1)), &parsed, &odd),
            Some(Chirality::AL(2))
        );
        assert_eq!(
            normalized_tetrahedral_chirality(Some(Chirality::OH(1)), &parsed, &odd),
            Some(Chirality::OH(1))
        );
    }

    #[test]
    fn projected_directional_bond_overrides_handle_multi_record_components() {
        let smiles: Smiles = "F/C=C/C=C/F".parse().unwrap();
        let preorder = identity_preorder(&smiles);
        let overrides = smiles.projected_directional_bond_overrides(&preorder);

        assert_eq!(overrides.len(), 3);
        assert!(overrides.has_semantic_endpoint(1));
        assert!(overrides.has_semantic_endpoint(4));
    }

    #[test]
    fn invert_tetrahedral_chirality_swaps_supported_pairs_and_leaves_others() {
        assert_eq!(invert_tetrahedral_chirality(Chirality::At), Chirality::AtAt);
        assert_eq!(invert_tetrahedral_chirality(Chirality::AtAt), Chirality::At);
        assert_eq!(invert_tetrahedral_chirality(Chirality::TH(2)), Chirality::TH(1));
        assert_eq!(invert_tetrahedral_chirality(Chirality::AL(2)), Chirality::AL(1));
        assert_eq!(invert_tetrahedral_chirality(Chirality::SP(3)), Chirality::SP(3));
    }
}
