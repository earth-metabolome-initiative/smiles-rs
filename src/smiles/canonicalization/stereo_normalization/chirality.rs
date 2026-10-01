use alloc::vec::Vec;

use super::directional_bonds::{StereoSubstituentIdentityKey, atom_based_substituent_priority_key};
use crate::{
    atom::{Atom, AtomSyntax, bracketed::chirality::Chirality},
    bond::Bond,
    smiles::{Smiles, StereoNeighbor, stereo::normalized_tetrahedral_chirality},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SquareShape {
    U,
    Z,
    Four,
}

pub(super) fn atom_with_chirality(atom: Atom, chirality: Option<Chirality>) -> Atom {
    match atom.syntax() {
        AtomSyntax::OrganicSubset => {
            debug_assert!(
                chirality.is_none(),
                "organic-subset atoms should not carry explicit chirality tags"
            );
            Atom::new_organic_subset(atom.symbol(), atom.aromatic())
        }
        AtomSyntax::Bracket => {
            let mut builder = Atom::builder()
                .with_symbol(atom.symbol())
                .with_aromatic(atom.aromatic())
                .with_hydrogens(atom.hydrogen_count())
                .with_charge(atom.charge())
                .with_class(atom.class());
            if let Some(isotope) = atom.isotope_mass_number() {
                builder = builder.with_isotope(isotope);
            }
            if let Some(chirality) = chirality {
                builder = builder.with_chirality(chirality);
            }
            builder.build()
        }
    }
}

fn canonical_stereo_neighbor_sort_key(
    neighbor: StereoNeighbor,
    new_index_of_old_node: &[usize],
) -> (u8, usize) {
    match neighbor {
        StereoNeighbor::Atom(node_id) => (0, new_index_of_old_node[node_id]),
        StereoNeighbor::ExplicitHydrogen => (1, usize::MAX),
    }
}

fn allene_like_stereo_center(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    node_id: usize,
) -> bool {
    smiles.edge_count_for_node(node_id) == 2
        && smiles
            .edges_for_node(node_id)
            .all(|edge| edge.bond() == Bond::Double && !edge.is_aromatic())
}

fn non_tetrahedral_default_chirality(
    chirality: Option<Chirality>,
    parsed_neighbors: &[StereoNeighbor],
) -> Option<Chirality> {
    match chirality {
        Some(Chirality::At) if parsed_neighbors.len() == 5 => Some(Chirality::TB(1)),
        Some(Chirality::AtAt) if parsed_neighbors.len() == 5 => Some(Chirality::TB(2)),
        Some(Chirality::At) if parsed_neighbors.len() == 6 => Some(Chirality::OH(1)),
        Some(Chirality::AtAt) if parsed_neighbors.len() == 6 => Some(Chirality::OH(2)),
        other => other,
    }
}

pub(crate) fn stereo_chirality_normal_form(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    node_id: usize,
    chirality: Option<Chirality>,
    parsed_neighbors: &[StereoNeighbor],
) -> Option<Chirality> {
    match chirality {
        Some(Chirality::At) => {
            if allene_like_stereo_center(smiles, node_id) {
                Some(Chirality::AL(1))
            } else if parsed_neighbors.len() == 4 {
                Some(Chirality::TH(1))
            } else if parsed_neighbors.len() == 5 {
                Some(Chirality::TB(1))
            } else if parsed_neighbors.len() == 6 {
                Some(Chirality::OH(1))
            } else {
                Some(Chirality::At)
            }
        }
        Some(Chirality::AtAt) => {
            if allene_like_stereo_center(smiles, node_id) {
                Some(Chirality::AL(2))
            } else if parsed_neighbors.len() == 4 {
                Some(Chirality::TH(2))
            } else if parsed_neighbors.len() == 5 {
                Some(Chirality::TB(2))
            } else if parsed_neighbors.len() == 6 {
                Some(Chirality::OH(2))
            } else {
                Some(Chirality::AtAt)
            }
        }
        other => non_tetrahedral_default_chirality(other, parsed_neighbors),
    }
}

fn supports_stereo_neighbor_normalization(chirality: Option<Chirality>) -> bool {
    matches!(
        chirality,
        Some(
            Chirality::At
                | Chirality::AtAt
                | Chirality::TH(_)
                | Chirality::AL(_)
                | Chirality::SP(_)
                | Chirality::TB(_)
                | Chirality::OH(_)
        )
    )
}

fn expected_stereo_neighbor_count(chirality: Chirality) -> usize {
    match chirality {
        Chirality::At
        | Chirality::AtAt
        | Chirality::TH(_)
        | Chirality::AL(_)
        | Chirality::SP(_) => 4,
        Chirality::TB(_) => 5,
        Chirality::OH(_) => 6,
    }
}

fn stereo_neighbors_with_implicit_hydrogens(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    node_id: usize,
    chirality: Chirality,
    parsed_neighbors: &[StereoNeighbor],
) -> Vec<StereoNeighbor> {
    let mut neighbors = parsed_neighbors.to_vec();
    let expected_count = expected_stereo_neighbor_count(chirality);
    let atom_hydrogens = usize::from(smiles.nodes()[node_id].hydrogen_count());
    let present_implicit_hydrogens =
        neighbors.iter().filter(|neighbor| **neighbor == StereoNeighbor::ExplicitHydrogen).count();
    let missing_implicit_hydrogens = atom_hydrogens.saturating_sub(present_implicit_hydrogens);
    let missing_capacity = expected_count.saturating_sub(neighbors.len());

    for _ in 0..missing_implicit_hydrogens.min(missing_capacity) {
        neighbors.push(StereoNeighbor::ExplicitHydrogen);
    }
    neighbors
}

pub(crate) fn canonical_stereo_neighbors_row(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    node_id: usize,
    chirality: Option<Chirality>,
    parsed_neighbors: &[StereoNeighbor],
    new_index_of_old_node: &[usize],
    rooted_classes: &[usize],
    refined_classes: &[usize],
) -> Vec<StereoNeighbor> {
    let chirality = stereo_chirality_normal_form(smiles, node_id, chirality, parsed_neighbors);
    if !supports_stereo_neighbor_normalization(chirality) {
        return parsed_neighbors.to_vec();
    }
    let expanded_neighbors = stereo_neighbors_with_implicit_hydrogens(
        smiles,
        node_id,
        chirality.unwrap_or_else(|| unreachable!()),
        parsed_neighbors,
    );

    match chirality {
        Some(Chirality::At | Chirality::AtAt | Chirality::TH(_) | Chirality::AL(_)) => {
            let mut neighbors = expanded_neighbors;
            neighbors.sort_unstable_by_key(|neighbor| {
                canonical_stereo_neighbor_sort_key(*neighbor, new_index_of_old_node)
            });
            neighbors
        }
        Some(Chirality::SP(kind)) => {
            square_assignment_from_shape(Chirality::SP(kind), &expanded_neighbors).map_or_else(
                || parsed_neighbors.to_vec(),
                |assignment| {
                    canonical_square_planar_neighbors(
                        smiles,
                        node_id,
                        &assignment,
                        new_index_of_old_node,
                        rooted_classes,
                        refined_classes,
                    )
                },
            )
        }
        Some(Chirality::TB(_)) => {
            canonical_tb_neighbors(chirality, &expanded_neighbors, new_index_of_old_node)
                .unwrap_or_else(|| parsed_neighbors.to_vec())
        }
        Some(Chirality::OH(_)) => {
            canonical_oh_neighbors(chirality, &expanded_neighbors, new_index_of_old_node)
                .unwrap_or_else(|| parsed_neighbors.to_vec())
        }
        None => parsed_neighbors.to_vec(),
    }
}

pub(super) fn normalized_stereo_chirality(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    node_id: usize,
    chirality: Option<Chirality>,
    parsed_neighbors: &[StereoNeighbor],
    normalized_neighbors: &[StereoNeighbor],
    rooted_classes: &[usize],
    refined_classes: &[usize],
) -> Option<Chirality> {
    let chirality = stereo_chirality_normal_form(smiles, node_id, chirality, parsed_neighbors)?;
    match chirality {
        Chirality::At | Chirality::AtAt | Chirality::TH(_) | Chirality::AL(_) => {
            if matches!(chirality, Chirality::At | Chirality::AtAt | Chirality::TH(_))
                && !tetrahedral_like_is_stereogenic(
                    smiles,
                    node_id,
                    chirality,
                    parsed_neighbors,
                    rooted_classes,
                    refined_classes,
                )
            {
                return None;
            }
            normalized_tetrahedral_chirality(
                Some(chirality),
                parsed_neighbors,
                normalized_neighbors,
            )
        }
        Chirality::SP(_) => Some(Chirality::SP(1)),
        Chirality::TB(_) => Some(Chirality::TB(1)),
        Chirality::OH(_) => Some(Chirality::OH(1)),
    }
}

fn stereo_substituent_identity_key(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    endpoint: usize,
    neighbor: StereoNeighbor,
    rooted_classes: &[usize],
    refined_classes: &[usize],
) -> StereoSubstituentIdentityKey {
    match neighbor {
        StereoNeighbor::ExplicitHydrogen => StereoSubstituentIdentityKey::ExplicitHydrogen,
        StereoNeighbor::Atom(node_id) => {
            StereoSubstituentIdentityKey::Atom(atom_based_substituent_priority_key(
                smiles,
                endpoint,
                node_id,
                rooted_classes,
                refined_classes,
            ))
        }
    }
}

pub(crate) fn tetrahedral_like_is_stereogenic(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    node_id: usize,
    chirality: Chirality,
    parsed_neighbors: &[StereoNeighbor],
    rooted_classes: &[usize],
    refined_classes: &[usize],
) -> bool {
    let mut keys =
        stereo_neighbors_with_implicit_hydrogens(smiles, node_id, chirality, parsed_neighbors)
            .into_iter()
            .map(|neighbor| {
                stereo_substituent_identity_key(
                    smiles,
                    node_id,
                    neighbor,
                    rooted_classes,
                    refined_classes,
                )
            })
            .collect::<Vec<_>>();
    keys.sort_unstable();
    keys.windows(2).all(|window| window[0] != window[1])
}

fn stereo_neighbor_sequence_key(
    neighbors: &[StereoNeighbor],
    new_index_of_old_node: &[usize],
) -> Vec<(u8, usize)> {
    neighbors
        .iter()
        .copied()
        .map(|neighbor| canonical_stereo_neighbor_sort_key(neighbor, new_index_of_old_node))
        .collect()
}

fn square_planar_identity_sequence_key(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    node_id: usize,
    neighbors: &[StereoNeighbor],
    rooted_classes: &[usize],
    refined_classes: &[usize],
) -> Vec<StereoSubstituentIdentityKey> {
    neighbors
        .iter()
        .copied()
        .map(|neighbor| {
            stereo_substituent_identity_key(
                smiles,
                node_id,
                neighbor,
                rooted_classes,
                refined_classes,
            )
        })
        .collect()
}

fn canonicalize_equivalent_square_planar_neighbors(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    node_id: usize,
    neighbors: &[StereoNeighbor],
    new_index_of_old_node: &[usize],
    rooted_classes: &[usize],
    refined_classes: &[usize],
) -> Vec<StereoNeighbor> {
    let mut canonicalized = neighbors.to_vec();
    let mut keyed_positions = neighbors
        .iter()
        .copied()
        .enumerate()
        .map(|(index, neighbor)| {
            (
                index,
                stereo_substituent_identity_key(
                    smiles,
                    node_id,
                    neighbor,
                    rooted_classes,
                    refined_classes,
                ),
            )
        })
        .collect::<Vec<_>>();
    keyed_positions.sort_unstable_by_key(|&(index, key)| (key, index));

    let mut start = 0;
    while start < keyed_positions.len() {
        let key = keyed_positions[start].1;
        let mut end = start + 1;
        while end < keyed_positions.len() && keyed_positions[end].1 == key {
            end += 1;
        }

        let mut positions =
            keyed_positions[start..end].iter().map(|(index, _key)| *index).collect::<Vec<_>>();
        positions.sort_unstable();
        let mut group_neighbors =
            positions.iter().map(|&index| neighbors[index]).collect::<Vec<_>>();
        group_neighbors.sort_unstable_by_key(|neighbor| {
            canonical_stereo_neighbor_sort_key(*neighbor, new_index_of_old_node)
        });
        for (position, neighbor) in positions.into_iter().zip(group_neighbors) {
            canonicalized[position] = neighbor;
        }

        start = end;
    }

    canonicalized
}

fn canonical_square_planar_candidate_key(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    node_id: usize,
    neighbors: &[StereoNeighbor],
    new_index_of_old_node: &[usize],
    rooted_classes: &[usize],
    refined_classes: &[usize],
) -> (Vec<StereoSubstituentIdentityKey>, Vec<(u8, usize)>) {
    (
        square_planar_identity_sequence_key(
            smiles,
            node_id,
            neighbors,
            rooted_classes,
            refined_classes,
        ),
        stereo_neighbor_sequence_key(neighbors, new_index_of_old_node),
    )
}

const fn square_shape_path(shape: SquareShape) -> [usize; 4] {
    match shape {
        SquareShape::U => [0, 1, 2, 3],
        SquareShape::Z => [0, 1, 3, 2],
        SquareShape::Four => [0, 2, 1, 3],
    }
}

fn square_shape_for_chirality(chirality: Chirality) -> Option<SquareShape> {
    match chirality {
        Chirality::SP(1) => Some(SquareShape::U),
        Chirality::SP(2) => Some(SquareShape::Four),
        Chirality::SP(3) => Some(SquareShape::Z),
        _ => None,
    }
}

fn canonical_square_planar_neighbors(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    node_id: usize,
    parsed_neighbors: &[StereoNeighbor],
    new_index_of_old_node: &[usize],
    rooted_classes: &[usize],
    refined_classes: &[usize],
) -> Vec<StereoNeighbor> {
    let assignment =
        [parsed_neighbors[0], parsed_neighbors[1], parsed_neighbors[2], parsed_neighbors[3]];
    let mut best = canonicalize_equivalent_square_planar_neighbors(
        smiles,
        node_id,
        &assignment,
        new_index_of_old_node,
        rooted_classes,
        refined_classes,
    );
    let mut best_key = canonical_square_planar_candidate_key(
        smiles,
        node_id,
        &best,
        new_index_of_old_node,
        rooted_classes,
        refined_classes,
    );
    for rotation in 0..4 {
        let rotated = [
            assignment[rotation % 4],
            assignment[(rotation + 1) % 4],
            assignment[(rotation + 2) % 4],
            assignment[(rotation + 3) % 4],
        ];
        let rotated = canonicalize_equivalent_square_planar_neighbors(
            smiles,
            node_id,
            &rotated,
            new_index_of_old_node,
            rooted_classes,
            refined_classes,
        );
        let rotated_key = canonical_square_planar_candidate_key(
            smiles,
            node_id,
            &rotated,
            new_index_of_old_node,
            rooted_classes,
            refined_classes,
        );
        if rotated_key < best_key {
            best = rotated;
            best_key = rotated_key;
        }

        let mirrored = [
            assignment[rotation % 4],
            assignment[(rotation + 3) % 4],
            assignment[(rotation + 2) % 4],
            assignment[(rotation + 1) % 4],
        ];
        let mirrored = canonicalize_equivalent_square_planar_neighbors(
            smiles,
            node_id,
            &mirrored,
            new_index_of_old_node,
            rooted_classes,
            refined_classes,
        );
        let mirrored_key = canonical_square_planar_candidate_key(
            smiles,
            node_id,
            &mirrored,
            new_index_of_old_node,
            rooted_classes,
            refined_classes,
        );
        if mirrored_key < best_key {
            best = mirrored;
            best_key = mirrored_key;
        }
    }
    best
}

fn square_assignment_from_shape(
    chirality: Chirality,
    sequence: &[StereoNeighbor],
) -> Option<[StereoNeighbor; 4]> {
    let path = square_shape_path(square_shape_for_chirality(chirality)?);
    let mut assignment = [StereoNeighbor::ExplicitHydrogen; 4];
    for (index, &position) in path.iter().enumerate() {
        assignment[position] =
            sequence.get(index).copied().unwrap_or(StereoNeighbor::ExplicitHydrogen);
    }
    Some(assignment)
}

fn tb_axis_and_order(chirality: Chirality) -> Option<(usize, usize, bool)> {
    match chirality {
        Chirality::TB(1) => Some((0, 4, false)),
        Chirality::TB(2) => Some((0, 4, true)),
        Chirality::TB(3) => Some((0, 3, false)),
        Chirality::TB(4) => Some((0, 3, true)),
        Chirality::TB(5) => Some((0, 2, false)),
        Chirality::TB(6) => Some((0, 2, true)),
        Chirality::TB(7) => Some((0, 1, false)),
        Chirality::TB(8) => Some((0, 1, true)),
        Chirality::TB(9) => Some((1, 4, false)),
        Chirality::TB(11) => Some((1, 4, true)),
        Chirality::TB(10) => Some((1, 3, false)),
        Chirality::TB(12) => Some((1, 3, true)),
        Chirality::TB(13) => Some((1, 2, false)),
        Chirality::TB(14) => Some((1, 2, true)),
        Chirality::TB(15) => Some((2, 4, false)),
        Chirality::TB(20) => Some((2, 4, true)),
        Chirality::TB(16) => Some((2, 3, false)),
        Chirality::TB(19) => Some((2, 3, true)),
        Chirality::TB(17) => Some((3, 4, false)),
        Chirality::TB(18) => Some((3, 4, true)),
        _ => None,
    }
}

fn canonical_tb_neighbors(
    chirality: Option<Chirality>,
    parsed_neighbors: &[StereoNeighbor],
    new_index_of_old_node: &[usize],
) -> Option<Vec<StereoNeighbor>> {
    let chirality = chirality?;
    let (axis_start_index, axis_end_index, clockwise) = tb_axis_and_order(chirality)?;
    let axis_start = parsed_neighbors.get(axis_start_index).copied()?;
    let axis_end = parsed_neighbors.get(axis_end_index).copied()?;
    let mut cycle = parsed_neighbors
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, neighbor)| {
            (index != axis_start_index && index != axis_end_index).then_some(neighbor)
        })
        .collect::<Vec<_>>();
    if clockwise {
        cycle.reverse();
    }
    if canonical_stereo_neighbor_sort_key(axis_end, new_index_of_old_node)
        < canonical_stereo_neighbor_sort_key(axis_start, new_index_of_old_node)
    {
        cycle.reverse();
        return Some(
            [
                vec![axis_end],
                minimal_cycle_rotation(&cycle, new_index_of_old_node),
                vec![axis_start],
            ]
            .concat(),
        );
    }
    Some(
        [vec![axis_start], minimal_cycle_rotation(&cycle, new_index_of_old_node), vec![axis_end]]
            .concat(),
    )
}

fn canonical_oh_neighbors(
    chirality: Option<Chirality>,
    parsed_neighbors: &[StereoNeighbor],
    new_index_of_old_node: &[usize],
) -> Option<Vec<StereoNeighbor>> {
    let permutation = octahedral_normalization_permutation(chirality?)?;
    let normalized = permutation
        .iter()
        .copied()
        .map(|index| parsed_neighbors.get(index).copied())
        .collect::<Option<Vec<_>>>()?;
    let mut best = normalized.clone();
    let mut best_key = stereo_neighbor_sequence_key(&best, new_index_of_old_node);

    for permutation in OCTAHEDRAL_OH1_EQUIVALENT_PERMUTATIONS {
        let candidate =
            permutation.iter().copied().map(|index| normalized[index]).collect::<Vec<_>>();
        let candidate_key = stereo_neighbor_sequence_key(&candidate, new_index_of_old_node);
        if candidate_key < best_key {
            best = candidate;
            best_key = candidate_key;
        }
    }

    Some(best)
}

fn minimal_cycle_rotation(
    cycle: &[StereoNeighbor],
    new_index_of_old_node: &[usize],
) -> Vec<StereoNeighbor> {
    let mut best = cycle.to_vec();
    let mut best_key = stereo_neighbor_sequence_key(&best, new_index_of_old_node);
    for rotation in 1..cycle.len() {
        let rotated =
            cycle[rotation..].iter().chain(cycle[..rotation].iter()).copied().collect::<Vec<_>>();
        let rotated_key = stereo_neighbor_sequence_key(&rotated, new_index_of_old_node);
        if rotated_key < best_key {
            best = rotated;
            best_key = rotated_key;
        }
    }
    best
}

const fn octahedral_normalization_permutation(chirality: Chirality) -> Option<[usize; 6]> {
    match chirality {
        Chirality::OH(1) => Some([0, 1, 2, 3, 4, 5]),
        Chirality::OH(2) => Some([0, 1, 4, 3, 2, 5]),
        Chirality::OH(3) => Some([0, 1, 2, 3, 5, 4]),
        Chirality::OH(4) => Some([0, 1, 2, 4, 3, 5]),
        Chirality::OH(5) => Some([0, 1, 2, 5, 3, 4]),
        Chirality::OH(6) => Some([0, 1, 2, 4, 5, 3]),
        Chirality::OH(7) => Some([0, 1, 2, 5, 4, 3]),
        Chirality::OH(8) => Some([0, 1, 3, 2, 4, 5]),
        Chirality::OH(9) => Some([0, 1, 3, 2, 5, 4]),
        Chirality::OH(10) => Some([0, 1, 4, 2, 3, 5]),
        Chirality::OH(11) => Some([0, 1, 5, 2, 3, 4]),
        Chirality::OH(12) => Some([0, 1, 4, 2, 5, 3]),
        Chirality::OH(13) => Some([0, 1, 5, 2, 4, 3]),
        Chirality::OH(14) => Some([0, 1, 3, 4, 2, 5]),
        Chirality::OH(15) => Some([0, 1, 3, 5, 2, 4]),
        Chirality::OH(16) => Some([0, 1, 5, 3, 2, 4]),
        Chirality::OH(17) => Some([0, 1, 4, 5, 2, 3]),
        Chirality::OH(18) => Some([0, 1, 5, 4, 2, 3]),
        Chirality::OH(19) => Some([0, 1, 3, 4, 5, 2]),
        Chirality::OH(20) => Some([0, 1, 3, 5, 4, 2]),
        Chirality::OH(21) => Some([0, 1, 4, 3, 5, 2]),
        Chirality::OH(22) => Some([0, 1, 5, 3, 4, 2]),
        Chirality::OH(23) => Some([0, 1, 4, 5, 3, 2]),
        Chirality::OH(24) => Some([0, 1, 5, 4, 3, 2]),
        Chirality::OH(25) => Some([0, 2, 3, 4, 5, 1]),
        Chirality::OH(26) => Some([0, 2, 3, 5, 4, 1]),
        Chirality::OH(27) => Some([0, 2, 4, 3, 5, 1]),
        Chirality::OH(28) => Some([0, 2, 5, 3, 4, 1]),
        Chirality::OH(29) => Some([0, 2, 4, 5, 3, 1]),
        Chirality::OH(30) => Some([0, 2, 5, 4, 3, 1]),
        _ => None,
    }
}

const OCTAHEDRAL_OH1_EQUIVALENT_PERMUTATIONS: [[usize; 6]; 24] = [
    [0, 1, 2, 3, 4, 5],
    [0, 2, 3, 4, 1, 5],
    [0, 3, 4, 1, 2, 5],
    [0, 4, 1, 2, 3, 5],
    [1, 0, 4, 5, 2, 3],
    [1, 2, 0, 4, 5, 3],
    [1, 4, 5, 2, 0, 3],
    [1, 5, 2, 0, 4, 3],
    [2, 0, 1, 5, 3, 4],
    [2, 1, 5, 3, 0, 4],
    [2, 3, 0, 1, 5, 4],
    [2, 5, 3, 0, 1, 4],
    [3, 0, 2, 5, 4, 1],
    [3, 2, 5, 4, 0, 1],
    [3, 4, 0, 2, 5, 1],
    [3, 5, 4, 0, 2, 1],
    [4, 0, 3, 5, 1, 2],
    [4, 1, 0, 3, 5, 2],
    [4, 3, 5, 1, 0, 2],
    [4, 5, 1, 0, 3, 2],
    [5, 1, 4, 3, 2, 0],
    [5, 2, 1, 4, 3, 0],
    [5, 3, 2, 1, 4, 0],
    [5, 4, 3, 2, 1, 0],
];
