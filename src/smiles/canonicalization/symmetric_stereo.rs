//! Canonical numbering by the smallest stereo image.

use alloc::vec::Vec;

use geometric_traits::{
    impls::PermutationGroup,
    traits::{CanonicalLabeling, CanonicalLabelingOptions, SparseMatrix2D, SparseValuedMatrixRef},
};

use super::{
    SmilesCanonicalLabeling,
    state::{canonical_chirality_key, stereo_neutral_canonical_atom_label},
};
use crate::{
    atom::bracketed::chirality::Chirality,
    bond::Bond,
    smiles::{
        Smiles, SmilesAtomPolicy, StereoNeighbor, double_bond_stereo::DoubleBondStereoConfig,
        edge_key, invariants::bond_entry_code, render_plan::preserve_raw_directional_single_around,
    },
};

/// Atom parity and raw bond directions that depend on traversal order.
struct NumberingRelativeStereo {
    parity_centers: Vec<bool>,
    raw_directional_edges: Vec<(usize, usize)>,
}

impl<AtomPolicy: SmilesAtomPolicy> Smiles<AtomPolicy> {
    /// Returns the smallest stereo image numbering fixing `root`.
    pub(crate) fn symmetric_stereo_labeling(
        &self,
        root: Option<usize>,
        rooted_classes: &[usize],
        refined_classes: &[usize],
    ) -> SmilesCanonicalLabeling {
        let records =
            self.double_bond_stereo_records_with_planning_classes(rooted_classes, refined_classes);
        let mut semantic_endpoint = alloc::vec![false; self.nodes().len()];
        let mut double_bond_configs: Vec<((usize, usize), DoubleBondStereoConfig)> = records
            .iter()
            .map(|record| {
                let (a, b) = (record.side_a().endpoint(), record.side_b().endpoint());
                semantic_endpoint[a] = true;
                semantic_endpoint[b] = true;
                (edge_key(a, b), record.config())
            })
            .collect();
        double_bond_configs.sort_unstable();
        let stereo = self.numbering_relative_stereo(&semantic_endpoint);

        let result = CanonicalLabeling::canonical_labeling_with_options(
            self,
            |node| {
                let atom = self.nodes()[node];
                let chirality = if stereo.parity_centers[node] {
                    (true, tetrahedral_family(atom.chirality()), 0)
                } else {
                    let (kind, value) = canonical_chirality_key(atom.chirality());
                    (false, kind, value)
                };
                (Some(node) != root, stereo_neutral_canonical_atom_label(atom), chirality)
            },
            |a, b| {
                let key = edge_key(a, b);
                let entry = self
                    .bond_entry_for_node_pair(key)
                    .unwrap_or_else(|| unreachable!("canonizer only queries existing edges"));
                let config = double_bond_configs
                    .binary_search_by_key(&key, |&(edge, _)| edge)
                    .ok()
                    .map(|index| double_bond_configs[index].1);
                let raw_directional = stereo.raw_directional_edges.binary_search(&key).is_ok();
                (bond_entry_code(entry), config, raw_directional)
            },
            CanonicalLabelingOptions {
                collect_automorphism_generators: true,
                ..Default::default()
            },
        );
        let skeleton_labeling = SmilesCanonicalLabeling::new(result.order);
        if result.automorphism_generators.is_empty()
            || (!stereo.parity_centers.contains(&true) && stereo.raw_directional_edges.is_empty())
        {
            return skeleton_labeling;
        }

        let canonical = self.exact_canonicalize_with_labeling(&skeleton_labeling);
        let new_index = skeleton_labeling.new_index_of_old_node();
        let generators = result.automorphism_generators.iter().map(|generator| {
            let mut conjugated = alloc::vec![0; generator.len()];
            for (old, &image) in generator.iter().enumerate() {
                conjugated[new_index[old]] = new_index[image];
            }
            conjugated
        });
        let parity_centers: Vec<bool> =
            skeleton_labeling.order().iter().map(|&old| stereo.parity_centers[old]).collect();
        let mut raw_directional_edges: Vec<(usize, usize)> = stereo
            .raw_directional_edges
            .iter()
            .map(|&(a, b)| edge_key(new_index[a], new_index[b]))
            .collect();
        raw_directional_edges.sort_unstable();

        let automorphism = canonical.stereo_smallest_image_automorphism(
            generators,
            &parity_centers,
            &raw_directional_edges,
        );
        let mut order = alloc::vec![0; automorphism.len()];
        for (old, &canonical_index) in new_index.iter().enumerate() {
            order[automorphism[canonical_index]] = old;
        }
        SmilesCanonicalLabeling::new(order)
    }

    fn numbering_relative_stereo(&self, semantic_endpoint: &[bool]) -> NumberingRelativeStereo {
        let parity_centers =
            (0..self.nodes().len()).map(|node| self.renders_parity_relative(node)).collect();
        let raw_directional_edges = self
            .bond_matrix()
            .sparse_entries()
            .filter(|&((row, column), entry)| {
                row < column
                    && matches!(entry.bond(), Bond::Up | Bond::Down)
                    && preserve_raw_directional_single_around(
                        self,
                        |node| semantic_endpoint[node],
                        row,
                        column,
                    )
            })
            .map(|((row, column), _entry)| (row, column))
            .collect();
        NumberingRelativeStereo { parity_centers, raw_directional_edges }
    }

    /// Whether the emitted neighbors support tag parity.
    fn renders_parity_relative(&self, node: usize) -> bool {
        let atom = self.nodes()[node];
        if tetrahedral_parity(atom.chirality()).is_none() {
            return false;
        }
        let mut emitted: Vec<StereoNeighbor> =
            self.bond_matrix().sparse_row(node).map(StereoNeighbor::Atom).collect();
        if atom.hydrogen_count() == 1 {
            emitted.push(StereoNeighbor::ExplicitHydrogen);
        }
        let mut parsed = self.parsed_stereo_neighbors_row(node).to_vec();
        emitted.sort_unstable();
        parsed.sort_unstable();
        parsed == emitted
    }

    /// Returns a skeleton automorphism minimizing the stereo points.
    fn stereo_smallest_image_automorphism(
        &self,
        generators: impl Iterator<Item = Vec<usize>>,
        parity_centers: &[bool],
        raw_directional_edges: &[(usize, usize)],
    ) -> Vec<usize> {
        let node_count = self.nodes().len();
        let degree = 2 * node_count + 2 * raw_directional_edges.len();
        let induced = generators.map(|generator| {
            let mut permutation = alloc::vec![0; degree];
            for node in 0..node_count {
                let flip = usize::from(self.neighbor_order_flips(node, &generator));
                permutation[2 * node] = 2 * generator[node] + flip;
                permutation[2 * node + 1] = 2 * generator[node] + 1 - flip;
            }
            for (index, &(a, b)) in raw_directional_edges.iter().enumerate() {
                let image = raw_directional_edges
                    .binary_search(&edge_key(generator[a], generator[b]))
                    .unwrap_or_else(|_| unreachable!("automorphisms keep raw directional edges"));
                let flip = usize::from(generator[a] > generator[b]);
                permutation[2 * node_count + 2 * index] = 2 * node_count + 2 * image + flip;
                permutation[2 * node_count + 2 * index + 1] = 2 * node_count + 2 * image + 1 - flip;
            }
            permutation
        });
        let group = PermutationGroup::new(degree, induced)
            .unwrap_or_else(|_| unreachable!("induced generators are permutations of the points"));

        let mut points: Vec<usize> = (0..node_count)
            .filter(|&node| parity_centers[node])
            .map(|node| 2 * node + usize::from(self.parity_bit(node)))
            .collect();
        points.extend(raw_directional_edges.iter().enumerate().map(|(index, &(a, b))| {
            let entry = self
                .bond_entry_for_node_pair((a, b))
                .unwrap_or_else(|| unreachable!("raw directional edges exist"));
            2 * node_count + 2 * index + usize::from(entry.bond() == Bond::Down)
        }));
        let smallest = group
            .smallest_image(&points)
            .unwrap_or_else(|_| unreachable!("stereo points are distinct and in range"));
        (0..node_count).map(|node| smallest.element[2 * node] / 2).collect()
    }

    /// Combines tag parity with parsed neighbor inversion parity.
    fn parity_bit(&self, node: usize) -> bool {
        let tag = tetrahedral_parity(self.nodes()[node].chirality())
            .unwrap_or_else(|| unreachable!("parity centers carry an invertible tag"));
        tag ^ odd_inversions(self.parsed_stereo_neighbors_row(node))
    }

    /// Whether `automorphism` reverses sorted neighbor parity.
    fn neighbor_order_flips(&self, node: usize, automorphism: &[usize]) -> bool {
        let images: Vec<usize> =
            self.bond_matrix().sparse_row(node).map(|neighbor| automorphism[neighbor]).collect();
        odd_inversions(&images)
    }
}

/// `Some(false)` for `@`, `@TH1` and `@AL1`, `Some(true)` for their inverses,
/// and `None` for tags `render` prints unchanged.
const fn tetrahedral_parity(chirality: Option<Chirality>) -> Option<bool> {
    match chirality {
        Some(Chirality::At | Chirality::TH(1) | Chirality::AL(1)) => Some(false),
        Some(Chirality::AtAt | Chirality::TH(2) | Chirality::AL(2)) => Some(true),
        _ => None,
    }
}

/// The tag family of a parity center, the same for a tag and its inverse.
const fn tetrahedral_family(chirality: Option<Chirality>) -> u8 {
    match chirality {
        Some(Chirality::TH(_)) => 2,
        Some(Chirality::AL(_)) => 3,
        _ => 1,
    }
}

fn odd_inversions<T: Ord>(sequence: &[T]) -> bool {
    let mut odd = false;
    for (index, right) in sequence.iter().enumerate() {
        for left in &sequence[..index] {
            odd ^= left > right;
        }
    }
    odd
}
