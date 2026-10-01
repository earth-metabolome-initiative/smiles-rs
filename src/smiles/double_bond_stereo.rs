use alloc::{collections::BTreeMap, vec::Vec};
use core::cmp::Ordering;

use geometric_traits::traits::{
    SparseMatrix2D, SparseValuedMatrix2DRef, SparseValuedMatrixRef, WeisfeilerLehmanColoring,
};

use super::{
    Smiles,
    invariants::{AtomInvariant, bond_entry_code},
};
use crate::bond::{Bond, bond_edge::BondEdge};

/// Semantic double-bond stereo configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DoubleBondStereoConfig {
    /// Opposite-side alkene geometry.
    E,
    /// Same-side alkene geometry.
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DoubleBondStereoSide {
    endpoint: usize,
    reference_atom: usize,
    reference_bond_is_up: bool,
}

impl DoubleBondStereoSide {
    #[must_use]
    pub(super) fn endpoint(self) -> usize {
        self.endpoint
    }

    #[must_use]
    pub(super) fn reference_atom(self) -> usize {
        self.reference_atom
    }

    #[must_use]
    pub(super) fn reference_bond_is_up(self) -> bool {
        self.reference_bond_is_up
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DoubleBondStereoRecord {
    double_bond: BondEdge,
    side_a: DoubleBondStereoSide,
    side_b: DoubleBondStereoSide,
    config: DoubleBondStereoConfig,
}

impl DoubleBondStereoRecord {
    #[must_use]
    pub(super) fn side_a(self) -> DoubleBondStereoSide {
        self.side_a
    }

    #[must_use]
    pub(super) fn side_b(self) -> DoubleBondStereoSide {
        self.side_b
    }

    #[must_use]
    pub(super) fn config(self) -> DoubleBondStereoConfig {
        self.config
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DirectionalNeighbor {
    neighbor: usize,
    bond: Bond,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DoubleBondStereoCandidate {
    endpoint_a: usize,
    endpoint_b: usize,
    double_bond: BondEdge,
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    #[must_use]
    pub(super) fn double_bond_stereo_records(&self) -> Vec<DoubleBondStereoRecord> {
        let candidates = self.double_bond_stereo_candidates();
        if candidates.is_empty() {
            return Vec::new();
        }

        let refined_classes = self.stereo_neutral_refined_classes();
        let rooted_classes = self.stereo_neutral_rooted_classes(&refined_classes);

        self.double_bond_stereo_records_from_classes(candidates, &rooted_classes, &refined_classes)
    }

    #[must_use]
    pub(crate) fn semantic_double_bond_stereo_config(
        &self,
        node_a: usize,
        node_b: usize,
    ) -> Option<DoubleBondStereoConfig> {
        let edge_key = crate::smiles::edge_key(node_a, node_b);
        self.double_bond_stereo_records().into_iter().find_map(|record| {
            (crate::smiles::edge_key(record.double_bond.source(), record.double_bond.target())
                == edge_key)
                .then_some(record.config())
        })
    }

    #[must_use]
    pub(super) fn double_bond_stereo_records_with_planning_classes(
        &self,
        rooted_classes: &[usize],
        refined_classes: &[usize],
    ) -> Vec<DoubleBondStereoRecord> {
        let candidates = self.double_bond_stereo_candidates();
        if candidates.is_empty() {
            return Vec::new();
        }

        self.double_bond_stereo_records_from_classes(candidates, rooted_classes, refined_classes)
    }

    fn double_bond_stereo_records_from_classes(
        &self,
        candidates: Vec<DoubleBondStereoCandidate>,
        rooted_classes: &[usize],
        refined_classes: &[usize],
    ) -> Vec<DoubleBondStereoRecord> {
        candidates
            .into_iter()
            .filter_map(|candidate| {
                let side_a = self.double_bond_stereo_side(
                    candidate.endpoint_a,
                    candidate.endpoint_b,
                    rooted_classes,
                    refined_classes,
                )?;
                let side_b = self.double_bond_stereo_side(
                    candidate.endpoint_b,
                    candidate.endpoint_a,
                    rooted_classes,
                    refined_classes,
                )?;

                let left = stereo_side_parity(side_a);
                let right = stereo_side_parity(side_b);
                let config = if left == right {
                    DoubleBondStereoConfig::E
                } else {
                    DoubleBondStereoConfig::Z
                };

                Some(DoubleBondStereoRecord {
                    double_bond: candidate.double_bond,
                    side_a,
                    side_b,
                    config,
                })
            })
            .collect()
    }

    fn double_bond_stereo_candidates(&self) -> Vec<DoubleBondStereoCandidate> {
        if !self.has_directional_single_bonds() || !self.has_double_bonds() {
            return Vec::new();
        }
        // Ring membership is cheaper and clearer than rerunning a bespoke
        // reachability search per double bond candidate.
        let ring_membership = self.ring_membership();

        self.bond_matrix()
            .sparse_entries()
            .filter_map(|((row, column), entry)| {
                (row < column && non_aromatic_double_bond(*entry)).then_some((
                    row,
                    column,
                    entry.to_bond_edge(row, column),
                ))
            })
            .filter_map(|(a, b, double_bond)| {
                if !self.double_bond_supports_semantic_stereo(a, b) {
                    return None;
                }
                self.has_directional_neighbor(a, b)?;
                self.has_directional_neighbor(b, a)?;
                if ring_membership.contains_edge(a, b) {
                    return None;
                }
                Some(DoubleBondStereoCandidate { endpoint_a: a, endpoint_b: b, double_bond })
            })
            .collect()
    }

    fn has_directional_single_bonds(&self) -> bool {
        self.bond_matrix()
            .sparse_entries()
            .any(|((_row, _column), entry)| matches!(entry.bond(), Bond::Up | Bond::Down))
    }

    fn has_double_bonds(&self) -> bool {
        self.bond_matrix()
            .sparse_entries()
            .any(|((_row, _column), entry)| non_aromatic_double_bond(*entry))
    }

    fn double_bond_supports_semantic_stereo(&self, node_a: usize, node_b: usize) -> bool {
        self.non_single_family_bond_count(node_a) == 1
            && self.non_single_family_bond_count(node_b) == 1
    }

    fn non_single_family_bond_count(&self, node_id: usize) -> usize {
        self.bond_matrix()
            .sparse_row_values_ref(node_id)
            .filter(|entry| non_single_family_bond(**entry))
            .count()
    }

    fn double_bond_stereo_side(
        &self,
        endpoint: usize,
        opposite_endpoint: usize,
        rooted_classes: &[usize],
        refined_classes: &[usize],
    ) -> Option<DoubleBondStereoSide> {
        let reference_atom = self.choose_reference_substituent(
            endpoint,
            opposite_endpoint,
            rooted_classes,
            refined_classes,
        )?;
        let reference_bond_is_up =
            self.reference_bond_is_up(endpoint, opposite_endpoint, reference_atom)?;
        Some(DoubleBondStereoSide { endpoint, reference_atom, reference_bond_is_up })
    }

    fn directional_neighbors(
        &self,
        endpoint: usize,
        opposite_endpoint: usize,
    ) -> Vec<DirectionalNeighbor> {
        self.bond_matrix()
            .sparse_row(endpoint)
            .zip(self.bond_matrix().sparse_row_values_ref(endpoint))
            .filter_map(|(neighbor, entry)| {
                if neighbor == opposite_endpoint {
                    return None;
                }
                match entry.bond() {
                    Bond::Up | Bond::Down => {
                        Some(DirectionalNeighbor { neighbor, bond: entry.bond() })
                    }
                    _ => None,
                }
            })
            .collect()
    }

    fn has_directional_neighbor(&self, endpoint: usize, opposite_endpoint: usize) -> Option<()> {
        (!self.directional_neighbors(endpoint, opposite_endpoint).is_empty()).then_some(())
    }

    fn reference_bond_is_up(
        &self,
        endpoint: usize,
        opposite_endpoint: usize,
        reference_atom: usize,
    ) -> Option<bool> {
        let directional_neighbors = self.directional_neighbors(endpoint, opposite_endpoint);

        if let Some(reference_direction) =
            directional_neighbors.iter().find(|directional| directional.neighbor == reference_atom)
        {
            return Some(matches!(reference_direction.bond, Bond::Up));
        }

        let mut parities = directional_neighbors.into_iter().map(|directional| {
            let mut parity = matches!(directional.bond, Bond::Up);
            if reference_atom != directional.neighbor {
                parity = !parity;
            }
            parity
        });

        let first = parities.next()?;
        parities.all(|parity| parity == first).then_some(first)
    }

    fn choose_reference_substituent(
        &self,
        endpoint: usize,
        opposite_endpoint: usize,
        rooted_classes: &[usize],
        refined_classes: &[usize],
    ) -> Option<usize> {
        let neighbors: Vec<usize> = self
            .bond_matrix()
            .sparse_row(endpoint)
            .filter(|&neighbor| neighbor != opposite_endpoint)
            .collect();
        let (&first, rest) = neighbors.split_first()?;
        let mut best = first;
        let mut best_key =
            substituent_priority_key(self, endpoint, best, rooted_classes, refined_classes);
        let mut unique_best = true;

        for &candidate in rest {
            let candidate_key = substituent_priority_key(
                self,
                endpoint,
                candidate,
                rooted_classes,
                refined_classes,
            );
            match candidate_key.cmp(&best_key) {
                Ordering::Greater => {
                    best = candidate;
                    best_key = candidate_key;
                    unique_best = true;
                }
                Ordering::Equal => {
                    unique_best = false;
                }
                Ordering::Less => {}
            }
        }

        unique_best.then_some(best)
    }

    pub(super) fn stereo_neutral_refined_classes(&self) -> Vec<usize> {
        let seed_colors: Vec<StereoNeutralAtomInvariantKey> =
            self.atom_invariants().into_iter().map(StereoNeutralAtomInvariantKey::from).collect();
        self.wl_coloring_with_seed_and_edge_colors(&seed_colors, |node, neighbor| {
            let entry =
                self.bond_entry_for_node_pair((node, neighbor)).unwrap_or_else(|| unreachable!());
            usize::from(bond_entry_code(entry))
        })
    }

    pub(super) fn stereo_neutral_rooted_classes(&self, initial_classes: &[usize]) -> Vec<usize> {
        let node_count = initial_classes.len();
        if node_count == 0 {
            return Vec::new();
        }

        let (directed_edges, directed_edge_ids) = self.stereo_neutral_directed_edges();
        if directed_edges.is_empty() {
            return initial_classes.to_vec();
        }

        let mut edge_classes: Vec<usize> =
            directed_edges.iter().map(|edge| initial_classes[edge.to]).collect();

        loop {
            let next_keys: Vec<StereoNeutralDirectedEdgeKey> = directed_edges
                .iter()
                .map(|edge| {
                    let mut neighborhood: Vec<(usize, usize)> = self
                        .bond_matrix()
                        .sparse_row(edge.to)
                        .zip(self.bond_matrix().sparse_row_values_ref(edge.to))
                        .filter_map(|(neighbor, entry)| {
                            if neighbor == edge.from {
                                None
                            } else {
                                Some((
                                    usize::from(bond_entry_code(*entry)),
                                    edge_classes[*directed_edge_ids
                                        .get(&(edge.to, neighbor))
                                        .unwrap_or_else(|| unreachable!())],
                                ))
                            }
                        })
                        .collect();
                    neighborhood.sort_unstable();
                    StereoNeutralDirectedEdgeKey {
                        node_class: initial_classes[edge.to],
                        neighborhood,
                    }
                })
                .collect();

            let next_classes = dense_ranks(&next_keys);
            if next_classes == edge_classes {
                break;
            }
            edge_classes = next_classes;
        }

        let rooted_keys: Vec<StereoNeutralRootedNodeKey> = (0..node_count)
            .map(|node_id| {
                let mut neighborhood: Vec<(usize, usize)> = self
                    .bond_matrix()
                    .sparse_row(node_id)
                    .zip(self.bond_matrix().sparse_row_values_ref(node_id))
                    .map(|(neighbor, entry)| {
                        (
                            usize::from(bond_entry_code(*entry)),
                            edge_classes[*directed_edge_ids
                                .get(&(node_id, neighbor))
                                .unwrap_or_else(|| unreachable!())],
                        )
                    })
                    .collect();
                neighborhood.sort_unstable();
                StereoNeutralRootedNodeKey { node_class: initial_classes[node_id], neighborhood }
            })
            .collect();

        dense_ranks(&rooted_keys)
    }

    fn stereo_neutral_directed_edges(
        &self,
    ) -> (Vec<DirectedEdge>, BTreeMap<(usize, usize), usize>) {
        let mut directed_edges = Vec::with_capacity(self.number_of_bonds() * 2);
        let mut directed_edge_ids = BTreeMap::new();

        for from in 0..self.nodes().len() {
            for (to, _entry) in self
                .bond_matrix()
                .sparse_row(from)
                .zip(self.bond_matrix().sparse_row_values_ref(from))
            {
                let directed_edge_id = directed_edges.len();
                directed_edges.push(DirectedEdge { from, to });
                directed_edge_ids.insert((from, to), directed_edge_id);
            }
        }

        (directed_edges, directed_edge_ids)
    }
}

fn stereo_side_parity(side: DoubleBondStereoSide) -> bool {
    side.reference_bond_is_up()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SubstituentPriorityKey {
    atomic_number: u8,
    isotope_mass_number: Option<u16>,
    charge: i8,
    aromatic: bool,
    bond_order_to_endpoint: u8,
    rooted_class: usize,
    refined_class: usize,
}

impl Ord for SubstituentPriorityKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.atomic_number
            .cmp(&other.atomic_number)
            .then_with(|| self.isotope_mass_number.cmp(&other.isotope_mass_number))
            .then_with(|| self.charge.cmp(&other.charge))
            .then_with(|| self.aromatic.cmp(&other.aromatic))
            .then_with(|| self.bond_order_to_endpoint.cmp(&other.bond_order_to_endpoint))
            .then_with(|| self.rooted_class.cmp(&other.rooted_class))
            .then_with(|| self.refined_class.cmp(&other.refined_class))
    }
}

impl PartialOrd for SubstituentPriorityKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn substituent_priority_key(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    endpoint: usize,
    neighbor: usize,
    rooted_classes: &[usize],
    refined_classes: &[usize],
) -> SubstituentPriorityKey {
    let atom = smiles.node_by_id(neighbor).unwrap_or_else(|| unreachable!());
    let atomic_number = atom.element().map_or(0, u8::from);
    let edge = smiles.edge_for_node_pair((endpoint, neighbor)).unwrap_or_else(|| unreachable!());
    let bond_order_to_endpoint = if edge.is_aromatic() { 1 } else { bond_priority(edge.bond()) };

    SubstituentPriorityKey {
        atomic_number,
        isotope_mass_number: atom.isotope_mass_number(),
        charge: atom.charge_value(),
        aromatic: atom.aromatic(),
        bond_order_to_endpoint,
        rooted_class: rooted_classes[neighbor],
        refined_class: refined_classes[neighbor],
    }
}

fn bond_priority(bond: Bond) -> u8 {
    match bond {
        Bond::Single | Bond::Up | Bond::Down => 1,
        Bond::Double => 2,
        Bond::Triple => 3,
        Bond::Quadruple => 4,
    }
}

#[inline]
fn non_aromatic_double_bond(entry: super::BondEntry) -> bool {
    entry.bond() == Bond::Double && !entry.aromatic()
}

#[inline]
fn non_single_family_bond(entry: super::BondEntry) -> bool {
    entry.aromatic() || !matches!(entry.bond(), Bond::Single | Bond::Up | Bond::Down)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct StereoNeutralAtomInvariantKey {
    syntax: u8,
    symbol_kind: u8,
    element_symbol: &'static str,
    isotope_mass_number: Option<u16>,
    aromatic: bool,
    hydrogens: u8,
    charge: i8,
    class: u16,
    chirality_kind: u8,
    chirality_value: u8,
    degree: usize,
    bond_kind_histogram: [usize; 5],
}

impl From<AtomInvariant> for StereoNeutralAtomInvariantKey {
    fn from(value: AtomInvariant) -> Self {
        let (symbol_kind, element_symbol) = match value.symbol.element() {
            Some(element) => (0, element.symbol()),
            None => (1, ""),
        };
        let (chirality_kind, chirality_value) = match value.chirality {
            None => (0, 0),
            Some(crate::atom::bracketed::chirality::Chirality::At) => (1, 0),
            Some(crate::atom::bracketed::chirality::Chirality::AtAt) => (2, 0),
            Some(crate::atom::bracketed::chirality::Chirality::TH(value)) => (3, value),
            Some(crate::atom::bracketed::chirality::Chirality::AL(value)) => (4, value),
            Some(crate::atom::bracketed::chirality::Chirality::SP(value)) => (5, value),
            Some(crate::atom::bracketed::chirality::Chirality::TB(value)) => (6, value),
            Some(crate::atom::bracketed::chirality::Chirality::OH(value)) => (7, value),
        };

        Self {
            syntax: match value.syntax {
                crate::atom::AtomSyntax::OrganicSubset => 0,
                crate::atom::AtomSyntax::Bracket => 1,
            },
            symbol_kind,
            element_symbol,
            isotope_mass_number: value.isotope_mass_number,
            aromatic: value.aromatic,
            hydrogens: value.hydrogens,
            charge: value.charge,
            class: value.class,
            chirality_kind,
            chirality_value,
            degree: value.degree,
            bond_kind_histogram: [
                value.bond_kind_histogram.count(Bond::Single)
                    + value.bond_kind_histogram.count(Bond::Up)
                    + value.bond_kind_histogram.count(Bond::Down),
                value.bond_kind_histogram.count(Bond::Double),
                value.bond_kind_histogram.count(Bond::Triple),
                value.bond_kind_histogram.count(Bond::Quadruple),
                value.bond_kind_histogram.aromatic_count(),
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DirectedEdge {
    from: usize,
    to: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct StereoNeutralDirectedEdgeKey {
    node_class: usize,
    neighborhood: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct StereoNeutralRootedNodeKey {
    node_class: usize,
    neighborhood: Vec<(usize, usize)>,
}

fn dense_ranks<K>(keys: &[K]) -> Vec<usize>
where
    K: Ord + Clone,
{
    let mut unique = keys.to_vec();
    unique.sort_unstable();
    unique.dedup();
    keys.iter().map(|key| unique.binary_search(key).unwrap_or_else(|_| unreachable!())).collect()
}

#[cfg(test)]
mod tests {
    use alloc::{string::ToString, vec::Vec};

    use super::{DoubleBondStereoConfig, Smiles};

    type SemanticDoubleBondStereoSignature =
        ([usize; 2], [(usize, usize); 2], DoubleBondStereoConfig);

    fn parse(smiles: &str) -> Smiles {
        smiles.parse().unwrap()
    }

    fn semantic_double_bond_stereo_signature(
        smiles: &Smiles,
    ) -> Vec<SemanticDoubleBondStereoSignature> {
        let refined = smiles.stereo_neutral_refined_classes();
        let rooted = smiles.stereo_neutral_rooted_classes(&refined);
        let mut signature = smiles
            .double_bond_stereo_records()
            .into_iter()
            .map(|record| {
                let mut endpoints =
                    [rooted[record.side_a().endpoint()], rooted[record.side_b().endpoint()]];
                endpoints.sort_unstable();

                let mut sides = [
                    (rooted[record.side_a().endpoint()], rooted[record.side_a().reference_atom()]),
                    (rooted[record.side_b().endpoint()], rooted[record.side_b().reference_atom()]),
                ];
                sides.sort_unstable();

                (endpoints, sides, record.config())
            })
            .collect::<Vec<_>>();
        signature.sort_unstable();
        signature
    }

    #[test]
    fn double_bond_stereo_of_empty_graph_is_empty() {
        assert!(
            Smiles::<crate::smiles::ConcreteAtoms>::new_for_policy()
                .double_bond_stereo_records()
                .is_empty()
        );
    }

    #[test]
    fn double_bond_stereo_matches_simple_rdkit_e_fixtures() {
        for smiles in ["F/C=C/F", "C/C=C/C", "C/C=C(/F)C"] {
            let records = parse(smiles).double_bond_stereo_records();
            assert_eq!(records.len(), 1, "{smiles}");
            assert_eq!(records[0].config(), DoubleBondStereoConfig::E, "{smiles}");
        }
    }

    #[test]
    fn double_bond_stereo_matches_simple_rdkit_z_fixtures() {
        for smiles in ["F/C=C\\F", "C/C=C\\C", "CC/C(Cl)=C(/F)C"] {
            let records = parse(smiles).double_bond_stereo_records();
            assert_eq!(records.len(), 1, "{smiles}");
            assert_eq!(records[0].config(), DoubleBondStereoConfig::Z, "{smiles}");
        }
    }

    #[test]
    fn double_bond_stereo_selects_reference_substituents_not_just_directional_neighbors() {
        let record = parse("CC/C(Cl)=C(/F)C").double_bond_stereo_records()[0];
        assert_eq!(record.side_a().reference_atom(), 3);
        assert_eq!(record.side_b().reference_atom(), 5);
        assert!(!record.side_a().reference_bond_is_up());
        assert!(record.side_b().reference_bond_is_up());
        assert_eq!(record.config(), DoubleBondStereoConfig::Z);
    }

    #[test]
    fn double_bond_stereo_omits_non_stereogenic_ring_alkene() {
        assert!(parse("C1CC/C=C/CC1").double_bond_stereo_records().is_empty());
    }

    #[test]
    fn double_bond_stereo_omits_cumulene_like_directional_case() {
        assert!(parse("C=C(=C/CO)/C#N").double_bond_stereo_records().is_empty());
    }

    #[test]
    fn double_bond_stereo_polyene_matches_rdkit_stereo_count() {
        let records = parse(
            "CC1CCC/C(C)=C1/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)/CCCC2(C)C",
        )
        .double_bond_stereo_records();
        assert_eq!(records.len(), 9);
        assert!(records.iter().all(|record| record.config() == DoubleBondStereoConfig::E));
    }

    #[test]
    fn render_roundtrip_preserves_semantic_double_bond_stereo() {
        for input in [
            "F/C=C/F",
            "F/C=C\\F",
            "CC/C(Cl)=C(/F)C",
            "C(=C/Cl)\\C=C\\Cl",
            "CC1CCC/C(C)=C1/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(\\C)CO",
            "CbbbbbbbbbbbbbbbbbbbbC/C=C\\C/C=C\\C/C=C\\CBrBrSbbbbbC#CC#CC#CC#CC#CC#CC#CC#CCCCCCCCC/C=C/C=C/CCcC/C=C/C(=C/C=C/C)CCC#CC#C",
            "C=C\\1C=CC=C/OSONNNNNbcNNN:NNNNNNNNNNNC1=C-2\\C=ONNNNNbcNNN:NNNNNNNNNNNC2=C-2\\C=CCC2",
        ] {
            let parsed = parse(input);
            let rendered = parsed.to_string();
            let reparsed = parse(&rendered);
            assert_eq!(
                semantic_double_bond_stereo_signature(&parsed),
                semantic_double_bond_stereo_signature(&reparsed),
                "semantic double-bond stereo changed after render roundtrip\ninput={input}\nrendered={rendered}",
            );
        }
    }
}
