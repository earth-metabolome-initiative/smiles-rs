use alloc::vec::Vec;

use geometric_traits::traits::SparseValuedMatrix2DRef;

use super::{BondEntry, Smiles};
use crate::{
    atom::{AtomSyntax, atom_symbol::AtomSymbol, bracketed::chirality::Chirality},
    bond::{Bond, BondDescriptor},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub(crate) struct BondKindHistogram {
    counts: [usize; 4],
    aromatic_count: usize,
}

impl BondKindHistogram {
    #[inline]
    pub(crate) fn record(&mut self, entry: BondEntry) {
        // The single/double kekule order retained on an aromatic bond is
        // non-semantic, so an aromatic bond is counted only as aromatic and
        // never under its kekule kind. This keeps the histogram (and every
        // ordering derived from it) identical for two representations that
        // differ only in the kekule assignment of their aromatic bonds.
        if entry.aromatic() {
            self.aromatic_count += 1;
        } else {
            self.counts[bond_kind_index(entry.bond())] += 1;
        }
    }

    #[inline]
    #[must_use]
    pub(crate) fn count(self, bond: Bond) -> usize {
        self.counts[bond_kind_index(bond)]
    }

    #[inline]
    #[must_use]
    pub(crate) const fn aromatic_count(self) -> usize {
        self.aromatic_count
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub(crate) struct AtomInvariant {
    pub(crate) syntax: AtomSyntax,
    pub(crate) symbol: AtomSymbol,
    pub(crate) isotope_mass_number: Option<u16>,
    pub(crate) aromatic: bool,
    pub(crate) hydrogens: u8,
    pub(crate) charge: i8,
    pub(crate) class: u16,
    pub(crate) chirality: Option<Chirality>,
    pub(crate) degree: usize,
    pub(crate) bond_kind_histogram: BondKindHistogram,
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    #[inline]
    #[must_use]
    pub(crate) fn atom_invariant(&self, node_id: usize) -> Option<AtomInvariant> {
        let atom = *self.node_by_id(node_id)?;
        let mut bond_kind_histogram = BondKindHistogram::default();
        let mut degree = 0;
        for entry in self.bond_matrix.sparse_row_values_ref(node_id) {
            degree += 1;
            bond_kind_histogram.record(*entry);
        }

        Some(AtomInvariant {
            syntax: atom.syntax(),
            symbol: atom.symbol(),
            isotope_mass_number: atom.isotope_mass_number(),
            aromatic: atom.aromatic(),
            hydrogens: atom.hydrogen_count(),
            charge: atom.charge_value(),
            class: atom.class(),
            chirality: atom.chirality(),
            degree,
            bond_kind_histogram,
        })
    }

    #[inline]
    #[must_use]
    pub(crate) fn atom_invariants(&self) -> Vec<AtomInvariant> {
        (0..self.atom_nodes.len()).map(|node_id| self.atom_invariant(node_id).unwrap()).collect()
    }
}

#[inline]
pub(crate) const fn bond_kind_index(bond: Bond) -> usize {
    bond_kind_code(bond) as usize
}

#[inline]
pub(crate) const fn bond_kind_code(bond: Bond) -> u8 {
    match bond {
        Bond::Single | Bond::Up | Bond::Down => 0,
        Bond::Double => 1,
        Bond::Triple => 2,
        Bond::Quadruple => 3,
    }
}

#[inline]
pub(crate) const fn bond_descriptor_code(descriptor: BondDescriptor) -> u8 {
    // An aromatic bond collapses to a single canonical code: its kekule order
    // must not influence ordering or signatures (see `BondKindHistogram`).
    if descriptor.is_aromatic() {
        return 8;
    }
    bond_kind_code(descriptor.bond())
}

#[inline]
pub(crate) const fn bond_descriptor_index(descriptor: BondDescriptor) -> usize {
    bond_descriptor_code(descriptor) as usize
}

#[inline]
pub(crate) const fn bond_entry_code(entry: BondEntry) -> u8 {
    // An aromatic bond collapses to a single canonical code: its kekule order
    // must not influence ordering or signatures (see `BondKindHistogram`).
    if entry.aromatic() {
        return 8;
    }
    bond_kind_code(entry.bond())
}

#[inline]
pub(crate) const fn planning_chirality_key(chirality: Option<Chirality>) -> (u8, u8) {
    match chirality {
        None => (0, 0),
        Some(Chirality::At | Chirality::AtAt) => (1, 0),
        Some(Chirality::TH(_)) => (2, 0),
        Some(Chirality::AL(_)) => (3, 0),
        Some(Chirality::SP(_)) => (4, 0),
        Some(Chirality::TB(_)) => (5, 0),
        Some(Chirality::OH(_)) => (6, 0),
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use elements_rs::Element;

    use super::{
        AtomInvariant, BondEntry, BondKindHistogram, Smiles, bond_entry_code, bond_kind_code,
        bond_kind_index, planning_chirality_key,
    };
    use crate::{
        atom::{
            Atom, AtomSyntax,
            atom_symbol::AtomSymbol,
            bracketed::{charge::Charge, chirality::Chirality},
        },
        bond::{
            Bond, BondDescriptor,
            bond_edge::{BondEdge, bond_edge},
        },
        smiles::BondMatrixBuilder,
    };

    fn smiles_from_edges(atom_nodes: Vec<Atom>, bond_edges: &[BondEdge]) -> Smiles {
        let mut builder = BondMatrixBuilder::with_capacity(bond_edges.len());
        for edge in bond_edges {
            builder
                .push_edge_with_descriptor(
                    edge.source(),
                    edge.target(),
                    edge.descriptor(),
                    edge.ring_num(),
                )
                .unwrap();
        }
        let number_of_nodes = atom_nodes.len();
        Smiles::from_bond_matrix_parts(atom_nodes, builder.finish(number_of_nodes))
    }

    #[test]
    fn atom_invariants_of_empty_graph_are_empty() {
        assert!(
            Smiles::<crate::smiles::ConcreteAtoms>::new_for_policy().atom_invariants().is_empty()
        );
    }

    #[test]
    fn atom_invariant_captures_local_atom_fields() {
        let atom = Atom::builder()
            .with_symbol(AtomSymbol::Element(Element::N))
            .with_isotope(15)
            .with_aromatic(true)
            .with_hydrogens(2)
            .with_charge(Charge::try_new(2).unwrap())
            .with_class(17)
            .with_chirality(Chirality::TH(2))
            .build();
        let smiles = smiles_from_edges(vec![atom], &[]);

        let invariant = smiles.atom_invariant(0).unwrap();
        assert_eq!(
            invariant,
            AtomInvariant {
                syntax: AtomSyntax::Bracket,
                symbol: AtomSymbol::Element(Element::N),
                isotope_mass_number: Some(15),
                aromatic: true,
                hydrogens: 2,
                charge: 2,
                class: 17,
                chirality: Some(Chirality::TH(2)),
                degree: 0,
                bond_kind_histogram: BondKindHistogram::default(),
            }
        );
    }

    #[test]
    fn atom_invariants_distinguish_organic_subset_bracket_and_wildcard_atoms() {
        let organic = Atom::new_organic_subset(AtomSymbol::Element(Element::C), false);
        let bracket = Atom::builder().with_symbol(AtomSymbol::Element(Element::C)).build();
        let wildcard = Atom::builder().with_symbol(AtomSymbol::WildCard).build();
        let smiles = smiles_from_edges(vec![organic, bracket, wildcard], &[]);

        let invariants = smiles.atom_invariants();
        assert_ne!(invariants[0], invariants[1]);
        assert_ne!(invariants[1], invariants[2]);
        assert_eq!(invariants[0].syntax, AtomSyntax::OrganicSubset);
        assert_eq!(invariants[1].syntax, AtomSyntax::Bracket);
        assert_eq!(invariants[2].symbol, AtomSymbol::WildCard);
    }

    #[test]
    fn atom_invariant_captures_degree_and_incident_bond_kinds() {
        let smiles = smiles_from_edges(
            vec![
                Atom::new_organic_subset(AtomSymbol::Element(Element::C), false),
                Atom::new_organic_subset(AtomSymbol::Element(Element::O), false),
                Atom::new_organic_subset(AtomSymbol::Element(Element::N), false),
                Atom::new_organic_subset(AtomSymbol::Element(Element::S), true),
            ],
            &[
                bond_edge(0, 1, Bond::Single, None),
                bond_edge(0, 2, Bond::Double, None),
                crate::bond::bond_edge::bond_edge_with_aromaticity(0, 3, Bond::Single, None, true),
            ],
        );

        let invariant = smiles.atom_invariant(0).unwrap();
        assert_eq!(invariant.degree, 3);
        // The aromatic bond is counted only as aromatic, never under its kekule
        // kind, so the plain single bond is the only one in the Single bucket.
        assert_eq!(invariant.bond_kind_histogram.count(Bond::Single), 1);
        assert_eq!(invariant.bond_kind_histogram.count(Bond::Double), 1);
        assert_eq!(invariant.bond_kind_histogram.aromatic_count(), 1);
    }

    #[test]
    fn atom_invariants_distinguish_equal_degree_but_different_bond_histograms() {
        let single_single = smiles_from_edges(
            vec![
                Atom::new_organic_subset(AtomSymbol::Element(Element::C), false),
                Atom::new_organic_subset(AtomSymbol::Element(Element::O), false),
                Atom::new_organic_subset(AtomSymbol::Element(Element::N), false),
            ],
            &[bond_edge(0, 1, Bond::Single, None), bond_edge(0, 2, Bond::Single, None)],
        );
        let single_double = smiles_from_edges(
            vec![
                Atom::new_organic_subset(AtomSymbol::Element(Element::C), false),
                Atom::new_organic_subset(AtomSymbol::Element(Element::O), false),
                Atom::new_organic_subset(AtomSymbol::Element(Element::N), false),
            ],
            &[bond_edge(0, 1, Bond::Single, None), bond_edge(0, 2, Bond::Double, None)],
        );

        assert_ne!(single_single.atom_invariant(0), single_double.atom_invariant(0));
    }

    #[test]
    fn atom_invariant_treats_directional_single_bonds_as_single_family() {
        let smiles = smiles_from_edges(
            vec![
                Atom::new_organic_subset(AtomSymbol::Element(Element::C), false),
                Atom::new_organic_subset(AtomSymbol::Element(Element::O), false),
                Atom::new_organic_subset(AtomSymbol::Element(Element::N), false),
            ],
            &[bond_edge(0, 1, Bond::Up, None), bond_edge(0, 2, Bond::Down, None)],
        );

        let invariant = smiles.atom_invariant(0).unwrap();
        assert_eq!(invariant.bond_kind_histogram.count(Bond::Single), 2);
        assert_eq!(invariant.bond_kind_histogram.count(Bond::Up), 2);
        assert_eq!(invariant.bond_kind_histogram.count(Bond::Down), 2);
    }

    #[test]
    fn bond_kind_helpers_group_directional_single_bonds_and_separate_others() {
        assert_eq!(bond_kind_code(Bond::Single), 0);
        assert_eq!(bond_kind_code(Bond::Up), 0);
        assert_eq!(bond_kind_code(Bond::Down), 0);
        assert_eq!(bond_kind_code(Bond::Double), 1);
        assert_eq!(bond_kind_code(Bond::Triple), 2);
        assert_eq!(bond_kind_code(Bond::Quadruple), 3);

        assert_eq!(bond_kind_index(Bond::Single), bond_kind_index(Bond::Up));
        assert_eq!(bond_kind_index(Bond::Up), bond_kind_index(Bond::Down));
        assert_ne!(bond_kind_index(Bond::Single), bond_kind_index(Bond::Double));
        assert_ne!(bond_kind_index(Bond::Triple), bond_kind_index(Bond::Quadruple));
    }

    #[test]
    fn bond_entry_code_collapses_aromatic_and_separates_kekule_kinds() {
        // An aromatic bond collapses to code 8 regardless of its kekule order,
        // so an aromatic single and an aromatic double share one code and never
        // collide with a plain single (0) or double (1).
        let aromatic_single =
            BondEntry::from_descriptor(BondDescriptor::aromatic(Bond::Single), None, 0);
        let aromatic_double =
            BondEntry::from_descriptor(BondDescriptor::aromatic(Bond::Double), None, 0);
        assert_eq!(bond_entry_code(aromatic_single), 8);
        assert_eq!(bond_entry_code(aromatic_double), 8);

        // Non-aromatic bonds keep their distinct kekule codes.
        assert_eq!(bond_entry_code(BondEntry::new(Bond::Single, None, 0)), 0);
        assert_eq!(bond_entry_code(BondEntry::new(Bond::Double, None, 0)), 1);
        assert_eq!(bond_entry_code(BondEntry::new(Bond::Triple, None, 0)), 2);
        assert_eq!(bond_entry_code(BondEntry::new(Bond::Quadruple, None, 0)), 3);
    }

    #[test]
    fn planning_chirality_key_groups_surface_variants_by_family() {
        assert_eq!(planning_chirality_key(None), (0, 0));
        assert_eq!(planning_chirality_key(Some(Chirality::At)), (1, 0));
        assert_eq!(planning_chirality_key(Some(Chirality::AtAt)), (1, 0));
        assert_eq!(planning_chirality_key(Some(Chirality::TH(1))), (2, 0));
        assert_eq!(planning_chirality_key(Some(Chirality::AL(2))), (3, 0));
        assert_eq!(planning_chirality_key(Some(Chirality::SP(3))), (4, 0));
        assert_eq!(planning_chirality_key(Some(Chirality::TB(20))), (5, 0));
        assert_eq!(planning_chirality_key(Some(Chirality::OH(30))), (6, 0));
    }
}
