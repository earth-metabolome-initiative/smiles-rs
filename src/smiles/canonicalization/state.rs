use alloc::vec::Vec;

use geometric_traits::traits::SparseValuedMatrixRef;

use crate::{
    atom::{Atom, AtomSyntax, atom_symbol::AtomSymbol, bracketed::chirality::Chirality},
    smiles::{BondEntry, Smiles, SmilesAtomPolicy, StereoNeighbor},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct CanonicalAtomLabel {
    syntax: u8,
    symbol: AtomSymbol,
    isotope_mass_number: Option<u16>,
    aromatic: bool,
    hydrogens: u8,
    charge: i8,
    class: u16,
    chirality_kind: u8,
    chirality_value: u8,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct CanonicalBondLabel(pub(super) u8);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CanonicalStereoNeighborKey(u8, usize);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CanonicalizationStateKey {
    atom_labels: Vec<CanonicalAtomLabel>,
    bond_edges: Vec<(usize, usize, CanonicalBondLabel)>,
    parsed_stereo_neighbors: Vec<Vec<CanonicalStereoNeighborKey>>,
    implicit_hydrogen_cache: Vec<u8>,
}

pub(super) fn canonical_atom_label(atom: Atom) -> CanonicalAtomLabel {
    let (chirality_kind, chirality_value) = canonical_chirality_key(atom.chirality());

    CanonicalAtomLabel {
        syntax: match atom.syntax() {
            AtomSyntax::OrganicSubset => 0,
            AtomSyntax::Bracket => 1,
        },
        symbol: atom.symbol(),
        isotope_mass_number: atom.isotope_mass_number(),
        aromatic: atom.aromatic(),
        hydrogens: atom.hydrogen_count(),
        charge: atom.charge_value(),
        class: atom.class(),
        chirality_kind,
        chirality_value,
    }
}

pub(super) fn stereo_neutral_canonical_atom_label(atom: Atom) -> CanonicalAtomLabel {
    let mut label = canonical_atom_label(atom);
    label.chirality_kind = 0;
    label.chirality_value = 0;
    label
}

pub(super) fn canonicalization_state_key<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
) -> CanonicalizationStateKey {
    let atom_labels = smiles.nodes().iter().copied().map(canonical_atom_label).collect();
    let bond_edges = smiles
        .bond_matrix()
        .sparse_entries()
        .filter(|((row, column), _)| row < column)
        .map(|((row, column), entry)| (row, column, canonical_bond_label(*entry)))
        .collect();
    let parsed_stereo_neighbors = smiles
        .parsed_stereo_neighbors
        .iter()
        .map(|row| row.iter().copied().map(canonical_stereo_neighbor_key).collect())
        .collect();

    CanonicalizationStateKey {
        atom_labels,
        bond_edges,
        parsed_stereo_neighbors,
        implicit_hydrogen_cache: smiles.implicit_hydrogen_cache.clone(),
    }
}

pub(super) const fn canonical_bond_label(entry: BondEntry) -> CanonicalBondLabel {
    // The aromatic flag fully determines an aromatic bond: the single/double
    // kekule order retained alongside the flag is non-semantic. Collapse every
    // aromatic bond to one canonical code so that two graphs which differ only
    // in the kekule assignment of their aromatic bonds (e.g. a raw aromatic
    // parse versus its kekulize-then-reperceive round-trip) canonicalize
    // identically, including on broken-ring fragments where the order would
    // otherwise leak into the rendered label.
    if entry.aromatic() {
        return CanonicalBondLabel(8);
    }
    let bond_code = match entry.bond() {
        crate::bond::Bond::Single => 0,
        crate::bond::Bond::Double => 1,
        crate::bond::Bond::Triple => 2,
        crate::bond::Bond::Quadruple => 3,
        crate::bond::Bond::Up => 4,
        crate::bond::Bond::Down => 5,
    };
    CanonicalBondLabel(bond_code)
}

pub(crate) const fn canonical_chirality_key(chirality: Option<Chirality>) -> (u8, u8) {
    match chirality {
        None => (0, 0),
        Some(Chirality::At) => (1, 0),
        Some(Chirality::AtAt) => (2, 0),
        Some(Chirality::TH(value)) => (3, value),
        Some(Chirality::AL(value)) => (4, value),
        Some(Chirality::SP(value)) => (5, value),
        Some(Chirality::TB(value)) => (6, value),
        Some(Chirality::OH(value)) => (7, value),
    }
}

pub(crate) fn canonical_stereo_neighbor_key(
    neighbor: StereoNeighbor,
) -> CanonicalStereoNeighborKey {
    match neighbor {
        StereoNeighbor::Atom(node_id) => CanonicalStereoNeighborKey(0, node_id),
        StereoNeighbor::ExplicitHydrogen => CanonicalStereoNeighborKey(1, usize::MAX),
    }
}
