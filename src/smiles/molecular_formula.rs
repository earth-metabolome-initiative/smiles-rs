use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use core::str::FromStr;

use elements_rs::{Element, Isotope};
use molecular_formulas::{ChargeLike, ChemicalFormula, CountLike};
use thiserror::Error;

use super::{Smiles, SmilesAtomPolicy, WildcardSmiles};

/// Error raised while converting a [`WildcardSmiles`] graph into a molecular
/// formula.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum WildcardMolecularFormulaConversionError {
    /// A wildcard atom cannot be represented by an exact molecular formula.
    #[error("cannot convert wildcard atom at index {atom_id} to a molecular formula")]
    WildcardAtom {
        /// Index of the wildcard atom in the SMILES graph.
        atom_id: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FormulaSpecies {
    Element(Element),
    Isotope { element: Element, mass_number: u16 },
}

impl FormulaSpecies {
    #[inline]
    fn element(self) -> Element {
        match self {
            Self::Element(element) | Self::Isotope { element, .. } => element,
        }
    }

    #[inline]
    fn isotope_mass_number(self) -> Option<u16> {
        match self {
            Self::Element(_) => None,
            Self::Isotope { mass_number, .. } => Some(mass_number),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FormulaSpeciesCount {
    species: FormulaSpecies,
    count: u32,
}

#[derive(Debug, Clone, Default)]
struct ComponentFormula {
    species_counts: Vec<FormulaSpeciesCount>,
    charge: i32,
}

impl ComponentFormula {
    fn add_species(&mut self, species: FormulaSpecies, count: u32) {
        if count == 0 {
            return;
        }

        if let Some(entry) = self.species_counts.iter_mut().find(|entry| entry.species == species) {
            entry.count = entry.count.checked_add(count).unwrap_or_else(|| {
                unreachable!("parsed SMILES atom and hydrogen counts should fit into u32")
            });
        } else {
            self.species_counts.push(FormulaSpeciesCount { species, count });
        }
    }

    fn write_formula(&mut self, target: &mut String) {
        self.sort_species();
        for entry in &self.species_counts {
            write_species(target, entry.species);
            if entry.count != 1 {
                target.push_str(&entry.count.to_string());
            }
        }
        write_charge(target, self.charge);
    }

    fn sort_species(&mut self) {
        let has_carbon =
            self.species_counts.iter().any(|entry| entry.species.element() == Element::C);
        self.species_counts.sort_by(|left, right| {
            formula_species_order(left.species, has_carbon)
                .cmp(&formula_species_order(right.species, has_carbon))
        });
    }
}

impl<Count, Charge> From<&Smiles> for ChemicalFormula<Count, Charge>
where
    Count: CountLike,
    Charge: ChargeLike + TryFrom<Count>,
    Isotope: TryFrom<(Element, Count), Error = elements_rs::errors::Error>,
{
    fn from(smiles: &Smiles) -> Self {
        let formula = strict_smiles_formula_string(smiles);
        parse_generated_formula(&formula)
    }
}

impl<Count, Charge> From<Smiles> for ChemicalFormula<Count, Charge>
where
    Count: CountLike,
    Charge: ChargeLike + TryFrom<Count>,
    Isotope: TryFrom<(Element, Count), Error = elements_rs::errors::Error>,
{
    fn from(smiles: Smiles) -> Self {
        Self::from(&smiles)
    }
}

impl<Count, Charge> TryFrom<&WildcardSmiles> for ChemicalFormula<Count, Charge>
where
    Count: CountLike,
    Charge: ChargeLike + TryFrom<Count>,
    Isotope: TryFrom<(Element, Count), Error = elements_rs::errors::Error>,
{
    type Error = WildcardMolecularFormulaConversionError;

    fn try_from(smiles: &WildcardSmiles) -> Result<Self, Self::Error> {
        let formula = smiles_formula_string(smiles.inner())?;
        Ok(parse_generated_formula(&formula))
    }
}

impl<Count, Charge> TryFrom<WildcardSmiles> for ChemicalFormula<Count, Charge>
where
    Count: CountLike,
    Charge: ChargeLike + TryFrom<Count>,
    Isotope: TryFrom<(Element, Count), Error = elements_rs::errors::Error>,
{
    type Error = WildcardMolecularFormulaConversionError;

    fn try_from(smiles: WildcardSmiles) -> Result<Self, Self::Error> {
        Self::try_from(&smiles)
    }
}

fn strict_smiles_formula_string(smiles: &Smiles) -> String {
    smiles_formula_string(smiles).unwrap_or_else(|error| {
        match error {
            WildcardMolecularFormulaConversionError::WildcardAtom { .. } => {
                unreachable!("strict Smiles cannot contain wildcard atoms")
            }
        }
    })
}

fn smiles_formula_string<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
) -> Result<String, WildcardMolecularFormulaConversionError> {
    debug_assert!(
        !smiles.nodes().is_empty(),
        "parsed SMILES graphs are non-empty; empty graphs are crate-internal only"
    );

    let components = smiles.connected_components();
    let mut component_formulas =
        vec![ComponentFormula::default(); components.number_of_components()];
    let component_ids = components.component_identifiers().collect::<Vec<_>>();

    for (atom_id, atom) in smiles.nodes().iter().enumerate() {
        let component = &mut component_formulas[component_ids[atom_id]];
        let element = atom
            .element()
            .ok_or(WildcardMolecularFormulaConversionError::WildcardAtom { atom_id })?;
        let species = atom.isotope_mass_number().map_or(FormulaSpecies::Element(element), |mass| {
            FormulaSpecies::Isotope { element, mass_number: mass }
        });
        component.add_species(species, 1);
        let hydrogen_count =
            u32::from(atom.hydrogen_count()) + u32::from(smiles.implicit_hydrogen_count(atom_id));
        component.add_species(FormulaSpecies::Element(Element::H), hydrogen_count);
        component.charge = component
            .charge
            .checked_add(i32::from(atom.charge_value()))
            .unwrap_or_else(|| unreachable!("parsed SMILES formal charges should fit into i32"));
    }

    let mut formula = String::new();
    for (index, component) in component_formulas.iter_mut().enumerate() {
        if index != 0 {
            formula.push('.');
        }
        component.write_formula(&mut formula);
    }
    Ok(formula)
}

fn parse_generated_formula<Count, Charge>(formula: &str) -> ChemicalFormula<Count, Charge>
where
    Count: CountLike,
    Charge: ChargeLike + TryFrom<Count>,
    Isotope: TryFrom<(Element, Count), Error = elements_rs::errors::Error>,
{
    ChemicalFormula::from_str(formula).unwrap_or_else(|error| {
        unreachable!("generated formula `{formula}` should parse as ChemicalFormula: {error}")
    })
}

fn formula_species_order(
    species: FormulaSpecies,
    has_carbon: bool,
) -> (u8, &'static str, u8, bool, Option<u16>) {
    let element = species.element();
    let isotope_mass_number = species.isotope_mass_number();
    let is_isotope = isotope_mass_number.is_some();
    let atomic_number = u8::from(element);

    if has_carbon {
        match (element, is_isotope) {
            (Element::C, _) => (0, "", atomic_number, is_isotope, isotope_mass_number),
            (Element::H, _) => (1, "", atomic_number, is_isotope, isotope_mass_number),
            (_, false) => (2, element.symbol(), 0, false, None),
            (_, true) => (3, "", atomic_number, true, isotope_mass_number),
        }
    } else {
        match (element, is_isotope) {
            (Element::H, _) => (0, "", atomic_number, is_isotope, isotope_mass_number),
            (_, false) => (1, element.symbol(), 0, false, None),
            (_, true) => (2, "", atomic_number, true, isotope_mass_number),
        }
    }
}

fn write_species(target: &mut String, species: FormulaSpecies) {
    match species {
        FormulaSpecies::Element(element) => target.push_str(element.symbol()),
        FormulaSpecies::Isotope { element, mass_number } => {
            target.push('[');
            target.push_str(&mass_number.to_string());
            target.push_str(element.symbol());
            target.push(']');
        }
    }
}

fn write_charge(target: &mut String, charge: i32) {
    match charge.cmp(&0) {
        core::cmp::Ordering::Less => {
            target.push('-');
            if charge != -1 {
                target.push_str(&charge.unsigned_abs().to_string());
            }
        }
        core::cmp::Ordering::Equal => {}
        core::cmp::Ordering::Greater => {
            target.push('+');
            if charge != 1 {
                target.push_str(&charge.to_string());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use molecular_formulas::{ChargedMolecularFormula, MolecularFormula};

    use super::*;

    type TestFormula = ChemicalFormula<u32, i32>;

    #[test]
    fn formula_string_counts_implicit_hydrogens() {
        let smiles: Smiles = "c1ccccc1".parse().unwrap();

        assert_eq!(strict_smiles_formula_string(&smiles), "C6H6");
    }

    #[test]
    fn formula_conversion_preserves_isotopes_explicit_hydrogens_and_charge() {
        let smiles: Smiles = "[13CH3][NH3+]".parse().unwrap();
        let formula: TestFormula = ChemicalFormula::from(&smiles);

        assert_eq!(formula.to_string(), "[¹³C]H₆N⁺");
        assert_eq!(formula.count_of_element::<u32>(Element::C), Ok(1));
        assert_eq!(formula.count_of_element::<u32>(Element::H), Ok(6));
        assert_eq!(formula.count_of_element::<u32>(Element::N), Ok(1));
        assert!((formula.charge() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn formula_conversion_preserves_disconnected_components() {
        let smiles: Smiles = "[Na+].[Cl-]".parse().unwrap();
        let formula: TestFormula = ChemicalFormula::from(smiles);

        assert_eq!(formula.to_string(), "Na⁺.Cl⁻");
        assert!(formula.charge().abs() < f64::EPSILON);
    }

    #[test]
    fn formula_conversion_uses_explicit_count_and_charge_types() {
        let smiles: Smiles = "C".parse().unwrap();
        let compact: ChemicalFormula<u8, i8> = ChemicalFormula::from(&smiles);
        let wide: ChemicalFormula<u32, i32> = ChemicalFormula::from(&smiles);

        assert_eq!(compact.to_string(), "CH₄");
        assert_eq!(wide.to_string(), compact.to_string());
    }

    #[test]
    fn formula_conversion_matches_rdkit_non_carbon_isotope_ordering() {
        let smiles: Smiles = "C([18OH])([131I])Cl".parse().unwrap();
        let formula: TestFormula = ChemicalFormula::from(smiles);
        let rdkit_formula = TestFormula::try_from("CH2Cl[18O][131I]").unwrap();

        assert_eq!(formula, rdkit_formula);
    }

    #[test]
    fn formula_conversion_rejects_wildcards() {
        let smiles: WildcardSmiles = "*".parse().unwrap();
        let error = TestFormula::try_from(smiles).unwrap_err();

        assert_eq!(error, WildcardMolecularFormulaConversionError::WildcardAtom { atom_id: 0 });
    }
}
