use alloc::vec::Vec;

use elements_rs::{AllowedValences, ChargedValences, Element};
use geometric_traits::traits::SparseValuedMatrixRef;
use hashbrown::{HashMap, HashSet};

use super::{
    AromaticityAssignment, AromaticityDiagnostic, AromaticityModel, AromaticityRingFamilyKind,
    AromaticityStatus, RdkitDefaultAromaticity, RdkitMdlAromaticity, RdkitSimpleAromaticity,
    Smiles,
};
#[cfg(test)]
use crate::smiles::RingComponent;
use crate::{
    atom::Atom,
    bond::Bond,
    smiles::{RingMembership, SmilesAtomPolicy, SymmSssrStatus, cycle_edges, rdkit_symm_sssr},
};

impl AromaticityModel for RdkitDefaultAromaticity {
    fn assignment<AtomPolicy: SmilesAtomPolicy>(
        &self,
        smiles: &Smiles<AtomPolicy>,
    ) -> AromaticityAssignment {
        RdkitAromaticityFlavor::Default.assignment(smiles)
    }
}

impl AromaticityModel for RdkitMdlAromaticity {
    fn assignment<AtomPolicy: SmilesAtomPolicy>(
        &self,
        smiles: &Smiles<AtomPolicy>,
    ) -> AromaticityAssignment {
        RdkitAromaticityFlavor::Mdl.assignment(smiles)
    }
}

impl AromaticityModel for RdkitSimpleAromaticity {
    fn assignment<AtomPolicy: SmilesAtomPolicy>(
        &self,
        smiles: &Smiles<AtomPolicy>,
    ) -> AromaticityAssignment {
        RdkitAromaticityFlavor::Simple.assignment(smiles)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum RdkitAromaticityFlavor {
    Default,
    Simple,
    Mdl,
}

impl RdkitAromaticityFlavor {
    fn candidate_rules(self) -> RdkitDefaultCandidateRules {
        match self {
            Self::Default | Self::Simple => RdkitDefaultCandidateRules::DEFAULT,
            Self::Mdl => RdkitDefaultCandidateRules::MDL,
        }
    }

    fn exocyclic_bonds_steal_electrons(self) -> bool {
        match self {
            Self::Default | Self::Simple | Self::Mdl => true,
        }
    }

    fn allows_standalone_five_member_aromatic_cycles(self) -> bool {
        match self {
            Self::Default | Self::Simple => true,
            Self::Mdl => false,
        }
    }

    fn allows_simple_cycle_size(self, ring_size: usize) -> bool {
        match self {
            Self::Default | Self::Mdl => ring_size >= 3,
            Self::Simple => matches!(ring_size, 5 | 6),
        }
    }

    fn assignment<AtomPolicy: SmilesAtomPolicy>(
        self,
        smiles: &Smiles<AtomPolicy>,
    ) -> AromaticityAssignment {
        if let Some(prepared_smiles) = RdkitPreAromaticityNormalization::apply(smiles) {
            self.assignment_with_overrides(
                &prepared_smiles.smiles,
                &prepared_smiles.implicit_hydrogen_overrides,
            )
        } else {
            let implicit_hydrogen_overrides = HashMap::new();
            self.assignment_with_overrides(smiles, &implicit_hydrogen_overrides)
        }
    }

    fn assignment_with_overrides<AtomPolicy: SmilesAtomPolicy>(
        self,
        smiles: &Smiles<AtomPolicy>,
        implicit_hydrogen_overrides: &HashMap<usize, u8>,
    ) -> AromaticityAssignment {
        match self.quick_assignment(smiles, implicit_hydrogen_overrides) {
            QuickAssignment::Ready(assignment) => assignment,
            QuickAssignment::NeedsFullEvaluation { ring_membership, atom_states } => {
                RdkitDefaultContext::new_with_ring_membership_and_atom_states(
                    self,
                    smiles,
                    &ring_membership,
                    atom_states,
                )
                .assignment()
            }
        }
    }

    fn quick_assignment<AtomPolicy: SmilesAtomPolicy>(
        self,
        smiles: &Smiles<AtomPolicy>,
        implicit_hydrogen_overrides: &HashMap<usize, u8>,
    ) -> QuickAssignment {
        let ring_membership = smiles.ring_membership();
        if ring_membership.atom_ids().is_empty() {
            return QuickAssignment::Ready(AromaticityAssignment::new(
                AromaticityStatus::Complete,
                Vec::new(),
                Vec::new(),
            ));
        }

        let atom_states = RdkitDefaultElectronModel::build_states(
            self,
            smiles,
            &ring_membership,
            implicit_hydrogen_overrides,
        );
        if !ring_membership.atom_ids().iter().copied().any(|atom_id| atom_states[atom_id].candidate)
        {
            return QuickAssignment::Ready(AromaticityAssignment::new(
                AromaticityStatus::Complete,
                Vec::new(),
                Vec::new(),
            ));
        }

        QuickAssignment::NeedsFullEvaluation { ring_membership, atom_states }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QuickAssignment {
    Ready(AromaticityAssignment),
    NeedsFullEvaluation {
        ring_membership: RingMembership,
        atom_states: Vec<RdkitPerAtomAromaticityState>,
    },
}

/// Compatibility normalization for the subset of `RDKit` `cleanUp` behavior
/// that materially changes donor typing before aromaticity assignment.
///
/// This is intentionally narrow. We only rewrite neutral five-valent
/// phosphorus motifs with one `P=O` and another `P=C` or `P=N` bond where the
/// carbon or nitrogen neighbor has degree >= 2. Those are the sanitize-time
/// cleanup cases that change the aromatic ring model here, instead of trying to
/// mirror every `RDKit` cleanup transformation up front.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
struct RdkitPreAromaticityNormalization;

#[derive(Debug)]
struct RdkitNormalizedSmiles<AtomPolicy> {
    smiles: Smiles<AtomPolicy>,
    implicit_hydrogen_overrides: HashMap<usize, u8>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct RdkitPhosphorusCleanupTarget {
    phosphorus_atom_id: usize,
    oxygen_atom_id: usize,
}

impl RdkitPreAromaticityNormalization {
    fn apply<AtomPolicy: SmilesAtomPolicy>(
        smiles: &Smiles<AtomPolicy>,
    ) -> Option<RdkitNormalizedSmiles<AtomPolicy>> {
        let phosphorus_targets = Self::phosphorus_cleanup_targets(smiles);
        if phosphorus_targets.is_empty() {
            return None;
        }

        let mut atom_nodes = smiles.nodes().to_vec();
        let mut bonds_to_single =
            HashSet::<(usize, usize)>::with_capacity(phosphorus_targets.len());
        let mut implicit_hydrogen_overrides =
            HashMap::<usize, u8>::with_capacity(phosphorus_targets.len() * 2);

        for target in phosphorus_targets {
            atom_nodes[target.phosphorus_atom_id] =
                atom_nodes[target.phosphorus_atom_id].with_charge_value(1);
            atom_nodes[target.oxygen_atom_id] =
                atom_nodes[target.oxygen_atom_id].with_charge_value(-1);
            implicit_hydrogen_overrides.insert(target.phosphorus_atom_id, 0);
            implicit_hydrogen_overrides.insert(target.oxygen_atom_id, 0);
            bonds_to_single
                .insert(crate::smiles::edge_key(target.phosphorus_atom_id, target.oxygen_atom_id));
        }

        let bond_matrix = crate::smiles::BondMatrix::from_sorted_upper_triangular_entries(
            atom_nodes.len(),
            smiles.bond_matrix().sparse_entries().filter_map(|((row, column), entry)| {
                (row < column).then_some((
                    row,
                    column,
                    if bonds_to_single.contains(&(row, column)) {
                        entry.with_bond(Bond::Single)
                    } else {
                        *entry
                    },
                ))
            }),
        )
        .unwrap_or_else(|_| unreachable!("existing bond matrix entries are already valid"));

        Some(RdkitNormalizedSmiles {
            smiles: Smiles::<AtomPolicy>::from_bond_matrix_parts(atom_nodes, bond_matrix),
            implicit_hydrogen_overrides,
        })
    }

    fn phosphorus_cleanup_targets<AtomPolicy: SmilesAtomPolicy>(
        smiles: &Smiles<AtomPolicy>,
    ) -> Vec<RdkitPhosphorusCleanupTarget> {
        let mut targets = Vec::<RdkitPhosphorusCleanupTarget>::new();

        for phosphorus_atom_id in 0..smiles.nodes().len() {
            let atom = &smiles.nodes()[phosphorus_atom_id];
            if atom.element() != Some(Element::P) || atom.charge_value() != 0 {
                continue;
            }
            if raw_total_valence(smiles, phosphorus_atom_id) != 5 {
                continue;
            }

            if smiles.edge_count_for_node(phosphorus_atom_id) != 3 {
                continue;
            }
            let incident_bonds = smiles.edges_for_node(phosphorus_atom_id);

            let mut oxygen_atom_id = None;
            let mut has_double_to_carbon_or_nitrogen = false;

            for edge in incident_bonds {
                let bond = edge.bond().without_direction();
                let neighbor_atom_id = edge.target();
                let neighbor_atom = &smiles.nodes()[neighbor_atom_id];

                if bond == Bond::Double
                    && neighbor_atom.element() == Some(Element::O)
                    && neighbor_atom.charge_value() == 0
                {
                    oxygen_atom_id = Some(neighbor_atom_id);
                } else if bond == Bond::Double
                    && matches!(neighbor_atom.element(), Some(Element::C | Element::N))
                    && smiles.edge_count_for_node(neighbor_atom_id) >= 2
                {
                    has_double_to_carbon_or_nitrogen = true;
                }
            }

            if has_double_to_carbon_or_nitrogen && let Some(oxygen_atom_id) = oxygen_atom_id {
                targets.push(RdkitPhosphorusCleanupTarget { phosphorus_atom_id, oxygen_atom_id });
            }
        }

        targets
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ElectronDonorType {
    Vacant,
    One,
    Two,
    Any,
    None,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
struct RdkitDefaultCandidateRules {
    allow_third_row: bool,
    allow_triple_bonds: bool,
    allow_higher_exceptions: bool,
    only_c_or_n: bool,
    allow_exocyclic_multiple_bonds: bool,
    only_one_electron_donors: bool,
}

impl RdkitDefaultCandidateRules {
    const DEFAULT: Self = Self {
        allow_third_row: true,
        allow_triple_bonds: true,
        allow_higher_exceptions: true,
        only_c_or_n: false,
        allow_exocyclic_multiple_bonds: true,
        only_one_electron_donors: false,
    };

    const MDL: Self = Self {
        allow_third_row: false,
        allow_triple_bonds: false,
        allow_higher_exceptions: false,
        only_c_or_n: true,
        allow_exocyclic_multiple_bonds: false,
        only_one_electron_donors: true,
    };
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RdkitAtomAromContext {
    element: Option<Element>,
    formal_charge: i8,
    total_hydrogens: u8,
    radical_electrons: u8,
    degree: usize,
    degree_plus_total_h: usize,
    total_valence: usize,
    total_unsaturations: usize,
    incident_multiple_bond: bool,
    incident_cyclic_multiple_bond: bool,
    incident_noncyclic_multiple_bond_to: Option<usize>,
    incident_noncyclic_multiple_bond_to_more_electronegative_atom: bool,
    multiple_bond_kinds: Vec<Bond>,
}

impl RdkitAtomAromContext {
    fn build<AtomPolicy: SmilesAtomPolicy>(
        smiles: &Smiles<AtomPolicy>,
        ring_membership: &RingMembership,
        implicit_hydrogen_overrides: &HashMap<usize, u8>,
        radical_electrons: &[u8],
        atom_id: usize,
    ) -> Self {
        let atom = &smiles.nodes()[atom_id];
        let element = atom.element();
        let explicit_hydrogens = atom.hydrogen_count();
        let implicit_hydrogens = implicit_hydrogen_overrides
            .get(&atom_id)
            .copied()
            .unwrap_or_else(|| smiles.implicit_hydrogen_count(atom_id));
        let total_hydrogens = explicit_hydrogens.saturating_add(implicit_hydrogens);
        let degree = smiles.edge_count_for_node(atom_id);
        let edges = smiles.edges_for_node(atom_id);
        let degree_plus_total_h = degree + usize::from(total_hydrogens);
        let mut explicit_valence = usize::from(explicit_hydrogens);
        let mut incident_multiple_bond = false;
        let mut incident_cyclic_multiple_bond = false;
        let mut incident_noncyclic_multiple_bond_to = None;
        let mut multiple_bond_kinds = Vec::<Bond>::new();

        for edge in edges {
            let bond = edge.bond().without_direction();
            explicit_valence += bond_valence_contribution(bond);
            if edge.is_aromatic() || matches!(bond, Bond::Double | Bond::Triple | Bond::Quadruple) {
                incident_multiple_bond = true;
                if ring_membership.contains_edge(atom_id, edge.target()) {
                    incident_cyclic_multiple_bond = true;
                }
            }
            if matches!(bond, Bond::Double | Bond::Triple | Bond::Quadruple) {
                multiple_bond_kinds.push(bond);
                if !ring_membership.contains_edge(atom_id, edge.target())
                    && incident_noncyclic_multiple_bond_to.is_none()
                {
                    incident_noncyclic_multiple_bond_to = Some(edge.target());
                }
            }
        }

        let total_valence = explicit_valence + usize::from(implicit_hydrogens);
        let total_unsaturations = explicit_valence.saturating_sub(degree);
        let incident_noncyclic_multiple_bond_to_more_electronegative_atom =
            incident_noncyclic_multiple_bond_to
                .and_then(|other_atom_id| smiles.node_by_id(other_atom_id).and_then(Atom::element))
                .zip(element)
                .is_some_and(|(other, current)| more_electronegative(other, current));

        Self {
            element,
            formal_charge: atom.charge_value(),
            total_hydrogens,
            radical_electrons: radical_electrons[atom_id],
            degree,
            degree_plus_total_h,
            total_valence,
            total_unsaturations,
            incident_multiple_bond,
            incident_cyclic_multiple_bond,
            incident_noncyclic_multiple_bond_to,
            incident_noncyclic_multiple_bond_to_more_electronegative_atom,
            multiple_bond_kinds,
        }
    }

    fn neutral_default_valence(&self) -> Option<u8> {
        let element = self.element?;
        default_valence_for(element, 0)
    }

    fn adjusted_default_valence(&self) -> Option<u8> {
        let element = self.element?;
        rdkit_adjusted_default_valence(element, self.formal_charge)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RdkitPerAtomAromaticityState {
    donor_type: ElectronDonorType,
    candidate: bool,
}

impl Default for RdkitPerAtomAromaticityState {
    fn default() -> Self {
        Self { donor_type: ElectronDonorType::None, candidate: false }
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
struct RdkitDefaultElectronModel;

impl RdkitDefaultElectronModel {
    fn build_states<AtomPolicy: SmilesAtomPolicy>(
        flavor: RdkitAromaticityFlavor,
        smiles: &Smiles<AtomPolicy>,
        ring_membership: &RingMembership,
        implicit_hydrogen_overrides: &HashMap<usize, u8>,
    ) -> Vec<RdkitPerAtomAromaticityState> {
        let radical_electrons = assign_radicals(smiles);
        let mut atom_states = vec![RdkitPerAtomAromaticityState::default(); smiles.nodes().len()];
        for atom_id in ring_membership.atom_ids().iter().copied() {
            let context = RdkitAtomAromContext::build(
                smiles,
                ring_membership,
                implicit_hydrogen_overrides,
                &radical_electrons,
                atom_id,
            );
            let donor_type = Self::donor_type(&context, flavor.exocyclic_bonds_steal_electrons());
            let candidate = Self::is_candidate(&context, donor_type, flavor.candidate_rules());
            atom_states[atom_id] = RdkitPerAtomAromaticityState { donor_type, candidate };
        }
        atom_states
    }

    fn count_atom_electrons(context: &RdkitAtomAromContext) -> i8 {
        let Some(default_valence) = context.neutral_default_valence() else {
            return -1;
        };
        if default_valence <= 1 {
            return -1;
        }
        if context.degree_plus_total_h > 3 {
            return -1;
        }

        let Some(element) = context.element else {
            return -1;
        };
        let outer_electrons = i16::from(rdkit_outer_electrons(element));
        let mut lone_pair_electrons = outer_electrons - i16::from(default_valence);
        lone_pair_electrons = (lone_pair_electrons - i16::from(context.formal_charge)).max(0);

        let mut available_electrons = i16::from(default_valence)
            - i16::try_from(context.degree_plus_total_h).unwrap_or(i16::MAX)
            + lone_pair_electrons
            - i16::from(context.radical_electrons);
        if available_electrons > 1 && context.total_unsaturations > 1 {
            available_electrons = 1;
        }

        i8::try_from(available_electrons).unwrap_or(i8::MAX)
    }

    fn donor_type(
        context: &RdkitAtomAromContext,
        exocyclic_bonds_steal_electrons: bool,
    ) -> ElectronDonorType {
        if context.element.is_none() {
            return if context.incident_cyclic_multiple_bond {
                ElectronDonorType::One
            } else {
                ElectronDonorType::Any
            };
        }

        let mut electrons = Self::count_atom_electrons(context);
        if electrons < 0 {
            return ElectronDonorType::None;
        }

        match electrons {
            0 => {
                if context.incident_noncyclic_multiple_bond_to.is_some() {
                    ElectronDonorType::Vacant
                } else if context.incident_cyclic_multiple_bond {
                    ElectronDonorType::One
                } else {
                    ElectronDonorType::None
                }
            }
            1 => {
                if context.incident_noncyclic_multiple_bond_to.is_some() {
                    if exocyclic_bonds_steal_electrons
                        && context.incident_noncyclic_multiple_bond_to_more_electronegative_atom
                    {
                        ElectronDonorType::Vacant
                    } else {
                        ElectronDonorType::One
                    }
                } else if context.incident_multiple_bond {
                    ElectronDonorType::One
                } else if context.formal_charge == 1 {
                    ElectronDonorType::Vacant
                } else {
                    ElectronDonorType::None
                }
            }
            _ => {
                if exocyclic_bonds_steal_electrons
                    && context.incident_noncyclic_multiple_bond_to_more_electronegative_atom
                {
                    electrons -= 1;
                }
                if electrons % 2 == 1 { ElectronDonorType::One } else { ElectronDonorType::Two }
            }
        }
    }

    fn is_candidate(
        context: &RdkitAtomAromContext,
        donor_type: ElectronDonorType,
        candidate_rules: RdkitDefaultCandidateRules,
    ) -> bool {
        if candidate_rules.only_c_or_n && !element_is_carbon_or_nitrogen(context.element) {
            return false;
        }
        if !candidate_rules.allow_third_row && element_is_beyond_second_row(context.element) {
            return false;
        }
        if element_is_beyond_argon(context.element)
            && (!candidate_rules.allow_higher_exceptions || !element_is_se_or_te(context.element))
        {
            return false;
        }
        if !matches!(
            donor_type,
            ElectronDonorType::Vacant
                | ElectronDonorType::One
                | ElectronDonorType::Two
                | ElectronDonorType::Any
        ) {
            return false;
        }
        if candidate_rules.only_one_electron_donors && !matches!(donor_type, ElectronDonorType::One)
        {
            return false;
        }
        if let Some(default_valence) = context.adjusted_default_valence()
            && default_valence > 0
            && context.total_valence > usize::from(default_valence)
        {
            return false;
        }

        if context.radical_electrons > 0
            && (!matches!(context.element, Some(Element::C)) || context.formal_charge != 0)
        {
            return false;
        }

        if context.total_unsaturations > 1 {
            let mut multiple_bond_count = 0_usize;
            for bond_kind in &context.multiple_bond_kinds {
                match bond_kind {
                    Bond::Single | Bond::Up | Bond::Down => {}
                    Bond::Double | Bond::Quadruple => multiple_bond_count += 1,
                    Bond::Triple => {
                        if !candidate_rules.allow_triple_bonds {
                            return false;
                        }
                        multiple_bond_count += 1;
                    }
                }
                if multiple_bond_count > 1 {
                    return false;
                }
            }
        }

        if !candidate_rules.allow_exocyclic_multiple_bonds
            && context.incident_noncyclic_multiple_bond_to.is_some()
        {
            return false;
        }

        true
    }

    fn electron_range(donor_type: ElectronDonorType) -> (u8, u8) {
        match donor_type {
            ElectronDonorType::Any => (1, 2),
            ElectronDonorType::One => (1, 1),
            ElectronDonorType::Two => (2, 2),
            ElectronDonorType::Vacant | ElectronDonorType::None => (0, 0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RdkitDefaultRingKind {
    SimpleCycle,
    FusedComponent,
}

impl From<RdkitDefaultRingKind> for AromaticityRingFamilyKind {
    fn from(value: RdkitDefaultRingKind) -> Self {
        match value {
            RdkitDefaultRingKind::SimpleCycle => Self::SimpleCycle,
            RdkitDefaultRingKind::FusedComponent => Self::FusedComponent,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RdkitDefaultRingFamily {
    kind: RdkitDefaultRingKind,
    member_cycles: Vec<Vec<usize>>,
    member_cycle_bond_edges: Vec<Vec<[usize; 2]>>,
    atom_ids: Vec<usize>,
    bond_edges: Vec<[usize; 2]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RdkitCycleFamilySeed {
    member_cycles: Vec<Vec<usize>>,
    member_cycle_bond_edges: Vec<Vec<[usize; 2]>>,
}

#[derive(Debug, Default)]
struct AromaticityDraft {
    status: AromaticityStatusAccumulator,
    atom_ids: Vec<usize>,
    atom_set: HashSet<usize>,
    bond_edges: Vec<[usize; 2]>,
    diagnostics: Vec<AromaticityDiagnostic>,
}

impl AromaticityDraft {
    fn extend_subgraph(&mut self, delocalization_subgraph: RdkitDelocalizationSubgraph) {
        for atom_id in delocalization_subgraph.atom_ids {
            if self.atom_set.insert(atom_id) {
                self.atom_ids.push(atom_id);
            }
        }
        self.bond_edges.extend(delocalization_subgraph.bond_edges);
    }

    fn push_diagnostic(&mut self, diagnostic: AromaticityDiagnostic) {
        self.diagnostics.push(diagnostic);
    }

    fn extend_diagnostics(&mut self, diagnostics: Vec<AromaticityDiagnostic>) {
        self.diagnostics.extend(diagnostics);
    }

    fn finish(self) -> AromaticityAssignment {
        AromaticityAssignment::new_with_diagnostics(
            self.status.finish(),
            self.atom_ids,
            self.bond_edges,
            self.diagnostics,
        )
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum FamilyEvaluationStatus {
    Supported,
    Partial,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FamilyEvaluationOutcome {
    status: FamilyEvaluationStatus,
    evaluation: Option<RdkitFamilyEvaluation>,
    diagnostics: Vec<AromaticityDiagnostic>,
}

impl From<Option<RdkitFamilyEvaluation>> for FamilyEvaluationOutcome {
    fn from(evaluation: Option<RdkitFamilyEvaluation>) -> Self {
        Self { status: FamilyEvaluationStatus::Supported, evaluation, diagnostics: Vec::new() }
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
struct AromaticityStatusAccumulator {
    saw_partial: bool,
    saw_supported: bool,
    saw_unsupported: bool,
}

impl AromaticityStatusAccumulator {
    fn mark_partial(&mut self) {
        self.saw_partial = true;
    }

    fn mark_supported(&mut self) {
        self.saw_supported = true;
    }

    fn mark_unsupported(&mut self) {
        self.saw_unsupported = true;
    }

    fn finish(self) -> AromaticityStatus {
        if self.saw_partial || (self.saw_unsupported && self.saw_supported) {
            AromaticityStatus::Partial
        } else if self.saw_unsupported {
            AromaticityStatus::Unsupported
        } else {
            AromaticityStatus::Complete
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RdkitFamilyEvaluation {
    delocalization_subgraph: RdkitDelocalizationSubgraph,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RdkitPreparedFusedFamily {
    member_cycle_bond_indices: Vec<Vec<usize>>,
    ring_neighbors: Vec<Vec<usize>>,
    family_bond_edges: Vec<[usize; 2]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RdkitDelocalizationSubgraph {
    atom_ids: Vec<usize>,
    bond_edges: Vec<[usize; 2]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RdkitSubsystemScratch {
    atom_multiplicity: Vec<u8>,
    active_atom_ids: Vec<usize>,
    active_atom_positions: Vec<usize>,
    non_candidate_atom_count: usize,
    lower_bound: u8,
    upper_bound: u8,
    any_donor_count: usize,
    contributing_atom_count: usize,
    bond_multiplicity: Vec<u8>,
    unique_bond_indices: Vec<usize>,
    unique_bond_positions: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RdkitFusedFamilyAccumulator {
    aromatic_bond_seen: Vec<bool>,
    aromatic_bond_indices: Vec<usize>,
    done_bond_seen: Vec<bool>,
    done_bond_count: usize,
}

impl RdkitFamilyEvaluation {
    fn into_delocalization_subgraph(self) -> RdkitDelocalizationSubgraph {
        self.delocalization_subgraph
    }
}

#[derive(Debug)]
struct RdkitDefaultContext<'a, AtomPolicy = crate::smiles::ConcreteAtoms> {
    flavor: RdkitAromaticityFlavor,
    smiles: &'a Smiles<AtomPolicy>,
    atom_states: Vec<RdkitPerAtomAromaticityState>,
    symm_sssr_cycles: Vec<Vec<usize>>,
    symm_sssr_status: SymmSssrStatus,
}

impl<AtomPolicy: SmilesAtomPolicy> RdkitDefaultContext<'_, AtomPolicy> {
    const MAX_FUSED_AROMATIC_RING_SIZE: usize = 24;

    #[cfg(test)]
    fn new<'a>(
        flavor: RdkitAromaticityFlavor,
        smiles: &'a Smiles<AtomPolicy>,
        implicit_hydrogen_overrides: &HashMap<usize, u8>,
    ) -> RdkitDefaultContext<'a, AtomPolicy> {
        let ring_membership = smiles.ring_membership();
        let atom_states = RdkitDefaultElectronModel::build_states(
            flavor,
            smiles,
            &ring_membership,
            implicit_hydrogen_overrides,
        );
        Self::new_with_ring_membership_and_atom_states(
            flavor,
            smiles,
            &ring_membership,
            atom_states,
        )
    }

    fn new_with_ring_membership_and_atom_states<'a>(
        flavor: RdkitAromaticityFlavor,
        smiles: &'a Smiles<AtomPolicy>,
        ring_membership: &RingMembership,
        atom_states: Vec<RdkitPerAtomAromaticityState>,
    ) -> RdkitDefaultContext<'a, AtomPolicy> {
        let symm_sssr =
            rdkit_symm_sssr::symmetrize_sssr_with_ring_membership(smiles, ring_membership);
        RdkitDefaultContext {
            flavor,
            smiles,
            atom_states,
            symm_sssr_cycles: symm_sssr.cycles().to_vec(),
            symm_sssr_status: symm_sssr.status(),
        }
    }

    fn assignment(&self) -> AromaticityAssignment {
        let mut draft = AromaticityDraft::default();
        if !self.symm_sssr_status.is_complete() {
            draft.status.mark_partial();
            if self.symm_sssr_status.used_fallback() {
                draft.push_diagnostic(AromaticityDiagnostic::SymmSssrUsedFallback);
            }
            if self.symm_sssr_status.hit_queue_cutoff() {
                draft.push_diagnostic(AromaticityDiagnostic::SymmSssrHitQueueCutoff);
            }
        }

        if self.flavor == RdkitAromaticityFlavor::Simple {
            self.extend_simple_cycle_assignment(&mut draft);
            return draft.finish();
        }

        for family in self.ring_families() {
            let outcome = self.evaluate_family(&family);
            match outcome.status {
                FamilyEvaluationStatus::Supported => draft.status.mark_supported(),
                FamilyEvaluationStatus::Partial => draft.status.mark_partial(),
                FamilyEvaluationStatus::Unsupported => draft.status.mark_unsupported(),
            }
            draft.extend_diagnostics(outcome.diagnostics);
            if let Some(evaluation) = outcome.evaluation {
                draft.extend_subgraph(evaluation.into_delocalization_subgraph());
            }
        }

        draft.finish()
    }

    fn extend_simple_cycle_assignment(&self, draft: &mut AromaticityDraft) {
        for cycle in &self.symm_sssr_cycles {
            let bond_edges = cycle_edges(cycle);
            if let Some(evaluation) = self.evaluate_simple_cycle_parts(cycle, &bond_edges) {
                draft.status.mark_supported();
                draft.extend_subgraph(evaluation.into_delocalization_subgraph());
            }
        }
    }

    fn ring_families(&self) -> Vec<RdkitDefaultRingFamily> {
        Smiles::<AtomPolicy>::rdkit_fused_symm_sssr_cycle_families_from_cycles(
            &self.symm_sssr_cycles,
            Self::MAX_FUSED_AROMATIC_RING_SIZE,
        )
        .into_iter()
        .map(|family_seed| {
            let RdkitCycleFamilySeed { member_cycles, member_cycle_bond_edges } = family_seed;
            if member_cycles.len() == 1 {
                RdkitDefaultRingFamily::simple_cycle_with_bond_edges(
                    member_cycles.into_iter().next().unwrap_or_default(),
                    member_cycle_bond_edges.into_iter().next().unwrap_or_default(),
                )
            } else {
                RdkitDefaultRingFamily::fused_component(member_cycles, member_cycle_bond_edges)
            }
        })
        .collect()
    }

    fn evaluate_family(&self, family: &RdkitDefaultRingFamily) -> FamilyEvaluationOutcome {
        if !family.is_supported() {
            return FamilyEvaluationOutcome {
                status: FamilyEvaluationStatus::Unsupported,
                evaluation: None,
                diagnostics: vec![AromaticityDiagnostic::UnsupportedRingFamily {
                    kind: family.kind.into(),
                    ring_count: family.member_cycles.len(),
                    atom_count: family.atom_ids.len(),
                    bond_count: family.bond_edges.len(),
                }],
            };
        }

        match family.kind {
            RdkitDefaultRingKind::SimpleCycle => self.evaluate_simple_cycle(family).into(),
            RdkitDefaultRingKind::FusedComponent => self.evaluate_fused_family(family),
        }
    }

    fn evaluate_simple_cycle(
        &self,
        family: &RdkitDefaultRingFamily,
    ) -> Option<RdkitFamilyEvaluation> {
        self.evaluate_simple_cycle_parts(&family.atom_ids, &family.bond_edges)
    }

    fn evaluate_simple_cycle_parts(
        &self,
        atom_ids: &[usize],
        bond_edges: &[[usize; 2]],
    ) -> Option<RdkitFamilyEvaluation> {
        if !self.flavor.allows_simple_cycle_size(atom_ids.len()) {
            return None;
        }
        if !self.flavor.allows_standalone_five_member_aromatic_cycles() && atom_ids.len() == 5 {
            return None;
        }
        if !atom_ids.iter().copied().all(|atom_id| self.atom_states[atom_id].candidate) {
            return None;
        }
        if !self.apply_huckel(atom_ids) {
            return None;
        }

        Some(RdkitFamilyEvaluation {
            delocalization_subgraph: RdkitDelocalizationSubgraph {
                atom_ids: atom_ids.to_vec(),
                bond_edges: bond_edges.to_vec(),
            },
        })
    }

    fn evaluate_fused_family(&self, family: &RdkitDefaultRingFamily) -> FamilyEvaluationOutcome {
        if self.flavor == RdkitAromaticityFlavor::Simple {
            return self.evaluate_simple_fused_family(family).into();
        }
        self.evaluate_fused_family_with_budget(family, RdkitFusedSubsystemBudget::default())
    }

    fn evaluate_simple_fused_family(
        &self,
        family: &RdkitDefaultRingFamily,
    ) -> Option<RdkitFamilyEvaluation> {
        let mut atom_ids = Vec::<usize>::new();
        let mut bond_edges = Vec::<[usize; 2]>::new();

        for (cycle, cycle_bond_edges) in
            family.member_cycles.iter().zip(family.member_cycle_bond_edges.iter())
        {
            if !self.flavor.allows_simple_cycle_size(cycle.len()) {
                continue;
            }
            let Some(evaluation) = self.evaluate_simple_cycle_parts(cycle, cycle_bond_edges) else {
                continue;
            };
            atom_ids.extend(evaluation.delocalization_subgraph.atom_ids);
            bond_edges.extend(evaluation.delocalization_subgraph.bond_edges);
        }

        if bond_edges.is_empty() {
            return None;
        }

        atom_ids.sort_unstable();
        atom_ids.dedup();
        bond_edges.sort_unstable();
        bond_edges.dedup();

        Some(RdkitFamilyEvaluation {
            delocalization_subgraph: RdkitDelocalizationSubgraph { atom_ids, bond_edges },
        })
    }

    fn evaluate_fused_family_with_budget(
        &self,
        family: &RdkitDefaultRingFamily,
        budget: RdkitFusedSubsystemBudget,
    ) -> FamilyEvaluationOutcome {
        let prepared = RdkitPreparedFusedFamily::new(family);
        let ring_count = family.member_cycles.len();
        let ring_bond_count = family.bond_edges.len();
        let mut scratch =
            RdkitSubsystemScratch::new(self.smiles.nodes().len(), prepared.family_bond_edges.len());
        let mut family_accumulator =
            RdkitFusedFamilyAccumulator::new(prepared.family_bond_edges.len());
        let mut hit_budget_cap = false;
        let mut hit_visit_budget = false;
        let mut subsystem_visits: usize = 0;
        let connected_subsystem_search =
            RdkitConnectedSubsystemSearch::new(&prepared.ring_neighbors);
        let max_subsystem_size = ring_count.min(budget.max_fused_subsystem_rings);

        'subsystem_sizes: for subsystem_size in 1..=max_subsystem_size {
            if subsystem_size == 2 && ring_count > budget.max_ring_combination_search {
                hit_budget_cap = true;
                break;
            }

            let should_continue =
                connected_subsystem_search.for_each_exact_size_stateful(subsystem_size, |event| {
                    match event {
                        RdkitConnectedSubsystemEvent::Enter(cycle_index) => {
                            scratch.push_cycle(family, &prepared, &self.atom_states, cycle_index);
                            true
                        }
                        RdkitConnectedSubsystemEvent::Exit(cycle_index) => {
                            scratch.pop_cycle(family, &prepared, &self.atom_states, cycle_index);
                            true
                        }
                        RdkitConnectedSubsystemEvent::Visit(_) => {
                            subsystem_visits = subsystem_visits.saturating_add(1);
                            if subsystem_visits > budget.max_subsystem_visits {
                                hit_visit_budget = true;
                                return false;
                            }
                            if Self::evaluate_current_subsystem(&mut scratch) {
                                family_accumulator.record_subsystem(&scratch);
                            }

                            family_accumulator.done_bond_count() < ring_bond_count
                        }
                    }
                });
            if !should_continue {
                break 'subsystem_sizes;
            }
        }

        if hit_visit_budget {
            hit_budget_cap = true;
        }

        if !hit_budget_cap
            && budget.max_fused_subsystem_rings < ring_count
            && family_accumulator.done_bond_count() < ring_bond_count
        {
            hit_budget_cap = true;
        }

        let evaluation = family_accumulator.finish(&prepared.family_bond_edges);
        if evaluation.is_none() {
            return FamilyEvaluationOutcome {
                status: if hit_budget_cap {
                    FamilyEvaluationStatus::Partial
                } else {
                    FamilyEvaluationStatus::Supported
                },
                evaluation: None,
                diagnostics: Self::fused_budget_diagnostics(
                    family,
                    budget,
                    hit_budget_cap,
                    hit_visit_budget,
                ),
            };
        }

        FamilyEvaluationOutcome {
            status: if hit_budget_cap {
                FamilyEvaluationStatus::Partial
            } else {
                FamilyEvaluationStatus::Supported
            },
            evaluation,
            diagnostics: Self::fused_budget_diagnostics(
                family,
                budget,
                hit_budget_cap,
                hit_visit_budget,
            ),
        }
    }

    fn fused_budget_diagnostics(
        family: &RdkitDefaultRingFamily,
        budget: RdkitFusedSubsystemBudget,
        hit_budget_cap: bool,
        hit_visit_budget: bool,
    ) -> Vec<AromaticityDiagnostic> {
        if !hit_budget_cap {
            return Vec::new();
        }

        if hit_visit_budget {
            return vec![AromaticityDiagnostic::SubsystemVisitBudgetExceeded {
                ring_count: family.member_cycles.len(),
                budget: budget.max_subsystem_visits,
            }];
        }

        vec![AromaticityDiagnostic::FusedSubsystemBudgetExceeded {
            ring_count: family.member_cycles.len(),
            max_fused_subsystem_rings: budget.max_fused_subsystem_rings,
            max_ring_combination_search: budget.max_ring_combination_search,
        }]
    }

    fn evaluate_current_subsystem(scratch: &mut RdkitSubsystemScratch) -> bool {
        if !Self::subsystem_atoms_are_candidates_and_satisfy_huckel(scratch) {
            return false;
        }

        if scratch.unique_bond_indices.is_empty() {
            return false;
        }

        true
    }

    fn subsystem_atoms_are_candidates_and_satisfy_huckel(scratch: &RdkitSubsystemScratch) -> bool {
        if scratch.non_candidate_atom_count > 0 {
            return false;
        }
        if scratch.any_donor_count > 1 {
            return false;
        }
        if scratch.contributing_atom_count == 0 {
            return false;
        }

        if scratch.upper_bound >= 6 {
            (scratch.lower_bound..=scratch.upper_bound).any(|electrons| (electrons - 2) % 4 == 0)
        } else {
            scratch.upper_bound == 2
        }
    }

    fn apply_huckel(&self, atom_ids: &[usize]) -> bool {
        let mut lower_bound = 0_u8;
        let mut upper_bound = 0_u8;
        let mut any_donor_count = 0_usize;

        for atom_id in atom_ids {
            let donor_type = self.atom_states[*atom_id].donor_type;
            if donor_type == ElectronDonorType::Any {
                any_donor_count += 1;
                if any_donor_count > 1 {
                    return false;
                }
            }
            let (atom_lower_bound, atom_upper_bound) =
                RdkitDefaultElectronModel::electron_range(donor_type);
            lower_bound = lower_bound.saturating_add(atom_lower_bound);
            upper_bound = upper_bound.saturating_add(atom_upper_bound);
        }

        if upper_bound >= 6 {
            (lower_bound..=upper_bound).any(|electrons| (electrons - 2) % 4 == 0)
        } else {
            upper_bound == 2
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(clippy::struct_field_names)]
struct RdkitFusedSubsystemBudget {
    max_fused_subsystem_rings: usize,
    max_ring_combination_search: usize,
    max_subsystem_visits: usize,
}

impl Default for RdkitFusedSubsystemBudget {
    fn default() -> Self {
        Self {
            max_fused_subsystem_rings: 6,
            max_ring_combination_search: 300,
            max_subsystem_visits: 100_000,
        }
    }
}

impl RdkitPreparedFusedFamily {
    fn new(family: &RdkitDefaultRingFamily) -> Self {
        let eligible_cycle_indices = vec![true; family.member_cycles.len()];
        let ring_neighbors =
            rdkit_fused_cycle_neighbors(&family.member_cycle_bond_edges, &eligible_cycle_indices);
        let member_cycle_bond_indices = family
            .member_cycle_bond_edges
            .iter()
            .map(|cycle_edges| {
                cycle_edges
                    .iter()
                    .map(|edge| {
                        family.bond_edges.binary_search(edge).unwrap_or_else(|_| {
                            unreachable!("family bond edges include each member cycle bond edge")
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self {
            member_cycle_bond_indices,
            ring_neighbors,
            family_bond_edges: family.bond_edges.clone(),
        }
    }
}

impl RdkitSubsystemScratch {
    const MISSING_POSITION: usize = usize::MAX;

    fn new(atom_count: usize, bond_count: usize) -> Self {
        Self {
            atom_multiplicity: vec![0_u8; atom_count],
            active_atom_ids: Vec::new(),
            active_atom_positions: vec![Self::MISSING_POSITION; atom_count],
            non_candidate_atom_count: 0,
            lower_bound: 0,
            upper_bound: 0,
            any_donor_count: 0,
            contributing_atom_count: 0,
            bond_multiplicity: vec![0_u8; bond_count],
            unique_bond_indices: Vec::new(),
            unique_bond_positions: vec![Self::MISSING_POSITION; bond_count],
        }
    }

    fn push_cycle(
        &mut self,
        family: &RdkitDefaultRingFamily,
        prepared: &RdkitPreparedFusedFamily,
        atom_states: &[RdkitPerAtomAromaticityState],
        cycle_index: usize,
    ) {
        for atom_id in family.member_cycles[cycle_index].iter().copied() {
            self.increment_atom(atom_id, &atom_states[atom_id]);
        }
        for bond_index in prepared.member_cycle_bond_indices[cycle_index].iter().copied() {
            self.increment_bond(bond_index);
        }
    }

    fn pop_cycle(
        &mut self,
        family: &RdkitDefaultRingFamily,
        prepared: &RdkitPreparedFusedFamily,
        atom_states: &[RdkitPerAtomAromaticityState],
        cycle_index: usize,
    ) {
        for atom_id in family.member_cycles[cycle_index].iter().copied().rev() {
            self.decrement_atom(atom_id, &atom_states[atom_id]);
        }
        for bond_index in prepared.member_cycle_bond_indices[cycle_index].iter().copied().rev() {
            self.decrement_bond(bond_index);
        }
    }

    fn increment_atom(&mut self, atom_id: usize, atom_state: &RdkitPerAtomAromaticityState) {
        let old_multiplicity = self.atom_multiplicity[atom_id];
        match old_multiplicity {
            0 => {
                self.add_active_atom(atom_id);
                if !atom_state.candidate {
                    self.non_candidate_atom_count += 1;
                }
                self.add_electron_contribution(atom_state.donor_type);
            }
            2 => self.remove_electron_contribution(atom_state.donor_type),
            _ => {}
        }
        self.atom_multiplicity[atom_id] = old_multiplicity + 1;
    }

    fn decrement_atom(&mut self, atom_id: usize, atom_state: &RdkitPerAtomAromaticityState) {
        let old_multiplicity = self.atom_multiplicity[atom_id];
        debug_assert!(old_multiplicity > 0);
        match old_multiplicity {
            1 => {
                self.remove_electron_contribution(atom_state.donor_type);
                if !atom_state.candidate {
                    self.non_candidate_atom_count -= 1;
                }
                self.remove_active_atom(atom_id);
            }
            3 => self.add_electron_contribution(atom_state.donor_type),
            _ => {}
        }
        self.atom_multiplicity[atom_id] = old_multiplicity - 1;
    }

    fn increment_bond(&mut self, bond_index: usize) {
        let old_multiplicity = self.bond_multiplicity[bond_index];
        match old_multiplicity {
            0 => self.add_unique_bond(bond_index),
            1 => self.remove_unique_bond(bond_index),
            _ => {}
        }
        self.bond_multiplicity[bond_index] = old_multiplicity + 1;
    }

    fn decrement_bond(&mut self, bond_index: usize) {
        let old_multiplicity = self.bond_multiplicity[bond_index];
        debug_assert!(old_multiplicity > 0);
        match old_multiplicity {
            1 => self.remove_unique_bond(bond_index),
            2 => self.add_unique_bond(bond_index),
            _ => {}
        }
        self.bond_multiplicity[bond_index] = old_multiplicity - 1;
    }

    fn add_unique_bond(&mut self, bond_index: usize) {
        let position = self.unique_bond_positions[bond_index];
        if position != Self::MISSING_POSITION {
            return;
        }
        self.unique_bond_positions[bond_index] = self.unique_bond_indices.len();
        self.unique_bond_indices.push(bond_index);
    }

    fn remove_unique_bond(&mut self, bond_index: usize) {
        let position = self.unique_bond_positions[bond_index];
        debug_assert_ne!(position, Self::MISSING_POSITION);
        let removed_bond_index = self.unique_bond_indices.swap_remove(position);
        debug_assert_eq!(removed_bond_index, bond_index);
        if let Some(&moved_bond_index) = self.unique_bond_indices.get(position) {
            self.unique_bond_positions[moved_bond_index] = position;
        }
        self.unique_bond_positions[bond_index] = Self::MISSING_POSITION;
    }

    fn add_active_atom(&mut self, atom_id: usize) {
        let position = self.active_atom_positions[atom_id];
        if position != Self::MISSING_POSITION {
            return;
        }
        self.active_atom_positions[atom_id] = self.active_atom_ids.len();
        self.active_atom_ids.push(atom_id);
    }

    fn remove_active_atom(&mut self, atom_id: usize) {
        let position = self.active_atom_positions[atom_id];
        debug_assert_ne!(position, Self::MISSING_POSITION);
        let removed_atom_id = self.active_atom_ids.swap_remove(position);
        debug_assert_eq!(removed_atom_id, atom_id);
        if let Some(&moved_atom_id) = self.active_atom_ids.get(position) {
            self.active_atom_positions[moved_atom_id] = position;
        }
        self.active_atom_positions[atom_id] = Self::MISSING_POSITION;
    }

    fn add_electron_contribution(&mut self, donor_type: ElectronDonorType) {
        let (lower_bound, upper_bound) = RdkitDefaultElectronModel::electron_range(donor_type);
        self.lower_bound = self.lower_bound.saturating_add(lower_bound);
        self.upper_bound = self.upper_bound.saturating_add(upper_bound);
        if donor_type == ElectronDonorType::Any {
            self.any_donor_count += 1;
        }
        self.contributing_atom_count += 1;
    }

    fn remove_electron_contribution(&mut self, donor_type: ElectronDonorType) {
        let (lower_bound, upper_bound) = RdkitDefaultElectronModel::electron_range(donor_type);
        self.lower_bound -= lower_bound;
        self.upper_bound -= upper_bound;
        if donor_type == ElectronDonorType::Any {
            self.any_donor_count -= 1;
        }
        self.contributing_atom_count -= 1;
    }
}

impl RdkitFusedFamilyAccumulator {
    fn new(bond_count: usize) -> Self {
        Self {
            aromatic_bond_seen: vec![false; bond_count],
            aromatic_bond_indices: Vec::new(),
            done_bond_seen: vec![false; bond_count],
            done_bond_count: 0,
        }
    }

    fn record_subsystem(&mut self, scratch: &RdkitSubsystemScratch) {
        for bond_index in scratch.unique_bond_indices.iter().copied() {
            if !self.done_bond_seen[bond_index] {
                self.done_bond_seen[bond_index] = true;
                self.done_bond_count += 1;
            }
            if !self.aromatic_bond_seen[bond_index] {
                self.aromatic_bond_seen[bond_index] = true;
                self.aromatic_bond_indices.push(bond_index);
            }
        }
    }

    fn done_bond_count(&self) -> usize {
        self.done_bond_count
    }

    fn finish(self, family_bond_edges: &[[usize; 2]]) -> Option<RdkitFamilyEvaluation> {
        if self.aromatic_bond_indices.is_empty() {
            return None;
        }

        let mut atom_ids = Vec::<usize>::new();
        let bond_edges = self
            .aromatic_bond_indices
            .into_iter()
            .map(|bond_index| family_bond_edges[bond_index])
            .inspect(|edge| {
                atom_ids.push(edge[0]);
                atom_ids.push(edge[1]);
            })
            .collect::<Vec<_>>();
        atom_ids.sort_unstable();
        atom_ids.dedup();

        Some(RdkitFamilyEvaluation {
            delocalization_subgraph: RdkitDelocalizationSubgraph { atom_ids, bond_edges },
        })
    }
}

impl RdkitDefaultRingFamily {
    fn simple_cycle_with_bond_edges(cycle: Vec<usize>, bond_edges: Vec<[usize; 2]>) -> Self {
        Self {
            kind: RdkitDefaultRingKind::SimpleCycle,
            member_cycles: vec![cycle.clone()],
            member_cycle_bond_edges: vec![bond_edges.clone()],
            atom_ids: cycle,
            bond_edges,
        }
    }

    fn fused_component(
        member_cycles: Vec<Vec<usize>>,
        member_cycle_bond_edges: Vec<Vec<[usize; 2]>>,
    ) -> Self {
        debug_assert_eq!(member_cycles.len(), member_cycle_bond_edges.len());
        let mut atom_ids =
            member_cycles.iter().flat_map(|cycle| cycle.iter().copied()).collect::<Vec<_>>();
        let mut bond_edges = member_cycle_bond_edges
            .iter()
            .flat_map(|cycle_bond_edges| cycle_bond_edges.iter().copied())
            .collect::<Vec<_>>();
        atom_ids.sort_unstable();
        atom_ids.dedup();
        bond_edges.sort_unstable();
        bond_edges.dedup();

        Self {
            kind: RdkitDefaultRingKind::FusedComponent,
            member_cycles,
            member_cycle_bond_edges,
            atom_ids,
            bond_edges,
        }
    }

    fn is_supported(&self) -> bool {
        match self.kind {
            RdkitDefaultRingKind::SimpleCycle => {
                self.atom_ids.len() >= 3 && self.member_cycles.len() == 1
            }
            RdkitDefaultRingKind::FusedComponent => {
                !self.member_cycles.is_empty()
                    && !self.atom_ids.is_empty()
                    && !self.bond_edges.is_empty()
            }
        }
    }
}

fn default_valence_for(element: Element, charge: i8) -> Option<u8> {
    element
        .valences_at_charge(charge)
        .first()
        .copied()
        .or_else(|| element.valences_at_charge(0).first().copied())
}

fn rdkit_adjusted_default_valence(element: Element, formal_charge: i8) -> Option<u8> {
    let adjusted_atomic_number = i16::from(u8::from(element)) - i16::from(formal_charge);
    let adjusted_element = u8::try_from(adjusted_atomic_number)
        .ok()
        .and_then(|atomic_number| Element::try_from(atomic_number).ok())?;
    default_valence_for(adjusted_element, 0)
}

fn assign_radicals<AtomPolicy: SmilesAtomPolicy>(smiles: &Smiles<AtomPolicy>) -> Vec<u8> {
    smiles
        .nodes()
        .iter()
        .enumerate()
        .map(|(atom_id, atom)| assign_radicals_for_atom(smiles, atom_id, atom))
        .collect()
}

fn assign_radicals_for_atom<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    atom_id: usize,
    atom: &Atom,
) -> u8 {
    if !atom.is_bracket_atom() {
        return 0;
    }
    let Some(element) = atom.element() else {
        return 0;
    };

    let charge = i16::from(atom.charge_value());
    let outer_electrons = i16::from(rdkit_outer_electrons(element));
    let total_valence = raw_total_valence(smiles, atom_id);
    let allowed_valences = element.allowed_valences();

    if !allowed_valences.is_empty() {
        let base_count = if matches!(element, Element::H | Element::He) { 2 } else { 8 };
        let mut radical_electrons = base_count - outer_electrons - total_valence + charge;
        if radical_electrons < 0 {
            radical_electrons = 0;
            if allowed_valences.len() > 1 {
                for valence in allowed_valences {
                    let candidate = i16::from(*valence) - total_valence + charge;
                    if candidate >= 0 {
                        radical_electrons = candidate;
                        break;
                    }
                }
            }
        }
        let earlier_element_radicals = outer_electrons - total_valence - charge;
        if earlier_element_radicals >= 0 {
            radical_electrons = radical_electrons.min(earlier_element_radicals);
        }
        return u8::try_from(radical_electrons).unwrap_or(0);
    }

    if smiles.edge_count_for_node(atom_id) != 0 {
        return 0;
    }
    let available_valence = (outer_electrons - charge).max(0);
    u8::try_from(available_valence % 2).unwrap_or(0)
}

fn raw_total_valence<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    atom_id: usize,
) -> i16 {
    i16::from(smiles.total_valence(atom_id))
}

pub(crate) fn rdkit_smarts_total_valence(
    smiles: &Smiles<impl SmilesAtomPolicy>,
    atom_id: usize,
    aromaticity: &AromaticityAssignment,
) -> u8 {
    let raw_total_valence = smiles.total_valence(atom_id);
    if !aromaticity.contains_atom(atom_id) {
        return raw_total_valence;
    }

    let Some(atom) = smiles.node_by_id(atom_id) else {
        unreachable!("caller validates atom id before delegating to RDKit SMARTS valence helper");
    };
    let Some(element) = atom.element() else {
        return raw_total_valence;
    };

    rdkit_adjusted_default_valence(element, atom.charge_value())
        .map_or(raw_total_valence, |default_valence| raw_total_valence.max(default_valence))
}

fn bond_valence_contribution(bond: Bond) -> usize {
    match bond {
        Bond::Single | Bond::Up | Bond::Down => 1,
        Bond::Double => 2,
        Bond::Triple => 3,
        Bond::Quadruple => 4,
    }
}

fn more_electronegative(other: Element, current: Element) -> bool {
    let other_outer = rdkit_outer_electrons(other);
    let current_outer = rdkit_outer_electrons(current);
    if other_outer != current_outer {
        return other_outer > current_outer;
    }
    u8::from(other) < u8::from(current)
}

const RDKIT_OUTER_ELECTRONS: [u8; 119] = [
    0, 1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2,
    3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 3, 4, 5,
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4,
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
];

fn rdkit_outer_electrons(element: Element) -> u8 {
    let atomic_number = usize::from(u8::from(element));
    RDKIT_OUTER_ELECTRONS.get(atomic_number).copied().unwrap_or_else(|| {
        unreachable!("RDKit outer-electron table missing atomic number {atomic_number}")
    })
}

fn element_is_carbon_or_nitrogen(element: Option<Element>) -> bool {
    matches!(element, Some(Element::C | Element::N))
}

fn element_is_beyond_second_row(element: Option<Element>) -> bool {
    element.is_some_and(|element| {
        !matches!(
            element,
            Element::H
                | Element::He
                | Element::Li
                | Element::Be
                | Element::B
                | Element::C
                | Element::N
                | Element::O
                | Element::F
                | Element::Ne
        )
    })
}

fn element_is_beyond_argon(element: Option<Element>) -> bool {
    element.is_some_and(|element| {
        !matches!(
            element,
            Element::H
                | Element::He
                | Element::Li
                | Element::Be
                | Element::B
                | Element::C
                | Element::N
                | Element::O
                | Element::F
                | Element::Ne
                | Element::Na
                | Element::Mg
                | Element::Al
                | Element::Si
                | Element::P
                | Element::S
                | Element::Cl
                | Element::Ar
        )
    })
}

fn element_is_se_or_te(element: Option<Element>) -> bool {
    matches!(element, Some(Element::Se | Element::Te))
}

fn rdkit_fused_cycle_neighbors(
    member_cycle_bond_edges: &[Vec<[usize; 2]>],
    eligible_cycle_indices: &[bool],
) -> Vec<Vec<usize>> {
    // The original pairwise cycle comparison was quadratic in the family size.
    // Build the same "share exactly one bond" relation by inverting the
    // bond-to-cycle incidence once, then counting shared edges per cycle pair.
    let mut bond_to_cycle_indices =
        HashMap::<[usize; 2], Vec<usize>>::with_capacity(member_cycle_bond_edges.len() * 6);

    for (cycle_index, cycle_edges) in member_cycle_bond_edges.iter().enumerate() {
        if !eligible_cycle_indices[cycle_index] {
            continue;
        }
        for &edge in cycle_edges {
            bond_to_cycle_indices.entry(edge).or_default().push(cycle_index);
        }
    }

    let mut shared_edge_counts =
        HashMap::<[usize; 2], u8>::with_capacity(member_cycle_bond_edges.len() * 2);
    for cycle_indices in bond_to_cycle_indices.values() {
        for left_position in 0..cycle_indices.len() {
            for right_position in left_position + 1..cycle_indices.len() {
                let pair = [cycle_indices[left_position], cycle_indices[right_position]];
                *shared_edge_counts.entry(pair).or_insert(0) += 1;
            }
        }
    }

    let mut ring_neighbors = vec![Vec::<usize>::new(); member_cycle_bond_edges.len()];
    for ([left_index, right_index], shared_edge_count) in shared_edge_counts {
        if shared_edge_count == 1 {
            ring_neighbors[left_index].push(right_index);
            ring_neighbors[right_index].push(left_index);
        }
    }
    for neighbors in &mut ring_neighbors {
        neighbors.sort_unstable();
    }

    ring_neighbors
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
struct RdkitConnectedSubsystemSearch<'a> {
    ring_neighbors: &'a [Vec<usize>],
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RdkitConnectedSubsystemState {
    selected_cycle_indices: Vec<usize>,
    selected_cycle_seen: Vec<bool>,
    seen_subsystems_small: HashSet<u128>,
    seen_subsystems_large: HashSet<Vec<usize>>,
    frontier_buffers: Vec<Vec<usize>>,
    sorted_cycle_indices: Vec<usize>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct RdkitConnectedSubsystemQuery {
    root_ring_index: usize,
    subsystem_size: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum RdkitConnectedSubsystemEvent<'a> {
    Enter(usize),
    Exit(usize),
    Visit(&'a [usize]),
}

impl<'a> RdkitConnectedSubsystemSearch<'a> {
    fn new(ring_neighbors: &'a [Vec<usize>]) -> Self {
        Self { ring_neighbors }
    }

    #[cfg(test)]
    fn for_each_exact_size<F>(&self, subsystem_size: usize, mut visit: F) -> bool
    where
        F: FnMut(&[usize]) -> bool,
    {
        self.for_each_exact_size_stateful(subsystem_size, |event| {
            match event {
                RdkitConnectedSubsystemEvent::Enter(_) | RdkitConnectedSubsystemEvent::Exit(_) => {
                    true
                }
                RdkitConnectedSubsystemEvent::Visit(subsystem) => visit(subsystem),
            }
        })
    }

    fn for_each_exact_size_stateful<F>(&self, subsystem_size: usize, mut visit: F) -> bool
    where
        F: FnMut(RdkitConnectedSubsystemEvent<'_>) -> bool,
    {
        if subsystem_size == 0 || subsystem_size > self.ring_neighbors.len() {
            return true;
        }
        if subsystem_size == 1 {
            for ring_index in 0..self.ring_neighbors.len() {
                let should_continue = visit(RdkitConnectedSubsystemEvent::Enter(ring_index))
                    && visit(RdkitConnectedSubsystemEvent::Visit(&[ring_index]));
                let exit_result = visit(RdkitConnectedSubsystemEvent::Exit(ring_index));
                if !should_continue || !exit_result {
                    return false;
                }
            }
            return true;
        }

        let mut search_state = RdkitConnectedSubsystemState {
            selected_cycle_indices: Vec::<usize>::with_capacity(subsystem_size),
            selected_cycle_seen: vec![false; self.ring_neighbors.len()],
            seen_subsystems_small: HashSet::<u128>::new(),
            seen_subsystems_large: HashSet::<Vec<usize>>::new(),
            frontier_buffers: vec![Vec::<usize>::new(); subsystem_size.saturating_sub(1)],
            sorted_cycle_indices: Vec::<usize>::with_capacity(subsystem_size),
        };

        for root_ring_index in 0..self.ring_neighbors.len() {
            let query = RdkitConnectedSubsystemQuery { root_ring_index, subsystem_size };
            search_state.selected_cycle_indices.clear();
            search_state.selected_cycle_indices.push(root_ring_index);
            search_state.selected_cycle_seen[root_ring_index] = true;
            let enter_result = visit(RdkitConnectedSubsystemEvent::Enter(root_ring_index));
            if !enter_result {
                search_state.selected_cycle_seen[root_ring_index] = false;
                return false;
            }
            let mut frontier = self.ring_neighbors[root_ring_index]
                .iter()
                .copied()
                .filter(|neighbor_ring_index| *neighbor_ring_index > root_ring_index)
                .collect::<Vec<_>>();
            frontier.sort_unstable();
            frontier.dedup();
            let should_continue =
                self.expand_from_root(query, &mut search_state, &frontier, &mut visit);
            let exit_result = visit(RdkitConnectedSubsystemEvent::Exit(root_ring_index));
            search_state.selected_cycle_seen[root_ring_index] = false;
            if !should_continue || !exit_result {
                return false;
            }
        }

        true
    }

    #[cfg(test)]
    fn enumerate_exact_size(&self, subsystem_size: usize) -> Vec<Vec<usize>> {
        let mut connected_subsystems = Vec::<Vec<usize>>::new();
        let _ = self.for_each_exact_size(subsystem_size, |subsystem| {
            connected_subsystems.push(subsystem.to_vec());
            true
        });
        connected_subsystems
    }

    fn expand_from_root<F>(
        &self,
        query: RdkitConnectedSubsystemQuery,
        search_state: &mut RdkitConnectedSubsystemState,
        frontier: &[usize],
        visit: &mut F,
    ) -> bool
    where
        F: FnMut(RdkitConnectedSubsystemEvent<'_>) -> bool,
    {
        if search_state.selected_cycle_indices.len() == query.subsystem_size {
            if self.ring_neighbors.len() <= u128::BITS as usize {
                let mut subsystem_key = 0_u128;
                search_state.sorted_cycle_indices.clear();
                for (ring_index, is_selected) in search_state.selected_cycle_seen.iter().enumerate()
                {
                    if *is_selected {
                        subsystem_key |= 1_u128 << ring_index;
                        search_state.sorted_cycle_indices.push(ring_index);
                    }
                }
                if search_state.seen_subsystems_small.insert(subsystem_key) {
                    return visit(RdkitConnectedSubsystemEvent::Visit(
                        &search_state.sorted_cycle_indices,
                    ));
                }
            } else {
                let mut subsystem = search_state.selected_cycle_indices.clone();
                subsystem.sort_unstable();
                if search_state.seen_subsystems_large.insert(subsystem.clone()) {
                    return visit(RdkitConnectedSubsystemEvent::Visit(&subsystem));
                }
            }
            return true;
        }

        for (position, next_ring_index) in frontier.iter().copied().enumerate() {
            search_state.selected_cycle_indices.push(next_ring_index);
            search_state.selected_cycle_seen[next_ring_index] = true;
            let enter_result = visit(RdkitConnectedSubsystemEvent::Enter(next_ring_index));
            if !enter_result {
                let _ = visit(RdkitConnectedSubsystemEvent::Exit(next_ring_index));
                search_state.selected_cycle_seen[next_ring_index] = false;
                search_state.selected_cycle_indices.pop();
                return false;
            }
            let frontier_buffer_index = search_state.selected_cycle_indices.len() - 2;
            let mut next_frontier =
                core::mem::take(&mut search_state.frontier_buffers[frontier_buffer_index]);
            self.build_next_frontier(
                query,
                search_state,
                &frontier[position + 1..],
                next_ring_index,
                &mut next_frontier,
            );

            let should_continue = self.expand_from_root(query, search_state, &next_frontier, visit);
            search_state.frontier_buffers[frontier_buffer_index] = next_frontier;
            let exit_result = visit(RdkitConnectedSubsystemEvent::Exit(next_ring_index));
            search_state.selected_cycle_seen[next_ring_index] = false;
            search_state.selected_cycle_indices.pop();
            if !should_continue || !exit_result {
                return false;
            }
        }

        true
    }

    fn build_next_frontier(
        &self,
        query: RdkitConnectedSubsystemQuery,
        search_state: &RdkitConnectedSubsystemState,
        frontier_suffix: &[usize],
        next_ring_index: usize,
        next_frontier: &mut Vec<usize>,
    ) {
        next_frontier.clear();

        let neighbor_ring_indices = &self.ring_neighbors[next_ring_index];
        let mut frontier_position = 0_usize;
        let mut neighbor_position = 0_usize;

        while frontier_position < frontier_suffix.len()
            || neighbor_position < neighbor_ring_indices.len()
        {
            while neighbor_position < neighbor_ring_indices.len() {
                let neighbor_ring_index = neighbor_ring_indices[neighbor_position];
                if neighbor_ring_index <= query.root_ring_index
                    || search_state.selected_cycle_seen[neighbor_ring_index]
                {
                    neighbor_position += 1;
                } else {
                    break;
                }
            }

            let next_candidate = match (
                frontier_suffix.get(frontier_position).copied(),
                neighbor_ring_indices.get(neighbor_position).copied(),
            ) {
                (Some(frontier_ring_index), Some(neighbor_ring_index)) => {
                    match frontier_ring_index.cmp(&neighbor_ring_index) {
                        core::cmp::Ordering::Less => {
                            frontier_position += 1;
                            frontier_ring_index
                        }
                        core::cmp::Ordering::Greater => {
                            neighbor_position += 1;
                            neighbor_ring_index
                        }
                        core::cmp::Ordering::Equal => {
                            frontier_position += 1;
                            neighbor_position += 1;
                            frontier_ring_index
                        }
                    }
                }
                (Some(frontier_ring_index), None) => {
                    frontier_position += 1;
                    frontier_ring_index
                }
                (None, Some(neighbor_ring_index)) => {
                    neighbor_position += 1;
                    neighbor_ring_index
                }
                (None, None) => break,
            };

            if next_frontier.last().copied() != Some(next_candidate) {
                next_frontier.push(next_candidate);
            }
        }
    }
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    fn rdkit_fused_symm_sssr_cycle_families_from_cycles(
        cycles: &[Vec<usize>],
        max_fused_ring_size: usize,
    ) -> Vec<RdkitCycleFamilySeed> {
        let cycle_bond_edges = cycles.iter().map(|cycle| cycle_edges(cycle)).collect::<Vec<_>>();
        let eligible_cycle_indices = cycle_bond_edges
            .iter()
            .map(|cycle_edges| cycle_edges.len() <= max_fused_ring_size)
            .collect::<Vec<_>>();
        let ring_neighbors =
            rdkit_fused_cycle_neighbors(&cycle_bond_edges, &eligible_cycle_indices);
        let mut families = Vec::<RdkitCycleFamilySeed>::new();
        let mut seen_cycles = vec![false; cycles.len()];

        for cycle_index in 0..cycles.len() {
            if seen_cycles[cycle_index] {
                continue;
            }

            let mut pending_cycle_indices = vec![cycle_index];
            let mut family_cycle_indices = Vec::<usize>::new();

            while let Some(current_cycle_index) = pending_cycle_indices.pop() {
                if seen_cycles[current_cycle_index] {
                    continue;
                }
                seen_cycles[current_cycle_index] = true;
                family_cycle_indices.push(current_cycle_index);

                for &next_cycle_index in &ring_neighbors[current_cycle_index] {
                    if !seen_cycles[next_cycle_index] {
                        pending_cycle_indices.push(next_cycle_index);
                    }
                }
            }

            families.push(RdkitCycleFamilySeed {
                member_cycles: family_cycle_indices
                    .iter()
                    .map(|&index| cycles[index].clone())
                    .collect::<Vec<_>>(),
                member_cycle_bond_edges: family_cycle_indices
                    .into_iter()
                    .map(|index| cycle_bond_edges[index].clone())
                    .collect::<Vec<_>>(),
            });
        }

        families
    }

    #[cfg(test)]
    pub(crate) fn fused_symm_sssr_components(&self) -> Vec<RingComponent> {
        Self::rdkit_fused_symm_sssr_cycle_families_from_cycles(
            self.symm_sssr_result().cycles(),
            RdkitDefaultContext::<AtomPolicy>::MAX_FUSED_AROMATIC_RING_SIZE,
        )
        .into_iter()
        .map(|family_seed| {
            RingComponent {
                atom_ids: family_seed
                    .member_cycles
                    .iter()
                    .flat_map(|cycle| cycle.iter().copied())
                    .collect::<Vec<_>>(),
                bond_edges: family_seed
                    .member_cycle_bond_edges
                    .iter()
                    .flat_map(|cycle_edges| cycle_edges.iter().copied())
                    .collect::<Vec<_>>(),
            }
        })
        .map(|mut component| {
            component.atom_ids.sort_unstable();
            component.atom_ids.dedup();
            component.bond_edges.sort_unstable();
            component.bond_edges.dedup();
            component
        })
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use elements_rs::Element;
    use hashbrown::{HashMap, HashSet};

    use super::{
        AromaticityStatusAccumulator, ElectronDonorType, RdkitAromaticityFlavor,
        RdkitAtomAromContext, RdkitConnectedSubsystemEvent, RdkitConnectedSubsystemQuery,
        RdkitConnectedSubsystemSearch, RdkitConnectedSubsystemState, RdkitDefaultContext,
        RdkitDefaultElectronModel, RdkitDefaultRingKind, RdkitFusedSubsystemBudget,
        RdkitPreAromaticityNormalization, Smiles, assign_radicals, more_electronegative,
    };
    use crate::bond::Bond;

    #[test]
    fn rdkit_electronegativity_order_prefers_more_outer_electrons() {
        assert!(more_electronegative(Element::O, Element::C));
        assert!(more_electronegative(Element::Se, Element::C));
        assert!(!more_electronegative(Element::C, Element::Se));
    }

    #[test]
    fn rdkit_electronegativity_order_breaks_ties_by_lower_atomic_number() {
        assert!(more_electronegative(Element::S, Element::Se));
        assert!(!more_electronegative(Element::Se, Element::S));
    }

    #[test]
    fn rdkit_electronegativity_order_keeps_zinc_less_electronegative_than_carbon() {
        assert!(!more_electronegative(Element::Zn, Element::C));
    }

    #[test]
    fn connected_subsystem_search_matches_bruteforce_on_path_graph() {
        let ring_neighbors = vec![vec![1], vec![0, 2], vec![1, 3], vec![2]];
        assert_connected_subsystem_search_matches_bruteforce(&ring_neighbors, 4);
    }

    #[test]
    fn connected_subsystem_search_matches_bruteforce_on_branched_graph() {
        let ring_neighbors = vec![vec![1, 2], vec![0, 2, 3], vec![0, 1], vec![1]];
        assert_connected_subsystem_search_matches_bruteforce(&ring_neighbors, 4);
    }

    #[test]
    fn connected_subsystem_search_short_circuits_zero_and_oversized_sizes() {
        let ring_neighbors = vec![vec![1], vec![0]];
        let search = RdkitConnectedSubsystemSearch::new(&ring_neighbors);

        assert!(search.for_each_exact_size(0, |_| false));
        assert!(search.for_each_exact_size(3, |_| false));
    }

    #[test]
    fn connected_subsystem_search_stops_when_root_enter_callback_rejects() {
        let ring_neighbors = vec![vec![1], vec![0]];
        let search = RdkitConnectedSubsystemSearch::new(&ring_neighbors);

        let should_continue = search.for_each_exact_size_stateful(2, |event| {
            !matches!(event, RdkitConnectedSubsystemEvent::Enter(0))
        });

        assert!(!should_continue);
    }

    #[test]
    fn connected_subsystem_search_stops_when_child_enter_callback_rejects() {
        let ring_neighbors = vec![vec![1], vec![0]];
        let search = RdkitConnectedSubsystemSearch::new(&ring_neighbors);
        let mut events = Vec::<(&'static str, usize)>::new();

        let should_continue = search.for_each_exact_size_stateful(2, |event| {
            let owned = match event {
                RdkitConnectedSubsystemEvent::Enter(index) => ("enter", index),
                RdkitConnectedSubsystemEvent::Exit(index) => ("exit", index),
                RdkitConnectedSubsystemEvent::Visit(subsystem) => ("visit", subsystem[0]),
            };
            events.push(owned);
            !matches!(event, RdkitConnectedSubsystemEvent::Enter(1))
        });

        assert!(!should_continue);
        assert_eq!(events, vec![("enter", 0), ("enter", 1), ("exit", 1), ("exit", 0),]);
    }

    #[test]
    fn connected_subsystem_search_uses_large_seen_set_for_graphs_over_u128_width() {
        let ring_neighbors = (0..129_usize)
            .map(|index| {
                let mut neighbors = Vec::new();
                if index > 0 {
                    neighbors.push(index - 1);
                }
                if index + 1 < 129 {
                    neighbors.push(index + 1);
                }
                neighbors
            })
            .collect::<Vec<_>>();
        let search = RdkitConnectedSubsystemSearch::new(&ring_neighbors);

        let connected_pairs = search.enumerate_exact_size(2);

        assert_eq!(connected_pairs.len(), 128);
        assert_eq!(connected_pairs.first(), Some(&vec![0, 1]));
        assert_eq!(connected_pairs.last(), Some(&vec![127, 128]));
    }

    #[test]
    fn build_next_frontier_covers_less_greater_and_equal_merge_orderings() {
        let ring_neighbors = vec![vec![], vec![3], vec![], vec![], vec![]];
        let search = RdkitConnectedSubsystemSearch::new(&ring_neighbors);
        let query = RdkitConnectedSubsystemQuery { root_ring_index: 0, subsystem_size: 2 };
        let state = RdkitConnectedSubsystemState {
            selected_cycle_indices: vec![0],
            selected_cycle_seen: vec![false; ring_neighbors.len()],
            seen_subsystems_small: HashSet::new(),
            seen_subsystems_large: HashSet::new(),
            frontier_buffers: Vec::new(),
            sorted_cycle_indices: Vec::new(),
        };
        let mut next_frontier = Vec::new();

        search.build_next_frontier(query, &state, &[2], 1, &mut next_frontier);
        assert_eq!(next_frontier, vec![2, 3]);

        search.build_next_frontier(query, &state, &[4], 1, &mut next_frontier);
        assert_eq!(next_frontier, vec![3, 4]);

        search.build_next_frontier(query, &state, &[3], 1, &mut next_frontier);
        assert_eq!(next_frontier, vec![3]);
    }

    fn assert_connected_subsystem_search_matches_bruteforce(
        ring_neighbors: &[Vec<usize>],
        ring_count: usize,
    ) {
        let search = RdkitConnectedSubsystemSearch::new(ring_neighbors);

        for subsystem_size in 1..=ring_count {
            let exact = search.enumerate_exact_size(subsystem_size);
            let brute_force = brute_force_connected_subsystems_exact_size(
                ring_neighbors,
                ring_count,
                subsystem_size,
            );

            assert_eq!(
                exact, brute_force,
                "connected subsystem search mismatch for size {subsystem_size}"
            );
        }
    }

    fn brute_force_connected_subsystems_exact_size(
        ring_neighbors: &[Vec<usize>],
        ring_count: usize,
        subsystem_size: usize,
    ) -> Vec<Vec<usize>> {
        let mut connected_subsystems = Vec::<Vec<usize>>::new();
        let mut combination = (0..subsystem_size).collect::<Vec<_>>();
        loop {
            if brute_force_subsystem_is_connected(ring_neighbors, &combination) {
                connected_subsystems.push(combination.clone());
            }
            if !advance_combination_bruteforce(&mut combination, ring_count) {
                break;
            }
        }
        connected_subsystems.sort_unstable();
        connected_subsystems
    }

    fn brute_force_subsystem_is_connected(
        ring_neighbors: &[Vec<usize>],
        selected_cycle_indices: &[usize],
    ) -> bool {
        if selected_cycle_indices.len() <= 1 {
            return true;
        }

        let mut selected = vec![false; ring_neighbors.len()];
        for cycle_index in selected_cycle_indices.iter().copied() {
            selected[cycle_index] = true;
        }
        let mut seen = vec![false; ring_neighbors.len()];
        let mut pending = vec![selected_cycle_indices[0]];
        let mut seen_count = 0_usize;

        while let Some(cycle_index) = pending.pop() {
            if seen[cycle_index] {
                continue;
            }
            seen[cycle_index] = true;
            seen_count += 1;
            for next_cycle_index in ring_neighbors[cycle_index].iter().copied() {
                if selected[next_cycle_index] && !seen[next_cycle_index] {
                    pending.push(next_cycle_index);
                }
            }
        }

        seen_count == selected_cycle_indices.len()
    }

    fn advance_combination_bruteforce(combination: &mut [usize], item_count: usize) -> bool {
        if combination.is_empty() {
            return false;
        }

        let combination_len = combination.len();
        for position in (0..combination_len).rev() {
            let max_value = item_count - (combination_len - position);
            if combination[position] < max_value {
                combination[position] += 1;
                for next_position in position + 1..combination_len {
                    combination[next_position] = combination[next_position - 1] + 1;
                }
                return true;
            }
        }

        false
    }

    #[test]
    fn aromaticity_status_accumulator_maps_partial_and_unsupported_honestly() {
        let complete = AromaticityStatusAccumulator::default().finish();

        let mut partial = AromaticityStatusAccumulator::default();
        partial.mark_partial();

        let mut unsupported = AromaticityStatusAccumulator::default();
        unsupported.mark_unsupported();

        let mut mixed = AromaticityStatusAccumulator::default();
        mixed.mark_supported();
        mixed.mark_unsupported();

        assert_eq!(complete, super::AromaticityStatus::Complete);
        assert_eq!(partial.finish(), super::AromaticityStatus::Partial);
        assert_eq!(unsupported.finish(), super::AromaticityStatus::Unsupported);
        assert_eq!(mixed.finish(), super::AromaticityStatus::Partial);
    }

    #[test]
    fn prepared_fused_family_maps_member_cycle_bond_edges_without_recomputing_cycles() {
        let family = super::RdkitDefaultRingFamily::fused_component(
            vec![vec![0, 1, 2, 3], vec![2, 3, 4, 5]],
            vec![vec![[0, 1], [1, 2], [2, 3], [0, 3]], vec![[2, 3], [3, 4], [4, 5], [2, 5]]],
        );

        let prepared = super::RdkitPreparedFusedFamily::new(&family);

        assert_eq!(
            prepared.family_bond_edges,
            vec![[0, 1], [0, 3], [1, 2], [2, 3], [2, 5], [3, 4], [4, 5]]
        );
        assert_eq!(prepared.member_cycle_bond_indices, vec![vec![0, 2, 3, 1], vec![3, 5, 6, 4]]);
        assert_eq!(prepared.ring_neighbors, vec![vec![1], vec![0]]);
    }

    #[test]
    fn quick_assignment_short_circuits_acyclic_molecule() {
        let smiles: Smiles = "CCO".parse().expect("valid acyclic molecule");

        let assignment = RdkitAromaticityFlavor::Default.quick_assignment(&smiles, &HashMap::new());

        assert_eq!(
            assignment,
            super::QuickAssignment::Ready(super::AromaticityAssignment::new(
                super::AromaticityStatus::Complete,
                Vec::new(),
                Vec::new(),
            ))
        );
    }

    #[test]
    fn quick_assignment_short_circuits_ring_without_candidates() {
        let smiles: Smiles = "C1CCCCC1".parse().expect("valid cyclohexane");

        let assignment = RdkitAromaticityFlavor::Default.quick_assignment(&smiles, &HashMap::new());

        assert_eq!(
            assignment,
            super::QuickAssignment::Ready(super::AromaticityAssignment::new(
                super::AromaticityStatus::Complete,
                Vec::new(),
                Vec::new(),
            ))
        );
    }

    #[test]
    fn quick_assignment_does_not_short_circuit_benzene() {
        let smiles: Smiles = "C1=CC=CC=C1".parse().expect("valid benzene");

        let assignment = RdkitAromaticityFlavor::Default.quick_assignment(&smiles, &HashMap::new());

        assert!(matches!(assignment, super::QuickAssignment::NeedsFullEvaluation { .. }));
    }

    #[test]
    fn rdkit_default_chalcogen_gap_atom_states_match_current_expectation() {
        let smiles: Smiles = "CN1C2=CC=CC=C2SC1=[Se]".parse().expect("valid chalcogen gap case");
        let ring_membership = smiles.ring_membership();
        let implicit_hydrogen_overrides = HashMap::new();
        let atom_states = RdkitDefaultElectronModel::build_states(
            RdkitAromaticityFlavor::Default,
            &smiles,
            &ring_membership,
            &implicit_hydrogen_overrides,
        );

        assert_eq!(atom_states[1].donor_type, ElectronDonorType::Two);
        assert!(atom_states[1].candidate);
        assert_eq!(atom_states[8].donor_type, ElectronDonorType::Two);
        assert!(atom_states[8].candidate);
        assert_eq!(atom_states[9].donor_type, ElectronDonorType::Vacant);
        assert!(atom_states[9].candidate);
    }

    #[test]
    fn rdkit_default_assigns_radical_to_kekule_radical_gap_case() {
        let smiles: Smiles = "C1=CC(=CC=[C]1)[N+](=O)[O-]".parse().expect("valid radical gap case");
        let radical_electrons = assign_radicals(&smiles);
        assert_eq!(radical_electrons, vec![0, 0, 0, 0, 0, 1, 0, 0, 0]);

        let ring_membership = smiles.ring_membership();
        let implicit_hydrogen_overrides = HashMap::new();
        let atom_states = RdkitDefaultElectronModel::build_states(
            RdkitAromaticityFlavor::Default,
            &smiles,
            &ring_membership,
            &implicit_hydrogen_overrides,
        );
        assert_eq!(atom_states[5].donor_type, ElectronDonorType::One);
        assert!(atom_states[5].candidate);
    }

    #[test]
    fn rdkit_default_phosphorus_cleanup_enables_phosphinine_oxide_candidate() {
        let smiles: Smiles = "C1=CC=C2C(=C1)C=CC=P2=O".parse().expect("valid pnictogen gap case");
        let cleaned = RdkitPreAromaticityNormalization::apply(&smiles)
            .expect("RDKit pre-aromaticity phosphorus cleanup should apply");

        assert_eq!(cleaned.smiles.nodes()[9].charge_value(), 1);
        assert_eq!(cleaned.smiles.nodes()[10].charge_value(), -1);
        assert_eq!(cleaned.implicit_hydrogen_overrides.get(&9), Some(&0));
        assert_eq!(cleaned.implicit_hydrogen_overrides.get(&10), Some(&0));
        assert_eq!(
            cleaned.smiles.edge_for_node_pair((9, 10)).expect("cleaned P-O bond").bond(),
            Bond::Single
        );

        let ring_membership = cleaned.smiles.ring_membership();
        let radical_electrons = assign_radicals(&cleaned.smiles);
        let phosphorus_context = RdkitAtomAromContext::build(
            &cleaned.smiles,
            &ring_membership,
            &cleaned.implicit_hydrogen_overrides,
            &radical_electrons,
            9,
        );
        assert_eq!(phosphorus_context.degree, 3);
        assert_eq!(phosphorus_context.degree_plus_total_h, 3);
        assert_eq!(phosphorus_context.total_valence, 4);
        assert_eq!(phosphorus_context.total_unsaturations, 1);
        assert_eq!(RdkitDefaultElectronModel::count_atom_electrons(&phosphorus_context), 1);
        let atom_states = RdkitDefaultElectronModel::build_states(
            RdkitAromaticityFlavor::Default,
            &cleaned.smiles,
            &ring_membership,
            &cleaned.implicit_hydrogen_overrides,
        );
        assert_eq!(atom_states[9].donor_type, ElectronDonorType::One);
        assert!(atom_states[9].candidate);
    }

    #[test]
    fn rdkit_pre_aromaticity_cleanup_rewrites_phosphinine_oxide() {
        let smiles: Smiles =
            "C1=CC=C2C(=C1)C=CC=P2=O".parse().expect("valid phosphinine oxide case");
        let cleaned = RdkitPreAromaticityNormalization::apply(&smiles)
            .expect("cleanup should rewrite phosphorus");

        assert_eq!(cleaned.smiles.nodes()[9].charge_value(), 1);
        assert_eq!(cleaned.smiles.nodes()[10].charge_value(), -1);
        assert_eq!(cleaned.implicit_hydrogen_overrides.get(&9), Some(&0));
        assert_eq!(cleaned.implicit_hydrogen_overrides.get(&10), Some(&0));
        assert_eq!(
            cleaned.smiles.edge_for_node_pair((9, 10)).expect("phosphorus oxygen bond").bond(),
            Bond::Single
        );
    }

    #[test]
    fn rdkit_pre_aromaticity_cleanup_skips_simple_nonring_p_oxide_examples() {
        for smiles_text in ["C=P(=O)O", "N=P(=O)O"] {
            let smiles: Smiles = smiles_text.parse().expect("valid simple phosphorus oxide case");
            assert!(
                RdkitPreAromaticityNormalization::apply(&smiles).is_none(),
                "cleanup should not rewrite non-ring phosphorus oxide case {smiles_text}"
            );
        }
    }

    #[test]
    fn rdkit_default_without_cleanup_underassigns_phosphinine_oxide() {
        let smiles: Smiles =
            "C1=CC=C2C(=C1)C=CC=P2=O".parse().expect("valid phosphinine oxide case");
        let implicit_hydrogen_overrides = HashMap::new();

        let raw_assignment = RdkitDefaultContext::new(
            RdkitAromaticityFlavor::Default,
            &smiles,
            &implicit_hydrogen_overrides,
        )
        .assignment();
        let cleaned_assignment = smiles.aromaticity_assignment();

        assert!(
            raw_assignment.atom_ids().len() < cleaned_assignment.atom_ids().len(),
            "cleanup should increase aromatic assignment for phosphinine oxide"
        );
        assert!(
            !raw_assignment.atom_ids().contains(&9),
            "raw assignment should not treat neutral phosphorus as aromatic"
        );
        assert!(
            cleaned_assignment.atom_ids().contains(&9),
            "cleaned assignment should recover the RDKit aromatic phosphorus"
        );
        assert_eq!(cleaned_assignment.status(), super::AromaticityStatus::Complete);
    }

    #[test]
    fn rdkit_default_cleaned_phosphinine_oxide_keeps_phosphorus_as_candidate() {
        let smiles: Smiles =
            "C1=CC=C2C(=C1)C=CC=P2=O".parse().expect("valid phosphinine oxide case");
        let cleaned = RdkitPreAromaticityNormalization::apply(&smiles)
            .expect("cleanup should rewrite phosphorus");
        let ring_membership = cleaned.smiles.ring_membership();
        let atom_states = RdkitDefaultElectronModel::build_states(
            RdkitAromaticityFlavor::Default,
            &cleaned.smiles,
            &ring_membership,
            &cleaned.implicit_hydrogen_overrides,
        );

        assert_eq!(atom_states[9].donor_type, ElectronDonorType::One);
        assert!(atom_states[9].candidate);
    }

    #[test]
    fn rdkit_default_reports_partial_for_large_polycycle_sssr_frontier_case() {
        let smiles: Smiles = "CC1=C2CC3=C(C4=C5C=C3[C@@H]6C2=CC7=C1CC8=C(C9=C1C=C8[C@@H]7CCC[C@@H]2C3=C7CC8=C2C=C2C%10CCCC%11C%12=CC%13=C%14CC(=C7C)C(=C3)[C@@H]%13CCC[C@H]3C7=C%13CC%15=C3C=C3C(CCCC%16C%17=CC(=C(C4)C(=C%17CC4=C%16C=C%16[C@H](CCC6)C(=C7)C(=C%13C)CC%16=C4C)C)C5CCCC1C1=CC%10=C(CC2=C8C)C(=C1C9)C)C1=CC%11=C(CC%12=C%14C)C(=C1CC3=C%15C)C)C)C"
            .parse()
            .expect("valid large-polycycle frontier case");
        let assignment = smiles.aromaticity_assignment();

        assert_eq!(assignment.status(), super::AromaticityStatus::Partial);
        assert!(!assignment.atom_ids().is_empty());
        assert!(!assignment.bond_edges().is_empty());
    }

    #[test]
    fn fused_family_ring_budget_marks_azulene_partial_when_two_ring_search_is_disallowed() {
        let smiles: Smiles = "c1cccc2cccc2c1".parse().expect("valid azulene");
        let context =
            RdkitDefaultContext::new(RdkitAromaticityFlavor::Default, &smiles, &HashMap::new());
        let family = context
            .ring_families()
            .into_iter()
            .find(|family| family.kind == RdkitDefaultRingKind::FusedComponent)
            .expect("azulene should produce a fused family");

        let default_outcome = context
            .evaluate_fused_family_with_budget(&family, RdkitFusedSubsystemBudget::default());
        let capped_outcome = context.evaluate_fused_family_with_budget(
            &family,
            RdkitFusedSubsystemBudget {
                max_fused_subsystem_rings: 1,
                max_ring_combination_search: usize::MAX,
                max_subsystem_visits: usize::MAX,
            },
        );

        assert_eq!(default_outcome.status, super::FamilyEvaluationStatus::Supported);
        assert!(default_outcome.evaluation.is_some());
        assert_eq!(capped_outcome.status, super::FamilyEvaluationStatus::Partial);
        assert_eq!(
            capped_outcome.diagnostics,
            vec![super::AromaticityDiagnostic::FusedSubsystemBudgetExceeded {
                ring_count: 2,
                max_fused_subsystem_rings: 1,
                max_ring_combination_search: usize::MAX,
            }]
        );
        assert!(capped_outcome.evaluation.is_none());
    }

    #[test]
    fn fused_family_combination_budget_marks_azulene_partial_when_two_ring_step_is_blocked() {
        let smiles: Smiles = "c1cccc2cccc2c1".parse().expect("valid azulene");
        let context =
            RdkitDefaultContext::new(RdkitAromaticityFlavor::Default, &smiles, &HashMap::new());
        let family = context
            .ring_families()
            .into_iter()
            .find(|family| family.kind == RdkitDefaultRingKind::FusedComponent)
            .expect("azulene should produce a fused family");

        let capped_outcome = context.evaluate_fused_family_with_budget(
            &family,
            RdkitFusedSubsystemBudget {
                max_fused_subsystem_rings: 2,
                max_ring_combination_search: 1,
                max_subsystem_visits: usize::MAX,
            },
        );

        assert_eq!(family.member_cycles.len(), 2);
        assert_eq!(capped_outcome.status, super::FamilyEvaluationStatus::Partial);
        assert_eq!(
            capped_outcome.diagnostics,
            vec![super::AromaticityDiagnostic::FusedSubsystemBudgetExceeded {
                ring_count: 2,
                max_fused_subsystem_rings: 2,
                max_ring_combination_search: 1,
            }]
        );
        assert!(capped_outcome.evaluation.is_none());
    }

    #[test]
    fn unsupported_family_reports_explicit_diagnostic() {
        let smiles: Smiles = "c1ccccc1".parse().expect("valid benzene");
        let context =
            RdkitDefaultContext::new(RdkitAromaticityFlavor::Default, &smiles, &HashMap::new());
        let unsupported_family = super::RdkitDefaultRingFamily {
            kind: super::RdkitDefaultRingKind::SimpleCycle,
            member_cycles: Vec::new(),
            member_cycle_bond_edges: Vec::new(),
            atom_ids: Vec::new(),
            bond_edges: Vec::new(),
        };

        let outcome = context.evaluate_family(&unsupported_family);

        assert_eq!(outcome.status, super::FamilyEvaluationStatus::Unsupported);
        assert_eq!(
            outcome.diagnostics,
            vec![super::AromaticityDiagnostic::UnsupportedRingFamily {
                kind: super::AromaticityRingFamilyKind::SimpleCycle,
                ring_count: 0,
                atom_count: 0,
                bond_count: 0,
            }]
        );
        assert!(outcome.evaluation.is_none());
    }
}
