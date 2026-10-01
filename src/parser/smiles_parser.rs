//! Parser state used while turning tokenized SMILES into a graph.

use alloc::vec::Vec;
use core::marker::PhantomData;

use elements_rs::{Element, Isotope};

use crate::{
    atom::Atom,
    bond::{Bond, BondDescriptor, ring_num::RingNum},
    errors::{SmilesError, SmilesErrorWithSpan},
    parser::token_iter::TokenIter,
    smiles::{BondMatrixBuilder, Smiles, SmilesAtomPolicy, StereoNeighbor, WildcardAtoms},
    token::{Token, TokenKind, TokenWithSpan},
};

#[inline]
fn next_token(tokens: &mut TokenIter<'_>) -> Result<Option<TokenWithSpan>, SmilesErrorWithSpan> {
    match tokens.next() {
        Some(Ok(token)) => Ok(Some(token)),
        Some(Err(error)) => Err(error),
        None => Ok(None),
    }
}

pub(crate) fn parse_smiles(input: &str) -> Result<Smiles, SmilesErrorWithSpan> {
    parse_smiles_with_policy(input)
}

pub(crate) fn parse_wildcard_smiles(
    input: &str,
) -> Result<Smiles<WildcardAtoms>, SmilesErrorWithSpan> {
    parse_smiles_with_policy(input)
}

pub(crate) fn parse_smiles_with_policy<AtomPolicy: SmilesAtomPolicy>(
    input: &str,
) -> Result<Smiles<AtomPolicy>, SmilesErrorWithSpan> {
    if input.is_empty() {
        return Err(SmilesErrorWithSpan::new(SmilesError::MissingElement, 0, 0));
    }

    let mut tokens = TokenIter::from(input);
    let mut parser_state = ParserState::<AtomPolicy>::new_for_policy(input.len());
    let mut previous = None;
    let mut current = next_token(&mut tokens)?;
    let mut next = next_token(&mut tokens)?;

    while let Some(token_with_span) = current.take() {
        let (start, end) = (token_with_span.start(), token_with_span.end());
        let token = token_with_span.token();
        let token_kind = token.kind();
        let next_kind = next.as_ref().map(TokenWithSpan::token_kind);

        parser_state.update_last_span((start, end));
        match token {
            Token::Atom(atom) => parser_state.add_atom(atom, start, end)?,
            Token::Bond(bond) => {
                parser_state.validate_and_add_bond(start, end, bond, next_kind)?;
            }
            Token::LeftParentheses => {
                parser_state.validate_branch_open(start, end, next_kind)?;
            }
            Token::NonBond => {
                parser_state.validate_component_boundary()?;
                ParserState::<AtomPolicy>::validate_non_bond(previous, next_kind, start, end)?;
            }
            Token::RingClosure(ring_num) => {
                parser_state.validate_and_add_ring_num(start, end, ring_num)?;
            }
            Token::RightParentheses => {
                parser_state.validate_branch_close(start, end)?;
            }
        }

        previous = Some(token_kind);
        current = next.take();
        next = next_token(&mut tokens)?;
    }

    parser_state.validate_all_closed()?;
    Ok(parser_state.into_smiles())
}

/// Structure containing parser state.
struct ParserState<AtomPolicy = crate::smiles::ConcreteAtoms> {
    /// Nodes accumulated during parsing.
    atom_nodes: Vec<Atom>,
    /// Bonds accumulated during parsing.
    bond_matrix: BondMatrixBuilder,
    /// The last seen atom if present  
    last_atom: Option<usize>,
    /// A pending bond that needs to be connected to a second atom
    pending_bond: Option<BondDescriptor>,
    /// The stack of branch anchor atoms
    branch_stack: Vec<usize>,
    /// Open ring closures indexed by ring label.
    ring_open: [Option<(usize, Option<BondDescriptor>)>; 100],
    /// Parsed lexical stereo neighbor order per atom, preserving ring-digit
    /// position.
    parsed_stereo_neighbors: Vec<Vec<PendingStereoNeighbor>>,
    /// The last used span
    last_span: (usize, usize),
    atom_policy: PhantomData<fn() -> AtomPolicy>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingStereoNeighbor {
    Atom(usize),
    ExplicitHydrogen,
    RingLabel(RingNum),
}

impl<AtomPolicy: SmilesAtomPolicy> ParserState<AtomPolicy> {
    /// Creates a new initial state for the parser.
    #[must_use]
    fn new_for_policy(input_len: usize) -> Self {
        Self {
            atom_nodes: Vec::with_capacity(input_len),
            bond_matrix: BondMatrixBuilder::with_capacity(input_len),
            last_atom: None,
            pending_bond: None,
            branch_stack: Vec::with_capacity(input_len.min(16)),
            ring_open: [None; 100],
            parsed_stereo_neighbors: Vec::with_capacity(input_len),
            last_span: (0, 0),
            atom_policy: PhantomData,
        }
    }
    /// Updates the last span field.
    fn update_last_span(&mut self, last_span: (usize, usize)) {
        self.last_span = last_span;
    }
    /// Returns the previous atom ID if there is one.
    #[must_use]
    fn last_atom(&self) -> Option<usize> {
        self.last_atom
    }
    /// Updates the last atom ID.
    fn update_last_atom(&mut self, id: Option<usize>) {
        self.last_atom = id;
    }
    /// Returns the pending bond if present
    #[must_use]
    fn pending_bond(&self) -> Option<BondDescriptor> {
        self.pending_bond
    }
    /// Updates the pending bond field.
    fn update_pending_bond(&mut self, bond: Option<BondDescriptor>) {
        self.pending_bond = bond;
    }
    /// Pops a value off the the branch stack and returns it.
    fn pop_branch_stack(&mut self) -> Option<usize> {
        self.branch_stack.pop()
    }
    /// Pushes a branch anchor to the branch stack.
    fn push_stack(&mut self, anchor: usize) {
        self.branch_stack.push(anchor);
    }
    /// Checks if the branch stack is empty.
    #[must_use]
    fn stack_empty(&self) -> bool {
        self.branch_stack.is_empty()
    }
    /// Removes and returns the specified ring open field entry if present.
    fn remove_ring_open(&mut self, ring_num: RingNum) -> Option<(usize, Option<BondDescriptor>)> {
        self.ring_open[usize::from(ring_num.get())].take()
    }
    /// Checks if the ring open field is currently empty.
    #[must_use]
    fn ring_open_empty(&self) -> bool {
        self.ring_open.iter().all(Option::is_none)
    }
    /// Inserts the given ring into the ring open field
    fn insert_ring(&mut self, ring_num: RingNum, pending: (usize, Option<BondDescriptor>)) {
        self.ring_open[usize::from(ring_num.get())] = Some(pending);
    }
    #[must_use]
    fn nodes(&self) -> &[Atom] {
        &self.atom_nodes
    }
    /// Consumes the parser state and returns the parsed SMILES graph.
    #[must_use]
    fn into_smiles(self) -> Smiles<AtomPolicy> {
        let number_of_nodes = self.atom_nodes.len();
        let parsed_stereo_neighbors = self
            .parsed_stereo_neighbors
            .into_iter()
            .map(|neighbors| {
                neighbors
                    .into_iter()
                    .map(|neighbor| {
                        match neighbor {
                            PendingStereoNeighbor::Atom(atom) => StereoNeighbor::Atom(atom),
                            PendingStereoNeighbor::ExplicitHydrogen => {
                                StereoNeighbor::ExplicitHydrogen
                            }
                            PendingStereoNeighbor::RingLabel(_) => {
                                unreachable!(
                                    "all ring labels must be resolved before parse completion"
                                )
                            }
                        }
                    })
                    .collect()
            })
            .collect();
        Smiles::from_bond_matrix_parts_with_parsed_stereo(
            self.atom_nodes,
            self.bond_matrix.finish(number_of_nodes),
            parsed_stereo_neighbors,
        )
    }
    /// Returns whether there is an edge for the given pair of nodes.
    #[must_use]
    fn edge_for_node_pair_exists(&self, nodes: (usize, usize)) -> bool {
        self.bond_matrix.contains_edge(nodes.0, nodes.1)
    }
    /// Pushes an [`Atom`] into the parsed [`Smiles`] graph.
    fn push_node(&mut self, node: Atom) {
        self.atom_nodes.push(node);
        self.parsed_stereo_neighbors.push(Vec::new());
    }

    #[inline]
    #[must_use]
    fn node_has_chirality(&self, node_id: usize) -> bool {
        self.atom_nodes.get(node_id).and_then(Atom::chirality).is_some()
    }

    fn append_stereo_neighbor(&mut self, node_id: usize, neighbor: PendingStereoNeighbor) {
        if self.node_has_chirality(node_id) {
            self.parsed_stereo_neighbors[node_id].push(neighbor);
        }
    }

    fn resolve_ring_label_neighbor(&mut self, node_id: usize, ring_num: RingNum, neighbor: usize) {
        if !self.node_has_chirality(node_id) {
            return;
        }
        let slot = self.parsed_stereo_neighbors[node_id]
            .iter_mut()
            .find(|entry| **entry == PendingStereoNeighbor::RingLabel(ring_num))
            .unwrap_or_else(|| unreachable!("ring opening placeholder must exist"));
        *slot = PendingStereoNeighbor::Atom(neighbor);
    }
    #[inline]
    fn push_edge_verified(
        &mut self,
        node_a: usize,
        node_b: usize,
        bond: BondDescriptor,
        ring_num: Option<RingNum>,
    ) -> Result<(), SmilesError> {
        self.bond_matrix.push_edge_with_descriptor(node_a, node_b, bond, ring_num)
    }
    /// Adds an atom to the SMILES graph, either bracketed or unbracketed.
    ///
    /// # Errors
    /// - Returns [`SmilesError::InvalidIsotope`] if a concrete isotope label is
    ///   not known by `elements-rs`.
    /// - Returns [`SmilesError::InvalidHydrogenWithExplicitHydrogensFound`] if
    ///   a bracketed hydrogen carries an explicit hydrogen count greater than
    ///   one.
    /// - Returns [`SmilesError::NodeIdInvalid`] if the new atom would be
    ///   connected to a nonexistent previous atom.
    /// - Propagates edge-construction errors for duplicate edges or self-loops.
    fn add_atom(
        &mut self,
        atom: Atom,
        start: usize,
        end: usize,
    ) -> Result<(), SmilesErrorWithSpan> {
        if !AtomPolicy::ALLOW_WILDCARDS && atom.symbol().is_wildcard() {
            return Err(SmilesErrorWithSpan::new(SmilesError::WildcardAtomNotAllowed, start, end));
        }
        validate_concrete_isotope(&atom, start, end)?;

        let id = self.atom_nodes.len();
        let previous_atom = self.last_atom();
        if matches!(atom.element(), Some(Element::H)) && atom.hydrogen_count() > 1 {
            return Err(SmilesErrorWithSpan::new(
                SmilesError::InvalidHydrogenWithExplicitHydrogensFound,
                start,
                end,
            ));
        }
        let mut stereo_neighbors = Vec::new();
        if atom.chirality().is_some() {
            if let Some(previous) = previous_atom {
                stereo_neighbors.push(PendingStereoNeighbor::Atom(previous));
            }
            if atom.hydrogen_count() == 1 {
                stereo_neighbors.push(PendingStereoNeighbor::ExplicitHydrogen);
            }
        }
        self.push_node(atom);
        *self.parsed_stereo_neighbors.last_mut().unwrap_or_else(|| unreachable!()) =
            stereo_neighbors;
        if let Some(prev) = previous_atom {
            let bond = self.pending_bond().unwrap_or_else(|| default_bond(self.nodes(), prev, id));
            self.push_edge_verified(prev, id, bond, None)
                .map_err(|e| SmilesErrorWithSpan::new(e, start, end))?;
            self.append_stereo_neighbor(prev, PendingStereoNeighbor::Atom(id));
        }
        self.update_last_atom(Some(id));
        self.update_pending_bond(None);
        Ok(())
    }
    /// Validates that at the current point in parsing there are no hanging
    /// bonds, branches, or ring closures.
    ///
    /// # Errors
    /// - Returns [`SmilesError::IncompleteBond`] if a bond is left open at the
    ///   non-bond.
    /// - Returns [`SmilesError::UnclosedBranch`] if a branch is left open at
    ///   the non-bond.
    /// - Returns [`SmilesError::UnclosedRing`] if a ring is left open at the
    ///   non-bond.
    fn validate_all_closed(&mut self) -> Result<(), SmilesErrorWithSpan> {
        let (start, end) = self.last_span;
        let start = start.min(end.saturating_sub(1));
        let end = end.max(start.saturating_add(1));

        if let Some(bond) = self.pending_bond {
            return Err(SmilesErrorWithSpan::new(SmilesError::IncompleteBond(bond), start, end));
        }
        if !self.stack_empty() {
            return Err(SmilesErrorWithSpan::new(SmilesError::UnclosedBranch, start, end));
        }
        if !self.ring_open_empty() {
            return Err(SmilesErrorWithSpan::new(SmilesError::UnclosedRing, start, end));
        }
        self.update_last_atom(None);
        self.update_pending_bond(None);
        Ok(())
    }

    /// Validates a component boundary introduced by `.`.
    ///
    /// Unlike [`Self::validate_all_closed`], this allows open ring labels to
    /// remain pending so disconnected ring-closure notation such as
    /// `C1.C1`-style forms can be resolved by later tokens.
    fn validate_component_boundary(&mut self) -> Result<(), SmilesErrorWithSpan> {
        let (start, end) = self.last_span;
        let start = start.min(end.saturating_sub(1));
        let end = end.max(start.saturating_add(1));

        if let Some(bond) = self.pending_bond {
            return Err(SmilesErrorWithSpan::new(SmilesError::IncompleteBond(bond), start, end));
        }
        if !self.stack_empty() {
            return Err(SmilesErrorWithSpan::new(SmilesError::UnclosedBranch, start, end));
        }
        self.update_last_atom(None);
        self.update_pending_bond(None);
        Ok(())
    }

    /// This method validates and adds the ring number to the relevant bond in
    /// the graph.
    ///
    /// # Errors
    /// - Returns [`SmilesError::InvalidRingNumber`] if a previous atom for the
    ///   bond is not found or a relevant edge between the vertices is not
    ///   found.
    /// - Returns [`SmilesError::NodeIdInvalid`] if a node cannot be found in
    ///   the edge list
    fn validate_and_add_ring_num(
        &mut self,
        start: usize,
        end: usize,
        ring_num: RingNum,
    ) -> Result<(), SmilesErrorWithSpan> {
        let Some(current) = self.last_atom() else {
            return Err(SmilesErrorWithSpan::new(SmilesError::InvalidRingNumber, start, end));
        };
        if let Some((other, stored_bond)) = self.remove_ring_open(ring_num) {
            if current == other {
                return Err(SmilesErrorWithSpan::new(SmilesError::InvalidRingNumber, start, end));
            }
            if self.edge_for_node_pair_exists((current, other)) {
                return Err(SmilesErrorWithSpan::new(SmilesError::InvalidRingNumber, start, end));
            }
            let bond = self
                .pending_bond()
                .or(stored_bond)
                .unwrap_or_else(|| default_bond(self.nodes(), current, other));

            self.push_edge_verified(current, other, bond, Some(ring_num))
                .map_err(|e| SmilesErrorWithSpan::new(e, start, end))?;
            self.append_stereo_neighbor(current, PendingStereoNeighbor::Atom(other));
            self.resolve_ring_label_neighbor(other, ring_num, current);

            self.update_pending_bond(None);
        } else {
            self.append_stereo_neighbor(current, PendingStereoNeighbor::RingLabel(ring_num));
            self.insert_ring(ring_num, (current, self.pending_bond()));
            self.update_pending_bond(None);
        }

        Ok(())
    }
    /// Evaluates that a left token parentheses has a valid anchor atom.
    ///
    /// # Errors
    /// - Returns [`SmilesError::UnexpectedLeftParentheses`] if a valid anchor
    ///   is not found to associate with the left parentheses.
    fn validate_branch_open(
        &mut self,
        start: usize,
        end: usize,
        next_token: Option<TokenKind>,
    ) -> Result<(), SmilesErrorWithSpan> {
        if let Some(token) = next_token {
            if token == TokenKind::LeftParentheses {
                return Err(SmilesErrorWithSpan::new(
                    SmilesError::UnexpectedLeftParentheses,
                    start,
                    end,
                ));
            } else if token == TokenKind::RightParentheses {
                return Err(SmilesErrorWithSpan::new(SmilesError::EmptyBranch, start, end));
            }
        }
        let Some(anchor) = self.last_atom() else {
            return Err(SmilesErrorWithSpan::new(
                SmilesError::UnexpectedLeftParentheses,
                start,
                end,
            ));
        };
        self.push_stack(anchor);
        Ok(())
    }
    /// Evaluates that a right token parentheses has a valid anchor atom.
    ///
    /// # Errors
    /// - Returns [`SmilesError::UnexpectedRightParentheses`] if a valid anchor
    ///   is not found to associate with the right parentheses.
    fn validate_branch_close(
        &mut self,
        start: usize,
        end: usize,
    ) -> Result<(), SmilesErrorWithSpan> {
        let Some(anchor) = self.pop_branch_stack() else {
            return Err(SmilesErrorWithSpan::new(
                SmilesError::UnexpectedRightParentheses,
                start,
                end,
            ));
        };
        if let Some(last_atom) = self.last_atom
            && last_atom == anchor
        {
            return Err(SmilesErrorWithSpan::new(SmilesError::InvalidBranch, start, end));
        }
        self.update_last_atom(Some(anchor));
        Ok(())
    }
    /// Checks that there is an existing atom before the current bond, then
    /// updates the pending bond field with the bond.
    ///
    /// # Errors
    /// - Returns [`SmilesError::IncompleteBond`] if a previous atom is not
    ///   found.
    /// - Returns [`SmilesError::InvalidBond`] if bond is not binding two valid
    ///   nodes
    fn validate_and_add_bond(
        &mut self,
        start: usize,
        end: usize,
        bond: BondDescriptor,
        next_token: Option<TokenKind>,
    ) -> Result<(), SmilesErrorWithSpan> {
        if self.last_atom().is_none() {
            return Err(SmilesErrorWithSpan::new(SmilesError::IncompleteBond(bond), start, end));
        }
        if let Some(token) = next_token
            && matches!(token, TokenKind::Bond | TokenKind::LeftParentheses)
        {
            return Err(SmilesErrorWithSpan::new(SmilesError::InvalidBond, start, end));
        }
        self.update_pending_bond(Some(bond));
        Ok(())
    }
    /// Validates that a [`Token::NonBond`] is preceded and proceeded by valid
    /// tokens
    ///
    /// # Errors
    /// - Returns [`SmilesError::InvalidNonBondToken`] if there isn't a valid
    ///   token before or after the non bond
    fn validate_non_bond(
        last_token: Option<TokenKind>,
        next_token: Option<TokenKind>,
        start: usize,
        end: usize,
    ) -> Result<(), SmilesErrorWithSpan> {
        if let Some(last) = last_token {
            match last {
                TokenKind::NonBond | TokenKind::Bond | TokenKind::LeftParentheses => {
                    return Err(SmilesErrorWithSpan::new(
                        SmilesError::InvalidNonBondToken,
                        start,
                        end,
                    ));
                }
                _ => {}
            }
        } else {
            return Err(SmilesErrorWithSpan::new(SmilesError::InvalidNonBondToken, start, end));
        }
        if let Some(next) = next_token {
            if next != TokenKind::Atom {
                return Err(SmilesErrorWithSpan::new(SmilesError::InvalidNonBondToken, start, end));
            }
        } else {
            return Err(SmilesErrorWithSpan::new(SmilesError::InvalidNonBondToken, start, end));
        }
        Ok(())
    }
}

fn validate_concrete_isotope(
    atom: &Atom,
    start: usize,
    end: usize,
) -> Result<(), SmilesErrorWithSpan> {
    let Some(mass_number) = atom.isotope_mass_number() else {
        return Ok(());
    };
    let Some(element) = atom.element() else {
        // Isotope-qualified wildcards appear in placeholder corpora such as
        // PubChem. They still cannot be converted to exact formulas, but they
        // do not name a concrete isotope to validate here.
        return Ok(());
    };

    Isotope::try_from((element, mass_number))
        .map(|_| ())
        .map_err(|_| SmilesErrorWithSpan::new(SmilesError::InvalidIsotope, start, end))
}

#[cfg(test)]
impl ParserState {
    #[must_use]
    fn new(input_len: usize) -> Self {
        Self::new_for_policy(input_len)
    }
}

#[inline]
fn default_bond(nodes: &[Atom], id_a: usize, id_b: usize) -> BondDescriptor {
    let node_a = &nodes[id_a];
    let node_b = &nodes[id_b];
    if node_a.aromatic() && node_b.aromatic() {
        BondDescriptor::aromatic(Bond::Single)
    } else {
        Bond::Single.into()
    }
}

#[cfg(test)]
mod tests {

    use elements_rs::Element;

    use crate::{
        Smiles, SmilesError,
        atom::{Atom, atom_symbol::AtomSymbol},
        bond::{Bond, BondDescriptor, ring_num::RingNum},
        parser::smiles_parser::{ParserState, default_bond},
        token::TokenKind,
    };

    fn atom(element: Element, aromatic: bool) -> Atom {
        Atom::new_organic_subset(AtomSymbol::Element(element), aromatic)
    }

    #[test]
    fn parser_state_new_is_empty() {
        let state = ParserState::new(0);
        assert_eq!(state.last_span, (0, 0));
        assert_eq!(state.last_atom(), None);
        assert_eq!(state.pending_bond(), None);
        assert!(state.branch_stack.is_empty());
        assert!(state.stack_empty());
        assert!(state.ring_open_empty());
        assert!(state.nodes().is_empty());
        let smiles = state.into_smiles();
        assert_eq!(smiles.number_of_bonds(), 0);
    }

    #[test]
    fn parser_state_span_updates_work() {
        let mut state = ParserState::new(0);

        state.update_last_span((3, 7));
        assert_eq!(state.last_span, (3, 7));
    }

    #[test]
    fn parser_state_last_atom_and_pending_bond_updates_work() {
        let mut state = ParserState::new(0);

        state.update_last_atom(Some(4));
        assert_eq!(state.last_atom(), Some(4));
        state.update_last_atom(None);
        assert_eq!(state.last_atom(), None);

        state.update_pending_bond(Some(Bond::Triple.into()));
        assert_eq!(state.pending_bond(), Some(Bond::Triple.into()));
        state.update_pending_bond(None);
        assert_eq!(state.pending_bond(), None);
    }

    #[test]
    fn parser_state_branch_stack_methods_work() {
        let mut state = ParserState::new(0);

        assert!(state.stack_empty());
        assert!(state.branch_stack.is_empty());

        state.push_stack(1);
        state.push_stack(3);

        assert!(!state.stack_empty());
        assert_eq!(state.branch_stack, [1, 3]);
        assert_eq!(state.pop_branch_stack(), Some(3));
        assert_eq!(state.pop_branch_stack(), Some(1));
        assert_eq!(state.pop_branch_stack(), None);
        assert!(state.stack_empty());
    }

    #[test]
    fn parser_state_ring_open_methods_work() {
        let mut state = ParserState::new(0);
        let ring = RingNum::try_new(7).unwrap();

        assert!(state.ring_open_empty());
        assert_eq!(state.remove_ring_open(ring), None);

        state.insert_ring(ring, (9, Some(Bond::Double.into())));
        assert!(!state.ring_open_empty());
        assert_eq!(state.remove_ring_open(ring), Some((9, Some(Bond::Double.into()))));
        assert!(state.ring_open_empty());
    }

    #[test]
    fn parser_state_push_node_appends_in_order() {
        let mut state = ParserState::new(0);

        state.push_node(atom(Element::C, false));
        state.push_node(atom(Element::O, false));

        assert_eq!(state.nodes().len(), 2);
    }

    #[test]
    fn parser_state_into_smiles_returns_owned_graph() {
        let mut state = ParserState::new(0);
        state.push_node(atom(Element::C, false));
        state.push_node(atom(Element::O, false));
        state.push_edge_verified(0, 1, Bond::Single.into(), None).unwrap();

        let smiles = state.into_smiles();
        assert_eq!(smiles.nodes().len(), 2);
        assert_eq!(smiles.number_of_bonds(), 1);
    }

    #[test]
    fn parser_state_add_atom_adds_first_atom() {
        let mut state = ParserState::new(0);

        state
            .add_atom(Atom::new_organic_subset(AtomSymbol::Element(Element::C), false), 0, 1)
            .unwrap();

        assert_eq!(state.last_atom(), Some(0));
        assert_eq!(state.pending_bond(), None);
        assert_eq!(state.nodes().len(), 1);
        let smiles = state.into_smiles();
        assert_eq!(smiles.number_of_bonds(), 0);
    }

    #[test]
    fn parser_state_add_atom_connects_with_pending_bond() {
        let mut state = ParserState::new(0);

        state
            .add_atom(Atom::new_organic_subset(AtomSymbol::Element(Element::C), false), 0, 1)
            .unwrap();
        state.update_pending_bond(Some(Bond::Triple.into()));
        state
            .add_atom(Atom::new_organic_subset(AtomSymbol::Element(Element::N), false), 1, 2)
            .unwrap();

        assert_eq!(state.last_atom(), Some(1));
        assert_eq!(state.pending_bond(), None);
        let smiles = state.into_smiles();
        assert_eq!(smiles.nodes().len(), 2);
        assert_eq!(smiles.number_of_bonds(), 1);
        assert_eq!(smiles.edge_for_node_pair((0, 1)).unwrap().bond(), Bond::Triple);
    }

    #[test]
    fn parser_state_validate_all_closed_ok_resets_last_atom_and_pending_bond() {
        let mut state = ParserState::new(0);
        state.update_last_span((4, 5));
        state.update_last_atom(Some(9));
        state.update_pending_bond(None);

        state.validate_all_closed().unwrap();

        assert_eq!(state.last_atom(), None);
        assert_eq!(state.pending_bond(), None);
    }

    #[test]
    fn parser_state_validate_all_closed_errors_for_incomplete_bond() {
        let mut state = ParserState::new(0);
        state.update_last_span((2, 3));
        state.update_pending_bond(Some(Bond::Double.into()));

        let err = state.validate_all_closed().expect_err("expected incomplete bond");

        assert_eq!(err.smiles_error(), SmilesError::IncompleteBond(Bond::Double.into()));
        assert_eq!(err.start(), 2);
        assert_eq!(err.end(), 3);
    }

    #[test]
    fn parser_state_validate_all_closed_errors_for_unclosed_branch() {
        let mut state = ParserState::new(0);
        state.update_last_span((2, 3));
        state.push_stack(0);

        let err = state.validate_all_closed().expect_err("expected unclosed branch");

        assert_eq!(err.smiles_error(), SmilesError::UnclosedBranch);
        assert_eq!(err.start(), 2);
        assert_eq!(err.end(), 3);
    }

    #[test]
    fn parser_state_validate_all_closed_errors_for_unclosed_ring() {
        let mut state = ParserState::new(0);
        state.update_last_span((2, 3));
        state.insert_ring(RingNum::try_new(1).unwrap(), (0, None));

        let err = state.validate_all_closed().expect_err("expected unclosed ring");

        assert_eq!(err.smiles_error(), SmilesError::UnclosedRing);
        assert_eq!(err.start(), 2);
        assert_eq!(err.end(), 3);
    }

    #[test]
    fn parser_state_validate_component_boundary_allows_open_ring_labels() {
        let mut state = ParserState::new(0);
        state.update_last_span((2, 3));
        state.insert_ring(RingNum::try_new(1).unwrap(), (0, None));
        state.update_last_atom(Some(0));

        state.validate_component_boundary().unwrap();

        assert_eq!(state.last_atom(), None);
        assert_eq!(state.pending_bond(), None);
        assert!(!state.ring_open_empty());
    }

    #[test]
    fn parser_state_validate_component_boundary_rejects_incomplete_bond() {
        let mut state = ParserState::new(0);
        state.update_last_span((2, 3));
        state.update_pending_bond(Some(Bond::Double.into()));

        let err = state.validate_component_boundary().expect_err("expected incomplete bond");

        assert_eq!(err.smiles_error(), SmilesError::IncompleteBond(Bond::Double.into()));
        assert_eq!(err.start(), 2);
        assert_eq!(err.end(), 3);
    }

    #[test]
    fn parser_state_validate_component_boundary_rejects_open_branch() {
        let mut state = ParserState::new(0);
        state.update_last_span((2, 3));
        state.push_stack(0);

        let err = state.validate_component_boundary().expect_err("expected unclosed branch");

        assert_eq!(err.smiles_error(), SmilesError::UnclosedBranch);
        assert_eq!(err.start(), 2);
        assert_eq!(err.end(), 3);
    }

    #[test]
    fn parser_state_validate_branch_open_errors_without_anchor() {
        let mut state = ParserState::new(0);

        let err = state.validate_branch_open(3, 4, None).expect_err("expected missing anchor");

        assert_eq!(err.smiles_error(), SmilesError::UnexpectedLeftParentheses);
        assert_eq!(err.start(), 3);
        assert_eq!(err.end(), 4);
    }

    #[test]
    fn parser_state_validate_branch_close_errors_without_open_branch() {
        let mut state = ParserState::new(0);

        let err = state.validate_branch_close(4, 5).expect_err("expected missing branch anchor");

        assert_eq!(err.smiles_error(), SmilesError::UnexpectedRightParentheses);
        assert_eq!(err.start(), 4);
        assert_eq!(err.end(), 5);
    }

    #[test]
    fn validate_non_bond_rejects_invalid_left_context() {
        let err = ParserState::<crate::smiles::ConcreteAtoms>::validate_non_bond(
            Some(TokenKind::Bond),
            Some(TokenKind::Atom),
            4,
            5,
        )
        .expect_err("expected invalid non-bond token");
        assert_eq!(err.smiles_error(), SmilesError::InvalidNonBondToken);
        assert_eq!(err.start(), 4);
        assert_eq!(err.end(), 5);
    }

    #[test]
    fn parser_state_validate_and_add_bond_sets_pending_bond() {
        let mut state = ParserState::new(0);
        state.update_last_atom(Some(0));

        state.validate_and_add_bond(1, 2, BondDescriptor::aromatic(Bond::Single), None).unwrap();

        assert_eq!(state.pending_bond(), Some(BondDescriptor::aromatic(Bond::Single)));
    }

    #[test]
    fn parser_state_validate_and_add_bond_errors_without_left_atom() {
        let mut state = ParserState::new(0);

        let err = state
            .validate_and_add_bond(1, 2, Bond::Single.into(), None)
            .expect_err("expected incomplete bond");

        assert_eq!(err.smiles_error(), SmilesError::IncompleteBond(Bond::Single.into()));
        assert_eq!(err.start(), 1);
        assert_eq!(err.end(), 2);
    }

    #[test]
    fn parser_state_validate_and_add_ring_num_opens_ring_and_clears_pending_bond() {
        let mut state = ParserState::new(0);
        let ring = RingNum::try_new(6).unwrap();

        state.push_node(atom(Element::C, false));
        state.update_last_atom(Some(0));
        state.update_pending_bond(Some(Bond::Double.into()));

        state.validate_and_add_ring_num(1, 2, ring).unwrap();

        assert_eq!(state.pending_bond(), None);
        assert_eq!(state.remove_ring_open(ring), Some((0, Some(Bond::Double.into()))));
    }

    #[test]
    fn parser_state_validate_and_add_ring_num_closes_ring_and_adds_edge() {
        let mut state = ParserState::new(0);
        let ring = RingNum::try_new(4).unwrap();

        state.push_node(atom(Element::C, false));
        state.push_node(atom(Element::N, false));
        state.insert_ring(ring, (0, Some(Bond::Triple.into())));
        state.update_last_atom(Some(1));

        state.validate_and_add_ring_num(2, 3, ring).unwrap();

        assert_eq!(state.pending_bond(), None);
        assert!(state.edge_for_node_pair_exists((0, 1)));
        let smiles = state.into_smiles();
        let edge = smiles.edge_for_node_pair((0, 1)).unwrap();
        assert_eq!(smiles.number_of_bonds(), 1);
        assert_eq!(edge.bond(), Bond::Triple);
        assert_eq!(crate::bond::bond_edge::bond_edge_ring_num_val(edge), Some(4));
    }

    #[test]
    fn parser_state_validate_and_add_ring_num_prefers_current_pending_bond() {
        let mut state = ParserState::new(0);
        let ring = RingNum::try_new(5).unwrap();

        state.push_node(atom(Element::C, false));
        state.push_node(atom(Element::O, false));
        state.insert_ring(ring, (0, Some(Bond::Double.into())));
        state.update_last_atom(Some(1));
        state.update_pending_bond(Some(Bond::Quadruple.into()));

        state.validate_and_add_ring_num(2, 3, ring).unwrap();

        let smiles = state.into_smiles();
        assert_eq!(smiles.edge_for_node_pair((0, 1)).unwrap().bond(), Bond::Quadruple);
    }

    #[test]
    fn parser_state_validate_and_add_ring_num_errors_without_current_atom() {
        let mut state = ParserState::new(0);
        let ring = RingNum::try_new(1).unwrap();

        let err =
            state.validate_and_add_ring_num(7, 8, ring).expect_err("expected invalid ring number");

        assert_eq!(err.smiles_error(), SmilesError::InvalidRingNumber);
        assert_eq!(err.start(), 7);
        assert_eq!(err.end(), 8);
    }

    #[test]
    fn parser_state_validate_and_add_ring_num_errors_for_self_loop() {
        let mut state = ParserState::new(0);
        let ring = RingNum::try_new(2).unwrap();

        state.push_node(atom(Element::C, false));
        state.insert_ring(ring, (0, None));
        state.update_last_atom(Some(0));

        let err = state
            .validate_and_add_ring_num(3, 4, ring)
            .expect_err("expected invalid self-loop ring");

        assert_eq!(err.smiles_error(), SmilesError::InvalidRingNumber);
        assert_eq!(err.start(), 3);
        assert_eq!(err.end(), 4);
    }

    #[test]
    fn parser_state_validate_and_add_ring_num_errors_for_duplicate_edge() {
        let mut state = ParserState::new(0);
        let ring = RingNum::try_new(3).unwrap();

        state.push_node(atom(Element::C, false));
        state.push_node(atom(Element::O, false));
        state.push_edge_verified(0, 1, Bond::Single.into(), None).unwrap();
        state.insert_ring(ring, (0, None));
        state.update_last_atom(Some(1));

        let err = state
            .validate_and_add_ring_num(4, 5, ring)
            .expect_err("expected duplicate edge ring failure");

        assert_eq!(err.smiles_error(), SmilesError::InvalidRingNumber);
        assert_eq!(err.start(), 4);
        assert_eq!(err.end(), 5);
    }

    #[test]
    fn parse_smiles_allows_disconnected_ring_closure_stereo_forms() {
        let left = Smiles::from_str("[C@@]1(Cl)(F)(I).Br1").unwrap();
        let right = Smiles::from_str("[C@@](Cl)(F)(I)1.Br1").unwrap();

        assert_eq!(left.nodes().len(), 5);
        assert_eq!(right.nodes().len(), 5);
        assert_eq!(left.number_of_bonds(), 4);
        assert_eq!(right.number_of_bonds(), 4);
    }

    #[test]
    fn default_bond_is_aromatic_for_two_aromatic_atoms() {
        let nodes = vec![atom(Element::C, true), atom(Element::N, true)];

        assert_eq!(default_bond(&nodes, 0, 1), BondDescriptor::aromatic(Bond::Single));
    }

    #[test]
    fn default_bond_is_single_when_atoms_are_not_both_aromatic() {
        let nodes = vec![atom(Element::C, true), atom(Element::O, false)];

        assert_eq!(default_bond(&nodes, 0, 1), Bond::Single.into());
    }
}
