//! Maximum common edge subgraph (MCES) between two [`Smiles`] molecules.
//!
//! The heavy lifting lives in the `geometric-traits` crate. This module adds a
//! smiles-flavored entry point so callers can write `a.mces(&b)` for the common
//! case and reach for [`Smiles::mces_with`] when they need to tune the search.
//!
//! MCES is NP-hard. The default [`Smiles::mces`] runs an unbounded search,
//! which is fine for small molecules but can be expensive for large ones. Use
//! [`SmilesMces::search_budget`] to cap the work, then read
//! [`McesResult::search_completed`] to learn whether the returned match is a
//! proven maximum or only a lower bound.
//!
//! Matching is labeled: two bonds are compatible only when their endpoint atom
//! types agree and their [`BondEntry`](crate::smiles::BondEntry) values compare
//! equal, where aromatic bonds ignore their kekule order.
//!
//! # Examples
//!
//! ```
//! use smiles_rs::prelude::{GraphSimilarities, Smiles};
//!
//! let benzene: Smiles = "c1ccccc1".parse()?;
//! let pyridine: Smiles = "c1ccncc1".parse()?;
//!
//! let result = benzene.mces(&pyridine);
//! assert_eq!(result.matched_edges().len(), 4);
//! assert!(result.johnson_similarity() < 1.0);
//! # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
//! ```
pub use geometric_traits::traits::{
    GraphSimilarities, InitialProductVertexOrdering, LargestFragmentMetric, McesBuilder,
    McesResult, McesSearchMode,
};

use crate::smiles::Smiles;

impl Smiles {
    /// Computes the maximum common edge subgraph against `other` with default
    /// settings (a labeled, unbounded search).
    ///
    /// This is the convenient one-shot form of [`Smiles::mces_with`]. For large
    /// molecules prefer the builder so a [`search
    /// budget`](SmilesMces::search_budget) can bound the NP-hard search.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let a: Smiles = "CCO".parse()?;
    /// let b: Smiles = "CCN".parse()?;
    /// assert_eq!(a.mces(&b).matched_edges().len(), 1);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn mces(&self, other: &Self) -> McesResult<usize> {
        self.mces_with(other).compute()
    }

    /// Starts a configurable MCES query against `other`.
    ///
    /// The returned [`SmilesMces`] exposes the tuning knobs of the underlying
    /// search. Call [`SmilesMces::compute`] to run it.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let a: Smiles = "c1ccccc1".parse()?;
    /// let b: Smiles = "c1ccccc1C".parse()?;
    /// let result = a.mces_with(&b).search_budget(50_000).compute();
    /// assert!(result.search_completed());
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn mces_with<'g>(&'g self, other: &'g Self) -> SmilesMces<'g> {
        SmilesMces::new(self, other)
    }
}

/// A configurable maximum common edge subgraph query between two [`Smiles`]
/// molecules.
///
/// Created by [`Smiles::mces_with`]. Each setter returns the builder so calls
/// can be chained, and [`SmilesMces::compute`] runs the labeled search.
///
/// This surfaces the plain-value tuning knobs of the underlying search. The
/// advanced, type-changing hooks (custom pair filters, edge comparators,
/// disambiguation closures, rankers, and precomputed edge contexts) are left to
/// `geometric_traits::traits::McesBuilder`, which callers can drive directly
/// since [`Smiles`] implements the required graph traits.
#[derive(Clone, Debug)]
pub struct SmilesMces<'g> {
    first: &'g Smiles,
    second: &'g Smiles,
    largest_fragment_metric: LargestFragmentMetric,
    product_vertex_ordering: InitialProductVertexOrdering,
    partition: bool,
    search_mode: McesSearchMode,
    delta_y: bool,
    ignore_bond_orders: bool,
    similarity_threshold: Option<f64>,
    distance_threshold: Option<f64>,
    search_budget: Option<usize>,
}

impl<'g> SmilesMces<'g> {
    /// Builds a query with the same defaults as the underlying search.
    #[inline]
    #[must_use]
    fn new(first: &'g Smiles, second: &'g Smiles) -> Self {
        Self {
            first,
            second,
            largest_fragment_metric: LargestFragmentMetric::Edges,
            product_vertex_ordering: InitialProductVertexOrdering::None,
            partition: true,
            search_mode: McesSearchMode::PartialEnumeration,
            delta_y: true,
            ignore_bond_orders: false,
            similarity_threshold: None,
            distance_threshold: None,
            search_budget: None,
        }
    }

    /// Chooses the fragment-size metric used to break ties between cliques of
    /// equal matched-edge count.
    ///
    /// The default is [`LargestFragmentMetric::Edges`]. RDKit-style comparisons
    /// often prefer [`LargestFragmentMetric::Atoms`] because RDKit's
    /// `LargestFragSize` is atom-based.
    #[inline]
    #[must_use]
    pub fn largest_fragment_metric(mut self, metric: LargestFragmentMetric) -> Self {
        self.largest_fragment_metric = metric;
        self
    }

    /// Selects how modular-product vertices are ordered before clique search.
    ///
    /// Ordering does not change the optimal matched-edge count, but it changes
    /// how quickly the branch-and-bound search reaches and proves that optimum,
    /// so it interacts with [`search_budget`](Self::search_budget). The default
    /// is [`InitialProductVertexOrdering::None`] (construction order). The
    /// other policies (`EdgeSignature`, `LineGraphWL`, `Degree`,
    /// `PageRank`) order vertices by source-edge signatures, line-graph
    /// Weisfeiler-Lehman colors, product degree, or product PageRank.
    #[inline]
    #[must_use]
    pub fn product_vertex_ordering(mut self, ordering: InitialProductVertexOrdering) -> Self {
        self.product_vertex_ordering = ordering;
        self
    }

    /// Enables or disables partition-aware maximum clique search.
    ///
    /// Defaults to enabled, which mirrors RDKit's partitioned behavior and is
    /// generally faster. Disabling it falls back to the legacy single accepted
    /// maximum path.
    #[inline]
    #[must_use]
    pub fn partition(mut self, enabled: bool) -> Self {
        self.partition = enabled;
        self
    }

    /// Selects how the clique stage explores tied maximum solutions.
    ///
    /// The default [`McesSearchMode::PartialEnumeration`] keeps strict pruning
    /// and retains the equal-size maxima it accepts during the search.
    /// [`McesSearchMode::AllBest`] enumerates every tied maximum clique before
    /// ranking, which is only needed when a custom ranker must choose among all
    /// of them.
    #[inline]
    #[must_use]
    pub fn search_mode(mut self, search_mode: McesSearchMode) -> Self {
        self.search_mode = search_mode;
        self
    }

    /// Enables or disables Delta-Y exchange filtering.
    ///
    /// Defaults to enabled. When enabled, cliques whose matched edge subgraphs
    /// have different sorted degree sequences in the two molecules are
    /// discarded, which catches the Whitney triangle versus claw exception.
    #[inline]
    #[must_use]
    pub fn delta_y(mut self, enabled: bool) -> Self {
        self.delta_y = enabled;
        self
    }

    /// Ignores bond orders when matching, mirroring RDKit's `ignoreBondOrders`.
    ///
    /// Defaults to disabled. When enabled, two bonds match on the endpoint atom
    /// types alone and the stored bond value is ignored. Note that aromatic
    /// bonds already ignore their kekule order in the default matching, so this
    /// setting mainly collapses the single versus double versus triple
    /// distinction among non-aromatic bonds.
    #[inline]
    #[must_use]
    pub fn ignore_bond_orders(mut self, enabled: bool) -> Self {
        self.ignore_bond_orders = enabled;
        self
    }

    /// Sets a minimum similarity threshold for cheap pre-screening
    /// (RASCAL-style).
    ///
    /// Before the expensive pipeline runs, an upper bound on similarity is
    /// computed from degree sequences. If it is below `threshold`, the search
    /// is skipped and an empty result is returned. Typical values are 0.5
    /// to 0.7.
    #[inline]
    #[must_use]
    pub fn similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = Some(threshold);
        self
    }

    /// Sets a maximum distance threshold for cheap pre-screening
    /// (myopic-style).
    ///
    /// Before the expensive pipeline runs, a lower bound on edit distance is
    /// computed from degree sequences. If it exceeds `threshold`, the search is
    /// skipped and an empty result is returned.
    #[inline]
    #[must_use]
    pub fn distance_threshold(mut self, threshold: f64) -> Self {
        self.distance_threshold = Some(threshold);
        self
    }

    /// Caps the maximum-clique stage at `max_nodes` branch-and-bound nodes.
    ///
    /// The cap is the main lever against MCES NP-hardness. The node count is
    /// deterministic, so a given input and budget always stop at the same
    /// point. When the cap is hit the search returns the best clique found
    /// so far, a valid lower bound, and [`McesResult::search_completed`]
    /// reports `false`. The default is unbounded.
    #[inline]
    #[must_use]
    pub fn search_budget(mut self, max_nodes: usize) -> Self {
        self.search_budget = Some(max_nodes);
        self
    }

    /// Runs the configured labeled MCES search.
    #[must_use]
    pub fn compute(self) -> McesResult<usize> {
        let mut builder = McesBuilder::new(self.first, self.second)
            .with_largest_fragment_metric(self.largest_fragment_metric)
            .with_initial_product_vertex_ordering(self.product_vertex_ordering)
            .with_partition(self.partition)
            .with_search_mode(self.search_mode)
            .with_delta_y(self.delta_y)
            .with_ignore_edge_values(self.ignore_bond_orders);
        if let Some(threshold) = self.similarity_threshold {
            builder = builder.with_similarity_threshold(threshold);
        }
        if let Some(threshold) = self.distance_threshold {
            builder = builder.with_distance_threshold(threshold);
        }
        if let Some(max_nodes) = self.search_budget {
            builder = builder.with_search_budget(max_nodes);
        }
        builder.compute_labeled()
    }
}

#[cfg(test)]
mod tests {
    use geometric_traits::traits::GraphSimilarities;

    use super::{InitialProductVertexOrdering, LargestFragmentMetric, McesSearchMode};
    use crate::smiles::Smiles;

    fn smiles(input: &str) -> Smiles {
        input.parse().unwrap()
    }

    #[test]
    fn mces_of_identical_molecules_is_a_full_match() {
        let benzene = smiles("c1ccccc1");
        let result = benzene.mces(&smiles("c1ccccc1"));

        assert_eq!(result.matched_edges().len(), 6);
        assert!((result.johnson_similarity() - 1.0).abs() < 1.0e-9);
        assert!(result.search_completed());
    }

    #[test]
    fn mces_distinguishes_atom_identity() {
        let ethanol = smiles("CCO");
        let ethylamine = smiles("CCN");

        let result = ethanol.mces(&ethylamine);

        assert_eq!(result.matched_edges().len(), 1);
        assert!(result.johnson_similarity() < 1.0);
    }

    #[test]
    fn ignore_bond_orders_collapses_non_aromatic_kekule_distinction() {
        // A carbon-carbon single bond versus a carbon-carbon double bond.
        let single = smiles("CC");
        let double = smiles("C=C");

        assert_eq!(single.mces(&double).matched_edges().len(), 0);
        assert_eq!(
            single.mces_with(&double).ignore_bond_orders(true).compute().matched_edges().len(),
            1
        );
    }

    #[test]
    fn zero_search_budget_returns_an_incomplete_lower_bound() {
        let a = smiles("c1ccccc1C(=O)O");
        let b = smiles("c1ccccc1CC(=O)O");

        let result = a.mces_with(&b).search_budget(0).compute();

        assert!(!result.search_completed());
    }

    #[test]
    fn similarity_threshold_can_short_circuit_dissimilar_molecules() {
        let a = smiles("CCCCCCCCCC");
        let b = smiles("c1ccccc1");

        let result = a.mces_with(&b).similarity_threshold(0.9).compute();

        assert_eq!(result.matched_edges().len(), 0);
        assert_eq!(result.search_nodes(), 0);
    }

    #[test]
    fn distance_threshold_short_circuits_when_edit_distance_bound_is_exceeded() {
        // Hexane and pentane share a four-edge carbon path, so an unbounded
        // search matches four bonds. The degree-sequence edit-distance lower
        // bound for the pair sits between 0.5 and 1.0.
        let hexane = smiles("CCCCCC");
        let pentane = smiles("CCCCC");

        // A threshold below the bound skips the search entirely.
        let skipped = hexane.mces_with(&pentane).distance_threshold(0.0).compute();
        assert_eq!(skipped.matched_edges().len(), 0);
        assert_eq!(skipped.search_nodes(), 0);

        // A threshold above the bound runs and reaches the full four-bond
        // match.
        let run = hexane.mces_with(&pentane).distance_threshold(100.0).compute();
        assert_eq!(run.matched_edges().len(), 4);
        assert!(run.search_nodes() > 0);
        assert_eq!(hexane.mces(&pentane).matched_edges().len(), 4);
    }

    #[test]
    fn delta_y_filters_the_whitney_triangle_versus_claw_false_positive() {
        // Cyclopropane is a triangle (K3) and isobutane is a claw (K1,3). Their
        // line graphs are both triangles, so without Delta-Y filtering the
        // search reports a spurious three-bond match. Delta-Y rejects it
        // because the matched edge subgraphs have different degree
        // sequences.
        let cyclopropane = smiles("C1CC1");
        let isobutane = smiles("CC(C)C");

        let filtered = cyclopropane.mces_with(&isobutane).delta_y(true).compute();
        let unfiltered = cyclopropane.mces_with(&isobutane).delta_y(false).compute();

        assert_eq!(filtered.matched_edges().len(), 2);
        assert_eq!(unfiltered.matched_edges().len(), 3);
    }

    #[test]
    fn all_best_search_mode_enumerates_more_tied_cliques() {
        // Benzene against itself has many symmetry-equivalent maximum matches.
        // PartialEnumeration retains a single best clique, while AllBest
        // enumerates every tied maximum. Both agree on the matched-edge count.
        let benzene = smiles("c1ccccc1");

        let partial = benzene.mces_with(&smiles("c1ccccc1")).compute();
        let all_best =
            benzene.mces_with(&smiles("c1ccccc1")).search_mode(McesSearchMode::AllBest).compute();

        assert_eq!(partial.matched_edges().len(), 6);
        assert_eq!(all_best.matched_edges().len(), 6);
        assert_eq!(partial.all_cliques().len(), 1);
        assert!(all_best.all_cliques().len() > partial.all_cliques().len());
    }

    #[test]
    fn search_path_tuning_knobs_preserve_the_optimum() {
        // Partitioning, the fragment-size tie-break metric, and the product
        // vertex ordering steer how the search runs but must not change the
        // optimal matched-edge count.
        let acid = smiles("c1ccccc1C(=O)O");
        let amide = smiles("c1ccccc1CC(=O)N");
        let optimum = acid.mces(&amide).matched_edges().len();
        assert_eq!(optimum, 7);

        assert_eq!(
            acid.mces_with(&amide).partition(false).compute().matched_edges().len(),
            optimum
        );
        assert_eq!(
            acid.mces_with(&amide)
                .largest_fragment_metric(LargestFragmentMetric::Atoms)
                .compute()
                .matched_edges()
                .len(),
            optimum
        );

        for ordering in [
            InitialProductVertexOrdering::None,
            InitialProductVertexOrdering::EdgeSignature,
            InitialProductVertexOrdering::LineGraphWL,
            InitialProductVertexOrdering::Degree,
            InitialProductVertexOrdering::PageRank,
        ] {
            let matched = acid
                .mces_with(&amide)
                .product_vertex_ordering(ordering)
                .compute()
                .matched_edges()
                .len();
            assert_eq!(matched, optimum, "ordering {ordering:?} changed the optimum");
        }
    }
}
