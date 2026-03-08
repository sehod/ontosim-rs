use crate::node::Node;

// ---------------------------------------------------------------------------
// NodeMapping
// ---------------------------------------------------------------------------

/// A single node-to-node pairing produced by the similarity algorithm.
///
/// Each mapping pairs one node from the left-hand tree with one node from
/// the right-hand tree, along with their pairwise similarity score.
///
/// Use [`Tree::subtree`](crate::Tree::subtree) with the stored indices to
/// look up the corresponding [`Node`] and its label:
///
/// ```
/// # use ontosim::{Tree, similarity};
/// # use ontosim::matching::ExactMatching;
/// let t1: Tree = "{A{B}}".parse().unwrap();
/// let t2: Tree = "{A{B}}".parse().unwrap();
/// let result = similarity::compute(&t1, &t2, &ExactMatching);
///
/// for m in &result.mappings {
///     let lhs_label = m.lhs.map(|i| t1.subtree(i).label.as_str());
///     let rhs_label = m.rhs.map(|j| t2.subtree(j).label.as_str());
///     println!("{:?} <-> {:?} (score: {:.2})", lhs_label, rhs_label, m.sim);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct NodeMapping {
    /// 0-based postorder index into the left-hand tree's arena, or `None` if
    /// this side was unmatched during the bipartite assignment step.
    pub lhs: Option<usize>,
    /// 0-based postorder index into the right-hand tree's arena, or `None` if
    /// this side was unmatched during the bipartite assignment step.
    pub rhs: Option<usize>,
    /// Pairwise similarity score for this node pairing, as determined by the
    /// [`Matching`](crate::matching::Matching) strategy.
    pub sim: f64,
}

// ---------------------------------------------------------------------------
// SimilarityResult
// ---------------------------------------------------------------------------

/// The output of a similarity comparison between two trees (or subtrees).
///
/// Contains the aggregate similarity score and the individual node-level
/// [`NodeMapping`]s that produced it.
///
/// When returned from [`similarity::compute`](crate::similarity::compute),
/// the [`mappings`](Self::mappings) are sorted in **descending order** by
/// score, so the strongest pairings appear first.
///
/// Implements [`Ord`] by [`sim`](Self::sim) so that `max()` naturally
/// selects the best result from a set of candidates.
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// The aggregate similarity score — the sum of all individual mapping
    /// scores. This is **not** normalized to `[0.0, 1.0]`; its magnitude
    /// depends on how many nodes were successfully paired.
    pub sim: f64,
    /// The node-level pairings that contribute to [`sim`](Self::sim).
    ///
    /// Sorted in descending order by score when returned from
    /// [`similarity::compute`](crate::similarity::compute).
    pub mappings: Vec<NodeMapping>,
}

impl SimilarityResult {
    /// Returns a result with score `0.0` and no mappings.
    pub fn zero() -> Self {
        Self {
            sim: 0.0,
            mappings: Vec::new(),
        }
    }

    /// Combines two results by summing their scores and concatenating mappings.
    pub fn plus(&self, other: &SimilarityResult) -> SimilarityResult {
        SimilarityResult {
            sim: self.sim + other.sim,
            mappings: self
                .mappings
                .iter()
                .chain(other.mappings.iter())
                .cloned()
                .collect(),
        }
    }
}

impl PartialEq for SimilarityResult {
    fn eq(&self, other: &Self) -> bool {
        self.sim == other.sim
    }
}

impl Eq for SimilarityResult {}

impl PartialOrd for SimilarityResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SimilarityResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sim.total_cmp(&other.sim)
    }
}

// ---------------------------------------------------------------------------
// SimilarityCache
// ---------------------------------------------------------------------------

/// Dynamic-programming cache for subtree and subforest similarity scores and
/// mappings.
///
/// Stores four flat matrices (subforest scores, subforest mappings, subtree
/// scores, subtree mappings) indexed by `(i, j)` where `i` is a postorder
/// index in T1 and `j` is a postorder index in T2.
pub struct SimilarityCache {
    m: usize,
    n: usize,
    /// Subforest similarity scores   — S_F in the paper.
    sf: Vec<f64>,
    /// Subforest mappings            — M_F in the paper.
    mf: Vec<Vec<NodeMapping>>,
    /// Subtree similarity scores     — S_T in the paper.
    st: Vec<f64>,
    /// Subtree mappings              — M_T in the paper.
    mt: Vec<Vec<NodeMapping>>,
}

impl SimilarityCache {
    /// Creates a new cache for trees of size `m` and `n`.
    /// All scores default to 0.0 and all mapping lists default to empty.
    pub fn new(m: usize, n: usize) -> Self {
        let len = m * n;
        Self {
            m,
            n,
            sf: vec![0.0; len],
            mf: vec![Vec::new(); len],
            st: vec![0.0; len],
            mt: vec![Vec::new(); len],
        }
    }

    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.m && j < self.n);
        i * self.n + j
    }

    /// Resolves a pair of nodes to their 0-based cache coordinates.
    #[inline]
    fn node_idx(&self, lhs: &Node, rhs: &Node) -> usize {
        self.idx(lhs.code - 1, rhs.code - 1)
    }

    // -- Subforest accessors (by index) --

    pub fn get_subforest(&self, i: usize, j: usize) -> SimilarityResult {
        let idx = self.idx(i, j);
        SimilarityResult {
            sim: self.sf[idx],
            mappings: self.mf[idx].clone(),
        }
    }

    pub fn get_subforest_sim(&self, i: usize, j: usize) -> f64 {
        self.sf[self.idx(i, j)]
    }

    pub fn set_subforest(&mut self, i: usize, j: usize, result: &SimilarityResult) {
        let idx = self.idx(i, j);
        self.sf[idx] = result.sim;
        self.mf[idx] = result.mappings.clone();
    }

    // -- Subtree accessors (by index) --

    pub fn get_subtree(&self, i: usize, j: usize) -> SimilarityResult {
        let idx = self.idx(i, j);
        SimilarityResult {
            sim: self.st[idx],
            mappings: self.mt[idx].clone(),
        }
    }

    pub fn get_subtree_sim(&self, i: usize, j: usize) -> f64 {
        self.st[self.idx(i, j)]
    }

    pub fn get_subtree_mappings(&self, i: usize, j: usize) -> &[NodeMapping] {
        &self.mt[self.idx(i, j)]
    }

    pub fn set_subtree(&mut self, i: usize, j: usize, result: &SimilarityResult) {
        let idx = self.idx(i, j);
        self.st[idx] = result.sim;
        self.mt[idx] = result.mappings.clone();
    }

    // -- Node-based accessors (convenience for algorithm code) --

    pub fn get_subforest_by_node(&self, lhs: &Node, rhs: &Node) -> SimilarityResult {
        let idx = self.node_idx(lhs, rhs);
        SimilarityResult {
            sim: self.sf[idx],
            mappings: self.mf[idx].clone(),
        }
    }

    pub fn get_subtree_by_node(&self, lhs: &Node, rhs: &Node) -> SimilarityResult {
        let idx = self.node_idx(lhs, rhs);
        SimilarityResult {
            sim: self.st[idx],
            mappings: self.mt[idx].clone(),
        }
    }

    pub fn get_subtree_sim_by_node(&self, lhs: &Node, rhs: &Node) -> f64 {
        self.st[self.node_idx(lhs, rhs)]
    }

    pub fn get_subtree_mappings_by_node(&self, lhs: &Node, rhs: &Node) -> &[NodeMapping] {
        &self.mt[self.node_idx(lhs, rhs)]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SimilarityResult --

    #[test]
    fn zero_result() {
        let r = SimilarityResult::zero();
        assert_eq!(r.sim, 0.0);
        assert!(r.mappings.is_empty());
    }

    #[test]
    fn result_ordering() {
        let a = SimilarityResult {
            sim: 0.5,
            mappings: vec![],
        };
        let b = SimilarityResult {
            sim: 0.8,
            mappings: vec![],
        };
        let c = SimilarityResult {
            sim: 0.8,
            mappings: vec![],
        };
        assert!(a < b);
        assert!(b > a);
        assert_eq!(b, c);
    }

    #[test]
    fn result_max_selects_highest() {
        let results = vec![
            SimilarityResult {
                sim: 0.1,
                mappings: vec![],
            },
            SimilarityResult {
                sim: 0.9,
                mappings: vec![],
            },
            SimilarityResult {
                sim: 0.4,
                mappings: vec![],
            },
        ];
        let best = results.into_iter().max().unwrap();
        assert_eq!(best.sim, 0.9);
    }

    #[test]
    fn result_plus_combines() {
        let a = SimilarityResult {
            sim: 0.5,
            mappings: vec![NodeMapping {
                lhs: Some(0),
                rhs: Some(1),
                sim: 0.5,
            }],
        };
        let b = SimilarityResult {
            sim: 0.3,
            mappings: vec![NodeMapping {
                lhs: Some(2),
                rhs: Some(3),
                sim: 0.3,
            }],
        };
        let combined = a.plus(&b);
        assert!((combined.sim - 0.8).abs() < 1e-9);
        assert_eq!(combined.mappings.len(), 2);
    }

    // -- SimilarityCache --

    #[test]
    fn cache_defaults_to_zero() {
        let cache = SimilarityCache::new(3, 4);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(cache.get_subtree_sim(i, j), 0.0);
                assert_eq!(cache.get_subforest_sim(i, j), 0.0);
                assert!(cache.get_subtree(i, j).mappings.is_empty());
                assert!(cache.get_subforest(i, j).mappings.is_empty());
            }
        }
    }

    #[test]
    fn cache_set_get_subtree_roundtrip() {
        let mut cache = SimilarityCache::new(3, 3);
        let result = SimilarityResult {
            sim: 0.75,
            mappings: vec![NodeMapping {
                lhs: Some(1),
                rhs: Some(2),
                sim: 0.75,
            }],
        };
        cache.set_subtree(1, 2, &result);

        assert_eq!(cache.get_subtree_sim(1, 2), 0.75);
        let retrieved = cache.get_subtree(1, 2);
        assert_eq!(retrieved.sim, 0.75);
        assert_eq!(retrieved.mappings.len(), 1);

        // other cells remain untouched
        assert_eq!(cache.get_subtree_sim(0, 0), 0.0);
    }

    #[test]
    fn cache_set_get_subforest_roundtrip() {
        let mut cache = SimilarityCache::new(4, 5);
        let result = SimilarityResult {
            sim: 1.5,
            mappings: vec![
                NodeMapping {
                    lhs: Some(0),
                    rhs: Some(0),
                    sim: 0.8,
                },
                NodeMapping {
                    lhs: Some(1),
                    rhs: Some(1),
                    sim: 0.7,
                },
            ],
        };
        cache.set_subforest(2, 3, &result);

        assert_eq!(cache.get_subforest_sim(2, 3), 1.5);
        let retrieved = cache.get_subforest(2, 3);
        assert_eq!(retrieved.mappings.len(), 2);
    }

    #[test]
    fn cache_subtree_and_subforest_are_independent() {
        let mut cache = SimilarityCache::new(2, 2);
        let st_result = SimilarityResult {
            sim: 0.9,
            mappings: vec![],
        };
        let sf_result = SimilarityResult {
            sim: 0.3,
            mappings: vec![],
        };
        cache.set_subtree(0, 1, &st_result);
        cache.set_subforest(0, 1, &sf_result);

        assert_eq!(cache.get_subtree_sim(0, 1), 0.9);
        assert_eq!(cache.get_subforest_sim(0, 1), 0.3);
    }

    #[test]
    fn cache_node_based_accessors() {
        let mut cache = SimilarityCache::new(3, 3);
        let result = SimilarityResult {
            sim: 0.6,
            mappings: vec![NodeMapping {
                lhs: Some(1),
                rhs: Some(0),
                sim: 0.6,
            }],
        };
        // code is 1-based, so code=2 → index 1, code=1 → index 0
        cache.set_subtree(1, 0, &result);

        let lhs = Node {
            code: 2,
            label: "a".to_string(),
            leftmost_child: None,
            sibling: None,
            embedding: None,
        };
        let rhs = Node {
            code: 1,
            label: "b".to_string(),
            leftmost_child: None,
            sibling: None,
            embedding: None,
        };

        assert_eq!(cache.get_subtree_sim_by_node(&lhs, &rhs), 0.6);
        let retrieved = cache.get_subtree_by_node(&lhs, &rhs);
        assert_eq!(retrieved.sim, 0.6);
        assert_eq!(retrieved.mappings.len(), 1);
    }
}
