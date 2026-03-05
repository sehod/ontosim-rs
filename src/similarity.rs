//! Core tree similarity algorithm.
//!
//! Implements the bottom-up dynamic-programming procedure from the paper:
//! every pair of subtrees/subforests is compared in postorder, with optimal
//! child-pairing solved via the Hungarian method.

use crate::Tree;
use crate::assignment;
use crate::cache::{NodeMapping, SimilarityCache, SimilarityResult};
use crate::matching::Matching;

/// Computes the structural and semantic similarity between two ontology trees.
///
/// The algorithm decomposes `t1` and `t2` into subtrees at every node,
/// computes pairwise similarities bottom-up in postorder, and returns the
/// overall similarity score together with the optimal node-level mappings.
///
/// # Arguments
///
/// * `t1` — the left-hand tree
/// * `t2` — the right-hand tree
/// * `matching` — strategy for computing pairwise node similarity (e.g.
///   [`ExactMatching`](crate::matching::ExactMatching) or
///   [`EmbeddingMatching`](crate::matching::EmbeddingMatching))
///
/// # Example
///
/// ```
/// use ontosim::{Tree, similarity};
/// use ontosim::matching::ExactMatching;
///
/// let t1: Tree = "{A{B}{C}}".parse().unwrap();
/// let t2: Tree = "{A{B}{D}}".parse().unwrap();
///
/// let result = similarity::compute(&t1, &t2, &ExactMatching);
/// assert!(result.sim > 0.0);
/// ```
pub fn compute(t1: &Tree, t2: &Tree, matching: &dyn Matching) -> SimilarityResult {
    let m = t1.size();
    let n = t2.size();
    let mut cache = SimilarityCache::new(m, n);

    for i in 0..m {
        let ci = t1.subforest(i);

        for j in 0..n {
            let cj = t2.subforest(j);

            if let Some(sf) = compute_subforest_similarity(t1, t2, i, j, &ci, &cj, &cache) {
                cache.set_subforest(i, j, &sf);
            }

            if let Some(st) = compute_subtree_similarity(t1, t2, i, j, &ci, &cj, &cache, matching) {
                cache.set_subtree(i, j, &st);
            }
        }
    }

    cache.get_subtree(m - 1, n - 1)
}

/// Subforest similarity: S_TREE(SF(i), SF(j))
///
/// MAX of:
///   Case 1 — SF(i) is one part of SF(j): max over children of j
///   Case 2 — SF(j) is one part of SF(i): max over children of i
///   Case 3 — KM optimal matching of SF(i) and SF(j)
fn compute_subforest_similarity(
    t1: &Tree,
    t2: &Tree,
    i: usize,
    j: usize,
    ci: &[&crate::Node],
    cj: &[&crate::Node],
    cache: &SimilarityCache,
) -> Option<SimilarityResult> {
    // Case 1: max { subforest(node_i, child_of_j) } for each child of j
    let case1 = cj
        .iter()
        .map(|child_j| cache.get_subforest_by_node(t1.subtree(i), child_j))
        .max();

    // Case 2: max { subforest(child_of_i, node_j) } for each child of i
    let case2 = ci
        .iter()
        .map(|child_i| cache.get_subforest_by_node(child_i, t2.subtree(j)))
        .max();

    // Case 3: KM assignment of the two subforests
    let sf1: Vec<&crate::Node> = t1.subforest(i);
    let sf2: Vec<&crate::Node> = t2.subforest(j);
    let case3 = assignment::compute_optimal_mappings(&sf1, &sf2, cache);

    [case1, case2, Some(case3)]
        .into_iter()
        .flatten()
        .max()
        .filter(|r| r.sim > 0.0)
}

/// Subtree similarity: S_TREE(ST(i), ST(j))
///
/// MAX of:
///   Case 1 — ST(i) is one part of ST(j): max over children of j
///   Case 2 — ST(j) is one part of ST(i): max over children of i
///   Case 3 — mutual match: subforest(i,j) + sim(root_i, root_j)
#[allow(clippy::too_many_arguments)] // Internal DP helper; bundling args into a struct adds indirection without clarity.
fn compute_subtree_similarity(
    t1: &Tree,
    t2: &Tree,
    i: usize,
    j: usize,
    ci: &[&crate::Node],
    cj: &[&crate::Node],
    cache: &SimilarityCache,
    matching: &dyn Matching,
) -> Option<SimilarityResult> {
    let lhs = t1.subtree(i);
    let rhs = t2.subtree(j);

    // Case 1: max { subtree(node_i, child_of_j) } for each child of j
    let case1 = cj
        .iter()
        .map(|child_j| cache.get_subtree_by_node(t1.subtree(i), child_j))
        .max();

    // Case 2: max { subtree(child_of_i, node_j) } for each child of i
    let case2 = ci
        .iter()
        .map(|child_i| cache.get_subtree_by_node(child_i, t2.subtree(j)))
        .max();

    // Case 3: subforest(i,j) + sim(root_i, root_j)
    let root_sim = matching.similarity(lhs, rhs);
    let root_mapping = SimilarityResult {
        sim: root_sim,
        mappings: vec![NodeMapping {
            lhs: Some(i),
            rhs: Some(j),
            sim: root_sim,
        }],
    };
    let case3 = cache.get_subforest_by_node(lhs, rhs).plus(&root_mapping);

    [case1, case2, Some(case3)]
        .into_iter()
        .flatten()
        .max()
        .filter(|r| r.sim > 0.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Node;
    use crate::matching::{ExactMatching, Matching};

    /// Test-only matcher that reproduces the paper's hardcoded similarity
    /// values for the travel/tour example.
    struct HardcodedMatching;

    impl Matching for HardcodedMatching {
        fn similarity(&self, lhs: &Node, rhs: &Node) -> f64 {
            match (lhs.label.as_str(), rhs.label.as_str()) {
                (a, b) if a == b => 1.0,
                ("travel", "tour") | ("tour", "travel") => 1.0,
                ("visitor", "tourist") | ("tourist", "visitor") => 0.9,
                ("traffic", "transport") | ("transport", "traffic") => 0.85,
                ("land", "road") | ("road", "land") => 0.7,
                _ => 0.0,
            }
        }
    }

    fn travel_tree() -> Tree {
        "{travel{traffic{ship}{train}{land{bus}}}{visitor}{sights}}"
            .parse()
            .unwrap()
    }

    fn tour_tree() -> Tree {
        "{tour{transport{road{bus}{light bus}}}{tourist{business}}}"
            .parse()
            .unwrap()
    }

    // -- Basic sanity --

    #[test]
    fn identical_trees_score_perfectly() {
        let t = travel_tree();
        let result = compute(&t, &t, &ExactMatching);
        // Every node maps to itself with sim 1.0, so total = tree size.
        assert!((result.sim - t.size() as f64).abs() < 1e-9);
        assert_eq!(result.mappings.len(), t.size());
    }

    #[test]
    fn completely_disjoint_trees_score_zero() {
        let t1: Tree = "{A{B}{C}}".parse().unwrap();
        let t2: Tree = "{X{Y}{Z}}".parse().unwrap();
        let result = compute(&t1, &t2, &ExactMatching);
        assert_eq!(result.sim, 0.0);
        assert!(result.mappings.is_empty());
    }

    #[test]
    fn single_node_trees_matching() {
        let t1: Tree = "{hello}".parse().unwrap();
        let t2: Tree = "{hello}".parse().unwrap();
        let result = compute(&t1, &t2, &ExactMatching);
        assert!((result.sim - 1.0).abs() < 1e-9);
        assert_eq!(result.mappings.len(), 1);
    }

    #[test]
    fn single_node_trees_not_matching() {
        let t1: Tree = "{hello}".parse().unwrap();
        let t2: Tree = "{world}".parse().unwrap();
        let result = compute(&t1, &t2, &ExactMatching);
        assert_eq!(result.sim, 0.0);
    }

    // -- Paper example: travel vs tour --

    #[test]
    fn travel_tour_has_positive_similarity() {
        let result = compute(&travel_tree(), &tour_tree(), &HardcodedMatching);
        assert!(result.sim > 0.0);
    }

    #[test]
    fn travel_tour_bus_mapping_exists() {
        let t1 = travel_tree();
        let t2 = tour_tree();
        let result = compute(&t1, &t2, &HardcodedMatching);

        // "bus" is index 2 in travel (code 3) and index 0 in tour (code 1)
        let bus_mapping = result.mappings.iter().find(|m| {
            m.lhs.map(|i| t1.subtree(i).label.as_str()) == Some("bus")
                && m.rhs.map(|j| t2.subtree(j).label.as_str()) == Some("bus")
        });
        assert!(bus_mapping.is_some(), "expected bus<->bus mapping");
        assert!((bus_mapping.unwrap().sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn travel_tour_root_mapping_exists() {
        let t1 = travel_tree();
        let t2 = tour_tree();
        let result = compute(&t1, &t2, &HardcodedMatching);

        let root_mapping = result.mappings.iter().find(|m| {
            m.lhs.map(|i| t1.subtree(i).label.as_str()) == Some("travel")
                && m.rhs.map(|j| t2.subtree(j).label.as_str()) == Some("tour")
        });
        assert!(
            root_mapping.is_some(),
            "expected travel<->tour root mapping"
        );
        assert!((root_mapping.unwrap().sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn travel_tour_expected_mappings() {
        let t1 = travel_tree();
        let t2 = tour_tree();
        let result = compute(&t1, &t2, &HardcodedMatching);

        let mapping_labels: Vec<(Option<&str>, Option<&str>)> = result
            .mappings
            .iter()
            .map(|m| {
                (
                    m.lhs.map(|i| t1.subtree(i).label.as_str()),
                    m.rhs.map(|j| t2.subtree(j).label.as_str()),
                )
            })
            .collect();

        // Verify the key semantic pairings the algorithm should find
        assert!(mapping_labels.contains(&(Some("bus"), Some("bus"))));
        assert!(mapping_labels.contains(&(Some("land"), Some("road"))));
        assert!(mapping_labels.contains(&(Some("traffic"), Some("transport"))));
        assert!(mapping_labels.contains(&(Some("visitor"), Some("tourist"))));
        assert!(mapping_labels.contains(&(Some("travel"), Some("tour"))));
    }

    #[test]
    fn travel_tour_similarity_score() {
        let t1 = travel_tree();
        let t2 = tour_tree();
        let result = compute(&t1, &t2, &HardcodedMatching);

        // Expected: bus(1.0) + land/road(0.7) + traffic/transport(0.85) +
        //           visitor/tourist(0.9) + travel/tour(1.0) = 4.45
        assert!(
            (result.sim - 4.45).abs() < 1e-9,
            "expected 4.45, got {}",
            result.sim
        );
    }

    // -- Partial overlap --

    #[test]
    fn partial_overlap_with_exact_matching() {
        let t1: Tree = "{A{B}{C}{D}}".parse().unwrap();
        let t2: Tree = "{A{B}{E}}".parse().unwrap();
        let result = compute(&t1, &t2, &ExactMatching);
        // A matches A, B matches B → sim = 2.0
        assert!((result.sim - 2.0).abs() < 1e-9);
        assert_eq!(result.mappings.len(), 2);
    }
}
