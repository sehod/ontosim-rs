use std::collections::{HashMap, HashSet};

use crate::Tree;
use crate::node::Node;

/// Computes pairwise semantic similarity between two ontology nodes.
///
/// Implementations define how node labels (or embeddings) are compared to
/// produce a similarity score in the range `[0.0, 1.0]`.
pub trait Matching {
    fn similarity(&self, lhs: &Node, rhs: &Node) -> f64;
}

/// Produces embedding vectors from node labels.
///
/// Implement this trait to plug in any embedding backend (local model,
/// remote API, lookup table, etc.). The required method is [`embed`](Embedder::embed);
/// override [`embed_batch`](Embedder::embed_batch) when your backend supports
/// efficient batch requests.
pub trait Embedder {
    /// Computes an embedding vector for a single label.
    fn embed(&self, label: &str) -> Vec<f32>;

    /// Computes embedding vectors for a batch of labels.
    ///
    /// The default implementation calls [`embed`](Embedder::embed) sequentially.
    /// Override this when your backend can process multiple labels in one round-trip.
    fn embed_batch(&self, labels: &[&str]) -> Vec<Vec<f32>> {
        labels.iter().map(|l| self.embed(l)).collect()
    }
}

/// Populates embedding vectors on every node across all provided trees.
///
/// Labels are deduplicated across trees so that each unique label is embedded
/// exactly once, even if it appears in multiple trees.
///
/// # Example
///
/// ```
/// use ontosim::Tree;
/// use ontosim::matching::{Embedder, embed_trees};
///
/// struct UnitEmbedder;
/// impl Embedder for UnitEmbedder {
///     fn embed(&self, _label: &str) -> Vec<f32> { vec![1.0] }
/// }
///
/// let mut t1: Tree = "{A{B}}".parse().unwrap();
/// let mut t2: Tree = "{A{C}}".parse().unwrap();
/// embed_trees(&mut [&mut t1, &mut t2], &UnitEmbedder);
///
/// assert!(t1.subtree(0).embedding.is_some());
/// assert!(t2.subtree(0).embedding.is_some());
/// ```
pub fn embed_trees(trees: &mut [&mut Tree], embedder: &dyn Embedder) {
    let mut seen = HashSet::new();
    let mut unique_labels = Vec::new();
    for tree in trees.iter() {
        for i in 0..tree.size() {
            let label = &tree.subtree(i).label;
            if seen.insert(label.clone()) {
                unique_labels.push(label.clone());
            }
        }
    }

    let label_refs: Vec<&str> = unique_labels.iter().map(|s| s.as_str()).collect();
    let embeddings = embedder.embed_batch(&label_refs);

    let lookup: HashMap<&str, &Vec<f32>> = unique_labels
        .iter()
        .zip(embeddings.iter())
        .map(|(l, e)| (l.as_str(), e))
        .collect();

    for tree in trees.iter_mut() {
        for i in 0..tree.size() {
            let label = tree.subtree(i).label.clone();
            if let Some(emb) = lookup.get(label.as_str()) {
                tree.subtree_mut(i).embedding = Some((*emb).clone());
            }
        }
    }
}

/// Scores 1.0 for identical labels, 0.0 otherwise.
///
/// Useful as a baseline or when no external semantic information is available.
///
/// ```
/// use ontosim::{Tree, matching::{Matching, ExactMatching}};
///
/// let tree: Tree = "{A{B}{C}}".parse().unwrap();
/// let m = ExactMatching;
/// assert_eq!(m.similarity(tree.subtree(0), tree.subtree(0)), 1.0);
/// assert_eq!(m.similarity(tree.subtree(0), tree.subtree(1)), 0.0);
/// ```
pub struct ExactMatching;

impl Matching for ExactMatching {
    fn similarity(&self, lhs: &Node, rhs: &Node) -> f64 {
        if lhs.label == rhs.label { 1.0 } else { 0.0 }
    }
}

/// Computes similarity via cosine similarity of pre-populated embedding vectors.
///
/// Panics if either node is missing its embedding. Use [`embed_trees`] with an
/// [`Embedder`] implementation to populate embeddings before calling this, or
/// set them manually via [`Tree::subtree_mut`](crate::Tree::subtree_mut).
pub struct EmbeddingMatching;

impl Matching for EmbeddingMatching {
    fn similarity(&self, lhs: &Node, rhs: &Node) -> f64 {
        let e1 = lhs
            .embedding
            .as_ref()
            .unwrap_or_else(|| panic!("missing embedding on left node '{}'", lhs.label));
        let e2 = rhs
            .embedding
            .as_ref()
            .unwrap_or_else(|| panic!("missing embedding on right node '{}'", rhs.label));
        cosine_similarity(e1, e2)
    }
}

/// Cosine similarity between two equal-length vectors.
///
/// Returns 0.0 if either vector has zero magnitude.
///
/// # Panics
///
/// Panics if the vectors have different lengths.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum();
    let mag_a: f64 = a
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    let mag_b: f64 = b
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();

    let magnitude = mag_a * mag_b;
    if magnitude == 0.0 {
        0.0
    } else {
        dot / magnitude
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tree;

    fn make_node(label: &str, embedding: Option<Vec<f32>>) -> Node {
        Node {
            code: 1,
            label: label.to_string(),
            leftmost_child: None,
            sibling: None,
            embedding,
        }
    }

    // -- cosine_similarity --

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-9);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn cosine_zero_vector_returns_zero() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
        assert_eq!(cosine_similarity(&b, &a), 0.0);
    }

    #[test]
    #[should_panic(expected = "vector dimensions must match")]
    fn cosine_dimension_mismatch_panics() {
        cosine_similarity(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn cosine_known_value() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // dot = 32, |a| = sqrt(14), |b| = sqrt(77)
        let expected = 32.0 / (14.0_f64.sqrt() * 77.0_f64.sqrt());
        let sim = cosine_similarity(&a, &b);
        assert!((sim - expected).abs() < 1e-9);
    }

    // -- ExactMatching --

    #[test]
    fn exact_matching_same_label() {
        let m = ExactMatching;
        let a = make_node("bus", None);
        let b = make_node("bus", None);
        assert_eq!(m.similarity(&a, &b), 1.0);
    }

    #[test]
    fn exact_matching_different_label() {
        let m = ExactMatching;
        let a = make_node("bus", None);
        let b = make_node("train", None);
        assert_eq!(m.similarity(&a, &b), 0.0);
    }

    // -- EmbeddingMatching --

    #[test]
    fn embedding_matching_uses_cosine() {
        let m = EmbeddingMatching;
        let a = make_node("ship", Some(vec![1.0, 0.0, 0.0]));
        let b = make_node("boat", Some(vec![1.0, 0.0, 0.0]));
        assert!((m.similarity(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn embedding_matching_orthogonal() {
        let m = EmbeddingMatching;
        let a = make_node("up", Some(vec![1.0, 0.0]));
        let b = make_node("right", Some(vec![0.0, 1.0]));
        assert!(m.similarity(&a, &b).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "missing embedding")]
    fn embedding_matching_panics_without_embeddings() {
        let m = EmbeddingMatching;
        let a = make_node("x", None);
        let b = make_node("y", Some(vec![1.0]));
        m.similarity(&a, &b);
    }

    // -- Trait object usage --

    #[test]
    fn matching_as_trait_object() {
        let tree: Tree = "{A{B}{C}}".parse().unwrap();
        let matchers: Vec<Box<dyn Matching>> = vec![Box::new(ExactMatching)];
        for m in &matchers {
            assert_eq!(m.similarity(tree.subtree(0), tree.subtree(0)), 1.0);
            assert_eq!(m.similarity(tree.subtree(0), tree.subtree(1)), 0.0);
        }
    }

    // -- Embedder + embed_trees --

    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Test embedder that produces a deterministic 2-d vector from the first
    /// byte of the label. Tracks how many times `embed_batch` is called and
    /// the total number of labels it receives.
    struct TestEmbedder {
        batch_calls: AtomicUsize,
        labels_seen: AtomicUsize,
    }

    impl TestEmbedder {
        fn new() -> Self {
            Self {
                batch_calls: AtomicUsize::new(0),
                labels_seen: AtomicUsize::new(0),
            }
        }
    }

    impl Embedder for TestEmbedder {
        fn embed(&self, label: &str) -> Vec<f32> {
            let byte = label.as_bytes().first().copied().unwrap_or(0) as f32;
            vec![byte, byte * 0.5]
        }

        fn embed_batch(&self, labels: &[&str]) -> Vec<Vec<f32>> {
            self.batch_calls.fetch_add(1, Ordering::Relaxed);
            self.labels_seen.fetch_add(labels.len(), Ordering::Relaxed);
            labels.iter().map(|l| self.embed(l)).collect()
        }
    }

    #[test]
    fn embed_trees_populates_all_nodes() {
        let mut t1: Tree = "{A{B}{C}}".parse().unwrap();
        let mut t2: Tree = "{X{Y}}".parse().unwrap();
        let embedder = TestEmbedder::new();

        embed_trees(&mut [&mut t1, &mut t2], &embedder);

        for i in 0..t1.size() {
            assert!(t1.subtree(i).embedding.is_some(), "t1 node {} missing", i);
        }
        for i in 0..t2.size() {
            assert!(t2.subtree(i).embedding.is_some(), "t2 node {} missing", i);
        }
    }

    #[test]
    fn embed_trees_deduplicates_shared_labels() {
        let mut t1: Tree = "{A{bus}{C}}".parse().unwrap();
        let mut t2: Tree = "{X{bus}}".parse().unwrap();
        let embedder = TestEmbedder::new();

        embed_trees(&mut [&mut t1, &mut t2], &embedder);

        assert_eq!(embedder.batch_calls.load(Ordering::Relaxed), 1);
        // 5 unique labels: A, bus, C, X — not 5 (bus counted once)
        assert_eq!(embedder.labels_seen.load(Ordering::Relaxed), 4);

        // Both "bus" nodes get the same embedding.
        let bus_t1 = t1.subtree(0).embedding.as_ref().unwrap();
        let bus_t2 = t2.subtree(0).embedding.as_ref().unwrap();
        assert_eq!(bus_t1, bus_t2);
    }

    #[test]
    fn embed_trees_single_tree() {
        let mut t: Tree = "{root{leaf}}".parse().unwrap();
        let embedder = TestEmbedder::new();

        embed_trees(&mut [&mut t], &embedder);

        assert!(t.subtree(0).embedding.is_some());
        assert!(t.subtree(1).embedding.is_some());
    }

    #[test]
    fn embed_trees_then_similarity_roundtrip() {
        let mut t1: Tree = "{A{B}{C}}".parse().unwrap();
        let mut t2: Tree = "{A{B}{D}}".parse().unwrap();
        let embedder = TestEmbedder::new();

        embed_trees(&mut [&mut t1, &mut t2], &embedder);

        let result = crate::similarity::compute(&t1, &t2, &EmbeddingMatching);
        assert!(result.sim > 0.0, "expected positive similarity");
        assert!(!result.mappings.is_empty());
    }
}
