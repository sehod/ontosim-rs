use crate::node::Node;

/// Computes pairwise semantic similarity between two ontology nodes.
///
/// Implementations define how node labels (or embeddings) are compared to
/// produce a similarity score in the range `[0.0, 1.0]`.
pub trait Matching {
    fn similarity(&self, lhs: &Node, rhs: &Node) -> f64;
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
        if lhs.label == rhs.label {
            1.0
        } else {
            0.0
        }
    }
}

/// Computes similarity via cosine similarity of pre-populated embedding vectors.
///
/// Panics if either node is missing its embedding. Populate embeddings on the
/// tree nodes (via [`Tree::subtree_mut`](crate::Tree::subtree_mut)) before use.
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

    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum();
    let mag_a: f64 = a.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();

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
}
