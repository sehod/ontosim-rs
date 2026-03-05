/// A node within a [`Tree`](crate::Tree)'s postorder-indexed arena.
///
/// Nodes are not constructed directly — they are produced internally when a
/// `Tree` builds its postorder representation. Children and siblings are
/// referenced by index into the arena rather than by pointer.
#[derive(Debug, Clone)]
pub struct Node {
    /// 1-based postorder number (matches the paper's convention).
    pub code: usize,
    /// The concept label for this node.
    pub label: String,
    /// Arena index of the leftmost child, if any.
    pub leftmost_child: Option<usize>,
    /// Arena index of the next right sibling, if any.
    pub sibling: Option<usize>,
    /// Optional embedding vector, populated externally before running
    /// similarity with [`EmbeddingMatching`](crate::matching::EmbeddingMatching).
    pub embedding: Option<Vec<f32>>,
}
