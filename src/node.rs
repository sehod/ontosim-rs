/// A single concept node within a [`Tree`](crate::Tree)'s postorder-indexed
/// arena.
///
/// `Node`s are created automatically when a [`Tree`](crate::Tree) is
/// constructed — you do not need to build them by hand. Fields are public
/// for convenient read access when inspecting similarity results; the
/// [`Tree`](crate::Tree) owns the arena and manages construction.
///
/// Children and siblings are referenced by 0-based index into the arena
/// rather than by pointer, following an adjacency-list style.
#[derive(Debug, Clone)]
pub struct Node {
    /// 1-based postorder sequence number (matches the paper's convention).
    ///
    /// Note: this is **not** the same as the 0-based arena index used by
    /// [`Tree::subtree`](crate::Tree::subtree); `code` is always
    /// `arena_index + 1`.
    pub code: usize,
    /// The concept label for this node (e.g. `"travel"`, `"bus"`).
    pub label: String,
    /// Arena index of the leftmost (first) child, or `None` for leaf nodes.
    pub leftmost_child: Option<usize>,
    /// Arena index of the next right sibling, or `None` if this is the
    /// rightmost child of its parent.
    pub sibling: Option<usize>,
    /// Optional embedding vector for this node's label.
    ///
    /// Populated via [`embed_trees`](crate::matching::embed_trees) or
    /// manually through [`Tree::subtree_mut`](crate::Tree::subtree_mut)
    /// before running similarity with
    /// [`EmbeddingMatching`](crate::matching::EmbeddingMatching).
    pub embedding: Option<Vec<f32>>,
}
