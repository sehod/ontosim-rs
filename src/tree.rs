use std::fmt;
use std::str::FromStr;

use crate::node::Node;

/// An n-ary labeled tree representing an ontology concept hierarchy.
///
/// A `Tree` stores the human-readable recursive structure (label + children)
/// and eagerly computes a flat postorder-indexed node arena for efficient
/// algorithmic access.
///
/// # Construction
///
/// Trees can be built from bracket notation via [`FromStr`]:
///
/// ```
/// use ontosim::Tree;
///
/// let tree: Tree = "{travel{traffic{ship}{train}{land{bus}}}{visitor}{sights}}"
///     .parse()
///     .unwrap();
///
/// assert_eq!(tree.label(), "travel");
/// assert_eq!(tree.size(), 8);
/// ```
#[derive(Debug, Clone)]
pub struct Tree {
    label: String,
    children: Vec<Tree>,
    /// Postorder-indexed arena of all nodes — built once at construction time.
    postorder: Vec<Node>,
}

impl Tree {
    /// Creates a new tree with the given label and child subtrees.
    pub fn new(label: impl Into<String>, children: Vec<Tree>) -> Self {
        let label = label.into();
        let mut tree = Self {
            label,
            children,
            postorder: Vec::new(),
        };
        tree.postorder = build_postorder(&tree);
        tree
    }

    /// The root label of this tree.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Number of nodes (concepts) in the tree.
    pub fn size(&self) -> usize {
        self.postorder.len()
    }

    /// Returns the node at postorder position `i` (0-based index).
    pub fn subtree(&self, i: usize) -> &Node {
        &self.postorder[i]
    }

    /// Returns the child nodes of the node at postorder position `i`,
    /// following the leftmost-child / sibling chain.
    pub fn subforest(&self, i: usize) -> Vec<&Node> {
        let mut children = Vec::new();
        let mut next = self.postorder[i].leftmost_child;
        while let Some(idx) = next {
            let child = &self.postorder[idx];
            children.push(child);
            next = child.sibling;
        }
        children
    }

    /// Returns a mutable reference to the node at postorder position `i`.
    ///
    /// Primarily used to set embeddings before running similarity with
    /// [`EmbeddingMatching`](crate::matching::EmbeddingMatching).
    pub fn subtree_mut(&mut self, i: usize) -> &mut Node {
        &mut self.postorder[i]
    }
}

// ---------------------------------------------------------------------------
// Postorder arena construction
// ---------------------------------------------------------------------------

fn build_postorder(tree: &Tree) -> Vec<Node> {
    let mut nodes = Vec::new();
    visit_postorder(tree, &mut nodes);
    nodes
}

fn visit_postorder(tree: &Tree, nodes: &mut Vec<Node>) -> usize {
    let mut leftmost_child: Option<usize> = None;
    let mut prev_child: Option<usize> = None;

    for (i, child) in tree.children.iter().enumerate() {
        let child_idx = visit_postorder(child, nodes);
        if i == 0 {
            leftmost_child = Some(child_idx);
        }
        if let Some(prev) = prev_child {
            nodes[prev].sibling = Some(child_idx);
        }
        prev_child = Some(child_idx);
    }

    let idx = nodes.len();
    nodes.push(Node {
        code: idx + 1,
        label: tree.label.clone(),
        leftmost_child,
        sibling: None,
        embedding: None,
    });
    idx
}

// ---------------------------------------------------------------------------
// Bracket notation: parsing (FromStr) and display (Display / fmt)
// ---------------------------------------------------------------------------

/// Bracket notation encodes trees with nested braces, e.g.
/// `{A{B{X}{Y}{F}}{C}}` — root A with children B (which has X, Y, F) and C.
impl FromStr for Tree {
    type Err = ParseTreeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut tree_stack: Vec<(Tree, usize)> = Vec::new();
        let mut label_stack: Vec<String> = Vec::new();

        for ch in s.chars() {
            match ch {
                '{' => label_stack.push(String::new()),
                '}' => {
                    let label = label_stack.pop().ok_or(ParseTreeError::UnbalancedBraces)?;
                    let depth = label_stack.len();

                    let mut children: Vec<Tree> = Vec::new();
                    while tree_stack.last().is_some_and(|(_, d)| *d > depth) {
                        children.push(tree_stack.pop().unwrap().0);
                    }
                    children.reverse();

                    tree_stack.push((Tree::new(label, children), depth));
                }
                _ => {
                    label_stack
                        .last_mut()
                        .ok_or(ParseTreeError::UnbalancedBraces)?
                        .push(ch);
                }
            }
        }

        if label_stack.is_empty() && tree_stack.len() == 1 {
            Ok(tree_stack.pop().unwrap().0)
        } else {
            Err(ParseTreeError::UnbalancedBraces)
        }
    }
}

/// Serializes back to bracket notation via the `Display` trait.
impl fmt::Display for Tree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{{}", self.label)?;
        for child in &self.children {
            write!(f, "{child}")?;
        }
        write!(f, "}}")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseTreeError {
    UnbalancedBraces,
}

impl fmt::Display for ParseTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnbalancedBraces => write!(f, "unbalanced braces in bracket notation"),
        }
    }
}

impl std::error::Error for ParseTreeError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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

    // -- Bracket notation round-trip --

    #[test]
    fn parse_and_display_travel() {
        let tree = travel_tree();
        assert_eq!(
            tree.to_string(),
            "{travel{traffic{ship}{train}{land{bus}}}{visitor}{sights}}"
        );
    }

    #[test]
    fn parse_and_display_tour() {
        let tree = tour_tree();
        assert_eq!(
            tree.to_string(),
            "{tour{transport{road{bus}{light bus}}}{tourist{business}}}"
        );
    }

    #[test]
    fn parse_single_node() {
        let tree: Tree = "{root}".parse().unwrap();
        assert_eq!(tree.label(), "root");
        assert_eq!(tree.size(), 1);
        assert_eq!(tree.to_string(), "{root}");
    }

    #[test]
    fn parse_error_on_unbalanced() {
        assert!("{foo".parse::<Tree>().is_err());
        assert!("foo}".parse::<Tree>().is_err());
        assert!("foo".parse::<Tree>().is_err());
    }

    // -- Postorder numbering --

    #[test]
    fn travel_tree_size() {
        assert_eq!(travel_tree().size(), 8);
    }

    #[test]
    fn tour_tree_size() {
        assert_eq!(tour_tree().size(), 7);
    }

    #[test]
    fn postorder_codes_are_sequential() {
        let tree = travel_tree();
        for (i, node) in tree.postorder.iter().enumerate() {
            assert_eq!(node.code, i + 1, "node '{}' has wrong code", node.label);
        }
    }

    #[test]
    fn travel_postorder_labels() {
        let tree = travel_tree();
        let labels: Vec<&str> = tree.postorder.iter().map(|n| n.label.as_str()).collect();
        // Postorder: leaves first, then parents.
        // ship(1), train(2), bus(3), land(4), traffic(5), visitor(6), sights(7), travel(8)
        assert_eq!(
            labels,
            vec![
                "ship", "train", "bus", "land", "traffic", "visitor", "sights", "travel"
            ]
        );
    }

    #[test]
    fn tour_postorder_labels() {
        let tree = tour_tree();
        let labels: Vec<&str> = tree.postorder.iter().map(|n| n.label.as_str()).collect();
        // bus(1), light bus(2), road(3), transport(4), business(5), tourist(6), tour(7)
        assert_eq!(
            labels,
            vec![
                "bus",
                "light bus",
                "road",
                "transport",
                "business",
                "tourist",
                "tour"
            ]
        );
    }

    // -- subtree / subforest accessors --

    #[test]
    fn subtree_returns_correct_node() {
        let tree = travel_tree();
        assert_eq!(tree.subtree(0).label, "ship");
        assert_eq!(tree.subtree(0).code, 1);
        assert_eq!(tree.subtree(7).label, "travel");
        assert_eq!(tree.subtree(7).code, 8);
    }

    #[test]
    fn subforest_of_leaf_is_empty() {
        let tree = travel_tree();
        // "ship" is a leaf (index 0)
        assert!(tree.subforest(0).is_empty());
    }

    #[test]
    fn subforest_of_traffic() {
        let tree = travel_tree();
        // "traffic" is at index 4; children are ship, train, land
        let children: Vec<&str> = tree.subforest(4).iter().map(|n| n.label.as_str()).collect();
        assert_eq!(children, vec!["ship", "train", "land"]);
    }

    #[test]
    fn subforest_of_root_travel() {
        let tree = travel_tree();
        // "travel" is at index 7; children are traffic, visitor, sights
        let children: Vec<&str> = tree.subforest(7).iter().map(|n| n.label.as_str()).collect();
        assert_eq!(children, vec!["traffic", "visitor", "sights"]);
    }

    #[test]
    fn subforest_of_land() {
        let tree = travel_tree();
        // "land" is at index 3; single child: bus
        let children: Vec<&str> = tree.subforest(3).iter().map(|n| n.label.as_str()).collect();
        assert_eq!(children, vec!["bus"]);
    }

    #[test]
    fn tour_subforest_of_root() {
        let tree = tour_tree();
        // "tour" at index 6; children are transport, tourist
        let children: Vec<&str> = tree.subforest(6).iter().map(|n| n.label.as_str()).collect();
        assert_eq!(children, vec!["transport", "tourist"]);
    }

    #[test]
    fn tour_subforest_of_road() {
        let tree = tour_tree();
        // "road" at index 2; children are bus, light bus
        let children: Vec<&str> = tree.subforest(2).iter().map(|n| n.label.as_str()).collect();
        assert_eq!(children, vec!["bus", "light bus"]);
    }
}
