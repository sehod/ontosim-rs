//! Structural and semantic similarity comparison of ontology trees.
//!
//! This crate implements a tree similarity algorithm based on
//! [*"A mapping-based tree similarity algorithm and its application to
//! ontology alignment"*](https://www.sciencedirect.com/science/article/abs/pii/S0950705113003523)
//! (Zhu et al., 2013). Given two [`Tree`]s and a [`Matching`](matching::Matching)
//! strategy, the algorithm computes an overall similarity score together with
//! the optimal node-level mappings between the two hierarchies.
//!
//! # Quick start — exact label matching
//!
//! ```
//! use ontosim::{Tree, similarity};
//! use ontosim::matching::ExactMatching;
//!
//! let t1: Tree = "{travel{traffic{ship}{train}}{visitor}}".parse().unwrap();
//! let t2: Tree = "{travel{traffic{ship}{bus}}{visitor}}".parse().unwrap();
//!
//! let result = similarity::compute(&t1, &t2, &ExactMatching);
//!
//! println!("score: {}", result.sim);
//! for m in &result.mappings {
//!     let lhs = m.lhs.map(|i| t1.subtree(i).label.as_str());
//!     let rhs = m.rhs.map(|j| t2.subtree(j).label.as_str());
//!     println!("  {:?} <-> {:?}  (sim {:.2})", lhs, rhs, m.sim);
//! }
//! ```
//!
//! # Embedding-based matching
//!
//! For semantic (rather than string-equal) comparison, implement the
//! [`Embedder`](matching::Embedder) trait to provide vector embeddings, then
//! use [`EmbeddingMatching`](matching::EmbeddingMatching):
//!
//! ```
//! use ontosim::{Tree, similarity};
//! use ontosim::matching::{Embedder, EmbeddingMatching, embed_trees};
//!
//! struct UnitEmbedder;
//! impl Embedder for UnitEmbedder {
//!     fn embed(&self, _label: &str) -> Vec<f32> { vec![1.0, 0.0] }
//! }
//!
//! let mut t1: Tree = "{A{B}}".parse().unwrap();
//! let mut t2: Tree = "{A{C}}".parse().unwrap();
//!
//! embed_trees(&mut [&mut t1, &mut t2], &UnitEmbedder);
//! let result = similarity::compute(&t1, &t2, &EmbeddingMatching);
//! assert!(result.sim > 0.0);
//! ```

pub mod assignment;
pub mod cache;
pub mod matching;
mod node;
pub mod similarity;
mod tree;

pub use node::Node;
pub use tree::{ParseTreeError, Tree};
