//! Structural and semantic similarity comparison of ontology trees.
//!
//! This crate implements a tree similarity algorithm based on
//! [*"A mapping-based tree similarity algorithm and its application to
//! ontology alignment"*](https://www.sciencedirect.com/science/article/abs/pii/S0950705113003523)
//! (Zhu et al., 2013).

pub mod assignment;
pub mod cache;
pub mod matching;
mod node;
pub mod similarity;
mod tree;

pub use node::Node;
pub use tree::{ParseTreeError, Tree};
