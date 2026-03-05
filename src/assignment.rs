//! Hungarian method (Kuhn-Munkres) for optimal bipartite graph assignment.
//!
//! Used internally by the tree similarity algorithm to find the optimal
//! pairing of child subtrees between two nodes (the KM step in the paper).

use std::collections::BTreeSet;

use crate::cache::{SimilarityCache, SimilarityResult};
use crate::node::Node;

/// Solves the optimal assignment problem for two subforests using the
/// Hungarian method, returning the best-scoring set of subtree pairings.
///
/// `subforest1` and `subforest2` are the child nodes of two nodes being
/// compared. `cache` must already contain subtree similarity scores for
/// every `(lhs, rhs)` pair in the two subforests.
pub fn compute_optimal_mappings(
    subforest1: &[&Node],
    subforest2: &[&Node],
    cache: &SimilarityCache,
) -> SimilarityResult {
    let dim = subforest1.len().max(subforest2.len());
    if dim == 0 {
        return SimilarityResult::zero();
    }

    // Build cost matrix: negated similarities (because the Hungarian method minimises).
    // Padding cells default to 1.0 (from init), real cells are overwritten.
    let mut costs = vec![vec![1.0; dim]; dim];
    let mut max_possible_sim = 0.0;

    for (i, lhs) in subforest1.iter().enumerate() {
        for (j, rhs) in subforest2.iter().enumerate() {
            let sim = cache.get_subtree_sim_by_node(lhs, rhs);
            costs[i][j] = -sim;
            max_possible_sim += -costs[i][j];
        }
    }

    if max_possible_sim == 0.0 {
        return SimilarityResult::zero();
    }

    // For non-square matrices: minimise the real dimension first, pad dimension second.
    if subforest1.len() >= subforest2.len() {
        minimize_rows(&mut costs, dim);
        minimize_cols(&mut costs, dim);
    } else {
        minimize_cols(&mut costs, dim);
        minimize_rows(&mut costs, dim);
    }

    loop {
        let covered = find_min_lines(&costs, dim);
        if covered.needed_lines >= dim {
            return make_assignment(covered, subforest1, subforest2, cache);
        }
        minimize_uncovered(&mut costs, &covered, dim);
    }
}

// ---------------------------------------------------------------------------
// Step 1 & 2: Row / column minimisation
// ---------------------------------------------------------------------------

fn minimize_rows(costs: &mut [Vec<f64>], dim: usize) {
    for row in costs.iter_mut().take(dim) {
        let min = row.iter().copied().take(dim).fold(f64::INFINITY, f64::min);
        for v in row.iter_mut().take(dim) {
            *v -= min;
        }
    }
}

fn minimize_cols(costs: &mut [Vec<f64>], dim: usize) {
    for j in 0..dim {
        let min = (0..dim).map(|i| costs[i][j]).fold(f64::INFINITY, f64::min);
        for row in costs.iter_mut().take(dim) {
            row[j] -= min;
        }
    }
}

// ---------------------------------------------------------------------------
// Step 3: Find minimum covering lines
// ---------------------------------------------------------------------------

/// Which dimension a line covers.
#[derive(Debug, Clone)]
enum LineKind {
    Row(usize),
    Col(usize),
}

#[derive(Debug, Clone)]
struct Line {
    kind: LineKind,
    zero_positions: BTreeSet<usize>,
}

impl Line {
    fn num_zeros(&self) -> usize {
        self.zero_positions.len()
    }
}

struct LineCoverage {
    /// Original (unmodified) lines keyed by kind for the assignment step.
    row_lines: Vec<Line>,
    col_lines: Vec<Line>,
    /// How many lines were drawn through each cell (0, 1, or 2).
    covered: Vec<Vec<i32>>,
    needed_lines: usize,
}

fn find_min_lines(costs: &[Vec<f64>], dim: usize) -> LineCoverage {
    // Build row and column lines with their zero positions.
    let mut row_lines: Vec<Line> = (0..dim)
        .map(|i| Line {
            kind: LineKind::Row(i),
            zero_positions: BTreeSet::new(),
        })
        .collect();
    let mut col_lines: Vec<Line> = (0..dim)
        .map(|j| Line {
            kind: LineKind::Col(j),
            zero_positions: BTreeSet::new(),
        })
        .collect();

    let mut zero_count = 0usize;
    for i in 0..dim {
        for j in 0..dim {
            if costs[i][j] == 0.0 {
                zero_count += 1;
                row_lines[i].zero_positions.insert(j);
                col_lines[j].zero_positions.insert(i);
            }
        }
    }

    // Working copies sorted by descending zero count.
    // Each entry is (original_kind, zero_positions_copy).
    let mut temp: Vec<Line> = row_lines.iter().chain(col_lines.iter()).cloned().collect();
    temp.sort_by(|a, b| b.num_zeros().cmp(&a.num_zeros()));

    let mut covered = vec![vec![0i32; dim]; dim];
    let mut covered_zero_count = 0usize;
    let mut needed_lines = 0usize;

    while !temp.is_empty() && covered_zero_count < zero_count {
        let line = temp.remove(0);
        needed_lines += 1;

        match line.kind {
            LineKind::Row(row) => {
                for j in 0..dim {
                    covered[row][j] += 1;
                    if line.zero_positions.contains(&j) && covered[row][j] == 1 {
                        covered_zero_count += 1;
                        // Reduce the corresponding column line's zero set.
                        if let Some(pos) = temp
                            .iter()
                            .position(|l| matches!(l.kind, LineKind::Col(c) if c == j))
                        {
                            temp[pos].zero_positions.remove(&row);
                            if temp[pos].zero_positions.is_empty() {
                                temp.remove(pos);
                            } else {
                                temp.sort_by(|a, b| b.num_zeros().cmp(&a.num_zeros()));
                            }
                        }
                    }
                }
            }
            LineKind::Col(col) => {
                for i in 0..dim {
                    covered[i][col] += 1;
                    if line.zero_positions.contains(&i) && covered[i][col] == 1 {
                        covered_zero_count += 1;
                        if let Some(pos) = temp
                            .iter()
                            .position(|l| matches!(l.kind, LineKind::Row(r) if r == i))
                        {
                            temp[pos].zero_positions.remove(&col);
                            if temp[pos].zero_positions.is_empty() {
                                temp.remove(pos);
                            } else {
                                temp.sort_by(|a, b| b.num_zeros().cmp(&a.num_zeros()));
                            }
                        }
                    }
                }
            }
        }
    }

    LineCoverage {
        row_lines,
        col_lines,
        covered,
        needed_lines,
    }
}

// ---------------------------------------------------------------------------
// Step 4: Make assignment from the zero positions
// ---------------------------------------------------------------------------

fn make_assignment(
    coverage: LineCoverage,
    subforest1: &[&Node],
    subforest2: &[&Node],
    cache: &SimilarityCache,
) -> SimilarityResult {
    // Build a mutable map of lines (row lines + col lines) keyed for lookup.
    // We'll consume lines as assignments are made.
    struct AssignLine {
        is_row: bool,
        index: usize,
        zeros: BTreeSet<usize>,
    }

    let mut lines: Vec<AssignLine> = coverage
        .row_lines
        .into_iter()
        .map(|l| {
            let idx = match l.kind {
                LineKind::Row(r) => r,
                _ => unreachable!(),
            };
            AssignLine {
                is_row: true,
                index: idx,
                zeros: l.zero_positions,
            }
        })
        .chain(coverage.col_lines.into_iter().map(|l| {
            let idx = match l.kind {
                LineKind::Col(c) => c,
                _ => unreachable!(),
            };
            AssignLine {
                is_row: false,
                index: idx,
                zeros: l.zero_positions,
            }
        }))
        .collect();

    let mut sim = 0.0;
    let mut mappings = Vec::new();

    while !lines.is_empty() {
        // Find the line with the fewest zeros.
        let min_idx = lines
            .iter()
            .enumerate()
            .min_by_key(|(_, l)| l.zeros.len())
            .map(|(idx, _)| idx)
            .unwrap();
        let min_line = lines.remove(min_idx);

        let (row, col) = if min_line.is_row {
            let col = match min_line.zeros.iter().next() {
                Some(&c) => c,
                None => continue,
            };
            (min_line.index, col)
        } else {
            let row = match min_line.zeros.iter().next() {
                Some(&r) => r,
                None => continue,
            };
            (row, min_line.index)
        };

        // Remove the complementary line for the assigned row/col.
        lines.retain(|l| !(l.is_row && l.index == row) && !(!l.is_row && l.index == col));

        // Update remaining lines: the assigned column can't be used by any row
        // line, and the assigned row can't be used by any column line.
        let mut to_remove = Vec::new();
        for (i, l) in lines.iter_mut().enumerate() {
            if l.is_row {
                l.zeros.remove(&col);
            } else {
                l.zeros.remove(&row);
            }
            if l.zeros.is_empty() {
                to_remove.push(i);
            }
        }
        for i in to_remove.into_iter().rev() {
            lines.remove(i);
        }

        // Look up the actual subtree result from the cache (not the cost matrix).
        if let (Some(lhs), Some(rhs)) = (subforest1.get(row), subforest2.get(col)) {
            let st_sim = cache.get_subtree_sim_by_node(lhs, rhs);
            if st_sim > 0.0 {
                sim += st_sim;
                mappings.extend_from_slice(cache.get_subtree_mappings_by_node(lhs, rhs));
            }
        }
    }

    SimilarityResult { sim, mappings }
}

// ---------------------------------------------------------------------------
// Step 5: Minimise uncovered cells
// ---------------------------------------------------------------------------

fn minimize_uncovered(costs: &mut [Vec<f64>], coverage: &LineCoverage, dim: usize) {
    let mut min_value = f64::INFINITY;
    for i in 0..dim {
        for j in 0..dim {
            if coverage.covered[i][j] == 0 && costs[i][j] < min_value {
                min_value = costs[i][j];
            }
        }
    }

    for i in 0..dim {
        for j in 0..dim {
            if costs[i][j] != 0.0 {
                let cover_count = coverage.covered[i][j];
                if cover_count == 0 {
                    costs[i][j] -= min_value;
                } else if cover_count == 2 {
                    costs[i][j] += min_value;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::NodeMapping;

    fn make_node(code: usize, label: &str) -> Node {
        Node {
            code,
            label: label.to_string(),
            leftmost_child: None,
            sibling: None,
            embedding: None,
        }
    }

    #[test]
    fn empty_subforests_return_zero() {
        let cache = SimilarityCache::new(1, 1);
        let result = compute_optimal_mappings(&[], &[], &cache);
        assert_eq!(result.sim, 0.0);
        assert!(result.mappings.is_empty());
    }

    #[test]
    fn single_pair_returns_cached_similarity() {
        let mut cache = SimilarityCache::new(1, 1);
        cache.set_subtree(
            0,
            0,
            &SimilarityResult {
                sim: 0.85,
                mappings: vec![NodeMapping {
                    lhs: Some(0),
                    rhs: Some(0),
                    sim: 0.85,
                }],
            },
        );

        let n1 = make_node(1, "a");
        let n2 = make_node(1, "b");
        let result = compute_optimal_mappings(&[&n1], &[&n2], &cache);
        assert!((result.sim - 0.85).abs() < 1e-9);
        assert_eq!(result.mappings.len(), 1);
    }

    #[test]
    fn two_by_two_optimal_assignment() {
        // Set up a 2x2 case where optimal is (0→1, 1→0):
        //   cache[0,0]=0.1  cache[0,1]=0.9
        //   cache[1,0]=0.8  cache[1,1]=0.2
        // Optimal: pair (0,1) + (1,0) = 0.9 + 0.8 = 1.7
        let mut cache = SimilarityCache::new(2, 2);
        let pairs = [(0, 0, 0.1), (0, 1, 0.9), (1, 0, 0.8), (1, 1, 0.2)];
        for &(i, j, sim) in &pairs {
            cache.set_subtree(
                i,
                j,
                &SimilarityResult {
                    sim,
                    mappings: vec![NodeMapping {
                        lhs: Some(i),
                        rhs: Some(j),
                        sim,
                    }],
                },
            );
        }

        let n1a = make_node(1, "a");
        let n1b = make_node(2, "b");
        let n2a = make_node(1, "x");
        let n2b = make_node(2, "y");
        let result = compute_optimal_mappings(&[&n1a, &n1b], &[&n2a, &n2b], &cache);
        assert!((result.sim - 1.7).abs() < 1e-9);
        assert_eq!(result.mappings.len(), 2);
    }

    #[test]
    fn non_square_more_rows() {
        // 3 nodes vs 2 nodes — one row must go unmatched.
        //   cache[0,0]=0.5  cache[0,1]=0.1
        //   cache[1,0]=0.2  cache[1,1]=0.9
        //   cache[2,0]=0.3  cache[2,1]=0.4
        // Best: (0,0)=0.5 + (1,1)=0.9 = 1.4  (row 2 unmatched)
        let mut cache = SimilarityCache::new(3, 2);
        let pairs = [
            (0, 0, 0.5),
            (0, 1, 0.1),
            (1, 0, 0.2),
            (1, 1, 0.9),
            (2, 0, 0.3),
            (2, 1, 0.4),
        ];
        for &(i, j, sim) in &pairs {
            cache.set_subtree(
                i,
                j,
                &SimilarityResult {
                    sim,
                    mappings: vec![NodeMapping {
                        lhs: Some(i),
                        rhs: Some(j),
                        sim,
                    }],
                },
            );
        }

        let nodes1: Vec<Node> = (0..3).map(|i| make_node(i + 1, &format!("l{i}"))).collect();
        let nodes2: Vec<Node> = (0..2).map(|j| make_node(j + 1, &format!("r{j}"))).collect();
        let refs1: Vec<&Node> = nodes1.iter().collect();
        let refs2: Vec<&Node> = nodes2.iter().collect();
        let result = compute_optimal_mappings(&refs1, &refs2, &cache);
        assert!((result.sim - 1.4).abs() < 1e-9);
    }

    #[test]
    fn non_square_more_cols() {
        // 2 nodes vs 3 nodes — one column must go unmatched.
        //   cache[0,0]=0.1  cache[0,1]=0.8  cache[0,2]=0.3
        //   cache[1,0]=0.7  cache[1,1]=0.2  cache[1,2]=0.4
        // Best: (0,1)=0.8 + (1,0)=0.7 = 1.5
        let mut cache = SimilarityCache::new(2, 3);
        let pairs = [
            (0, 0, 0.1),
            (0, 1, 0.8),
            (0, 2, 0.3),
            (1, 0, 0.7),
            (1, 1, 0.2),
            (1, 2, 0.4),
        ];
        for &(i, j, sim) in &pairs {
            cache.set_subtree(
                i,
                j,
                &SimilarityResult {
                    sim,
                    mappings: vec![NodeMapping {
                        lhs: Some(i),
                        rhs: Some(j),
                        sim,
                    }],
                },
            );
        }

        let nodes1: Vec<Node> = (0..2).map(|i| make_node(i + 1, &format!("l{i}"))).collect();
        let nodes2: Vec<Node> = (0..3).map(|j| make_node(j + 1, &format!("r{j}"))).collect();
        let refs1: Vec<&Node> = nodes1.iter().collect();
        let refs2: Vec<&Node> = nodes2.iter().collect();
        let result = compute_optimal_mappings(&refs1, &refs2, &cache);
        assert!((result.sim - 1.5).abs() < 1e-9);
    }

    #[test]
    fn all_zero_similarities_returns_zero() {
        let cache = SimilarityCache::new(2, 2);
        let n1a = make_node(1, "a");
        let n1b = make_node(2, "b");
        let n2a = make_node(1, "x");
        let n2b = make_node(2, "y");
        let result = compute_optimal_mappings(&[&n1a, &n1b], &[&n2a, &n2b], &cache);
        assert_eq!(result.sim, 0.0);
        assert!(result.mappings.is_empty());
    }
}
