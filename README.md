# ontosim

Rust library for computing structural and semantic similarity between ontology trees.

Implements the algorithm from [*A mapping-based tree similarity algorithm and its application to ontology alignment*](https://www.sciencedirect.com/science/article/abs/pii/S0950705113003523) (Zhu et al., 2013).

## Quick start

```rust
use ontosim::{Tree, similarity};
use ontosim::matching::ExactMatching;

let t1: Tree = "{travel{traffic{ship}{train}{land{bus}}}{visitor}{sights}}".parse().unwrap();
let t2: Tree = "{tour{transport{road{bus}{light bus}}}{tourist{business}}}".parse().unwrap();

let result = similarity::compute(&t1, &t2, &ExactMatching);
println!("similarity: {}", result.sim);
```

## How it works

1. Each tree is decomposed into subtrees indexed in postorder.
2. Pairwise node similarity is computed via a pluggable `Matching` trait (label equality, cosine similarity of embeddings, or custom).
3. Optimal child-pairing at each level is solved with the Hungarian method (Kuhn-Munkres).
4. A bottom-up dynamic-programming pass produces the overall similarity score and node-level mappings.

## Matching strategies

| Strategy | Description |
|---|---|
| `ExactMatching` | 1.0 for identical labels, 0.0 otherwise |
| `EmbeddingMatching` | Cosine similarity of pre-populated embedding vectors |
| Custom `impl Matching` | Any user-defined pairwise node similarity function |

## Building & testing

```sh
cargo build
cargo test
```

## License

MIT
