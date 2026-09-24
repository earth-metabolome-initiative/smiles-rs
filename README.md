# smiles-rs
[![crates.io](https://img.shields.io/crates/v/smiles-rs.svg)](https://crates.io/crates/smiles-rs)
[![docs.rs](https://img.shields.io/docsrs/smiles-rs)](https://docs.rs/smiles-rs)
[![downloads](https://img.shields.io/crates/d/smiles-rs.svg)](https://crates.io/crates/smiles-rs)
[![Rust CI](https://github.com/earth-metabolome-initiative/smiles-rs/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/earth-metabolome-initiative/smiles-rs/actions/workflows/rust.yml)
[![codecov](https://codecov.io/gh/earth-metabolome-initiative/smiles-rs/graph/badge.svg)](https://codecov.io/gh/earth-metabolome-initiative/smiles-rs)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/earth-metabolome-initiative/smiles-rs/blob/main/LICENSE)
[![MSRV](https://img.shields.io/badge/rustc-1.92%2B-orange.svg)](https://blog.rust-lang.org/)

Parses SMILES strings into molecular graphs, following the [OpenSMILES specification](http://opensmiles.org/opensmiles.html). `no_std` with `alloc`, no unsafe code.

Beyond parsing it canonicalizes molecules through `canonicalize` and `canonical_labeling`, perceives aromaticity under the RDKit default, MDL and simple models, and kekulizes back to alternating bonds. It computes symmetrized SSSR rings and ring membership, fragments and connected components, radius-bounded atom environments for MAP4-style fingerprints, and the maximum common edge subgraph of two molecules through [`geometric-traits`](https://crates.io/crates/geometric-traits). Tetrahedral and double bond stereochemistry survive canonicalization, formulas convert into [`molecular-formulas`](https://crates.io/crates/molecular-formulas) types, and `WildcardSmiles` accepts `*` atoms with a fallible conversion back into `Smiles`.

## Example

```rust
use core::str::FromStr;

use molecular_formulas::prelude::ChemicalFormula;
use smiles_rs::prelude::Smiles;

let ethanol = Smiles::from_str("CCO")?;

assert_eq!(ethanol.nodes().len(), 3);
assert_eq!(ethanol.number_of_bonds(), 2);
assert_eq!(ethanol.render(), "CCO");

let formula: ChemicalFormula<u32, i32> = ChemicalFormula::from(&ethanol);
assert_eq!(formula.to_string(), "C₂H₆O");
# Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
```

## Features

| Feature | Effect |
| ------- | ------ |
| `datasets` | Streams PubChem, ZINC20, COCONUT, LOTUS and MassSpecGym from a local cache, as text or parsed into `Smiles` or `WildcardSmiles`, requires `std`. See [`datasets`](https://docs.rs/smiles-rs/latest/smiles_rs/datasets/). |
| `fuzzing` | Exposes the parser internals the fuzz targets drive. |
