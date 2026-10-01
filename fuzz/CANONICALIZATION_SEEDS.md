Canonicalization fuzzing uses `fuzz/corpus/canonicalization` directly as the default libFuzzer corpus.

That corpus already contains many discovered inputs with hash-like names. This file documents the additional curated `seed-*` entries added on purpose.

The goal of the curated seeds is not volume. It is coverage of classes that are easy to lose during corpus minimization:

- basic valid graphs
- disconnected graphs
- parser-error inputs
- aromatic and Kekule equivalents
- tetrahedral, alkene, and non-tetrahedral stereo
- wildcard-heavy edge cases
- historical canonicalization regressions

Curated seeds currently added:

- `seed-basic-methane`
- `seed-basic-disconnected`
- `seed-basic-isotope-bracket`
- `seed-invalid-unclosed-ring`
- `seed-invalid-unclosed-bracket`
- `seed-aromatic-benzene`
- `seed-aromatic-benzene-kekule`
- `seed-aromatic-naphthalene`
- `seed-aromatic-indole`
- `seed-aromatic-imidazole`
- `seed-stereo-tetrahedral`
- `seed-stereo-alkene-e`
- `seed-stereo-alkene-z`
- `seed-stereo-square-planar`
- `seed-stereo-nontetrahedral`
- `seed-stereo-disconnected-ring-closure`
- `seed-wildcard-quadruple`
- `seed-wildcard-aromatic-triangle`
- `seed-regression-phosphorus-wildcard`
- `seed-regression-partial-diagnostics`
- `seed-regression-tetrahedral-index`

These are small enough to keep startup cheap, but targeted enough to exercise the canonicalizer beyond random parser junk.
