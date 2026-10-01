Each fuzz target reads its curated seeds from `fuzz/corpus/<target>/seed-*`. `cargo fuzz run <target>` uses that directory as its default libFuzzer corpus, and `.clusterfuzzlite/build.sh` zips it into `<target>_seed_corpus.zip`, refusing a target that has none. Inputs libFuzzer discovers locally land in the same directory and stay untracked.

A seed file holds one SMILES string with no trailing newline, because the parser rejects the newline and the seed would only reach the error path.

The seeds are few on purpose. They cover classes that corpus minimization easily loses, chosen for what each target admits.

- `roundtrip` renders and reparses strict and wildcard SMILES, so its seeds span the grammar with bracket atoms carrying isotopes, charges, hydrogen counts and classes, every bond order, two-digit and reused ring closures, stereo of each kind, wildcards, and malformed inputs for the error renderer.
- `canonicalization` checks the canonicalization invariants on strict and wildcard SMILES. Its seeds are basic and disconnected graphs, aromatic and Kekule forms of the same ring, tetrahedral, alkene and non-tetrahedral stereo, wildcard edge cases, parser errors, and the inputs of past canonicalization regressions.
- `aromaticity_kekulization_roundtrip` skips any input with an aromatic atom or bond, so its seeds are Kekule rings (benzenoid, fused, heteroaromatic, charged) next to rings that must stay non-aromatic, such as cyclobutadiene, cyclooctatetraene, fulvene and quinone.
- `rooted_render` takes strict SMILES of at most 48 atoms and re-roots the render at every atom, so its seeds carry directional bonds, including the ones that a rooted render turns into ring closures, tetrahedral centres in and out of rings, and disconnected components.
