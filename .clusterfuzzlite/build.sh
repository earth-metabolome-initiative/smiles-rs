#!/bin/bash
set -eu

cd "$SRC/smiles-rs"
cargo fuzz build -O --debug-assertions --fuzz-dir fuzz

targets=$(cargo fuzz list --fuzz-dir fuzz)
if [ -z "$targets" ]; then
    echo "cargo fuzz list named no target" >&2
    exit 1
fi

target_dir=fuzz/target/x86_64-unknown-linux-gnu/release
for name in $targets; do
    seeds=(fuzz/corpus/"$name"/seed-*)
    if [ ! -e "${seeds[0]}" ]; then
        echo "fuzz target $name has no fuzz/corpus/$name/seed-* files" >&2
        exit 1
    fi
    cp "$target_dir/$name" "$OUT/"
    # the runner unpacks <target>_seed_corpus.zip as the starting corpus
    zip -qj "$OUT/${name}_seed_corpus.zip" "${seeds[@]}"
done
