#!/usr/bin/env bash
#
# Launches the smiles-parser fuzz targets in parallel, one per tmux pane
# (requires the nightly toolchain, `cargo install cargo-fuzz`, and tmux).
#
# Usage:
#   fuzz/run.sh                      # all targets, 60s each, in a tmux session
#   fuzz/run.sh rooted_render        # one target
#   fuzz/run.sh roundtrip canonicalization
#   MAX_TIME=900 fuzz/run.sh         # 15 minutes per target
#
# Environment:
#   MAX_TIME  seconds per target (default 60)
#   SESSION   tmux session name (default smiles-fuzz)
#
# Each pane runs one target and stays open after it finishes so you can read the
# final stats. Detach with Ctrl-b d; reattach with `tmux attach -t smiles-fuzz`.
set -euo pipefail

cd "$(dirname "$0")/.."

ALL_TARGETS=(
    roundtrip
    aromaticity_kekulization_roundtrip
    canonicalization
    rooted_render
)

MAX_TIME="${MAX_TIME:-60}"
SESSION="${SESSION:-smiles-fuzz}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required but not installed" >&2
    exit 1
fi

if [ "$#" -gt 0 ]; then
    targets=("$@")
else
    targets=("${ALL_TARGETS[@]}")
fi

# Build every target up front so the parallel panes do not race on the cargo
# build lock when they start (cargo fuzz build takes one target at a time).
echo "building fuzz targets..."
for target in "${targets[@]}"; do
    cargo +nightly fuzz build "$target"
done

pane_command() {
    local target="$1"
    printf 'cargo +nightly fuzz run %q -- -max_total_time=%q -print_final_stats=1; echo; echo "=== %s finished (press enter to close) ==="; read _' \
        "$target" "$MAX_TIME" "$target"
}

tmux kill-session -t "$SESSION" 2>/dev/null || true
pane=$(tmux new-session -d -s "$SESSION" -n fuzz -P -F '#{pane_id}' "$(pane_command "${targets[0]}")")
tmux select-pane -t "$pane" -T "${targets[0]}"
for target in "${targets[@]:1}"; do
    pane=$(tmux split-window -t "$SESSION" -P -F '#{pane_id}' "$(pane_command "$target")")
    tmux select-pane -t "$pane" -T "$target"
    tmux select-layout -t "$SESSION" tiled >/dev/null
done
tmux select-layout -t "$SESSION" tiled >/dev/null

# Label each pane with its harness name along the pane border.
tmux set-option -t "$SESSION" pane-border-status top
tmux set-option -t "$SESSION" pane-border-format ' #{pane_title} '

echo "started ${#targets[@]} target(s) in tmux session '$SESSION' for ${MAX_TIME}s each"
tmux attach -t "$SESSION"
