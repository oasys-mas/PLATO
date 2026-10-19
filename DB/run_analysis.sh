#!/usr/bin/env bash
# Run ablation + main analysis in parallel, for two episode thresholds.
# Usage: bash DB/run_analysis.sh
# Run from: /lustre/work/soh/alirezas/PLATO/

set -euo pipefail

BASE="/lustre/work/soh/alirezas/PLATO/DB/results"
mkdir -p "$BASE/ep1" "$BASE/ep15000"

S="python DB/analyze_results_by_gridsize.py"

echo "[$(date '+%H:%M:%S')] Launching 8 parallel runs..."

# ── ep1 (no effective filter) ──────────────────────────────────────────────
$S --min_episode 1                    > "$BASE/ep1/ablation_plain.txt" 2>&1 & PIDS+=($!)
$S --min_episode 1 --latex            > "$BASE/ep1/ablation_latex.txt" 2>&1 & PIDS+=($!)
$S --min_episode 1 --ablation         > "$BASE/ep1/main_plain.txt"     2>&1 & PIDS+=($!)
$S --min_episode 1 --ablation --latex > "$BASE/ep1/main_latex.txt"     2>&1 & PIDS+=($!)

# ── ep15000 ────────────────────────────────────────────────────────────────
$S --min_episode 15000                    > "$BASE/ep15000/ablation_plain.txt" 2>&1 & PIDS+=($!)
$S --min_episode 15000 --latex            > "$BASE/ep15000/ablation_latex.txt" 2>&1 & PIDS+=($!)
$S --min_episode 15000 --ablation         > "$BASE/ep15000/main_plain.txt"     2>&1 & PIDS+=($!)
$S --min_episode 15000 --ablation --latex > "$BASE/ep15000/main_latex.txt"     2>&1 & PIDS+=($!)

echo "  PIDs: ${PIDS[*]}"
echo "  Waiting for all to finish..."

for pid in "${PIDS[@]}"; do wait "$pid"; done

echo "[$(date '+%H:%M:%S')] All done."
echo ""
echo "── ep1 ──────────────────────────────"
ls -lh "$BASE/ep1/"
echo ""
echo "── ep15000 ──────────────────────────"
ls -lh "$BASE/ep15000/"
