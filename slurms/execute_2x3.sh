#!/bin/bash
#SBATCH --job-name=2x3_smarts
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-11
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

source ~/.bashrc
conda activate PLATO_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source DB/postgres_env.sh
sleep $((RANDOM % 90 + 10))  # stagger DB connections across array tasks

# Native execution on grid 2 (2x3): one array task per (baseline, variant).
# Each task scans all matching checkpoints across all setup*_2x3 configs sequentially.
# No overlap: each task filters by a unique (policy, attention, nognn) combination.
EXPERIMENTS=(
    # PLATO main variants
    "PLATO,mlp,additive,"
    "PLATO,lstm,additive,"
    "PLATO,mlp,dotproduct,"
    "PLATO,lstm,dotproduct,"
    # PLATO ablations
    "PLATO,mlp_noteamstats,additive,"
    "PLATO,mlp_nopointer,additive,"
    "PLATO,mlp,additive,nognn"
    # Other baselines
    "DICG,,,"
    "DGN,,,"
    "MOHITO,,,"
    "DEEPSET,,,"
)

EXP_STR=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID-1]}
IFS=',' read -r BASELINE POLICY ATTENTION EXTRA <<< "$EXP_STR"

echo "▶ Job $SLURM_ARRAY_TASK_ID: Baseline=$BASELINE | policy=$POLICY | attention=$ATTENTION | extra=$EXTRA | Grid=2"

CMD="python execution/execute.py --type smart --baseline $BASELINE --grid_size 2 --domain wildfire"
[ -n "$POLICY" ]    && CMD="$CMD --pointer_policy $POLICY"
[ -n "$ATTENTION" ] && CMD="$CMD --attention_type $ATTENTION"
[[ "$EXTRA" == "nognn" ]] && CMD="$CMD --filter_nognn"

echo "▶ Command: $CMD"
eval $CMD
EXIT_CODE=$?
exit $EXIT_CODE
