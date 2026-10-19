#!/bin/bash
#SBATCH --job-name=zs_3x3_to_4x4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-11
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

source ~/.bashrc
conda activate PLATO_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source DB/postgres_env.sh
sleep $((RANDOM % 90 + 10))  # stagger DB connections across array tasks

# Zero-shot: trained on 3x3, evaluated on setup*_4x4_zs configs.
# One array task per (baseline, variant) — no checkpoint overlap between tasks.
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

echo "▶ Job $SLURM_ARRAY_TASK_ID: Zero-shot 3x3->4x4_zs | Baseline=$BASELINE | policy=$POLICY | attention=$ATTENTION | extra=$EXTRA"

CMD="python execution/execute.py --type smart --baseline $BASELINE --grid_size 3 --zero_shot --zero_shot_target_grids_size 4 --zero_shot_mode zs --no-load_from_pkl --domain wildfire"
[ -n "$POLICY" ]    && CMD="$CMD --pointer_policy $POLICY"
[ -n "$ATTENTION" ] && CMD="$CMD --attention_type $ATTENTION"
[[ "$EXTRA" == "nognn" ]] && CMD="$CMD --filter_nognn"

echo "▶ Command: $CMD"
eval $CMD
EXIT_CODE=$?
exit $EXIT_CODE
