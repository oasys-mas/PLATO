#!/bin/bash
#SBATCH --job-name=naive_all
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-4
#SBATCH --time=6:00:00        
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

source ~/.bashrc
conda activate PLATO_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source DB/postgres_env.sh
sleep $((RANDOM % 90 + 10))  # stagger DB connections across array tasks

# Each job runs one naive baseline across all settings in general_config.SETUPS
# (which should include setup*_2x3, setup*_3x3, setup*_4x4)
BASELINES=(
    "NOOP"
    "Random"
    "Weakest"
    "Strongest"
)

BASELINE=${BASELINES[$SLURM_ARRAY_TASK_ID-1]}

echo "▶ Job $SLURM_ARRAY_TASK_ID: Naive baseline=$BASELINE on all configured setups"

CMD="python execution/execute.py --type naive --baseline $BASELINE --domain wildfire"

echo "▶ Command: $CMD"
eval $CMD
EXIT_CODE=$?
exit $EXIT_CODE
