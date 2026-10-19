#!/bin/bash
#SBATCH --job-name=wild_2x3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-40
#SBATCH --time=6-00:00:00         
#SBATCH --cpus-per-task=12
#SBATCH --mem=8G


source ~/.bashrc
conda activate PLATO_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source DB/postgres_env.sh

# All experiment combinations (2x3 grids only)
EXPERIMENTS=(
# PLATO pointer_mlp additive
"PLATO_wildfire.py,pointer_mlp,additive,setup0_2x3,2"
"PLATO_wildfire.py,pointer_mlp,additive,setup1_2x3,2"
"PLATO_wildfire.py,pointer_mlp,additive,setup2_2x3,2"
"PLATO_wildfire.py,pointer_mlp,additive,setup3_2x3,2"

# PLATO LSTM additive
"PLATO_wildfire.py,pointer_lstm,additive,setup0_2x3,2"
"PLATO_wildfire.py,pointer_lstm,additive,setup1_2x3,2"
"PLATO_wildfire.py,pointer_lstm,additive,setup2_2x3,2"
"PLATO_wildfire.py,pointer_lstm,additive,setup3_2x3,2"

# PLATO pointer_mlp dotproduct
"PLATO_wildfire.py,pointer_mlp,dotproduct,setup0_2x3,2"
"PLATO_wildfire.py,pointer_mlp,dotproduct,setup1_2x3,2"
"PLATO_wildfire.py,pointer_mlp,dotproduct,setup2_2x3,2"
"PLATO_wildfire.py,pointer_mlp,dotproduct,setup3_2x3,2"

# PLATO LSTM additive
"PLATO_wildfire.py,pointer_lstm,dotproduct,setup0_2x3,2"
"PLATO_wildfire.py,pointer_lstm,dotproduct,setup1_2x3,2"
"PLATO_wildfire.py,pointer_lstm,dotproduct,setup2_2x3,2"
"PLATO_wildfire.py,pointer_lstm,dotproduct,setup3_2x3,2"

# Ablations (MLP + additive, all setups)
"PLATO_wildfire.py,pointer_mlp_noteamstats,additive,setup0_2x3,2"
"PLATO_wildfire.py,pointer_mlp_noteamstats,additive,setup1_2x3,2"
"PLATO_wildfire.py,pointer_mlp_noteamstats,additive,setup2_2x3,2"
"PLATO_wildfire.py,pointer_mlp_noteamstats,additive,setup3_2x3,2"
"PLATO_wildfire.py,pointer_mlp_nopointer,additive,setup0_2x3,2"
"PLATO_wildfire.py,pointer_mlp_nopointer,additive,setup1_2x3,2"
"PLATO_wildfire.py,pointer_mlp_nopointer,additive,setup2_2x3,2"
"PLATO_wildfire.py,pointer_mlp_nopointer,additive,setup3_2x3,2"
"PLATO_wildfire.py,pointer_mlp,additive,setup0_2x3,2,nognn"
"PLATO_wildfire.py,pointer_mlp,additive,setup1_2x3,2,nognn"
"PLATO_wildfire.py,pointer_mlp,additive,setup2_2x3,2,nognn"
"PLATO_wildfire.py,pointer_mlp,additive,setup3_2x3,2,nognn"

# MOHITO
"MOHITO_wildfire.py,,,setup0_2x3,2"
"MOHITO_wildfire.py,,,setup1_2x3,2"
"MOHITO_wildfire.py,,,setup2_2x3,2"
"MOHITO_wildfire.py,,,setup3_2x3,2"

# DGN
"DGN_wildfire.py,,,setup0_2x3,2"
"DGN_wildfire.py,,,setup1_2x3,2"
"DGN_wildfire.py,,,setup2_2x3,2"
"DGN_wildfire.py,,,setup3_2x3,2"

# DICG_DE
"dicg_de_wildfire.py,,,setup0_2x3,2"
"dicg_de_wildfire.py,,,setup1_2x3,2"
"dicg_de_wildfire.py,,,setup2_2x3,2"
"dicg_de_wildfire.py,,,setup3_2x3,2"


)

EXP_STR=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID-1]}
IFS=',' read -r SCRIPT POLICY ATTENTION SETTING GRID CRITIC_FLAG <<< "$EXP_STR"

sleep $((SLURM_ARRAY_TASK_ID - 1))


# Build unique experiment name for logs/output/etc
if [[ "$SCRIPT" == "PLATO_wildfire.py" ]]; then
    EXP_NAME="${SCRIPT%%.*}_${POLICY}_${ATTENTION}_${SETTING}_${GRID}"
else
    EXP_NAME="${SCRIPT%%.*}_${SETTING}_${GRID}"
fi

EXP_NAME=${EXP_NAME//__/_}
EXP_NAME=${EXP_NAME//,}
EXP_NAME=$(echo "$EXP_NAME" | sed 's/_$//g')


echo "▶ Experiment: $EXP_NAME"
echo "▶ Script: $SCRIPT  | Policy: $POLICY | Attn: $ATTENTION | Setting: $SETTING | Grid: $GRID"

CMD="python $SCRIPT --experiment_name $EXP_NAME --setting_type $SETTING --grid_size $GRID"
if [ -n "$POLICY" ]; then
    CMD="$CMD --pointer_policy $POLICY"
fi
if [ -n "$ATTENTION" ]; then
    CMD="$CMD --attention_type $ATTENTION"
fi
if [[ "$CRITIC_FLAG" == "nognn" ]]; then
    CMD="$CMD --ablation_no_gnn_critic"
fi

echo "▶ Command: $CMD"
eval $CMD