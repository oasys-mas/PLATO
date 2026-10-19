#!/bin/bash
#SBATCH --job-name=plato_analysis
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-6
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

source ~/.bashrc
conda activate PLATO_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source DB/postgres_env.sh
sleep $((RANDOM % 60 + 5))

OUT_DIR="$(pwd)/DB/results"
mkdir -p "$OUT_DIR"

# 1-2 : native only        (plain + latex)
# 3-4 : zeroshot only      (plain + latex)
# 5-6 : ablation (native + zs together, all 7 PLATO variants)  (plain + latex)
case $SLURM_ARRAY_TASK_ID in
  1) FLAGS="--section native";          OUTFILE="$OUT_DIR/native_plain.txt"    ;;
  2) FLAGS="--section native --latex";  OUTFILE="$OUT_DIR/native_latex.txt"    ;;
  3) FLAGS="--section zs";              OUTFILE="$OUT_DIR/zs_plain.txt"        ;;
  4) FLAGS="--section zs --latex";      OUTFILE="$OUT_DIR/zs_latex.txt"        ;;
  5) FLAGS="--ablation";                OUTFILE="$OUT_DIR/ablation_plain.txt"  ;;
  6) FLAGS="--ablation --latex";        OUTFILE="$OUT_DIR/ablation_latex.txt"  ;;
esac

echo "▶ Task $SLURM_ARRAY_TASK_ID: flags=$FLAGS"
echo "▶ Output: $OUTFILE"

python -u DB/analyze_results_by_gridsize.py $FLAGS 2>&1 | tee "$OUTFILE"
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Done → $OUTFILE"
else
    echo "❌ Failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
