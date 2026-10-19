# PLATO: Pointer-network Actor with Team-aware Attention for Open systems

## Citation

If you use this code, please cite:

```bibtex
@article{abadi2026plato,
  title={PLATO: Pointer Learner for Agent and Task Openness},
  author={Abadi, Alireza Saleh and Soh, Leen-Kiat and Redder, Daniel Alan and Eck, Adam and Doshi, Prashant},
  journal={arXiv preprint arXiv:2607.25082},
  year={2026}
}
```

PLATO is a multi-agent reinforcement learning method for **open agent systems**, where
the set of active agents and the set of available tasks change online. It combines:

1. a **pointer-network actor** that outputs a distribution directly over the *current* task
   set, conditioned on a permutation-invariant summary of the *current* team, and
2. a **centralized GNN critic** that evaluates open global states via a bipartite
   agent–task graph whose structure changes with system composition.

This design avoids fixed action indexing, padding-based bounds, and retraining to handle
openness. The benchmark domain here is **wildfire** from
[free-range-zoo](https://github.com/oasys-mas/dev-free-range-zoo).

---

## 1. Repository layout

| Path | Purpose |
|------|---------|
| `PLATO_wildfire.py` | Train PLATO (and its ablations) on wildfire |
| `DGN_wildfire.py`, `dicg_de_wildfire.py`, `MOHITO_wildfire.py`, `DEEPSET_wildfire.py` | Train the baselines |
| `execution/execute.py` | Evaluate trained checkpoints and the naive baselines |
| `execution/*_evaluator.py` | Per-method evaluators used by `execute.py` |
| `utils/helper.py` | Command-line arguments (`parse_args`) and config loading |
| `configs.json` | All hyperparameters, seeds, setups, and evaluation settings |
| `experiment_logger_postgres.py` / `experiment_logger_sqlite.py` | Result loggers (see §6) |
| `DB/analyze_results_by_gridsize.py` | Aggregate results into native / zero-shot tables |
| `slurms/` | Example SLURM scripts for training, evaluation, and the DB server |
| `checkpoints/` | Trained model checkpoints (distributed separately, see §8) |

---

## 2. Installation

The environment is pinned in `requirements.txt`. It installs `free_range_zoo` directly
from GitHub at the exact commit the experiments used, so the environment is reproducible
from a single file.

```bash
# Python 3.12
conda create -n PLATO_ENV python=3.12 -y
conda activate PLATO_ENV

pip install -r requirements.txt
```

`requirements.txt` lists the core runtime dependencies; the exact, fully-pinned
environment used for the paper is in `requirements-lock.txt`. Key versions:

| Package | Version |
|---------|---------|
| python | 3.12 |
| torch | 2.6.0+cu124 (CUDA 12.4) |
| numpy | 1.26.4 |
| SQLAlchemy | 2.0.49 |
| dataset | 2.0.0 |
| psycopg2 | (PostgreSQL driver) |
| scipy | (significance tests) |
| free_range_zoo | git `dd1b306` (wildfire environment) |

> **CUDA note.** `torch==2.6.0+cu124` targets CUDA 12.4. For a CPU-only or different-CUDA
> machine, install the matching PyTorch build first, then `pip install -r requirements.txt`.
> Training and evaluation both run on CPU or GPU (set `General_Config.DEVICE` in `configs.json`).

---

## 3. Configuration (`configs.json`)

Everything is driven by `configs.json`:

- `General_Config` — global settings: `DEVICE`, `DB` (`"postgres"` or `"sqlite"`), seeds
  (training / validation / execution), `MAX_STEPS` (episode horizon = 100),
  `SAVE_EVERY` (checkpoint stride = 500), and `SETUPS` (the list of wildfire configurations).
- `PLATO` — actor/critic hyperparameters (learning rates, PPO clip, hidden dims, etc.).
- `DGN`, `DICG`, `MOHITO`, `DEEPSET` — baseline hyperparameters.
- `Eval` — evaluation settings: `checkpoint_folder`, `checkpoint_frequency` (1000),
  `starting_episode`, `max_episode` (30000), `episodes_per_run` (50).

A **setup** is named `setup{0..3}_{grid}` (e.g. `setup2_3x3`). The four setups vary the
openness and dynamics (fire spread, ignition, suppressant refill); grids are `2x3`, `3x3`,
`4x4` (and `5x5`/`6x6` as unseen zero-shot targets).

---

## 4. Training

Train one configuration at a time. The variant is selected by `--pointer_policy`
(`pointer_mlp` / `pointer_lstm`) and `--attention_type` (`additive` / `dotproduct`):

```bash
# PLATO (main model): pointer-MLP + additive attention, setup2 on the 3x3 grid
python PLATO_wildfire.py \
    --pointer_policy pointer_mlp \
    --attention_type additive \
    --setting_type setup2_3x3 \
    --grid_size 3
```

Ablations are expressed through the policy name (e.g. `pointer_mlp_noteamstats`,
`pointer_mlp_nopointer`) and the `--filter_nognn` flag for the no-GNN-critic critic variant.

Baselines use the same argument interface:

```bash
python DGN_wildfire.py        --setting_type setup2_3x3 --grid_size 3
python dicg_de_wildfire.py    --setting_type setup2_3x3 --grid_size 3
python MOHITO_wildfire.py     --setting_type setup2_3x3 --grid_size 3
python DEEPSET_wildfire.py    --setting_type setup2_3x3 --grid_size 3
```

> **MOHITO baseline** additionally needs the `mohito` package (vendored under
> `baselines/mohito`). PLATO and the other baselines do not depend on it.

Checkpoints are written to `checkpoints/` every `SAVE_EVERY` (500) episodes, up to
`EPISODES` (30000). See `slurms/train_3x3.sh` for a complete SLURM array that trains
every (method, variant, setup) on one grid.

---

## 5. Evaluation

Evaluation runs `episodes_per_run` (50) episodes per checkpoint with **sampled** (stochastic)
action selection, scanning checkpoints at `checkpoint_frequency` (every 1000 episodes).

```bash
# Native: evaluate PLATO (main) on the 3x3 grid, all setups in configs.json
python execution/execute.py --type smart --baseline PLATO \
    --grid_size 3 --domain wildfire \
    --pointer_policy mlp --attention_type additive

# Other smart methods
python execution/execute.py --type smart --baseline DGN    --grid_size 3 --domain wildfire
python execution/execute.py --type smart --baseline DICG   --grid_size 3 --domain wildfire
python execution/execute.py --type smart --baseline MOHITO --grid_size 3 --domain wildfire
python execution/execute.py --type smart --baseline DEEPSET --grid_size 3 --domain wildfire

# Naive baselines (NOOP / Random / Weakest / Strongest), all setups and grids
python execution/execute.py --type naive --baseline NOOP --domain wildfire
```

**Zero-shot (cross-grid) evaluation** — load checkpoints trained on `--grid_size` and
evaluate on an unseen target grid:

```bash
# Trained on 3x3, evaluated on the 4x4 zero-shot setups
python execution/execute.py --type smart --baseline PLATO \
    --grid_size 3 --zero_shot --zero_shot_target_grids_size 4 \
    --zero_shot_mode zs --no-load_from_pkl --domain wildfire \
    --pointer_policy mlp --attention_type additive
```

> Note: training uses `--pointer_policy pointer_mlp`, while evaluation uses the short form
> `--pointer_policy mlp`. See `slurms/execute_*.sh` for the full evaluation matrix
> (native 2x3/3x3/4x4 and the four zero-shot grid pairs).

---

## 6. Results database — why PostgreSQL

Each evaluated episode logs far more than a scalar return. The schema records, per episode
and **per step**, the agents' actions, suppressant levels, and every fire's intensity and
fuel (tables `wildfire_episodes`, `wildfire_steps`, `wildfire_step_agents`,
`wildfire_step_fires`), alongside the per-checkpoint aggregates in `eval_results`. This
fine-grained logging is what enables:

- **statistical tests** (paired / effect-size comparisons across methods, e.g. the `dz`
  columns in the analysis tables),
- **cross-checkpoint analysis** (30 checkpoints × 50 episodes per setup), and
- **highly parallel evaluation** — dozens of SLURM array tasks writing concurrently.

A single-file SQLite database cannot take concurrent writers safely, so for the full
parallel sweep we use **PostgreSQL**. Set `General_Config.DB` accordingly:

- `"postgres"` — for parallel sweeps and the fine-grained analysis above.
- `"sqlite"` — a zero-setup single-process fallback (`experiment_logger_sqlite.py`,
  writes a local `.db` file); fine for a handful of sequential runs.

### Starting PostgreSQL

The database runs from the official `postgres:11` container (via Apptainer/Docker). An
example server launcher is in `slurms/postgres_setup.sh`; it writes connection details to
`DB/postgres_env.sh`, which training/evaluation jobs then `source`. The logger reads
`POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_DB` from the environment and a
password file.

```bash
# start the server (SLURM example)
sbatch slurms/postgres_setup.sh
# then, in each job:
source DB/postgres_env.sh
```

---

## 7. Analysis / result tables

`DB/analyze_results_by_gridsize.py` aggregates the database into the paper's tables
(per grid, per setup; Episode Return plus putouts, burnouts, fires, etc.), selecting
checkpoints and reporting significance markers.

```bash
source DB/postgres_env.sh   # (or set DB="sqlite" in configs.json)

python DB/analyze_results_by_gridsize.py --section native          # native-grid table
python DB/analyze_results_by_gridsize.py --section zs              # zero-shot table
python DB/analyze_results_by_gridsize.py --ablation                # all 7 PLATO variants
python DB/analyze_results_by_gridsize.py --section native --latex  # LaTeX output
```

`slurms/run_analysis.sh` produces all of the above (plain + LaTeX) in one array job.

---

## 8. Checkpoints

Trained checkpoints are **not distributed with this repository** (they are large and are
kept in the authors' archive). To get results you can either:

- **train your own** checkpoints by following §4 (they are written to `checkpoints/`), or
- request the trained checkpoints from the authors.

Place them in any directory and point `Eval.checkpoint_folder` in `configs.json` at it.
Each checkpoint folder is named `{METHOD}_wildfire_{variant}_{setup}__{GRID}` and contains
per-episode checkpoint files (every 500 episodes, up to 30000).

---

## 9. Reproducing the paper numbers

- Evaluation is **sampled** (stochastic), not greedy. Returns therefore reproduce within
  sampling noise rather than bit-for-bit; saturated low-return setups match exactly.
- Checkpoints are scanned at `checkpoint_frequency = 1000` (episodes 1000…30000 = 30 per run).
- Seeds for training, validation, and execution are fixed in `General_Config`
  (`Global_Seed = 42`, and the incremental validation/execution seeds).

---

## 10. Citation

If you use this code, please cite the PLATO paper. (See `CITATION` / the paper for the
BibTeX entry.)
