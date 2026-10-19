import numpy as np
from scipy import stats
from tabulate import tabulate
from collections import defaultdict
import sys
import os
import argparse
import traceback


def _excepthook(exc_type, exc_value, exc_tb):
    print("\n" + "=" * 80, flush=True)
    print("[UNCAUGHT EXCEPTION]", flush=True)
    print(f"  Type : {exc_type.__name__}", flush=True)
    print(f"  Value: {exc_value}", flush=True)
    print(f"  POSTGRES_HOST = {os.environ.get('POSTGRES_HOST', 'NOT SET')}", flush=True)
    print(f"  POSTGRES_PORT = {os.environ.get('POSTGRES_PORT', 'NOT SET')}", flush=True)
    print(f"  POSTGRES_USER = {os.environ.get('POSTGRES_USER', 'NOT SET')}", flush=True)
    print(f"  POSTGRES_DB   = {os.environ.get('POSTGRES_DB',   'NOT SET')}", flush=True)
    print("=" * 80, flush=True)
    traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stdout)
    sys.stdout.flush()


sys.excepthook = _excepthook

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.experiment_manager import load_config

general_config = load_config("General_Config")

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--latex', action='store_true', help='Output tables in LaTeX format')
parser.add_argument('--ablation', action='store_true', help='Compare all PLATO variants')
parser.add_argument('--zs_mode', type=str, choices=['fixteam', 'fixteam_transition', 'normal'], default=None,
                    help='Filter zero-shot results by mode: normal (original), fixteam, or fixteam_transition')
parser.add_argument('--settings', type=str, nargs='+', default=None,
                    help='Filter by specific setting types (e.g. --settings AO TO FO fully-closed)')
parser.add_argument('--domain', type=str, choices=['wildfire', 'rideshare'], default='wildfire',
                    help='Environment domain: wildfire or rideshare')
parser.add_argument('--min_episode', type=int, default=0,
                    help='Only consider checkpoints at or after this training episode (0 = no filter)')
parser.add_argument('--section', type=str, choices=['native', 'zs', 'all'], default='all',
                    help='Which section to run: native only, zs only, or all (default)')
args = parser.parse_args()

# Significance levels
ALPHA_NORMAL = 0.05 / 8      # native: PLATO mlp_add + DGN + MOHITO + DICG + 4 naive = 8 models
ALPHA_ZEROSHOT = 0.05 / 4    # zero-shot: PLATO + DGN + MOHITO + DICG = 4 models
ALPHA_ABLATION = 0.05 / 7    # ablation: 7 PLATO variants (mlp/lstm × add/dot + noteamstats + nopointer + nognn)

# the number of runs to include
NUM_RUNS = 50

# only consider checkpoints at or after this training episode
MIN_EPISODE = args.min_episode

# Ablation mode: if True, compare all PLATO variants against each other
# If False, only show mlp additive and ignore other variants in p-test
ablation = args.ablation

# LaTeX output mode: if True, output tables in LaTeX format
latex = args.latex

# Which sections to run
run_native = args.section in ('native', 'all')
run_zs = args.section in ('zs', 'all')

# Variations to ignore in p-test comparisons (only compare against mlp additive)
# When ablation=True, we don't ignore any PLATO variations
ignore_variations_for_ptest = [] if ablation else ['lstm', 'dotproduct', 'noteamstats', 'nopointer', 'nognn']

# Connect to database based on config
if general_config.DB == "postgres":
    import psycopg2
    conn = psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'localhost'),
        port=os.environ.get('POSTGRES_PORT', '5432'),
        user=os.environ.get('POSTGRES_USER', 'postgres'),
        password=os.environ.get('POSTGRES_PASSWORD', None),
        database=os.environ.get('POSTGRES_DB', 'mydb')
    )
    cursor = conn.cursor()
    USE_POSTGRES = True
else:
    import sqlite3
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    USE_POSTGRES = False


def exec_query(query, params=None):
    """Execute query with correct placeholder/boolean syntax for SQLite/Postgres."""
    if not USE_POSTGRES:
        query = query.replace('%s', '?')
        query = query.replace('FALSE', '0').replace('TRUE', '1')
    else:
        # PostgreSQL: zero_shot column is text, cast before comparing to boolean
        query = query.replace('zero_shot = FALSE', 'zero_shot::boolean = FALSE')
        query = query.replace('zero_shot = TRUE', 'zero_shot::boolean = TRUE')
    if params is not None:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    return cursor.fetchall()


def grid_label_for(grid_size):
    """Dynamic grid label."""
    mapping = {2: '2x3', 3: '3x3', 4: '4x4', 5: '5x5', 6: '6x6'}
    return mapping.get(grid_size, f'{grid_size}x{grid_size}')


def select_best_eval_id_for_experiments(exp_ids):
    """
    Select best eval_id across all checkpoints across all experiments in exp_ids.
    Criterion: max mean reward over last NUM_RUNS runs, episode >= MIN_EPISODE.
    Postgres: single CTE query. SQLite: original loop.
    """
    if not exp_ids:
        return None, None

    if USE_POSTGRES:
        cursor.execute("""
            WITH last_n AS (
                SELECT
                    er.id          AS eval_id,
                    ru.reward,
                    ROW_NUMBER() OVER (PARTITION BY er.id ORDER BY ru.id DESC) AS rn
                FROM eval_results er
                JOIN eval_runs ru ON ru.eval_id = er.id
                WHERE er.experiment_id = ANY(%s)
                  AND (er.episode IS NULL OR er.episode::integer >= %s)
            ),
            best AS (
                SELECT eval_id, AVG(reward) AS avg_reward
                FROM last_n
                WHERE rn <= %s
                GROUP BY eval_id
            )
            SELECT eval_id, avg_reward
            FROM best
            ORDER BY avg_reward DESC
            LIMIT 1
        """, (list(exp_ids), MIN_EPISODE, NUM_RUNS))
        rows = cursor.fetchall()
        if not rows:
            return None, None
        return rows[0][0], float(rows[0][1])

    # SQLite fallback
    best_eval_id = None
    best_avg_reward = None
    for exp_id in exp_ids:
        eval_rows = exec_query("""
            SELECT id, episode FROM eval_results WHERE experiment_id = %s
        """, (exp_id,))
        for eval_id, episode in eval_rows:
            if episode is not None and episode < MIN_EPISODE:
                continue
            avg_rows = exec_query("""
                SELECT AVG(reward) FROM (
                    SELECT reward FROM eval_runs WHERE eval_id = %s ORDER BY id DESC LIMIT %s
                ) AS last_runs
            """, (eval_id, NUM_RUNS))
            if not avg_rows or avg_rows[0][0] is None:
                continue
            avg_reward = avg_rows[0][0]
            if best_eval_id is None or avg_reward > best_avg_reward:
                best_eval_id = eval_id
                best_avg_reward = avg_reward
    return best_eval_id, best_avg_reward


# Define models and settings (domain-aware)
domain = args.domain

if domain == 'rideshare':
    naive_baselines = ["NOOP", "Random", "FIFO", "GreedyTFocus", "GreedyTGlobal"]
    metrics = ['reward', 'pickup', 'dropoff', 'accept', 'noop', 'action']
    metric_names = {
        'reward': 'Reward',
        'pickup': 'Pickups',
        'dropoff': 'Dropoffs',
        'accept': 'Accepts',
        'noop': 'Noops',
        'action': 'Total Actions',
    }
    higher_is_better = {'reward', 'pickup', 'dropoff', 'accept'}
    lower_is_better = {'noop', 'action'}
    latex_metric_labels = {
        'reward': 'reward',
        'pickup': 'pickups',
        'dropoff': 'dropoffs',
        'accept': 'accepts',
        'noop': 'noop',
        'action': 'actions',
    }
    latex_metric_captions = {
        'reward': 'Episode return',
        'pickup': 'Pickups',
        'dropoff': 'Dropoffs',
        'accept': 'Accepts',
        'noop': 'NOOP percentage',
        'action': 'Total actions',
    }
else:
    naive_baselines = ["NOOP", "Random", "Weakest", "Strongest"]
    metrics = ['reward', 'burnout', 'putout', 'noop', 'action', 'reward_per_fight', 'new_fire_ignited', 'zero_suppressant']
    metric_names = {
        'reward': 'Reward',
        'burnout': 'Burnouts',
        'putout': 'Putouts',
        'noop': 'Noops',
        'action': 'Total Actions',
        'reward_per_fight': 'Reward Per Fight',
        'new_fire_ignited': 'New Fires Ignited',
        'zero_suppressant': 'Zero Suppressant Count'
    }
    higher_is_better = {'reward', 'putout', 'reward_per_fight'}
    lower_is_better = {'burnout', 'noop', 'new_fire_ignited', 'zero_suppressant', 'action'}
    latex_metric_labels = {
        'reward': 'reward',
        'burnout': 'burnouts',
        'putout': 'putouts',
        'noop': 'noop',
        'action': 'actions',
        'reward_per_fight': 'rpf',
        'new_fire_ignited': 'newfire',
        'zero_suppressant': 'zerosup'
    }
    latex_metric_captions = {
        'reward': 'Episode return',
        'burnout': 'Burnouts',
        'putout': 'Putouts',
        'noop': 'NOOP percentage',
        'action': 'Total actions',
        'reward_per_fight': 'Reward per fight',
        'new_fire_ignited': 'New fires ignited',
        'zero_suppressant': 'Zero suppressant count'
    }

plato_policy_variations = ["mlp", "lstm", "mlp_noteamstats", "mlp_nopointer"]
plato_attention_variations = ["additive", "dotproduct"]

def format_latex_value(mean, std, is_significant, metric, min_dz=None):
    if metric == 'noop':
        val = f"{mean:.2f}$\\pm${std:.2f}\\%"
    else:
        val = f"{mean:.2f}$\\pm${std:.2f}"
    if is_significant:
        dz_str = f"{{\\tiny($d_z{{\\geq}}{min_dz:.2f}$)}}" if min_dz is not None else ""
        val = f"\\textbf{{{val}}}{dz_str}"
    return val

def print_latex_table(results, setup_list, metric, grid_str, setup_significant, is_zeroshot=False):
    if not results:
        return
    
    grid_display = grid_str.replace('x', r'\times ')
    metric_label = latex_metric_labels.get(metric, metric)
    metric_caption = latex_metric_captions.get(metric, metric_names[metric])
    
    zs_prefix = 'zs_' if is_zeroshot else ''
    num_setups = len(setup_list)
    total_cols = 1 + num_setups  # Model + setup columns
    
    print(f"% {metric_caption} ({grid_str})")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{2.5pt}")
    print(r"\renewcommand{\arraystretch}{1.03}")
    col_spec = 'l' + '|l' * num_setups
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print(r"\toprule")
    print(f"\\multicolumn{{{total_cols}}}{{c}}{{${grid_display}$}} \\\\")
    print(r"\midrule")
    # Dynamic column headers from setup_list
    header_cells = ['Model']
    for i, setup in enumerate(setup_list):
        sep_char = '|' if i < num_setups - 1 else ''
        header_cells.append(f"\\multicolumn{{1}}{{c{sep_char}}}{{{setup}}}")
    print(' & '.join(header_cells) + r' \\')
    print(r"\midrule")
    
    for row in results:
        model = row['Model']
        cells = [model]
        for i, setup in enumerate(setup_list):
            val = row.get(setup, '-')
            if val != '-':
                cells.append(val)
            else:
                cells.append('-')
        
        sep = ' & '
        line = sep.join(cells) + r' \\'
        print(line)
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    alpha_val = ALPHA_ABLATION if ablation else (ALPHA_ZEROSHOT if is_zeroshot else ALPHA_NORMAL)
    print(f"\\caption{{{metric_caption} (mean $\\pm$ std) on the ${grid_display}$ grid. Bold indicates statistically significant best performance within each setup (one-sided paired $t$-test; $\\alpha={alpha_val:.5f}$, Bonferroni-corrected).}}")
    print(f"\\label{{tab:{zs_prefix}{metric_label}_{grid_str}}}")
    print(r"\end{table}")
    print()

try:
    db_grid_sizes = [row[0] for row in exec_query("""
        SELECT DISTINCT grid_size
        FROM experiments
        WHERE run_type = 'execution'
          AND (domain = %s OR (domain IS NULL AND %s = 'wildfire'))
        ORDER BY grid_size
    """, (domain, domain))]
except Exception as e:
    print(f"\n[FATAL] Could not connect to / query database.", flush=True)
    print(f"  POSTGRES_HOST = {os.environ.get('POSTGRES_HOST', 'NOT SET')}", flush=True)
    print(f"  POSTGRES_PORT = {os.environ.get('POSTGRES_PORT', 'NOT SET')}", flush=True)
    print(f"  POSTGRES_USER = {os.environ.get('POSTGRES_USER', 'NOT SET')}", flush=True)
    print(f"  POSTGRES_DB   = {os.environ.get('POSTGRES_DB',   'NOT SET')}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("=" * 120)
print(f"EXPERIMENT RESULTS ANALYSIS BY GRID SIZE — {domain.upper()}")
print("=" * 120)

# =============================================================================
# SECTION 1: NATIVE RESULTS BY GRID SIZE
# =============================================================================
for grid_size in (db_grid_sizes if run_native else []):
    grid_str = grid_label_for(grid_size)
    print(f"\n\n{'='*120}")
    print(f"GRID SIZE: {grid_str}")
    print(f"{'='*120}\n")

    experiments = []

    # PLATO variations (exclude nognn here; fetched separately below)
    for policy_var in plato_policy_variations:
        for attention_var in plato_attention_variations:
            experiments.extend(exec_query("""
                SELECT DISTINCT
                    e.id,
                    e.model_family,
                    e.name || '_' || COALESCE(e.variation1, '') || '_' || COALESCE(e.variation2, '') AS exp_name,
                    e.setting_type,
                    e.variation1,
                    e.variation2
                FROM experiments e
                WHERE e.run_type = 'execution' AND e.grid_size = %s
                  AND e.model_family = 'PLATO'
                  AND e.variation1 = %s AND e.variation2 = %s
                  AND LOWER(e.name) NOT LIKE '%%nognn%%'
                  AND (e.zero_shot IS NULL OR e.zero_shot = FALSE)
                  AND (e.domain = %s OR (e.domain IS NULL AND %s = 'wildfire'))
                ORDER BY e.setting_type
            """, (grid_size, policy_var, attention_var, domain, domain)))

    # PLATO nognn ablation variant (variation1/2 same as mlp+additive but name contains 'nognn')
    for row in exec_query("""
        SELECT DISTINCT
            e.id,
            e.model_family,
            e.name || '_mlp_additive_nognn' AS exp_name,
            e.setting_type,
            e.variation1,
            e.variation2
        FROM experiments e
        WHERE e.run_type = 'execution' AND e.grid_size = %s
          AND e.model_family = 'PLATO'
          AND e.variation1 = 'mlp' AND e.variation2 = 'additive'
          AND LOWER(e.name) LIKE '%%nognn%%'
          AND (e.zero_shot IS NULL OR e.zero_shot = FALSE)
          AND (e.domain = %s OR (e.domain IS NULL AND %s = 'wildfire'))
        ORDER BY e.setting_type
    """, (grid_size, domain, domain)):
        exp_id, model_family, exp_name, setting_type, var1, var2 = row
        experiments.append((exp_id, model_family, exp_name, setting_type, 'mlp_nognn', var2))

    # Other smart algorithms (handle naming variants)
    smart_algo_names = ['DICG', 'DICG_wildfire', 'DGN', 'MOHITO', 'MOHITO_wildfire']
    if domain == 'rideshare':
        smart_algo_names = ['DICG', 'DGN', 'MOHITO']  # rideshare uses clean names
    for algo in smart_algo_names:
        experiments.extend(exec_query("""
            SELECT DISTINCT
                e.id,
                e.model_family,
                e.name || '_' || COALESCE(e.variation1, '') || '_' || COALESCE(e.variation2, '') AS exp_name,
                e.setting_type,
                e.variation1,
                e.variation2
            FROM experiments e
            WHERE e.run_type = 'execution' AND e.grid_size = %s
              AND e.model_family = %s
              AND (e.zero_shot IS NULL OR e.zero_shot = FALSE)
              AND (e.domain = %s OR (e.domain IS NULL AND %s = 'wildfire'))
            ORDER BY e.setting_type
        """, (grid_size, algo, domain, domain)))

    # Baselines (skip for grid_size >= 4 on wildfire only)
    skip_baselines = (grid_size >= 4) if domain == 'wildfire' else False
    if not skip_baselines:
        experiments.extend(exec_query("""
            SELECT DISTINCT
                e.id,
                e.model_family,
                e.name || '_' || COALESCE(e.variation1, '') || '_' || COALESCE(e.variation2, '') AS exp_name,
                e.setting_type,
                e.variation1,
                e.variation2
            FROM experiments e
            WHERE e.run_type = 'execution' AND e.grid_size = %s
              AND e.model_family = 'Baseline'
              AND (e.zero_shot IS NULL OR e.zero_shot = FALSE)
              AND (e.domain = %s OR (e.domain IS NULL AND %s = 'wildfire'))
            ORDER BY e.setting_type
        """, (grid_size, domain, domain)))

    # Organize experiments
    setting_types = []
    model_combos = []
    exp_names = defaultdict(lambda: defaultdict(list))  # setting_type -> (model_family,var1,var2) -> [(exp_id, exp_name)]

    for exp_id, model_family, exp_name, setting_type, variation1, variation2 in experiments:
        if setting_type not in setting_types:
            setting_types.append(setting_type)

        model_combo = (model_family, variation1, variation2)
        if model_combo not in model_combos:
            model_combos.append(model_combo)

        if (exp_id, exp_name) not in exp_names[setting_type][model_combo]:
            exp_names[setting_type][model_combo].append((exp_id, exp_name))

    # Aggregate best checkpoint per (setting_type, model_combo)
    def _make_combo_data():
        d = {'all_rewards': {}, 'all_noops': {}, 'all_actions': {}, 'count': 0}
        if domain == 'rideshare':
            d.update({'all_pickups': {}, 'all_dropoffs': {}, 'all_accepts': {}})
        else:
            d.update({'all_burnouts': {}, 'all_putouts': {}, 'all_reward_per_fight': {},
                      'all_new_fire_ignited': {}, 'all_zero_suppressant': {}})
        return d

    combinations_data = defaultdict(_make_combo_data)

    for setting_type in setting_types:
        for model_combo in model_combos:
            exp_list = exp_names[setting_type][model_combo]
            if not exp_list:
                continue

            exp_ids = [eid for (eid, _) in exp_list]
            best_eval_id, best_avg_reward = select_best_eval_id_for_experiments(exp_ids)
            if best_eval_id is None:
                continue

            model_family, variation1, variation2 = model_combo
            combo_key = (model_family, variation1, variation2, setting_type)

            if domain == 'rideshare':
                runs = exec_query("""
                    SELECT seed, reward, pickups, dropoffs, episode_accept_count,
                           episode_noop_count, episode_pick_count, episode_drop_count
                    FROM eval_runs
                    WHERE eval_id = %s
                    ORDER BY seed ASC
                    LIMIT %s
                """, (best_eval_id, NUM_RUNS))

                for seed, reward, pickups, dropoffs, accept_count, noop_count, pick_count, drop_count in runs:
                    if seed is not None:
                        combinations_data[combo_key]['all_rewards'][seed] = reward
                        combinations_data[combo_key]['all_pickups'][seed] = pickups or 0
                        combinations_data[combo_key]['all_dropoffs'][seed] = dropoffs or 0
                        combinations_data[combo_key]['all_accepts'][seed] = accept_count or 0
                        combinations_data[combo_key]['all_noops'][seed] = noop_count or 0
                        total_actions = (accept_count or 0) + (pick_count or 0) + (drop_count or 0) + (noop_count or 0)
                        combinations_data[combo_key]['all_actions'][seed] = total_actions
                        combinations_data[combo_key]['count'] += 1
            else:
                runs = exec_query("""
                    SELECT seed, reward, burnouts, putouts, episode_noop_count,
                           episode_fight_count, reward_per_fight, new_fire_ignited_count,
                           episode_zero_suppressant_count
                    FROM eval_runs
                    WHERE eval_id = %s
                    ORDER BY seed ASC
                    LIMIT %s
                """, (best_eval_id, NUM_RUNS))

                for seed, reward, burnouts, putouts, noop_count, fight_count, reward_per_fight, new_fire_ignited, zero_suppressant in runs:
                    if seed is not None:
                        combinations_data[combo_key]['all_rewards'][seed] = reward
                        combinations_data[combo_key]['all_burnouts'][seed] = burnouts
                        combinations_data[combo_key]['all_putouts'][seed] = putouts
                        combinations_data[combo_key]['all_noops'][seed] = noop_count
                        combinations_data[combo_key]['all_actions'][seed] = (fight_count or 0) + (noop_count or 0)
                        combinations_data[combo_key]['all_reward_per_fight'][seed] = reward_per_fight
                        combinations_data[combo_key]['all_new_fire_ignited'][seed] = new_fire_ignited
                        combinations_data[combo_key]['all_zero_suppressant'][seed] = zero_suppressant
                        combinations_data[combo_key]['count'] += 1

    # Compute means/stds per combination
    models_data = defaultdict(list)
    for (model_family, variation1, variation2, setting_type), data in combinations_data.items():
        if not data['all_rewards']:
            continue

        actual_model = model_family
        actual_variation = f"{variation1 if variation1 else ''} {variation2 if variation2 else ''}".strip()
        if model_family == 'Baseline':
            actual_model = variation1 if variation1 else 'Baseline'
            actual_variation = "NAIVE"

        # Per-seed noop percentage (correct std, avoids dividing count-std by mean-action)
        noop_pcts = [
            (data['all_noops'][s] / data['all_actions'][s] * 100) if data['all_actions'].get(s, 0) > 0 else 0.0
            for s in data['all_noops']
        ]
        entry = {
            'actual_variation': actual_variation,
            'setting_type': setting_type,
            'reward_mean': float(np.mean(list(data['all_rewards'].values()))),
            'reward_std': float(np.std(list(data['all_rewards'].values()), ddof=1)),
            'noop_mean': float(np.mean(noop_pcts)) if noop_pcts else 0.0,
            'noop_std': float(np.std(noop_pcts, ddof=1)) if len(noop_pcts) > 1 else 0.0,
            'action_mean': float(np.mean(list(data['all_actions'].values()))),
            'action_std': float(np.std(list(data['all_actions'].values()), ddof=1)),
        }
        if domain == 'rideshare':
            entry.update({
                'pickup_mean': float(np.mean(list(data['all_pickups'].values()))),
                'pickup_std': float(np.std(list(data['all_pickups'].values()), ddof=1)),
                'dropoff_mean': float(np.mean(list(data['all_dropoffs'].values()))),
                'dropoff_std': float(np.std(list(data['all_dropoffs'].values()), ddof=1)),
                'accept_mean': float(np.mean(list(data['all_accepts'].values()))),
                'accept_std': float(np.std(list(data['all_accepts'].values()), ddof=1)),
            })
        else:
            entry.update({
                'burnout_mean': float(np.mean(list(data['all_burnouts'].values()))),
                'burnout_std': float(np.std(list(data['all_burnouts'].values()), ddof=1)),
                'putout_mean': float(np.mean(list(data['all_putouts'].values()))),
                'putout_std': float(np.std(list(data['all_putouts'].values()), ddof=1)),
                'reward_per_fight_mean': float(np.mean(list(data['all_reward_per_fight'].values()))),
                'reward_per_fight_std': float(np.std(list(data['all_reward_per_fight'].values()), ddof=1)),
                'new_fire_ignited_mean': float(np.mean(list(data['all_new_fire_ignited'].values()))),
                'new_fire_ignited_std': float(np.std(list(data['all_new_fire_ignited'].values()), ddof=1)),
                'zero_suppressant_mean': float(np.mean(list(data['all_zero_suppressant'].values()))),
                'zero_suppressant_std': float(np.std(list(data['all_zero_suppressant'].values()), ddof=1)),
            })
        models_data[actual_model].append(entry)

    # Dynamic: use setting_types actually found in DB for this grid_size
    # Exclude _zs suffixed settings — those belong to zero-shot section only
    native_setting_types = [s for s in setting_types if not s.endswith('_zs')]
    if args.settings:
        setup_list = [s for s in args.settings if s in native_setting_types]
    else:
        setup_list = sorted(native_setting_types)

    for metric in metrics:
        print(f"\n{'-'*200}")
        print(f"{metric_names[metric].upper()}")
        print(f"{'-'*200}\n")

        # Reorganize data by model and variation, then by setup
        model_var_data = defaultdict(lambda: defaultdict(dict))
        for model_name, entries in models_data.items():
            for exp_data in entries:
                var_key = exp_data['actual_variation']
                setup_type = exp_data['setting_type']
                # noop_mean/noop_std already store per-seed percentages
                mean_val = exp_data[f'{metric}_mean']
                std_val = exp_data[f'{metric}_std']
                model_var_data[model_name][var_key][setup_type] = {'mean': mean_val, 'std': std_val}

        # Raw data for tests (seed -> value dicts)
        setup_raw_data = defaultdict(dict)
        for (model_family, variation1, variation2, setting_type), data in combinations_data.items():
            actual_model = model_family
            actual_variation = f"{variation1 if variation1 else ''} {variation2 if variation2 else ''}".strip()
            if model_family == 'Baseline':
                actual_model = variation1 if variation1 else 'Baseline'
                actual_variation = "NAIVE"

            model_var_key = f"{actual_model}|{actual_variation}"

            if metric == 'reward':
                raw_values = data['all_rewards']  # {seed: value}
            elif metric == 'burnout':
                raw_values = data.get('all_burnouts', {})
            elif metric == 'putout':
                raw_values = data.get('all_putouts', {})
            elif metric == 'pickup':
                raw_values = data.get('all_pickups', {})
            elif metric == 'dropoff':
                raw_values = data.get('all_dropoffs', {})
            elif metric == 'accept':
                raw_values = data.get('all_accepts', {})
            elif metric == 'noop':
                raw_values = {}
                for seed, noop in data['all_noops'].items():
                    action = data['all_actions'].get(seed, 1)
                    raw_values[seed] = (noop / action * 100) if action > 0 else 0.0
            elif metric == 'action':
                raw_values = data['all_actions']
            elif metric == 'reward_per_fight':
                raw_values = data.get('all_reward_per_fight', {})
            elif metric == 'new_fire_ignited':
                raw_values = data.get('all_new_fire_ignited', {})
            elif metric == 'zero_suppressant':
                raw_values = data.get('all_zero_suppressant', {})
            else:
                raw_values = {}

            if raw_values:
                setup_raw_data[setting_type][model_var_key] = raw_values

        # Significance marking per setup (one-sided paired t-test on matching seeds)
        setup_significant = defaultdict(set)
        setup_dz = defaultdict(dict)   # setup -> model_key -> min Cohen's d_z
        alpha_native = ALPHA_ABLATION if ablation else ALPHA_NORMAL
        for setup in setup_list:
            if setup not in setup_raw_data:
                continue

            filtered_raw_data = {}
            for k, v in setup_raw_data[setup].items():
                # In ablation mode: compare PLATO variants only
                if ablation and not k.startswith('PLATO|'):
                    continue
                if any(ignored in k.lower() for ignored in ignore_variations_for_ptest):
                    continue
                filtered_raw_data[k] = v

            if not filtered_raw_data:
                continue

            setup_means = {k: float(np.mean(list(v.values()))) for k, v in filtered_raw_data.items()}
            if not setup_means:
                continue

            if metric in higher_is_better:
                best_mean = max(setup_means.values())
                best_models = [k for k, v in setup_means.items() if v == best_mean]
            else:
                best_mean = min(setup_means.values())
                best_models = [k for k, v in setup_means.items() if v == best_mean]

            for best_model in best_models:
                best_data = filtered_raw_data[best_model]

                if len(filtered_raw_data) == 1:
                    continue

                is_significant = True
                min_dz = float('inf')
                alternative = 'greater' if metric in higher_is_better else 'less'
                for other_model, other_data in filtered_raw_data.items():
                    if other_model == best_model:
                        continue
                    try:
                        # Pair by matching seeds
                        common_seeds = set(best_data.keys()) & set(other_data.keys())
                        if len(common_seeds) < 2:
                            is_significant = False
                            break
                        best_paired = [best_data[s] for s in sorted(common_seeds)]
                        other_paired = [other_data[s] for s in sorted(common_seeds)]

                        # One-sided paired t-test: best > other (or best < other for lower_is_better)
                        _, p_value = stats.ttest_rel(best_paired, other_paired, alternative=alternative)
                        if np.isnan(p_value):
                            is_significant = False
                            break
                        if p_value >= alpha_native:
                            is_significant = False
                            break

                        # Cohen's d_z (signed toward advantage of best model)
                        if metric in higher_is_better:
                            diffs = [b - o for b, o in zip(best_paired, other_paired)]
                        else:
                            diffs = [o - b for b, o in zip(best_paired, other_paired)]
                        s_d = float(np.std(diffs, ddof=1))
                        if s_d > 0:
                            min_dz = min(min_dz, float(np.mean(diffs)) / s_d)
                    except Exception:
                        is_significant = False
                        break

                if is_significant:
                    setup_significant[setup].add(best_model)
                    if min_dz < float('inf'):
                        setup_dz[setup][best_model] = min_dz

        # Build results table
        results = []

        # PLATO first (if present)
        if 'PLATO' in model_var_data:
            plato_vars = sorted(model_var_data['PLATO'].keys(), key=lambda x: (x.count('dotproduct'), x), reverse=True)
            # When ablation=False, only show mlp additive
            if not ablation:
                plato_vars = [v for v in plato_vars if v.strip() == 'mlp additive']
            for var in plato_vars:
                row = {'Model': 'PLATO', 'Variation': var}
                model_var_key = f"PLATO|{var}"

                for setup in setup_list:
                    if setup in model_var_data['PLATO'][var]:
                        d = model_var_data['PLATO'][var][setup]
                        is_sig = model_var_key in setup_significant.get(setup, set())
                        min_dz = setup_dz.get(setup, {}).get(model_var_key) if is_sig else None
                        if latex:
                            val_str = format_latex_value(d['mean'], d['std'], is_sig, metric, min_dz)
                        else:
                            val_str = f"{d['mean']:.2f} ± {d['std']:.2f}"
                            if metric == 'noop':
                                val_str += '%'
                            if is_sig:
                                dz_str = f" (dz≥{min_dz:.2f})" if min_dz is not None else ""
                                val_str = '* ' + val_str + dz_str
                        row[setup] = val_str
                    else:
                        row[setup] = '-'

                results.append(row)

        # Other smart algos (both naming variants, if present)
        for algo in ['DICG', 'DICG_wildfire', 'DGN', 'MOHITO', 'MOHITO_wildfire']:
            if algo not in model_var_data:
                continue
            for var in sorted(model_var_data[algo].keys()):
                row = {'Model': algo, 'Variation': var}
                model_var_key = f"{algo}|{var}"

                for setup in setup_list:
                    if setup in model_var_data[algo][var]:
                        d = model_var_data[algo][var][setup]
                        is_sig = model_var_key in setup_significant.get(setup, set())
                        min_dz = setup_dz.get(setup, {}).get(model_var_key) if is_sig else None
                        if latex:
                            val_str = format_latex_value(d['mean'], d['std'], is_sig, metric, min_dz)
                        else:
                            val_str = f"{d['mean']:.2f} ± {d['std']:.2f}"
                            if metric == 'noop':
                                val_str += '%'
                            if is_sig:
                                dz_str = f" (dz≥{min_dz:.2f})" if min_dz is not None else ""
                                val_str = '* ' + val_str + dz_str
                        row[setup] = val_str
                    else:
                        row[setup] = '-'

                results.append(row)

        # Baselines
        for baseline in naive_baselines:
            if baseline not in model_var_data:
                continue
            for var in sorted(model_var_data[baseline].keys()):
                row = {'Model': baseline, 'Variation': var}
                model_var_key = f"{baseline}|{var}"

                for setup in setup_list:
                    if setup in model_var_data[baseline][var]:
                        d = model_var_data[baseline][var][setup]
                        is_sig = model_var_key in setup_significant.get(setup, set())
                        min_dz = setup_dz.get(setup, {}).get(model_var_key) if is_sig else None
                        if latex:
                            val_str = format_latex_value(d['mean'], d['std'], is_sig, metric, min_dz)
                        else:
                            val_str = f"{d['mean']:.2f} ± {d['std']:.2f}"
                            if metric == 'noop':
                                val_str += '%'
                            if is_sig:
                                dz_str = f" (dz≥{min_dz:.2f})" if min_dz is not None else ""
                                val_str = '* ' + val_str + dz_str
                        row[setup] = val_str
                    else:
                        row[setup] = '-'

                results.append(row)

        if results:
            if latex:
                print_latex_table(results, setup_list, metric, grid_str, setup_significant, is_zeroshot=False)
            else:
                print(tabulate(results, headers="keys", tablefmt="grid"))
        else:
            print(f"No data for {metric_names[metric]}")
        print()

# =============================================================================
# SECTION 2: ZERO-SHOT (Cross-Grid) RESULTS — wildfire only
# =============================================================================
if not run_zs:
    conn.close()
    sys.exit(0)

if domain == 'rideshare':
    print("\n\nZERO-SHOT: Skipped (not applicable for rideshare domain)")
    conn.close()
    sys.exit(0)

# Determine zero-shot mode filter for display and queries
zs_mode_display = args.zs_mode.upper() if args.zs_mode else "ALL"
print("\n\n" + "=" * 120)
print(f"ZERO-SHOT (CROSS-GRID) RESULTS - Smart Models Only - Mode: {zs_mode_display}")
print("=" * 120)

# Build the mode suffix pattern for SQL LIKE queries
def get_zs_like_pattern(source_grid, target_grid, zs_mode):
    """Build LIKE pattern based on zero-shot mode filter."""
    base = f'from_setup%_{source_grid}_to_setup%_{target_grid}'
    if zs_mode == 'fixteam':
        return base + '_fixteam'
    elif zs_mode == 'fixteam_transition':
        return base + '_fixteam_transition'
    elif zs_mode == 'zs':
        return base + '_zs'
    elif zs_mode == 'bigteam':
        return base + '_bigteam'
    elif zs_mode == 'normal':
        # Match only patterns that DON'T have mode suffix - need exact match
        return base  # Exact match without trailing %
    else:
        # None - match all modes
        return base + '%'

# 4x4 is split into two distinct entries: 'zs' (matched agent count) and 'bigteam' (full native team).
# None means use the global --zs_mode filter from args.
zero_shot_groups = [
    ('2x3', [('3x3', None), ('4x4', 'zs'), ('4x4', 'bigteam'), ('5x5', None), ('6x6', None)]),
    ('3x3', [('2x3', None), ('4x4', 'zs'), ('4x4', 'bigteam'), ('5x5', None), ('6x6', None)]),
    ('4x4', [('2x3', None), ('3x3', None), ('5x5', None)]),
]
zero_shot_combinations = [(src, tgt, mode) for src, tgts in zero_shot_groups for tgt, mode in tgts]

_zs_source_seen = set()
for source_grid, target_grid, combo_mode in zero_shot_combinations:
    # combo_mode overrides global --zs_mode for entries with an explicit mode (e.g. 4x4 zs/bigteam)
    effective_mode = combo_mode if combo_mode is not None else args.zs_mode
    target_label = f"{target_grid} [{combo_mode.upper()}]" if combo_mode else target_grid
    effective_mode_display = effective_mode.upper() if effective_mode else 'ALL'
    latex_grid_str = f"{target_grid}_{combo_mode}" if combo_mode else target_grid
    if source_grid not in _zs_source_seen:
        _zs_source_seen.add(source_grid)
        print(f"\n\n{'#'*120}")
        print(f"ZERO-SHOT GROUP: TRAINED ON {source_grid}")
        print(f"{'#'*120}")
    print(f"\n\n{'='*120}")
    print(f"ZERO-SHOT: {source_grid} -> {target_label} (Mode: {effective_mode_display})")
    print(f"{'='*120}\n")

    # Get the LIKE pattern for this source/target/mode combination
    zs_pattern = get_zs_like_pattern(source_grid, target_grid, effective_mode)
    print(f"  Collecting ZS data: {source_grid} {target_label} ...", flush=True)

    experiments = []

    # PLATO variations
    # PLATO variations (exclude nognn here; fetched separately below)
    for policy_var in plato_policy_variations:
        for attention_var in plato_attention_variations:
            experiments.extend(exec_query("""
                SELECT DISTINCT
                    e.id,
                    e.model_family,
                    e.name || '_' || COALESCE(e.variation1, '') || '_' || COALESCE(e.variation2, '') AS exp_name,
                    e.setting_type,
                    e.variation1,
                    e.variation2,
                    e.zero_shot_target_grids_size_detail
                FROM experiments e
                WHERE e.run_type = 'execution' AND e.zero_shot = TRUE
                  AND e.model_family = 'PLATO'
                  AND e.variation1 = %s AND e.variation2 = %s
                  AND LOWER(e.name) NOT LIKE '%%nognn%%'
                  AND e.zero_shot_target_grids_size_detail LIKE %s
                ORDER BY e.setting_type
            """, (policy_var, attention_var, zs_pattern)))

    # PLATO nognn ablation variant (relabel variation1 as 'mlp_nognn')
    for row in exec_query("""
        SELECT DISTINCT
            e.id,
            e.model_family,
            e.name || '_mlp_additive_nognn' AS exp_name,
            e.setting_type,
            e.variation1,
            e.variation2,
            e.zero_shot_target_grids_size_detail
        FROM experiments e
        WHERE e.run_type = 'execution' AND e.zero_shot = TRUE
          AND e.model_family = 'PLATO'
          AND e.variation1 = 'mlp' AND e.variation2 = 'additive'
          AND LOWER(e.name) LIKE '%%nognn%%'
          AND e.zero_shot_target_grids_size_detail LIKE %s
        ORDER BY e.setting_type
    """, (zs_pattern,)):
        exp_id, model_family, exp_name, setting_type, var1, var2, zs_detail = row
        experiments.append((exp_id, model_family, exp_name, setting_type, 'mlp_nognn', var2, zs_detail))

    # Other smart algorithms
    for algo in ['DICG', 'DICG_wildfire', 'DGN', 'MOHITO', 'MOHITO_wildfire']:
        experiments.extend(exec_query("""
            SELECT DISTINCT
                e.id,
                e.model_family,
                e.name || '_' || COALESCE(e.variation1, '') || '_' || COALESCE(e.variation2, '') AS exp_name,
                e.setting_type,
                e.variation1,
                e.variation2,
                e.zero_shot_target_grids_size_detail
            FROM experiments e
            WHERE e.run_type = 'execution' AND e.zero_shot = TRUE
              AND e.model_family = %s
              AND e.zero_shot_target_grids_size_detail LIKE %s
            ORDER BY e.setting_type
        """, (algo, zs_pattern)))

    # Parse target setting, group experiments
    exp_with_target = []
    setting_types = []
    model_combos = []
    exp_names = defaultdict(lambda: defaultdict(list))  # target_setting -> model_combo -> [(exp_id, exp_name)]

    for exp_id, model_family, exp_name, setting_type, variation1, variation2, zs_detail in experiments:
        if zs_detail and '_to_' in zs_detail:
            target_setting = zs_detail.split('_to_')[1]  # "setupX_YY" or "setupX_YY_fixteam"
            # Strip mode suffix if present (e.g., _fixteam, _fixteam_transition)
            for mode_suffix in ['_fixteam_transition', '_fixteam']:
                if target_setting.endswith(mode_suffix):
                    target_setting = target_setting[:-len(mode_suffix)]
                    break
        else:
            target_setting = setting_type

        exp_with_target.append((exp_id, model_family, exp_name, target_setting, variation1, variation2))

        if target_setting not in setting_types:
            setting_types.append(target_setting)

        model_combo = (model_family, variation1, variation2)
        if model_combo not in model_combos:
            model_combos.append(model_combo)

        if (exp_id, exp_name) not in exp_names[target_setting][model_combo]:
            exp_names[target_setting][model_combo].append((exp_id, exp_name))

    # Aggregate best checkpoint per (target_setting, model_combo)
    combinations_data = defaultdict(lambda: {
        'all_rewards': {},  # seed -> value
        'all_burnouts': {},
        'all_putouts': {},
        'all_noops': {},
        'all_actions': {},
        'all_reward_per_fight': {},
        'all_new_fire_ignited': {},
        'all_zero_suppressant': {},
        'count': 0
    })

    for target_setting in setting_types:
        for model_combo in model_combos:
            exp_list = exp_names[target_setting][model_combo]
            if not exp_list:
                continue

            exp_ids = [eid for (eid, _) in exp_list]
            best_eval_id, best_avg_reward = select_best_eval_id_for_experiments(exp_ids)
            if best_eval_id is None:
                continue

            model_family, variation1, variation2 = model_combo
            combo_key = (model_family, variation1, variation2, target_setting)

            runs = exec_query("""
                SELECT seed, reward, burnouts, putouts, episode_noop_count,
                       episode_fight_count, reward_per_fight, new_fire_ignited_count,
                       episode_zero_suppressant_count
                FROM eval_runs
                WHERE eval_id = %s
                ORDER BY seed ASC
                LIMIT %s
            """, (best_eval_id, NUM_RUNS))

            for seed, reward, burnouts, putouts, noop_count, fight_count, reward_per_fight, new_fire_ignited, zero_suppressant in runs:
                if seed is not None:
                    combinations_data[combo_key]['all_rewards'][seed] = reward
                    combinations_data[combo_key]['all_burnouts'][seed] = burnouts
                    combinations_data[combo_key]['all_putouts'][seed] = putouts
                    combinations_data[combo_key]['all_noops'][seed] = noop_count
                    combinations_data[combo_key]['all_actions'][seed] = fight_count + noop_count
                    combinations_data[combo_key]['all_reward_per_fight'][seed] = reward_per_fight
                    combinations_data[combo_key]['all_new_fire_ignited'][seed] = new_fire_ignited
                    combinations_data[combo_key]['all_zero_suppressant'][seed] = zero_suppressant
                    combinations_data[combo_key]['count'] += 1

    # Compute means/stds
    models_data = defaultdict(list)
    for (model_family, variation1, variation2, target_setting), data in combinations_data.items():
        if not data['all_rewards']:
            continue

        actual_model = model_family
        actual_variation = f"{variation1 if variation1 else ''} {variation2 if variation2 else ''}".strip()

        noop_pcts = [
            (data['all_noops'][s] / data['all_actions'][s] * 100) if data['all_actions'].get(s, 0) > 0 else 0.0
            for s in data['all_noops']
        ]
        models_data[actual_model].append({
            'actual_variation': actual_variation,
            'setting_type': target_setting,
            'reward_mean': float(np.mean(list(data['all_rewards'].values()))),
            'reward_std': float(np.std(list(data['all_rewards'].values()), ddof=1)),
            'burnout_mean': float(np.mean(list(data['all_burnouts'].values()))),
            'burnout_std': float(np.std(list(data['all_burnouts'].values()), ddof=1)),
            'putout_mean': float(np.mean(list(data['all_putouts'].values()))),
            'putout_std': float(np.std(list(data['all_putouts'].values()), ddof=1)),
            'noop_mean': float(np.mean(noop_pcts)) if noop_pcts else 0.0,
            'noop_std': float(np.std(noop_pcts, ddof=1)) if len(noop_pcts) > 1 else 0.0,
            'action_mean': float(np.mean(list(data['all_actions'].values()))),
            'action_std': float(np.std(list(data['all_actions'].values()), ddof=1)),
            'reward_per_fight_mean': float(np.mean(list(data['all_reward_per_fight'].values()))),
            'reward_per_fight_std': float(np.std(list(data['all_reward_per_fight'].values()), ddof=1)),
            'new_fire_ignited_mean': float(np.mean(list(data['all_new_fire_ignited'].values()))),
            'new_fire_ignited_std': float(np.std(list(data['all_new_fire_ignited'].values()), ddof=1)),
            'zero_suppressant_mean': float(np.mean(list(data['all_zero_suppressant'].values()))),
            'zero_suppressant_std': float(np.std(list(data['all_zero_suppressant'].values()), ddof=1)),
        })

    # Dynamic: use setting_types actually found in DB for this zero-shot combination
    if args.settings:
        # Preserve user-specified order, only include those with data
        setup_list = [s for s in args.settings if s in setting_types]
    else:
        setup_list = sorted(setting_types)

    for metric in metrics:
        print(f"\n{'-'*200}")
        print(f"{metric_names[metric].upper()}")
        print(f"{'-'*200}\n")

        model_var_data = defaultdict(lambda: defaultdict(dict))
        for model_name, entries in models_data.items():
            for exp_data in entries:
                var_key = exp_data['actual_variation']
                setup_type = exp_data['setting_type']
                # noop_mean/noop_std already store per-seed percentages
                mean_val = exp_data[f'{metric}_mean']
                std_val = exp_data[f'{metric}_std']
                model_var_data[model_name][var_key][setup_type] = {'mean': mean_val, 'std': std_val}

        # Raw data for tests (smart only, seed -> value dicts)
        setup_raw_data = defaultdict(dict)
        for (model_family, variation1, variation2, target_setting), data in combinations_data.items():
            model_var_key = f"{model_family}|{(f'{variation1 if variation1 else ''} {variation2 if variation2 else ''}'.strip())}"

            if metric == 'reward':
                raw_values = data['all_rewards']  # {seed: value}
            elif metric == 'burnout':
                raw_values = data.get('all_burnouts', {})
            elif metric == 'putout':
                raw_values = data.get('all_putouts', {})
            elif metric == 'pickup':
                raw_values = data.get('all_pickups', {})
            elif metric == 'dropoff':
                raw_values = data.get('all_dropoffs', {})
            elif metric == 'accept':
                raw_values = data.get('all_accepts', {})
            elif metric == 'noop':
                raw_values = {}
                for seed, noop in data['all_noops'].items():
                    action = data['all_actions'].get(seed, 1)
                    raw_values[seed] = (noop / action * 100) if action > 0 else 0.0
            elif metric == 'action':
                raw_values = data['all_actions']
            elif metric == 'reward_per_fight':
                raw_values = data.get('all_reward_per_fight', {})
            elif metric == 'new_fire_ignited':
                raw_values = data.get('all_new_fire_ignited', {})
            elif metric == 'zero_suppressant':
                raw_values = data.get('all_zero_suppressant', {})
            else:
                raw_values = {}

            if raw_values:
                setup_raw_data[target_setting][model_var_key] = raw_values

        # Significance (one-sided paired t-test on matching seeds)
        alpha_zs = ALPHA_ABLATION if ablation else ALPHA_ZEROSHOT
        setup_significant = defaultdict(set)
        setup_dz = defaultdict(dict)
        for setup in setup_list:
            if setup not in setup_raw_data:
                continue

            filtered_raw_data = {}
            for k, v in setup_raw_data[setup].items():
                # In ablation mode: compare PLATO variants only
                if ablation and not k.startswith('PLATO|'):
                    continue
                if any(ignored in k.lower() for ignored in ignore_variations_for_ptest):
                    continue
                filtered_raw_data[k] = v

            if not filtered_raw_data:
                continue

            setup_means = {k: float(np.mean(list(v.values()))) for k, v in filtered_raw_data.items()}
            if not setup_means:
                continue

            if metric in higher_is_better:
                best_mean = max(setup_means.values())
                best_models = [k for k, v in setup_means.items() if v == best_mean]
            else:
                best_mean = min(setup_means.values())
                best_models = [k for k, v in setup_means.items() if v == best_mean]

            for best_model in best_models:
                best_data = filtered_raw_data[best_model]
                if len(filtered_raw_data) == 1:
                    continue

                is_significant = True
                min_dz = float('inf')
                alternative = 'greater' if metric in higher_is_better else 'less'
                for other_model, other_data in filtered_raw_data.items():
                    if other_model == best_model:
                        continue
                    try:
                        # Pair by matching seeds
                        common_seeds = set(best_data.keys()) & set(other_data.keys())
                        if len(common_seeds) < 2:
                            is_significant = False
                            break
                        best_paired = [best_data[s] for s in sorted(common_seeds)]
                        other_paired = [other_data[s] for s in sorted(common_seeds)]

                        # One-sided paired t-test
                        _, p_value = stats.ttest_rel(best_paired, other_paired, alternative=alternative)
                        if np.isnan(p_value):
                            is_significant = False
                            break
                        if p_value >= alpha_zs:
                            is_significant = False
                            break

                        # Cohen's d_z
                        if metric in higher_is_better:
                            diffs = [b - o for b, o in zip(best_paired, other_paired)]
                        else:
                            diffs = [o - b for b, o in zip(best_paired, other_paired)]
                        s_d = float(np.std(diffs, ddof=1))
                        if s_d > 0:
                            min_dz = min(min_dz, float(np.mean(diffs)) / s_d)
                    except Exception:
                        is_significant = False
                        break

                if is_significant:
                    setup_significant[setup].add(best_model)
                    if min_dz < float('inf'):
                        setup_dz[setup][best_model] = min_dz

        # Table (smart only)
        results = []

        if 'PLATO' in model_var_data:
            plato_vars = sorted(model_var_data['PLATO'].keys(), key=lambda x: (x.count('dotproduct'), x), reverse=True)
            # When ablation=False, only show mlp additive
            if not ablation:
                plato_vars = [v for v in plato_vars if v.strip() == 'mlp additive']
            for var in plato_vars:
                row = {'Model': 'PLATO', 'Variation': var}
                model_var_key = f"PLATO|{var}"

                for setup in setup_list:
                    if setup in model_var_data['PLATO'][var]:
                        d = model_var_data['PLATO'][var][setup]
                        is_sig = model_var_key in setup_significant.get(setup, set())
                        min_dz = setup_dz.get(setup, {}).get(model_var_key) if is_sig else None
                        if latex:
                            val_str = format_latex_value(d['mean'], d['std'], is_sig, metric, min_dz)
                        else:
                            val_str = f"{d['mean']:.2f} ± {d['std']:.2f}"
                            if metric == 'noop':
                                val_str += '%'
                            if is_sig:
                                dz_str = f" (dz≥{min_dz:.2f})" if min_dz is not None else ""
                                val_str = '* ' + val_str + dz_str
                        row[setup] = val_str
                    else:
                        row[setup] = '-'

                results.append(row)

        for algo in ['DICG', 'DICG_wildfire', 'DGN', 'MOHITO', 'MOHITO_wildfire']:
            if algo not in model_var_data:
                continue
            for var in sorted(model_var_data[algo].keys()):
                row = {'Model': algo, 'Variation': var}
                model_var_key = f"{algo}|{var}"

                for setup in setup_list:
                    if setup in model_var_data[algo][var]:
                        d = model_var_data[algo][var][setup]
                        is_sig = model_var_key in setup_significant.get(setup, set())
                        min_dz = setup_dz.get(setup, {}).get(model_var_key) if is_sig else None
                        if latex:
                            val_str = format_latex_value(d['mean'], d['std'], is_sig, metric, min_dz)
                        else:
                            val_str = f"{d['mean']:.2f} ± {d['std']:.2f}"
                            if metric == 'noop':
                                val_str += '%'
                            if is_sig:
                                dz_str = f" (dz≥{min_dz:.2f})" if min_dz is not None else ""
                                val_str = '* ' + val_str + dz_str
                        row[setup] = val_str
                    else:
                        row[setup] = '-'

                results.append(row)

        if results:
            if latex:
                print_latex_table(results, setup_list, metric, latex_grid_str, setup_significant, is_zeroshot=True)
            else:
                print(tabulate(results, headers="keys", tablefmt="grid"))
        else:
            print(f"No data for {metric_names[metric]}")
        print()

conn.close()