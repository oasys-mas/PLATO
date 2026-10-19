import json
import os
import time

import numpy as np
import pandas as pd
import json
import torch
from types import SimpleNamespace

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


def save_experiment_config(args, hparams=None,checkpoint_path=None, save_dir="."):
    """
    Append experiment configuration (args + hyperparameters) to experiments.json instead of overwriting.
    Each experiment entry is stored as a dictionary with a unique timestamp.
    """
    # Collect all arguments
    cfg = {**vars(args)}



    cfg["hyperparameters"] = hparams  # Store hyperparameters in a nested dictionary

    # Add checkpoint path
    if checkpoint_path:
        cfg["checkpoint_path"] = checkpoint_path

    # Add timestamp and experiment name
    cfg["timestamp"] = time.strftime("%Y-%m-%d_%H-%M-%S")
    cfg["experiment_name"] = getattr(args, "experiment_name", "unnamed_experiment")

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "experiments.json")

    # Load existing experiments if file exists
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            try:
                experiments = json.load(f)
                if not isinstance(experiments, list):
                    experiments = [experiments]
            except json.JSONDecodeError:
                experiments = []
    else:
        experiments = []

    # Append new experiment config
    experiments.append(cfg)

    # Save updated JSON
    with open(config_path, "w") as f:
        json.dump(experiments, f, indent=4)

    return cfg
parent_dir = os.path.dirname(os.path.dirname(__file__))
default_path = os.path.join(parent_dir, "configs.json")
def load_config(model_name: str, config_file: str = default_path):
    with open(config_file, "r") as f:
        cfg_all = json.load(f)

    if model_name not in cfg_all:
        raise ValueError(f"Model '{model_name}' not found in {config_file}")

    cfg = cfg_all[model_name]

    # resolve device strings
    dev = cfg.get("DEVICE", None)
    if isinstance(dev, str):
        if dev.lower() == "cuda":
            cfg["DEVICE"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            cfg["DEVICE"] = torch.device(dev.lower())

    return SimpleNamespace(**cfg)

plato_config = load_config("PLATO")

def collect_and_print_run_stats(run_idx, run_rewards, total_noop, total_fight, total_actions, episodes_per_run, num_runs,run_episode_burnouts,run_episode_putouts):
    """Collect run statistics, print them, and return a dict for results list."""
    eval = load_config("Eval")

        
        
    # Reward stats
    avg_reward = np.mean(run_rewards)
    std_reward = np.std(run_rewards)

    
    avg_burnouts = np.mean(run_episode_burnouts)
    avg_putouts = np.mean(run_episode_putouts)
    

    # Per-episode averages
    avg_NOOP = total_noop / episodes_per_run
    avg_FIGHT = total_fight / episodes_per_run
    avg_TOTAL = total_actions / episodes_per_run

    # Print
    print(
        f"Run {run_idx}/{num_runs}:\n"
        f"  Reward mean ± std: {avg_reward:.4f} ± {std_reward:.4f}\n"
        f"  NOOP avg/total: {avg_NOOP:.2f} / {total_noop}\n"
        f"  FIGHT avg/total: {avg_FIGHT:.2f} / {total_fight}\n"
        f"  TOTAL actions avg/total: {avg_TOTAL:.2f} / {total_actions}"
    )

    # Return stats dict
    return {
        'run': run_idx,
        'average_reward': avg_reward,
        'std_reward': std_reward,
        'avg_putouts': avg_putouts,
        'avg_burnouts': avg_burnouts,
        'avg_NOOP': avg_NOOP,
        'avg_FIGHT': avg_FIGHT,
        'avg_TOTAL_actions': avg_TOTAL,
        'total_noop_actions': total_noop,
        'total_fight_actions': total_fight,
        'total_actions': total_actions,
    }

def average_results(results, experiment, output_dir="evaluation_results", placeholder=""):
    """
    Save evaluation results to per-experiment CSV and also update master_summary.csv (no duplicates).
    """
    # Create DataFrame for this experiment
    df = pd.DataFrame(results)

    # Normalized reward
    df["normalized_reward"] = (
        (df["average_reward"] - df["average_reward"].min()) /
        (df["average_reward"].max() - df["average_reward"].min() + 1e-8)
    )

    # Per-experiment CSV
    exp_id = f"{experiment.algorithm}_{experiment.pointer_policy}_{experiment.attention_type}_{experiment.setting_type}_{experiment.grid_size}X{experiment.grid_size}"

    # Summary stats
    avg_reward_mean = df['average_reward'].mean()
    avg_reward_std = df['average_reward'].std()
    
    avg_burnuts = df['avg_burnouts'].mean()
    avg_putouts = df['avg_putouts'].mean()
    
    avg_std_reward = df['std_reward'].mean()
    avg_noop = df['avg_NOOP'].mean()
    avg_fight = df['avg_FIGHT'].mean()
    avg_total = df['avg_TOTAL_actions'].mean()
    # if plato_config.Shuffle_Tasks and 'PLATO' in experiment.algorithm:
    #     experiment.algorithm=experiment.algorithm+'_sh'
    average_row = {
        'baseline':experiment.algorithm,
        'checkpoint_name':experiment.checkpoint,
        
        "experiment_id": exp_id,
        "avg_reward_mean": avg_reward_mean,
        "avg_putouts": avg_putouts,
        "avg_burnuts": avg_burnuts,
        "avg_reward_std": avg_reward_std,
        "avg_std_reward": avg_std_reward,
        "avg_NOOP": avg_noop,
        "avg_FIGHT": avg_fight,
        "avg_TOTAL_actions": avg_total
    }
    return average_row
    
def save_and_summarize_results(results, experiment, output_dir="evaluation_results", placeholder=""):
    """
    Save evaluation results to per-experiment CSV and also update master_summary.csv (no duplicates).
    """
    # Create DataFrame for this experiment
    df = pd.DataFrame(results)

    # Normalized reward
    df["normalized_reward"] = (
        (df["average_reward"] - df["average_reward"].min()) /
        (df["average_reward"].max() - df["average_reward"].min() + 1e-8)
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Per-experiment CSV
    exp_id = f"{experiment.algorithm}_{experiment.pointer_policy}_{experiment.attention_type}_{experiment.setting_type}_{experiment.grid_size}X{experiment.grid_size}"
    csv_filename = f"{exp_id}_{placeholder}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"\nEvaluation completed! Results saved to: {csv_path}")

    # Summary stats
    avg_reward_mean = df['average_reward'].mean()
    avg_reward_std = df['average_reward'].std()
    
    avg_burnuts = df['avg_burnouts'].mean()
    avg_putouts = df['avg_putouts'].mean()
    
    avg_std_reward = df['std_reward'].mean()
    avg_noop = df['avg_NOOP'].mean()
    avg_fight = df['avg_FIGHT'].mean()
    avg_total = df['avg_TOTAL_actions'].mean()

    print("\nSummary Statistics:")
    print(f"  Average reward across runs: {avg_reward_mean:.4f} ± {avg_reward_std:.4f}")
    print(f"  Average per-run std_reward: {avg_std_reward:.4f}")
    print(f"  Average NOOP actions/episode: {avg_noop:.2f}")
    print(f"  Average FIGHT actions/episode: {avg_fight:.2f}")
    print(f"  Average TOTAL actions/episode: {avg_total:.2f}")




    # Master summary CSV
    master_csv_path = os.path.join(output_dir, "master_summary.csv")
    new_row = {
        "experiment_id": exp_id,
        "avg_reward_mean": avg_reward_mean,
        "avg_putouts": avg_putouts,
        "avg_burnuts": avg_burnuts,
        "avg_reward_std": avg_reward_std,
        "avg_std_reward": avg_std_reward,
        "avg_NOOP": avg_noop,
        "avg_FIGHT": avg_fight,
        "avg_TOTAL_actions": avg_total
    }

    if os.path.exists(master_csv_path):
        master_df = pd.read_csv(master_csv_path)
        # Remove any old entry with the same experiment_id
        master_df = master_df[master_df["experiment_id"] != exp_id]
    else:
        master_df = pd.DataFrame(columns=new_row.keys())

    # Append new row and save
    master_df = pd.concat([master_df, pd.DataFrame([new_row])], ignore_index=True)
    master_df.to_csv(master_csv_path, index=False)

    return csv_path


def _parse_experiment_id(exp_id: str):
    """
    Parse experiment_id strings to extract:
    algorithm, pointer_policy, attention_type, setting_type, grid_size, and a label
    that includes ONLY algo/policy/attention (NO setting).
    """
    
    s = exp_id.strip()
    s = s.replace("___", "_")
    s = s.replace("stochastically_completable", "stochasticallycompletable")
    s = s.replace("time_critical","timecritical")
    s = s.replace("resource_limited", "resourcelimited")
    s = re.sub(r"_+", "_", s)
    tokens = [t for t in s.split("_") if t]

    KNOWN_SETUPS = {"completable", "incompletable", "overkillable", "stochasticallycompletable"  ,"resourcelimited", "timecritical"}
    ATTN_SET = {"additive", "dotproduct"}
    POLICY_SET = {"lstm", "gru"}

    grid_token = next((t for t in tokens if re.fullmatch(r"\d+[xX]\d+", t)), "")
    m = re.search(r"(\d+)[xX](\d+)", grid_token)
    grid_size = int(m.group(1)) if m else np.nan
    if grid_token:
        tokens.remove(grid_token)

    setting_type = ""
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i].lower() in KNOWN_SETUPS:
            setting_type = tokens[i].lower()
            tokens.pop(i)
            break
    def canonical_algo(head):
        h = head.lower()
        if h in {"dgn"}: return "DGN"
        if h in {"dicg"}: return "DICG"
        if h in {"qmix"}: return "QMIX"
        if h in {"coma"}: return "COMA"
        if h in {"vanilla_mappo", "vanilla", "mappo"}: return "Vanilla_MAPPO"
        if h in {"PLATO", "pointerlearner", "pointer", "pinterlearner"}: return "PLATO"
        if h in {"noop"}: return "Naive-NOOP"
        if h in {"random"}: return "Naive-Random"
        if h in {"weakest"}: return "Naive-Weakest"
        if h in {"strongest"}: return "Naive-Strongest"
        return None

    algorithm = ""
    if len(tokens) >= 2:
        two = f"{tokens[0]}_{tokens[1]}"
        cand = canonical_algo(two)
        if cand:
            algorithm, tokens = cand, tokens[2:]
    if not algorithm and tokens:
        cand = canonical_algo(tokens[0])
        if cand:
            algorithm, tokens = cand, tokens[1:]
    if not algorithm:
        algorithm = tokens.pop(0) if tokens else "UNKNOWN"

    pointer_policy, attention_type = "", ""
    for t in list(tokens):
        tl = t.lower()
        if not pointer_policy and tl in POLICY_SET:
            pointer_policy = tl
        elif not attention_type and tl in ATTN_SET:
            attention_type = tl

    def pretty(s): return s.replace("_", " ").title()

    # Label WITHOUT setting
    label_parts = [pretty(algorithm)]
    if pointer_policy: label_parts.append(pointer_policy.upper())
    if attention_type: label_parts.append(attention_type)
    label = "\n".join(label_parts)

    return algorithm, pointer_policy, attention_type, setting_type, grid_size, label

def plot_pvalue_heatmap_for_grid(sub_df, out_path, episode_data_dict=None, n_sim_eps=50):
    """
    Pairwise p-value (Mann-Whitney U) heatmap with larger cells and smaller number font.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu

    exp_ids = sub_df["experiment_id"].tolist()

    # Abbreviate labels
    short_labels = []
    for algo, pol, attn in zip(sub_df["algorithm"], sub_df["pointer_policy"], sub_df["attention_type"]):
        p = pol.upper() if pol else ""
        a = attn if attn != "none" else ""
        short = "-".join(filter(None, [algo.split("_")[0], p, a]))
        short_labels.append(short)

    # Gather episodes
    episodes_per_exp = []
    for exp_id in exp_ids:
        if episode_data_dict and exp_id in episode_data_dict:
            episodes = np.asarray(episode_data_dict[exp_id], dtype=float)
        else:
            row = sub_df[sub_df["experiment_id"] == exp_id].iloc[0]
            mean_r = float(row["avg_reward_mean"])
            std_r = float(row["avg_reward_std"])
            rng = np.random.default_rng(hash(exp_id) % (2**32))
            episodes = rng.normal(mean_r, std_r + 1e-12, n_sim_eps)
        episodes_per_exp.append(episodes)

    # Compute p-values
    n = len(exp_ids)
    P = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                _, p = mannwhitneyu(episodes_per_exp[i], episodes_per_exp[j], alternative='two-sided')
            except Exception:
                p = np.nan
            P[i, j] = P[j, i] = p
    np.fill_diagonal(P, 0.0)

    # Bigger figure each cell gets more space
    fig_size = max(8, 1.0 * n)  # was 0.6*n, now 1.0*n for larger cells
    plt.figure(figsize=(fig_size, fig_size))

    # Plot heatmap
    im = plt.imshow(P, cmap="coolwarm_r", interpolation="nearest", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="p-value (Mann–Whitney U)")

    plt.xticks(range(n), short_labels, rotation=45, ha="right", fontsize=10)
    plt.yticks(range(n), short_labels, fontsize=10)
    plt.title("Pairwise p-value Heatmap - 3x3 - Completable", fontsize=14, pad=12)

    # Smaller annotation font
    for i in range(n):
        for j in range(n):
            if i == j:
                txt = "—"
            else:
                pval = P[i, j]
                if not np.isfinite(pval):
                    txt = "nan"
                else:
                    txt = f"{pval:.3f}"
                # Bold significant values
                if np.isfinite(pval) and pval < 0.05:
                    plt.text(j, i, txt, ha="center", va="center", fontsize=7, fontweight="bold", color="black")
                else:
                    plt.text(j, i, txt, ha="center", va="center", fontsize=7, color="black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_master_summary(master_csv_path="evaluation_results/master_summary.csv",
                        out_dir="evaluation_results/plots",
                        episode_data_dict=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not os.path.exists(master_csv_path):
        raise FileNotFoundError(f"Not found: {master_csv_path}")

    df = pd.read_csv(master_csv_path)
    parsed = df["experiment_id"].apply(_parse_experiment_id)
    (df["algorithm"], df["pointer_policy"], df["attention_type"],
     df["setting_type"], df["grid_size"], df["label"]) = zip(*parsed)

    # clean fields
    df["attention_type"] = df["attention_type"].fillna("").replace("", "none")
    df["noop_pct"] = 100.0 * (df["avg_NOOP"] / df["avg_TOTAL_actions"].replace(0, np.nan))
    df["noop_pct"] = df["noop_pct"].fillna(0.0)
    df.sort_values(["grid_size", "avg_reward_mean"], ascending=[True, False], inplace=True)

    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for g in sorted(df["grid_size"].dropna().unique()):
        sub_g = df[df["grid_size"] == g].copy()
        if sub_g.empty:
            continue

        # iterate EACH setting within this grid size
        for setting in sorted(sub_g["setting_type"].dropna().unique()):
            sub = sub_g[sub_g["setting_type"] == setting].copy()
            if sub.empty:
                continue

            x = np.arange(len(sub))

            # color per bar
            cmap = plt.cm.get_cmap("tab20", max(20, len(sub)))
            bar_colors = [cmap(i % cmap.N) for i in range(len(sub))]

            fig_w = max(8, len(sub) * 1.2)

            # ---- Plot 1: Average reward ± std ----
            fig = plt.figure(figsize=(fig_w, 5))
            heights = sub["avg_reward_mean"].values.astype(float)
            yerr    = sub["avg_reward_std"].values.astype(float)
            plt.bar(x, heights, yerr=yerr, capsize=4, color=bar_colors)
            plt.xticks(x, sub["label"].tolist(), rotation=30, ha="right")
            plt.ylabel("Average Reward")
            plt.title(f"{setting.replace('_',' ').title()} — Average Reward ± Std — Grid {int(g)}x{int(g)}")
            for xi, h in zip(x, heights):
                plt.text(xi - 0.15, h, f"{h:.2f}", ha="right", va="bottom", fontsize=9)

            f1 = os.path.join(out_dir, f"grid{int(g)}_{setting}_avg_reward.png")
            plt.tight_layout(); plt.savefig(f1, dpi=200, bbox_inches="tight"); plt.close(fig)
            saved.append(f1)

            # ---- Plot 2: NOOP percentage ----
            fig = plt.figure(figsize=(fig_w, 5))
            heights2 = sub["noop_pct"].values.astype(float)
            plt.bar(x, heights2, color=bar_colors)
            plt.xticks(x, sub["label"].tolist(), rotation=30, ha="right")
            plt.ylabel("NOOP (%)")
            plt.title(f"{setting.replace('_',' ').title()} — NOOP Percentage — Grid {int(g)}x{int(g)}")
            for xi, h in zip(x, heights2):
                plt.text(xi, h, f"{h:.2f}%", ha="center", va="bottom", fontsize=9)

            f2 = os.path.join(out_dir, f"grid{int(g)}_{setting}_noop_pct.png")
            plt.tight_layout(); plt.savefig(f2, dpi=200, bbox_inches="tight"); plt.close(fig)
            saved.append(f2)

            # ---- Plot 3: p-value heatmap (per setting) ----
            f3 = os.path.join(out_dir, f"grid{int(g)}_{setting}_pvalue_heatmap.png")
            plot_pvalue_heatmap_for_grid(sub, f3, episode_data_dict=episode_data_dict)
            saved.append(f3)

                        # ---- Plot 3: Performance (avg_putouts vs avg_burnouts) ----
            fig = plt.figure(figsize=(fig_w, 5))

            # Pull metrics; column names match what you save in master_summary.csv
            putouts = sub["avg_putouts"].astype(float).values
            burnouts = sub["avg_burnuts"].astype(float).values  # note: 'avg_burnuts' (as saved)

            # Side-by-side bars per label
            bar_width = 0.38
            x_left  = x - bar_width/2
            x_right = x + bar_width/2

            plt.bar(x_left,  putouts,  width=bar_width, label="Putouts",  color="green")
            plt.bar(x_right, burnouts, width=bar_width, label="Burnouts", color="red")

            # X axis labels
            plt.xticks(x, sub["label"].tolist(), rotation=30, ha="right")

            plt.ylabel("Average per episode")
            plt.title(f"{setting.replace('_',' ').title()} — Performance — Grid {int(g)}x{int(g)}")

            # Numbers on bars (like your other charts)
            for xi, h in zip(x_left, putouts):
                plt.text(xi, h, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
            for xi, h in zip(x_right, burnouts):
                plt.text(xi, h, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

            plt.legend()

            f3_perf = os.path.join(out_dir, f"grid{int(g)}_{setting}_performance.png")
            plt.tight_layout(); plt.savefig(f3_perf, dpi=200, bbox_inches="tight"); plt.close(fig)
            saved.append(f3_perf)

    return saved

def export_setting_tables(master_csv_path="evaluation_results/master_summary.csv",
                          out_dir="evaluation_results/summary_tables"):
    """
    Build 2 tables per grid size with mean ± std formatting:
      1) avg_reward_mean ± avg_reward_std (per family × setting)
      2) noop_pct_mean ± noop_pct_std (per family × setting)

    Rows are ordered:
      POINTER_LSTM_ADDITIVE
      POINTER_LSTM_DOTPRODUCT

      MAPPO, QMIX, COMA, Weakest, Strongest, Random, NOOP

    Columns are setting types: completable, incompletable, overkillable, stochastically_completable, Average
    """
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(master_csv_path):
        raise FileNotFoundError(f"Not found: {master_csv_path}")

    df = pd.read_csv(master_csv_path)

    # Parse
    parsed = df["experiment_id"].apply(_parse_experiment_id)
    (df["algorithm"], df["pointer_policy"], df["attention_type"],
     df["setting_type"], df["grid_size"], df["label"]) = zip(*parsed)

    # Normalize settings for headers
    df["setting_type"] = df["setting_type"].fillna("").replace({
    "stochasticallycompletable": "stochastically_completable",
    "resourcelimited": "resource_limited",
    "timecritical": "time_critical",
    })

    
    # Compute NOOP %
    df["noop_pct"] = 100.0 * (df["avg_NOOP"] / df["avg_TOTAL_actions"].replace(0, np.nan))
    df["noop_pct"] = df["noop_pct"].fillna(0.0)

    # Build row labels (families)
    def make_family(row):
        algo = (row["algorithm"] or "").lower()
        if "pointer" in algo:
            pol = (row["pointer_policy"] or "").upper()
            attn = (row["attention_type"] or "").upper()
            return f"POINTER_{pol}_{attn}"
        if "dgn" in algo or "DGN" in algo:
            return "DGN"
        if "DICG" in algo or "dicg" in algo:
            return "DICG"
        if "vanilla_map" in algo or "mappo" in algo:
            return "MAPPO"
        if "qmix" in algo:
            return "QMIX"
        if "coma" in algo:
            return "COMA"
        if "weakest" in algo:
            return "Weakest"
        if "strongest" in algo:
            return "Strongest"
        if "random" in algo:
            return "Random"
        if "noop" in algo:
            return "NOOP"
        return row["algorithm"] or "UNKNOWN"

    df["family"] = df.apply(make_family, axis=1)

    row_order = [
        "POINTER_LSTM_ADDITIVE",
        "POINTER_LSTM_DOTPRODUCT",
        "DGN",
        "DICG",
        "MOHITO",
        # "MAPPO",
        # "QMIX",
        # "COMA",
        "Weakest",
        "Strongest",
        "Random",
        "NOOP",
    ]
    col_order = ["completable", "incompletable", "overkillable", "stochastically_completable","resource_limited","time_critical"]

    def fmt(mean, std):
        if np.isnan(mean):
            return ""
        if np.isnan(std):
            return f"{mean:.2f}"
        return f"{mean:.2f} ± {std:.1f}"

    for g in sorted(df["grid_size"].dropna().unique()):
        sub = df[df["grid_size"] == g].copy()
        if sub.empty:
            continue

        # Aggregate per (family, setting)
        grouped = (sub.groupby(["family", "setting_type"], dropna=False)
                     .agg(
                         # reward mean and std (use mean of per-run stds as your std summary)
                         reward_mean=("avg_reward_mean", "mean"),
                         reward_std=("avg_reward_std", "mean"),
                         # noop mean and std derived from noop_pct across entries
                         noop_mean=("noop_pct", "mean"),
                         
                         noop_std=("noop_pct", "std"),
                         
                         avg_burnuts_mean =("avg_burnuts", "mean"),
                         avg_burnut_std =("avg_burnuts", "std"),
                         
                         avg_puouts_mean =("avg_putouts", "mean"),
                         avg_puouts_std =("avg_putouts", "std"),
                         
                     )
                     .reset_index())

        def make_table(metric_mean: str, metric_std: str, out_name: str):
            # Pivot for means and stds separately
            piv_mean = grouped.pivot(index="family", columns="setting_type", values=metric_mean)
            piv_std  = grouped.pivot(index="family", columns="setting_type", values=metric_std)

            # Ensure column order & include missing cols
            for c in col_order:
                if c not in piv_mean.columns:
                    piv_mean[c] = np.nan
                if c not in piv_std.columns:
                    piv_std[c] = np.nan
            piv_mean = piv_mean[col_order]
            piv_std  = piv_std[col_order]

            # Ensure all requested rows exist
            for r in row_order:
                if r not in piv_mean.index:
                    piv_mean.loc[r] = np.nan
                    piv_std.loc[r]  = np.nan
            piv_mean = piv_mean.loc[row_order]
            piv_std  = piv_std.loc[row_order]

            # Format mean ± std per cell
            formatted = piv_mean.copy()
            for r in formatted.index:
                for c in col_order:
                    formatted.at[r, c] = fmt(piv_mean.at[r, c], piv_std.at[r, c])

            # "Average" column = mean of means across settings, std = mean of stds across settings
            avg_mean = np.nanmean(piv_mean.values, axis=1)
            avg_std  = np.nanmean(piv_std.values, axis=1)
            formatted["Average"] = [fmt(m, s) for m, s in zip(avg_mean, avg_std)]

            # Save as CSV (string cells)
            out_path = os.path.join(out_dir, f"{out_name}_grid{int(g)}.csv")
            formatted.to_csv(out_path, index_label="family",encoding="utf-8")
            print(f"Saved: {out_path}")

        # 1) Reward table (mean ± std)
        make_table("reward_mean", "reward_std", "table_avg_reward_by_setting")

        # 2) NOOP% table (mean ± std)
        make_table("noop_mean", "noop_std", "table_noop_pct_by_setting")
        
        make_table("avg_burnuts_mean", "avg_burnut_std", "table_burnout_performance_by_setting")
        make_table("avg_puouts_mean", "avg_puouts_std", "table_putout_performance_by_setting")
                         

# Example:
# files = plot_master_summary()
# print("\n".join(files))

