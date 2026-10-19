import os
import sys
import sqlite3
import itertools
import pandas as pd
from scipy.stats import mannwhitneyu

# Make project root importable
this_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(this_file))
sys.path.append(project_root)
from experiment_logger_sqlite import DB_PATH, init_db

def load_data_option_1():
    """Compare ALL experiments within each (grid_size, setting_type)"""
    return """
    WITH best_eval AS (
        SELECT x.*
        FROM (
            SELECT
                er.*,
                ROW_NUMBER() OVER (
                    PARTITION BY experiment_id
                    ORDER BY avg_reward DESC
                ) AS rk
            FROM eval_results er
        ) x
        WHERE x.rk = 1
    )
    SELECT
        e.id AS experiment_id,
        (e.name || '_' || COALESCE(e.variation1, '') || '_' || COALESCE(e.variation2, '')) AS exp_name,
        e.grid_size,
        e.setting_type,
        e.model_family,
        e.variation1,
        e.variation2,
        e.run_type,
        r.run_index,
        r.reward
    FROM experiments e
    JOIN best_eval be ON be.experiment_id = e.id
    JOIN eval_runs r ON r.eval_id = be.id
    WHERE e.run_type = 'execution'
    ORDER BY e.grid_size, e.setting_type, e.model_family, experiment_id, r.run_index
    """

def load_data_option_2():
    """Top 5 experiments per (grid_size, setting_type)"""
    return """
    WITH best_eval AS (
        SELECT x.*
        FROM (
            SELECT
                er.*,
                ROW_NUMBER() OVER (
                    PARTITION BY experiment_id
                    ORDER BY avg_reward DESC
                ) AS rk
            FROM eval_results er
        ) x
        WHERE x.rk = 1
    ),
    ranked_experiments AS (
        SELECT
            e.id           AS experiment_id,
            e.grid_size,
            e.setting_type,
            e.run_type,
            e.model_family,
            e.variation1,
            e.variation2,
            be.id          AS eval_id,
            be.avg_reward,
            ROW_NUMBER() OVER (
                PARTITION BY e.grid_size, e.setting_type
                ORDER BY be.avg_reward DESC
            ) AS rank_by_reward
        FROM experiments e
        JOIN best_eval be ON be.experiment_id = e.id
        WHERE e.run_type = 'execution'
    ),
    top_experiments AS (
        SELECT *
        FROM ranked_experiments
        WHERE rank_by_reward <= 5
    )
    SELECT
        e.id AS experiment_id,
        (e.name || '_' || COALESCE(e.variation1, '') || '_' || COALESCE(e.variation2, '')) AS exp_name,
        e.grid_size,
        e.setting_type,
        e.run_type,
        r.run_index,
        r.reward
    FROM top_experiments t
    JOIN experiments e ON e.id = t.experiment_id
    JOIN eval_runs   r ON r.eval_id = t.eval_id
    ORDER BY e.grid_size, e.setting_type, experiment_id, r.run_index
    """

def load_data_original():
    """Your original query"""
    return """
    WITH best_eval AS (
        SELECT x.*
        FROM (
            SELECT
                er.*,
                ROW_NUMBER() OVER (
                    PARTITION BY experiment_id
                    ORDER BY avg_reward DESC
                ) AS rk
            FROM eval_results er
        ) x
        WHERE x.rk = 1
    ),
    ranked_experiments AS (
        SELECT
            e.id           AS experiment_id,
            e.grid_size,
            e.setting_type,
            e.run_type,
            e.model_family,
            e.variation1,
            e.variation2,
            be.id          AS eval_id,
            be.avg_reward,
            ROW_NUMBER() OVER (
                PARTITION BY e.model_family, e.variation1, e.variation2, e.setting_type
                ORDER BY be.avg_reward DESC
            ) AS rank_by_reward
        FROM experiments e
        JOIN best_eval be ON be.experiment_id = e.id
        WHERE e.run_type = 'execution'
    ),
    top3_experiments AS (
        SELECT *
        FROM ranked_experiments
        WHERE rank_by_reward <= 3
    )
    SELECT
        e.id AS experiment_id,
        (e.name || '_' || COALESCE(e.variation1, '') || '_' || COALESCE(e.variation2, '')) AS exp_name,
        e.grid_size,
        e.setting_type,
        e.run_type,
        r.run_index,
        r.reward
    FROM top3_experiments t
    JOIN experiments e ON e.id = t.experiment_id
    JOIN eval_runs   r ON r.eval_id = t.eval_id
    ORDER BY e.grid_size, e.setting_type, experiment_id, r.run_index
    """

def run_statistical_analysis(query_func, option_name):
    """Run the statistical analysis with a given query"""
    print(f"\n=== RUNNING ANALYSIS: {option_name} ===")
    
    init_db()
    conn = sqlite3.connect(DB_PATH)
    
    query = query_func()
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded rows: {len(df)}")
    
    if len(df) == 0:
        print("No data loaded!")
        return None
    
    # Show breakdown by group
    breakdown = df.groupby(['grid_size', 'setting_type']).agg({
        'experiment_id': 'nunique',
        'reward': 'count'
    }).rename(columns={'experiment_id': 'unique_experiments', 'reward': 'total_runs'})
    print("Breakdown by (grid_size, setting_type):")
    print(breakdown)
    
    # Compute Mann–Whitney U p-values for all pairs
    rows = []
    total_comparisons = 0
    
    for (grid_size, setting_type), gdf in df.groupby(["grid_size", "setting_type"]):
        exps = sorted(gdf["experiment_id"].unique())
        group_comparisons = len(list(itertools.combinations(exps, 2)))
        total_comparisons += group_comparisons
        
        print(f"\nGroup ({grid_size}, {setting_type}): {len(exps)} experiments, {group_comparisons} pairwise comparisons")
        
        for exp_a, exp_b in itertools.combinations(exps, 2):
            ra = gdf.loc[gdf["experiment_id"] == exp_a, "reward"].values
            rb = gdf.loc[gdf["experiment_id"] == exp_b, "reward"].values
            
            if len(ra) == 0 or len(rb) == 0:
                continue
            
            name_a = gdf.loc[gdf["experiment_id"] == exp_a, "exp_name"].iloc[0]
            name_b = gdf.loc[gdf["experiment_id"] == exp_b, "exp_name"].iloc[0]
            
            # Mann–Whitney U test, two sided
            u_stat, p_value = mannwhitneyu(ra, rb, alternative="two-sided")
            
            rows.append({
                "grid_size": grid_size,
                "setting_type": setting_type,
                "exp_id_a": exp_a,
                "name_a": name_a,
                "n_a": len(ra),
                "exp_id_b": exp_b,
                "name_b": name_b,
                "n_b": len(rb),
                "u_stat": u_stat,
                "p_value": p_value,
            })
    
    print(f"\nTotal pairwise comparisons: {total_comparisons}")
    
    if len(rows) == 0:
        print("No pairwise comparisons could be made!")
        return None
    
    p_df = pd.DataFrame(rows)
    
    # Show summary stats
    print(f"Generated {len(p_df)} p-value comparisons")
    print(f"Significant results (p < 0.05): {(p_df['p_value'] < 0.05).sum()}")
    print(f"Highly significant (p < 0.01): {(p_df['p_value'] < 0.01).sum()}")
    
    # Save results
    filename = f"pairwise_pvalues_{option_name.lower().replace(' ', '_')}_mannwhitney.csv"
    out_path = os.path.join(project_root, filename)
    p_df.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}")
    
    return p_df

if __name__ == "__main__":
    # Try different options
    options = [
        (load_data_original, "Original (Top 3 per complex group)"),
        (load_data_option_2, "Option 2 (Top 5 per grid_size+setting)"),
        (load_data_option_1, "Option 1 (All experiments)")
    ]
    
    for query_func, option_name in options:
        try:
            run_statistical_analysis(query_func, option_name)
        except Exception as e:
            print(f"Error with {option_name}: {e}")
    
    print("\n=== ANALYSIS COMPLETE ===")