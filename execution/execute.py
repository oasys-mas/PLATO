#!/usr/bin/env python3
"""
Unified model evaluation script for SLURM job submission.
Handles both smart algorithms and naive baselines.
"""
import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="free_range_zoo")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from execution import PLATOEvaluator, DICGEvaluator, DGNEvaluator, MOHITOEvaluator, DEEPSETEvaluator
from execution.naive_baseline_evaluator import NaiveBaselineEvaluator, create_baseline_experiment
from utils.experiment_manager import load_config
from evaluation_helper import scan_experiments_with_frequency
from utils.helper import setup_torch

setup_torch()
eval_config = load_config("Eval")
general_config = load_config("General_Config")


if general_config.DB == "postgres":
    from experiment_logger_postgres import ExperimentDB, RunLogger
else:
    from experiment_logger_sqlite import ExperimentDB, RunLogger
    

def run_smart_baselines_evaluation_best_checkpoint(baseline_name='PLATO', zero_shot=False, gridsize=None, zero_shot_target_grids_size='', zero_shot_mode=None, setups=None, domain='wildfire', pointer_policy=None, attention_type=None, filter_nognn=None) -> bool:
    """Run smart algorithm evaluation with frequency-based checkpoint scanning."""
    
    #  Create ONE shared database connection for this entire job (MOVED OUTSIDE ALL LOOPS)
    shared_db = ExperimentDB()
    print("Created shared database connection for smart algorithm evaluation")
    
    # Smart algorithm evaluator mapping
    evaluator_map = {
        'PLATO': PLATOEvaluator,
        'DEEPSET': DEEPSETEvaluator,
        'DICG': DICGEvaluator,
        'DGN': DGNEvaluator,
        'MOHITO': MOHITOEvaluator
    }
      
    if baseline_name not in evaluator_map:
        print(f"Unsupported smart algorithm: {baseline_name}")
        print(f"   Available: {list(evaluator_map.keys())}")
        return False
    
    total_experiments = 0
    successful_experiments = 0
    

    print(f"\n{'='*50}")
    print(f"Processing grid size: {gridsize}x{gridsize}")
    print(f"{'='*50}")
    
    baselines = scan_experiments_with_frequency(
        eval_config.checkpoint_folder,
        baseline=baseline_name,
        frequent=eval_config.checkpoint_frequency,
        gridsize=gridsize,
        starting_episode=eval_config.starting_episode,
        max_episode=eval_config.max_episode,
        zero_shot=zero_shot,
        zero_shot_target_grids_size=zero_shot_target_grids_size,
        zero_shot_mode=zero_shot_mode,
        filter_pointer_policy=pointer_policy,
        filter_attention_type=attention_type,
        filter_nognn=filter_nognn,
    )
    
    for baseline in baselines:
        for experiment in baseline:
            if experiment.grid_size != gridsize:
                continue
                
            if experiment.algorithm not in evaluator_map:
                print(f"Unsupported algorithm: {experiment.algorithm}")
                continue
                
            total_experiments += 1
            print(f"Evaluating: {experiment.name} (Episode {getattr(experiment, 'episode', 'N/A')})")
            
            try:
                # Reuse the shared database connection (NO NEW CONNECTION CREATED)
                evaluator = evaluator_map[experiment.algorithm](experiment, shared_db=shared_db)  
                success = evaluator.evaluate()  # Results automatically logged to database
                
                if success:
                    successful_experiments += 1
                    print(f"Successfully evaluated {experiment.name}")
                else:
                    print(f"Failed to evaluate {experiment.name}")
                    
            except Exception as e:
                print(f"Error evaluating {experiment.name}: {str(e)}")
                import traceback
                traceback.print_exc()

    # Print summary
    print(f"\n{'='*50}")
    print(f"SMART ALGORITHM EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {total_experiments - successful_experiments}")
    if total_experiments > 0:
        print(f"Success rate: {100 * successful_experiments / total_experiments:.1f}%")
    
    print("Shared database connection used for all experiments")
    return total_experiments > 0 and successful_experiments == total_experiments


def run_naive_baselines_evaluation(baseline_name=None, setups=None, domain='wildfire') -> bool:
    """Run naive baseline evaluation on all settings and grid sizes."""
    
    # Create ONE shared database connection for all naive baselines
    shared_db = ExperimentDB()
    print("Created shared database connection for naive baseline evaluation")
    
    # Available naive baselines per domain
    wildfire_baselines = ["NOOP", "Random", "Weakest", "Strongest"]
    rideshare_baselines = ["NOOP", "Random", "FIFOTfocus", "FIFOTglobal", "GreedyTFocus", "GreedyTGlobal"]
    available_baselines = rideshare_baselines if domain == 'rideshare' else wildfire_baselines
    
    if baseline_name and baseline_name not in available_baselines:
        print(f"Unsupported naive baseline for {domain}: {baseline_name}")
        print(f"   Available: {available_baselines}")
        return False
    
    baselines_to_run = [baseline_name] if baseline_name else available_baselines
    
    print(f"\n{'='*50}")
    print(f"NAIVE BASELINE EVALUATION")
    print(f"{'='*50}")
    print(f"Baselines: {baselines_to_run}")
    print(f"Setups: {setups if setups is not None else general_config.SETUPS}")
    
    total_evaluations = 0
    successful_evaluations = 0
    
    for baseline_name in baselines_to_run:
        for setting in (setups if setups is not None else general_config.SETUPS):
            # Skip zero-shot target configs — naive baselines only run on native setups.
            if setting.endswith('_zs'):
                continue

            total_evaluations += 1
            grid_size = None
            print(f"\nEvaluating {baseline_name} on {setting} ")
            if '2x3' in setting:
                grid_size=2
            if '3x3' in setting:
                grid_size=3
            if '4x4' in setting:
                grid_size=4
            if '5x5' in setting:
                grid_size=5
            if '6x6' in setting:
                grid_size=6
            if not grid_size:
                grid_size=3  # Default to 3x3 if not specified    
            if grid_size not in eval_config.NAIVE_GRID_SIZES:
                print(f"Skipping {baseline_name} on {setting} due to unsupported grid size")
                continue
            try:
                # Create experiment object for this baseline
                experiment = create_baseline_experiment(baseline_name, setting, grid_size, domain=domain)
                
                # Create evaluator with shared database connection
                evaluator = NaiveBaselineEvaluator(experiment, shared_db=shared_db)
                success = evaluator.evaluate()
                
                if success:
                    successful_evaluations += 1
                    print(f"Completed {baseline_name} on {setting} {grid_size}x{grid_size}")
                else:
                    print(f"Failed {baseline_name} on {setting} {grid_size}x{grid_size}")
                    
            except Exception as e:
                print(f"Error evaluating {baseline_name} on {setting} {grid_size}x{grid_size}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"NAIVE BASELINE EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total configurations: {total_evaluations}")
    print(f"Successful: {successful_evaluations}")
    print(f"Failed: {total_evaluations - successful_evaluations}")
    if total_evaluations > 0:
        print(f"Success rate: {100 * successful_evaluations / total_evaluations:.1f}%")
    
    print("Shared database connection used for all baselines")
    return total_evaluations > 0 and successful_evaluations == total_evaluations


def parse_args():
    parser = argparse.ArgumentParser(description="Run unified evaluation (smart algorithms + naive baselines)")
    
    # Baseline/algorithm to evaluate
    parser.add_argument(
        "--baseline",
        type=str,
        default="PLATO",
        help="baseline name to evaluate"
    )  
    parser.add_argument('--grid_size', type=int, choices=[2,3,4,5,6],default=2, help='size of the grid')

    
    # Evaluation type
    parser.add_argument(
        "--type",
        type=str,
        default="smart",
        choices=["smart", "naive"],
        help="Type of evaluation: 'smart' (baselines) or 'naive' (baselines)"
    )
    parser.add_argument(
        "--zero_shot",
        action="store_true", default=False,
        help="Enable zero-shot evaluation mode."
    )
    parser.add_argument(
        "--zero_shot_target_grids_size",
        choices=[2,3,4,5,6], type=int,default=4,
        help="The target configuration for zero shot experiments, if null, its not a zero shot experiment."
    )
    parser.add_argument(
        "--zero_shot_mode",
        type=str,
        choices=["fixteam", "fixteam_transition", "zs", "bigteam"],
        default="zs",
        help="Zero-shot mode: 'fixteam' (same team config, different grid), 'fixteam_transition' (same + 20%% harder transitions), 'zs' (use dedicated _zs grid setups with matched agent count), or 'bigteam' (use native target grid setup with its own 6-agent team)"
    )
    parser.add_argument(
        "--load_from_pkl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, load setups from configuration_pkls instead of configs.json. Use --no-load_from_pkl to disable."
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["wildfire", "rideshare"],
        default="wildfire",
        help="Environment domain: 'wildfire' or 'rideshare'"
    )
    parser.add_argument(
        "--setting",
        type=str,
        default=None,
        help="Specific setting/config to evaluate (e.g. 'rideshare_0', 'setup1_2x3'). If not set, runs all setups."
    )
    parser.add_argument(
        "--pointer_policy",
        type=str,
        choices=["mlp", "lstm", "mlp_noteamstats", "mlp_nopointer"],
        default=None,
        help="Filter PLATO checkpoints by pointer policy. Ablation variants: mlp_noteamstats, mlp_nopointer."
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        choices=["additive", "dotproduct"],
        default=None,
        help="Filter PLATO checkpoints by attention type (additive or dotproduct). If omitted, all attention types are evaluated."
    )
    parser.add_argument(
        "--filter_nognn",
        action="store_true",
        default=False,
        help="When set, evaluate only the nognn ablation (ablation_no_gnn_critic) checkpoints."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    smart_algorithms = ['PLATO', 'DICG', 'DGN', 'MOHITO', 'DEEPSET']
    wildfire_naive = ["NOOP", "Random", "Weakest", "Strongest"]
    rideshare_naive = ["NOOP", "Random", "FIFOTfocus", "FIFOTglobal", "GreedyTFocus", "GreedyTGlobal"]

    print(f"Starting {args.type} evaluation")
    success = True
    # Determine setups source
    setups = None
    if args.load_from_pkl:
        setups = general_config.configuration_pkls
        print("Loaded setups from configuration_pkls")
    else:
        setups = general_config.SETUPS
        print("Loaded setups from setups")

    # Filter to a single setting if --setting is provided
    if args.setting:
        setups = [args.setting]
        print(f"Filtered to single setting: {args.setting}")
    else:
        print("Running all configured setups")

    if args.type == "smart":
        baselines_to_run = [args.baseline] if args.baseline else smart_algorithms
        print(f"Evaluating smart algorithms: {baselines_to_run}")
        for baseline in baselines_to_run:
            if baseline not in smart_algorithms:
                print(f"Unknown smart algorithm: {baseline}")
                success = False
                continue
            print(f"\nRunning smart algorithm: {baseline}")
            success &= run_smart_baselines_evaluation_best_checkpoint(
                baseline,
                zero_shot=args.zero_shot,
                gridsize=args.grid_size,
                zero_shot_target_grids_size=args.zero_shot_target_grids_size,
                zero_shot_mode=args.zero_shot_mode,
                setups=setups,
                domain=args.domain,
                pointer_policy=args.pointer_policy,
                attention_type=args.attention_type,
                filter_nognn=True if args.filter_nognn else None,
            )
    elif args.type == "naive":
        naive_baselines = rideshare_naive if args.domain == 'rideshare' else wildfire_naive
        baselines_to_run = [args.baseline] if args.baseline else naive_baselines
        print(f"Evaluating naive baselines ({args.domain}): {baselines_to_run}")
        for baseline in baselines_to_run:
            if baseline not in naive_baselines:
                print(f"Unknown naive baseline for {args.domain}: {baseline}")
                success = False
                continue
            print(f"\nRunning naive baseline: {baseline}")
            success &= run_naive_baselines_evaluation(baseline, setups=setups, domain=args.domain)
    if success:
        print("\nAll evaluations completed successfully!")
    else:
        print("\nSome evaluations failed")
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)