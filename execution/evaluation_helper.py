from dataclasses import dataclass
import os
import re
from typing import List, Dict, Optional

from utils.experiment_manager import load_config
@dataclass
class Experiment:
    name: str
    path: str
    algorithm: Optional[str]
    pointer_policy: Optional[str]
    attention_type: Optional[str]
    setting_type: Optional[str]
    grid_size: Optional[int]
    checkpoint_path: Optional[str]
    checkpoint: Optional[str]
    episode: Optional[str]
    zero_shot: Optional[str] = None
    zero_shot_target_grids_size : Optional[str] = None
    zero_shot_target_grids_size_detail : Optional[str] = None
    zero_shot_mode: Optional[str] = None  # None, "fixteam", or "fixteam_transition"
    domain: str = "wildfire"  # "wildfire" or "rideshare"
    ablation_no_gnn_critic: bool = False  # True when critic is MLPValueCritic
    
ALGORITHMS = ("COMA", "Vanilla_MAPPO", "QMIX", "PLATO","DGN","DICG","MOHITO","DEEPSET")
general_config = load_config("General_Config")


_alg_prefix_re = re.compile(r"^(COMA|Vanilla_MAPPO|QMIX|PLATO|DGN|DICG|MOHITO|DEEPSET|DEEPSET)", re.IGNORECASE)
# Captures full variant: mlp, lstm, mlp_noteamstats, lstm_noteamstats, mlp_nopointer, lstm_nopointer
_policy_re = re.compile(r"(?:^|_)pointer_((?:lstm|mlp)(?:_noteamstats|_nopointer)?)(?:_|$)", re.IGNORECASE)
_attention_re = re.compile(r"(?:^|_)(additive|dotproduct)(?:_|$)", re.IGNORECASE)
_grid_re = re.compile(r"(\d+)[xX](\d+)")
_nognn_re = re.compile(r"(?:^|_)nognn(?:_|$)", re.IGNORECASE)

def scan_experiments_with_frequency(
    root_dir="./checkpoints",
    baseline='',
    frequent=200,gridsize=3,
    starting_episode=500000,
    max_episode=100000,
    zero_shot=None,
    zero_shot_target_grids_size: Optional[int]=None,
    zero_shot_mode: Optional[str]=None,
    filter_pointer_policy: Optional[str]=None,
    filter_attention_type: Optional[str]=None,
    filter_nognn: Optional[bool]=None,
) -> List[List[Dict]]:
    """
    Walks `root_dir`, finds subfolders containing `baseline` in their name,
    then collects checkpoint files at regular frequency intervals.
    
    Args:
        root_dir: Root directory to scan
        baseline: Substring to match in folder names (case-insensitive, e.g., 'mohito', 'pointer')
        frequent: Frequency interval for checkpoints (e.g., 200 means collect *200.pt, *400.pt, *600.pt...)
    
    Returns:
        List of lists, where each inner list contains experiment dicts for checkpoints
        at each frequency milestone (200, 400, 600, etc.)
    """
    results: List[List[Dict]] = []
    
    if not os.path.isdir(root_dir):
        return results
    
    # Convert baseline to lowercase for case-insensitive matching
    baseline_lower = baseline.lower()
    
    for name in sorted(os.listdir(root_dir)):
        sub_path = os.path.join(root_dir, name)
        if not os.path.isdir(sub_path):
            continue
        
        # Filter: only process folders with baseline name in them
        if baseline_lower and baseline_lower not in name.lower():
            continue
        
        # Normalize folder name
        norm_name = name.replace("PinterLearner", "PLATO")
        norm_name = norm_name.replace("stochastically_completable", "stochasticallycompletable")
        norm_name = norm_name.replace("time_critical", "timecritical")
        norm_name = norm_name.replace("resource_limited", "resourcelimited")
        
        # Extract metadata (algorithm, policy, attention, setting, grid_size)
        alg_match = _alg_prefix_re.search(norm_name)
        algorithm = None
        if alg_match:
            alg_raw = alg_match.group(1)
            algorithm = next((a for a in ALGORITHMS if a.lower() == alg_raw.lower()), alg_raw)
        
        pointer_policy = None
        pm = _policy_re.search(norm_name)
        if pm:
            pointer_policy = pm.group(1).lower()

        no_gnn_critic = bool(_nognn_re.search(norm_name))

        # Attention
        attention_type = None
        am = _attention_re.search(norm_name)
        if am:
            attention_type = am.group(1).lower()

        # Exclude nognn folders unless explicitly requested, and vice-versa.
        # Without this, pointer_mlp+additive scans would pick up nognn folders too.
        if filter_nognn is True and not no_gnn_critic:
            continue
        if filter_nognn is not True and no_gnn_critic:
            continue

        # Filter by policy/attention type if requested
        if filter_pointer_policy and (pointer_policy is None or pointer_policy != filter_pointer_policy.lower()):
            continue
        if filter_attention_type and (attention_type is None or attention_type != filter_attention_type.lower()):
            continue

        setting_type = None
        tokens = re.split(r"_+", norm_name.lower())
        # Match longest setting first to avoid partial matches (e.g. "TO" in "MOHITO","DEEPSET")
        all_settings = sorted(
            (general_config.SETUPS or []) + (general_config.configuration_pkls or []),
            key=len, reverse=True
        )
        for s in all_settings:
            # Use word-boundary regex to avoid substring false positives
            if re.search(r'(?:^|[_\-])' + re.escape(s) + r'(?:[_\-]|$)', norm_name, re.IGNORECASE):
                setting_type = s
                break
        
        grid_size = None
        gm = _grid_re.search(norm_name)
        if gm:
            try:
                grid_size = int(gm.group(1))
            except ValueError:
                grid_size = None
        if grid_size and grid_size !=gridsize:
            continue 
        # Find all .pt files in the subfolder
        pt_files = []
        if os.path.isdir(sub_path):
            for file in os.listdir(sub_path):
                if file.endswith('.pt'):
                    pt_files.append(file)
        
        # Filter for files ending with frequency pattern (*200.pt, *400.pt, *600.pt, etc.)
        frequency_checkpoints = []
        for pt_file in pt_files:
            # Match files ending with {number}.pt (e.g., "checkpoint_200.pt", "model_400.pt")
            match = re.search(r'(\d+)\.pt$', pt_file)
            if match:
                checkpoint_num = int(match.group(1))
                # Check if it's a multiple of the frequency
                if checkpoint_num % frequent == 0 and checkpoint_num >= starting_episode and checkpoint_num <= max_episode:
                    frequency_checkpoints.append((checkpoint_num, pt_file))
        
        # Sort by checkpoint number
        frequency_checkpoints.sort(key=lambda x: x[0])
        
        # Create an experiment dict for each frequency checkpoint
        folder_experiments = []
        for checkpoint_num, pt_file in frequency_checkpoints:
            checkpoint_path = os.path.join(sub_path, pt_file)
            folder_experiments.append(Experiment(
                name=f"{name}_step_{checkpoint_num}",
                episode=f"{checkpoint_num}",
                path=sub_path,
                algorithm=algorithm,
                pointer_policy=pointer_policy,
                attention_type=attention_type,
                setting_type=setting_type,
                grid_size=grid_size,
                checkpoint_path=checkpoint_path,
                checkpoint=pt_file,
                zero_shot=zero_shot,
                zero_shot_target_grids_size_detail='',
                zero_shot_target_grids_size=zero_shot_target_grids_size,
                zero_shot_mode=zero_shot_mode,
                domain="rideshare" if "rideshare" in name.lower() else "wildfire",
                ablation_no_gnn_critic=no_gnn_critic,
            ))
        
        # Add this folder's experiments to results
        if folder_experiments:
            results.append(folder_experiments)
    
    return results