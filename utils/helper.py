# helper.py
import argparse
from enum import Enum
import logging
import pickle
import random
import sys
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.experiment_manager import load_config
# TODO, load from params
DEBUG=True
from utils.custom_configuration import create_wildfire_config
import torch._dynamo as dynamo
dynamo.config.suppress_errors = True  # do this before any torch.compile happens
general_config = load_config("General_Config")


# === checkpoint_utils.py ===
import os, re, json, torch, tempfile, shutil
from typing import Dict, Any, Optional, Union, Iterable

def shuffle_task_features(task_features):
    """
    shuffle task features.
    
    Args:
        task_features (torch.Tensor): Task features tensor of shape [num_tasks, task_feat_dim]
        
    Returns:
        torch.Tensor: Shuffled task features with same shape
    """
    if task_features.shape[0] == 0:
        return task_features
    shuffleed_task_features=task_features[torch.randperm(len(task_features))]
    return shuffleed_task_features



def get_training_seed(episode, general_config):
    """
    Get the seed for training based on configuration.
    
    Args:
        episode (int): Current episode number
        general_config: Configuration object with Incremental_Training_Seed and Training_Starting_Seed
        
    Returns:
        int: The seed to use for this episode
    """
    if general_config.Incremental_Training_Seed:
        return general_config.Training_Starting_Seed + episode
    else:
        return general_config.Training_Starting_Seed


def get_validation_seed(episode, general_config):
    """
    Get the seed for validation based on configuration.
    
    Args:
        episode (int): Current episode number
        general_config: Configuration object with Incremental_Training_Seed and Training_Starting_Seed
        
    Returns:
        int: The seed to use for this episode
    """
    if general_config.Incremental_Validation_Seed:
        return general_config.Validation_Starting_Seed + episode
    else:
        return general_config.Validation_Starting_Seed

def get_execution_seed(episode, general_config):
    """
    Get the seed for execution based on configuration.
    
    Args:
        episode (int): Current episode number
        general_config: Configuration object with Incremental_Training_Seed and Training_Starting_Seed
        
    Returns:
        int: The seed to use for this episode
    """
    if general_config.Incremental_Execution_Seed:
        return general_config.Execution_Starting_Seed + episode
    else:
        return general_config.Execution_Starting_Seed

def _atomic_torch_save(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)  # atomic on POSIX
    except Exception:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        raise

def save_checkpoint(
    payload: Dict[str, Any],
    base_path: str,
    step: Optional[Union[int, str]] = None,
    keep_last: int = 5,
    file_prefix: str = "ckpt",
) -> str:
    """
    payload: dict of anything with .state_dict() or plain values (we handle both).
            e.g. {'actor': actor, 'critic': critic, 'opt_actor': optA, 'opt_critic': optC,
                  'step': step, 'config': config_dict}
    base_path: directory or file-prefix path (e.g., '/path/to/checkpoints/PLATO')
    step: number or string appended to filename
    keep_last: keep only most recent K checkpoints (by step number if available, else mtime)
    file_prefix: filename prefix
    """
    # Convert objects with .state_dict() into dictionaries
    to_save = {}
    for k, v in payload.items():
        if hasattr(v, "state_dict") and callable(v.state_dict):
            try:
                to_save[k] = v.state_dict()
            except Exception:
                to_save[k] = v  # fallback
        else:
            to_save[k] = v

    # Build path
    dir_path = os.path.dirname(base_path) if os.path.splitext(base_path)[1] else base_path
    name = os.path.basename(base_path) if os.path.splitext(base_path)[1] else os.path.basename(base_path.rstrip("/"))
    if not name: name = "checkpoint"
    os.makedirs(dir_path or ".", exist_ok=True)

    suffix = f"_{step}" if step is not None else ""
    path = os.path.join(dir_path, f"{file_prefix}_{name}{suffix}.pt")

    # Save atomically
    _atomic_torch_save(to_save, path)

    # Prune old files
    if keep_last is not None and keep_last > 0:
        pattern = re.compile(rf"^{re.escape(file_prefix)}_{re.escape(name)}_(\d+)\.pt$")
        cands = []
        for f in os.listdir(dir_path):
            if f.startswith(f"{file_prefix}_{name}_") and f.endswith(".pt"):
                m = pattern.match(f)
                step_num = int(m.group(1)) if m else None
                fp = os.path.join(dir_path, f)
                cands.append((step_num, os.path.getmtime(fp), fp))
        # Sort by step if available, else by mtime
        cands.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else x[1]))
        while len(cands) > keep_last:
            _, _, fp = cands.pop(0)
            try: os.remove(fp)
            except FileNotFoundError: pass

    return path


def orthogonal_init(module, gain=1.0):
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
def collect_future_rewards(rewards, k=3):
    """Collect future rewards as list of lists (per step)"""
    future_rewards = []
    for i in range(len(rewards)):
        seq = []
        for j in range(k):
            if i + j < len(rewards):
                seq.append(rewards[i + j])
            else:
                seq.append(0.0)
        future_rewards.append(seq)
    return future_rewards

def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight MAPPO Training")
    
    # DICG
    
    
    
    
    
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint path',default='') 
    parser.add_argument('--experiment_name', type=str, help='name of the experiment',default='') # pointer learner - COMA - QMIX

    
      
    attention_type = ['additive', 'dotproduct','cosine_sim','']# only for pointer learner
    parser.add_argument('--attention_type', type=str, choices=attention_type, default="", help='attention type of pointer policy')


  
  

    parser.add_argument('--setting_type', type=str, choices=general_config.SETUPS, default="", help='setting type of the grid')
    parser.add_argument('--configuration_pkl', type=str, choices=general_config.configuration_pkls, default="", help='pkl configuration to use (overrides setting_type if provided)')
    parser.add_argument('--grid_size', type=int, choices=[2,3,4,5,6], default=None, help='size of the grid')
    pointer_policies = [
        'pointer_lstm', 'pointer_mlp', '',
        # ablation variants (actor side — encoded in checkpoint name automatically)
        'pointer_mlp_noteamstats', 'pointer_lstm_noteamstats',
        'pointer_mlp_nopointer',   'pointer_lstm_nopointer',
    ]
    parser.add_argument('--pointer_policy', type=str, choices=pointer_policies, default="", help='policy to utilize')

    parser.add_argument("--episodes", type=int, default=30000,
                        help="Number of episodes")
    parser.add_argument("--debug", action="store_true", default=True,
                        help="Enable debug output")
    parser.add_argument("--isframeopen", action="store_true", default=True,
                        help="Enable isframeopen ")
    parser.add_argument("--critic_type", choices=["deepset", "gnn"], default="gnn")
    parser.add_argument("--mohito_config_file",  default="baselines/MOHITO/wildfire/configs/wf_frank_no_drop.yaml")

    # ── Ablation toggle: critic side only (actor ablations encoded in pointer_policy name) ──
    parser.add_argument('--ablation_no_gnn_critic', action='store_true', default=False,
                        help='Ablation: replace GNN critic with a simple MLP critic')

    naive_baselines = ['noop', 'random', 'strongest', 'weakest']

    general = parser.add_argument_group('General')

    # general.add_argument('namespace', type=str,default="test", help='namespace for the configuration group')
    # general.add_argument('baseline', type=str, choices=naive_baselines, help='naive baseline policy to utilize')
    general.add_argument('--cuda', action='store_true', help='Utilize cuda if available')
    general.add_argument('--threads', type=int, default=1, help='utilize this many threads for the experiment')
    general.add_argument('--override', action='store_true', help='override file safety checks and allow data overwrite')
    general.add_argument('--silence', action='store_true', help='silence all stdout output from the script')

    reproducible = parser.add_argument_group('Reproducibility')
    reproducible.add_argument('--seed', type=int, default=42, help='seed for the experiment')
    reproducible.add_argument('--dataset_global_seed', type=int, default=None, help='seed for initializing the configuration dataset')

    validation = parser.add_argument_group('Validation')
    validation.add_argument('--testing_episodes', type=int, default=1, help='number of episodes to run per test')

    
    return parser.parse_args()
FORMAT_STRING = "[%(asctime)s] [%(levelname)8s] [%(name)10s] [%(filename)21s:%(lineno)03d] %(message)s"

def save_hyperparameters(writer, args, caller_globals=None):
    """
    Collects hyperparameters dynamically from both global constants (UPPERCASE)
    and argparse args, and logs them to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter instance.
        args: Namespace object from argparse.
        caller_globals: (dict) globals() from the calling module, for dynamic UPPERCASE scanning.

    Returns:
        hparams (dict): Combined hyperparameters and args.
    """
    # Use caller's globals if passed, otherwise use local globals
    caller_globals = caller_globals or globals()

    # Collect all UPPERCASE constants (hyperparameters)
    hyperparams = {
        k: v for k, v in caller_globals.items()
        if k.isupper() and not k.startswith("__") and isinstance(v, (int, float, str, bool))
    }

    # Merge argparse arguments
    hyperparams.update(vars(args))

    # Log to TensorBoard (human-readable)
    formatted = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
    writer.add_text("Hyperparameters", formatted)


    return hyperparams

def setup_logging(name):
    logging.basicConfig(format=FORMAT_STRING, level=logging.INFO)
    return logging.getLogger(name)

import os, random
import numpy as np
import torch

def set_global_seed(seed: int, deterministic: bool = True) -> None:
    # Python / NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # not just current device

    if deterministic:
        # Deterministic behavior (may reduce speed, some ops may warn/fallback)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Required for some CUDA deterministic matmul paths
        # (PyTorch will complain if needed and not set)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class EnvironmentType(Enum):
    WILDFIRE = "wildfire"
    
#wilfire
def load_configuration(setting_type, grid_size=None, zero_shot_mode=None, 
                       source_setting_type=None, target_grid_size=None):
    """
    Load wildfire configuration.
    
    Args:
        setting_type: The setting type to load
        grid_size: Grid size (optional, unused directly but kept for compatibility)
        zero_shot_mode: None, "fixteam", or "fixteam_transition"
        source_setting_type: Original training setting (for fixteam modes)
        target_grid_size: Target grid size (for fixteam modes)
    
    Returns:
        WildfireConfiguration
    """
    termination_reward=0.0
    # if grid_size == 2:
    #     termination_reward = 0.0
    # if grid_size == 3:
    #     termination_reward = 0.0
    # elif grid_size == 4:
    #     termination_reward = 0.0
    # elif grid_size == 5:
    #     termination_reward = 0.0
        
        
    config = create_wildfire_config(
            setting_type=setting_type,
            termination_reward=termination_reward,

            zero_shot_mode=zero_shot_mode,
            source_setting_type=source_setting_type,
            target_grid_size=target_grid_size,
        )
    return config        
    



def load_configuration_pkl(configuration):
    
    """
    Load pkl configuration.
    
    Args:
        configuration
    
    Returns:
        Configuration object (RideshareConfiguration or WildfireConfiguration)
    """

    # Try to load from pickle file first
    if configuration:
        try:
            dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            path = os.path.join(dir, "pkls", f"{configuration}.pkl")
            print(f"Loading config from {path}")
            with open(path, 'rb') as f:
                config=pickle.load(f)
                _fix_pkl_dtypes(config)
                return config
        except (FileNotFoundError, Exception) as e:
            print(f"Failed to load pickle file: {e}")
    else:
        return 


def _fix_pkl_dtypes(config):
    """Cast int32 tensor fields in a loaded pkl config to float32.
    
    Some pkl configs store `possible_capacities` and `capacity_probabilities`
    as int32 tensors, but the environment's capacity transition expects float32.
    """
    ac = getattr(config, 'agent_config', None)
    if ac is None:
        return
    for attr in ('possible_capacities', 'capacity_probabilities'):
        val = getattr(ac, attr, None)
        if isinstance(val, torch.Tensor) and val.dtype != torch.float32:
            setattr(ac, attr, val.float())


def resolve_configuration(args):
    """
    Unified configuration loader for all algorithms.
    If configuration_pkl is provided (not None/empty), load from pkl.
    Otherwise, fall back to load_configuration (wildfire setups.json).
    
    Args:
        args: Parsed argparse namespace with configuration_pkl, setting_type, grid_size
    
    Returns:
        Configuration object (RideshareConfiguration or WildfireConfiguration)
    """
    pkl = getattr(args, 'configuration_pkl', None)
    setting_type = getattr(args, 'setting_type', None)

    # Wildfire runs use setup names like "setup1_2x3". The shared parser currently
    # defaults configuration_pkl to a rideshare pickle, which should not override a
    # valid wildfire setup selection.
    if setting_type in (general_config.SETUPS or []):
        return load_configuration(setting_type, getattr(args, 'grid_size', None))

    if pkl and pkl.strip():
        return load_configuration_pkl(pkl)
    return load_configuration(setting_type, getattr(args, 'grid_size', None))


def debug_print(*args, **kwargs):
    """Print only if debug mode is enabled"""
    if DEBUG:
        print(*args, **kwargs)
       

def enable_gpu_optimizations():
    """Enable various GPU optimizations for better performance"""
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Performance optimizations - disable determinism for speed
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True 
        

        
        print("GPU optimizations enabled")
    else:
        print(" No GPU available, using CPU")
def setup_torch():
    
    torch.use_deterministic_algorithms(True, warn_only=True)
import torch

def count_zero_suppressant_agents(state) -> int:
    """
    Count agents whose suppressant level is exactly 0.

    Assumes state.suppressants is a tensor of shape (1, num_agents)
    or (batch_size, num_agents). For non-batched WildfireState, use
    the first row.
    """
    suppressants = state.suppressants

    # If there is a batch dimension, pick the first batch element
    if suppressants.dim() == 2 and suppressants.size(0) == 1:
        suppressants = suppressants[0]

    zero_mask = (suppressants == 0)
    return int(zero_mask.sum().item())

def optimize_tensor_conversion(data, device, dtype=torch.float32, non_blocking=True):
    """Optimized tensor conversion with minimal memory operations"""
    if isinstance(data, torch.Tensor):
        if data.device == device and data.dtype == dtype:
            return data  # Already optimal
        return data.to(device=device, dtype=dtype, non_blocking=non_blocking)
    elif isinstance(data, (list, tuple)):
        if isinstance(data[0], torch.Tensor):
            # Batch convert tensors
            return torch.stack([optimize_tensor_conversion(x, device, dtype, non_blocking) for x in data])
        else:
            return torch.tensor(data, device=device, dtype=dtype)
    else:
        return torch.tensor(data, device=device, dtype=dtype)
def setup_tensorboard_logging(args):
    """Setup tensorboard logging"""
    nognn_suffix = '_nognn' if getattr(args, 'ablation_no_gnn_critic', False) else ''
    base = (args.experiment_name + '_' + args.pointer_policy + '_' + args.attention_type + '_'
            + args.setting_type + '_' + args.configuration_pkl + '_'
            + str(args.grid_size) + 'X' + str(args.grid_size) + nognn_suffix)
    log_path = f"tensorboard/{base}_{int(time.time())}"
    checkpoint_log_path = f"checkpoints/{base}"
    writer = SummaryWriter(log_dir=log_path)
    return writer, checkpoint_log_path    


# === checkpoint_utils.py ===
import os, re, json, torch, tempfile, shutil
from typing import Dict, Any, Optional, Union, Iterable

def _atomic_torch_save(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)  # atomic on POSIX
    except Exception:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        raise

def save_checkpoint(
    payload: Dict[str, Any],
    base_path: str,
    episode: int = 0,
    file_prefix: str = "ckpt",
) -> str:
    """
    payload: dict of anything with .state_dict() or plain values (we handle both).
            e.g. {'actor': actor, 'critic': critic, 'opt_actor': optA, 'opt_critic': optC,
                  'step': step, 'config': config_dict}
    base_path: directory or file-prefix path (e.g., '/path/to/checkpoints/PLATO')
    step: number or string appended to filename
    keep_last: keep only most recent K checkpoints (by step number if available, else mtime)
    file_prefix: filename prefix
    """
    # Convert objects with .state_dict() into dictionaries
    to_save = {}
    for k, v in payload.items():
        if hasattr(v, "state_dict") and callable(v.state_dict):
            try:
                to_save[k] = v.state_dict()
            except Exception:
                to_save[k] = v  # fallback
        else:
            to_save[k] = v

    # Build path
    dir_path = os.path.dirname(base_path) if os.path.splitext(base_path)[1] else base_path
    name = os.path.basename(base_path) if os.path.splitext(base_path)[1] else os.path.basename(base_path.rstrip("/"))
    if not name: name = "checkpoint"
    os.makedirs(dir_path or ".", exist_ok=True)


    path = os.path.join(dir_path, f"{file_prefix}_episode_{episode}.pt")

    # Save atomically
    _atomic_torch_save(to_save, path)
    path = os.path.join(dir_path, f"{file_prefix}.pt")

    # Save atomically
    _atomic_torch_save(to_save, path)



    return path
