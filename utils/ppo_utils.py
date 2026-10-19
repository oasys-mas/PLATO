# mappo_utils.py

import torch
import numpy as np
import math

def safe_int(x, default=0):
    """Convert to int safely."""
    if x is None:
        return default
    try:
        v = float(x)
    except:
        return default
    if not math.isfinite(v):
        return default
    return int(v)

def in_range(agent_tensor, task_tensor, configuration):
    """
    Checks if task is within agent's range using Manhattan distance.
    """
    # Get (y, x) for agent and task
    if agent_tensor.ndim == 2:
        agent_tensor = agent_tensor[0]
    if task_tensor.ndim == 2:
        task_tensor = task_tensor[0]

    ay, ax = agent_tensor[0].item(), agent_tensor[1].item()
    ty, tx = task_tensor[0].item(), task_tensor[1].item()

    # Get scalar attack range
    if isinstance(configuration.agent_config.attack_range, torch.Tensor):
        attack_range = configuration.agent_config.attack_range[0].item()
    else:
        attack_range = configuration.agent_config.attack_range

    return abs(ay - ty) + abs(ax - tx) <= attack_range


def build_action_mask(agent_obs, configuration):
    grid_size = configuration.grid_height * configuration.grid_width
    max_tasks = grid_size
    result = []

    def in_range(agent_tensor, task_tensor, config):
        ay, ax = agent_tensor[0].item(), agent_tensor[1].item()
        ty, tx = task_tensor[0].item(), task_tensor[1].item()
        max_range = int(config.agent_config.attack_range[0].item())
        return abs(ay - ty) + abs(ax - tx) <= max_range

    active_idx = 0
    for task in agent_obs['tasks'][0]:
        intensity = task[2].item()

        if intensity >= 1 and in_range(agent_obs['self'][0], task, configuration):
            result.append((active_idx, 0))   # fight
            result.append((active_idx, -1))  # noop
        else:
            result.append((active_idx, -85))  # fight not allowed
            result.append((active_idx, -85))  # noop not allowed

        active_idx += 1

    # Pad remaining tasks (if fewer than max_tasks)
    while active_idx < max_tasks:
        result.append((active_idx, -85))
        result.append((active_idx, -85))
        active_idx += 1

    return result

def index_to_action(index):
    task_id = index // 2
    action_type = 0 if index % 2 == 0 else -1
    return task_id, action_type
def action_to_index(task_id, action_type):
    return task_id * 2 + (0 if action_type == 0 else 1)



def discount_cumsum(rewards, discount):
    """Compute discounted cumulative sums of vectors."""
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = rewards[t] + discount * running_add
        discounted[t] = running_add
    return discounted

def pad_to_last(arr, total_length, axis=0, pad_value=0):
    """Pad a numpy array to total_length along the specified axis."""
    pad_size = total_length - arr.shape[axis]
    if pad_size <= 0:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, pad_size)
    return np.pad(arr, pad_width, mode='constant', constant_values=pad_value)

def filter_valids(tensor, valids):
    """Filter valid steps from a padded tensor."""
    return [t[:int(v)] for t, v in zip(tensor, valids)]

def compute_advantages(discount, gae_lambda, max_path_length, baselines, rewards):
    """Compute GAE-based advantages."""
    advantages = []
    for path_baseline, path_reward in zip(baselines, rewards):
        path_baseline = np.append(path_baseline, 0)
        deltas = path_reward + discount * path_baseline[1:] - path_baseline[:-1]
        advantages.append(discount_cumsum(deltas, discount * gae_lambda))
    return np.array(advantages, dtype=np.float32)

def flatten_batch(tensor):
    """Flatten leading dimensions (e.g., [B, T, ...] -> [B*T, ...])"""
    shape = tensor.shape
    return tensor.reshape(-1, *shape[2:]) if len(shape) > 2 else tensor.reshape(-1)

def explained_variance_1d(y, y_pred):
    """Compute 1D explained variance."""
    assert y.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y)
    return 1 - np.var(y - y_pred) / (var_y + 1e-8)

def normalize(x, eps=1e-8):
    """Normalize a vector."""
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + eps)
def clip_gradients(parameters, max_norm):
    torch.nn.utils.clip_grad_norm_(parameters, max_norm)