#PLATO - Wildfire
from copy import deepcopy
import sys
import tabulate
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import logging
import os
import time
import torch.nn.functional as F
from datetime import datetime

from modules.centralized_critic_modules import GNNValueCritic, MLPValueCritic

from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from utils.graph_utils_ptr import action_info, episode_infoes, get_episode_fire_counts, print_wildfire_grid
from utils.network_logger import log_gradient_flow, log_gradient_flow_graph, log_model_architecture, log_network_graph, log_training_flow_graph, save_to_csv
from policies.pointer_GRU_policy import Pointer_GRU_Actor
from policies.pointer_LSTM_policy import Pointer_LSTM_Actor
from policies.pointer_MLP_policy import Pointer_MLP_Actor
from utils.observation_graph_helper import observations_to_features,create_global_bipartite_graph_wildfire,extract_wildfire_action_features,normalize_wildfire_agent_features
from utils.helper import *
from utils.experiment_manager import load_config
from utils.ppo_utils import safe_int
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="free_range_zoo")

config = load_config("PLATO")
general_config = load_config("General_Config")
TUNING_MODE = general_config.TUNING_MODE

if general_config.DB == "postgres":
    from experiment_logger_postgres import ExperimentDB, RunLogger
else:
    from experiment_logger_sqlite import ExperimentDB, RunLogger
    
# ============================================================
# DEBUG LOGGER - Comprehensive Pipeline Diagnostics
# ============================================================
class DebugLogger:
    """Centralized debug logging for pipeline verification"""
    
    def __init__(self, log_file="outputs.txt", enabled=False, detailed_episodes=5, detailed_steps=3):
        self.enabled = enabled
        self.log_file = log_file
        self.step_count = 0
        self.episode_count = 0
        self.detailed_episodes = detailed_episodes  # Only log detailed info for first N episodes
        self.detailed_steps = detailed_steps  # Only log detailed step info for first N steps per episode
        
        # ACTION PATTERN TRACKING
        self.episode_actions = {}  # {agent: [action_indices]}
        self.global_action_counts = {}  # {agent: {action_idx: count}}
        self.action_history_window = []  # Rolling window of last N episode action distributions
        self.window_size = 50  # Track last 50 episodes
        
        # Clear existing log
        if enabled:
            with open(log_file, 'w') as f:
                f.write(f"=== PLATO Debug Log Started at {datetime.now()} ===\n")
                f.write(f"Detailed logging for first {detailed_episodes} episodes, first {detailed_steps} steps each\n\n")
    
    def reset_episode_actions(self):
        """Reset per-episode action tracking"""
        self.episode_actions = {}
    
    def track_action(self, agent, action_idx, num_available_actions):
        """Track action selection for pattern analysis"""
        if agent not in self.episode_actions:
            self.episode_actions[agent] = []
        self.episode_actions[agent].append((action_idx, num_available_actions))
        
        # Update global counts
        if agent not in self.global_action_counts:
            self.global_action_counts[agent] = {}
        if action_idx not in self.global_action_counts[agent]:
            self.global_action_counts[agent][action_idx] = 0
        self.global_action_counts[agent][action_idx] += 1
    
    def log_action_pattern_summary(self, episode):
        """Log action pattern analysis for the episode"""
        if not self.enabled:
            return
            
        with open(self.log_file, 'a') as f:
            f.write(f"\n[ACTION_PATTERN] Episode {episode} Action Distribution:\n")
            
            episode_summary = {}
            for agent, actions in self.episode_actions.items():
                # Count actions in this episode
                action_counts = {}
                for action_idx, num_actions in actions:
                    if action_idx not in action_counts:
                        action_counts[action_idx] = 0
                    action_counts[action_idx] += 1
                
                total = sum(action_counts.values())
                f.write(f"[ACTION_PATTERN]   {agent}: ")
                
                # Sort by action index for consistent display
                sorted_actions = sorted(action_counts.items())
                parts = []
                for action_idx, count in sorted_actions:
                    pct = 100 * count / total if total > 0 else 0
                    parts.append(f"action_{action_idx}={count}({pct:.1f}%)")
                f.write(", ".join(parts) + "\n")
                
                episode_summary[agent] = action_counts
            
            # Store for rolling window analysis
            self.action_history_window.append(episode_summary)
            if len(self.action_history_window) > self.window_size:
                self.action_history_window.pop(0)
            
            # Every 100 episodes, log global pattern
            if episode % 100 == 0:
                f.write(f"\n[ACTION_PATTERN] === GLOBAL ACTION PATTERN (Episodes 1-{episode}) ===\n")
                for agent, counts in self.global_action_counts.items():
                    total = sum(counts.values())
                    f.write(f"[ACTION_PATTERN]   {agent}: ")
                    sorted_counts = sorted(counts.items())
                    parts = []
                    for action_idx, count in sorted_counts:
                        pct = 100 * count / total if total > 0 else 0
                        parts.append(f"action_{action_idx}={count}({pct:.1f}%)")
                    f.write(", ".join(parts) + "\n")
                
                # Log rolling window trend (last N episodes)
                if len(self.action_history_window) >= 10:
                    f.write(f"\n[ACTION_PATTERN] === RECENT TREND (Last {len(self.action_history_window)} episodes) ===\n")
                    recent_counts = {}
                    for ep_summary in self.action_history_window:
                        for agent, counts in ep_summary.items():
                            if agent not in recent_counts:
                                recent_counts[agent] = {}
                            for action_idx, count in counts.items():
                                if action_idx not in recent_counts[agent]:
                                    recent_counts[agent][action_idx] = 0
                                recent_counts[agent][action_idx] += count
                    
                    for agent, counts in recent_counts.items():
                        total = sum(counts.values())
                        f.write(f"[ACTION_PATTERN]   {agent}: ")
                        sorted_counts = sorted(counts.items())
                        parts = []
                        for action_idx, count in sorted_counts:
                            pct = 100 * count / total if total > 0 else 0
                            parts.append(f"action_{action_idx}={count}({pct:.1f}%)")
                        f.write(", ".join(parts) + "\n")
    
    def _should_log_detailed(self):
        """Check if we should log detailed info"""
        return self.episode_count <= self.detailed_episodes
    
    def _should_log_step_detailed(self):
        """Check if we should log detailed step info"""
        return self.episode_count <= self.detailed_episodes and self.step_count < self.detailed_steps
    
    def log(self, msg, category="INFO"):
        if not self.enabled:
            return
        with open(self.log_file, 'a') as f:
            f.write(f"[{category}] {msg}\n")
    
    def log_section(self, title):
        if not self.enabled:
            return
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*60}\n{title}\n{'='*60}\n")
    
    def log_episode_start(self, episode):
        self.episode_count = episode
        self.step_count = 0
        if self._should_log_detailed():
            self.log_section(f"EPISODE {episode} START")
    
    def log_step(self, step):
        self.step_count = step
        if self._should_log_step_detailed():
            self.log(f"--- Step {step} ---", "STEP")
    
    def log_action_features(self, agent, env_id, action_features, action_mapping):
        """Log what features and mappings are available for action selection"""
        if not self._should_log_step_detailed():
            return
        self.log(f"Agent={agent}, Env={env_id}", "ACTION_FEATURES")
        self.log(f"  action_features shape: {action_features.shape}", "ACTION_FEATURES")
        self.log(f"  action_features:\n{action_features}", "ACTION_FEATURES")
        self.log(f"  action_mapping: {action_mapping}", "ACTION_FEATURES")
        # In this codebase, action_features already includes NOOP as the LAST row.
        num_tasks_excluding_noop = max(int(action_features.shape[0]) - 1, 0)
        self.log(f"  num_tasks (excluding NOOP): {num_tasks_excluding_noop}", "ACTION_FEATURES")
    
    def log_actor_forward(self, task_features, agent_features, logits):
        """Log actor network inputs and outputs"""
        if not self._should_log_step_detailed():
            return
        self.log(f"Actor Forward Pass:", "ACTOR")
        self.log(f"  task_features shape: {task_features.shape}", "ACTOR")
        self.log(f"  task_features:\n{task_features}", "ACTOR")
        self.log(f"  agent_features shape: {agent_features.shape}", "ACTOR")
        self.log(f"  agent_features:\n{agent_features}", "ACTOR")
    
    def log_action_selection(self, action, log_prob, action_mapping, selected_action):
        """Log the action selection process"""
        if not self._should_log_step_detailed():
            return
        self.log(f"Action Selection:", "ACTION")
        self.log(f"  sampled action index: {action.item()}", "ACTION")
        self.log(f"  log_prob: {log_prob.item():.4f}", "ACTION")
        self.log(f"  mapped to env_action: {selected_action}", "ACTION")
        
        # Check mapping validity
        if action.item() >= len(action_mapping):
            self.log(f"  ERROR: action index {action.item()} >= mapping length {len(action_mapping)}", "ACTION")
    
    def log_critic_forward(self, agent_nodes, fire_nodes, edge_index, value):
        """Log critic network inputs and output"""
        if not self._should_log_step_detailed():
            return
        self.log(f"Critic Forward Pass:", "CRITIC")
        self.log(f"  agent_nodes shape: {agent_nodes.shape}", "CRITIC")
        self.log(f"  agent_nodes:\n{agent_nodes}", "CRITIC")
        if fire_nodes is not None:
            self.log(f"  fire_nodes shape: {fire_nodes.shape}", "CRITIC")
            self.log(f"  fire_nodes:\n{fire_nodes}", "CRITIC")
        if edge_index is not None:
            self.log(f"  edge_index shape: {edge_index.shape}", "CRITIC")
        self.log(f"  output value: {value:.4f}", "CRITIC")
    
    def log_rewards(self, reward, done):
        """Log reward received"""
        if not self._should_log_step_detailed():
            return
        self.log(f"Reward: {reward:.4f}, Done: {done}", "REWARD")
    
    def log_buffer_add(self, agent, env_id, action, log_prob, value, reward):
        """Log what's being added to buffer"""
        if not self._should_log_step_detailed():
            return
        self.log(f"Buffer Add: agent={agent}, env={env_id}, action={action}, log_prob={log_prob:.4f}, value={value:.4f}", "BUFFER")
    
    def log_gae_computation(self, rewards, values, advantages, returns):
        """Log GAE computation results"""
        if not self._should_log_detailed():
            return
        self.log_section("GAE COMPUTATION")
        self.log(f"Num samples: {len(rewards)}", "GAE")
        self.log(f"Rewards: min={min(rewards):.4f}, max={max(rewards):.4f}, mean={np.mean(rewards):.4f}", "GAE")
        self.log(f"Values: min={min(values):.4f}, max={max(values):.4f}, mean={np.mean(values):.4f}", "GAE")
        self.log(f"Advantages: min={min(advantages):.4f}, max={max(advantages):.4f}, mean={np.mean(advantages):.4f}, std={np.std(advantages):.4f}", "GAE")
        self.log(f"Returns: min={min(returns):.4f}, max={max(returns):.4f}, mean={np.mean(returns):.4f}", "GAE")
    
    def log_training_update(self, batch_idx, actor_loss, critic_loss, entropy, approx_kl):
        """Log training update metrics"""
        if not self._should_log_detailed():
            return
        self.log(f"Update batch {batch_idx}: actor_loss={actor_loss:.4f}, critic_loss={critic_loss:.4f}, entropy={entropy:.4f}, kl={approx_kl:.4f}", "TRAIN")
    
    def log_episode_summary(self, episode, total_reward, steps, actor_loss, critic_loss):
        """Log episode summary - always log this"""
        self.log_section(f"EPISODE {episode} SUMMARY")
        self.log(f"Total Reward: {total_reward:.4f}", "SUMMARY")
        self.log(f"Steps: {steps}", "SUMMARY")
        self.log(f"Actor Loss: {actor_loss:.4f}", "SUMMARY")
        self.log(f"Critic Loss: {critic_loss:.4f}", "SUMMARY")

    def log_network_params(self, actor, critic):
        """Log network parameter statistics"""
        self.log_section("NETWORK PARAMETERS")
        
        # Actor stats
        actor_params = []
        for name, param in actor.named_parameters():
            if param.requires_grad:
                actor_params.append({
                    'name': name,
                    'shape': list(param.shape),
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item()
                })
        
        self.log("Actor Parameters:", "PARAMS")
        for p in actor_params:
            self.log(f"  {p['name']}: shape={p['shape']}, mean={p['mean']:.4f}, std={p['std']:.4f}", "PARAMS")
        
        # Critic stats
        critic_params = []
        for name, param in critic.named_parameters():
            if param.requires_grad:
                critic_params.append({
                    'name': name,
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item()
                })
        
        self.log("Critic Parameters:", "PARAMS")
        for p in critic_params[:5]:  # First 5
            self.log(f"  {p['name']}: mean={p['mean']:.4f}, std={p['std']:.4f}", "PARAMS")

# Global debug logger instance
debug_logger = DebugLogger(log_file="outputs.txt", enabled=general_config.log_in_file, detailed_episodes=5, detailed_steps=3)

    
    
if general_config.DEVICE != (None or ''):
    DEVICE=general_config.DEVICE
else:    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()  
    ]
)

class RolloutBuffer:
    """Efficient centralized rollout buffer for MAPPO"""
    
    def __init__(self, agents):
        self.agents = agents
        self.clear()
    
    def clear(self):
        # Centralized storage instead of per-agent
        self.data = {
            'obs': [], 'cent_obs': [], 'actions': [], 'log_probs': [],
            'values': [], 'rewards': [], 'masks': [], 'returns': [],
            'advantages': [], 'task_features': [], 'agent_features': [],
            'valid_indices': [], 'hidden_states': [], 'policy_hidden': [],
            'agents': []  # Track which agent each sample belongs to
        }
    
    def add_step(self, agent, **kwargs):
        """Add a single step's data - centralized storage"""
        for key, value in kwargs.items():
            if key in self.data:
                self.data[key].append(value)
        self.data['agents'].append(agent)
    
    def compute_gae(self, gamma=0.99, lam=0.95, v_last=None):
        """Centralized GAE computation - one value per environment like DICG"""
        if not self.data['rewards']:
            return
            
        # Convert to tensors
        rewards = torch.as_tensor(self.data['rewards'], device=DEVICE, dtype=torch.float32)
        values = torch.as_tensor(self.data['values'], device=DEVICE, dtype=torch.float32) 
        masks = torch.as_tensor(self.data['masks'], device=DEVICE, dtype=torch.float32)
        
        # Group by timestep.
        # IMPORTANT: In this runner, per-timestep samples are appended in the nested loop order:
        #   for agent in agents:
        #     for env_id in range(E):
        #       append(...)
        # i.e. agent-major then env-major within a timestep: index = a * E + e.
        # (Not env-major.)
        E = config.NUM_PARALLEL_ENVS
        A = len(self.agents)
        
        if rewards.numel() % (E * A) != 0:
            print(f"Warning: Total samples {rewards.numel()} not divisible by {E * A}")
            # Pad to make it divisible
            pad_size = (E * A) - (rewards.numel() % (E * A))
            rewards = torch.cat([rewards, torch.zeros(pad_size, device=DEVICE)])
            values = torch.cat([values, torch.zeros(pad_size, device=DEVICE)])
            masks = torch.cat([masks, torch.zeros(pad_size, device=DEVICE)])
        
        T = rewards.numel() // (E * A)
        
        # Reshape to [T, E*A] then group and average by environment per timestep
        rewards = rewards.view(T, E * A)
        values = values.view(T, E * A)
        masks = masks.view(T, E * A)
        
        # Average across agents per environment per timestep (like DICG)
        env_rewards = torch.zeros(T, E, device=DEVICE)
        env_values = torch.zeros(T, E, device=DEVICE)
        env_masks = torch.zeros(T, E, device=DEVICE)
        
        for e in range(E):
            for a in range(A):
                # Stored column order is (a * E + e)
                idx = a * E + e
                env_rewards[:, e] += rewards[:, idx] / A
                env_values[:, e] += values[:, idx] / A
                env_masks[:, e] = masks[:, idx]  # Take any agent's mask (should be same)
        
        # v_last per environment
        if v_last is None:
            v_last_tensor = torch.zeros(E, device=DEVICE)
        else:
            v_last_tensor = torch.as_tensor(v_last, device=DEVICE)
            if len(v_last_tensor.shape) > 1:
                v_last_tensor = v_last_tensor.mean(dim=-1)  # Average if per-agent
        
        # Compute GAE per environment (like DICG)
        advantages = torch.zeros(T, E, device=DEVICE)
        returns = torch.zeros(T, E, device=DEVICE)
        
        for e in range(E):
            r = env_rewards[:, e]
            v = env_values[:, e] 
            m = env_masks[:, e]
            
            next_v = torch.cat([v[1:], v_last_tensor[e].unsqueeze(0)])
            delta = r + gamma * next_v * m - v
            
            gae = 0.0
            for t in range(T - 1, -1, -1):
                gae = delta[t] + gamma * lam * m[t] * gae
                advantages[t, e] = gae
            returns[:, e] = advantages[:, e] + v
        
        # Expand back to per-agent format for compatibility
        expanded_advantages = torch.zeros(T * E * A, device=DEVICE)
        expanded_returns = torch.zeros(T * E * A, device=DEVICE)
        
        for t in range(T):
            for e in range(E):
                for a in range(A):
                    # Expand back in the same stored order: (a * E + e)
                    idx = t * (E * A) + a * E + e
                    expanded_advantages[idx] = advantages[t, e]
                    expanded_returns[idx] = returns[t, e]
        
        # Trim back to original size
        original_size = len(self.data['rewards'])
        expanded_advantages = expanded_advantages[:original_size]
        expanded_returns = expanded_returns[:original_size]
        
        # # Always normalize advantages
        if expanded_advantages.std() > 1e-8:
            expanded_advantages = (expanded_advantages - expanded_advantages.mean()) / (expanded_advantages.std() + 1e-8)
        
        self.data['advantages'] = expanded_advantages.cpu().tolist()
        self.data['returns'] = expanded_returns.cpu().tolist()
        
        # DEBUG: Log GAE computation
        debug_logger.log_gae_computation(
            self.data['rewards'], 
            self.data['values'],
            self.data['advantages'],
            self.data['returns']
        )
        
        if general_config.DEBUG:
            print(f"DICG-style GAE - Mean: {expanded_advantages.mean().item():.4f}, Std: {expanded_advantages.std().item():.4f}")
    
    def get_batches(self, batch_size):
        """Create mini-batches from centralized buffer data"""
        if not self.data['obs']:
            print("Warning: No samples in buffer!")
            return
            
        # Get all samples - centralized approach
        num_samples = len(self.data['obs'])
        all_samples = []
        
        for i in range(num_samples):
            sample = tuple(self.data[key][i] for key in [
                'obs', 'cent_obs', 'actions', 'values', 'returns',
                'log_probs', 'advantages', 'task_features', 
                'agent_features', 'valid_indices', 'policy_hidden'
            ])
            all_samples.append(sample)
        
        # Shuffle and create batches
        indices = torch.randperm(len(all_samples))
        for start in range(0, len(all_samples), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = [all_samples[i] for i in batch_indices]
            yield list(zip(*batch))


class ActorWrapper:
    """Wrapper for actor policy with action sampling"""
    
    def __init__(self, actor, device):
        self.actor = actor
        self.device = device
        self.hidden_states = {}
        self.current_episode = 0  #
    def init_hidden(self, agents, num_envs):
        """Initialize hidden states for all agents"""
        for agent in agents:
            if hasattr(self.actor, 'memory_lstm') or 'LSTM' in str(type(self.actor)):
                h = torch.zeros(num_envs, config.ACTOR_HIDDEN_DIM, device=self.device)
                c = torch.zeros(num_envs, config.ACTOR_HIDDEN_DIM, device=self.device)
                self.hidden_states[agent] = (h, c)
            else:
                self.hidden_states[agent] = torch.zeros(num_envs, config.ACTOR_HIDDEN_DIM, device=self.device)
    def gumbel_softmax_sample(self, logits, temperature=1.0, hard=False):
        """Gumbel softmax sampling"""
        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        
        # Apply temperature and softmax
        y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
        
        if hard:
            # Straight-through estimator for discrete sampling
            index = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            return y_hard - y_soft.detach() + y_soft, index.squeeze(-1)
        
        return y_soft, None
    
    
    def get_action(self, task_features, agent_features, valid_indices, agent, env_id):
        """Get action for single environment"""
        is_recurrent = isinstance(self.actor, Pointer_LSTM_Actor)
        # Get hidden state - LSTM expects 2D tensors [1, hidden_dim]
        if isinstance(self.hidden_states[agent], tuple):
            h, c = self.hidden_states[agent]
            hidden = (h[env_id:env_id+1], c[env_id:env_id+1])  # [1, hidden_dim]
        else:
            hidden = self.hidden_states[agent][env_id:env_id+1]  # [1, hidden_dim]
        
        with torch.no_grad():
            if task_features.shape[0] == 0:
                debug_logger.log(f"Agent {agent} Env {env_id}: Only NOOP available (no tasks)", "ACTION")
                # Only NOOP available
                action = torch.tensor(0, device=self.device)
                log_prob = torch.tensor(0.0, device=self.device)
                logits = torch.tensor([0.0], device=self.device) 
                next_hidden = hidden
            else:
                # Forward pass - DEBUG LOGGING
                debug_logger.log(f"Agent {agent} Env {env_id}: Forward pass with {task_features.shape[0]} tasks", "ACTOR")
                debug_logger.log_actor_forward(task_features, agent_features, torch.zeros(1))  # placeholder
                
                logits, next_hidden = self.actor(task_features, agent_features, hidden)
                
                # Log actual logits after forward pass
                debug_logger.log(f"  Actual logits: {logits.tolist()}", "ACTOR")
                # IMPORTANT:
                # task_features already includes NOOP as the last row (see _extract_action_features).
                # Therefore logits MUST have exactly the same length as task_features.
                expected_size = task_features.shape[0]
                debug_logger.log(f"  Logits shape: {logits.shape}, expected: ({expected_size},)", "ACTOR")

                # CRITICAL CHECK: Verify logits size matches the provided action feature list
                if logits.shape[0] != expected_size:
                    debug_logger.log(f"  CRITICAL ERROR: logits size {logits.shape[0]} != expected {expected_size}", "ACTOR")
                
                # Log probability distribution
                probs = torch.softmax(logits, dim=-1)
                debug_logger.log(f"  Softmax probs: {probs.tolist()}", "ACTOR")
                
                # Apply action mask.
                # If valid_indices is provided, only allow those indices.
                # Otherwise allow all logits.
                if valid_indices is not None and len(valid_indices) > 0:
                    mask = torch.full_like(logits, -1e9)
                    valid_tensor = torch.tensor(valid_indices, device=logits.device, dtype=torch.long)
                    # Guard against any out-of-range indices
                    valid_tensor = valid_tensor[(valid_tensor >= 0) & (valid_tensor < logits.shape[0])]
                    mask[valid_tensor] = 0.0
                else:
                    mask = torch.zeros_like(logits)
                
                if config.use_gumbel:
                    masked_logits=logits + mask

                                # Gumbel softmax with annealed temperature
                    temperature = max(0.1, 2.0 * (0.995 ** self.current_episode))  # Anneal from 2.0 to 0.1
                    
                    if self.actor.training:
                        # Use soft sampling during training
                        probs, _ = self.gumbel_softmax_sample(masked_logits, temperature, hard=False)
                        dist = Categorical(probs=probs)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                    else:
                        # Use hard sampling during evaluation
                        _, action = self.gumbel_softmax_sample(masked_logits, temperature=0.1, hard=True)
                        # Still compute log_prob for consistency
                        dist = Categorical(logits=masked_logits)
                        log_prob = dist.log_prob(action)                    
                    # Compute log prob for PPO
                    dist = Categorical(logits=masked_logits)
                    log_prob = dist.log_prob(action)
                else:
                    temperature = getattr(config, 'logit_temperature', 2.0)
                    logits = logits / temperature
                    dist = Categorical(logits=logits + mask)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                # DEBUG: Log action selection
                debug_logger.log(f"  Sampled action: {action.item()}, log_prob: {log_prob.item():.4f}", "ACTION")
                debug_logger.log(f"  Action distribution entropy: {dist.entropy().item():.4f}", "ACTION")
            
        if is_recurrent:    
            # Update hidden state - store back as [hidden_dim] per environment
            if isinstance(next_hidden, tuple):
                h_next, c_next = next_hidden
                self.hidden_states[agent] = (
                    self.hidden_states[agent][0].clone(),
                    self.hidden_states[agent][1].clone()
                )
                self.hidden_states[agent][0][env_id] = h_next.squeeze(0)  # [hidden_dim]
                self.hidden_states[agent][1][env_id] = c_next.squeeze(0)  # [hidden_dim]
            else:
                self.hidden_states[agent] = self.hidden_states[agent].clone()
                self.hidden_states[agent][env_id] = next_hidden.squeeze(0)  # [hidden_dim]
    
            
        return action, log_prob,logits
    
    def evaluate_actions(self, task_features_batch, agent_features_batch, 
                        actions_batch, valid_indices_batch, policy_hiddens_batch):
        """Evaluate actions for training using stored policy hidden states"""
        log_probs = []
        entropies = []
        
        for task_feat, agent_feat, action, valid_idx, pol_hidden in zip(
            task_features_batch, agent_features_batch, actions_batch, 
            valid_indices_batch, policy_hiddens_batch
        ):
            # Use stored hidden, ensure it's 2D [1, hidden_dim] for LSTM
            if isinstance(pol_hidden, tuple):
                h, c = pol_hidden
                if h.dim() == 1:
                    h = h.unsqueeze(0)
                if c.dim() == 1:
                    c = c.unsqueeze(0)
                hidden = (h, c)
            else:
                if pol_hidden.dim() == 1:
                    hidden = pol_hidden.unsqueeze(0)
                else:
                    hidden = pol_hidden
            
            if task_feat.shape[0] == 0:
                dummy_param = next(self.actor.parameters())
                log_prob = dummy_param.sum() * 0.0
                entropy = dummy_param.sum() * 0.0
            else:
                logits, _ = self.actor(task_feat, agent_feat, hidden)
                temperature = getattr(config, 'logit_temperature', 2.0)
                logits = logits / temperature
                mask = torch.full_like(logits, -float('1e9'))
                if len(valid_idx) > 0:
                    mask[valid_idx] = 0.0
                mask[-1] = 0.0
                
                dist = Categorical(logits=logits + mask)
                log_prob = dist.log_prob(action.to(self.device))
                entropy = dist.entropy()
            
            log_probs.append(log_prob)
            entropies.append(entropy)
        
        return torch.stack(log_probs), torch.stack(entropies).mean()

class MAPPOTrainer:
    """Clean MAPPO trainer implementation"""
    
    def __init__(self, actor, critic,num_agents, device):
        self.actor_wrapper = ActorWrapper(actor, device)
        self.critic = critic
        self.device = device
        self.num_agents = num_agents 
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.LR_ACTOR,eps=general_config.Optimizer_Epsilon)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.LR_CRITIC,eps=general_config.Optimizer_Epsilon)
        
        # PPO parameters
        self.clip_param = config.CLIP_PARAM
        self.value_clip_param = config.VALUE_CLIP_PARAM  # For value clipping
        self.entropy_coef = config.ENTROPY_COEF
        self.max_grad_norm = config.MAX_GRAD_NORM
        self.update_count = 0  # Track update count for logging
        
        # Running statistics for return normalization
        self.return_mean = 0.0
        self.return_std = 1.0
        self.return_count = 0
    
    def update(self, batch):
        """Single PPO update step with DICG-style stability"""
        self.update_count += 1
        
        (obs_batch, cent_obs_batch, actions_batch, values_batch, returns_batch,
         old_log_probs_batch, advantages_batch, task_features_batch,
         agent_features_batch, valid_indices_batch, policy_hiddens_batch) = batch
        
        # Convert to tensors with explicit float32 dtype (like DICG)
        returns = torch.tensor(returns_batch, device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs_batch, device=self.device, dtype=torch.float32)
        advantages = torch.tensor(advantages_batch, device=self.device, dtype=torch.float32)
        advantages = advantages.clamp(-5.0, 5.0)

        old_values = torch.tensor(values_batch, device=self.device, dtype=torch.float32)
        
        # DEBUG: Log batch statistics
        debug_logger.log(f"Update #{self.update_count} - Batch size: {len(returns)}", "TRAIN")
        debug_logger.log(f"  Returns: min={returns.min():.4f}, max={returns.max():.4f}, mean={returns.mean():.4f}", "TRAIN")
        debug_logger.log(f"  Advantages: min={advantages.min():.4f}, max={advantages.max():.4f}, mean={advantages.mean():.4f}, std={advantages.std():.4f}", "TRAIN")
        debug_logger.log(f"  Old log_probs: min={old_log_probs.min():.4f}, max={old_log_probs.max():.4f}", "TRAIN")
        
        if general_config.DEBUG:
            if advantages.abs().sum() < 1e-6:
                print("advantages are zero for this batch")

        # Get new action log probs and entropy using stored policy hidden states
        new_log_probs, entropy = self.actor_wrapper.evaluate_actions(
            task_features_batch, agent_features_batch, actions_batch, 
            valid_indices_batch, policy_hiddens_batch
        )
        
        # Get new values
        new_values = self._get_values(cent_obs_batch)
        
        # Ensure tensor sizes match (handle potential mismatches)
        min_size = min(len(new_log_probs), len(old_log_probs), len(advantages), 
                      len(new_values), len(returns))
        
        new_log_probs = new_log_probs[:min_size]
        old_log_probs = old_log_probs[:min_size] 
        advantages = advantages[:min_size]
        new_values = new_values[:min_size].float()  # Explicit float32
        returns = returns[:min_size]
        old_values = old_values[:min_size]
        
        # DEBUG: Log new vs old values
        debug_logger.log(f"  New log_probs: min={new_log_probs.min():.4f}, max={new_log_probs.max():.4f}", "TRAIN")
        debug_logger.log(f"  New values: min={new_values.min():.4f}, max={new_values.max():.4f}, mean={new_values.mean():.4f}", "TRAIN")
        
        # Actor loss (PPO clipped objective)
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # DEBUG: Log ratio statistics
        debug_logger.log(f"  Ratio: min={ratio.min():.4f}, max={ratio.max():.4f}, mean={ratio.mean():.4f}", "TRAIN")
        
        # Clamp ratio to prevent extreme values (DICG-style stability)
        ratio = torch.clamp(ratio, 0.1, 10.0)  
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
        
        # Normalize returns using running statistics (reduces variance)
        batch_mean = returns.mean().item()
        batch_std = returns.std().item() + 1e-8
        
        # Update running statistics with exponential moving average
        if self.return_count == 0:
            self.return_mean = batch_mean
            self.return_std = batch_std
        else:
            alpha = 0.01  # Slow update for stability
            self.return_mean = (1 - alpha) * self.return_mean + alpha * batch_mean
            self.return_std = (1 - alpha) * self.return_std + alpha * batch_std
        self.return_count += 1
        
        # Normalize returns for critic training
        normalized_returns = (returns - self.return_mean) / (self.return_std + 1e-8)
        normalized_new_values = (new_values - self.return_mean) / (self.return_std + 1e-8)
        normalized_old_values = (old_values - self.return_mean) / (self.return_std + 1e-8)
        
        # Clipped value loss (PPO-style) to prevent large value updates
        value_pred_clipped = normalized_old_values + torch.clamp(
            normalized_new_values - normalized_old_values,
            -self.value_clip_param,
            self.value_clip_param
        )
        
        # Use Huber loss instead of MSE for robustness to outliers
        value_loss_unclipped = F.huber_loss(normalized_new_values, normalized_returns, reduction='none', delta=10.0)
        value_loss_clipped = F.huber_loss(value_pred_clipped, normalized_returns, reduction='none', delta=10.0)
        
        # Take the max to be conservative (like PPO actor loss)
        critic_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
        
        value_mse = F.mse_loss(new_values, returns)
        explained_variance = 1 - torch.var(returns - new_values) / (torch.var(returns) + 1e-8)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # DEBUG: Log gradient norms before clipping
        actor_grad_norm = sum(p.grad.norm().item() for p in self.actor_wrapper.actor.parameters() if p.grad is not None)
        debug_logger.log(f"  Actor grad norm (before clip): {actor_grad_norm:.4f}", "TRAIN")
        
        nn.utils.clip_grad_norm_(self.actor_wrapper.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Update critic with stricter gradient clipping
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # DEBUG: Log gradient norms before clipping
        critic_grad_norm = sum(p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None)
        debug_logger.log(f"  Critic grad norm (before clip): {critic_grad_norm:.4f}", "TRAIN")
        
        # Use stricter gradient clipping for critic (half of actor's threshold)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm * 0.5)
        self.critic_optimizer.step()
        
        log_ratio = new_log_probs - old_log_probs
        approx_kl = (log_ratio.exp() - 1 - log_ratio).mean()
        
        # DEBUG: Log training update summary
        debug_logger.log_training_update(self.update_count, actor_loss.item(), critic_loss.item(), entropy.item(), approx_kl.item() if torch.is_tensor(approx_kl) else approx_kl)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            # Keep these as floats; casting to int makes them look like 0 and breaks KL early-stop.
            'approx_kl': float(approx_kl.item()) if torch.is_tensor(approx_kl) else float(approx_kl),
            'value_mse': float(value_mse.item()) if torch.is_tensor(value_mse) else float(value_mse),
            'explained_variance': float(explained_variance.item()) if torch.is_tensor(explained_variance) else float(explained_variance)
        }
    
    def _get_values(self, cent_obs_batch):
        """Get critic values using GNNValueCritic with enriched node features (no edge weights, no global context)"""
        values = []
        for obs_item in cent_obs_batch:
            # Unpack - supports both old (3-tuple) and new (5-tuple) formats
            if len(obs_item) == 3:
                agent_nodes, task_nodes, edge_index = obs_item
            else:
                # For backward compatibility, ignore edge_weight and global_features
                agent_nodes, task_nodes, edge_index = obs_item[0], obs_item[1], obs_item[2]
            
            value = self.critic(
                agent_nodes.to(self.device),
                task_nodes.to(self.device) if task_nodes is not None else None,
                edge_index.to(self.device) if edge_index is not None else None
            )
            values.append(value.squeeze())
        return torch.stack(values) if values else torch.empty(0, device=self.device)


class MAPPORunner:
    """Main training runner"""
    
    def __init__(self, args):
        self.args = args
        self.device = DEVICE
        self.best_eval_reward = float("-inf")

        # Setup environment and models
        self.env = self._init_env()
        self.actor, self.critic, self.agents = self._init_models(args)

        # Derive grid size from environment state (set after _init_models calls env.reset)
        _fires = self.env.state().fires[0].squeeze(0)
        self.grid_size = (_fires.shape[0], _fires.shape[1])

        # Setup trainer and buffer
        self.trainer = MAPPOTrainer(self.actor, self.critic,len(self.agents), self.device)
        self.buffer = RolloutBuffer(self.agents)
        
        # Setup logging
        self.writer, self.checkpoint_dir = setup_tensorboard_logging(args)
        if config.log_network:
            log_dir=self.checkpoint_dir.replace('checkpoints','network_logs')
            log_model_architecture(self.actor, self.critic,log_dir)
            log_network_graph(self.actor, self.critic,log_dir)  
            log_training_flow_graph(log_dir)
        
        # DEBUG: Log network parameters at initialization
        debug_logger.log_section("INITIALIZATION")
        debug_logger.log_network_params(self.actor, self.critic)
        debug_logger.log(f"Actor type: {type(self.actor).__name__}", "INIT")
        debug_logger.log(f"Critic type: {type(self.critic).__name__}", "INIT")
        debug_logger.log(f"Number of agents: {len(self.agents)}", "INIT")
        debug_logger.log(f"Config: LR_ACTOR={config.LR_ACTOR}, LR_CRITIC={config.LR_CRITIC}", "INIT")
        debug_logger.log(f"Config: CLIP_PARAM={config.CLIP_PARAM}, ENTROPY_COEF={config.ENTROPY_COEF}", "INIT")
        debug_logger.log(f"Config: ACTOR_HIDDEN_DIM={config.ACTOR_HIDDEN_DIM}, CRITIC_HIDDEN_DIM={config.CRITIC_HIDDEN_DIM}", "INIT")
            
        # Setup DB logger
        self.db = ExperimentDB()    
        experiment_meta = dict(
            run_type="training",
            model_family="PLATO",
            model_type="smart",
            name=args.experiment_name,
            variation1=args.pointer_policy,
            variation2=args.attention_type,
            grid_size=self.args.grid_size,
            setting_type=self.args.setting_type,
            # seed=self.args.seed,
            parent_experiment_id=None,
            checkpoint_path=self.checkpoint_dir,
            notes=None,
        )   
    
        self.run_logger = RunLogger(
            writer=self.writer,
            db=self.db,
            experiment_meta=experiment_meta,
            log_every=general_config.LOG_EVERY,
        )
    def _init_env(self):
        """Initialize environment"""
        configuration = resolve_configuration(self.args)
        env = wildfire_v0.parallel_env(
            configuration=configuration,
            parallel_envs=config.NUM_PARALLEL_ENVS,
            max_steps=general_config.MAX_STEPS,
            device=self.device,     
            single_seeding=False,
            show_bad_actions=False,
            observe_other_power=True,
            observe_other_suppressant=True,
            override_initialization_check=True,
            log_directory="outputs/",

        )
        return action_mapping_wrapper_v0(env)
    
    def _init_models(self,args):
        """Initialize actor and critic models"""
        observations, _ = self.env.reset()
        agents = self.env.agents
        # Derive actor ablation flags from the pointer_policy name
        pol = args.pointer_policy
        no_team_stats = 'noteamstats' in pol
        no_pointer    = 'nopointer'   in pol
        # Critic ablation is still a separate flag (not actor-side)
        no_gnn_critic = getattr(args, 'ablation_no_gnn_critic', False)
        if 'lstm' in pol:
            actor = Pointer_LSTM_Actor(
                    task_feat_dim=7,  # [|dx|, |dy|, dist, in_range, intensity, fuel, agents_in_range]
                    agent_feat_dim=general_config.FEAT_DIM,
                    hidden_dim=config.ACTOR_HIDDEN_DIM,
                    attention_type=self.args.attention_type,                 
                    dropout=config.ACTOR_DROPOUT,
                    ablation_no_team_stats=no_team_stats,
                    ablation_no_pointer=no_pointer,
                ).to(self.device)
        elif 'mlp' in pol or pol == '':
            actor = Pointer_MLP_Actor(
                task_feat_dim=7,  # [|dx|, |dy|, dist, in_range, intensity, fuel, agents_in_range]
                agent_feat_dim=general_config.FEAT_DIM,
                hidden_dim=config.ACTOR_HIDDEN_DIM,
                attention_type=self.args.attention_type,
                ablation_no_team_stats=no_team_stats,
                ablation_no_pointer=no_pointer,
            ).to(self.device)
        if no_gnn_critic:
            critic = MLPValueCritic(agent_feat_dim=11, task_feat_dim=9, hidden_dim=64).to(self.device)
        else:
            critic = GNNValueCritic(agent_feat_dim=11, task_feat_dim=9, hidden_dim=64).to(self.device)

        # self._orthogonal_init(actor)
        # self._orthogonal_init(critic)
        return actor, critic, agents
    
    def _orthogonal_init(self, model):
        """Apply orthogonal initialization to model parameters"""
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Orthogonal initialization for linear and convolutional layers
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                # Orthogonal initialization for RNN layers
                for name, param in module.named_parameters():
                    if 'weight_ih' in name or 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
            elif isinstance(module, nn.Embedding):
                # Orthogonal initialization for embedding layers
                nn.init.orthogonal_(module.weight)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                # Standard initialization for normalization layers
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    
    def evaluate(self):
        """Evaluate current policy"""
        total_rewards = []
        total_burnouts = []
        total_putouts = []
        total_lengths = []
        eval_seeds = []
        
        # Set models to eval mode
        self.actor.eval()
        
        with torch.no_grad():
            for eval_episode in range(general_config.VALIDATION_EPISODES):
                seed = get_validation_seed(eval_episode, general_config)
                observations, _ = self.env.reset(seed=seed)
                eval_seeds.append(seed)
                self.trainer.actor_wrapper.init_hidden(self.agents, config.NUM_PARALLEL_ENVS)
                
                episode_reward = 0
                episode_putouts = 0
                episode_burnouts = 0
                episode_supressant_wait_time=0
                steps = 0
                
                while (not torch.all(self.env.finished)):
                    obs = {agent: observations[agent][0] for agent in self.agents}
                    
                    # Get observation features
                    obs_features = observations_to_features(
                        config.NUM_PARALLEL_ENVS, obs, 3, self.env, 
                        config.ONLY_AVAILABLE_ACTIONS, self.device, shuffle_tasks=config.Shuffle_Tasks
                    )
                    
                    # Extract action features with 7-dim task features (same as training)
                    action_features = self._extract_action_features(obs, self.agents)
                    
                    # Collect actions (no buffer storage, just execution)
                    actions_dict = {}
                    for agent in self.agents:
                        agent_actions = []
                        for env_id in range(config.NUM_PARALLEL_ENVS):
                            agent_feat = obs_features[env_id][agent]
                            action_data = action_features[(agent, env_id)]
                            
                            # Normalize agent features: [y_norm, x_norm, power, suppressant]
                            self_features = agent_feat['self']
                            others_features = agent_feat.get('others', torch.empty(0, device=self.device))
                            if self_features.numel() == 0:
                                agent_features = torch.zeros(1, general_config.FEAT_DIM, device=self.device)
                            else:
                                agent_features = self._normalize_agent_features(self_features, others_features)
                            
                            action, log_prob, logits = self.trainer.actor_wrapper.get_action(
                                action_data['action_features'], agent_features, 
                                list(range(len(action_data['action_mapping']))), agent, env_id
                            )
                            
                            # Use action mapping from action_features (same as training)
                            env_action = action_data['action_mapping'][action.item()]
                            
                            agent_actions.append(torch.tensor(env_action, device=self.device))
                        
                        actions_dict[agent] = torch.stack(agent_actions)
                    
                    # Environment step
                    observations, rewards, terms, truncs, infoes = self.env.step(actions_dict)
                    
                    # Process rewards and collect metrics
                    for env_id in range(config.NUM_PARALLEL_ENVS):
                        reward = np.mean([r[env_id].mean().item() for r in rewards.values()])
                        burnouts = np.mean([r.item() for r in infoes['burnouts']])
                        putouts = np.mean([r.item() for r in infoes['putouts']])
                        
                        episode_reward += reward
                        episode_burnouts += burnouts
                        episode_putouts += putouts
                    
                    steps += 1
                
                # Store episode metrics
                total_rewards.append(episode_reward / config.NUM_PARALLEL_ENVS)
                total_burnouts.append(episode_burnouts / config.NUM_PARALLEL_ENVS)
                total_putouts.append(episode_putouts / config.NUM_PARALLEL_ENVS)
                total_lengths.append(steps)
        
        # Set models back to train mode
        self.actor.train()
        
        # Calculate averages
        avg_reward = np.mean(total_rewards)
        avg_burnouts = np.mean(total_burnouts)
        avg_putouts = np.mean(total_putouts)
        avg_length = np.mean(total_lengths)
        
        return avg_reward, avg_burnouts, avg_putouts, avg_length, str(eval_seeds)

    def _normalize_agent_features(self, self_features, others_features):
        """Delegate to shared utility (see utils/observation_graph_helper.py)."""
        return normalize_wildfire_agent_features(self_features, others_features, self.env)

    def _extract_action_features(self, obs, agents, shuffle=False):
        """Extract egocentric, direction-agnostic task features for all agents.
        Delegates to the shared extract_wildfire_action_features() utility.
        Task features: [|dx|, |dy|, dist, in_range, intensity, fuel, agents_in_range] = 7 dims
        """
        return extract_wildfire_action_features(
            self.env, agents, config.NUM_PARALLEL_ENVS, self.device,
            shuffle_tasks=shuffle
        )
    def collect_rollout(self,episode):
        """Collect a single rollout"""
        # DEBUG: Log episode start
        debug_logger.log_episode_start(episode)
        debug_logger.reset_episode_actions()  # Reset action tracking for new episode

        self.trainer.actor_wrapper.current_episode = episode
        seed = get_training_seed(episode, general_config)
        observations, _ = self.env.reset(seed=seed)
        env_seeds = str(seed)
        
        initial_state=deepcopy(self.env.state())
        
        # print(self.env.state())
        self.trainer.actor_wrapper.init_hidden(self.agents, config.NUM_PARALLEL_ENVS)
        
        episode_reward = 0
        episode_putouts = 0
        episode_burnouts = 0
        episode_noop_count = 0
        episode_fight_count = 0
        episode_supressant_wait_time = 0

        steps = 0
        
        while (not torch.all(self.env.finished)):
            # DEBUG: Log step
            debug_logger.log_step(steps)
            
            obs = {agent: observations[agent][0] for agent in self.agents}
            
            # Get graph data for critic (with NOOP node and enhanced features)
            graph_data = create_global_bipartite_graph_wildfire(
                config.NUM_PARALLEL_ENVS, self.env.state(), self.device,
                grid_size=self.grid_size
            )

            # Get observation features
            obs_features = observations_to_features(
                config.NUM_PARALLEL_ENVS, obs, 3, self.env, 
                config.ONLY_AVAILABLE_ACTIONS, self.device,shuffle_tasks= config.Shuffle_Tasks
            )
            
            action_features = self._extract_action_features(obs, self.agents,
                                                              shuffle=config.Shuffle_Tasks)
            # Collect actions
            actions_dict = {}
            for agent in self.agents:
                agent_actions = []
                for env_id in range(config.NUM_PARALLEL_ENVS):
                    agent_feat = obs_features[env_id][agent]
                    
                    # Normalize agent features: [power, suppressant, is_ready, rel_dist_norm]
                    self_features = agent_feat['self']
                    others_features = agent_feat.get('others', torch.empty(0, device=self.device))
                    if self_features.numel() == 0:
                        agent_features = torch.zeros(1, general_config.FEAT_DIM, device=self.device)
                    else:
                        agent_features = self._normalize_agent_features(self_features, others_features)
                    
                    # Get hidden state for evaluation storage
                    if isinstance(self.trainer.actor_wrapper.hidden_states[agent], tuple):
                        h, c = self.trainer.actor_wrapper.hidden_states[agent]
                        # Store as 2D tensors [1, hidden_dim] for LSTM compatibility
                        h_for_eval = (h[env_id:env_id+1].detach().clone(), 
                                     c[env_id:env_id+1].detach().clone())
                    else:
                        # Store as 2D tensor [1, hidden_dim] for GRU compatibility  
                        h_for_eval = self.trainer.actor_wrapper.hidden_states[agent][env_id:env_id+1].detach().clone()
                    
                    action_data = action_features[(agent, env_id)]
                    
                    # DEBUG: Log action features BEFORE actor forward
                    debug_logger.log_action_features(
                        agent, env_id, 
                        action_data['action_features'], 
                        action_data['action_mapping']
                    )
                    debug_logger.log(f"  Agent features shape: {agent_features.shape}", "ACTION_FEATURES")
                    debug_logger.log(f"  Agent features:\n{agent_features}", "ACTION_FEATURES")
                    
                    action, log_prob, logits = self.trainer.actor_wrapper.get_action(
                        action_data['action_features'], agent_features, 
                        list(range(len(action_data['action_mapping']))), agent, env_id
                    )
                    
                    # DEBUG: Log action mapping AFTER sampling
                    debug_logger.log(f"Action Mapping Debug:", "MAPPING")
                    debug_logger.log(f"  action_features shape: {action_data['action_features'].shape}", "MAPPING")
                    debug_logger.log(f"  logits shape: {logits.shape}", "MAPPING")
                    debug_logger.log(f"  action_mapping: {action_data['action_mapping']}", "MAPPING")
                    debug_logger.log(f"  sampled action index: {action.item()}", "MAPPING")
                    
                    # CRITICAL CHECK: Verify action index is valid
                    if action.item() >= len(action_data['action_mapping']):
                        debug_logger.log(f"  ERROR: action index {action.item()} >= mapping length {len(action_data['action_mapping'])}", "MAPPING")
                        debug_logger.log(f"  This means network output {logits.shape[0]} logits but mapping has {len(action_data['action_mapping'])} entries", "MAPPING")
                    
                    selected_action = action_data['action_mapping'][action.item()]
                    env_action = selected_action
                    
                    debug_logger.log_action_selection(action, log_prob, action_data['action_mapping'], selected_action)
                    
                    # TRACK ACTION PATTERN
                    debug_logger.track_action(agent, action.item(), len(action_data['action_mapping']))
                    
                    agent_actions.append(torch.tensor(env_action, device=self.device))
                    
                    # Get critic value (using task_nodes which includes NOOP)
                    with torch.no_grad():
                        value = self.trainer._get_values([
                            (graph_data[env_id]['agent_nodes'],
                             graph_data[env_id]['task_nodes'],
                             graph_data[env_id]['edge_index']
                    )
                        ])
                        value_scalar = value[0].item() if len(value) > 0 else 0.0
                    
                    # DEBUG: Log critic value
                    debug_logger.log_critic_forward(
                        graph_data[env_id]['agent_nodes'],
                        graph_data[env_id]['task_nodes'],
                        graph_data[env_id]['edge_index'],
                        value_scalar
                    )
                    
                    # Store in buffer (using task_nodes which includes NOOP)
                    self.buffer.add_step(agent,
                        obs=obs[agent],
                        cent_obs=(graph_data[env_id]['agent_nodes'], 
                                graph_data[env_id]['task_nodes'],
                                graph_data[env_id]['edge_index']
                ),
                        actions=action,
                        log_probs=log_prob.item(),
                        values=value_scalar,
                        task_features=action_data['action_features'],
                        valid_indices=list(range(len(action_data['action_mapping']))),          
                        agent_features=agent_features,
                        policy_hidden=h_for_eval  # Store hidden state used for action
                    )
                    
                    # DEBUG: Log buffer addition
                    debug_logger.log_buffer_add(agent, env_id, action.item(), log_prob.item(), value_scalar, None)
                
                actions_dict[agent] = torch.stack(agent_actions)
            # print_wildfire_grid(self.env.state())
            # Environment step
            observations, rewards, terms, truncs, infoes = self.env.step(actions_dict)
            noop_counts,fight_counts=action_info(actions_dict)
            
            reward = np.mean([r.mean().item() for r in rewards.values()])
            burnouts=np.mean([r.item() for r in infoes['burnouts']])
            putouts=np.mean([r.item() for r in infoes['putouts']])
            step_supressant_wait_time=count_zero_suppressant_agents(self.env.state())
            
            # DEBUG: Log reward
            debug_logger.log(f"Step Reward: {reward:.4f}, Burnouts: {burnouts:.1f}, Putouts: {putouts:.1f}", "REWARD")
            
            # Process rewards and masks, per agent per env
            for agent in self.agents:
                for env_id in range(config.NUM_PARALLEL_ENVS):
                    done = any(t[env_id].item() for t in terms.values())
                    env_reward = np.mean([r[env_id].item() for r in rewards.values()])
                    self.buffer.add_step(
                        agent,
                        rewards=env_reward,
                        masks=0.0 if done else 1.0
                    )

            episode_reward += reward
            episode_burnouts += burnouts
            episode_putouts += putouts
            episode_noop_count += noop_counts
            episode_fight_count += fight_counts
            episode_supressant_wait_time+=step_supressant_wait_time
            
            steps += 1
        
        
        
        last_state=self.env.state()
        episode_total_fires_count,new_fire_ignited_count= get_episode_fire_counts(initial_state, last_state,episode_burnouts,episode_putouts,config.NUM_PARALLEL_ENVS)
        
        # DEBUG: Log episode end
        debug_logger.log(f"Episode finished after {steps} steps, total reward: {episode_reward:.4f}", "EPISODE")
        
        return episode_reward , steps,episode_burnouts,episode_putouts,episode_total_fires_count,new_fire_ignited_count,episode_noop_count,episode_fight_count,env_seeds,episode_supressant_wait_time
    
    def train(self):
        """Main training loop"""
        for episode in range(1, general_config.EPISODES + 1):
                    # start timer
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Collect rollout
            episode_reward, episode_steps,episode_burnouts,episode_putouts,episode_total_fires_count,new_fire_ignited_count,episode_noop_count,episode_fight_count,env_seeds,episode_supressant_wait_time = self.collect_rollout(episode)
            
            # Compute v_last from final state for proper GAE bootstrapping (DICG-style)
            final_graph_data = create_global_bipartite_graph_wildfire(
                config.NUM_PARALLEL_ENVS, self.env.state(), self.device,
                grid_size=self.grid_size
            )
            with torch.no_grad():
                v_last_list = []
                for e in range(config.NUM_PARALLEL_ENVS):
                    # Get one value per environment (not per agent like DICG)
                    v = self.trainer._get_values([(final_graph_data[e]['agent_nodes'],
                                                final_graph_data[e]['task_nodes'],
                                                final_graph_data[e]['edge_index']
                                      )])
                    v_last_list.append(v[0].item() if len(v) > 0 else 0.0)
                self.buffer.compute_gae(v_last=v_last_list)

            
            # Training updates
            train_metrics = {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0, 'approx_kl': 0,'value_mse':0,'explained_variance':0}
            num_updates = 0
            
            debug_logger.log_section(f"TRAINING EPISODE {episode}")
            
            for epoch in range(config.PPO_EPOCH):
                for batch in self.buffer.get_batches(config.BATCH_SIZE // config.NUM_MINI_BATCH):
                    metrics = self.trainer.update(batch)
                    for key in train_metrics:
                        train_metrics[key] += metrics[key]
                    num_updates += 1
                    
                    # Early stopping on high KL divergence
                    if metrics['approx_kl'] > config.TARGET_KL:  # Conservative threshold
                        break
                if metrics['approx_kl'] > config.TARGET_KL:
                    break
            if config.log_network:    
                grad_summary = log_gradient_flow(self.actor, self.critic, episode, self.writer)
            else:
                grad_summary = {
                    'actor_vanishing': 0,
                    'actor_exploding': 0,
                    'critic_vanishing':0,
                    'critic_exploding':0,
                    'actor_max_grad':  0,
                    'critic_max_grad': 0,
                }
    
            # Average metrics
            for key in train_metrics:
                train_metrics[key] /= max(num_updates, 1)
            
            # DEBUG: Log episode summary
            debug_logger.log_episode_summary(
                episode, episode_reward, episode_steps, 
                train_metrics['actor_loss'], train_metrics['critic_loss']
            )
            
            # DEBUG: Log action pattern for this episode
            debug_logger.log_action_pattern_summary(episode)

            total_actor_grad = sum(p.grad.norm().item() for p in self.actor.parameters() if p.grad is not None)
            total_critic_grad = sum(p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None)


            elapsed = time.perf_counter() - t0
            fps = episode_steps / max(elapsed, 1e-9)

            # Log learning progress indicators
            if episode % general_config.LOG_NETWORK_EVERY == 0:
                if config.log_network:
                    log_gradient_flow_graph(self.actor, self.critic, episode)
                    
            if episode % general_config.LOG_EVERY == 0:                        
                    
                episode_stats = dict(
                    reward=episode_reward,
                    burnouts=episode_burnouts,
                    putouts=episode_putouts,
                    episode_total_fires_count=episode_total_fires_count,
                    new_fire_ignited_count=new_fire_ignited_count,
                    episode_noop_count=episode_noop_count,
                    episode_fight_count=episode_fight_count,
                    episode_length=episode_steps,
                    env_seeds=env_seeds,
                    episode_supressant_wait_time=episode_supressant_wait_time
                )

                self.run_logger.log_episode(self.args.experiment_name,
                    episode=episode,
                    episode_stats=episode_stats,
                    train_metrics=train_metrics,
                    grad_summary=grad_summary,
                    total_actor_grad=total_actor_grad,
                    total_critic_grad=total_critic_grad,
                    elapsed=elapsed,
                    fps=fps,
                    
                )

            # Run evaluation every VALIDATION_EVERY episodes
            if episode % general_config.VALIDATION_EVERY == 0:
                eval_reward, eval_burnouts, eval_putouts, eval_length, eval_seeds = self.evaluate()
                #THis could be inside the eval part.due to debugging its a separate eval func.

                self.run_logger.log_eval(
                    episode=episode,
                    eval_reward=eval_reward,
                    eval_burnouts=eval_burnouts,
                    eval_putouts=eval_putouts,
                    eval_length=eval_length,
                    eval_seeds=eval_seeds,
                    avg_noop= 0
                )
                if TUNING_MODE and eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                
                logging.info(f"Evaluation at episode {episode}: "
                            f"Reward: {eval_reward:.4f}, "
                            f"Burnouts: {eval_burnouts:.4f}, "
                            f"Putouts: {eval_putouts:.4f}, "
                            f"Length: {eval_length:.2f}")
            
    

            
            # Save checkpoint
            if episode % general_config.SAVE_EVERY == 0:
                save_checkpoint({
                    "actor": self.actor,
                    "critic": self.critic,
                    "episode": episode,
                    "reward": episode_reward
                }, self.checkpoint_dir, episode, self.args.experiment_name)
            
            # Clear buffer
            self.buffer.clear()
                    # stop timer
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            
        if TUNING_MODE:
            return self.best_eval_reward


def main():
    """Main function"""
    args = parse_args()
    args.experiment_name = 'PLATO_wildfire'
    if(general_config.use_global_seed):
        set_global_seed(general_config.Global_Seed,general_config.use_deterministic)
    
    # Add this in main() after loading configs:
    print("=== CONFIG Initialized ===")
    print(f"NUM_PARALLEL_ENVS: {config.NUM_PARALLEL_ENVS}")
    print(f"ONLY_AVAILABLE_ACTIONS: {config.ONLY_AVAILABLE_ACTIONS}")
    print(f"MAX_STEPS: {general_config.MAX_STEPS}")
    print(f"FEAT_DIM: {general_config.FEAT_DIM}")
    print(f"NumPy: {np.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Python: {sys.version}")


        

    
    runner = MAPPORunner(args)
    runner.train()


if __name__ == '__main__':
    main()