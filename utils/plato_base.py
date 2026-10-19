# PLATO Base Module - Shared components for wildfire and rideshare domains
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


def orthogonal_init(model):
    """Apply orthogonal initialization to model parameters"""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for name, param in module.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.orthogonal_(module.weight)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
