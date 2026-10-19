# DICG for Rideshare Environment - Competitive Individual Rewards
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import tabulate
from colorama import Fore
import gc
import time
from tqdm import trange


from utils.experiment_manager import load_config
from free_range_zoo.envs import rideshare_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from utils.graph_utils_ptr import convert_actions_to_tensor, get_available_actions
from utils.graph_utils import print_rideshare_grid
from utils.helper import (debug_print, get_training_seed, get_validation_seed, 
                          resolve_configuration, parse_args, save_checkpoint, 
                          set_global_seed, setup_logging, setup_tensorboard_logging)
from utils.rideshare_capacity_filter import apply_capacity_filter, RideshareMetricTracker, RideshareEpisodeMetrics
from copy import deepcopy



config = load_config("DICG")
general_config = load_config("General_Config")

if general_config.DB == "postgres":
    from experiment_logger_postgres import ExperimentDB, RunLogger
else:
    from experiment_logger_sqlite import ExperimentDB, RunLogger
    

if general_config.DEVICE != (None or ''):
    DEVICE = general_config.DEVICE
else:    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def safe_orthogonal_init(module):
    """Safe orthogonal initialization"""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


class DICGEncoder(nn.Module):
    """Parameter-sharing encoder for DICG following Equation (3)"""
    def __init__(self, obs_dim, embedding_dim, hidden_sizes=(config.HIDDEN_DIM,)):
        super().__init__()
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.Tanh()])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, embedding_dim))
        self.encoder = nn.Sequential(*layers)
        safe_orthogonal_init(self)
    
    def forward(self, obs):
        """ei = Encoder(oi; θe), i=1,...,n"""
        batch_size, n_agents, obs_dim = obs.shape
        obs_flat = obs.view(-1, obs_dim)
        embeddings = self.encoder(obs_flat)
        return embeddings.view(batch_size, n_agents, -1)


class DICGAttention(nn.Module):
    """Attention mechanism following Equations (4) and (5)"""
    def __init__(self, embedding_dim):
        super().__init__()
        self.W_a = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.orthogonal_(self.W_a.weight)
    
    def forward(self, embeddings):
        """Compute μij with Attention(ei, ej, Wa) = ej^T Wa ei"""
        transformed = self.W_a(embeddings)
        attention_scores = torch.bmm(transformed, embeddings.transpose(1, 2))
        attention_weights = F.softmax(attention_scores, dim=-1)
        return attention_weights


class DICGGraphConv(nn.Module):
    """Graph Convolution following Equation (7)"""
    def __init__(self, embedding_dim, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        self.conv_layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=False) for _ in range(n_layers)
        ])
        safe_orthogonal_init(self)
    
    def forward(self, embeddings, attention_weights):
        """H^(l+1) = σ(M * H^(l) * Wc^(l))"""
        H = embeddings
        for layer in self.conv_layers:
            messages = torch.bmm(attention_weights, H)
            H = torch.tanh(layer(messages))
        return H


class DICGModule(nn.Module):
    """Complete DICG module"""
    def __init__(self, obs_dim, embedding_dim=config.EMBEDDING_DIM, encoder_hidden_sizes=(config.HIDDEN_DIM,), n_gcn_layers=2):
        super().__init__()
        self.encoder = DICGEncoder(obs_dim, embedding_dim, encoder_hidden_sizes)
        self.attention = DICGAttention(embedding_dim)
        self.graph_conv = DICGGraphConv(embedding_dim, n_gcn_layers)
    
    def forward(self, obs):
        """Complete DICG forward pass with residual connection"""
        E_0 = self.encoder(obs)
        attention_weights = self.attention(E_0)
        E_m = self.graph_conv(E_0, attention_weights)
        E_tilde = E_0 + E_m  # Residual connection
        return E_tilde, attention_weights


class DICGActor(nn.Module):
    """DICG Actor for rideshare environment"""
    def __init__(self, obs_dim, action_dim, embedding_dim=config.EMBEDDING_DIM, hidden_dim=config.HIDDEN_DIM):
        super().__init__()
        self.dicg = DICGModule(obs_dim, embedding_dim)
        self.action_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        safe_orthogonal_init(self)

    def forward(self, obs, action_mask=None):
        final_embeddings, attention_weights = self.dicg(obs)
        logits = self.action_head(final_embeddings)

        # Hard mask invalid actions exactly (like MAPPO pointer)
        if action_mask is not None:
            masked_logits = logits.masked_fill(action_mask == 0, float('-inf'))
        else:
            masked_logits = logits

        action_probs = F.softmax(masked_logits, dim=-1)
        return action_probs, masked_logits


class DICGCritic(nn.Module):
    """DICG Critic with per-agent value heads for competitive/individual rewards.
    
    Each agent gets its own value estimate based on its DICG embedding,
    enabling individual reward credit assignment in competitive settings.
    """
    def __init__(self, obs_dim, n_agents, embedding_dim=config.EMBEDDING_DIM, aggregator_type='sum'):
        super().__init__()
        self.dicg = DICGModule(obs_dim, embedding_dim)
        self.n_agents = n_agents
        
        # Per-agent value head: maps each agent's embedding to a scalar value
        self.agent_value_head = nn.Sequential(
            nn.Linear(embedding_dim, config.EMBEDDING_DIM),
            nn.Tanh(),
            nn.Linear(config.EMBEDDING_DIM, 1)
        )
        safe_orthogonal_init(self)
    
    def forward(self, obs):
        """Returns per-agent values [batch, n_agents]"""
        final_embeddings, _ = self.dicg(obs)  # [batch, n_agents, embedding_dim]
        # Apply shared value head to each agent's embedding
        per_agent_values = self.agent_value_head(final_embeddings).squeeze(-1)  # [batch, n_agents]
        return per_agent_values


def pad_observations(observations, max_obs_dim=config.OBS_DIM, env_id=0):
    """Build fixed-size obs vectors for rideshare."""
    TASK_FEAT_DIM = 8   # y, x, y_dest, x_dest, accepted_by, riding_by, fare, entered_step
    AGENT_FEAT_DIM = 4  # y, x, num_accepted, num_riding

    padded_obs = {}
    for agent, obs in observations.items():
        agent_obs_dict = obs[env_id]

        if 'tasks' in agent_obs_dict and agent_obs_dict['tasks'].numel() > 0:
            tasks = agent_obs_dict['tasks'][env_id].float()
            if tasks.ndim == 1:
                tasks = tasks.unsqueeze(0)
        else:
            tasks = torch.empty(0, TASK_FEAT_DIM, device=DEVICE)

        self_f = (agent_obs_dict['self'][env_id].float()
                  if 'self' in agent_obs_dict
                  else torch.empty(0, device=DEVICE))

        if 'others' in agent_obs_dict and agent_obs_dict['others'].numel() > 0:
            others = agent_obs_dict['others'][env_id].float()
            if others.ndim == 1:
                others = others.unsqueeze(0)
        else:
            others = torch.empty(0, AGENT_FEAT_DIM, device=DEVICE)

        task_fdim = tasks.shape[1] if tasks.ndim == 2 and tasks.numel() > 0 else TASK_FEAT_DIM
        other_fdim = others.shape[1] if others.ndim == 2 and others.numel() > 0 else AGENT_FEAT_DIM

        num_tasks = min(tasks.shape[0], general_config.MAX_TASKS) if tasks.numel() > 0 else 0
        num_others = min(others.shape[0], general_config.MAX_TASKS) if others.numel() > 0 else 0

        padded_tasks = torch.full((general_config.MAX_TASKS, task_fdim), -1.0, device=DEVICE)
        padded_others = torch.full((general_config.MAX_TASKS, other_fdim), -1.0, device=DEVICE)
        if num_tasks > 0: 
            padded_tasks[:num_tasks] = tasks[:num_tasks]
        if num_others > 0: 
            padded_others[:num_others] = others[:num_others]

        agent_vec = torch.cat([padded_tasks.flatten(), self_f.flatten(), padded_others.flatten()])
        if agent_vec.numel() < max_obs_dim:
            out = torch.zeros(max_obs_dim, device=DEVICE)
            out[:agent_vec.numel()] = agent_vec
        else:
            out = agent_vec[:max_obs_dim]
        padded_obs[agent] = out
    return padded_obs


def create_action_masks(env, agents, observations, infos, env_id=0):
    """State-dependent masks and aligned mapping per agent (single env)."""
    action_masks = {}
    action_mappings = {}

    for agent in agents:
        # Preferred: env/infos should expose a per-step action mask
        inferred = False
        if infos and agent in infos and "action_mask" in infos[agent]:
            m = infos[agent]["action_mask"][env_id].float().to(DEVICE)
            # Pad/crop to MAX_ACTIONS_PER_AGENT
            if m.numel() < general_config.MAX_ACTIONS_PER_AGENT:
                pad = torch.zeros(general_config.MAX_ACTIONS_PER_AGENT - m.numel(), device=DEVICE)
                m = torch.cat([m, pad], dim=0)
            else:
                m = m[:general_config.MAX_ACTIONS_PER_AGENT]
            mask = m
            # Build mapping list aligned with mask=1 positions if available
            amap = env.unwrapped.agent_action_mapping[agent]
            action_mappings[agent] = amap[:general_config.MAX_ACTIONS_PER_AGENT]
            action_masks[agent] = mask
            inferred = True

        if not inferred:
            # Fallback: derive availability now (must reflect current state)
            amap = env.unwrapped.agent_action_mapping[agent]
            avail = get_available_actions(amap)  # ensure this uses CURRENT state
            mask = torch.zeros(general_config.MAX_ACTIONS_PER_AGENT, device=DEVICE)
            if len(avail) > 0:
                count = min(len(avail), general_config.MAX_ACTIONS_PER_AGENT)
                mask[:count] = 1.0
                action_mappings[agent] = avail[:count]
            else:
                # Always allow a real NOOP slot
                mask[-1] = 1.0
                action_mappings[agent] = [(-1, -1)]  # adapt if your env uses a different NOOP token
            action_masks[agent] = mask

        # Ensure NOOP is allowed (MAPPO convention): force last index on if env supports that
        action_masks[agent][-1] = 1.0

    return action_masks, action_mappings


class RolloutBuffer:
    """Rollout buffer for collecting experiences"""
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.data = {
            "observations": [],
            "actions": [],
            "action_log_probs": [],
            "values": [],
            "rewards": [],
            "masks": [],
            "action_masks": [],
            "advantages": [],
            "returns": []
        }
    
    def add_step(self, obs_batch, actions_batch, log_probs_batch, values_batch, 
                 rewards_batch, masks_batch, action_masks_batch):
        self.data["observations"].append(obs_batch)
        self.data["actions"].append(actions_batch)
        self.data["action_log_probs"].append(log_probs_batch)
        self.data["values"].append(values_batch)
        self.data["rewards"].append(rewards_batch)
        self.data["masks"].append(masks_batch)
        self.data["action_masks"].append(action_masks_batch)
    
    def compute_gae(self, gamma=0.99, lam=0.95):
        """Compute per-agent GAE for competitive/individual rewards.
        
        rewards: [T, n_agents] - individual reward per agent per timestep
        values:  [T, n_agents] - per-agent value estimates
        masks:   [T] - episode continuation mask (shared)
        """
        if not self.data["rewards"]:
            return
        
        # Ensure all tensors are float32
        rewards = torch.stack(self.data["rewards"]).float()   # [T, n_agents]
        values = torch.stack(self.data["values"]).float()     # [T, n_agents]
        masks = torch.stack(self.data["masks"]).float()       # [T]
        
        T = len(rewards)
        N = rewards.shape[-1] if rewards.dim() > 1 else 1
        
        # Ensure proper shapes for per-agent computation
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)  # [T, 1]
        if values.dim() == 1:
            values = values.unsqueeze(-1)    # [T, 1]
        
        advantages = torch.zeros_like(rewards, dtype=torch.float32)  # [T, N]
        
        # Per-agent GAE computation
        for agent_idx in range(N):
            gae = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = 0.0
                else:
                    next_value = values[t + 1, agent_idx]
                
                delta = rewards[t, agent_idx] + gamma * next_value * masks[t] - values[t, agent_idx]
                gae = delta + gamma * lam * masks[t] * gae
                advantages[t, agent_idx] = gae
        
        returns = advantages + values  # [T, N]
        
        # Normalize advantages per agent independently (competitive setting)
        for agent_idx in range(N):
            agent_adv = advantages[:, agent_idx]
            if agent_adv.numel() > 1 and agent_adv.std() > 1e-8:
                advantages[:, agent_idx] = (agent_adv - agent_adv.mean()) / (agent_adv.std() + 1e-8)
        
        # Ensure float32 dtype
        advantages = advantages.float()
        returns = returns.float()
        
        self.data["advantages"] = [adv for adv in advantages]
        self.data["returns"] = [ret for ret in returns]

    def get_batches(self, batch_size):
        """Create mini-batches from buffer data"""
        if not self.data["rewards"]:
            return
        
        # Get total number of samples
        total_samples = len(self.data["observations"])
        
        # Create shuffled indices
        indices = torch.randperm(total_samples)
        
        # Yield mini-batches
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_indices = indices[start:end]
            
            # Extract batch data
            batch_data = {}
            for key in ["observations", "actions", "action_log_probs", "values", 
                    "action_masks", "advantages", "returns"]:
                batch_data[key] = [self.data[key][i] for i in batch_indices]
            
            yield batch_data


def init_env(configuration):
    """Initialize rideshare environment with capacity filter"""
    env = rideshare_v0.parallel_env(
        configuration=configuration,
        parallel_envs=config.NUM_PARALLEL_ENVS,
        max_steps=general_config.MAX_STEPS,
        device=DEVICE,
        single_seeding=False
    )
    env.reset()
    apply_capacity_filter(env)
    return action_mapping_wrapper_v0(env)


class DICGRunner:
    def __init__(self, args):
        self.args = args
        self.device = DEVICE
        self.best_eval_reward = float("-inf")
        
        # Initialize environment
        configuration = resolve_configuration(args)
        self.env = init_env(configuration)
        self.metric_tracker = RideshareMetricTracker(self.env)
        self.episode_metrics = RideshareEpisodeMetrics()
        
        # Get environment info
        observations, infos = self.env.reset()
        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        
        # Initialize models
        self.actor = DICGActor(
            obs_dim=config.OBS_DIM,
            action_dim=general_config.MAX_ACTIONS_PER_AGENT,
            embedding_dim=config.EMBEDDING_DIM
        ).to(self.device)
        
        self.critic = DICGCritic(
            obs_dim=config.OBS_DIM,
            n_agents=self.n_agents,
            embedding_dim=config.EMBEDDING_DIM
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr_policy, eps=general_config.Optimizer_Epsilon)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.LR_CRITIC, eps=general_config.Optimizer_Epsilon)
        
        self.buffer = RolloutBuffer()
        self.writer, self.checkpoint_dir = setup_tensorboard_logging(args)
        
        # Setup DB logger 
        self.db = ExperimentDB()    
        experiment_meta = dict(
            run_type="training",
            model_family="DICG_de_rideshare",
            model_type="smart",
            name=args.experiment_name,
            variation1="dicg",
            variation2="attention",
            grid_size=self.args.grid_size,
            setting_type=self.args.setting_type,
            parent_experiment_id=None,
            checkpoint_path=self.checkpoint_dir,
            notes="Competitive individual rewards",
        )   
    
        self.run_logger = RunLogger(
            writer=self.writer,
            db=self.db,
            experiment_meta=experiment_meta,
            log_every=general_config.LOG_EVERY,
        )

    def evaluate(self):
        """Evaluate current policy - rideshare metrics"""
        total_rewards = []
        total_pickups = []
        total_dropoffs = []
        total_lengths = []
        eval_seeds = []
        
        # Per-agent reward tracking for competitive environment
        agent_rewards = {agent: [] for agent in self.agents}
        
        # Set models to eval mode
        self.actor.eval()
        self.critic.eval()
        
        with torch.no_grad():
            for eval_episode in range(general_config.VALIDATION_EPISODES):
                seed = get_validation_seed(eval_episode, general_config)
                observations, infos = self.env.reset(seed=seed)
                eval_seeds.append(seed)
                self.metric_tracker.reset()
                
                episode_reward = 0
                episode_agent_rewards = {agent: 0 for agent in self.agents}
                episode_pickups = 0
                episode_dropoffs = 0
                steps = 0
                
                while not torch.all(self.env.finished):
                    current_obs = {agent: observations[agent][0] for agent in self.agents}
                    
                    # Use same action collection as training but deterministic
                    padded_obs = pad_observations(current_obs, env_id=0)
                    action_masks, action_mappings = create_action_masks(self.env, self.agents, current_obs, infos, env_id=0)

                    obs_list = [padded_obs[agent].float() for agent in self.agents]
                    mask_list = [action_masks[agent].float() for agent in self.agents]
                    obs_batch = torch.stack(obs_list).unsqueeze(0)
                    mask_batch = torch.stack(mask_list).unsqueeze(0)

                    action_probs, _ = self.actor(obs_batch, mask_batch)
                    
                    # Use deterministic actions (argmax) for evaluation
                    env_actions = {}
                    for i, agent in enumerate(self.agents):
                        action_idx = torch.argmax(action_probs[0, i]).item()
                        
                        # Map to env action
                        if action_idx == (mask_batch.shape[-1] - 1):
                            env_action = (-1, -1)  # NOOP
                        else:
                            avail = action_mappings[agent]
                            if action_idx < len(avail):
                                env_action = avail[action_idx]
                            else:
                                env_action = (-1, -1)

                        env_actions[agent] = torch.tensor(env_action, device=DEVICE).unsqueeze(0)
                    
                    # Environment step
                    self.metric_tracker.pre_step()
                    observations, reward_dict, term, trunc, infos = self.env.step(
                        convert_actions_to_tensor(env_actions, DEVICE)
                    )
                    step_pickups, step_dropoffs, _ = self.metric_tracker.post_step()
                    
                    # Process rideshare metrics
                    episode_pickups += step_pickups
                    episode_dropoffs += step_dropoffs
                    
                    # Track individual rewards (competitive)
                    for agent, reward in reward_dict.items():
                        r = reward[0].item() if torch.is_tensor(reward) else float(reward)
                        episode_agent_rewards[agent] += r
                    
                    # Calculate step reward (sum of individual rewards)
                    step_reward = sum(r[0].item() if torch.is_tensor(r) else float(r) for r in reward_dict.values())
                    episode_reward += step_reward
                    
                    steps += 1
                
                # Store episode metrics
                total_rewards.append(episode_reward)
                total_pickups.append(episode_pickups)
                total_dropoffs.append(episode_dropoffs)
                total_lengths.append(steps)
                
                # Store per-agent rewards
                for agent in self.agents:
                    agent_rewards[agent].append(episode_agent_rewards[agent])
        
        # Set models back to train mode
        self.actor.train()
        self.critic.train()
        
        # Calculate averages
        avg_reward = np.mean(total_rewards)
        avg_pickups = np.mean(total_pickups)
        avg_dropoffs = np.mean(total_dropoffs)
        avg_length = np.mean(total_lengths)
        
        return avg_reward, avg_pickups, avg_dropoffs, avg_length, str(eval_seeds)

    def collect_actions(self, observations, infos, env_id=0):
        padded_obs = pad_observations(observations, env_id=env_id)
        action_masks, action_mappings = create_action_masks(self.env, self.agents, observations, infos, env_id)

        obs_list = [padded_obs[agent].float() for agent in self.agents]
        mask_list = [action_masks[agent].float() for agent in self.agents]
        obs_batch = torch.stack(obs_list).unsqueeze(0)   # [1, n_agents, obs_dim]
        mask_batch = torch.stack(mask_list).unsqueeze(0)  # [1, n_agents, n_actions]

        with torch.no_grad():
            action_probs, _ = self.actor(obs_batch, mask_batch)
            values = self.critic(obs_batch)

        actions_list, log_probs_list, env_actions = [], [], {}
        accept_count, pick_count, drop_count, noop_count = 0, 0, 0, 0
        
        for i, agent in enumerate(self.agents):
            dist = torch.distributions.Categorical(action_probs[0, i])
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)

            actions_list.append(action_idx.item())
            log_probs_list.append(log_prob.item())

            # Map to env action and track action types
            if action_idx.item() == (mask_batch.shape[-1] - 1):
                env_action = (-1, -1)  # NOOP
                noop_count += 1
            else:
                avail = action_mappings[agent]
                if action_idx.item() < len(avail):
                    env_action = avail[action_idx.item()]
                    action_type = env_action[1] if len(env_action) > 1 else -1
                    if action_type == -1 or env_action == (-1, -1):
                        noop_count += 1
                    elif action_type == 0:
                        accept_count += 1
                    elif action_type == 1:
                        pick_count += 1
                    elif action_type == 2:
                        drop_count += 1
                else:
                    env_action = (-1, -1)
                    noop_count += 1

            env_actions[agent] = torch.tensor(env_action, device=DEVICE).unsqueeze(0)

        actions_batch = torch.tensor(actions_list, device=DEVICE, dtype=torch.long)
        log_probs_batch = torch.tensor(log_probs_list, device=DEVICE, dtype=torch.float32)

        return (obs_batch.squeeze(0),
                actions_batch,
                log_probs_batch,
                values.squeeze(0).float(),
                mask_batch.squeeze(0),
                env_actions,
                accept_count,
                pick_count,
                drop_count,
                noop_count)

    def collect_rollout(self, episode):
        # Reset environment with training seed
        seed = get_training_seed(episode, general_config)
        observations, infos = self.env.reset(seed=seed)
        self.metric_tracker.reset()
        self.episode_metrics.reset()
        initial_state = deepcopy(self.env.state())
        episode_reward, step = 0.0, 0
        episode_pickups = 0
        episode_dropoffs = 0
        episode_accept_count = 0
        episode_pick_count = 0
        episode_drop_count = 0
        episode_noop_count = 0
        env_seeds = str(seed)
        
        # Per-agent reward tracking (competitive)
        episode_agent_rewards = {agent: 0.0 for agent in self.agents}

        while not torch.all(self.env.finished):
            current_obs = {agent: observations[agent][0] for agent in self.agents}

            (obs_batch, actions_batch, log_probs_batch,
             values_batch, mask_batch, env_actions, 
             step_accept, step_pick, step_drop, step_noop) = self.collect_actions(current_obs, infos, env_id=0)

            self.metric_tracker.pre_step()
            observations, reward_dict, term, trunc, infos = self.env.step(
                convert_actions_to_tensor(env_actions, DEVICE)
            )
            step_result = self.metric_tracker.post_step()
            step_pickups, step_dropoffs, _ = step_result

            # INDIVIDUAL REWARDS: Track per-agent rewards (competitive environment)
            per_agent_rewards = torch.zeros(len(self.agents), device=DEVICE, dtype=torch.float32)
            for i, agent in enumerate(self.agents):
                r = reward_dict[agent][0].item() if torch.is_tensor(reward_dict[agent]) else float(reward_dict[agent])
                r = np.clip(r, -10, 10)  # Clip for stability
                per_agent_rewards[i] = r
                episode_agent_rewards[agent] += r
            
            # Calculate step reward (sum of individual rewards for total logging)
            reward = per_agent_rewards.sum().item()
            done = any(t[0].item() if torch.is_tensor(t) else bool(t) for t in term.values())
            mask = 0.0 if done else 1.0

            episode_pickups += step_pickups
            episode_dropoffs += step_dropoffs
            episode_accept_count += step_accept
            episode_pick_count += step_pick
            episode_drop_count += step_drop
            episode_noop_count += step_noop

            self.buffer.add_step(
                obs_batch.float(),
                actions_batch,
                log_probs_batch.float(),
                values_batch.float(),  # [n_agents] per-agent values
                per_agent_rewards,     # [n_agents] per-agent rewards (competitive)
                torch.tensor(mask, device=DEVICE, dtype=torch.float32),
                mask_batch.float()
            )

            episode_reward += reward
            self.episode_metrics.record(step_result, noops=step_noop, accepts=step_accept)

            if general_config.DEBUG:
                print_rideshare_grid(
                    self.env.unwrapped.state(),
                    env_id=0,
                    rewards={agent: reward_dict[agent][0].item() if torch.is_tensor(reward_dict[agent]) else float(reward_dict[agent]) for agent in self.agents},
                    step=step,
                    agent_names=self.agents,
                )

            step += 1

        final_state = self.env.state()
        episode_total_tasks = episode_pickups + episode_dropoffs

        return (episode_reward, step, episode_pickups, episode_dropoffs, 
                episode_accept_count, episode_pick_count, episode_drop_count,
                episode_noop_count, episode_total_tasks, env_seeds)
        
    def update_models(self):
        """PPO update with shape-safe approx KL early stop."""
        if not self.buffer.data["rewards"]:
            return {}, {}

        # Compute advantages/returns
        self.buffer.compute_gae()

        actor_losses, critic_losses = [], []

        # Mini-batch sizing
        total_samples = len(self.buffer.data["observations"])
        mini_batch_size = max(1, total_samples // getattr(config, 'NUM_MINI_BATCH', 4))

        for epoch in range(config.PPO_EPOCH):
            epoch_actor_losses, epoch_critic_losses = [], []
            total_approx_kl, kl_batches = 0.0, 0

            for batch_data in self.buffer.get_batches(mini_batch_size):
                # ---- unpack batch (all as tensors) ----
                obs_data = torch.stack(batch_data["observations"]).float()       # [T, N, obs_dim]
                actions = torch.stack(batch_data["actions"]).long()              # [T, N]
                old_logp = torch.stack(batch_data["action_log_probs"]).float()   # [T, N]
                actionmask = torch.stack(batch_data["action_masks"]).float()     # [T, N, A]
                values = torch.stack(batch_data["values"]).float()               # [T, N] per-agent
                returns = torch.stack(batch_data["returns"]).float()             # [T, N] per-agent
                adv = torch.stack(batch_data["advantages"]).float()              # [T, N] per-agent

                # ---- actor forward ----
                action_probs, _ = self.actor(obs_data, actionmask)               # [T, N, A]
                T, N, A = action_probs.shape
                ap_flat = action_probs.view(T * N, A)
                act_flat = actions.view(T * N)

                dist = torch.distributions.Categorical(ap_flat)
                new_logp_flat = dist.log_prob(act_flat)                          # [T*N]
                new_logp = new_logp_flat.view(T, N)                              # [T, N]

                # PPO ratio with per-agent advantages (competitive/individual)
                ratio = torch.exp(new_logp - old_logp)                           # [T, N]
                surr1 = ratio * adv                                              # [T, N] per-agent
                surr2 = torch.clamp(ratio, 1.0 - config.CLIP_PARAM, 1.0 + config.CLIP_PARAM) * adv

                # Entropy bonus (mean over T,N)
                entropy = torch.distributions.Categorical(ap_flat).entropy().view(T, N).mean()

                actor_loss = -torch.min(surr1, surr2).mean() - config.ENTROPY_COEF * entropy

                # ---- critic forward (per-agent values) ----
                new_values = self.critic(obs_data).float()                       # [T, N] per-agent
                critic_loss = F.mse_loss(new_values, returns)                    # per-agent MSE

                # ---- optimize ----
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), config.MAX_GRAD_NORM)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), config.MAX_GRAD_NORM)
                self.critic_optimizer.step()

                epoch_actor_losses.append(actor_loss.item())
                epoch_critic_losses.append(critic_loss.item())

                # ---- approx KL (shape-safe, no grad) ----
                approx_kl_batch = 0.5 * (new_logp.detach() - old_logp.detach()).pow(2).mean().item()
                total_approx_kl += approx_kl_batch
                kl_batches += 1

            # ---- early stop by averaged KL across mini-batches ----
            if kl_batches > 0:
                avg_kl = total_approx_kl / kl_batches
                if avg_kl > 0.015:  # threshold can be moved to config
                    break

            actor_losses.extend(epoch_actor_losses)
            critic_losses.extend(epoch_critic_losses)

        return (
            {"actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0},
            {"critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0},
        )
    
    def train(self, args):
        """Main training loop with rideshare metrics"""
        for episode in range(1, general_config.EPISODES + 1):
            # Start timer 
            t0 = time.perf_counter()
            
            # Collect rollout
            (episode_reward, episode_steps, episode_pickups, episode_dropoffs,
             episode_accept_count, episode_pick_count, episode_drop_count,
             episode_noop_count, episode_total_tasks, env_seeds) = self.collect_rollout(episode)
            self.episode_metrics.log(self.writer, episode)
            
            # Update models
            actor_stats, critic_stats = self.update_models()
            
            # Performance metrics 
            elapsed = time.perf_counter() - t0
            fps = episode_steps / max(elapsed, 1e-9)
            
            # Create training metrics
            train_metrics = {
                'actor_loss': actor_stats.get("actor_loss", 0.0),
                'critic_loss': critic_stats.get("critic_loss", 0.0),
                'entropy': 0.0,  # DICG doesn't expose entropy separately
                'approx_kl': 0.0,
                'value_mse': critic_stats.get("critic_loss", 0.0),  # Use critic loss as proxy
                'explained_variance': 0.0
            }
            
            # Create episode statistics (rideshare metrics)
            episode_stats = dict(
                reward=episode_reward,
                pickups=episode_pickups,
                dropoffs=episode_dropoffs,
                episode_total_tasks=episode_total_tasks,
                new_tasks_completed=episode_dropoffs,
                episode_accept_count=episode_accept_count,
                episode_pick_count=episode_pick_count,
                episode_drop_count=episode_drop_count,
                episode_noop_count=episode_noop_count,
                episode_length=episode_steps,
                env_seeds=env_seeds,
            )

            # Gradient summary (simplified for DICG)
            grad_summary = {
                'actor_vanishing': 0,
                'actor_exploding': 0,
                'critic_vanishing': 0,
                'critic_exploding': 0,
                'actor_max_grad': 0,
                'critic_max_grad': 0,
            }

            # Log episode using RunLogger
            self.run_logger.log_episode(args.experiment_name,
                episode=episode,
                episode_stats=episode_stats,
                train_metrics=train_metrics,
                grad_summary=grad_summary,
                total_actor_grad=0.0,  # DICG doesn't expose grad norms
                total_critic_grad=0.0,
                elapsed=elapsed,
                fps=fps,
            )
            
            # Run evaluation every VALIDATION_EVERY episodes
            if episode % general_config.VALIDATION_EVERY == 0 and episode > 0:
                eval_reward, eval_pickups, eval_dropoffs, eval_length, eval_seeds = self.evaluate()
                
                # Track best for tuning mode
                if general_config.TUNING_MODE and eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
            
                # Log evaluation using RunLogger (map rideshare metrics to interface)
                self.run_logger.log_eval(
                    episode=episode,
                    eval_reward=eval_reward,
                    eval_burnouts=0.0,
                    eval_putouts=0.0,
                    avg_pickups=eval_pickups,
                    avg_dropoffs=eval_dropoffs,
                    avg_dropoffs_success=eval_dropoffs,
                    avg_accepts=0.0,
                    avg_noop=0.0,
                    eval_length=eval_length,
                    avg_convergence=None,
                )
                
                logging.info(f"Evaluation at episode {episode}: "
                            f"Reward: {eval_reward:.4f}, "
                            f"Pickups: {eval_pickups:.4f}, "
                            f"Dropoffs: {eval_dropoffs:.4f}, "
                            f"Length: {eval_length:.2f}")
            
            # Periodic logging (tabulated output)
            if episode % general_config.LOG_EVERY == 0:
                logging.info(tabulate.tabulate([
                    ['Episode', episode],
                    ['Reward', episode_reward],
                    ['Pickups', episode_pickups],
                    ['Dropoffs', episode_dropoffs],
                    ['Accept Actions', episode_accept_count],
                    ['Pick Actions', episode_pick_count],
                    ['Drop Actions', episode_drop_count],
                    ['NOOP Actions', episode_noop_count],
                    ['Actor Loss', train_metrics['actor_loss']],
                    ['Actor Grad Norm', 0.0],
                    ['Critic Loss', train_metrics['critic_loss']],
                    ['Critic Grad Norm', 0.0],
                    ['Entropy', train_metrics['entropy']],
                    ['KL Divergence', train_metrics['approx_kl']],
                    ['Episode Seconds', elapsed],
                    ['Steps/sec', fps],
                    ['Actor Vanishing', grad_summary['actor_vanishing']],
                    ['Actor Exploding', grad_summary['actor_exploding']],
                    ['Critic Vanishing', grad_summary['critic_vanishing']],
                    ['Critic Exploding', grad_summary['critic_exploding']],
                ], headers=['Name', 'Value'], tablefmt='orgtbl'))
            
            # Save checkpoint
            if episode % general_config.SAVE_EVERY == 0 and episode > 0:
                save_checkpoint({
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'episode': episode,
                    'reward': episode_reward
                }, self.checkpoint_dir, episode, args.experiment_name)
            
            self.buffer.clear()
            
            # Periodic cleanup
            if episode % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if general_config.TUNING_MODE:
            return self.best_eval_reward


def main():
    args = parse_args()
    

    args.experiment_name = 'DICG-rideshare'

   
    if general_config.use_global_seed:
        set_global_seed(general_config.Global_Seed, general_config.use_deterministic)
    
    logging.info("=== DICG Training on Rideshare Environment ===")
    logging.info(f"Reward Type: Competitive Individual Rewards")
    
    runner = DICGRunner(args)
    
    try:
        runner.train(args)
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise
    
    runner.writer.close()
    logging.info("Training completed")


if __name__ == "__main__":
    main()
