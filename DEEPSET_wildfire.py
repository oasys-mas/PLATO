#DeepSets baseline - Wildfire
#Deep Sets set encoder + fixed-index action head (rebuttal-promised baseline).
#Trained under the identical setup and protocol as PLATO_wildfire.py: same env,
#same seeds, same GNN critic, same MAPPO/PPO update, same logging/checkpointing.
from copy import deepcopy
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import logging
import time

from modules.centralized_critic_modules import GNNValueCritic, MLPValueCritic

from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from utils.graph_utils_ptr import action_info, get_episode_fire_counts
from utils.network_logger import log_gradient_flow, log_gradient_flow_graph, log_model_architecture, log_network_graph, log_training_flow_graph
from policies.deepset_policy import DeepSet_Actor
from utils.observation_graph_helper import observations_to_features, create_global_bipartite_graph_wildfire, extract_wildfire_action_features, normalize_wildfire_agent_features
from utils.helper import *
from utils.experiment_manager import load_config
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="free_range_zoo")

config = load_config("DEEPSET")
general_config = load_config("General_Config")
TUNING_MODE = general_config.TUNING_MODE

if general_config.DB == "postgres":
    from experiment_logger_postgres import ExperimentDB, RunLogger
else:
    from experiment_logger_sqlite import ExperimentDB, RunLogger


if general_config.DEVICE != (None or ''):
    DEVICE = general_config.DEVICE
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
            'valid_indices': [],
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

        rewards = torch.as_tensor(self.data['rewards'], device=DEVICE, dtype=torch.float32)
        values = torch.as_tensor(self.data['values'], device=DEVICE, dtype=torch.float32)
        masks = torch.as_tensor(self.data['masks'], device=DEVICE, dtype=torch.float32)

        # Group by timestep.
        # IMPORTANT: samples are appended agent-major then env-major within a
        # timestep: index = a * E + e (same layout as PLATO_wildfire.py).
        E = config.NUM_PARALLEL_ENVS
        A = len(self.agents)

        if rewards.numel() % (E * A) != 0:
            print(f"Warning: Total samples {rewards.numel()} not divisible by {E * A}")
            pad_size = (E * A) - (rewards.numel() % (E * A))
            rewards = torch.cat([rewards, torch.zeros(pad_size, device=DEVICE)])
            values = torch.cat([values, torch.zeros(pad_size, device=DEVICE)])
            masks = torch.cat([masks, torch.zeros(pad_size, device=DEVICE)])

        T = rewards.numel() // (E * A)

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
                v_last_tensor = v_last_tensor.mean(dim=-1)

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
                    idx = t * (E * A) + a * E + e
                    expanded_advantages[idx] = advantages[t, e]
                    expanded_returns[idx] = returns[t, e]

        original_size = len(self.data['rewards'])
        expanded_advantages = expanded_advantages[:original_size]
        expanded_returns = expanded_returns[:original_size]

        # Always normalize advantages
        if expanded_advantages.std() > 1e-8:
            expanded_advantages = (expanded_advantages - expanded_advantages.mean()) / (expanded_advantages.std() + 1e-8)

        self.data['advantages'] = expanded_advantages.cpu().tolist()
        self.data['returns'] = expanded_returns.cpu().tolist()

        if general_config.DEBUG:
            print(f"DICG-style GAE - Mean: {expanded_advantages.mean().item():.4f}, Std: {expanded_advantages.std().item():.4f}")

    def get_batches(self, batch_size):
        """Create mini-batches from centralized buffer data"""
        if not self.data['obs']:
            print("Warning: No samples in buffer!")
            return

        num_samples = len(self.data['obs'])
        all_samples = []

        for i in range(num_samples):
            sample = tuple(self.data[key][i] for key in [
                'obs', 'cent_obs', 'actions', 'values', 'returns',
                'log_probs', 'advantages', 'task_features',
                'agent_features', 'valid_indices'
            ])
            all_samples.append(sample)

        indices = torch.randperm(len(all_samples))
        for start in range(0, len(all_samples), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = [all_samples[i] for i in batch_indices]
            yield list(zip(*batch))


class ActorWrapper:
    """Wrapper for the DeepSet actor with fixed-index head sampling.

    The actor outputs a FIXED-length logit vector [max_tasks + 1] (NOOP last).
    Head slots >= the current task count are masked out, and the sampled head
    index is mapped back to the env action mapping by the caller.
    """

    def __init__(self, actor, device):
        self.actor = actor
        self.device = device
        self.current_episode = 0

    def get_action(self, task_features, agent_features, agent, env_id):
        """Get action for single environment. Returns a HEAD index."""
        with torch.no_grad():
            if task_features.shape[0] == 0:
                # Only NOOP available (no tasks): pick the dedicated NOOP slot
                action = torch.tensor(self.actor.max_tasks, device=self.device)
                log_prob = torch.tensor(0.0, device=self.device)
                logits = torch.zeros(self.actor.max_tasks + 1, device=self.device)
                return action, log_prob, logits

            logits, _ = self.actor(task_features, agent_features, None)

            num_real_tasks = task_features.shape[0] - 1  # exclude the NOOP row
            valid = self.actor.valid_head_indices(num_real_tasks)
            mask = torch.full_like(logits, -1e9)
            mask[torch.tensor(valid, device=logits.device, dtype=torch.long)] = 0.0

            temperature = getattr(config, 'logit_temperature', 2.0)
            dist = Categorical(logits=logits / temperature + mask)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, logits

    def evaluate_actions(self, task_features_batch, agent_features_batch,
                         actions_batch, valid_indices_batch):
        """Evaluate actions for training (fixed-index head)."""
        log_probs = []
        entropies = []

        for task_feat, agent_feat, action, valid_idx in zip(
            task_features_batch, agent_features_batch, actions_batch,
            valid_indices_batch
        ):
            if task_feat.shape[0] == 0:
                dummy_param = next(self.actor.parameters())
                log_prob = dummy_param.sum() * 0.0
                entropy = dummy_param.sum() * 0.0
            else:
                logits, _ = self.actor(task_feat, agent_feat, None)
                temperature = getattr(config, 'logit_temperature', 2.0)
                logits = logits / temperature
                mask = torch.full_like(logits, -float('1e9'))
                if len(valid_idx) > 0:
                    mask[valid_idx] = 0.0
                mask[-1] = 0.0  # NOOP slot always selectable

                dist = Categorical(logits=logits + mask)
                log_prob = dist.log_prob(action.to(self.device))
                entropy = dist.entropy()

            log_probs.append(log_prob)
            entropies.append(entropy)

        return torch.stack(log_probs), torch.stack(entropies).mean()


class MAPPOTrainer:
    """Clean MAPPO trainer implementation (identical update rule to PLATO)"""

    def __init__(self, actor, critic, num_agents, device):
        self.actor_wrapper = ActorWrapper(actor, device)
        self.critic = critic
        self.device = device
        self.num_agents = num_agents
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.LR_ACTOR, eps=general_config.Optimizer_Epsilon)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.LR_CRITIC, eps=general_config.Optimizer_Epsilon)

        # PPO parameters
        self.clip_param = config.CLIP_PARAM
        self.value_clip_param = config.VALUE_CLIP_PARAM
        self.entropy_coef = config.ENTROPY_COEF
        self.max_grad_norm = config.MAX_GRAD_NORM
        self.update_count = 0

        # Running statistics for return normalization
        self.return_mean = 0.0
        self.return_std = 1.0
        self.return_count = 0

    def update(self, batch):
        """Single PPO update step with DICG-style stability"""
        self.update_count += 1

        (obs_batch, cent_obs_batch, actions_batch, values_batch, returns_batch,
         old_log_probs_batch, advantages_batch, task_features_batch,
         agent_features_batch, valid_indices_batch) = batch

        returns = torch.tensor(returns_batch, device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs_batch, device=self.device, dtype=torch.float32)
        advantages = torch.tensor(advantages_batch, device=self.device, dtype=torch.float32)
        advantages = advantages.clamp(-5.0, 5.0)

        old_values = torch.tensor(values_batch, device=self.device, dtype=torch.float32)

        if general_config.DEBUG:
            if advantages.abs().sum() < 1e-6:
                print("advantages are zero for this batch")

        # Get new action log probs and entropy
        new_log_probs, entropy = self.actor_wrapper.evaluate_actions(
            task_features_batch, agent_features_batch, actions_batch,
            valid_indices_batch
        )

        # Get new values
        new_values = self._get_values(cent_obs_batch)

        # Ensure tensor sizes match (handle potential mismatches)
        min_size = min(len(new_log_probs), len(old_log_probs), len(advantages),
                       len(new_values), len(returns))

        new_log_probs = new_log_probs[:min_size]
        old_log_probs = old_log_probs[:min_size]
        advantages = advantages[:min_size]
        new_values = new_values[:min_size].float()
        returns = returns[:min_size]
        old_values = old_values[:min_size]

        # Actor loss (PPO clipped objective)
        ratio = torch.exp(new_log_probs - old_log_probs)
        ratio = torch.clamp(ratio, 0.1, 10.0)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

        # Normalize returns using running statistics (reduces variance)
        batch_mean = returns.mean().item()
        batch_std = returns.std().item() + 1e-8

        if self.return_count == 0:
            self.return_mean = batch_mean
            self.return_std = batch_std
        else:
            alpha = 0.01  # Slow update for stability
            self.return_mean = (1 - alpha) * self.return_mean + alpha * batch_mean
            self.return_std = (1 - alpha) * self.return_std + alpha * batch_std
        self.return_count += 1

        normalized_returns = (returns - self.return_mean) / (self.return_std + 1e-8)
        normalized_new_values = (new_values - self.return_mean) / (self.return_std + 1e-8)
        normalized_old_values = (old_values - self.return_mean) / (self.return_std + 1e-8)

        # Clipped value loss (PPO-style) to prevent large value updates
        value_pred_clipped = normalized_old_values + torch.clamp(
            normalized_new_values - normalized_old_values,
            -self.value_clip_param,
            self.value_clip_param
        )

        value_loss_unclipped = nn.functional.huber_loss(normalized_new_values, normalized_returns, reduction='none', delta=10.0)
        value_loss_clipped = nn.functional.huber_loss(value_pred_clipped, normalized_returns, reduction='none', delta=10.0)

        critic_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

        value_mse = nn.functional.mse_loss(new_values, returns)
        explained_variance = 1 - torch.var(returns - new_values) / (torch.var(returns) + 1e-8)

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_wrapper.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Update critic with stricter gradient clipping
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm * 0.5)
        self.critic_optimizer.step()

        log_ratio = new_log_probs - old_log_probs
        approx_kl = (log_ratio.exp() - 1 - log_ratio).mean()

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
        """Get critic values using GNNValueCritic with enriched node features"""
        values = []
        for obs_item in cent_obs_batch:
            if len(obs_item) == 3:
                agent_nodes, task_nodes, edge_index = obs_item
            else:
                agent_nodes, task_nodes, edge_index = obs_item[0], obs_item[1], obs_item[2]

            value = self.critic(
                agent_nodes.to(self.device),
                task_nodes.to(self.device) if task_nodes is not None else None,
                edge_index.to(self.device) if edge_index is not None else None
            )
            values.append(value.squeeze())
        return torch.stack(values) if values else torch.empty(0, device=self.device)


class DEEPSETRunner:
    """Main training runner"""

    def __init__(self, args):
        self.args = args
        self.device = DEVICE
        self.best_eval_reward = float("-inf")

        # Setup environment and models
        self.env = self._init_env()

        # Derive grid size from environment state (after env creation, pre-reset state is set in _init_models)
        observations, _ = self.env.reset()
        _fires = self.env.state().fires[0].squeeze(0)
        self.grid_size = (_fires.shape[0], _fires.shape[1])
        # Fixed-index head capacity = grid cell count (upper bound on concurrent fires)
        self.max_tasks = int(self.grid_size[0] * self.grid_size[1])

        self.actor, self.critic, self.agents = self._init_models(args)

        # Setup trainer and buffer
        self.trainer = MAPPOTrainer(self.actor, self.critic, len(self.agents), self.device)
        self.buffer = RolloutBuffer(self.agents)

        # Setup logging
        self.writer, self.checkpoint_dir = setup_tensorboard_logging(args)
        if config.log_network:
            log_dir = self.checkpoint_dir.replace('checkpoints', 'network_logs')
            log_model_architecture(self.actor, self.critic, log_dir)
            log_network_graph(self.actor, self.critic, log_dir)
            log_training_flow_graph(log_dir)

        # Setup DB logger
        self.db = ExperimentDB()
        experiment_meta = dict(
            run_type="training",
            model_family="DEEPSET",
            model_type="smart",
            name=args.experiment_name,
            variation1="deepset",
            variation2=config.POOLING,
            grid_size=self.args.grid_size,
            setting_type=self.args.setting_type,
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

    def _init_models(self, args):
        """Initialize actor and critic models"""
        agents = self.env.agents
        actor = DeepSet_Actor(
            task_feat_dim=7,  # [|dx|, |dy|, dist, in_range, intensity, fuel, agents_in_range]
            agent_feat_dim=general_config.FEAT_DIM,
            hidden_dim=config.ACTOR_HIDDEN_DIM,
            max_tasks=self.max_tasks,
            dropout=config.ACTOR_DROPOUT,
            pooling=config.POOLING,
        ).to(self.device)

        # Critic identical to PLATO's (isolates the actor architecture)
        if getattr(args, 'ablation_no_gnn_critic', False):
            critic = MLPValueCritic(agent_feat_dim=11, task_feat_dim=9, hidden_dim=64).to(self.device)
        else:
            critic = GNNValueCritic(agent_feat_dim=11, task_feat_dim=9, hidden_dim=64).to(self.device)

        return actor, critic, agents

    def _model_config(self):
        """Constructor arguments saved with each checkpoint for exact rebuild."""
        return dict(
            task_feat_dim=7,
            agent_feat_dim=general_config.FEAT_DIM,
            hidden_dim=config.ACTOR_HIDDEN_DIM,
            max_tasks=self.max_tasks,
            dropout=config.ACTOR_DROPOUT,
            pooling=config.POOLING,
        )

    def _head_to_env_action(self, action_item, action_mapping):
        """Map a sampled head index back to the env action mapping."""
        mapping_index = self.actor.head_index_to_mapping_index(action_item, len(action_mapping))
        return action_mapping[mapping_index]

    def evaluate(self):
        """Evaluate current policy"""
        total_rewards = []
        total_burnouts = []
        total_putouts = []
        total_lengths = []
        eval_seeds = []

        self.actor.eval()

        with torch.no_grad():
            for eval_episode in range(general_config.VALIDATION_EPISODES):
                seed = get_validation_seed(eval_episode, general_config)
                observations, _ = self.env.reset(seed=seed)
                eval_seeds.append(seed)

                episode_reward = 0
                episode_putouts = 0
                episode_burnouts = 0
                steps = 0

                while (not torch.all(self.env.finished)):
                    obs = {agent: observations[agent][0] for agent in self.agents}

                    obs_features = observations_to_features(
                        config.NUM_PARALLEL_ENVS, obs, 3, self.env,
                        config.ONLY_AVAILABLE_ACTIONS, self.device, shuffle_tasks=config.Shuffle_Tasks
                    )

                    action_features = self._extract_action_features(obs, self.agents)

                    actions_dict = {}
                    for agent in self.agents:
                        agent_actions = []
                        for env_id in range(config.NUM_PARALLEL_ENVS):
                            agent_feat = obs_features[env_id][agent]
                            action_data = action_features[(agent, env_id)]

                            self_features = agent_feat['self']
                            others_features = agent_feat.get('others', torch.empty(0, device=self.device))
                            if self_features.numel() == 0:
                                agent_features = torch.zeros(1, general_config.FEAT_DIM, device=self.device)
                            else:
                                agent_features = self._normalize_agent_features(self_features, others_features)

                            action, log_prob, logits = self.trainer.actor_wrapper.get_action(
                                action_data['action_features'], agent_features, agent, env_id
                            )

                            env_action = self._head_to_env_action(action.item(), action_data['action_mapping'])

                            agent_actions.append(torch.tensor(env_action, device=self.device))

                        actions_dict[agent] = torch.stack(agent_actions)

                    observations, rewards, terms, truncs, infoes = self.env.step(actions_dict)

                    for env_id in range(config.NUM_PARALLEL_ENVS):
                        reward = np.mean([r[env_id].mean().item() for r in rewards.values()])
                        burnouts = np.mean([r.item() for r in infoes['burnouts']])
                        putouts = np.mean([r.item() for r in infoes['putouts']])

                        episode_reward += reward
                        episode_burnouts += burnouts
                        episode_putouts += putouts

                    steps += 1

                total_rewards.append(episode_reward / config.NUM_PARALLEL_ENVS)
                total_burnouts.append(episode_burnouts / config.NUM_PARALLEL_ENVS)
                total_putouts.append(episode_putouts / config.NUM_PARALLEL_ENVS)
                total_lengths.append(steps)

        self.actor.train()

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
        Task features: [|dx|, |dy|, dist, in_range, intensity, fuel, agents_in_range] = 7 dims
        """
        return extract_wildfire_action_features(
            self.env, agents, config.NUM_PARALLEL_ENVS, self.device,
            shuffle_tasks=shuffle
        )

    def collect_rollout(self, episode):
        """Collect a single rollout"""
        self.trainer.actor_wrapper.current_episode = episode
        seed = get_training_seed(episode, general_config)
        observations, _ = self.env.reset(seed=seed)
        env_seeds = str(seed)

        initial_state = deepcopy(self.env.state())

        episode_reward = 0
        episode_putouts = 0
        episode_burnouts = 0
        episode_noop_count = 0
        episode_fight_count = 0
        episode_supressant_wait_time = 0

        steps = 0

        while (not torch.all(self.env.finished)):
            obs = {agent: observations[agent][0] for agent in self.agents}

            # Get graph data for critic (with NOOP node and enhanced features)
            graph_data = create_global_bipartite_graph_wildfire(
                config.NUM_PARALLEL_ENVS, self.env.state(), self.device,
                grid_size=self.grid_size
            )

            obs_features = observations_to_features(
                config.NUM_PARALLEL_ENVS, obs, 3, self.env,
                config.ONLY_AVAILABLE_ACTIONS, self.device, shuffle_tasks=config.Shuffle_Tasks
            )

            action_features = self._extract_action_features(obs, self.agents,
                                                            shuffle=config.Shuffle_Tasks)
            actions_dict = {}
            for agent in self.agents:
                agent_actions = []
                for env_id in range(config.NUM_PARALLEL_ENVS):
                    agent_feat = obs_features[env_id][agent]

                    self_features = agent_feat['self']
                    others_features = agent_feat.get('others', torch.empty(0, device=self.device))
                    if self_features.numel() == 0:
                        agent_features = torch.zeros(1, general_config.FEAT_DIM, device=self.device)
                    else:
                        agent_features = self._normalize_agent_features(self_features, others_features)

                    action_data = action_features[(agent, env_id)]

                    action, log_prob, logits = self.trainer.actor_wrapper.get_action(
                        action_data['action_features'], agent_features, agent, env_id
                    )

                    env_action = self._head_to_env_action(action.item(), action_data['action_mapping'])

                    agent_actions.append(torch.tensor(env_action, device=self.device))

                    # Get critic value (using task_nodes which includes NOOP)
                    with torch.no_grad():
                        value = self.trainer._get_values([
                            (graph_data[env_id]['agent_nodes'],
                             graph_data[env_id]['task_nodes'],
                             graph_data[env_id]['edge_index'])
                        ])
                        value_scalar = value[0].item() if len(value) > 0 else 0.0

                    num_real_tasks = action_data['action_features'].shape[0] - 1
                    self.buffer.add_step(agent,
                        obs=obs[agent],
                        cent_obs=(graph_data[env_id]['agent_nodes'],
                                  graph_data[env_id]['task_nodes'],
                                  graph_data[env_id]['edge_index']),
                        actions=action,
                        log_probs=log_prob.item(),
                        values=value_scalar,
                        task_features=action_data['action_features'],
                        # valid HEAD indices (fixed-index slots + NOOP slot)
                        valid_indices=self.actor.valid_head_indices(num_real_tasks),
                        agent_features=agent_features,
                    )

                actions_dict[agent] = torch.stack(agent_actions)

            # Environment step
            observations, rewards, terms, truncs, infoes = self.env.step(actions_dict)
            noop_counts, fight_counts = action_info(actions_dict)

            reward = np.mean([r.mean().item() for r in rewards.values()])
            burnouts = np.mean([r.item() for r in infoes['burnouts']])
            putouts = np.mean([r.item() for r in infoes['putouts']])
            step_supressant_wait_time = count_zero_suppressant_agents(self.env.state())

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
            episode_supressant_wait_time += step_supressant_wait_time

            steps += 1

        last_state = self.env.state()
        episode_total_fires_count, new_fire_ignited_count = get_episode_fire_counts(initial_state, last_state, episode_burnouts, episode_putouts, config.NUM_PARALLEL_ENVS)

        return episode_reward, steps, episode_burnouts, episode_putouts, episode_total_fires_count, new_fire_ignited_count, episode_noop_count, episode_fight_count, env_seeds, episode_supressant_wait_time

    def train(self):
        """Main training loop"""
        for episode in range(1, general_config.EPISODES + 1):
            t0 = time.perf_counter()

            # Collect rollout
            episode_reward, episode_steps, episode_burnouts, episode_putouts, episode_total_fires_count, new_fire_ignited_count, episode_noop_count, episode_fight_count, env_seeds, episode_supressant_wait_time = self.collect_rollout(episode)

            # Compute v_last from final state for proper GAE bootstrapping (DICG-style)
            final_graph_data = create_global_bipartite_graph_wildfire(
                config.NUM_PARALLEL_ENVS, self.env.state(), self.device,
                grid_size=self.grid_size
            )
            with torch.no_grad():
                v_last_list = []
                for e in range(config.NUM_PARALLEL_ENVS):
                    v = self.trainer._get_values([(final_graph_data[e]['agent_nodes'],
                                                   final_graph_data[e]['task_nodes'],
                                                   final_graph_data[e]['edge_index'])])
                    v_last_list.append(v[0].item() if len(v) > 0 else 0.0)
                self.buffer.compute_gae(v_last=v_last_list)

            # Training updates
            train_metrics = {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0, 'approx_kl': 0, 'value_mse': 0, 'explained_variance': 0}
            num_updates = 0

            for epoch in range(config.PPO_EPOCH):
                for batch in self.buffer.get_batches(config.BATCH_SIZE // config.NUM_MINI_BATCH):
                    metrics = self.trainer.update(batch)
                    for key in train_metrics:
                        train_metrics[key] += metrics[key]
                    num_updates += 1

                    # Early stopping on high KL divergence
                    if metrics['approx_kl'] > config.TARGET_KL:
                        break
                if metrics['approx_kl'] > config.TARGET_KL:
                    break
            if config.log_network:
                grad_summary = log_gradient_flow(self.actor, self.critic, episode, self.writer)
            else:
                grad_summary = {
                    'actor_vanishing': 0,
                    'actor_exploding': 0,
                    'critic_vanishing': 0,
                    'critic_exploding': 0,
                    'actor_max_grad': 0,
                    'critic_max_grad': 0,
                }

            # Average metrics
            for key in train_metrics:
                train_metrics[key] /= max(num_updates, 1)

            total_actor_grad = sum(p.grad.norm().item() for p in self.actor.parameters() if p.grad is not None)
            total_critic_grad = sum(p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None)

            elapsed = time.perf_counter() - t0
            fps = episode_steps / max(elapsed, 1e-9)

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

                self.run_logger.log_eval(
                    episode=episode,
                    eval_reward=eval_reward,
                    eval_burnouts=eval_burnouts,
                    eval_putouts=eval_putouts,
                    avg_pickups=None,
                    avg_dropoffs=None,
                    avg_dropoffs_success=None,
                    avg_accepts=None,
                    avg_noop=0,
                    eval_length=eval_length,
                    avg_convergence=None,
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
                    "reward": episode_reward,
                    "model_config": self._model_config(),
                }, self.checkpoint_dir, episode, self.args.experiment_name)

            # Clear buffer
            self.buffer.clear()

        if TUNING_MODE:
            return self.best_eval_reward


def main():
    """Main function"""
    args = parse_args()
    args.experiment_name = 'DEEPSET_wildfire'
    if (general_config.use_global_seed):
        set_global_seed(general_config.Global_Seed, general_config.use_deterministic)

    print("=== CONFIG Initialized ===")
    print(f"NUM_PARALLEL_ENVS: {config.NUM_PARALLEL_ENVS}")
    print(f"ONLY_AVAILABLE_ACTIONS: {config.ONLY_AVAILABLE_ACTIONS}")
    print(f"MAX_STEPS: {general_config.MAX_STEPS}")
    print(f"FEAT_DIM: {general_config.FEAT_DIM}")
    print(f"POOLING: {config.POOLING}")
    print(f"NumPy: {np.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Python: {sys.version}")

    runner = DEEPSETRunner(args)
    print(f"Fixed-index head capacity (max_tasks): {runner.max_tasks} "
          f"(grid {runner.grid_size[0]}x{runner.grid_size[1]})")
    runner.train()


if __name__ == '__main__':
    main()
