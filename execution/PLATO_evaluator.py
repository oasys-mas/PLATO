"""
PLATO-specific evaluator implementation - wildfire & rideshare.
"""

import torch
import numpy as np
from typing import Dict, Tuple
from torch.distributions import Categorical

from base_evaluator import BaseEvaluator
from policies import pointer_MLP_policy
from policies.pointer_GRU_policy import Pointer_GRU_Actor
from policies.pointer_LSTM_policy import Pointer_LSTM_Actor
from policies.pointer_MLP_policy import Pointer_MLP_Actor
from utils.observation_graph_helper import observations_to_features, extract_wildfire_action_features, normalize_wildfire_agent_features
from utils.experiment_manager import load_config


class PLATOEvaluator(BaseEvaluator):
    """PLATO algorithm evaluator with proper hidden state management."""
    def __init__(self, experiment, shared_db=None):
        self.plato_config = load_config("PLATO")
        super().__init__(experiment, shared_db=shared_db)

        # Initialize hidden states storage
        self.hidden_states = {}
        self._init_hidden_states()

        # ── Logit diagnostics ──────────────────────────────────────────────────
        # Set debug_logits=True to print a per-step breakdown of task features,
        # query vector, raw logits, and chosen action for every agent decision.
        # Set debug_logits_max_steps to limit output (None = unlimited).
        self.debug_logits = self.general_config.DEBUG
        self.debug_logits_max_steps = 20   # print first N steps then go quiet
        self._debug_step = 0
    
    def _load_models(self) -> Dict:
        """Load PLATO models from checkpoint."""
        print(f"Loading PLATO checkpoint from: {self.experiment.checkpoint_path}")
        checkpoint = torch.load(self.experiment.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Detect RNN type from checkpoint keys
        state = checkpoint['actor']
        ckpt_keys = list(state.keys())
        has_lstm = any('lstm' in k.lower() for k in ckpt_keys)
        has_mlp = any('mlp' in k.lower() for k in ckpt_keys)
        
        # Detect attention type
        attn_type = getattr(self.experiment, 'attention_type', None)
        if attn_type is None:
            if any('attn.Wk' in k or 'attn.Wq' in k or 'attn.v' in k for k in ckpt_keys):
                attn_type = 'additive'
            else:
                attn_type = 'dotproduct'
        print(f"Detected attention_type={attn_type}")

        # Auto-detect task_feat_dim from checkpoint's task_encoder weight
        task_feat_dim = 4  # default wildfire
        for k in ckpt_keys:
            if 'task_encoder.0.weight' in k:
                task_feat_dim = state[k].shape[1]  # nn.Linear weight is [out, in]
                break
        print(f"Auto-detected task_feat_dim={task_feat_dim} from checkpoint")

        # Derive actor ablation flags from the experiment's pointer_policy name
        pol = getattr(self.experiment, 'pointer_policy', '') or ''
        no_team_stats = 'noteamstats' in pol
        no_pointer    = 'nopointer'   in pol
        if no_team_stats or no_pointer:
            print(f"Ablation flags: no_team_stats={no_team_stats}, no_pointer={no_pointer}")

        # Instantiate the matching actor
        if has_lstm:
            print(f"Loading LSTM actor (task_feat_dim={task_feat_dim})")
            actor = Pointer_LSTM_Actor(
                task_feat_dim=task_feat_dim,
                agent_feat_dim=self.general_config.FEAT_DIM,
                hidden_dim=self.plato_config.ACTOR_HIDDEN_DIM,
                attention_type=attn_type,
                ablation_no_team_stats=no_team_stats,
                ablation_no_pointer=no_pointer,
            ).to(self.device)
        else:
            print(f"Loading MLP actor (task_feat_dim={task_feat_dim})")
            actor = Pointer_MLP_Actor(
                task_feat_dim=task_feat_dim,
                agent_feat_dim=self.general_config.FEAT_DIM,
                hidden_dim=self.plato_config.ACTOR_HIDDEN_DIM,
                attention_type=attn_type,
                ablation_no_team_stats=no_team_stats,
                ablation_no_pointer=no_pointer,
            ).to(self.device)
 
        # Load weights
        missing, unexpected = actor.load_state_dict(state, strict=False)
        if missing:
            print(f"[load_state_dict] Missing keys: {missing}")
        if unexpected:
            print(f"[load_state_dict] Unexpected keys: {unexpected}")
        
        actor.eval()
        self.task_feat_dim = task_feat_dim
        return {'actor': actor}
    
    def _init_hidden_states(self):
        """Initialize hidden states for all agents."""
        for agent in self.agents:
            if hasattr(self.models['actor'], 'memory_lstm') or 'LSTM' in str(type(self.models.get('actor', ''))):
                h = torch.zeros(1, self.plato_config.ACTOR_HIDDEN_DIM, device=self.device)
                c = torch.zeros(1, self.plato_config.ACTOR_HIDDEN_DIM, device=self.device)
                self.hidden_states[agent] = (h, c)
            else:
                self.hidden_states[agent] = torch.zeros(1, self.plato_config.ACTOR_HIDDEN_DIM, device=self.device)
    
    def _extract_action_features(self, obs):
        """Extract action features for all agents (domain-aware).
        
        Wildfire: 7-dim direction-agnostic features [abs_dx_norm, abs_dy_norm,
                  dist_norm, in_range, intensity, fuel, agents_in_range]
                  NOOP included as last task with all-zeros sentinel.
        Rideshare: 4-dim raw task features from task_store
                   NOOP NOT included here (added before forward pass in _get_actions)
        """
        if self.domain == "rideshare":
            return self._extract_action_features_rideshare(obs)
        return self._extract_action_features_wildfire(obs)

    def _extract_action_features_rideshare(self, obs):
        """Extract task features for rideshare (matching training).
        
        Action mapping uses actual task stages from the action space, not hardcoded 0.
        """
        action_features_dict = {}
        
        for agent in self.agents:
            env_id = 0  # Single env for evaluation
            task_indices = self.env.unwrapped.agent_action_mapping[agent][env_id]
            task_stages, locked_idx = self._get_rideshare_task_stages(agent, env_id)
            
            action_features = []
            action_mapping = []
            
            for action_idx, task_id in enumerate(task_indices):
                task_feature = self.env.unwrapped.task_store[env_id][task_id]
                # Slice to model's expected task_feat_dim (env may provide more features)
                action_features.append(task_feature[:self.task_feat_dim].float())
                # Use actual stage from action space as action_type
                stage = task_stages.get(action_idx, 0)
                action_mapping.append((action_idx, stage))
            
            # NOOP added to mapping only - feature added in _get_actions
            action_mapping.append((-1, -1))
            
            if action_features:
                action_features_tensor = torch.stack(action_features)
            else:
                # Use model's task_feat_dim for empty tensor shape
                feat_dim = self.models['actor'].task_encoder[0].in_features
                action_features_tensor = torch.empty(0, feat_dim, device=self.device, dtype=torch.float32)
            
            action_features_dict[agent] = {
                'action_features': action_features_tensor,
                'action_mapping': action_mapping,
            }
        
        return action_features_dict

    def _extract_action_features_wildfire(self, obs):
        """Extract wildfire task features, automatically matching the checkpoint's task_feat_dim.

        Uses the shared utility which returns 7-dim direction-agnostic features:
        [abs_dx_norm, abs_dy_norm, dist_norm, in_range, intensity, fuel, agents_in_range].

        If the checkpoint was trained with a different dim (e.g. old 4-dim absolute),
        we rebuild features to match.
        """
        import math

        expected_dim = self.task_feat_dim

        # Fast path: utility produces 8-dim; if checkpoint expects 8, use directly
        raw = extract_wildfire_action_features(
            self.env, self.agents, 1, self.device,
            shuffle_tasks=False
        )
        sample_dim = next(iter(raw.values()))['action_features'].shape[-1]

        if sample_dim == expected_dim:
            return {agent: raw[(agent, 0)] for agent in self.agents}

        # -------------------------------------------------------------------
        # Dim mismatch: rebuild features to match the checkpoint's expectation.
        # Old 4-dim checkpoint: [x_norm, y_norm, intensity, fuel] (absolute)
        # task_store layout: [row, col, intensity, fuel, ...]
        # -------------------------------------------------------------------
        result = {}
        state = self.env.state()
        fires = state.fires[0].squeeze(0)
        grid_h, grid_w = fires.shape[0], fires.shape[1]
        max_col = float(max(grid_w - 1, 1))
        max_row = float(max(grid_h - 1, 1))

        for agent_idx, agent in enumerate(self.agents):
            env_id = 0
            src = raw[(agent, env_id)]
            action_mapping = src['action_mapping']
            task_indices = self.env.unwrapped.agent_action_mapping[agent][env_id]
            idx_to_task = {action_idx: task_id for action_idx, task_id in enumerate(task_indices)}

            rebuilt = []
            for entry in action_mapping:
                action_idx, _ = entry
                if action_idx == -1:
                    # NOOP sentinel matching old training: [-1, -1, 0, 0]
                    noop = torch.zeros(expected_dim, device=self.device, dtype=torch.float32)
                    noop[0] = -1.0
                    if expected_dim > 1:
                        noop[1] = -1.0
                    rebuilt.append(noop)
                else:
                    task_id = idx_to_task[action_idx]
                    full_feat = self.env.unwrapped.task_store[env_id][task_id].float()
                    # task_store: [row, col, intensity, fuel, ...]
                    fire_row = full_feat[0:1]
                    fire_col = full_feat[1:2]
                    intensity = full_feat[2:3]
                    fuel = full_feat[3:4]
                    full = torch.cat([
                        fire_col / max_col,   # x_norm = col / (width-1)
                        fire_row / max_row,   # y_norm = row / (height-1)
                        intensity,
                        fuel,
                    ])
                    rebuilt.append(full[:expected_dim])

            result[agent] = {
                'action_features': torch.stack(rebuilt),
                'action_mapping': action_mapping,
            }
        return result

    def _get_actions(self, observations, infos) -> Tuple[Dict, Dict]:
        """Get actions for all agents using PLATO policy (domain-aware)."""
        actions_dict = {}
        action_stats = {'noop': 0, 'fight': 0, 'total': 0}
        
        obs = {agent: observations[agent][0] for agent in self.agents}
        # shuffle_tasks=False during evaluation: no benefit, hurts reproducibility.
        obs_features = observations_to_features(
            1, obs, 3, self.env, 
            self.plato_config.ONLY_AVAILABLE_ACTIONS, self.device, 
            shuffle_tasks=False
        )
        
        action_features = self._extract_action_features(obs)
        
        for agent in self.agents:
            agent_feat = obs_features[0][agent]
            
            # Prepare agent features (matching training normalization exactly)
            self_features = agent_feat['self']
            others_features = agent_feat.get('others', torch.empty(0, device=self.device))

            if self_features.numel() == 0:
                agent_features = torch.zeros(1, self.general_config.FEAT_DIM, device=self.device)
            elif self.domain == "rideshare":
                # Rideshare: simple concat of self + others (matching PLATO_rideshare.py training)
                if others_features.numel() == 0:
                    if self_features.dim() == 1:
                        agent_features = self_features.unsqueeze(0)
                    else:
                        agent_features = self_features
                else:
                    if self_features.dim() == 1:
                        self_features = self_features.unsqueeze(0)
                    if self_features.shape[-1] == others_features.shape[-1]:
                        agent_features = torch.cat([self_features, others_features], dim=0)
                    else:
                        agent_features = self_features
            else:
                agent_features = normalize_wildfire_agent_features(self_features, others_features, self.env)

            action_data = action_features[agent]
            task_features = action_data['action_features']
            action_mapping = action_data['action_mapping']
            
            # Get hidden state - LSTM expects 2D tensors [1, hidden_dim]
            if isinstance(self.hidden_states[agent], tuple):
                h, c = self.hidden_states[agent]
                hidden = (h, c)
            else:
                hidden = self.hidden_states[agent]
            
            if self.domain == "rideshare":
                env_action, next_hidden = self._get_rideshare_action(
                    task_features, agent_features, hidden, action_mapping, agent
                )
            else:
                env_action, next_hidden = self._get_wildfire_action(
                    task_features, agent_features, hidden, action_mapping, agent=agent
                )
            
            # Update hidden state (for recurrent models like LSTM)
            is_recurrent = isinstance(self.models['actor'], Pointer_LSTM_Actor)
            if is_recurrent:
                self.hidden_states[agent] = next_hidden
            
            if env_action == (-1, -1):
                action_stats['noop'] += 1
            else:
                action_stats['fight'] += 1
            
            action_stats['total'] += 1
            actions_dict[agent] = torch.tensor([env_action], device=self.device)

        if self.debug_logits:
            self._debug_step += 1

        return actions_dict, action_stats

    def _log_wildfire_decision(self, agent, task_features, agent_features, logits_raw, logits_clamped, action_mapping, chosen_idx):
        """Print a rich per-decision breakdown for zero-shot debugging."""
        sep = "─" * 72
        print(f"\n{sep}")
        print(f"[PLATO debug] step={self._debug_step}  agent={agent}")
        print(f"{sep}")

        # ── Agent query features ──────────────────────────────────────────────
        print("  Agent features (input to query encoder):")
        af = agent_features.cpu()
        for row_i, row in enumerate(af):
            label = "self " if row_i == 0 else f"other{row_i}"
            print(f"    [{label}] {row.tolist()}")

        # ── Per-task feature breakdown ────────────────────────────────────────
        feat_names = ["|dx|", "|dy|", "dist_n", "in_rng", "intens", "fuel", "allies"]
        n_feats = task_features.shape[1]
        names = feat_names[:n_feats] + [f"f{i}" for i in range(n_feats - len(feat_names))]
        print(f"  Task features + logits  ({task_features.shape[0]} tasks incl. NOOP):")
        header = f"    {'idx':>4}  {'mapping':>14}  {'raw_logit':>10}  {'clamped':>10}  "
        header += "  ".join(f"{n:>8}" for n in names)
        print(header)
        for i, (feats, (am_act, am_type), r_l, c_l) in enumerate(
            zip(task_features.cpu(), action_mapping, logits_raw.cpu(), logits_clamped.cpu())
        ):
            tag = "NOOP" if am_act == -1 else f"({am_act},{am_type})"
            chosen_marker = " CHOSEN" if i == chosen_idx else ""
            feat_str = "  ".join(f"{v:8.4f}" for v in feats.tolist())
            print(f"    {i:>4}  {tag:>14}  {r_l.item():>10.4f}  {c_l.item():>10.4f}  {feat_str}{chosen_marker}")

        # ── Softmax probabilities ─────────────────────────────────────────────
        probs = torch.softmax(logits_clamped.cpu(), dim=0)
        prob_str = "  ".join(f"[{i}]={p:.3f}" for i, p in enumerate(probs.tolist()))
        print(f"  Softmax probs: {prob_str}")
        print(sep)

    def _get_wildfire_action(self, task_features, agent_features, hidden, action_mapping, agent=None):
        """Get wildfire action (original logic)."""
        logits_raw = None
        with torch.no_grad():
            if task_features.shape[0] == 0:
                action = torch.tensor(0, device=self.device)
                next_hidden = hidden
            else:
                logits, next_hidden = self.models['actor'](task_features, agent_features, hidden)
                logits_raw = logits.clone()
                mask = torch.zeros_like(logits)
                temperature = getattr(self.plato_config, 'logit_temperature', 2.0)
                logits = logits / temperature
                dist = Categorical(logits=logits + mask)
                action = dist.sample()

        chosen_idx = action.item()
        env_action = action_mapping[chosen_idx]

        if (
            self.debug_logits
            and logits_raw is not None
            and (self.debug_logits_max_steps is None or self._debug_step < self.debug_logits_max_steps)
        ):
            temperature = getattr(self.plato_config, 'logit_temperature', 2.0)
            logits_clamped = logits_raw / temperature
            self._log_wildfire_decision(
                agent or "?", task_features, agent_features,
                logits_raw, logits_clamped, action_mapping, chosen_idx
            )

        return env_action, next_hidden

    def _get_rideshare_action(self, task_features, agent_features, hidden, action_mapping, agent):
        """Get rideshare action with expanded logits and lock-based masking."""
        with torch.no_grad():
            if task_features.shape[0] == 0:
                # Only NOOP available
                return (-1, -1), hidden
            
            # Add NOOP feature as last task before forward pass (match actual feature dim)
            feat_dim = task_features.shape[1]
            noop_features = torch.zeros(feat_dim, device=self.device, dtype=torch.float32)
            task_features_with_noop = torch.cat([
                task_features, noop_features.unsqueeze(0)
            ], dim=0)
            
            logits, next_hidden = self.models['actor'](task_features_with_noop, agent_features, hidden)
            
            # Expand logits: 3 action types per task + 1 NOOP
            num_tasks = task_features_with_noop.shape[0]  # includes NOOP
            num_real_tasks = num_tasks - 1
            total_task_actions = num_real_tasks * 3
            expanded_logits = torch.zeros(total_task_actions + 1, device=self.device)
            
            for task_idx in range(num_real_tasks):
                for action_type in range(3):
                    expanded_idx = task_idx * 3 + action_type
                    expanded_logits[expanded_idx] = logits[task_idx]
            
            expanded_logits[-1] = logits[-1]  # NOOP
            
            # Rideshare lock-based masking using base class helper
            task_stages, locked_idx = self._get_rideshare_task_stages(agent, env_id=0)
            
            mask = torch.full_like(expanded_logits, -float('1e9'))
            
            if locked_idx is None:
                # No active job: allow ACCEPT (type 0) for stray tasks + NOOP
                for t_idx, stage in task_stages.items():
                    if t_idx < num_real_tasks and stage == 0:
                        mask[t_idx * 3 + 0] = 0.0  # Accept
                mask[-1] = 0.0  # NOOP
            else:
                # Active job: lock to specific action for that task
                if locked_idx < num_real_tasks:
                    s_locked = task_stages.get(locked_idx, -1)
                    if s_locked == 1:
                        mask[locked_idx * 3 + 1] = 0.0  # Pickup
                    elif s_locked == 2:
                        mask[locked_idx * 3 + 2] = 0.0  # Dropoff
            
            temperature = getattr(self.plato_config, 'logit_temperature', 2.0)
            expanded_logits = expanded_logits / temperature
            dist = Categorical(logits=expanded_logits + mask)
            action = dist.sample()
        
        # Map expanded action to env format
        action_idx = action.item()
        if action_idx >= total_task_actions:
            env_action = (-1, -1)  # NOOP
        else:
            task_idx = action_idx // 3
            action_type = action_idx % 3
            env_action = (task_idx, action_type)
        
        return env_action, next_hidden
    
    def _get_agent_action(self, task_features, agent_features, valid_indices, agent):
        """Get action for a single agent with proper hidden state management (matching training)."""
        is_recurrent = isinstance(self.models['actor'], Pointer_LSTM_Actor)
        
        # Get current hidden state - LSTM expects 2D tensors [1, hidden_dim]
        if isinstance(self.hidden_states[agent], tuple):
            h, c = self.hidden_states[agent]
            hidden = (h, c)
        else:
            hidden = self.hidden_states[agent]
        
        with torch.no_grad():
            if task_features.shape[0] == 0:
                # Only NOOP available (no tasks)
                action = torch.tensor(0, device=self.device)
                log_prob = torch.tensor(0.0, device=self.device)
                next_hidden = hidden
            else:
                # Forward pass with hidden state
                logits, next_hidden = self.models['actor'](task_features, agent_features, hidden)
                
                # Apply action mask (matching training logic)
                if valid_indices is not None and len(valid_indices) > 0:
                    mask = torch.full_like(logits, -1e9)
                    valid_tensor = torch.tensor(valid_indices, device=logits.device, dtype=torch.long)
                    # Guard against any out-of-range indices
                    valid_tensor = valid_tensor[(valid_tensor >= 0) & (valid_tensor < logits.shape[0])]
                    mask[valid_tensor] = 0.0
                else:
                    mask = torch.zeros_like(logits)
                
                # Temperature scaling (preserves logit ordering unlike clamping)
                temperature = getattr(self.plato_config, 'logit_temperature', 2.0)
                logits = logits / temperature
                dist = Categorical(logits=logits + mask)
                action = dist.sample()
                log_prob = dist.log_prob(action)
        
        # Update hidden state for recurrent models
        if is_recurrent:
            if isinstance(next_hidden, tuple):
                self.hidden_states[agent] = next_hidden
            else:
                self.hidden_states[agent] = next_hidden
        
        return action, log_prob

    def run_episode(self, seed=None):
        """Override to reset hidden states for each episode."""
        self._init_hidden_states()
        return super().run_episode(seed)
    
    def _get_model_family(self) -> str:
        """Return the model family name."""
        return "PLATO"
    
    def _get_variation1(self) -> str:
        """Return the pointer policy type."""
        return getattr(self.experiment, 'pointer_policy', 'mlp')
    
    def _get_variation2(self) -> str:
        """Return the attention type."""
        return getattr(self.experiment, 'attention_type', 'additive')