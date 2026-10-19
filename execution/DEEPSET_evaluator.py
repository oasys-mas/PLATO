"""
DeepSets baseline evaluator implementation - wildfire.

Loads the DeepSet_Actor (set encoder + fixed-index action head) and evaluates
it under the exact same protocol as the other smart baselines. Head slots
beyond the current task count are masked; in zero-shot transfer to larger
grids, tasks beyond the head's fixed capacity are unreachable by construction
(the structural limitation of a fixed-index head under task openness).
"""

import torch
from typing import Dict, Tuple
from torch.distributions import Categorical

from base_evaluator import BaseEvaluator
from policies.deepset_policy import DeepSet_Actor
from utils.observation_graph_helper import observations_to_features, extract_wildfire_action_features, normalize_wildfire_agent_features
from utils.experiment_manager import load_config


class DEEPSETEvaluator(BaseEvaluator):
    """DeepSets baseline evaluator (fixed-index head)."""

    def __init__(self, experiment, shared_db=None):
        self.deepset_config = load_config("DEEPSET")
        super().__init__(experiment, shared_db=shared_db)

    def _load_models(self) -> Dict:
        """Load the DeepSet actor from checkpoint."""
        print(f"Loading DEEPSET checkpoint from: {self.experiment.checkpoint_path}")
        checkpoint = torch.load(self.experiment.checkpoint_path, map_location=self.device, weights_only=False)

        state = checkpoint['actor']
        model_config = checkpoint.get('model_config')
        if model_config is None:
            # Fallback: infer constructor args from the state dict shapes
            # head[-1] weight is [max_tasks + 1, hidden_dim]; phi[0] weight is [hidden_dim, task_feat_dim]
            head_key = [k for k in state.keys() if k.startswith('head') and k.endswith('.weight')][-1]
            phi_key = [k for k in state.keys() if k.startswith('phi') and k.endswith('.weight')][0]
            agent_key = [k for k in state.keys() if k.startswith('agent_encoder') and k.endswith('.weight')][0]
            model_config = dict(
                task_feat_dim=state[phi_key].shape[1],
                agent_feat_dim=state[agent_key].shape[1] // 4,
                hidden_dim=state[phi_key].shape[0],
                max_tasks=state[head_key].shape[0] - 1,
                pooling=getattr(self.deepset_config, 'POOLING', 'sum'),
            )
            print(f"model_config missing from checkpoint; inferred: {model_config}")

        actor = DeepSet_Actor(
            task_feat_dim=model_config['task_feat_dim'],
            agent_feat_dim=model_config['agent_feat_dim'],
            hidden_dim=model_config['hidden_dim'],
            max_tasks=model_config['max_tasks'],
            dropout=model_config.get('dropout', 0.0),
            pooling=model_config.get('pooling', 'sum'),
        ).to(self.device)

        missing, unexpected = actor.load_state_dict(state, strict=False)
        if missing:
            print(f"[load_state_dict] Missing keys: {missing}")
        if unexpected:
            print(f"[load_state_dict] Unexpected keys: {unexpected}")

        actor.eval()
        self.task_feat_dim = model_config['task_feat_dim']
        print(f"Loaded DeepSet actor (max_tasks={actor.max_tasks}, pooling={actor.pooling})")
        return {'actor': actor}

    def _extract_action_features(self, obs):
        """Extract 7-dim direction-agnostic task features (NOOP last).

        The fixed-index head's logits are computed from a permutation-invariant sum over the
        task set, so logit i cannot depend on the features of the task presented at index i.
        Left on the environment's canonical ordering, the head can instead learn which index
        the worthwhile task occupies, which scores task selection it never performs. Randomising
        the presentation order (the same permutation applied to the features and to the action
        mapping, so the task set and every available action are unchanged) removes that
        side-channel. Content-based pointing is invariant to this by construction.
        """
        raw = extract_wildfire_action_features(
            self.env, self.agents, 1, self.device,
            shuffle_tasks=getattr(self.deepset_config, 'Shuffle_Tasks_Eval', False)
        )
        return {agent: raw[(agent, 0)] for agent in self.agents}

    def _get_actions(self, observations, infos) -> Tuple[Dict, Dict]:
        """Get actions for all agents using the fixed-index DeepSet policy."""
        actions_dict = {}
        action_stats = {'noop': 0, 'fight': 0, 'total': 0}

        obs = {agent: observations[agent][0] for agent in self.agents}
        # shuffle_tasks=False during evaluation: no benefit, hurts reproducibility.
        obs_features = observations_to_features(
            1, obs, 3, self.env,
            self.deepset_config.ONLY_AVAILABLE_ACTIONS, self.device,
            shuffle_tasks=False
        )

        action_features = self._extract_action_features(obs)
        actor = self.models['actor']

        for agent in self.agents:
            agent_feat = obs_features[0][agent]

            # Prepare agent features (matching training normalization exactly)
            self_features = agent_feat['self']
            others_features = agent_feat.get('others', torch.empty(0, device=self.device))

            if self_features.numel() == 0:
                agent_features = torch.zeros(1, self.general_config.FEAT_DIM, device=self.device)
            else:
                agent_features = normalize_wildfire_agent_features(self_features, others_features, self.env)

            action_data = action_features[agent]
            task_features = action_data['action_features']
            action_mapping = action_data['action_mapping']

            with torch.no_grad():
                if task_features.shape[0] == 0:
                    env_action = (-1, -1)  # NOOP
                else:
                    logits, _ = actor(task_features, agent_features, None)

                    num_real_tasks = task_features.shape[0] - 1  # exclude the NOOP row
                    valid = actor.valid_head_indices(num_real_tasks)
                    mask = torch.full_like(logits, -1e9)
                    mask[torch.tensor(valid, device=logits.device, dtype=torch.long)] = 0.0

                    temperature = getattr(self.deepset_config, 'logit_temperature', 2.0)
                    dist = Categorical(logits=logits / temperature + mask)
                    action = dist.sample()

                    mapping_index = actor.head_index_to_mapping_index(action.item(), len(action_mapping))
                    env_action = action_mapping[mapping_index]

            if env_action == (-1, -1):
                action_stats['noop'] += 1
            else:
                action_stats['fight'] += 1

            action_stats['total'] += 1
            actions_dict[agent] = torch.tensor([env_action], device=self.device)

        return actions_dict, action_stats

    def _get_model_family(self) -> str:
        """Return the model family name."""
        return "DEEPSET"

    def _get_variation1(self) -> str:
        """Return the actor architecture identifier."""
        return "deepset"

    def _get_variation2(self) -> str:
        """Return the pooling type."""
        return getattr(self.models['actor'], 'pooling', 'sum')
