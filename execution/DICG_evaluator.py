"""
DICG-specific evaluator implementation - supports wildfire and rideshare.
"""

import torch
import numpy as np
from typing import Dict, Tuple

from base_evaluator import BaseEvaluator
from dicg_de_wildfire import DICGActor, DICGCritic, pad_observations
from dicg_de_rideshare import (
    DICGActor as RideshareDICGActor,
    DICGCritic as RideshareDICGCritic,
    pad_observations as rideshare_pad_observations
)
from utils.experiment_manager import load_config
from utils.graph_utils_ptr import convert_actions_to_tensor, get_available_actions


class DICGEvaluator(BaseEvaluator):
    """DICG algorithm evaluator - supports wildfire and rideshare."""
        
    def __init__(self, experiment, shared_db=None):
        self.dicg_config = load_config("DICG")
        super().__init__(experiment, shared_db=shared_db)
    
    
    def _load_models(self) -> Dict:
        """Load DICG models from checkpoint."""
        print(f"Loading DICG checkpoint from: {self.experiment.checkpoint_path}")
        checkpoint = torch.load(self.experiment.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Select model classes based on domain
        if self.domain == "rideshare":
            ActorClass = RideshareDICGActor
            CriticClass = RideshareDICGCritic
        else:
            ActorClass = DICGActor
            CriticClass = DICGCritic
        
        # Create models with same architecture as training
        actor = ActorClass(
            obs_dim=self.dicg_config.OBS_DIM,
            action_dim=self.general_config.MAX_ACTIONS_PER_AGENT,
            embedding_dim=self.dicg_config.EMBEDDING_DIM
        ).to(self.device)
        
        critic = CriticClass(
            obs_dim=self.dicg_config.OBS_DIM,
            n_agents=self.n_agents,
            embedding_dim=self.dicg_config.EMBEDDING_DIM
        ).to(self.device)
        
        # Load state dictionaries
        if 'actor' in checkpoint and 'critic' in checkpoint:
            actor.load_state_dict(checkpoint['actor'])
            critic.load_state_dict(checkpoint['critic'])
        else:
            raise KeyError(f"Expected 'actor' and 'critic' keys in checkpoint, "
                          f"but got keys: {list(checkpoint.keys())}")
        
        # Set to evaluation mode
        actor.eval()
        critic.eval()
        
        return {'actor': actor, 'critic': critic}
    
    def _create_action_masks(self, observations, infos, env_id=0):
        """Create action masks for DICG evaluator - single env version.
        
        Wildfire: Action format (action_index, 0). NOOP: (-1, -1)
        Rideshare: Action format (task_idx, action_type) where action_type comes
                   from the task's current stage. Uses lock mechanism.
        """
        action_masks = {}
        action_mappings = {}

        for agent in self.agents:
            if self.domain == "rideshare":
                # Rideshare: determine action types from task stages
                task_stages, locked_idx = self._get_rideshare_task_stages(agent, env_id)
                task_indices = self.env.unwrapped.agent_action_mapping[agent][env_id]
                
                mask = torch.zeros(self.general_config.MAX_ACTIONS_PER_AGENT, device=self.device)
                
                if locked_idx is not None:
                    # Agent has a committed task - force that action, NO NOOP
                    stage = task_stages[locked_idx]
                    action_mappings[agent] = [(locked_idx, stage)]
                    mask[0] = 1.0  # Only one valid action (no NOOP while job active)
                elif len(task_stages) > 0:
                    # No committed task - allow accepting available tasks + NOOP
                    mappings = []
                    for t_idx, stage in task_stages.items():
                        mappings.append((t_idx, stage))
                    mappings = mappings[:self.general_config.MAX_ACTIONS_PER_AGENT]
                    count = len(mappings)
                    mask[:count] = 1.0
                    mask[-1] = 1.0  # Allow NOOP when no active job
                    action_mappings[agent] = mappings
                else:
                    mask[-1] = 1.0  # NOOP
                    action_mappings[agent] = [(-1, -1)]
                
                action_masks[agent] = mask
            else:
                # Wildfire: original logic
                if infos and agent in infos and "action_mask" in infos[agent]:
                    m = infos[agent]["action_mask"][env_id].float().to(self.device)
                    if m.numel() < self.general_config.MAX_ACTIONS_PER_AGENT:
                        pad = torch.zeros(self.general_config.MAX_ACTIONS_PER_AGENT - m.numel(), device=self.device)
                        m = torch.cat([m, pad], dim=0)
                    else:
                        m = m[:self.general_config.MAX_ACTIONS_PER_AGENT]
                    mask = m
                    task_indices = self.env.unwrapped.agent_action_mapping[agent][env_id]
                    action_mappings[agent] = [(i, 0) for i in range(len(task_indices))][:self.general_config.MAX_ACTIONS_PER_AGENT]
                else:
                    task_indices = self.env.unwrapped.agent_action_mapping[agent][env_id]
                    mask = torch.zeros(self.general_config.MAX_ACTIONS_PER_AGENT, device=self.device)
                    if len(task_indices) > 0:
                        count = min(len(task_indices), self.general_config.MAX_ACTIONS_PER_AGENT)
                        mask[:count] = 1.0
                        action_mappings[agent] = [(i, 0) for i in range(count)]
                    else:
                        mask[-1] = 1.0
                        action_mappings[agent] = [(-1, -1)]
                
                mask[-1] = 1.0  # Always allow NOOP
                action_masks[agent] = mask

        return action_masks, action_mappings
    
    def _get_actions(self, observations, infos) -> Tuple[Dict, Dict]:
        """Get actions for all agents using DICG policy.
        
        Uses action probabilities directly from the actor (already properly masked).
        The actor applies masked_fill(..., -inf) before softmax, so invalid actions
        have 0 probability. No reprocessing needed.
        """
        actions_dict = {}
        action_stats = {'noop': 0, 'fight': 0, 'total': 0}
        
        # Use domain-appropriate pad_observations
        if self.domain == "rideshare":
            padded_obs = rideshare_pad_observations(observations, env_id=0)
        else:
            padded_obs = pad_observations(observations, env_id=0)
        action_masks, action_mappings = self._create_action_masks(observations, infos, env_id=0)
        
        obs_list = [padded_obs[agent].float() for agent in self.agents]
        mask_list = [action_masks[agent].float() for agent in self.agents]
        obs_batch = torch.stack(obs_list).unsqueeze(0)
        mask_batch = torch.stack(mask_list).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, _ = self.models['actor'](obs_batch, mask_batch)
        
        for i, agent in enumerate(self.agents):
            # Use probabilities directly - already properly masked by the actor
            probs = action_probs[0, i]
            
            # Sample action from the categorical distribution (same as training)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            
            # Map to environment action
            # NOOP is at the last index (mask_batch.shape[-1] - 1)
            if action_idx.item() == (mask_batch.shape[-1] - 1):
                env_action = (-1, -1)
                action_stats['noop'] += 1
            else:
                avail = action_mappings[agent]
                if action_idx.item() < len(avail):
                    env_action = avail[action_idx.item()]
                    action_stats['fight'] += 1
                else:
                    # Action index beyond available actions = NOOP
                    env_action = (-1, -1)
                    action_stats['noop'] += 1
            
            action_stats['total'] += 1
            actions_dict[agent] = torch.tensor([env_action], device=self.device)
        
        return actions_dict, action_stats
    
    def _get_model_family(self) -> str:
        """Return the model family name."""
        return "DICG"
    
    def _get_variation1(self) -> str:
        """Return the first variation identifier."""
        return "DICG"
    
    def _get_variation2(self) -> str:
        """Return the second variation identifier."""
        return "default"