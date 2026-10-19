"""
DGN-specific evaluator implementation.
"""

import torch
import numpy as np
from typing import Dict, Tuple

from base_evaluator import BaseEvaluator
from baselines.DGN.model import DGN
from baselines.DGN.config import hidden_dim
from utils.experiment_manager import load_config
from utils.graph_utils_ptr import convert_actions_to_tensor, get_available_actions


class DGNEvaluator(BaseEvaluator):
    """DGN algorithm evaluator - supports wildfire and rideshare."""
    def __init__(self, experiment, shared_db=None):
        self.dgn_config = load_config("DGN")
        super().__init__(experiment, shared_db=shared_db)

    
    def _load_models(self) -> Dict:
        """Load DGN model from checkpoint."""
        print(f"Loading DGN checkpoint from: {self.experiment.checkpoint_path}")
        checkpoint = torch.load(self.experiment.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Infer obs_space from checkpoint weights to handle cross-execution scenarios
        # The encoder.fc.weight has shape [hidden_dim, obs_space]
        if 'model' in checkpoint:
            encoder_weight = checkpoint['model'].get('encoder.fc.weight')
            if encoder_weight is not None:
                obs_space = encoder_weight.shape[1]  # Input dimension from checkpoint
                # Infer n_agents_ckpt from obs_space (obs_space = OBS_DIM + n_agents)
                self.n_agents_ckpt = obs_space - self.dgn_config.OBS_DIM
                print(f"Inferred from checkpoint: obs_space={obs_space}, n_agents_ckpt={self.n_agents_ckpt}")
            else:
                # Fallback to current environment
                obs_space = self.dgn_config.OBS_DIM + self.n_agents
                self.n_agents_ckpt = self.n_agents
        else:
            obs_space = self.dgn_config.OBS_DIM + self.n_agents
            self.n_agents_ckpt = self.n_agents
        
        n_actions = self.general_config.MAX_ACTIONS_PER_AGENT
        
        # Create model with architecture matching the checkpoint (use n_agents_ckpt)
        model = DGN(self.n_agents_ckpt, obs_space, hidden_dim, n_actions).to(self.device)
        
        # Load state dictionary
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            raise KeyError(f"Expected 'model' key in checkpoint, "
                          f"but got keys: {list(checkpoint.keys())}")
        
        # Set to evaluation mode
        model.eval()
        
        return {'model': model}
    
    def _pad_observations(self, observations, max_obs_dim=None, env_id=0):
        """Build fixed-size obs vectors - works for both wildfire and rideshare."""
        if max_obs_dim is None:
            max_obs_dim = self.dgn_config.OBS_DIM
        
        # Domain-specific feature dimensions
        if self.domain == "rideshare":
            TASK_FEAT_DIM = 8   # y, x, y_dest, x_dest, accepted_by, riding_by, fare, entered_step
            AGENT_FEAT_DIM = 4  # y, x, num_accepted, num_riding
        else:
            TASK_FEAT_DIM = self.general_config.FEAT_DIM
            AGENT_FEAT_DIM = self.general_config.FEAT_DIM
            
        padded_obs = {}
        for agent, obs in observations.items():
            agent_obs_dict = obs[env_id]

            if 'tasks' in agent_obs_dict and agent_obs_dict['tasks'].numel() > 0:
                tasks = agent_obs_dict['tasks'][env_id].float()
                if tasks.ndim == 1:
                    tasks = tasks.unsqueeze(0)
            else:
                tasks = torch.empty(0, TASK_FEAT_DIM, device=self.device)
                
            self_f = (agent_obs_dict['self'][env_id].float()
                      if 'self' in agent_obs_dict
                      else torch.empty(0, device=self.device))
                      
            if 'others' in agent_obs_dict and agent_obs_dict['others'].numel() > 0:
                others = agent_obs_dict['others'][env_id].float()
                if others.ndim == 1:
                    others = others.unsqueeze(0)
            else:
                others = torch.empty(0, AGENT_FEAT_DIM, device=self.device)

            # Use dynamic feature dims from data shape, with domain-specific fallback
            task_fdim = tasks.shape[1] if tasks.ndim == 2 and tasks.numel() > 0 else TASK_FEAT_DIM
            other_fdim = others.shape[1] if others.ndim == 2 and others.numel() > 0 else AGENT_FEAT_DIM

            num_tasks = min(tasks.shape[0], self.general_config.MAX_TASKS) if tasks.numel() > 0 else 0
            num_others = min(others.shape[0], self.general_config.MAX_TASKS) if others.numel() > 0 else 0

            padded_tasks = torch.full((self.general_config.MAX_TASKS, task_fdim), -1.0, device=self.device)
            padded_others = torch.full((self.general_config.MAX_TASKS, other_fdim), -1.0, device=self.device)
            if num_tasks > 0: 
                padded_tasks[:num_tasks] = tasks[:num_tasks]
            if num_others > 0: 
                padded_others[:num_others] = others[:num_others]

            agent_vec = torch.cat([padded_tasks.flatten(), self_f.flatten(), padded_others.flatten()])
            if agent_vec.numel() < max_obs_dim:
                out = torch.zeros(max_obs_dim, device=self.device)
                out[:agent_vec.numel()] = agent_vec
            else:
                out = agent_vec[:max_obs_dim]
            padded_obs[agent] = out
        return padded_obs
    
    def _create_action_masks(self, observations, infos, env_id=0):
        """Create action masks for the environment.
        
        Wildfire: Action format (action_index, 0). NOOP: (-1, -1)
        Rideshare: Action format (task_idx, action_type) where action_type comes
                   from the task's current stage. Uses lock mechanism - if agent
                   has a committed task (pickup/dropoff stage), it must complete it.
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
                        mask[-1] = 1.0  # NOOP
                        action_mappings[agent] = [(-1, -1)]
                
                mask[-1] = 1.0  # Always allow NOOP
                action_masks[agent] = mask

        return action_masks, action_mappings
    
    def _get_obs(self, observations, env_id=0):
        """Convert observations to DGN format - works for both wildfire and rideshare.

        For cross-execution: uses n_agents_ckpt for agent ID dimension to match model.
        When env has fewer agents than checkpoint, pads with zero rows so the model
        always receives tensors of the expected size.
        """
        padded_obs = self._pad_observations(observations, env_id=env_id)
        obs_list = []

        n_agents_for_id = getattr(self, 'n_agents_ckpt', self.n_agents)

        for i, agent in enumerate(self.agents[:n_agents_for_id]):
            agent_id = np.zeros(n_agents_for_id)
            agent_id[i] = 1
            agent_obs = torch.cat([padded_obs[agent], torch.tensor(agent_id, device=self.device, dtype=torch.float32)])
            obs_list.append(agent_obs.cpu().numpy())

        # Pad with zero rows for missing agents (env has fewer agents than checkpoint)
        if len(obs_list) < n_agents_for_id:
            dummy = np.zeros_like(obs_list[0])
            while len(obs_list) < n_agents_for_id:
                obs_list.append(dummy)

        return np.array(obs_list)
    
    def _create_adjacency_matrix(self, comm_flag=1.0):
        """Create adjacency matrix for communication - same as training.
        
        For cross-execution: uses n_agents_ckpt to match model dimensions.
        """
        # Use n_agents_ckpt for adjacency matrix dimensions
        n_agents_for_adj = getattr(self, 'n_agents_ckpt', self.n_agents)
        adj = np.ones((n_agents_for_adj, n_agents_for_adj)) * comm_flag + np.eye(n_agents_for_adj)
        return adj
    
    def _get_actions(self, observations, infos) -> Tuple[Dict, Dict]:
        """Get actions for all agents using DGN policy."""
        actions_dict = {}
        action_stats = {'noop': 0, 'fight': 0, 'total': 0}
        
        # Use n_agents_ckpt to limit agents processed by the model
        n_agents_ckpt = getattr(self, 'n_agents_ckpt', self.n_agents)
        agents_to_process = self.agents[:n_agents_ckpt]
        
        # Get observations and masks
        obs = self._get_obs(observations, env_id=0)
        adj = self._create_adjacency_matrix(comm_flag=float(self.dgn_config.comm_flag))
        action_masks, action_mappings = self._create_action_masks(observations, infos, env_id=0)
        
        mask_list = [action_masks[agent].cpu().numpy() for agent in agents_to_process]
        # Pad mask rows for missing agents (env has fewer agents than checkpoint)
        if len(mask_list) < n_agents_ckpt:
            dummy_mask = np.zeros_like(mask_list[0])
            while len(mask_list) < n_agents_ckpt:
                mask_list.append(dummy_mask)
        mask = np.array(mask_list)
        
        # Get Q-values from DGN model
        with torch.no_grad():
            q = self.models['model'](
                torch.Tensor(np.array([obs])).to(self.device), 
                torch.Tensor(np.array([adj])).to(self.device)
            )[0]
        
        # Select actions via sampling for agents the model knows about
        for i, agent in enumerate(agents_to_process):
            # Sampling action selection (softmax over masked Q-values)
            masked_q = q[i].cpu().detach().clone()
            masked_q[torch.tensor(mask[i]) == 0] = float('-inf')
            probs = torch.softmax(masked_q, dim=-1)
            action_idx = torch.multinomial(probs, 1).item()
            
            # Map to environment action
            if action_idx >= len(action_mappings[agent]):
                env_action = (-1, -1)  # NOOP
                action_stats['noop'] += 1
            else:
                env_action = action_mappings[agent][action_idx]
                action_stats['fight'] += 1
            
            action_stats['total'] += 1
            actions_dict[agent] = torch.tensor([env_action], device=self.device)
        
        # For extra agents (in cross-execution), default to NOOP
        for agent in self.agents[n_agents_ckpt:]:
            actions_dict[agent] = torch.tensor([(-1, -1)], device=self.device)
            action_stats['noop'] += 1
            action_stats['total'] += 1
        
        return actions_dict, action_stats
    

    def _get_model_family(self) -> str:
        """Return the model family name."""
        return "DGN"
    
    def _get_variation1(self) -> str:
        """Return the first variation identifier."""
        return "DGN"
    
    def _get_variation2(self) -> str:
        """Return the second variation identifier."""
        return "default"