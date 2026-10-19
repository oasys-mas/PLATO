"""
MOHITO-specific evaluator implementation - supports wildfire and rideshare.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Tuple
from gymnasium.spaces import MultiDiscrete

from utils.custom_configuration import target_for_zero_shot, zero_shot_setting_detail, zero_shot_setting_type

# Add MOHITO to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
mohito_base = os.path.join(parent_dir, 'baselines/mohito/wildfire/mohito')
if mohito_base not in sys.path:
    sys.path.insert(0, mohito_base)

from base_evaluator import BaseEvaluator
from mohito.wrappers.action_task import action_mapping_wrapper_v0 as mohito_action_wrapper # type: ignore
from mohito.wrappers.mohito_wrapper import mohito_hypergraph_wrapper_v0 # type: ignore
from mohito.mohito import ActorReturn, CriticReturn, Mohito # type: ignore
from utils.experiment_manager import load_config
from utils.graph_utils_ptr import action_info, convert_actions_to_tensor
from utils.helper import load_configuration, load_configuration_pkl
from utils.rideshare_capacity_filter import apply_capacity_filter, rideshare_amap_fix_wrapper_v0


class MOHITOEvaluator(BaseEvaluator):
    """MOHITO algorithm evaluator - supports wildfire and rideshare."""
    
    def __init__(self, experiment, shared_db=None):
        self.mohito_config = load_config("MOHITO")
        super().__init__(experiment, shared_db=shared_db)
    
    def _init_environment(self):
        """Initialize environment with MOHITO-specific wrappers.
        Supports both wildfire and rideshare domains.
        """
        if self.domain == "rideshare":
            return self._init_rideshare_env_mohito()
        return self._init_wildfire_env_mohito()
    
    def _init_rideshare_env_mohito(self):
        """Initialize rideshare environment with MOHITO-specific wrappers."""
        configuration = load_configuration_pkl(self.experiment.setting_type)
        
        from free_range_zoo.envs import rideshare_v0
        
        env = rideshare_v0.parallel_env(
            max_steps=self.general_config.MAX_STEPS,
            parallel_envs=1,
            configuration=configuration,
            device=self.device,
            single_seeding=False
        )
        
        # Apply rideshare-specific wrappers (same order as training)
        env.reset()
        apply_capacity_filter(env)
        env = mohito_action_wrapper(env)
        env.reset()
        env = rideshare_amap_fix_wrapper_v0(env)
        env.reset()
        env = mohito_hypergraph_wrapper_v0(env)
        env.reset()
        
        return env
    
    def _init_wildfire_env_mohito(self):
        setting_type = zero_shot_setting_type(self.experiment)
        self.experiment.zero_shot_target_grids_size_detail = zero_shot_setting_detail(self.experiment)
        
        # Get zero-shot mode configuration
        zero_shot_mode = getattr(self.experiment, 'zero_shot_mode', None)
        source_setting_type = None
        target_grid_size = None
        
        if zero_shot_mode in ("fixteam", "fixteam_transition"):
            # For fixteam modes: use source setting for team config, target for grid size
            source_setting_type = self.experiment.setting_type  # Original training setting
            target_grid_size = self.experiment.zero_shot_target_grids_size
            print(f"   Zero-shot mode: {zero_shot_mode}")
            print(f"   Source setting: {source_setting_type}")
            print(f"   Target grid size: {target_grid_size}x{target_grid_size}")

        configuration = load_configuration(
            setting_type,
            zero_shot_mode=zero_shot_mode,
            source_setting_type=source_setting_type,
            target_grid_size=target_grid_size
        )
        
        from free_range_zoo.envs import wildfire_v0
        
        # Match training environment initialization (without extra params)
        env = wildfire_v0.parallel_env(
            max_steps=self.general_config.MAX_STEPS,
            parallel_envs=1,
            configuration=configuration,
            device=self.device
        )
        
        # Apply MOHITO-specific wrappers (same order as training)
        env.reset()
        env = mohito_action_wrapper(env)
        env.reset()
        env = mohito_hypergraph_wrapper_v0(env)
        env.reset()
        
        return env
    

    def _load_models(self) -> Dict:
        """Load MOHITO models from checkpoint."""
        print(f"Loading MOHITO checkpoint from: {self.experiment.checkpoint_path}")
        
        # Get MOHITO hyperparameters
        mohito_config_dict = (self.mohito_config if isinstance(self.mohito_config, dict) 
                             else vars(self.mohito_config))
        mohito_hyperparameters = {
            k: v for k, v in mohito_config_dict.items() 
            if k not in ['training', 'convergence', 'environment', 'validation']
        }
        
        # Load checkpoint
        checkpoint = torch.load(self.experiment.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Auto-detect node_feature_size from checkpoint weights
        # The first GATv2 layer weight 'actor.main.net.0.lin_l.weight' has shape [hidden_dim, node_feature_size]
        for agent_key in checkpoint:
            state_dict = checkpoint[agent_key]
            if isinstance(state_dict, dict) and 'actor.main.net.0.lin_l.weight' in state_dict:
                detected_dim = state_dict['actor.main.net.0.lin_l.weight'].shape[1]
                config_dim = mohito_hyperparameters.get('both', {}).get('node_feature_size', 5)
                if detected_dim != config_dim:
                    print(f"Auto-detected node_feature_size={detected_dim} from checkpoint (config had {config_dim})")
                    mohito_hyperparameters['both']['node_feature_size'] = detected_dim
                break
        
        # Create policies for each agent
        policies = {}
        for agent_name in self.agents:
            policy = Mohito(name=agent_name, **mohito_hyperparameters).to(self.device)
            
            # Try different checkpoint formats
            key1 = f"agent_{agent_name}"
            key2 = agent_name
            
            if key1 in checkpoint:
                print(f"Loading {agent_name} from key '{key1}'")
                policy.load_state_dict(checkpoint[key1])
            elif key2 in checkpoint:
                print(f"Loading {agent_name} from key '{key2}'")
                policy.load_state_dict(checkpoint[key2])
            else:
                # Handle missing agent policy during cross training
                if hasattr(self.experiment, 'zero_shot') and self.experiment.zero_shot:
                    # Find all available agent policies in checkpoint
                    available_agent_keys = []
                    for key in checkpoint.keys():
                        if key.startswith('agent_') and key != f'agent_{agent_name}':
                            available_agent_keys.append(key)
                        elif key in self.agents and key != agent_name:
                            available_agent_keys.append(key)
                    
                    if available_agent_keys:
                        # Randomly select an existing policy
                        selected_key = np.random.choice(available_agent_keys)
                        print(f"Cross training: Agent {agent_name} policy not found. "
                              f"Randomly selected policy from '{selected_key}'")
                        policy.load_state_dict(checkpoint[selected_key])
                    else:
                        # Try loading checkpoint directly if no agent-specific keys found
                        print(f"Warning: No specific key found for {agent_name}. "
                              f"Attempting to load checkpoint directly as state_dict")
                        try:
                            policy.load_state_dict(checkpoint)
                        except Exception as e:
                            raise ValueError(
                                f"Could not find checkpoint data for agent {agent_name}. "
                                f"Expected keys like '{key1}' or '{key2}' in checkpoint. "
                                f"Available keys: {list(checkpoint.keys())}"
                            ) from e
                else:
                    # Original behavior for non-cross training
                    print(f"Warning: No specific key found for {agent_name}. "
                          f"Attempting to load checkpoint directly as state_dict")
                    try:
                        policy.load_state_dict(checkpoint)
                    except Exception as e:
                        raise ValueError(
                            f"Could not find checkpoint data for agent {agent_name}. "
                            f"Expected keys like '{key1}' or '{key2}' in checkpoint. "
                            f"Available keys: {list(checkpoint.keys())}"
                        ) from e
            
            policy.eval()
            policies[agent_name] = policy
            print(f"Successfully loaded {agent_name}")
        
        return policies
    
    
    
    def _get_actions(self, observations, infos) -> Tuple[Dict, Dict]:
        """Get actions for all agents using MOHITO policies.
        
        """
        action_stats = {'noop': 0, 'fight': 0, 'total': 0}
        
        with torch.no_grad():
            # Call observe() first for all agents (same as training's evaluate)
            for agent_name in self.agents:
                obs = observations[agent_name]
                self.models[agent_name].observe(obs)
            
            # Then get actions using .act() with deterministic selection (no exploration)
            agent_actions = {
                agent_name: self.models[agent_name].act(
                    action_space=self.env.action_space(agent_name),
                    return_logits=False
                )
                for agent_name in self.agents
            }
            
            # Compute action stats
            action_unwrapped = {
                agent: self.env.modifiers[agent].modify_action(action) 
                for agent, action in agent_actions.items()
            }
            noop_counts, fight_counts = action_info(action_unwrapped)
            action_stats = {'noop': noop_counts, 'fight': fight_counts, 'total': noop_counts + fight_counts}
        
        return agent_actions, action_stats
    

    def _get_model_family(self) -> str:
        """Return the model family name."""
        return "MOHITO"
    
    def _get_variation1(self) -> str:
        """Return the first variation identifier."""
        return "MOHITO"
    
    def _get_variation2(self) -> str:
        """Return the second variation identifier."""
        return "default"

    def _decode_agent_action(self, agent: str, action_value):
        """Decode MOHITO action via its wrapper's modify_action, then extract task tuple.

        MOHITO's action objects must be unwrapped through the modifier that was applied
        during environment setup. The result is the same (task_idx, action_type) tuple
        format used by the base class, so fire-coordinate lookup works identically.
        """
        try:
            unwrapped = self.env.modifiers[agent].modify_action(action_value)
            # modify_action returns a tensor of shape [2] or a (task_idx, type) tuple
            import torch
            if torch.is_tensor(unwrapped):
                flat = unwrapped.view(-1)
                task_idx = int(flat[0].item())
                action_type = int(flat[1].item()) if flat.numel() > 1 else 0
                return task_idx, action_type
            if isinstance(unwrapped, (tuple, list)) and len(unwrapped) >= 2:
                return int(unwrapped[0]), int(unwrapped[1])
        except Exception:
            pass
        return -1, -1