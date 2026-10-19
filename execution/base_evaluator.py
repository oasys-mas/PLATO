"""
Base evaluator class containing common functionality for all algorithm evaluators.
Supports both wildfire and rideshare domains.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from copy import deepcopy

from utils.custom_configuration import target_for_zero_shot, zero_shot_setting_detail, zero_shot_setting_type

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from free_range_zoo.envs import wildfire_v0
from free_range_zoo.envs import rideshare_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from experiment_logger_postgres import ExperimentDB
from utils.helper import count_zero_suppressant_agents, get_execution_seed, load_configuration, load_configuration_pkl, set_global_seed
from utils.experiment_manager import load_config
from utils.graph_utils_ptr import get_episode_fire_counts, print_wildfire_grid
from utils.rideshare_capacity_filter import apply_capacity_filter, RideshareMetricTracker


class BaseEvaluator(ABC):
    """
    Abstract base class for all algorithm evaluators.
    Contains common functionality like environment initialization, database logging, etc.
    """
    
    def __init__(self, experiment, shared_db=None):
        """
        Initialize the base evaluator.
        
        Args:
            experiment: Experiment object containing metadata and checkpoint path
        """
        self.experiment = experiment
        self.shared_db = shared_db if shared_db is not None else ExperimentDB()  
        self.eval_config = load_config("Eval")
        self.general_config = load_config("General_Config")
        
        # Determine domain from experiment (auto-detected from checkpoint name)
        self.domain = getattr(experiment, 'domain', 'wildfire')
        
        # Set device
        if self.general_config.DEVICE not in (None, ''):
            self.device = torch.device(self.general_config.DEVICE)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize environment
        self.env = self._init_environment()
        
        # Get agents
        observations, _ = self.env.reset()
        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        
        # Domain-specific setup
        if self.domain == "rideshare":
            self.metric_tracker = RideshareMetricTracker(self.env)
        
        # Load models (algorithm-specific)
        self.models = self._load_models()
        
        # Debug: Print configuration
        print(f"Initialized {self.__class__.__name__} (domain={self.domain})")
        print(f"   Device: {self.device}")
        print(f"   Agents: {self.n_agents}")
        print(f"   Checkpoint: {experiment.checkpoint_path}")
        if self.domain == "wildfire" and hasattr(self.env.unwrapped, 'agent_config'):
            print(f"   Suppressant decrease probability: {self.env.unwrapped.agent_config.suppressant_decrease_probability}")
            print(f"   Whole env:")
            print(self.env.unwrapped)
    def _init_environment(self):
        """Initialize the environment based on domain."""
        if self.domain == "rideshare":
            return self._init_rideshare_env()
        return self._init_wildfire_env()
    
    def _init_rideshare_env(self):
        """Initialize the rideshare environment."""
        # Load configuration from pkl (rideshare configs are stored as pkl)
        setting_type = self.experiment.setting_type
        configuration = load_configuration_pkl(setting_type)
        
        env = rideshare_v0.parallel_env(
            configuration=configuration,
            parallel_envs=1,
            max_steps=self.general_config.MAX_STEPS,
            device=self.device,
            single_seeding=False
        )
        env.reset()
        apply_capacity_filter(env)
        return action_mapping_wrapper_v0(env)
    
    def _init_wildfire_env(self):
        """Initialize the wildfire environment with common settings."""
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
        
  
        env = wildfire_v0.parallel_env(
            configuration=configuration,
            parallel_envs=1,  # Single environment for evaluation
            max_steps=self.general_config.MAX_STEPS,
            device=self.device,
            single_seeding=False,  
            show_bad_actions=False,
            observe_other_power=True,
            observe_other_suppressant=True,
            override_initialization_check=True
        )
        
        return action_mapping_wrapper_v0(env)
    
    @abstractmethod
    def _load_models(self) -> Dict:
        """
        Load algorithm-specific models from checkpoint.
        Must be implemented by each algorithm evaluator.
        
        Returns:
            Dictionary containing loaded models
        """
        pass
    
    def _get_rideshare_task_stages(self, agent, env_id=0):
        """Get the current stage of each task for a rideshare agent.
        
        Returns:
            task_stages: dict mapping task_idx -> stage (0=accept, 1=pickup, 2=dropoff)
            locked_idx: index of the first committed task (stage 1 or 2), or None
        """
        task_stages = {}
        locked_idx = None
        
        action_space = self.env.action_space(agent)
        oneof = action_space.spaces[env_id]  # Space_OneOf for this env
        
        for t_idx, discrete_space in enumerate(oneof.spaces):
            stage = discrete_space.start
            if stage == -1:  # NOOP sentinel (always last)
                break
            task_stages[t_idx] = stage
            if stage in (1, 2) and locked_idx is None:
                locked_idx = t_idx
        
        return task_stages, locked_idx
    
    @abstractmethod
    def _get_actions(self, observations, infos) -> Tuple[Dict, Dict]:
        """
        Get actions for all agents given observations.
        Must be implemented by each algorithm evaluator.
        
        Args:
            observations: Environment observations
            infos: Environment infos
            
        Returns:
            Tuple of (actions_dict, action_stats)
        """
        pass

    def _decode_agent_action(self, agent: str, action_value) -> Tuple:
        """Decode an action value into (task_idx, action_type) or (-1, -1) for NOOP.

        Default implementation handles the standard torch.tensor([(task_idx, type)])
        format used by PLATO, DGN, and DICG evaluators.
        MOHITO overrides this with its own wrapper-based decoding.

        Returns:
            (task_idx, action_type) where task_idx == -1 means NOOP
        """
        try:
            if torch.is_tensor(action_value):
                flat = action_value.view(-1)
                task_idx = int(flat[0].item())
                action_type = int(flat[1].item()) if flat.numel() > 1 else 0
                return task_idx, action_type
        except Exception:
            pass
        return -1, -1

    def _collect_wildfire_step_snapshot(self, step: int, actions_dict: Dict) -> Dict:
        """Capture state + agent decisions BEFORE env.step() for one timestep.

        Returns a step_record dict with fires[] and agents[] lists (rewards are
        added after env.step() by the caller).
        """
        state = self.env.state()
        fires_tensor = state.fires[0]       # [H, W]
        intensity_tensor = state.intensity[0]  # [H, W]
        fuel_tensor = state.fuel[0]         # [H, W]
        # agents: [n_agents, 2] — no batch dim
        agents_pos = state.agents
        suppressants = state.suppressants[0]   # [n_agents, 1] or [n_agents]
        capacities = state.capacity[0]         # same
        equipments = state.equipment[0]        # same

        # ── Fire cells ──────────────────────────────────────────────────────
        nonzero_coords = (fires_tensor != 0).nonzero(as_tuple=False)  # [N, 2]
        fires_list = []
        active_fire_count = 0
        total_intensity_sum = 0.0
        for coord in nonzero_coords:
            r, c = int(coord[0].item()), int(coord[1].item())
            fire_val = int(fires_tensor[r, c].item())
            intensity = float(intensity_tensor[r, c].item())
            fuel = float(fuel_tensor[r, c].item())
            is_lit = 1 if fire_val > 0 else 0
            if is_lit:
                active_fire_count += 1
                total_intensity_sum += intensity
            fires_list.append({
                'row': r,
                'col': c,
                'intensity': intensity,
                'fuel': fuel,
                'is_lit': is_lit,
                'fire_type': abs(fire_val),
            })

        # ── Agent cells ─────────────────────────────────────────────────────
        env_id = 0
        agents_list = []
        for i, agent in enumerate(self.agents):
            pos = agents_pos[i]  # [2]
            agent_row = int(pos[0].item())
            agent_col = int(pos[1].item())
            sup = suppressants[i]
            sup_val = float(sup.item() if torch.is_tensor(sup) else sup)
            cap = capacities[i]
            cap_val = float(cap.item() if torch.is_tensor(cap) else cap)
            eq = equipments[i]
            eq_val = float(eq.item() if torch.is_tensor(eq) else eq)

            # Decode action
            action_value = actions_dict.get(agent)
            task_idx, action_type_int = self._decode_agent_action(agent, action_value)

            target_row = target_col = target_intensity = target_fuel = None
            action_type_str = 'unknown'

            if task_idx == -1:
                action_type_str = 'noop'
            else:
                action_type_str = 'fight'
                try:
                    task_indices = self.env.unwrapped.agent_action_mapping[agent][env_id]
                    task_id = int(task_indices[task_idx].item()
                                  if torch.is_tensor(task_indices[task_idx])
                                  else task_indices[task_idx])
                    feat = self.env.unwrapped.task_store[env_id][task_id]
                    target_row = float(feat[0].item())
                    target_col = float(feat[1].item())
                    target_intensity = float(feat[2].item())
                    target_fuel = float(feat[3].item())
                except Exception:
                    action_type_str = 'unknown'

            agents_list.append({
                'agent_id': str(agent),
                'agent_row': agent_row,
                'agent_col': agent_col,
                'suppressant': sup_val,
                'capacity': cap_val,
                'equipment': eq_val,
                'action_type': action_type_str,
                'target_fire_row': target_row,
                'target_fire_col': target_col,
                'target_fire_intensity': target_intensity,
                'target_fire_fuel': target_fuel,
                'agent_reward': None,  # filled in after env.step()
            })

        return {
            'step': step,
            'step_reward': None,   # filled in after env.step()
            'active_fire_count': active_fire_count,
            'total_intensity_sum': total_intensity_sum,
            'new_burnouts': 0,     # filled in after env.step()
            'new_putouts': 0,      # filled in after env.step()
            'fires': fires_list,
            'agents': agents_list,
        }
    
    def run_episode(self, seed: Optional[int] = None) -> Dict:
        """
        Run a single episode and return comprehensive metrics as a dict.
        This is the common episode loop used by all algorithms.
        Supports both wildfire and rideshare domains.
        
        Args:
            seed: Random seed for episode
            
        Returns:
            Dict containing episode metrics (domain-specific keys included)
        """
        if seed is not None:
            set_global_seed(seed)
        
        observations, infos = self.env.reset(seed=seed)
        
        # Initialize episode metrics
        episode_reward = 0.0
        action_counts = {'noop': 0, 'fight': 0, 'total': 0}
        steps = 0
        
        # Domain-specific initialization
        if self.domain == "wildfire":
            initial_state = deepcopy(self.env.state())
            episode_burnouts = 0
            episode_putouts = 0
            episode_zero_suppressant_agent_count = 0
        elif self.domain == "rideshare":
            self.metric_tracker.reset()
            episode_pickups = 0
            episode_dropoffs = 0
        # Step-detail collection (wildfire only, gated by config)
        log_step_detail = (self.domain == "wildfire" and
                           getattr(self.eval_config, 'log_step_detail', False))
        episode_step_data = [] if log_step_detail else None

        if self.general_config.DEBUG:  
            print(f"\n--- ************Starting episode (seed={seed}) *************---")
        # Run episode
        while (not torch.all(self.env.finished).item()):
            # Get actions from algorithm-specific implementation
            actions_dict, step_action_stats = self._get_actions(observations, infos)

            # Wildfire: snapshot state+actions BEFORE step
            if log_step_detail:
                step_snapshot = self._collect_wildfire_step_snapshot(steps, actions_dict)
            
            # Rideshare: snapshot state before step for metric tracking
            if self.domain == "rideshare":
                self.metric_tracker.pre_step()
            if self.domain == "wildfire" and self.general_config.DEBUG:  
                print_wildfire_grid(self.env.state())
                print(actions_dict)  # actions takeln by agents
            observations, rewards, terms, truncs, infos = self.env.step(actions_dict)
            if self.general_config.DEBUG:  
                print(f"Step {steps}: rewards={rewards}")
            # Update reward
            step_reward = np.mean([r.mean().item() if torch.is_tensor(r) else r 
                                 for r in rewards.values()])
            episode_reward += step_reward
            
            # Update action counts
            for key in action_counts:
                if key in step_action_stats:
                    action_counts[key] += step_action_stats[key]
            # Also accumulate any rideshare-specific action keys (accept, pickup, dropoff)
            for key in step_action_stats:
                if key not in action_counts:
                    action_counts[key] = step_action_stats[key]
                elif key not in ('noop', 'fight', 'total'):
                    action_counts[key] += step_action_stats[key]
            
            # Domain-specific step metrics
            if self.domain == "wildfire":
                episode_zero_suppressant_agent_count += count_zero_suppressant_agents(self.env.state())
                # Update burnouts and putouts
                step_burnouts = 0
                step_putouts = 0
                if hasattr(infos, '__getitem__'):
                    step_burnouts = int(infos.get("burnouts", 0))
                    step_putouts = int(infos.get("putouts", 0))
                    episode_burnouts += step_burnouts
                    episode_putouts += step_putouts
            elif self.domain == "rideshare":
                step_pickups, step_dropoffs, _ = self.metric_tracker.post_step()
                episode_pickups += step_pickups
                episode_dropoffs += step_dropoffs

            # Finalise wildfire step snapshot with post-step data
            if log_step_detail:
                step_snapshot['step_reward'] = float(step_reward)
                step_snapshot['new_burnouts'] = step_burnouts if self.domain == 'wildfire' else 0
                step_snapshot['new_putouts'] = step_putouts if self.domain == 'wildfire' else 0
                # Attach per-agent rewards
                for agent_rec in step_snapshot['agents']:
                    agent_name = agent_rec['agent_id']
                    r = rewards.get(agent_name, 0)
                    agent_rec['agent_reward'] = float(
                        r.mean().item() if torch.is_tensor(r) else r
                    )
                episode_step_data.append(step_snapshot)
            
            steps += 1
        
        # Build metrics dict (common fields)
        metrics = {
            'reward': episode_reward,
            'action_counts': action_counts,
            'steps': steps,
        }
        
        # Add domain-specific metrics
        if self.domain == "wildfire":
            final_state = self.env.state()
            episode_total_fires_count, new_fire_ignited_count = get_episode_fire_counts(
                initial_state, final_state, episode_burnouts, episode_putouts, 1
            )
            fight_count = action_counts['fight'] if action_counts['fight'] > 0 else 1
            metrics.update({
                'burnouts': episode_burnouts,
                'putouts': episode_putouts,
                'total_fires': episode_total_fires_count,
                'new_fires': new_fire_ignited_count,
                'zero_suppressant_count': episode_zero_suppressant_agent_count,
                'reward_per_fight': episode_reward / fight_count,
                'putouts_per_fight': episode_putouts / fight_count,
            })
        elif self.domain == "rideshare":
            metrics.update({
                'pickups': episode_pickups,
                'dropoffs': episode_dropoffs,
            })

        if log_step_detail:
            metrics['step_data'] = episode_step_data
        
        return metrics

    def evaluate(self) -> bool:
        """
        Run the complete evaluation process including database logging.
        Supports both wildfire and rideshare domains.
        
        Returns:
            True if evaluation completed successfully
        """
        print(f"Starting evaluation ({self.domain}): {self.eval_config.num_runs} runs, "
              f"{self.eval_config.episodes_per_run} episodes per run")
        
        # Setup experiment metadata
        experiment_meta = self._create_experiment_metadata()
        
        # Common accumulators
        eval_runs = []
        wildfire_step_data_list = [] if (self.domain == "wildfire" and
                                         getattr(self.eval_config, 'log_step_detail', False)) else None
        all_rewards = []
        all_noops = []
        all_total_actions = []
        all_steps = []
        
        # Domain-specific accumulators
        if self.domain == "wildfire":
            all_burnouts = []
            all_putouts = []
            all_total_fires = []
            all_new_fires = []
            all_reward_per_fight = []
            all_putouts_per_fight = []
            all_zero_suppressant_counts = []
        elif self.domain == "rideshare":
            all_pickups = []
            all_dropoffs = []
            all_accepts = []
        
        run_index = 0
        
        for run in range(self.eval_config.num_runs):
            print(f"Run {run + 1}/{self.eval_config.num_runs}")
            
            for episode in range(self.eval_config.episodes_per_run):
                seed = get_execution_seed(episode, self.general_config)
                
                # Run episode (returns dict)
                metrics = self.run_episode(seed)
                
                # Common metrics
                all_rewards.append(metrics['reward'])
                all_noops.append(metrics['action_counts']['noop'])
                all_total_actions.append(metrics['action_counts']['total'])
                all_steps.append(metrics['steps'])
                
                # Build eval run record (common fields)
                run_record = {
                    'run_index': run_index,
                    'seed': seed,
                    'reward': metrics['reward'],
                    'episode_length': metrics['steps'],
                    'episode_noop_count': metrics['action_counts']['noop'],
                }
                
                # Domain-specific accumulation and run record
                if self.domain == "wildfire":
                    all_burnouts.append(metrics['burnouts'])
                    all_putouts.append(metrics['putouts'])
                    all_total_fires.append(metrics['total_fires'])
                    all_new_fires.append(metrics['new_fires'])
                    all_reward_per_fight.append(metrics['reward_per_fight'])
                    all_putouts_per_fight.append(metrics['putouts_per_fight'])
                    all_zero_suppressant_counts.append(metrics['zero_suppressant_count'])
                    run_record.update({
                        'burnouts': metrics['burnouts'],
                        'putouts': metrics['putouts'],
                        'episode_total_fires_count': metrics['total_fires'],
                        'new_fire_ignited_count': metrics['new_fires'],
                        'episode_fight_count': metrics['action_counts'].get('fight', 0),
                        'reward_per_fight': metrics['reward_per_fight'],
                        'putouts_per_fight': metrics['putouts_per_fight'],
                        'episode_zero_suppressant_count': metrics['zero_suppressant_count'],
                    })
                elif self.domain == "rideshare":
                    all_pickups.append(metrics['pickups'])
                    all_dropoffs.append(metrics['dropoffs'])
                    all_accepts.append(metrics['action_counts'].get('accept', 0))
                    run_record.update({
                        'pickups': metrics['pickups'],
                        'dropoffs': metrics['dropoffs'],
                        'episode_accept_count': metrics['action_counts'].get('accept', 0),
                        'episode_pick_count': metrics['action_counts'].get('pickup', 0),
                        'episode_drop_count': metrics['action_counts'].get('dropoff', 0),
                        'episode_action_count': metrics['action_counts'].get('fight', 0)
                                               + metrics['action_counts'].get('accept', 0)
                                               + metrics['action_counts'].get('pickup', 0)
                                               + metrics['action_counts'].get('dropoff', 0),
                    })
                
                eval_runs.append(run_record)
                if wildfire_step_data_list is not None:
                    wildfire_step_data_list.append(metrics.get('step_data'))
                run_index += 1
        
        # Calculate aggregated results
        eval_results = {
            'avg_reward': np.mean(all_rewards),
            'avg_length': np.mean(all_steps),
            'avg_noop': np.mean(all_noops),
            'avg_total_actions': np.mean(all_total_actions),
        }
        
        if self.domain == "wildfire":
            eval_results.update({
                'avg_burnouts': np.mean(all_burnouts),
                'avg_putouts': np.mean(all_putouts),
                'avg_reward_per_fight': np.mean(all_reward_per_fight),
                'avg_putouts_per_fight': np.mean(all_putouts_per_fight),
                'avg_zero_suppressant_count': np.mean(all_zero_suppressant_counts),
            })
        elif self.domain == "rideshare":
            eval_results.update({
                'avg_pickups': np.mean(all_pickups),
                'avg_dropoffs': np.mean(all_dropoffs),
                'avg_accepts': np.mean(all_accepts),
            })
        
        # Log to database if enabled
        if getattr(self.eval_config, 'log_to_db', True):
            print("\n=== LOGGING TO DATABASE ===")
            self.shared_db.log_execution(
                experiment_meta, eval_results, eval_runs,
                wildfire_step_data_list=wildfire_step_data_list,
            )
        else:
            print("\n=== DATABASE LOGGING DISABLED (log_to_db=False) ===")
        
        # Print summary
        if self.domain == "wildfire":
            self._print_summary(all_rewards, all_burnouts, all_putouts, all_steps, 
                              all_total_fires, all_new_fires,
                              all_reward_per_fight, all_putouts_per_fight, all_zero_suppressant_counts)
        elif self.domain == "rideshare":
            self._print_rideshare_summary(all_rewards, all_pickups, all_dropoffs, all_steps)
        
        return True
    
    def _create_experiment_metadata(self) -> Dict:
        """Create experiment metadata for database logging."""
        return {
            'run_type': "execution",
            'model_family': self._get_model_family(),
            'model_type': "smart",
            'name': self.experiment.name,
            'variation1': self._get_variation1(),
            'variation2': self._get_variation2(),
            'grid_size': self.experiment.grid_size,
            'setting_type': self.experiment.setting_type,
            'seed': getattr(self.experiment, 'seed', None),
            'parent_experiment_id': None,
            'checkpoint_path': self.experiment.checkpoint_path,
            'zero_shot': self.experiment.zero_shot or None,
            'zero_shot_source_grids_size': self.experiment.grid_size if self.experiment.zero_shot else None,
            'zero_shot_target_grids_size': getattr(self.experiment, 'zero_shot_target_grids_size', None) if self.experiment.zero_shot else None,
            'zero_shot_target_grids_size_detail': getattr(self.experiment, 'zero_shot_target_grids_size_detail', None) or None,
            'zero_shot_mode': (getattr(self.experiment, 'zero_shot_mode', None) if self.experiment.zero_shot else None) or None,
            'domain': self.domain,
            'notes': "neurips 2026",
        }
    
    @abstractmethod
    def _get_model_family(self) -> str:
        """Return the model family name (e.g., 'PLATO', 'DICG', etc.)."""
        pass
    
    def _get_variation1(self) -> str:
        """Return the first variation identifier."""
        return getattr(self.experiment, 'pointer_policy', 'default')
    
    def _get_variation2(self) -> str:
        """Return the second variation identifier."""
        return getattr(self.experiment, 'attention_type', 'default')
    
    def _print_summary(self, rewards, burnouts, putouts, steps, total_fires, 
                      new_fires, reward_per_fight=None, putouts_per_fight=None, 
                      zero_suppressant_counts=None):
        """Print wildfire evaluation summary."""
        print(f"\n=== EVALUATION SUMMARY (wildfire) ===")
        print(f"Average reward: {np.mean(rewards):.3f}")
        print(f"Average burnouts: {np.mean(burnouts):.2f}")
        print(f"Average putouts: {np.mean(putouts):.2f}")
        print(f"Average episode length: {np.mean(steps):.1f}")
        print(f"Average total fires: {np.mean(total_fires):.2f}")
        print(f"Average new fires: {np.mean(new_fires):.2f}")

        if reward_per_fight is not None:
            print(f"Average reward per fight: {np.mean(reward_per_fight):.3f}")
        if putouts_per_fight is not None:
            print(f"Average putouts per fight: {np.mean(putouts_per_fight):.3f}")
        if zero_suppressant_counts is not None:
            print(f"Average zero suppressant agent count: {np.mean(zero_suppressant_counts):.2f}")
    
    def _print_rideshare_summary(self, rewards, pickups, dropoffs, steps):
        """Print rideshare evaluation summary."""
        print(f"\n=== EVALUATION SUMMARY (rideshare) ===")
        print(f"Average reward: {np.mean(rewards):.3f}")
        print(f"Average pickups: {np.mean(pickups):.2f}")
        print(f"Average dropoffs: {np.mean(dropoffs):.2f}")
        print(f"Average episode length: {np.mean(steps):.1f}")