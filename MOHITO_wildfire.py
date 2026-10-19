#MOHITO_wildfire
import argparse
import os
import sys
import random
import copy
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
import tabulate
import time

from free_range_zoo.utils.env import BatchedAECEnv
# from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from free_range_zoo.envs import wildfire_v0
current_dir = os.path.dirname(os.path.abspath(__file__))
mohito_base = os.path.join(current_dir, 'baselines/mohito/wildfire/mohito')  # this directory contains the top-level folder "mohito/"
if mohito_base not in sys.path:
    sys.path.insert(0, mohito_base)

from mohito.wrappers.action_task import action_mapping_wrapper_v0 # type: ignore 
from mohito.wrappers.mohito_wrapper import mohito_hypergraph_wrapper_v0 # type: ignore
from mohito.mohito import ActorReturn, CriticReturn, Mohito # type: ignore

from utils.helper import count_zero_suppressant_agents, get_training_seed, get_validation_seed, parse_args, set_global_seed, setup_tensorboard_logging, save_checkpoint, resolve_configuration
from utils.experiment_manager import load_config
from copy import deepcopy
from utils.graph_utils_ptr import action_info, get_episode_fire_counts, print_wildfire_grid
# Load configurations
mohito_config = load_config("MOHITO")
general_config = load_config("General_Config")
NUM_PARALLEL_ENVS=1
TUNING_MODE = general_config.TUNING_MODE


if general_config.DB == "postgres":
    from experiment_logger_postgres import ExperimentDB, RunLogger
else:
    from experiment_logger_sqlite import ExperimentDB, RunLogger
    

# Setup device
if general_config.DEVICE != (None or ''):
    DEVICE = general_config.DEVICE
else:    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


class MohitoCriticController:
    """
    Holds the replay buffer, handles exchanging observations + actions between policies at training time.
    """

    def __init__(self,
                 agent_names: List[str],
                 buffer_size: int,
                 batch_size: int,
                 experiences_before_update: int,
                 backups_per_update: int,
                 steps_per_checkpoint: int,
                 wait_until_full: bool = True):
        """
        Args:
            agent_names (List[str]): List of agent names to update.
            buffer_size (int): Size of the replay buffer.
            batch_size (int): Size of the batch to sample from the replay buffer.
            experiences_before_update (int): How many new experiences to add to the replay buffer before sampling.
            backups_per_update (int): How many times to backup / update per reaching the # of new experiences.
            steps_per_checkpoint (int): How many steps to take before saving a checkpoint.
            wait_until_full (bool): If True, will not sample from the replay buffer until it is full.
        """
        self.agent_names = agent_names

        self.buffer = []
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        assert self.batch_size <= self.buffer_size, "Batch size must be less than or equal to buffer size."

        self.experiences_before_update = experiences_before_update
        self.backups_per_update = backups_per_update
        self.wait_until_full = wait_until_full

        self._experience_counter = 0
        self._update_counter = 0
        self._checkpoint_counter = 0
        self._total_checkpoints = 0

        self.steps_per_checkpoint = steps_per_checkpoint

    def __call__(self, policies: Dict[str, Mohito], o, a, r, o_prime) -> bool:
        """
        Adds the experience to the replay buffer, then if the buffer is full, and it is time to update then sample and update policies.

        Args:
            policies (Dict[str, Mohito]): Dictionary of policies to update.
            o (Dict[str, Any]): Observations from the environment.
            a (Dict[str, Any]): Actions taken by the policies.
            r (Dict[str, float]): Rewards received from the environment.
            o_prime (Dict[str, Any]): Next observations from the environment.
        Returns:
            bool: Whether the buffer was updated
        """
        # Add experience to the replay buffer
        self.buffer.append({'o': o, 'a': a, 'r': r, 'o_prime': o_prime})

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            self._experience_counter += 1

        did_update = False
        losses = {}
        # Update
        if (len(self.buffer) == self.buffer_size or not self.wait_until_full):
            if (self._experience_counter % self.experiences_before_update == 0):
                for j in range(self.backups_per_update):
                    batch = random.sample(self.buffer, self.batch_size)
                    batch_losses = self.update(policies, batch)
                    if batch_losses:
                        losses.update(batch_losses)
                    self._update_counter += 1

                self._experience_counter = 0
                did_update = True

            # Checkpoint saving
            if (self._checkpoint_counter % self.steps_per_checkpoint == 0 
                    and self._checkpoint_counter > 0):
                self._total_checkpoints += 1
                self._checkpoint_counter = 0
            else:
                self._checkpoint_counter += 1

        return did_update,losses

    def update(self, policies: Dict[str, Mohito], batch: List[Tuple[Dict[str, Any]]]):
        # Get ed and ed_prime for updates
        hyperedges = {
            name: p.update_forward(batch)
            for name, p in policies.items()
        }

        # Update and collect losses
        losses = {}
        for name, p in policies.items():
            loss_dict = p.update(batch, hyperedges=hyperedges)
            if loss_dict:
                losses[name] = loss_dict
        
        return losses


class MOHITORunner:
    """Main MOHITO training runner"""
    
    def __init__(self, args):
        self.args = args
        self.device = DEVICE
        self.best_eval_reward = float("-inf")
        # Setup environment and models
        self.env = self._init_env()
        self.agent_policies, self.agent_names = self._init_models()
        
        # Setup controller
        training_params = mohito_config.training if isinstance(mohito_config.training, dict) else vars(mohito_config.training)
        self.update_controller = MohitoCriticController(
            agent_names=self.agent_names,
            buffer_size=training_params['buffer_size'],
            batch_size=training_params['batch_size'],
            experiences_before_update=training_params['experiences_before_update'],
            backups_per_update=training_params['backups_per_update'],
            steps_per_checkpoint=training_params['steps_per_checkpoint'],
            wait_until_full=training_params['wait_until_full']
        )
        
        # Setup logging
        self.writer, self.checkpoint_dir = setup_tensorboard_logging(args)
        
        # Setup DB logger 
        self.db = ExperimentDB()    
        experiment_meta = dict(
            run_type="training",
            model_family="MOHITO",
            model_type="smart",
            name=args.experiment_name,
            variation1="MOHITO",
            variation2="hypergraph",
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
        
        # Handle both dict and SimpleNamespace
        env_params = mohito_config.environment if isinstance(mohito_config.environment, dict) else vars(mohito_config.environment)
        training_params = mohito_config.training if isinstance(mohito_config.training, dict) else vars(mohito_config.training)
        
        env = wildfire_v0.parallel_env(
            configuration=configuration,
            parallel_envs=1,
            max_steps=general_config.MAX_STEPS,
            device=self.device,     
            single_seeding=False,
            show_bad_actions=False,
            observe_other_power=True,
            observe_other_suppressant=True,
            override_initialization_check=True,
            log_directory="outputs/",

        )
        
        # env.reset()
        env = action_mapping_wrapper_v0(env)
        # env.reset()
        env = mohito_hypergraph_wrapper_v0(env)
        # env.reset()
        
        return env
    
    def _init_models(self):
        """Initialize MOHITO agent policies"""
        observations, _ = self.env.reset()
        agent_names = copy.copy(self.env.agents)
        
        # Extract MOHITO hyperparameters - handle both dict and SimpleNamespace
        mohito_config_dict = mohito_config if isinstance(mohito_config, dict) else vars(mohito_config)
        mohito_hyperparameters = {
            k: v for k, v in mohito_config_dict.items() 
            if k not in ['training', 'environment','convergence', 'validation']
        }
        
        agent_policies = {
            name: Mohito(name=name, **mohito_hyperparameters)
            for name in agent_names
        }
        
        logging.info(f"Initialized {len(agent_names)} MOHITO agents: {agent_names}")
        
        return agent_policies, agent_names
    
    def evaluate(self):
        """Evaluate current policy"""
        total_rewards = []
        total_burnouts = []
        total_putouts = []
        total_lengths = []
        eval_seeds = []
        
        # Set models to eval mode
        for policy in self.agent_policies.values():
            policy.eval()
        
        with torch.no_grad():
            for eval_episode in range(general_config.VALIDATION_EPISODES):
                seed = get_validation_seed(eval_episode, general_config)
                observations, _ = self.env.reset(seed=seed)
                eval_seeds.append(seed)
                episode_reward = 0
                episode_putouts = 0
                episode_burnouts = 0
                steps = 0
                
                while (not torch.all(self.env.finished)) :

                    # Get actions from actors
                    actions_dict = {}
                    # Call observe() first for all agents
                    for agent_name in self.agent_names:
                        obs = observations[agent_name]
                        self.agent_policies[agent_name].observe(obs)

                    # Then get actions using .act()
                    agent_actions_raw = {
                        agent_name: self.agent_policies[agent_name].act(
                            action_space=self.env.action_space(agent_name),
                            return_logits=False)
                        for agent_name in self.agent_names
                    }

                    # Process actions for environment
                    actions_dict = {}
                    for agent_name in self.agent_names:
                        action = agent_actions_raw[agent_name]
                        if torch.is_tensor(action):
                            action = action.item()
                        
                        # Your existing action validation logic...
                        action_int = int(action)
                        asp = self.env.action_space(agent_name)
                        
                        if hasattr(asp, "spaces") and len(asp.spaces) == 1:
                            sub = asp.spaces[0]
                            valid = sub.start <= action_int < sub.start + sub.n
                        else:
                            valid = False
                        
                        if valid and action_int != -1:
                            env_action = (action_int, 0)
                        else:
                            env_action = (-1, -1)
                        
                        actions_dict[agent_name] = torch.tensor(env_action, device=self.device)
                    # Environment step
                    observations, rewards, terms, truncs, infoes = self.env.step(actions_dict)
                    
                    # Process metrics
                    episode_burnouts += int(infoes.get("burnouts", 0))
                    episode_putouts += int(infoes.get("putouts", 0))
                    
                    # Calculate step reward
                    step_reward = np.mean([r.mean().item() if torch.is_tensor(r) else r for r in rewards.values()])
                    episode_reward += step_reward
                    
                    steps += 1
                
                # Store episode metrics
                total_rewards.append(episode_reward)
                total_burnouts.append(episode_burnouts)
                total_putouts.append(episode_putouts)
                total_lengths.append(steps)
        
        # Set models back to train mode
        for policy in self.agent_policies.values():
            policy.train()
        
        # Calculate averages
        avg_reward = np.mean(total_rewards)
        avg_burnouts = np.mean(total_burnouts)
        avg_putouts = np.mean(total_putouts)
        avg_length = np.mean(total_lengths)
        
        return avg_reward, avg_burnouts, avg_putouts, avg_length, str(eval_seeds)
    def train(self):
        """Main training loop"""
        training_params = mohito_config.training if isinstance(mohito_config.training, dict) else vars(mohito_config.training)

        for ep_no in range(1, general_config.EPISODES + 1):
            # Start timer 
            t0 = time.perf_counter()
            
            # Reset environment with training seed
            seed = get_training_seed(ep_no, general_config)
            observations, infos = self.env.reset(seed=seed)
            initial_state = deepcopy(self.env.state())
            
            episode_reward = 0
            episode_steps = 0
            episode_burnouts = 0
            episode_putouts = 0
            episode_noop_count = 0
            episode_fight_count = 0
            episode_supressant_wait_time = 0

            env_seeds = str(seed)
            while (not torch.all(self.env.finished)) :

                #Call observe() first 
                actor_outputs = {
                        name: self.agent_policies[name](
                            observations[name],
                            # observations[name][0], tried
                            action_space=self.env.action_space(name),
                            explore=True,
                        )
                        for name in self.agent_names
                    }

                agent_actions = {
                        agent_name: actor_outputs[agent_name].task_action
                        for agent_name in self.agent_names
                    }

                action_unwrapped = {agent: self.env.modifiers[agent].modify_action(action) for agent, action in agent_actions.items()}
                noop_counts,fight_counts=action_info(action_unwrapped)
                # print_wildfire_grid(self.env.state())
                # print(agent_actions)
                
                observation_prime, rewards, terminations, truncations, infos = self.env.step(agent_actions)
                step_supressant_wait_time=count_zero_suppressant_agents(self.env.state())

                # Update policies
                did_update,losses  = self.update_controller(
                    policies=self.agent_policies,
                    o=observations,
                    a=agent_actions,
                    r=rewards,
                    o_prime=observation_prime,
                )
                if losses:
                    actor_loss = np.mean([l.get('actor_loss', 0) for l in losses.values()])
                    critic_loss = np.mean([l.get('critic_loss', 0) for l in losses.values()])
                else:
                    actor_loss = critic_loss = 0.0
                # Track metrics
                episode_reward += int(sum(rewards.values()) / len(rewards))
                episode_burnouts += int(infos.get("burnouts", 0))
                episode_putouts += int(infos.get("putouts", 0))
                episode_noop_count+=noop_counts
                episode_fight_count+=fight_counts
                episode_supressant_wait_time+=step_supressant_wait_time

                episode_steps += 1
                
                observations = observation_prime
            
            # Calculate fire counts after episode ends
            final_state = self.env.state()
            episode_total_fires_count, new_fire_ignited_count = get_episode_fire_counts(
                initial_state, final_state, episode_burnouts, episode_putouts, NUM_PARALLEL_ENVS
            )
            
            # Create training metrics (adapted for MOHITO)
            train_metrics = {
                'actor_loss': actor_loss,
                'critic_loss': critic_loss,
                'entropy': 0.0,
                'approx_kl': 0.0,
                'value_mse': 0.0,
                'explained_variance': 0.0
            }
            
            # Performance metrics 
            elapsed = time.perf_counter() - t0
            fps = episode_steps / max(elapsed, 1e-9)
            
            # Create episode statistics 
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

            # Gradient summary (simplified for MOHITO)
            grad_summary = {
                'actor_vanishing': 0,
                'actor_exploding': 0,
                'critic_vanishing': 0,
                'critic_exploding': 0,
                'actor_max_grad': 0,
                'critic_max_grad': 0,
            }

            # Log episode using RunLogger
            self.run_logger.log_episode(self.args.experiment_name,
                episode=ep_no,
                episode_stats=episode_stats,
                train_metrics=train_metrics,
                grad_summary=grad_summary,
                total_actor_grad=0.0, 
                total_critic_grad=0.0,
                elapsed=elapsed,
                fps=fps,
            )
            
            # Run evaluation every VALIDATION_EVERY episodes
            if ep_no % general_config.VALIDATION_EVERY == 0 and ep_no > 0:
                eval_reward, eval_burnouts, eval_putouts, eval_length, eval_seeds = self.evaluate()
                  
                # Log evaluation using RunLogger
                self.run_logger.log_eval(
                    episode=ep_no,
                    eval_reward=eval_reward,
                    eval_burnouts=eval_burnouts,
                    eval_putouts=eval_putouts,
                    eval_length=eval_length,
                    eval_seeds=eval_seeds,

                    avg_noop= 0
                )
                if TUNING_MODE and eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                logging.info(f"Evaluation at episode {ep_no}: "
                            f"Reward: {eval_reward:.4f}, "
                            f"Burnouts: {eval_burnouts:.4f}, "
                            f"Putouts: {eval_putouts:.4f}, "
                            f"Length: {eval_length:.2f}, ")
            
            # Periodic logging (tabulated output)
            if ep_no % general_config.LOG_EVERY == 0:
                logging.info(tabulate.tabulate(
                    [   
                        ['Episode', ep_no],
                        ['Reward', f"{float(episode_reward):.4f}"],
                        ['Burnouts', episode_burnouts],
                        ['Putouts', episode_putouts],
                        ['Steps', episode_steps],
                        ['Updates', self.update_controller._update_counter],
                        ['Episode Seconds', f"{elapsed:.2f}"],
                        ['Steps/sec', f"{fps:.2f}"],
                    ],
                    headers=['Name', 'Value'],
                    tablefmt='orgtbl'
                ))
            
            # Save checkpoint
            if ep_no % general_config.SAVE_EVERY == 0 and ep_no > 0:
                checkpoint_data = {
                    f"agent_{name}": policy.state_dict()
                    for name, policy in self.agent_policies.items()
                }
                checkpoint_data.update({
                    "episode": ep_no,
                    "reward": episode_reward
                })
                
                save_checkpoint(
                    checkpoint_data, 
                    self.checkpoint_dir, 
                    ep_no, 
                    self.args.experiment_name
                )  
        if TUNING_MODE:
            return self.best_eval_reward
       



def main():
    """Main function"""
    args = parse_args()
    args.experiment_name = 'MOHITO'
    if(general_config.use_global_seed):
        set_global_seed(general_config.Global_Seed,general_config.use_deterministic)
    
    # Set seeds for reproducibility
    training_params = mohito_config.training if isinstance(mohito_config.training, dict) else vars(mohito_config.training)

    # Print configuration
    logging.info("=== MOHITO Training Configuration ===")
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Episodes: {general_config.EPISODES}")
    logging.info(f"Buffer Size: {training_params['buffer_size']}")
    logging.info(f"Batch Size: {training_params['batch_size']}")
    
    # Initialize and run trainer
    runner = MOHITORunner(args)
    runner.train()


if __name__ == '__main__':
    main()