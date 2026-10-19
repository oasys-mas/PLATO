# MOHITO Base Module - Shared components for wildfire and rideshare domains
import random
from typing import Dict, List, Tuple, Any
import torch


class MohitoCriticController:
    """
    Holds the replay buffer, handles exchanging observations + actions between policies at training time.
    Shared across wildfire and rideshare domains.
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

    def __call__(self, policies: Dict, o, a, r, o_prime) -> Tuple[bool, Dict]:
        """
        Adds the experience to the replay buffer, then if the buffer is full, and it is time to update then sample and update policies.

        Args:
            policies (Dict): Dictionary of policies to update.
            o (Dict): Observations from the environment.
            a (Dict): Actions taken by the policies.
            r (Dict): Rewards received from the environment.
            o_prime (Dict): Next observations from the environment.
        Returns:
            Tuple[bool, Dict]: Whether the buffer was updated, and losses dict
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

        return did_update, losses

    def update(self, policies: Dict, batch: List[Tuple[Dict[str, Any]]]):
        """Update policies using batch of experiences"""
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


def get_training_params(config):
    """Extract training parameters from config (handles both dict and SimpleNamespace)"""
    return config.training if isinstance(config.training, dict) else vars(config.training)


def get_mohito_hyperparameters(config):
    """Extract MOHITO hyperparameters from config, excluding training/environment/validation sections"""
    config_dict = config if isinstance(config, dict) else vars(config)
    return {
        k: v for k, v in config_dict.items() 
        if k not in ['training', 'environment', 'convergence', 'validation']
    }


def create_default_train_metrics(actor_loss=0.0, critic_loss=0.0):
    """Create default training metrics dict for MOHITO (which doesn't compute PPO metrics)"""
    return {
        'actor_loss': actor_loss,
        'critic_loss': critic_loss,
        'entropy': 0.0,
        'approx_kl': 0.0,
        'value_mse': 0.0,
        'explained_variance': 0.0
    }


def create_default_grad_summary():
    """Create default gradient summary dict for MOHITO"""
    return {
        'actor_vanishing': 0,
        'actor_exploding': 0,
        'critic_vanishing': 0,
        'critic_exploding': 0,
        'actor_max_grad': 0,
        'critic_max_grad': 0,
    }
