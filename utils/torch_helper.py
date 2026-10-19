from collections import deque, namedtuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import numpy as np
from collections import deque


class ConvergenceDetector:
    """
    Simple convergence detector that tracks policy parameter changes.
    Stops training when policy updates become minimal over a window of episodes.
    """
    
    def __init__(self, 
                 threshold=5e-2,           # Minimum parameter change threshold
                 window_size=20,           # Number of episodes to check for convergence
                 min_episodes=500,         # Minimum episodes before checking convergence
                 patience=15,              # Number of consecutive convergence checks before stopping
                 adaptive_threshold=True): # Auto-adjust threshold based on initial changes
        """
        Args:
            threshold: Minimum L2 norm of parameter changes to consider as "no change"
            window_size: Number of recent episodes to average parameter changes over
            min_episodes: Minimum number of episodes before starting convergence checks
            patience: How many consecutive windows must show convergence before stopping
            adaptive_threshold: If True, automatically adjust threshold based on model scale
        """
        self.initial_threshold = threshold
        self.threshold = threshold
        self.window_size = window_size
        self.min_episodes = min_episodes
        self.patience = patience
        self.adaptive_threshold = adaptive_threshold
        
        # Track parameter changes
        self.param_changes = deque(maxlen=window_size)
        self.prev_params = None
        self.episode_count = 0
        self.convergence_count = 0
        
        # Adaptive threshold tracking
        self.initial_changes = []
        self.threshold_calibrated = False
        
        # Statistics
        self.last_param_change = float('inf')
        self.avg_param_change = float('inf')
        
    def update(self, model):
        """
        Update convergence detector with current model parameters.
        
        Args:
            model: PyTorch model (actor policy) to track
            
        Returns:
            bool: True if training should stop (converged), False otherwise
        """
        self.episode_count += 1
        
        # Get current parameters as a flat tensor
        current_params = self._get_flat_params(model)
        
        # Calculate parameter change if we have previous parameters
        if self.prev_params is not None:
            param_change = torch.norm(current_params - self.prev_params).item()
            self.param_changes.append(param_change)
            self.last_param_change = param_change
            
            # Collect initial changes for adaptive threshold
            if self.adaptive_threshold and len(self.initial_changes) < 20:
                self.initial_changes.append(param_change)
            
            # Calibrate threshold after collecting enough initial data
            if self.adaptive_threshold and not self.threshold_calibrated and len(self.initial_changes) >= 20:
                self._calibrate_threshold()
                self.threshold_calibrated = True
            
            # Calculate average change over the window
            if len(self.param_changes) >= self.window_size:
                self.avg_param_change = np.mean(list(self.param_changes))
        
        # Store current parameters for next comparison
        self.prev_params = current_params.clone()
        
        # Check for convergence only after minimum episodes
        if self.episode_count < self.min_episodes:
            return False
            
        # Check if we have enough data points
        if len(self.param_changes) < self.window_size:
            return False
            
        # Check convergence condition
        if self.avg_param_change < self.threshold:
            self.convergence_count += 1
            if self.convergence_count >= self.patience:
                return True  # Converged!
        else:
            self.convergence_count = 0  # Reset counter if not converging
            
        return False
    
    def _calibrate_threshold(self):
        """Automatically calibrate threshold based on initial parameter changes"""
        if len(self.initial_changes) < 10:
            return
        
        # Use median of initial changes as baseline
        median_change = np.median(self.initial_changes)
        
        # Set threshold to 1% of typical initial changes
        auto_threshold = median_change * 0.01
        
        # Use the smaller of the two thresholds (more conservative)
        self.threshold = min(self.initial_threshold, auto_threshold)
        
        print(f"Auto-calibrated convergence threshold: {self.threshold:.2e} "
              f"(median initial change: {median_change:.2e})")
    
    def _get_flat_params(self, model):
        """Get all model parameters as a single flattened tensor."""
        params = []
        for param in model.parameters():
            if param.requires_grad:
                params.append(param.data.view(-1))
        return torch.cat(params)
    
    def get_stats(self):
        """Get convergence statistics for logging."""
        return {
            'last_param_change': self.last_param_change,
            'avg_param_change': self.avg_param_change,
            'convergence_count': self.convergence_count,
            'episodes_checked': self.episode_count,
            'threshold': self.threshold,
            'calibrated_threshold': self.threshold_calibrated,
            'converged': self.convergence_count >= self.patience
        }
    
    def reset(self):
        """Reset the convergence detector."""
        self.param_changes.clear()
        self.prev_params = None
        self.episode_count = 0
        self.convergence_count = 0
        self.last_param_change = float('inf')
        self.avg_param_change = float('inf')
        self.initial_changes.clear()
        self.threshold_calibrated = False
        self.threshold = self.initial_threshold


class EarlyStoppingConvergence:
    """
    Alternative convergence detector that combines parameter changes with performance.
    Stops when both policy parameters and performance have stabilized.
    """
    
    def __init__(self,
                 param_threshold=1e-6,     # Parameter change threshold
                 reward_threshold=0.01,    # Reward improvement threshold
                 window_size=30,           # Window for averaging
                 min_episodes=300,         # Minimum episodes before checking
                 patience=5):              # Patience for convergence
        
        self.param_threshold = param_threshold
        self.reward_threshold = reward_threshold
        self.window_size = window_size
        self.min_episodes = min_episodes
        self.patience = patience
        
        # Track both parameters and rewards
        self.param_changes = deque(maxlen=window_size)
        self.rewards = deque(maxlen=window_size)
        self.prev_params = None
        self.episode_count = 0
        self.convergence_count = 0
        
    def update(self, model, episode_reward):
        """
        Update with both model parameters and episode reward.
        
        Args:
            model: PyTorch model to track
            episode_reward: Reward from the episode
            
        Returns:
            bool: True if converged, False otherwise
        """
        self.episode_count += 1
        self.rewards.append(episode_reward)
        
        # Track parameter changes
        current_params = self._get_flat_params(model)
        param_converged = False
        
        if self.prev_params is not None:
            param_change = torch.norm(current_params - self.prev_params).item()
            self.param_changes.append(param_change)
            
            if len(self.param_changes) >= self.window_size:
                avg_param_change = np.mean(list(self.param_changes))
                param_converged = avg_param_change < self.param_threshold
        
        self.prev_params = current_params.clone()
        
        # Check reward convergence
        reward_converged = False
        if len(self.rewards) >= self.window_size:
            recent_rewards = list(self.rewards)
            reward_std = np.std(recent_rewards)
            reward_converged = reward_std < self.reward_threshold
        
        # Check overall convergence only after minimum episodes
        if self.episode_count < self.min_episodes:
            return False
            
        # Both conditions must be met
        if param_converged and reward_converged:
            self.convergence_count += 1
            if self.convergence_count >= self.patience:
                return True
        else:
            self.convergence_count = 0
            
        return False
    
    def _get_flat_params(self, model):
        """Get all model parameters as a single flattened tensor."""
        params = []
        for param in model.parameters():
            if param.requires_grad:
                params.append(param.data.view(-1))
        return torch.cat(params)
    
    def get_stats(self):
        """Get convergence statistics for logging."""
        avg_param_change = np.mean(list(self.param_changes)) if self.param_changes else float('inf')
        reward_std = np.std(list(self.rewards)) if len(self.rewards) >= 2 else float('inf')
        
        return {
            'avg_param_change': avg_param_change,
            'reward_std': reward_std,
            'convergence_count': self.convergence_count,
            'param_threshold': self.param_threshold,
            'reward_threshold': self.reward_threshold,
            'converged': self.convergence_count >= self.patience
        }



# Residual Block Helper Class
class ResidualBlock(nn.Module):
    """Standard residual block with layer norm and dropout"""
    
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """Forward pass with residual connection"""
        residual = x
        x = self.layers(x)
        x = x + residual  # Residual connection
        x = self.layer_norm(x)  # Post-residual layer norm
        return x

# Enhanced initialization function for residual models
def apply_residual_orthogonal_init(module, gain=1.0):
    """Enhanced initialization for residual networks"""
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
        if hasattr(module, 'weight') and module.weight is not None:
            # Use smaller gain for residual connections to maintain stability
            torch.nn.init.orthogonal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d)):
        if hasattr(module, 'weight') and module.weight is not None:
            torch.nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, ResidualBlock):
        # Special initialization for residual blocks
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=0.5)  # Smaller gain for stability
                
# Add this function right after your imports, before the MAPPOTrainer class:

def randomize_actor_weights(actor):
    """
    Simple function to randomly reinitialize actor weights to break NOOP bias
    """
    print("Randomizing actor weights...")
    
    with torch.no_grad():
        for param in actor.parameters():
            # Add random noise to all parameters
            noise = torch.randn_like(param) * 0.1
            param.data += noise
    
    print("Actor weights randomized!")                
                
                
                
                
                
                

                