import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DeepSetAgentEncoder(nn.Module):
    """
    FIXED: More stable agent encoder
    """
    def __init__(self, agent_feat_dim: int, hidden_dim: int, dropout=0.1):
        super().__init__()
        
        self.phi = nn.Sequential(
            nn.Linear(agent_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),  # Added batch norm for stability
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, agent_features):
        if agent_features.shape[0] == 0:
            return torch.zeros(self.phi[0].out_features, device=agent_features.device, dtype=agent_features.dtype)
        
        # Handle single agent case for batch norm
        if agent_features.shape[0] == 1:
            # Skip batch norm for single sample
            x = agent_features
            for i, layer in enumerate(self.phi):
                if isinstance(layer, nn.BatchNorm1d):
                    continue  # Skip batch norm
                x = layer(x)
            transformed = x
        else:
            transformed = self.phi(agent_features)
        
        pooled = torch.mean(transformed, dim=0)
        embedded_agents = self.rho(pooled)
        embedded_agents = embedded_agents + pooled  # Residual connection
        
        return embedded_agents
