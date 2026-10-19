import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNValueCritic(torch.nn.Module):
    """
    GNN Critic with ENRICHED node features instead of distance weights/global context.
    
    Key Design Principles:
    - NO distance-weighted edges (all edges are unweighted)
    - NO global context node  
    - RICHER node features derived purely from env.state()
    - Uses GCN layers with skip connections for stability
    - Mean+Max pooling for capturing both average and worst-case states
    
    Agent Node Features (11 features):
        [x, y, suppressant, capacity, equipment, 
         suppressant_ratio, is_empty, is_full, x_norm, y_norm, has_equipment]
    
    Fire Node Features (8 features):
        [x, y, intensity, fuel, 
         intensity_ratio, fuel_ratio, is_critical, x_norm, y_norm]
         (Note: last feature is y_norm, but we pad to 8 for alignment)
    """
    
    def __init__(self, agent_feat_dim=11, task_feat_dim=9, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Separate encoders for agents and fires (with residual projections)
        self.agent_proj = torch.nn.Linear(agent_feat_dim, hidden_dim)
        self.agent_encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
        )
        
        self.fire_proj = torch.nn.Linear(task_feat_dim, hidden_dim)
        self.fire_encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
        )
        
        # GCN layers (2 layers for stability)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # Pre-head MLP - input is 2*hidden_dim due to mean+max pooling
        self.pre_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
        )
        
        # Value head
        self.value_head = torch.nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Weight initialization for stable training"""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, agent_nodes, fire_nodes=None, edge_index=None):
        """
        Forward pass with:
        - Unweighted message passing (no edge_weight)
        - Skip connections for gradient stability
        - Mean+max pooling for capturing both average and worst-case states
        
        Args:
            agent_nodes: [num_agents, agent_feat_dim] - enriched agent features
            fire_nodes: [num_fires, task_feat_dim] - enriched fire features (optional)
            edge_index: [2, num_edges] - edge connectivity (unweighted)
        """
        device = agent_nodes.device
        dtype = agent_nodes.dtype
        
        # Handle empty cases
        if agent_nodes.shape[0] == 0:
            return torch.zeros(1, device=device, dtype=dtype)
        
        # Encode agents with residual connection
        agent_proj = self.agent_proj(agent_nodes)
        agent_encoded = F.leaky_relu(agent_proj + self.agent_encoder(agent_proj))
        num_agents = agent_nodes.shape[0]
        
        # Handle fire nodes with residual connection
        if fire_nodes is not None and fire_nodes.shape[0] > 0:
            fire_proj = self.fire_proj(fire_nodes)
            fire_encoded = F.leaky_relu(fire_proj + self.fire_encoder(fire_proj))
            x = torch.cat([agent_encoded, fire_encoded], dim=0)
        else:
            x = agent_encoded
        
        num_nodes = x.shape[0]
        
        # Handle edge_index - create self-loops if empty
        if edge_index is None or edge_index.shape[1] == 0:
            edge_index = torch.stack([
                torch.arange(num_nodes, device=device, dtype=torch.long),
                torch.arange(num_nodes, device=device, dtype=torch.long)
            ], dim=0)
        
        # GCN layers WITH skip connections (NO edge weights - unweighted)
        x1 = F.leaky_relu(self.gcn1(x, edge_index)) + x
        x2 = F.leaky_relu(self.gcn2(x1, edge_index)) + x1
        
        # Deep residual: add initial encoding back (helps gradient flow)
        x2 = x2 + x
        
        # Mean+Max pooling to capture both average and worst-case structures
        mean_pool = torch.mean(x2, dim=0, keepdim=True)  # [1, hidden_dim]
        max_pool = torch.max(x2, dim=0, keepdim=True)[0]  # [1, hidden_dim]
        global_features_pooled = torch.cat([mean_pool, max_pool], dim=-1)  # [1, 2*hidden_dim]
        
        # Value prediction
        h = self.pre_head(global_features_pooled)
        value = self.value_head(h)
        
        return value.squeeze(-1)


class MLPValueCritic(torch.nn.Module):
    """
    Ablation critic: replaces GNNValueCritic with a simple MLP.
    Accepts the same call signature (agent_nodes, fire_nodes, edge_index) but
    ignores edge_index and uses mean-pooling over all nodes instead of GCN message passing.
    """

    def __init__(self, agent_feat_dim=11, task_feat_dim=9, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.agent_proj = torch.nn.Linear(agent_feat_dim, hidden_dim)
        self.fire_proj  = torch.nn.Linear(task_feat_dim,  hidden_dim)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
        )
        self.value_head = torch.nn.Linear(hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, agent_nodes, fire_nodes=None, edge_index=None):
        device = agent_nodes.device
        dtype  = agent_nodes.dtype

        if agent_nodes.shape[0] == 0:
            return torch.zeros(1, device=device, dtype=dtype)

        agent_enc = F.leaky_relu(self.agent_proj(agent_nodes))          # [A, H]
        agent_mean = agent_enc.mean(dim=0)                              # [H]
        agent_max  = agent_enc.max(dim=0)[0]                           # [H]

        if fire_nodes is not None and fire_nodes.shape[0] > 0:
            fire_enc  = F.leaky_relu(self.fire_proj(fire_nodes))        # [F, H]
            fire_mean = fire_enc.mean(dim=0)                            # [H]
            fire_max  = fire_enc.max(dim=0)[0]                         # [H]
        else:
            fire_mean = torch.zeros(self.hidden_dim, device=device, dtype=dtype)
            fire_max  = torch.zeros(self.hidden_dim, device=device, dtype=dtype)

        # Combine agent and fire summaries: mean of (agent_mean, fire_mean) and same for max
        combined_mean = (agent_mean + fire_mean) / 2.0                 # [H]
        combined_max  = torch.max(agent_max, fire_max)                 # [H]

        global_feat = torch.cat([combined_mean, combined_max], dim=-1).unsqueeze(0)  # [1, 2H]
        h = self.mlp(global_feat)
        return self.value_head(h).squeeze(-1)
