import torch
import torch.nn as nn
from torch.distributions import Categorical

from policies.policy_helper import compute_agents_encoding


# ------------------------------------------------------------
# Pointer Networks components - MLP version 
# ------------------------------------------------------------

class PointerNetworkEncoder(nn.Module):
    """
    MLP encoder over the (contextualized) candidate items.
    Processes each item independently.
    Expects [T, H] and returns [T, H].
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Simple MLP applied to each task independently
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            current_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights with Xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs: torch.Tensor, hidden_state=None):
        """
        inputs:  [T, H]
        hidden_state: ignored (kept for API compatibility)
        returns:
          enc_out:    [T, H]
          enc_hidden: None (no hidden state for MLP)
        """
        if inputs.dim() != 2:
            raise ValueError(f"Encoder expects [T, H], got {inputs.shape}")
        
        # Apply MLP to each row independently
        enc_out = self.mlp(inputs)  # [T, H]
        return enc_out, None


class AttentionMechanism(nn.Module):
    """
    Attention mechanism producing *scores* (logits), not probabilities.
    Supports:
      - "additive": Bahdanau-style attention
      - "dotproduct": Simple dot-product attention (no learned parameters)
      - "cosine_sim": Cosine similarity (normalized dot-product, no learned parameters)
    Decoder state is the Query; encoder outputs are Keys.
    """
    def __init__(self, hidden_dim, attention_type="additive"):
        super().__init__()
        self.attention_type = attention_type
        
        if attention_type == "additive":
            self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)  # keys (encoder outputs)
            self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)  # query (decoder state)
            self.v  = nn.Linear(hidden_dim, 1, bias=False)

            # Better initialization for attention
            for m in [self.Wk, self.Wq, self.v]:
                nn.init.xavier_uniform_(m.weight)
        elif attention_type in ("dotproduct", "cosine_sim"):
            # No learned parameters for dot-product or cosine similarity
            pass
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}. Use 'additive', 'dotproduct', or 'cosine_sim'.")

    def forward(self, encoder_outputs: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        encoder_outputs: [T, H]
        query:           [H]
        returns:
          scores/logits: [T]
        """
        if self.attention_type == "additive":
            T, H = encoder_outputs.size()
            q = query.unsqueeze(0).expand(T, -1)        # [T, H]
            e = torch.tanh(self.Wk(encoder_outputs) + self.Wq(q))  # [T, H]
            scores = self.v(e).squeeze(-1)              # [T]
        elif self.attention_type == "dotproduct":
            # Simple dot-product: scores[i] = encoder_outputs[i] · query
            scores = encoder_outputs @ query            # [T]
        elif self.attention_type == "cosine_sim":
            # Cosine similarity: normalized dot-product (loses magnitude info)
            enc_norm = torch.nn.functional.normalize(encoder_outputs, dim=1)  # [T, H]
            q_norm = torch.nn.functional.normalize(query, dim=0)              # [H]
            scores = enc_norm @ q_norm                  # [T]
        else:
            raise ValueError(f"Unknown attention_type: {self.attention_type}")
        return scores  # logits (no softmax)


class PointerNetworkDecoder(nn.Module):
    """
    Minimal Pointer decoder for one-step pointing:
    uses the agent/context vector as query to attend over encoder outputs.
    """
    def __init__(self, hidden_dim, attention_type="additive"):
        super().__init__()
        self.attn = AttentionMechanism(hidden_dim, attention_type)

    def forward(self, encoder_outputs: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        encoder_outputs: [T, H]
        query:           [H]  (e.g., agent context)
        returns:
          logits:        [T]  (attention scores)
        """
        return self.attn(encoder_outputs, query)


class ResidualBlock(nn.Module):
    """
    Residual block with weight normalization: x -> x + scale * act(WeightNorm(Linear)(x))
    Weight normalization decouples magnitude from direction, improving training stability.
    """
    def __init__(self, hidden_dim, activation=None, scale=0.1):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        # Use parametrizations API (newer, recommended)
        nn.utils.parametrizations.weight_norm(self.fc)
        self.act = activation if activation is not None else nn.ReLU()
        self.scale = scale
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return x + self.scale * self.act(self.fc(x))


# ------------------------------------------------------------
# Simplified Pointer policy (without LSTM memory)
# ------------------------------------------------------------

class Pointer_MLP_Actor(nn.Module):
    """
    Simplified Pointer Network policy (single-step pointing) without LSTM memory:
    - Encodes agent features into a context (query) 
    - Encodes candidate tasks (plus NOOP) with an MLP
    - Computes attention scores (logits) over tasks+NOOP
    - Returns logits (no softmax) - no hidden state needed

    Designed for policy-gradient algorithms (REINFORCE, PPO):
    -> build Categorical(logits=masked_logits) outside with your mask.
    """
    def __init__(
        self,
        task_feat_dim: int,
        agent_feat_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        attention_type: str = "additive",
        ablation_no_team_stats: bool = False,
        ablation_no_pointer: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.ablation_no_team_stats = ablation_no_team_stats
        self.ablation_no_pointer = ablation_no_pointer

        # NOTE: NOOP is now provided as input, not generated internally.
        # The caller must include NOOP as the last task in task_features.

        # Embed raw features to hidden_dim with better architecture
        self.agent_encoder = nn.Sequential(
            # 4 = mean - var - max - min stats
            nn.Linear(4 * agent_feat_dim, hidden_dim), 
            nn.ReLU(),
            # ResidualBlock(hidden_dim),
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(task_feat_dim, hidden_dim),  
            nn.ReLU(),
            # ResidualBlock(hidden_dim),
        )

        # Note: we encode tasks once with `task_encoder` and avoid an
        # extra per-key MLP to keep the model compact and fast.
        self.decoder = PointerNetworkDecoder(
            hidden_dim=hidden_dim,
            attention_type=attention_type,
        )

        # Ablation: independent task scoring head (used when ablation_no_pointer=True)
        self.no_pointer_head = nn.Linear(hidden_dim, 1)

        # Better initialization for all linears
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------- forward ----------
    def forward(
        self,
        task_features: torch.Tensor,   # [N_tasks, task_feat_dim] - MUST include NOOP as last task
        agent_features: torch.Tensor,  # [N_agents, agent_feat_dim]
        hidden_state=None,             # Kept for API compatibility, but ignored
    ):
        """
        IMPORTANT: task_features MUST include NOOP as the last task.
        The actor does NOT generate NOOP internally - it expects NOOP to be provided.
        
        Returns:
          logits:      [N_tasks]  (including NOOP which should be the last index)
          next_hidden: None (no memory)
        """
        device = task_features.device
        dtype  = task_features.dtype
        N = task_features.size(0)

        # --------- Edge case: no tasks -> return empty logits ---------
        if N == 0:
            # Should not happen if NOOP is always included, but handle gracefully
            return torch.tensor([0.0], device=device, dtype=dtype), None

        # --------- Agent context (decoder query) ---------
        if agent_features.numel() > 0:
            agents_stats = compute_agents_encoding(agent_features)         # [4 * agent_feat_dim]

            # Ablation: no team stats — zero out everything except self features
            if self.ablation_no_team_stats:
                feat_dim = agent_features.shape[1] if agent_features.dim() > 1 else agent_features.shape[0]
                agents_stats = agents_stats.clone()
                agents_stats[feat_dim:] = 0.0  # zero others_mean, others_var, relative_pos

            query = self.agent_encoder(agents_stats)                      # [H]
        else:
            query = torch.zeros(self.hidden_dim, device=device, dtype=dtype)

        # --------- Encode all tasks (including NOOP which is provided as input) ---------
        enc_tasks = self.task_encoder(task_features)                   # [N, H]

        # --------- Attention scores (logits) ---------
        if self.ablation_no_pointer:
            # Ablation: score tasks independently, ignoring agent context (query)
            logits = self.no_pointer_head(enc_tasks).squeeze(-1)       # [N]
        else:
            # The attention mechanism handles query-key interaction via Wk and Wq
            logits = self.decoder(enc_tasks, query)                    # [N]

        # Safety: replace non-finite with zeros (rare but protects training)
        if not torch.isfinite(logits).all():
            logits = torch.zeros_like(logits)

        return logits, None  # No hidden state