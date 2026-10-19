import torch
import torch.nn as nn
from torch.distributions import Categorical

from policies.policy_helper import compute_agents_encoding



# ------------------------------------------------------------
# Pointer Networks components (Vinyals et al., 2015) + RL tweaks
# Encoder-Decoder LSTM architecture following the original paper
# ------------------------------------------------------------

def _init_lstm(lstm: nn.LSTM):
    """Initialize LSTM weights following best practices."""
    for name, param in lstm.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param)
        elif "bias" in name:
            param.data.fill_(0.0)
            # forget gate bias = 1
            n = param.size(0)
            param.data[n // 4 : n // 2] = 1.0


class PointerNetworkEncoder(nn.Module):
    """
    LSTM encoder over the (contextualized) candidate items.
    Expects [T, H] and returns [T, H]; wraps a batch dim internally.
    Returns both outputs (for attention) and final hidden state (for decoder init).
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        _init_lstm(self.lstm)

    def forward(self, inputs: torch.Tensor, hidden_state=None):
        """
        inputs:  [T, H]
        hidden_state: optional (h, c) with shapes [L, B, H] (B must be 1)
        returns:
          enc_out:    [T, H]  - outputs at each position (for attention)
          enc_hidden: (h, c)  - final hidden state (for decoder init)
        """
        if inputs.dim() != 2:
            raise ValueError(f"Encoder expects [T, H], got {inputs.shape}")
        x = inputs.unsqueeze(0)   # [1, T, H]
        out, h = self.lstm(x, hidden_state)  # out: [1, T, H]
        return out.squeeze(0), h


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

            for m in [self.Wk, self.Wq, self.v]:
                nn.init.uniform_(m.weight, -0.1, 0.1)
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
    LSTM-based Pointer decoder following the original Pointer Networks paper.
    
    For single-step pointing:
    - Takes decoder input (agent context) and processes through LSTM
    - Uses LSTM hidden state as query to attend over encoder outputs
    - Hidden state can be initialized from encoder final state or passed across steps
    """
    def __init__(self, hidden_dim, num_layers=1, dropout=0.0, attention_type="additive"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Decoder LSTM: input is the context/agent encoding
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        _init_lstm(self.lstm)
        
        # Attention mechanism
        self.attn = AttentionMechanism(hidden_dim, attention_type)

    def forward(
        self, 
        encoder_outputs: torch.Tensor,  # [T, H]
        decoder_input: torch.Tensor,    # [H] - agent context
        hidden_state=None,              # (h, c) - can be encoder final state
    ):
        """
        encoder_outputs: [T, H] - encoder outputs for attention
        decoder_input:   [H]    - input to decoder LSTM (agent context)
        hidden_state:    (h, c) - decoder LSTM hidden state (optional)
        
        returns:
          logits:       [T]     - attention scores over encoder outputs
          next_hidden:  (h, c)  - updated decoder hidden state
        """
        # Process decoder input through LSTM
        # decoder_input: [H] -> [1, 1, H]
        dec_in = decoder_input.unsqueeze(0).unsqueeze(0)
        dec_out, next_hidden = self.lstm(dec_in, hidden_state)  # [1, 1, H]
        
        # Use decoder LSTM output as query for attention
        query = dec_out.squeeze(0).squeeze(0)  # [H]
        
        # Compute attention scores (logits)
        logits = self.attn(encoder_outputs, query)  # [T]
        
        return logits, next_hidden


# ------------------------------------------------------------
# Pointer policy (one-step selection per env step)
# Following original Pointer Networks: LSTM Encoder + LSTM Decoder
# ------------------------------------------------------------

class Pointer_LSTM_Actor(nn.Module):
    """
    Pointer Network policy following the original paper (Vinyals et al., 2015):
    - LSTM Encoder: processes candidate tasks sequence
    - LSTM Decoder: takes agent context, produces query for attention
    - Attention: points to one of the encoder outputs (tasks)
    
    Architecture:
    1. Embed tasks -> LSTM Encoder -> encoder outputs [T, H]
    2. Embed agent features -> decoder input [H]
    3. LSTM Decoder (initialized from encoder) -> query [H]
    4. Attention(query, encoder_outputs) -> logits [T]
    
    Hidden state is for the decoder LSTM, passed across environment steps.

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

        # Embed raw features to hidden_dim
        self.agent_encoder = nn.Sequential(
            # 4 = mean - var - max - min stats
            nn.Linear(4 * agent_feat_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),         nn.Tanh(),
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(task_feat_dim, hidden_dim),  nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Pointer Net Encoder: LSTM over task sequence
        self.encoder = PointerNetworkEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Pointer Net Decoder: LSTM that produces query for attention
        self.decoder = PointerNetworkDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            attention_type=attention_type,
        )

        # Ablation: independent task scoring head (used when ablation_no_pointer=True)
        self.no_pointer_head = nn.Linear(hidden_dim, 1)

        # Pointer-style init for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)

    # ---------- helpers ----------
    def _bootstrap_hidden(self, device, dtype=torch.float32):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=device, dtype=dtype)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=device, dtype=dtype)
        return (h0, c0)

    # ---------- forward ----------
    def forward(
        self,
        task_features: torch.Tensor,   # [N_tasks, task_feat_dim] - MUST include NOOP as last task
        agent_features: torch.Tensor,  # [N_agents, agent_feat_dim]
        hidden_state=None,             # (h, c) for decoder LSTM, optional
    ):
        """
        IMPORTANT: task_features MUST include NOOP as the last task.
        The actor does NOT generate NOOP internally - it expects NOOP to be provided.
        
        Returns:
          logits:      [N_tasks]  (including NOOP which should be the last index)
          next_hidden: (h, c) for decoder LSTM
        """
        device = task_features.device
        dtype  = task_features.dtype
        N = task_features.size(0)

        # --------- Edge case: no tasks -> return empty logits ---------
        if N == 0:
            # Should not happen if NOOP is always included, but handle gracefully
            if hidden_state is None:
                hidden_state = self._bootstrap_hidden(device=device, dtype=dtype)
            return torch.tensor([0.0], device=device, dtype=dtype), hidden_state

        # --------- Encode tasks with LSTM encoder ---------
        enc_tasks = self.task_encoder(task_features)           # [N, H]
        enc_out, enc_hidden = self.encoder(enc_tasks)          # enc_out: [N, H], enc_hidden: (h, c)

        # --------- Prepare decoder hidden state ---------
        # If no hidden state provided, initialize from encoder final state
        if hidden_state is None:
            dec_hidden = enc_hidden
        else:
            # Ensure format [L, B=1, H]
            h, c = hidden_state
            if h.dim() == 2:  # [L, H]
                h = h.unsqueeze(1)
                c = c.unsqueeze(1)
            dec_hidden = (h, c)

        # --------- Agent context (decoder input) ---------
        if agent_features.numel() > 0:
            agents_stats = compute_agents_encoding(agent_features)  # [4 * agent_feat_dim]

            # Ablation: no team stats — zero out everything except self features
            if self.ablation_no_team_stats:
                feat_dim = agent_features.shape[1] if agent_features.dim() > 1 else agent_features.shape[0]
                agents_stats = agents_stats.clone()
                agents_stats[feat_dim:] = 0.0  # zero others_mean, others_var, relative_pos

            dec_input = self.agent_encoder(agents_stats)            # [H]
        else:
            dec_input = torch.zeros(self.hidden_dim, device=device, dtype=dtype)

        # --------- Decode: LSTM + Attention ---------
        logits, next_hidden = self.decoder(enc_out, dec_input, dec_hidden)  # [N], (h, c)

        # Ablation: no pointer — override with independent task scoring
        if self.ablation_no_pointer:
            logits = self.no_pointer_head(enc_out).squeeze(-1)     # [N]

        # Safety: replace non-finite with zeros (rare but protects training)
        if not torch.isfinite(logits).all():
            logits = torch.zeros_like(logits)

        return logits, next_hidden
