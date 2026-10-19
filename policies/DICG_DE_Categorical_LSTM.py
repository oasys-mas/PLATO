import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


# --- minimal torch-only building blocks ---

class MLPEncoderModule(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=nn.Tanh,              # pass a Module class
        hidden_w_init=nn.init.xavier_uniform_,
        hidden_b_init=nn.init.zeros_,
        output_nonlinearity=None,                 # optional Module class
        output_w_init=nn.init.xavier_uniform_,
        output_b_init=nn.init.zeros_,
        layer_normalization=False,
    ):
        super().__init__()
        layers, prev = [], input_dim
        for hs in hidden_sizes:
            lin = nn.Linear(prev, hs)
            hidden_w_init(lin.weight); hidden_b_init(lin.bias)
            layers.append(lin)
            if layer_normalization:
                layers.append(nn.LayerNorm(hs))
            if hidden_nonlinearity:
                layers.append(hidden_nonlinearity())
            prev = hs
        out = nn.Linear(prev, output_dim)
        output_w_init(out.weight); output_b_init(out.bias)
        layers.append(out)
        if output_nonlinearity:
            layers.append(output_nonlinearity())
        self.net = nn.Sequential(*layers)

        # nice-to-have: orthogonal on linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)


class CategoricalLSTMModule(nn.Module):
    """LSTM categorical head (time-major input)."""
    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=False)
        self.out = nn.Linear(hidden_size, output_size)
        # init
        for name, p in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0.0)
        nn.init.orthogonal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, x, h=None, c=None):
        # x: (T, B, D)
        if h is not None and c is not None:
            y, (hn, cn) = self.lstm(x, (h, c))
        else:
            y, (hn, cn) = self.lstm(x)
        logits = self.out(y)               # (T, B, A)
        return Categorical(logits=logits), hn, cn


# --- torch-only decentralized categorical LSTM policy ---

class DecCategoricalLSTMPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,                         # per-agent obs dim
        action_dim: int,                      # per-agent discrete actions
        n_agents: int,
        encoder_hidden_sizes=(64, 64),
        embedding_dim=64,
        lstm_hidden_size=64,
        state_include_actions=False,
        hidden_nonlinearity=nn.Tanh,
        hidden_w_init=nn.init.xavier_uniform_,
        hidden_b_init=nn.init.zeros_,
        output_nonlinearity=None,
        output_w_init=nn.init.xavier_uniform_,
        output_b_init=nn.init.zeros_,
        layer_normalization=False,
        name='DecCategoricalLSTMPolicy',
    ):
        super().__init__()
        self.name = name
        self.centralized = True
        self.vectorized = True

        self._n_agents = n_agents
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._embedding_dim = embedding_dim
        self.state_include_actions = state_include_actions

        self._prev_actions = None   # (B, n_agents, action_dim) one-hot
        self._prev_h = None         # (1, B*n_agents, H)
        self._prev_c = None

        mlp_input = obs_dim + action_dim if state_include_actions else obs_dim
        self.encoder = MLPEncoderModule(
            input_dim=mlp_input,
            output_dim=embedding_dim,
            hidden_sizes=encoder_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization,
        )

        self.lstm_head = CategoricalLSTMModule(
            input_size=embedding_dim,
            output_size=action_dim,
            hidden_size=lstm_hidden_size,
        )
    @property
    def lstm_hidden(self):
        return self._prev_h, self._prev_c

    @lstm_hidden.setter
    def lstm_hidden(self, value):
        # The wrapper might pass None during resets
        if value is None:
            self._prev_h, self._prev_c = None, None
        else:
            self._prev_h, self._prev_c = value
    def grad_norm(self):
        total = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total += p.grad.norm(2).item() ** 2
        return float(np.sqrt(total))

    def forward(self, obs_n, avail_actions_n, actions_n=None):
            """
            Handles both batched trajectory forward (for hypothetical future use)
            and single-step batch forward (for PPO updates).
            """
            obs_n = torch.as_tensor(obs_n, dtype=torch.float32)
            avail_actions_n = torch.as_tensor(avail_actions_n, dtype=torch.float32)

            # --- Identify input format ---
            # Case A: Training update with a flat batch of (batch_size, 1, obs_dim)
            is_flat_batch_update = obs_n.dim() == 3 and obs_n.shape[2] == self._obs_dim
            # Case B: Hypothetical trajectory training with (N, T, n_agents, ...)
            is_trajectory_batch = obs_n.dim() == 4 or (obs_n.dim() == 3 and obs_n.shape[2] == self._n_agents * self._obs_dim)

            if is_flat_batch_update:
                N, T, feat_dim = obs_n.shape  # N=batch_size, T=1, feat_dim=obs_dim
                
                # Reshape obs for encoder: (N*T, obs_dim)
                mlp_input_flat = obs_n.view(N * T, feat_dim)

                if self.state_include_actions:
                    # For T=1, the shifted previous action will be all zeros, which is correct.
                    prev_actions_zeros = torch.zeros(N * T, self._action_dim, device=obs_n.device)
                    mlp_input_flat = torch.cat([mlp_input_flat, prev_actions_zeros], dim=-1)

                enc = self.encoder(mlp_input_flat)  # (N*T, embedding_dim)
                
                # Reshape for LSTM: (T, N, embedding_dim)
                enc_lstm = enc.view(N, T, self._embedding_dim).transpose(0, 1)

                # Use the hidden state that the PolicyWrapper has set on this object
                h, c = self._prev_h, self._prev_c

                dists_t, _, _ = self.lstm_head(enc_lstm, h, c)  # (T, N, action_dim)
                
                # Reshape back to (N, T, action_dim)
                logits = dists_t.logits.transpose(0, 1)
                
                probs = F.softmax(logits, dim=-1)
                probs = probs * avail_actions_n
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                
                return Categorical(probs=probs)

            elif is_trajectory_batch:
                # Original logic for processing full multi-agent trajectories
                if obs_n.dim() == 3:
                    N, T = obs_n.shape[:2]
                    obs_n = obs_n.view(N, T, self._n_agents, self._obs_dim)
                    avail_actions_n = avail_actions_n.view(N, T, self._n_agents, self._action_dim)
                else:
                    N, T = obs_n.shape[:2]

                if self.state_include_actions:
                    assert actions_n is not None, "actions_n required"
                    actions_n = torch.as_tensor(actions_n, dtype=torch.long).unsqueeze(-1)
                    onehot = torch.zeros(N, T, self._n_agents, self._action_dim, dtype=torch.float32, device=obs_n.device)
                    onehot.scatter_(-1, actions_n, 1.0)
                    onehot_shift = torch.cat(
                        [torch.zeros(N, 1, self._n_agents, self._action_dim, device=obs_n.device), onehot[:, :-1]], dim=1
                    )
                    obs_n = torch.cat([obs_n, onehot_shift], dim=-1)

                enc = self.encoder(obs_n)
                enc = enc.transpose(0, 1).contiguous().view(T, N * self._n_agents, self._embedding_dim)

                dists_t, _, _ = self.lstm_head(enc)  # Assumes zero initial hidden state
                
                logits = dists_t.logits.view(T, N, self._n_agents, self._action_dim)
                logits = logits.transpose(0, 1).contiguous()
                
                probs = F.softmax(logits, dim=-1)
                probs = probs * avail_actions_n
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                
                return Categorical(probs=probs)
            else:
                raise ValueError(f"Unsupported input shape for obs_n: {obs_n.shape}")
            
    # ---------- single-step forward (acting) ----------
    def step_forward(self, obs_n, avail_actions_n):
        """
        obs_n: (B, n_agents*obs_dim) or (B, n_agents, obs_dim)
        avail_actions_n: (B, n_agents*action_dim) or (B, n_agents, action_dim)
        returns: Categorical over (B, n_agents, action_dim)
        """
        obs_n = torch.as_tensor(obs_n, dtype=torch.float32)
        avail_actions_n = torch.as_tensor(avail_actions_n, dtype=torch.float32)

        if obs_n.dim() == 2:
            B = obs_n.shape[0]
            obs_n = obs_n.view(B, self._n_agents, self._obs_dim)
        else:
            B = obs_n.shape[0]

        if avail_actions_n.dim() == 2:
            avail_actions_n = avail_actions_n.view(B, self._n_agents, self._action_dim)

        if self.state_include_actions:
            if self._prev_actions is None:
                self._prev_actions = torch.zeros(B, self._n_agents, self._action_dim, dtype=torch.float32, device=obs_n.device)
            obs_n = torch.cat([obs_n, self._prev_actions], dim=-1)

        enc = self.encoder(obs_n)                                   # (B, n_agents, emb)
        enc = enc.view(1, B * self._n_agents, self._embedding_dim)  # (1, B*n_agents, emb)

        dists_t, h, c = self.lstm_head(enc, self._prev_h, self._prev_c)
        self._prev_h, self._prev_c = h, c

        probs = F.softmax(dists_t.logits.squeeze(0), dim=-1)        # (B*n_agents, A)
        probs = probs.view(B, self._n_agents, self._action_dim)

        # mask + renorm
        probs = probs * avail_actions_n
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        return Categorical(probs=probs)

    # ---------- convenience APIs ----------
    def get_actions(self, obs_n, avail_actions_n, greedy=False):
        with torch.no_grad():
            dist = self.step_forward(obs_n, avail_actions_n)
            actions = torch.argmax(dist.probs, dim=-1) if greedy else dist.sample()
            actions_np = actions.cpu().numpy()
            info = {"action_probs": [dist.probs[i].cpu().numpy() for i in range(actions.size(0))]}

            if self.state_include_actions:
                B = actions.size(0)
                onehot = torch.zeros(B, self._n_agents, self._action_dim, dtype=torch.float32, device=actions.device)
                onehot.scatter_(-1, actions.unsqueeze(-1), 1.0)
                self._prev_actions = onehot
            return actions_np, info

    def entropy(self, observations, avail_actions, actions=None):
        d = self.forward(observations, avail_actions, actions)
        return d.entropy().mean(dim=-1)

    def log_likelihood(self, observations, avail_actions, actions):
        d = self.forward(observations, avail_actions, actions)
        return d.log_prob(torch.as_tensor(actions)).sum(dim=-1)  # independent across agents

    def reset(self, dones=None):
        if dones is None or (isinstance(dones, (list, np.ndarray, torch.Tensor)) and bool(np.all(dones))):
            self._prev_actions = None
            self._prev_h = None
            self._prev_c = None

    @property
    def recurrent(self):
        return True
