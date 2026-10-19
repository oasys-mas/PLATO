import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class CategoricalMLPModule(nn.Module):
    """MLP categorical head."""
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=nn.Tanh,          # pass Module class
        hidden_w_init=nn.init.xavier_uniform_,
        hidden_b_init=nn.init.zeros_,
        output_nonlinearity=None,
        output_w_init=nn.init.xavier_uniform_,
        output_b_init=nn.init.zeros_,
        layer_normalization=False,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for hs in hidden_sizes:
            lin = nn.Linear(prev, hs)
            hidden_w_init(lin.weight); hidden_b_init(lin.bias)
            layers.append(lin)
            if layer_normalization:
                layers.append(nn.LayerNorm(hs))
            if hidden_nonlinearity:
                # hidden_nonlinearity is a Module *class*
                layers.append(hidden_nonlinearity())
            prev = hs
        out = nn.Linear(prev, output_dim)
        output_w_init(out.weight); output_b_init(out.bias)
        layers.append(out)
        if output_nonlinearity:
            # if you pass a Module class, call it; if callable, wrap
            layers.append(output_nonlinearity() if isinstance(output_nonlinearity, type) else output_nonlinearity)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.net(x)                     # (..., output_dim)
        return Categorical(logits=logits)        # distribution per last dim


class DecCategoricalMLPPolicy(nn.Module):
    """
    Decentralized categorical MLP policy (torch-only).

    Args:
        obs_dim: per-agent observation dimension (not global!)
        action_dim: number of discrete actions per agent
        n_agents: number of agents
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=nn.Tanh,
        hidden_w_init=nn.init.xavier_uniform_,
        hidden_b_init=nn.init.zeros_,
        output_nonlinearity=None,
        output_w_init=nn.init.xavier_uniform_,
        output_b_init=nn.init.zeros_,
        layer_normalization=False,
        name: str = "DecCategoricalMLPPolicy",
    ):
        super().__init__()
        self.name = name
        self.centralized = True      # centralized training flag (informational)
        self._n_agents = n_agents
        self._obs_dim = obs_dim
        self._action_dim = action_dim

        self.head = CategoricalMLPModule(
            input_dim=self._obs_dim,
            output_dim=self._action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization,
        )

        # orthogonal init (optional but nice)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    # -------- utility --------
    def grad_norm(self):
        total = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total += p.grad.norm(2).item() ** 2
        return float(np.sqrt(total))

    # -------- core forward --------
    def forward(self, obs, avail_actions):
        """
        obs:  tensor/np, shape (B, n_agents*obs_dim) or (B, n_agents, obs_dim)
        avail_actions: tensor/np, shape (B, n_agents*action_dim) or (B, n_agents, action_dim)
        returns: Categorical distribution over (B, n_agents, action_dim)
        """
        obs = torch.as_tensor(obs, dtype=torch.float32)
        avail_actions = torch.as_tensor(avail_actions, dtype=torch.float32)

        if obs.dim() == 2:
            # (B, n_agents*obs_dim) -> (B, n_agents, obs_dim)
            obs = obs.view(obs.shape[0], self._n_agents, self._obs_dim)
        if avail_actions.dim() == 2:
            avail_actions = avail_actions.view(avail_actions.shape[0], self._n_agents, self._action_dim)

        B = obs.shape[0]
        x = obs.reshape(B * self._n_agents, self._obs_dim)
        dist = self.head(x)  # Categorical over last dim for each agent row
        logits = dist.logits.view(B, self._n_agents, self._action_dim)

        # mask unavailable actions
        probs = F.softmax(logits, dim=-1)
        probs = probs * avail_actions
        probs_sum = probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # avoid NaN
        probs = probs / probs_sum

        return Categorical(probs=probs)

    # -------- API helpers --------
    def get_action(self, observation, available_actions=None, greedy=False):
        """
        observation: (n_agents*obs_dim,) or (n_agents, obs_dim)
        available_actions: (n_agents*action_dim,) or (n_agents, action_dim); if None, assume all 1s
        """
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            if obs.dim() == 2:  # (1, n_agents*obs_dim)
                obs = obs.view(1, self._n_agents, self._obs_dim)

            if available_actions is None:
                avail = torch.ones((1, self._n_agents, self._action_dim), dtype=torch.float32)
            else:
                avail = torch.as_tensor(available_actions, dtype=torch.float32).unsqueeze(0)
                if avail.dim() == 2:
                    avail = avail.view(1, self._n_agents, self._action_dim)

            dist = self.forward(obs.view(obs.size(0), -1), avail.view(avail.size(0), -1))
            if greedy:
                actions = torch.argmax(dist.probs, dim=-1)
            else:
                actions = dist.sample()
            log_probs = dist.log_prob(actions)  # (1, n_agents)
            return actions.squeeze(0).cpu().numpy(), {"probs": dist.probs.squeeze(0).cpu().numpy(),
                                                      "logp": log_probs.squeeze(0).sum().item()}

    def get_actions(self, observations, avail_actions, greedy=False):
        """
        observations: (B, n_agents*obs_dim) or (B, n_agents, obs_dim)
        avail_actions: (B, n_agents*action_dim) or (B, n_agents, action_dim)
        """
        with torch.no_grad():
            dist = self.forward(observations, avail_actions)
            if greedy:
                actions = torch.argmax(dist.probs, dim=-1)
            else:
                actions = dist.sample()
            actions_np = actions.cpu().numpy()
            agent_infos = {
                "action_probs": [dist.probs[i].cpu().numpy() for i in range(actions.size(0))]
            }
            return actions_np, agent_infos

    def log_likelihood(self, observations, avail_actions, actions):
        dist = self.forward(observations, avail_actions)
        ll = dist.log_prob(torch.as_tensor(actions))  # (B, n_agents)
        return ll.sum(dim=-1)                         # independent across agents

    def entropy(self, observations, avail_actions):
        dist = self.forward(observations, avail_actions)
        return dist.entropy().mean(dim=-1)

    def reset(self, dones=None):
        # stateless (MLP), nothing to reset; keep signature to match callers
        pass

    @property
    def vectorized(self):
        return True

    @property
    def recurrent(self):
        return False
