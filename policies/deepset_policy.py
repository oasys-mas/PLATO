import torch
import torch.nn as nn

from policies.policy_helper import compute_agents_encoding


# ------------------------------------------------------------
# Deep Sets baseline actor (fixed-index action head)
# ------------------------------------------------------------

class DeepSet_Actor(nn.Module):
    """
    Deep Sets baseline policy (Zaheer et al., 2017) with a FIXED-INDEX action head:
    - Encodes each task independently with a shared MLP (phi)
    - Pools the task embeddings with a permutation-invariant sum (rho after pooling)
    - Concatenates the pooled task summary with the team-statistics context
    - A fixed-size MLP head outputs logits over `max_tasks` index slots + NOOP

    This is the baseline promised in the NeurIPS rebuttal: the set encoder is
    permutation-invariant over variable-sized task sets, but the OUTPUT head is
    fixed-index. Slot i means "the i-th task in the current action mapping", so
    the head cannot track task identity when the task set changes within an
    episode, in contrast to PLATO's content-based pointing.

    Interface intentionally mirrors Pointer_MLP_Actor:
      forward(task_features, agent_features, hidden_state=None) -> (logits, None)
    but logits have FIXED length max_tasks + 1 (last index = NOOP), independent
    of the number of tasks currently observed. The caller must mask slots
    >= the current task count and map the sampled index back to the mapping.
    """

    def __init__(
        self,
        task_feat_dim: int,
        agent_feat_dim: int,
        hidden_dim: int,
        max_tasks: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        pooling: str = "sum",
    ):
        super().__init__()
        if pooling not in ("sum", "mean"):
            raise ValueError(f"Unknown pooling: {pooling}. Use 'sum' or 'mean'.")
        self.hidden_dim = hidden_dim
        self.max_tasks = max_tasks
        self.pooling = pooling

        # phi: shared per-task embedding
        phi_layers = []
        current_dim = task_feat_dim
        for _ in range(max(num_layers, 1)):
            phi_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ])
            current_dim = hidden_dim
        self.phi = nn.Sequential(*phi_layers)

        # rho: post-pooling transformation of the set summary
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Team context from the same 4 statistics as PLATO (mean/var/min-style stats)
        self.agent_encoder = nn.Sequential(
            nn.Linear(4 * agent_feat_dim, hidden_dim),
            nn.ReLU(),
        )

        # Fixed-index head: max_tasks task slots + 1 NOOP slot (last index)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_tasks + 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Small-gain policy-head init (standard for PPO): keeps the initial
        # policy near-uniform and slows premature entropy collapse.
        with torch.no_grad():
            self.head[-1].weight.mul_(0.01)

    # ---------- forward ----------
    def forward(
        self,
        task_features: torch.Tensor,   # [N_tasks, task_feat_dim] - includes NOOP as last row
        agent_features: torch.Tensor,  # [N_agents, agent_feat_dim]
        hidden_state=None,             # Kept for API compatibility, ignored
    ):
        """
        task_features follows the same convention as the pointer actors: the NOOP
        sentinel is the LAST row. It is excluded from the set encoding here (the
        head has its own dedicated NOOP slot at index max_tasks).

        Returns:
          logits:      [max_tasks + 1]  (fixed length; last index = NOOP)
          next_hidden: None (no memory)
        """
        device = task_features.device
        dtype = task_features.dtype

        # --------- Set summary over the real tasks (exclude the NOOP row) ---------
        real_tasks = task_features[:-1] if task_features.size(0) > 0 else task_features
        if real_tasks.size(0) > 0:
            embedded = self.phi(real_tasks)                       # [T, H]
            if self.pooling == "sum":
                pooled = embedded.sum(dim=0)                      # [H]
            else:
                pooled = embedded.mean(dim=0)                     # [H]
            set_summary = self.rho(pooled)                        # [H]
        else:
            set_summary = torch.zeros(self.hidden_dim, device=device, dtype=dtype)

        # --------- Team context (same statistics as PLATO's query) ---------
        if agent_features.numel() > 0:
            agents_stats = compute_agents_encoding(agent_features)  # [4 * agent_feat_dim]
            context = self.agent_encoder(agents_stats)              # [H]
        else:
            context = torch.zeros(self.hidden_dim, device=device, dtype=dtype)

        # --------- Fixed-index logits ---------
        logits = self.head(torch.cat([set_summary, context], dim=-1))  # [max_tasks + 1]

        # Safety: replace non-finite with zeros (rare but protects training)
        if not torch.isfinite(logits).all():
            logits = torch.zeros_like(logits)

        return logits, None  # No hidden state

    # ---------- fixed-head index helpers ----------
    def valid_head_indices(self, num_real_tasks: int):
        """Head indices that are selectable when `num_real_tasks` tasks are visible.

        Slots beyond max_tasks are unreachable by construction (the structural
        limitation of a fixed-index head under task openness).
        """
        n_slots = min(num_real_tasks, self.max_tasks)
        return list(range(n_slots)) + [self.max_tasks]

    def head_index_to_mapping_index(self, head_index: int, num_mapping_entries: int) -> int:
        """Map a sampled head index to an index into the action mapping
        (mapping convention: real tasks first, NOOP last)."""
        if head_index >= self.max_tasks:
            return num_mapping_entries - 1  # NOOP
        return head_index
