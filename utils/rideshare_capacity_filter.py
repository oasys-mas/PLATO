"""
Rideshare-specific fixes for MOHITO integration.

Two problems are addressed:

1. **Capacity filtering** (the ~-5000 rewards):
   The rideshare env exposes ALL unaccepted tasks (state==0) in every agent's
   action space, regardless of capacity.  When an agent at ``pool_limit``
   accepts another passenger, the env applies ``pool_limit_cost`` (−100)
   **every subsequent step**.  With random/early exploration, agents routinely
   exceed the limit and accumulate catastrophic negative rewards.

   Fix: monkey-patch the raw env so that ``update_actions``,
   ``update_observations``, and ``action_space`` exclude unaccepted (general)
   tasks from agents at or above ``pool_limit``.

2. **Action-mapping index mismatch**:
   ``action_task.py`` passes ``agent_action_mapping`` (within-batch passenger
   indices like [0, 1, 3, 7, 12]) to the MOHITO hypergraph wrapper as ``amap``.
   But the wrapper treats ``amap`` values as **observation-local** indices
   (0, 1, 2, …) for connecting graph nodes **and** for the action output that
   ``step_environment`` later looks up through ``agent_action_mapping`` *again*
   (double indirection).
   This works in wildfire (identity mapping) but is always wrong in rideshare
   when there are gaps in the passenger indices.

   Fix: insert a lightweight wrapper between ``action_mapping_wrapper_v0`` and
   ``mohito_hypergraph_wrapper_v0`` that replaces the batch-level indices with
   identity indices [0, 1, …, N−1].  ``step_environment`` still uses the real
   ``agent_action_mapping`` on the raw env for correct passenger lookup.

Usage (in _init_env):
    env = rideshare_v0.parallel_env(...)
    env.reset()
    apply_capacity_filter(env)                     # fix 1
    env = action_mapping_wrapper_v0(env)
    env.reset()
    env = rideshare_amap_fix_wrapper_v0(env)       # fix 2
    env.reset()
    env = mohito_hypergraph_wrapper_v0(env)
    env.reset()
"""

import types
import logging
import os
import sys

import torch
from tensordict.tensordict import TensorDict
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier

from free_range_zoo.envs.rideshare.env.spaces.actions import build_action_space

# Ensure MOHITO's package is importable
_mohito_base = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'baselines', 'mohito', 'wildfire', 'mohito',
)
if _mohito_base not in sys.path:
    sys.path.insert(0, _mohito_base)

from mohito.wrappers.wrapper_util import shared_wrapper  # type: ignore

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Fix 1  –  Capacity-aware filtering (monkey-patch on raw env)
# ═══════════════════════════════════════════════════════════════════════════

def _get_agent_assigned_counts(raw_env, agent_index: int) -> torch.IntTensor:
    """
    Return a (parallel_envs,) tensor with the number of passengers currently
    assigned to *agent_index* in each batched environment.
    """
    if raw_env._state.passengers is None or raw_env._state.passengers.shape[0] == 0:
        return torch.zeros(raw_env.parallel_envs, dtype=torch.int32, device=raw_env.device)

    assigned = raw_env._state.passengers[:, 7] == agent_index
    if assigned.sum() == 0:
        return torch.zeros(raw_env.parallel_envs, dtype=torch.int32, device=raw_env.device)

    batch_ids = raw_env._state.passengers[assigned][:, 0].long()
    counts = torch.zeros(raw_env.parallel_envs, dtype=torch.int32, device=raw_env.device)
    ones = torch.ones(batch_ids.shape[0], dtype=torch.int32, device=raw_env.device)
    counts.scatter_add_(0, batch_ids, ones)
    return counts


def _get_filtered_agent_tasks(raw_env, agent_index: int, pool_limit: int):
    """
    Build the boolean mask over *all* passengers that selects the tasks an agent
    should see.  When the agent is at or above ``pool_limit`` in a given batch,
    unaccepted (general) tasks for that batch are excluded.
    """
    general_tasks = raw_env._state.passengers[:, 6] == 0           # unaccepted
    exclusive_tasks = raw_env._state.passengers[:, 7] == agent_index  # assigned

    counts = _get_agent_assigned_counts(raw_env, agent_index)
    at_capacity = counts >= pool_limit  # (parallel_envs,) bool

    if at_capacity.any():
        passenger_batch = raw_env._state.passengers[:, 0].long()
        in_full_batch = at_capacity[passenger_batch]
        filtered_general = general_tasks & ~in_full_batch
        return filtered_general | exclusive_tasks

    return general_tasks | exclusive_tasks


# ---------------------------------------------------------------------------
# Patched methods – these replace the originals on the raw env instance
# ---------------------------------------------------------------------------

@torch.no_grad()
def _patched_update_actions(self):
    """update_actions with capacity-aware filtering."""
    self.environment_task_count = torch.bincount(
        self._state.passengers[:, 0], minlength=self.parallel_envs
    )

    index_shifts = torch.roll(
        torch.cumsum(self.environment_task_count, dim=0, dtype=torch.int32), 1, 0
    )
    index_shifts[0] = 0

    task_range = torch.arange(
        0, self.environment_task_count.sum(), device=self.device
    )
    task_indices = task_range - index_shifts[self._state.passengers[:, 0]].flatten()

    pool_limit = self.agent_config.pool_limit

    for agent_name, agent_index in self.agent_name_mapping.items():
        agent_tasks = _get_filtered_agent_tasks(self, agent_index, pool_limit)

        self.agent_task_count[agent_index] = torch.bincount(
            self._state.passengers[:, 0][agent_tasks],
            minlength=self.parallel_envs,
        )

        indices_nested = torch.nested.as_nested_tensor(
            task_indices[agent_tasks].split(
                self.agent_task_count[agent_index].tolist(), dim=0
            ),
            device=self.device,
            layout=torch.jagged,
        )

        self.agent_observation_mapping[agent_name] = indices_nested
        self.agent_action_mapping[agent_name] = indices_nested.clone()


@torch.no_grad()
def _patched_action_space(self, agent: str):
    """action_space with capacity-aware filtering."""
    agent_index = self.agent_name_mapping[agent]
    pool_limit = self.agent_config.pool_limit

    agent_tasks = _get_filtered_agent_tasks(self, agent_index, pool_limit)

    actions = self._state.passengers[agent_tasks][:, [0, 6]]
    return build_action_space(actions, self.agent_task_count[agent_index])


@torch.no_grad()
def _patched_update_observations(self):
    """update_observations with capacity-aware filtering to keep obs aligned with actions."""
    self.environment_task_count = self._state.passengers[:, 0].bincount(
        minlength=self.parallel_envs
    )

    # ---- task-level features (shared across agents) ----
    task_positional_info = self._state.passengers[:, 1:5]
    entered_info = self._state.passengers[:, [8]]

    accepted = self._state.passengers[:, [6]] == 1
    accepted_info = torch.where(accepted, self._state.passengers[:, [7]], -100)
    riding = self._state.passengers[:, [6]] == 2
    riding_info = torch.where(riding, self._state.passengers[:, [7]], -100)

    fare_info = self._state.passengers[:, [5]]

    task_observations = torch.cat(
        [task_positional_info, accepted_info, riding_info, fare_info, entered_info],
        dim=1,
    )
    self.task_store = torch.nested.as_nested_tensor(
        task_observations.split(self.environment_task_count.tolist()),
        device=self.device,
        layout=torch.jagged,
    )

    # ---- agent-level features ----
    agent_observations = torch.empty(
        (self.parallel_envs, self.agent_config.num_agents, 4),
        dtype=torch.int32,
        device=self.device,
    )
    agent_observations[:, :, :2] = self._state.agents[:, :, :2]
    for agent_name, agent_index in self.agent_name_mapping.items():
        exclusive_tasks = self._state.passengers[:, [7]] == agent_index
        accepted_info_count = (
            self._state.passengers[:, [0]][accepted & exclusive_tasks]
            .bincount(minlength=self.parallel_envs)
        )
        riding_info_count = (
            self._state.passengers[:, [0]][riding & exclusive_tasks]
            .bincount(minlength=self.parallel_envs)
        )
        agent_observations[:, agent_index, 2] = accepted_info_count
        agent_observations[:, agent_index, 3] = riding_info_count

    # ---- per-agent observations (capacity-filtered) ----
    pool_limit = self.agent_config.pool_limit
    self.observations = {}

    for agent_name, agent_index in self.agent_name_mapping.items():
        agent_tasks = _get_filtered_agent_tasks(self, agent_index, pool_limit)

        self.agent_task_count[agent_index] = (
            self._state.passengers[agent_tasks][:, 0]
            .bincount(minlength=self.parallel_envs)
        )

        nested_observation = torch.nested.as_nested_tensor(
            task_observations[agent_tasks].split(
                self.agent_task_count[agent_index].tolist()
            ),
            layout=torch.jagged,
        )

        agent_mask = torch.ones(
            self.agent_config.num_agents, dtype=torch.bool, device=self.device
        )
        agent_mask[agent_index] = False

        self.observations[agent_name] = TensorDict(
            {
                "self": agent_observations[:, agent_index].clone(),
                "others": agent_observations[:, agent_mask].clone(),
                "tasks": nested_observation.clone(),
            },
            batch_size=[self.parallel_envs],
            device=self.device,
        )


def apply_capacity_filter(parallel_env):
    """
    Monkey-patch the underlying raw rideshare environment so that agents at
    (or above) ``pool_limit`` can no longer see *unaccepted* tasks in their
    action space / observations.

    Call this **once** after the first ``env.reset()`` and **before** applying
    MOHITO wrappers.

    Args:
        parallel_env: The batched parallel rideshare environment.
    """
    raw = _unwrap_to_raw(parallel_env)
    pool_limit = raw.agent_config.pool_limit
    logger.info(
        f"Applying capacity filter: agents at pool_limit={pool_limit} "
        f"will not see unaccepted tasks (pool_limit_cost avoidance)."
    )

    raw.update_actions = types.MethodType(_patched_update_actions, raw)
    raw.action_space = types.MethodType(_patched_action_space, raw)
    raw.update_observations = types.MethodType(_patched_update_observations, raw)
    raw._capacity_filter_applied = True


# ═══════════════════════════════════════════════════════════════════════════
# Fix 2  –  Action-mapping identity wrapper (no author-code changes)
# ═══════════════════════════════════════════════════════════════════════════

class _AmapFixModifier(BaseModifier):
    """
    Intercept the ``(obs, action_map)`` tuple coming from
    ``action_mapping_wrapper_v0`` and replace the batch-level passenger indices
    in ``action_map['agent_action_mapping']`` with identity indices
    ``[0, 1, …, N-1]`` per batch.

    Why this is correct:
    - Both the action space and the observations are built from the **same**
      filtered task set in the **same** order.  So the i-th action entry always
      corresponds to the i-th observation task.
    - MOHITO uses ``amap`` for two things:
        (a) connecting hyperedge nodes to the correct task graph-node
        (b) storing task indices in ``_wrapper_task_action_mapping`` which
            ``modify_action`` returns during ``step``.
    - With identity ``amap``, MOHITO outputs task_idx = i.
      ``step_environment`` then correctly resolves it via
      ``agent_action_mapping[i]`` on the raw env (which still holds the real
      within-batch passenger indices).
    """
    env = True
    subject_agent = True

    def __init__(self, env, subject_agent: str):
        self.subject_agent = subject_agent

        # Navigate to the raw env to access agent_task_count, etc.
        raw = env
        while True:
            if hasattr(raw, 'aec_env'):
                raw = raw.aec_env
            elif hasattr(raw, 'env'):
                raw = raw.env
            else:
                break
        self.raw_env = raw
        self.cur_obs = None

    def modify_obs(self, observation):
        obs, action_map = observation

        agent_index = self.raw_env.agent_name_mapping[self.subject_agent]
        task_counts = self.raw_env.agent_task_count[agent_index]

        # Build identity mapping: [0, 1, ..., N-1] per batch
        identity = torch.nested.as_nested_tensor(
            [torch.arange(int(c.item()), dtype=torch.int32,
                          device=self.raw_env.device)
             for c in task_counts],
            device=self.raw_env.device,
            layout=torch.jagged,
        )

        self.cur_obs = obs, {'agent_action_mapping': identity}
        return self.cur_obs

    def observe(self):
        return self.cur_obs


def rideshare_amap_fix_wrapper_v0(env):
    """
    Wrap the environment so that the ``agent_action_mapping`` seen by the
    downstream MOHITO hypergraph wrapper contains observation-local identity
    indices instead of batch-level passenger indices.

    Place this **after** ``action_mapping_wrapper_v0`` and **before**
    ``mohito_hypergraph_wrapper_v0``.
    """
    return shared_wrapper(env, _AmapFixModifier)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

class RideshareStepResult:
    """Per-step metrics from RideshareMetricTracker.post_step().

    Backward-compatible: unpacks as (pickups, dropoffs, fares) so existing
    code ``p, d, _ = tracker.post_step()`` continues to work unchanged.
    """
    __slots__ = ('pickups', 'dropoffs', 'fares', 'successes')

    def __init__(self, pickups, dropoffs, fares, successes):
        self.pickups = pickups
        self.dropoffs = dropoffs
        self.fares = fares
        self.successes = successes  # agents currently at a passenger destination

    def __iter__(self):
        return iter((self.pickups, self.dropoffs, self.fares))


class RideshareMetricTracker:
    """
    Track actual pickups and dropoffs by monitoring passenger state changes
    in the raw environment (the env's `infos` dict does NOT contain these).

    Usage:
        tracker = RideshareMetricTracker(env)

    Then in the training loop:
        tracker.pre_step()
        observations, rewards, ... = env.step(actions)
        pickups, dropoffs, fares = tracker.post_step()
    """

    def __init__(self, parallel_env):
        self._raw = _unwrap_to_raw(parallel_env)
        self._prev_riding = 0
        self._prev_total = 0

    def pre_step(self):
        """Snapshot passenger states before env.step()."""
        p = self._raw._state.passengers
        if p is not None and p.numel() > 0:
            self._prev_riding = (p[:, 6] == 2).sum().item()
            self._prev_total = p.shape[0]
        else:
            self._prev_riding = 0
            self._prev_total = 0

    def post_step(self):
        """
        Compare passenger states after env.step() and return counts.

        Returns:
            (pickups, dropoffs, total_fare_reward)
            - pickups: passengers that transitioned to state 2 (riding)
            - dropoffs: passengers that exited the system (dropped off)
            - total_fare_reward: estimated positive reward from fares
        """
        p = self._raw._state.passengers
        if p is not None and p.numel() > 0:
            curr_riding = (p[:, 6] == 2).sum().item()
            curr_total = p.shape[0]
        else:
            curr_riding = 0
            curr_total = 0

        # In a single step the order is:
        #   1. movement_transition      (no state changes)
        #   2. passenger_state_transition  (accept: 01, pick: 12)
        #   3. passenger_exit_transition   (drop: removes state-2 passenger)
        #   4. passenger_entry_transition  (adds new state-0 passengers)
        #
        # So after one step:
        #   curr_riding = prev_riding + pickups - dropoffs
        #   curr_total  = prev_total  - dropoffs + entries
        #
        # Entries are always state-0 so they don't affect riding.
        # We can solve:
        #   entries   = curr_total - prev_total + dropoffs
        #   dropoffs  = prev_riding + pickups - curr_riding
        #
        # To resolve the two unknowns (pickups, dropoffs) we note that
        # pickups ≥ 0 and dropoffs ≥ 0.  In practice both happen rarely in the
        # same step, so the approximation below is very close:
        delta_riding = curr_riding - self._prev_riding
        if delta_riding >= 0:
            # Net increase in riding pickups dominate, dropoffs ≈ 0
            pickups = delta_riding
            dropoffs = 0
        else:
            # Net decrease dropoffs dominate, pickups ≈ 0
            pickups = 0
            dropoffs = -delta_riding

        successes = self._count_at_destinations()
        return RideshareStepResult(pickups, dropoffs, 0.0, successes)

    def _count_at_destinations(self):
        """Count agents currently positioned at any passenger's destination."""
        p = self._raw._state.passengers
        a = self._raw._state.agents  # [n_envs, n_agents, 2]
        if p is None or p.numel() == 0 or a is None or a.numel() == 0:
            return 0
        dests = p[:, 3:5]  # (dest_y, dest_x)
        n_envs, n_agents = a.shape[0], a.shape[1]
        count = 0
        for env_i in range(n_envs):
            env_mask = p[:, 0] == env_i
            env_dests = dests[env_mask]
            if env_dests.numel() == 0:
                continue
            for ag_i in range(n_agents):
                pos = a[env_i, ag_i]  # [2]
                if (env_dests == pos.unsqueeze(0)).all(dim=1).any().item():
                    count += 1
        return count

    def reset(self):
        """Call after env.reset() to re-initialize snapshot."""
        self.pre_step()  # snapshot initial state


class RideshareEpisodeMetrics:
    """Accumulates per-step rideshare metrics and logs them to TensorBoard.

    Usage::
        # at init:
        self.episode_metrics = RideshareEpisodeMetrics()

        # each episode start:
        self.episode_metrics.reset()

        # each step after post_step() and rideshare_action_info():
        self.episode_metrics.record(step_result, noops=noop_count, accepts=accept_count)

        # end of episode:
        self.episode_metrics.log(writer, episode)

    Tracked episode totals:
        pickups   — passengers picked up
        dropoffs  — passengers dropped off
        noops     — noop actions submitted
        accepts   — accept actions submitted
        successes — agent-at-destination events (agent pos == any passenger dest)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.pickups = 0
        self.dropoffs = 0
        self.noops = 0
        self.accepts = 0
        self.successes = 0

    def record(self, step_result, noops=0, accepts=0):
        """Accumulate metrics for one step.

        Args:
            step_result: RideshareStepResult from tracker.post_step().
            noops:   noop count for this step (from rideshare_action_info).
            accepts: accept count for this step (from rideshare_action_info).
        """
        if isinstance(step_result, RideshareStepResult):
            self.pickups += step_result.pickups
            self.dropoffs += step_result.dropoffs
            self.successes += step_result.successes
        else:
            self.pickups += step_result[0]
            self.dropoffs += step_result[1]
        self.noops += noops
        self.accepts += accepts

    def log(self, writer, global_step, prefix='rideshare'):
        """Write accumulated episode totals to TensorBoard."""
        writer.add_scalar(f'{prefix}/episode_pickups',   self.pickups,   global_step)
        writer.add_scalar(f'{prefix}/episode_dropoffs',  self.dropoffs,  global_step)
        writer.add_scalar(f'{prefix}/episode_noops',     self.noops,     global_step)
        writer.add_scalar(f'{prefix}/episode_accepts',   self.accepts,   global_step)
        writer.add_scalar(f'{prefix}/episode_successes', self.successes, global_step)


def _unwrap_to_raw(env):
    """Navigate wrapper layers to reach the raw rideshare env."""
    raw = env
    while True:
        if hasattr(raw, 'aec_env'):
            raw = raw.aec_env
        elif hasattr(raw, 'env'):
            raw = raw.env
        else:
            break
    return raw
