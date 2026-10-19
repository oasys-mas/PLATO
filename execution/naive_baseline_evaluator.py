"""
Naive baseline evaluator implementation.
Supports both wildfire and rideshare domains.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Type
from copy import deepcopy

from base_evaluator import BaseEvaluator
from free_range_zoo.envs.wildfire.baselines import (
    NoopBaseline as WildfireNoop,
    RandomBaseline as WildfireRandom,
    StrongestBaseline,
    WeakestBaseline,
)
from free_range_zoo.envs.rideshare.baselines import (
    NoopBaseline as RideshareNoop,
    RandomBaseline as RideshareRandom,
)
try:
    from free_range_zoo.envs.rideshare.baselines import (
        FirstInFirstOutTfocusBaseline,
        FirstInFirstOutTglobalBaseline,
    )
except ImportError:
    from free_range_zoo.envs.rideshare.baselines import FirstInFirstOutBaseline as FirstInFirstOutTfocusBaseline
    FirstInFirstOutTglobalBaseline = FirstInFirstOutTfocusBaseline
try:
    from free_range_zoo.envs.rideshare.baselines import GreedyTaskFocus, GreedyTaskGlobal
except ImportError:
    from free_range_zoo.envs.rideshare.baselines import GreedyBaseline as GreedyTaskFocus
    GreedyTaskGlobal = GreedyTaskFocus
from utils.experiment_manager import load_config
eval_config = load_config("Eval")
general_config = load_config("General_Config")


class NaiveBaselineEvaluator(BaseEvaluator):
    """
    Evaluator for naive baseline algorithms.
    Wildfire:  NOOP, Random, Strongest, Weakest
    Rideshare: NOOP, Random, FIFO, GreedyTFocus, GreedyTGlobal
    """

    # Domain-specific registries
    WILDFIRE_BASELINES = {
        'NOOP': WildfireNoop,
        'Random': WildfireRandom,
        'Strongest': StrongestBaseline,
        'Weakest': WeakestBaseline,
    }

    RIDESHARE_BASELINES = {
        'NOOP': RideshareNoop,
        'Random': RideshareRandom,
        'FIFOTfocus': FirstInFirstOutTfocusBaseline,
        'FIFOTglobal': FirstInFirstOutTglobalBaseline,
        'GreedyTFocus': GreedyTaskFocus,
        'GreedyTGlobal': GreedyTaskGlobal,
    }

    def __init__(self, experiment, shared_db=None):
        """
        Initialize naive baseline evaluator.

        Args:
            experiment: Experiment object with algorithm (baseline name) and domain
            shared_db: Optional shared database connection to reuse
        """
        self.baseline_name = experiment.algorithm
        domain = getattr(experiment, 'domain', 'wildfire')

        # Pick registry for this domain
        registry = self.RIDESHARE_BASELINES if domain == 'rideshare' else self.WILDFIRE_BASELINES
        if self.baseline_name not in registry:
            raise ValueError(
                f"Unsupported baseline '{self.baseline_name}' for domain '{domain}'. "
                f"Available: {list(registry.keys())}"
            )

        self.baseline_class = registry[self.baseline_name]
        self._configuration = None  # cached for rideshare greedy baselines
        super().__init__(experiment, shared_db=shared_db)

    # ------------------------------------------------------------------ env --
    def _init_environment(self):
        """Initialize environment based on domain."""
        if self.domain == 'rideshare':
            return self._init_rideshare_env_naive()
        return self._init_wildfire_env_naive()

    def _init_rideshare_env_naive(self):
        """Rideshare env for naive baselines (caches configuration for greedy agents)."""
        from utils.helper import load_configuration_pkl
        from free_range_zoo.envs import rideshare_v0
        from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
        from utils.rideshare_capacity_filter import apply_capacity_filter

        configuration = load_configuration_pkl(self.experiment.setting_type)
        self._configuration = configuration  # cache for greedy baseline agents

        env = rideshare_v0.parallel_env(
            configuration=configuration,
            parallel_envs=1,
            max_steps=self.general_config.MAX_STEPS,
            device=self.device,
            single_seeding=False,
        )
        env.reset()
        apply_capacity_filter(env)
        return action_mapping_wrapper_v0(env)

    def _init_wildfire_env_naive(self):
        """Wildfire env for naive baselines (simpler than smart wildfire init)."""
        configuration = self._load_configuration()

        from free_range_zoo.envs import wildfire_v0
        from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0

        env = wildfire_v0.parallel_env(
            configuration=configuration,
            parallel_envs=1,
            max_steps=self.general_config.MAX_STEPS,
            device=self.device,
            single_seeding=True,
            show_bad_actions=False,
            observe_other_power=True,
            observe_other_suppressant=True,
            override_initialization_check=True,
        )
        return action_mapping_wrapper_v0(env)

    def _load_configuration(self):
        """Load environment configuration (domain-aware)."""
        if self.domain == 'rideshare':
            from utils.helper import load_configuration_pkl
            cfg = load_configuration_pkl(self.experiment.setting_type)
            self._configuration = cfg
            return cfg
        else:
            from utils.helper import load_configuration
            return load_configuration(self.experiment.setting_type, self.experiment.grid_size)

    # -------------------------------------------------------------- models --
    def _load_models(self) -> Dict:
        """No trained models to load for naive baselines."""
        return {'baseline_class': self.baseline_class}

    # ------------------------------------------------------------- actions --
    def _create_baseline_agent(self, agent_name):
        """Instantiate one baseline agent (handles greedy kwargs)."""
        if self.baseline_class in (GreedyTaskFocus, GreedyTaskGlobal):
            return self.baseline_class(
                agent_name=agent_name,
                parallel_envs=1,
                agent_configuration=self._configuration.agent_config,
            )
        return self.baseline_class(agent_name=agent_name, parallel_envs=1)

    def _get_actions(self, observations, infos) -> Tuple[Dict, Dict]:
        """Get actions for all agents. Domain-aware stats tracking."""
        if self.domain == 'rideshare':
            return self._get_actions_rideshare(observations)
        return self._get_actions_wildfire(observations)

    def _get_actions_wildfire(self, observations) -> Tuple[Dict, Dict]:
        actions_dict = {}
        action_stats = {'noop': 0, 'fight': 0, 'total': 0}

        for agent_name in self.agents:
            agent = self._create_baseline_agent(agent_name)
            agent.observe(observations[agent_name])
            action_space = self.env.action_space(agent_name)
            agent_actions = agent.act(action_space)

            agent_actions = self._to_tensor(agent_actions)
            actions_dict[agent_name] = agent_actions

            action_type = agent_actions[0, 1].item() if agent_actions.shape[1] >= 2 else -1
            action_stats['total'] += 1
            if action_type == -1:
                action_stats['noop'] += 1
            elif action_type == 0:
                action_stats['fight'] += 1

        return actions_dict, action_stats

    def _get_actions_rideshare(self, observations) -> Tuple[Dict, Dict]:
        actions_dict = {}
        action_stats = {'noop': 0, 'accept': 0, 'pickup': 0, 'dropoff': 0, 'total': 0}

        for agent_name in self.agents:
            agent = self._create_baseline_agent(agent_name)
            agent.observe(observations[agent_name])
            action_space = self.env.action_space(agent_name)
            agent_actions = agent.act(action_space)

            agent_actions = self._to_tensor(agent_actions)
            actions_dict[agent_name] = agent_actions

            action_stats['total'] += 1

            # Rideshare action format: [[task_idx, action_type]]
            if agent_actions.shape[1] >= 2:
                task_idx = agent_actions[0, 0].item()
                action_type = agent_actions[0, 1].item()

                if task_idx == -1 or action_type == -1:
                    action_stats['noop'] += 1
                elif action_type == 0:
                    action_stats['accept'] += 1
                elif action_type == 1:
                    action_stats['pickup'] += 1
                elif action_type == 2:
                    action_stats['dropoff'] += 1
            else:
                action_stats['noop'] += 1

        return actions_dict, action_stats

    # ------------------------------------------------------------ helpers --
    def _to_tensor(self, agent_actions) -> torch.Tensor:
        """Convert baseline output to 2-D int32 tensor on device."""
        if isinstance(agent_actions, (list, tuple)):
            agent_actions = torch.tensor(agent_actions, device=self.device, dtype=torch.int32)
        elif not isinstance(agent_actions, torch.Tensor):
            agent_actions = torch.tensor([agent_actions], device=self.device, dtype=torch.int32)
        else:
            agent_actions = agent_actions.to(self.device)
        if agent_actions.dim() == 1:
            agent_actions = agent_actions.unsqueeze(0)
        return agent_actions

    # ------------------------------------------------------------ episode --
    def run_episode(self, seed=None):
        """
        Baselines need a fresh environment each episode.
        Re-init env, reset metric tracker for rideshare, then delegate to base.
        """
        if seed is not None:
            from utils.helper import set_global_seed
            set_global_seed(seed)

        # Reinitialize environment for this episode
        self.env = self._init_environment()
        observations, _ = self.env.reset()
        self.agents = self.env.agents

        # Rideshare: create a fresh metric tracker for the new env
        if self.domain == 'rideshare':
            from utils.rideshare_capacity_filter import RideshareMetricTracker
            self.metric_tracker = RideshareMetricTracker(self.env)

        return super().run_episode(seed)


    # ---------------------------------------------------------- metadata --
    def _create_experiment_metadata(self) -> Dict:
        """Create experiment metadata for baseline evaluation."""
        return {
            'run_type': "execution",
            'model_family': "Baseline",
            'model_type': "naive",
            'name': self.baseline_name,
            'variation1': self.baseline_name,
            'variation2': "default",
            'grid_size': self.experiment.grid_size,
            'setting_type': self.experiment.setting_type,
            'seed': getattr(self.experiment, 'seed', None),
            'parent_experiment_id': None,
            'checkpoint_path': "",
            'domain': self.domain,
            'notes': "coine4",
        }

    def _get_model_family(self) -> str:
        return "Baseline"

    def _get_variation1(self) -> str:
        return self.baseline_name

    def _get_variation2(self) -> str:
        return "default"


# ============================================================ convenience ==
def create_baseline_experiment(baseline_name: str, setting_type: str, grid_size: int, domain: str = 'wildfire'):
    """
    Create a mock experiment object for baseline evaluation.

    Args:
        baseline_name: Name of baseline (e.g. NOOP, Random, FIFO, GreedyTFocus …)
        setting_type: Environment setting type
        grid_size: Grid size
        domain: 'wildfire' or 'rideshare'

    Returns:
        Experiment-like object for baseline evaluation
    """

    class BaselineExperiment:
        def __init__(self, baseline_name, setting_type, grid_size, domain):
            self.algorithm = baseline_name
            self.name = f"{baseline_name}_{setting_type}_{grid_size}x{grid_size}"
            self.setting_type = setting_type
            self.grid_size = grid_size
            self.checkpoint_path = ""
            self.episode = "0"
            self.seed = None
            self.domain = domain
            self.zero_shot = False

    return BaselineExperiment(baseline_name, setting_type, grid_size, domain)