from typing import List, Literal, Tuple
import re
import torch
import json
import os

from free_range_zoo.envs.wildfire.env.structures.configuration import (
    WildfireConfiguration,
    FireConfiguration,
    AgentConfiguration,
    StochasticConfiguration,
    RewardConfiguration,
)

from utils.experiment_manager import load_config
general_config = load_config("General_Config")

# Load configuration
def _load_config(file_name="setups.json"):
    _config = None
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    default_path = os.path.join(parent_dir, file_name)
        
    with open(default_path, 'r') as f:
         _config = json.load(f)
    
    return _config


# Mapping from numeric grid_size argument to the grid suffix used in setup names.
# grid_size=2 corresponds to the 2x3 family (height=2, width=3).
_GRID_SIZE_TO_SUFFIX = {
    2: "2x3",
    3: "3x3",
    4: "4x4",
    5: "5x5",
    6: "6x6",
}

def zero_shot_setting_type(experiment):
    if experiment.zero_shot:
        target = experiment.zero_shot_target_grids_size
        tgt = _GRID_SIZE_TO_SUFFIX.get(int(target), f"{target}x{target}")
        # Replace any NxM grid-size token in the setting name with the target suffix
        result = re.sub(r'\d+[xX]\d+', tgt, experiment.setting_type)
        # Append _zs suffix for zero-shot grid setups
        if getattr(experiment, 'zero_shot_mode', None) == "zs":
            result += "_zs"
        return result
    return experiment.setting_type


def zero_shot_setting_detail(experiment):
    if experiment.zero_shot:
        to_type = zero_shot_setting_type(experiment)
        mode = getattr(experiment, 'zero_shot_mode', None)
        # Don't append mode suffix for 'zs' — the _zs suffix is already in the setup name
        mode_suffix = f"_{mode}" if mode and mode != "zs" else ""
        return f"from_{experiment.setting_type}_to_{to_type}{mode_suffix}"
    return experiment.setting_type


def target_for_zero_shot(trained_config: str) -> str:
    data = _load_config("cross_exection.json")  # list of {"trained_config","target_config"}
    for item in data:
        if item["trained_config"] == trained_config:
            return item["target_config"]
    raise KeyError(trained_config)


def create_wildfire_config(
    setting_type: str,
    generator: torch.Generator = None,
    termination_reward: float = 0.0,
    zero_shot_mode: str = None,
    source_setting_type: str = None,
    target_grid_size: int = None,
) -> WildfireConfiguration:
    """
    Create a wildfire configuration.
    
    Args:
        setting_type: The setting type to use for configuration
        generator: Random generator for initialization
        termination_reward: Reward for episode termination
        zero_shot_mode: None (normal), "fixteam" (same config, different grid), 
                       or "fixteam_transition" (same + 20% harder transitions)
        source_setting_type: Original training setting type (used in fixteam modes)
        target_grid_size: Target grid size for zero-shot (used in fixteam modes)
    
    Returns:
        WildfireConfiguration object
    """

   
    
    # Check if setting_type is in configuration_pkls (load from pickle)
    if setting_type in (general_config.configuration_pkls or []):
        from utils.helper import load_configuration_pkl
        return load_configuration_pkl(setting_type)
    
    config = _load_config()
    defaults = config["default_values"]

    # Determine which setup config to use for fires/agents/stochastic settings
    if zero_shot_mode in ("fixteam", "fixteam_transition") and source_setting_type:
        # Use source setting for team configuration (fires, agents, stochastic)
        setup_config = config["setups"][source_setting_type]
        # But use target grid dimensions
        setup_grid_height = target_grid_size
        setup_grid_width = target_grid_size
    else:
        # Normal mode: use setting_type for everything
        setup_config = config["setups"][setting_type]
        setup_grid_height = setup_config["grid_dimensions"]["height"]
        setup_grid_width = setup_config["grid_dimensions"]["width"]
    
    # Fire layout and types (from source config, sized to target grid)
    fire_types, fires_initially_lit, ignition_temperatures = _build_fire_layout_from_json(
        setup_config, setup_grid_height, setup_grid_width
    )

    # Reward calculation from config
    base_fire_reward = config["default_values"]["reward"]["base_fire_reward"]
    fire_rewards = torch.pow(base_fire_reward, fire_types.float())

    # Create fire config, potentially with harder transitions
    fire_config = _create_fire_configuration_from_json(
        config, setup_config, fire_types, fires_initially_lit, ignition_temperatures,
        harder_transitions=(zero_shot_mode == "fixteam_transition")
    )

    agent_config = _create_agent_configuration_from_json(
        config, setup_config,
        harder_transitions=(zero_shot_mode == "fixteam_transition")
    )

    reward_config = RewardConfiguration(
        fire_rewards=fire_rewards,
        burnout_penalty=config["default_values"]["reward"]["burnout_penalty"],
        bad_attack_penalty=config["default_values"]["reward"]["bad_attack_penalty"],
        termination_reward=termination_reward,
        termination_kappa=config["default_values"]["reward"]["termination_kappa"],
        burnout_penalty_scaled=config["default_values"]["reward"]["burnout_penalty_scaled"],
    )

    stochastic_config = _create_stochastic_configuration_from_json(setup_config,defaults)

    return WildfireConfiguration(
        grid_width=setup_grid_width,
        grid_height=setup_grid_height,
        fire_config=fire_config,
        agent_config=agent_config,
        reward_config=reward_config,
        stochastic_config=stochastic_config,
    )


def _build_fire_layout_from_json(
    setup_config: dict,
    grid_height: int,
    grid_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    fire_types = torch.zeros((grid_height, grid_width), dtype=torch.int32)
    ignition_temperatures = torch.ones((grid_height, grid_width), dtype=torch.int32)
    
    for fire in setup_config["fires"]:
        x, y = fire["position"]
        fire_types[y, x] = fire["type"]
        ignition_temperatures[y, x] = fire["ignition_temperature"]
    
    fires_initially_lit = fire_types != 0
    
    return fire_types, fires_initially_lit, ignition_temperatures


def _create_fire_configuration_from_json(
    config: dict,
    setup_config: dict,
    fire_types: torch.Tensor,
    fires_initially_lit: torch.Tensor,
    ignition_temperatures: torch.Tensor,
    harder_transitions: bool = False,
) -> FireConfiguration:
    
    fire_defaults = config["default_values"]["fire"]
    
    # Get base values from setup config
    base_spread_rate = setup_config["base_spread_rate"]
    random_ignition_probability = setup_config["random_ignition_probability"]
    
    # Apply 20% harder transitions if requested
    if harder_transitions:
        # Increase spread rate by 20% (capped at 1.0)
        base_spread_rate = min(1.0, base_spread_rate * 1.2 + 0.1)
        # Increase random ignition probability by 20% (capped at 1.0)
        random_ignition_probability = min(1.0, random_ignition_probability * 1.2 + 0.1)
    
    return FireConfiguration(
        fire_types=fire_types,
        num_fire_states=fire_defaults["states"],
        lit=fires_initially_lit,
        intensity_increase_probability=fire_defaults["intensity_increase_prob"],
        intensity_decrease_probability=fire_defaults["intensity_decrease_prob"],
        extra_power_decrease_bonus=fire_defaults["extra_power_decrease_bonus"],
        burnout_probability=fire_defaults["burnout_prob"],
        base_spread_rate=base_spread_rate,
        max_spread_rate=fire_defaults["max_spread_rate"],
        random_ignition_probability=random_ignition_probability,
        cell_size=fire_defaults["cell_size"],
        wind_direction=fire_defaults["wind_direction"],
        ignition_temp=ignition_temperatures,
        initial_fuel=fire_defaults["initial_fuel"],      
        # initial_fuel=fire_defaults["use_fire_fuel"]   
        # use_fire_fuel=fire_defaults["use_fire_fuel"],

    )


def _create_agent_configuration_from_json(
    config: dict,
    setup_config: dict,
    harder_transitions: bool = False,
) -> AgentConfiguration:
    
    agent_defaults = config["default_values"]["agent"]
    equipment_defaults = config["default_values"]["equipment"]
    

    # Use setup config
    agents_data = setup_config["agent_configs"]
    num_agents = len(agents_data)
    # JSON position is [x, y] = [col, row]; env expects [y, x] = [row, col]
    agents_tensor = torch.tensor(
        [[agent["position"][1], agent["position"][0]] for agent in agents_data], 
        dtype=torch.int32
    )
    initial_suppressant_value = agents_data[0]["initial_suppressant"]  # Use first agent's value
    agent_powers = [agent.get("power", agent_defaults["default_power"]) for agent in agents_data]
    agent_ranges = [agent.get("range", agent_defaults["default_range"]) for agent in agents_data]

    # Get refill probability from setup config
    suppressant_refill_probability = setup_config["suppressant_refill_probability"]
    
    # Apply 20% harder transitions if requested
    if harder_transitions:
        # Decrease refill probability by 20% (minimum 0.0)
        suppressant_refill_probability = max(0.0, suppressant_refill_probability * 0.8 - 0.1)

    equipment_config = {
        "initial_state": config["equipment_states"]["default"]["initial_state"],
        "states": torch.tensor(config["equipment_states"]["default"]["states"]),
    }
    repair_prob = equipment_defaults["frame_closed_repair_prob"]
    degrade_prob = equipment_defaults["frame_closed_degrade_prob"]
    
    # If suppressant_decrease is false in stochastic config, use high values (100) to prevent depletion
    stochastic_config = setup_config["stochastic_config"]
    if not stochastic_config.get("suppressant_decrease", True):
        suppressant_decrease_prob = 0
        suppressant_levels = agent_defaults.get("0_suppressant_levels", 100)
        tank_capacity = agent_defaults.get("0_tank_capacity", 100)
        possible_capacities = agent_defaults.get("0_possible_capacities", [1.0, 2.0, 3.0, 100.0])
        capacity_probabilities = agent_defaults.get("0_capacity_probabilities", [0.0, 0.0, 0.0, 1.0])
        initial_suppressant_value = agent_defaults.get("0_default_initial_suppressant", initial_suppressant_value)
    else:
        suppressant_decrease_prob = agent_defaults["suppressant_decrease_probability"]
        suppressant_levels = agent_defaults["suppressant_levels"]
        tank_capacity = agent_defaults["tank_capacity"]
        possible_capacities = agent_defaults["possible_capacities"]
        capacity_probabilities = agent_defaults["capacity_probabilities"]

    return AgentConfiguration(
        agents=agents_tensor,
        fire_reduction_power=torch.tensor(agent_powers, dtype=torch.int32),
        attack_range=torch.tensor(agent_ranges, dtype=torch.int32),
        suppressant_states=suppressant_levels,
        initial_suppressant=initial_suppressant_value,  # Single integer
        suppressant_decrease_probability=suppressant_decrease_prob,
        suppressant_refill_probability=suppressant_refill_probability,  # May be modified for harder_transitions
        initial_equipment_state=equipment_config["initial_state"],
        equipment_states=equipment_config["states"],
        repair_probability=repair_prob,
        degrade_probability=degrade_prob,
        critical_error_probability=agent_defaults["critical_error_probability"],
        initial_capacity=tank_capacity,
        tank_switch_probability=agent_defaults["tank_switch_probability"],
        possible_capacities=torch.tensor(possible_capacities),
        capacity_probabilities=torch.tensor(capacity_probabilities),
    )


def _get_equipment_states_from_json(config: dict, grid_width: int, grid_height: int) -> dict:
    
    equipment_states = config["equipment_states"]
    
    size_key = f"{grid_width}x{grid_height}"
    if size_key in equipment_states:
        equipment_config = equipment_states[size_key]
    else:
        equipment_config = equipment_states["default"]
    
    return {
        "initial_state": equipment_config["initial_state"],
        "states": torch.tensor(equipment_config["states"]),
    }


def _create_stochastic_configuration_from_json(
    setup_config: dict,
    defaults: dict,
) -> StochasticConfiguration:
    
    stochastic_config = setup_config["stochastic_config"]
    fire_defaults = defaults["fire"]

    return StochasticConfiguration(
        special_burnout_probability=stochastic_config["special_burnout_probability"],
        suppressant_refill=stochastic_config["suppressant_refill"],
        suppressant_decrease=stochastic_config["suppressant_decrease"],
        tank_switch=stochastic_config["tank_switch"],
        critical_error=stochastic_config["critical_error"],
        degrade=False,
        repair=False,
        fire_increase=stochastic_config["fire_increase"],
        fire_decrease=stochastic_config["fire_decrease"],
        fire_spread=stochastic_config["fire_spread"],
        realistic_fire_spread=stochastic_config["realistic_fire_spread"],
        random_fire_ignition=stochastic_config["random_fire_ignition"],
        # fire_fuel=1000,
        # fire_fuel=stochastic_config["fire_fuel"],
        fire_fuel=stochastic_config["fire_fuel"],
    )