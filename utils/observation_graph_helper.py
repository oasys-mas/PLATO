from enum import Enum
from typing import Dict
import tensordict
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
# Observation Vector Helper

import torch
from enum import Enum

from utils.helper import shuffle_task_features


def create_global_bipartite_graph_rideshare(num_parallel_envs: int, state, device, 
                                             grid_size: tuple = (5, 5)): 
    """
    Bipartite graph creation for rideshare environments.
    Creates graphs with agents and tasks (passengers) as two node types.
    
    Mirrors wildfire's graph structure with:
    - NOOP task node with sentinel features
    - Bidirectional edges (agent<->task) for GNN message passing
    - Agent-Agent edges for coordination
    - Normalized features
    
    Agent Node Features (5 features):
        [x_norm, y_norm, current_passengers, available_capacity, capacity_ratio]
        
    Task Node Features (7 features):
        [start_x_norm, start_y_norm, dest_x_norm, dest_y_norm, fare_norm, wait_time, status]
    
    NOOP Features: [-1, -1, -1, -1, 0, 0, 0] to distinguish from real tasks
    
    Edge Structure (unweighted, bidirectional):
        - Agent-Task: All agents connected to all tasks (bipartite, both directions)
        - Agent-Agent: All agents connected to each other (for coordination)
        - Agent-NOOP: All agents connected to NOOP node (both directions)
    """
    all_graphs = []
    dtype = torch.float32
    grid_h, grid_w = grid_size
    
    for env_id in range(num_parallel_envs):
        # ============================================================
        # Extract agent information
        # ============================================================
        if len(state.agents.shape) == 3:
            agents_positions = state.agents[env_id].to(device, non_blocking=True)
        else:
            agents_positions = state.agents.to(device, non_blocking=True)
        
        num_agents = agents_positions.shape[0]
        
        # Agent features with normalization
        agent_x = agents_positions[:, 0].float()
        agent_y = agents_positions[:, 1].float()
        
        # Normalized positions [0, 1]
        x_norm_agent = agent_x / max(grid_h - 1, 1)
        y_norm_agent = agent_y / max(grid_w - 1, 1)
        
        # Derived agent features
        current_passengers = torch.zeros(num_agents, dtype=dtype, device=device)
        max_capacity = 2.0
        available_capacity = torch.full((num_agents,), max_capacity, dtype=dtype, device=device)
        capacity_ratio = (available_capacity - current_passengers) / max(max_capacity, 1.0)
        
        agent_nodes = torch.stack([
            x_norm_agent, y_norm_agent, current_passengers, available_capacity, capacity_ratio
        ], dim=1)  # [num_agents, 5]
        
        # ============================================================
        # Handle tasks (passengers)
        # ============================================================
        num_tasks = state.passengers.shape[0] if state.passengers.numel() > 0 else 0
        
        if num_tasks > 0:
            if len(state.passengers.shape) >= 2 and state.passengers.shape[0] >= num_parallel_envs:
                try:
                    tasks_tensor = state.passengers[env_id].to(device, non_blocking=True)
                    num_tasks = tasks_tensor.shape[0]
                except IndexError:
                    tasks_tensor = torch.empty(0, 11, dtype=torch.int32, device=device)
                    num_tasks = 0
            else:
                tasks_tensor = state.passengers.to(device, non_blocking=True)
                num_tasks = tasks_tensor.shape[0]
        else:
            tasks_tensor = torch.empty(0, 11, dtype=torch.int32, device=device)
            num_tasks = 0
        
        # Create task nodes with normalized features
        if num_tasks > 0 and len(tasks_tensor.shape) >= 2 and tasks_tensor.shape[1] >= 7:
            task_start_x = tasks_tensor[:, 3].float()
            task_start_y = tasks_tensor[:, 2].float()
            task_dest_x = tasks_tensor[:, 5].float()
            task_dest_y = tasks_tensor[:, 4].float()
            task_fare = tasks_tensor[:, 6].float()
            
            # Normalize positions
            task_start_x_norm = task_start_x / max(grid_h - 1, 1)
            task_start_y_norm = task_start_y / max(grid_w - 1, 1)
            task_dest_x_norm = task_dest_x / max(grid_h - 1, 1)
            task_dest_y_norm = task_dest_y / max(grid_w - 1, 1)
            
            # Normalize fare using per-env max
            max_fare = torch.clamp(task_fare.max(), min=1.0)
            fare_norm = task_fare / max_fare
            
            wait_time = torch.zeros(num_tasks, dtype=dtype, device=device)
            status = torch.zeros(num_tasks, dtype=dtype, device=device)
            
            task_nodes = torch.stack([
                task_start_x_norm, task_start_y_norm, 
                task_dest_x_norm, task_dest_y_norm,
                fare_norm, wait_time, status
            ], dim=1)  # [num_tasks, 7]
        else:
            task_nodes = torch.empty(0, 7, dtype=dtype, device=device)
            num_tasks = 0
        
        # ============================================================
        # NOOP task node (always last, 7 features to match task nodes)
        # Sentinel values [-1, -1, -1, -1, 0, 0, 0] to distinguish from real tasks
        # ============================================================
        noop_features = torch.tensor([[-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0]], 
                                     dtype=dtype, device=device)
        
        # Combine real tasks + NOOP
        all_task_nodes = torch.cat([task_nodes, noop_features], dim=0) if num_tasks > 0 else noop_features
        num_all_tasks = num_tasks + 1  # includes NOOP
        
        # ============================================================
        # EDGE CONSTRUCTION (unweighted, bidirectional)
        # ============================================================
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        noop_idx = num_agents + num_tasks  # NOOP is last task node
        
        if num_agents > 0:
            agent_indices = torch.arange(num_agents, device=device, dtype=torch.long)
            
            # 1. Agent-Task edges (bipartite, all-to-all, BIDIRECTIONAL)
            if num_tasks > 0:
                task_indices = torch.arange(num_tasks, device=device, dtype=torch.long)
                agent_idx_expanded = agent_indices.unsqueeze(1).expand(-1, num_tasks).flatten()
                task_idx_expanded = task_indices.unsqueeze(0).expand(num_agents, -1).flatten()
                task_idx_offset = task_idx_expanded + num_agents
                
                # Both directions
                edge_index_agent_task = torch.stack([agent_idx_expanded, task_idx_offset], dim=0)
                edge_index_task_agent = torch.stack([task_idx_offset, agent_idx_expanded], dim=0)
                edge_index = torch.cat([edge_index, edge_index_agent_task, edge_index_task_agent], dim=1)
            
            # 2. Agent-NOOP edges (all agents to NOOP, bidirectional)
            edge_index_agent_noop = torch.stack([
                agent_indices, 
                torch.full((num_agents,), noop_idx, device=device, dtype=torch.long)
            ], dim=0)
            edge_index_noop_agent = torch.stack([
                torch.full((num_agents,), noop_idx, device=device, dtype=torch.long),
                agent_indices
            ], dim=0)
            edge_index = torch.cat([edge_index, edge_index_agent_noop, edge_index_noop_agent], dim=1)
            
            # 3. Agent-Agent edges (for coordination, all-to-all excluding self)
            if num_agents > 1:
                agent_pairs_src = []
                agent_pairs_dst = []
                for i in range(num_agents):
                    for j in range(num_agents):
                        if i != j:
                            agent_pairs_src.append(i)
                            agent_pairs_dst.append(j)
                if agent_pairs_src:
                    edge_index_agent_agent = torch.tensor(
                        [agent_pairs_src, agent_pairs_dst], 
                        dtype=torch.long, device=device
                    )
                    edge_index = torch.cat([edge_index, edge_index_agent_agent], dim=1)
        
        # ============================================================
        # Build graph data
        # ============================================================
        graph_data = {
            'agent_nodes': agent_nodes,           # [num_agents, 5]
            'task_nodes': all_task_nodes,          # [num_tasks + 1, 7] (includes NOOP)
            'fire_nodes': task_nodes,              # [num_tasks, 7] (without NOOP, for compatibility)
            'edge_index': edge_index,              # [2, num_edges] bidirectional
            'num_agents': num_agents,
            'num_tasks': num_all_tasks,            # includes NOOP
            'num_fires': num_tasks,                # without NOOP, for compatibility
            'num_passengers': num_tasks            # without NOOP, for compatibility
        }
        
        all_graphs.append(graph_data)
    
    return all_graphs

def create_global_bipartite_graph_wildfire(num_parallel_envs: int, wildfire_state, device, 
                                                grid_size: tuple = (3, 3)): 
    """
    ENRICHED bipartite graph creation WITHOUT distance weights or global context.
    
    Key Design: Instead of edge weights and global context, we embed MORE INFORMATION
    directly into the node features. The GCN can then learn to extract the relevant
    patterns through message passing.
    
    NO distance-weighted edges - all edges are unweighted (binary adjacency)
    NO global context vector - information is embedded in node features
    
    Agent Node Features (11 features):
        [x, y, suppressant, capacity, equipment,
         suppressant_ratio,    # suppressant / max(capacity, 1) - resource state
         is_empty,             # 1 if suppressant <= 0, else 0 - needs refill
         is_full,              # 1 if suppressant >= capacity, else 0 - ready to fight
         x_norm, y_norm,       # normalized position [0,1] based on grid size
         has_equipment]        # 1 if equipment > 0, else 0
    
    Fire/Task Node Features (9 features):
        [x, y, intensity, fuel,
         intensity_norm,       # intensity / max_possible_intensity (assume 10)
         fuel_norm,            # fuel / max_possible_fuel (assume 10)
         is_critical,          # 1 if intensity >= 3, else 0 - high priority
         x_norm, y_norm]       # normalized position [0,1]
    
    NOOP Features: [-1, -1, 0, 0, 0, 0, 0, -1, -1] to distinguish from real fires
    
    Edge Structure (unweighted):
        - Agent-Fire: All agents connected to all fires (bipartite)
        - Fire-Fire: 8-neighbor adjacency (for spread structure)
        - Agent-Agent: All agents connected to each other (for coordination)
        - NOOP: Connected to all agents
    
    Args:
        num_parallel_envs: Number of parallel environments
        wildfire_state: WildfireState object with fires, intensity, fuel, agents, etc.
        device: Torch device
        grid_size: (rows, cols) of the environment grid for normalization
    """
    all_graphs = []
    
    dtype = torch.float32
    grid_h, grid_w = grid_size
    # NOTE:
    # The wildfire env's raw `fuel` can be very large (e.g. 1e4), so using a fixed
    # max like 10.0 will explode feature magnitudes and destabilize the critic.
    # We therefore normalize using per-environment maxima (clamped) to keep
    # features roughly in [0, 1].
    
    for env_id in range(num_parallel_envs):
        # ============================================================
        # Extract raw state from environment
        # ============================================================
        fires_tensor = wildfire_state.fires[env_id].squeeze(0).to(device, non_blocking=True)
        intensity_tensor = wildfire_state.intensity[env_id].squeeze(0).to(device, non_blocking=True)
        fuel_tensor = wildfire_state.fuel[env_id].squeeze(0).to(device, non_blocking=True)

        # Per-env normalization constants (avoid divide-by-zero)
        max_intensity = torch.clamp(intensity_tensor.max().float(), min=1.0)
        max_fuel = torch.clamp(fuel_tensor.max().float(), min=1.0)
        
        # Agent state
        agents_positions = wildfire_state.agents.to(device, non_blocking=True)
        suppressants = wildfire_state.suppressants[env_id].squeeze(0).to(device, non_blocking=True)
        capacity = wildfire_state.capacity[env_id].squeeze(0).to(device, non_blocking=True)
        equipment = wildfire_state.equipment[env_id].squeeze(0).to(device, non_blocking=True)
        
        num_agents = agents_positions.shape[0]
        
        # ============================================================
        # ENRICHED AGENT NODE FEATURES (11 features)
        # state.agents: [row, col] (env convention: [y, x])
        # ============================================================
        agent_row = agents_positions[:, 0].float()  # row (y)
        agent_col = agents_positions[:, 1].float()  # col (x)
        agent_suppressant = suppressants.float()
        agent_capacity = capacity.float()
        agent_equipment = equipment.float()
        
        # Derived features (no external calculations, just from env.state())
        suppressant_ratio = agent_suppressant / torch.clamp(agent_capacity, min=1.0)
        is_empty = (agent_suppressant <= 0).float()
        is_full = (agent_suppressant >= agent_capacity).float()
        x_norm_agent = agent_col / max(grid_w - 1, 1)  # col / width
        y_norm_agent = agent_row / max(grid_h - 1, 1)  # row / height
        has_equipment = (agent_equipment > 0).float()
        
        agent_nodes = torch.stack([
            agent_row,
            agent_col,
            agent_suppressant,
            agent_capacity,
            agent_equipment,
            suppressant_ratio,
            is_empty,
            is_full,
            x_norm_agent,
            y_norm_agent,
            has_equipment
        ], dim=1)  # [num_agents, 11]
        
        # ============================================================
        # ENRICHED FIRE NODE FEATURES (9 features)
        # nonzero() on [H,W] gives [row, col]
        # ============================================================
        fire_mask = fires_tensor != 0
        fire_locations = torch.nonzero(fire_mask, as_tuple=False)
        num_fires = fire_locations.shape[0]
        
        if num_fires > 0:
            fire_row = fire_locations[:, 0].float()
            fire_col = fire_locations[:, 1].float()
            fire_intensity = intensity_tensor[fire_locations[:, 0], fire_locations[:, 1]].float()
            fire_fuel = fuel_tensor[fire_locations[:, 0], fire_locations[:, 1]].float()
            
            # Derived features
            # Scale raw magnitudes into a stable range.
            fire_intensity_scaled = fire_intensity / max_intensity
            fire_fuel_scaled = fire_fuel / max_fuel
            intensity_norm = fire_intensity_scaled
            fuel_norm = fire_fuel_scaled
            is_critical = (fire_intensity >= 3.0).float()
            x_norm_fire = fire_col / max(grid_w - 1, 1)  # col / width
            y_norm_fire = fire_row / max(grid_h - 1, 1)  # row / height
            
            fire_nodes = torch.stack([
                fire_row,
                fire_col,
                fire_intensity_scaled,
                fire_fuel_scaled,
                intensity_norm,
                fuel_norm,
                is_critical,
                x_norm_fire,
                y_norm_fire
            ], dim=1)  # [num_fires, 9]
            
            fire_positions = torch.stack([fire_row, fire_col], dim=1)
        else:
            fire_nodes = torch.empty(0, 9, dtype=dtype, device=device)
            fire_positions = torch.empty(0, 2, dtype=dtype, device=device)
        
        # ============================================================
        # NOOP task node (always last, 9 features to match fire nodes)
        # ============================================================
        noop_features = torch.tensor([[-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0]], 
                                     dtype=dtype, device=device)
        
        task_nodes = torch.cat([fire_nodes, noop_features], dim=0) if num_fires > 0 else noop_features
        num_tasks = num_fires + 1
        
        # ============================================================
        # UNWEIGHTED EDGE CONSTRUCTION
        # ============================================================
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        noop_idx = num_agents + num_fires  # NOOP is last task node
        
        if num_agents > 0:
            agent_indices = torch.arange(num_agents, device=device, dtype=torch.long)
            
            # 1. Agent-Fire edges (bipartite, all-to-all)
            if num_fires > 0:
                fire_indices = torch.arange(num_fires, device=device, dtype=torch.long)
                agent_idx_expanded = agent_indices.unsqueeze(1).expand(-1, num_fires).flatten()
                fire_idx_expanded = fire_indices.unsqueeze(0).expand(num_agents, -1).flatten()
                fire_idx_offset = fire_idx_expanded + num_agents
                
                edge_index_agent_fire = torch.stack([agent_idx_expanded, fire_idx_offset], dim=0)
                edge_index_fire_agent = torch.stack([fire_idx_offset, agent_idx_expanded], dim=0)
                edge_index = torch.cat([edge_index, edge_index_agent_fire, edge_index_fire_agent], dim=1)
            
            # 2. Agent-NOOP edges (all agents to NOOP)
            edge_index_agent_noop = torch.stack([agent_indices, 
                                                  torch.full((num_agents,), noop_idx, device=device, dtype=torch.long)], dim=0)
            edge_index_noop_agent = torch.stack([torch.full((num_agents,), noop_idx, device=device, dtype=torch.long),
                                                  agent_indices], dim=0)
            edge_index = torch.cat([edge_index, edge_index_agent_noop, edge_index_noop_agent], dim=1)
            
            # 3. Agent-Agent edges (for coordination - all-to-all, excluding self)
            if num_agents > 1:
                agent_pairs_src = []
                agent_pairs_dst = []
                for i in range(num_agents):
                    for j in range(num_agents):
                        if i != j:
                            agent_pairs_src.append(i)
                            agent_pairs_dst.append(j)
                if agent_pairs_src:
                    edge_index_agent_agent = torch.tensor([agent_pairs_src, agent_pairs_dst], 
                                                          dtype=torch.long, device=device)
                    edge_index = torch.cat([edge_index, edge_index_agent_agent], dim=1)
        
        # 4. Fire-Fire edges (8-neighbor adjacency for spread structure)
        if num_fires > 1:
            fire_pos_i = fire_positions.unsqueeze(1)  # [F, 1, 2]
            fire_pos_j = fire_positions.unsqueeze(0)  # [1, F, 2]
            fire_distances = torch.sqrt(((fire_pos_i - fire_pos_j) ** 2).sum(dim=2))
            
            # 8-neighbor: distance <= sqrt(2) ≈ 1.414
            adjacency_mask = (fire_distances > 0) & (fire_distances <= 1.5)
            fire_adj_pairs = torch.nonzero(adjacency_mask, as_tuple=False)
            
            if fire_adj_pairs.shape[0] > 0:
                fire_fire_src = fire_adj_pairs[:, 0] + num_agents
                fire_fire_dst = fire_adj_pairs[:, 1] + num_agents
                edge_index_fire_fire = torch.stack([fire_fire_src, fire_fire_dst], dim=0)
                edge_index = torch.cat([edge_index, edge_index_fire_fire], dim=1)
        
        # ============================================================
        # Build graph data (NO edge_weight, NO global_features)
        # ============================================================
        graph_data = {
            'task_nodes': task_nodes,     # fires + NOOP [num_tasks, 9]
            'fire_nodes': fire_nodes,     # just fires [num_fires, 9]
            'agent_nodes': agent_nodes,   # [num_agents, 11]
            'edge_index': edge_index,     # unweighted edges
            'num_fires': num_fires,
            'num_tasks': num_tasks,
            'num_agents': num_agents
        }
        
        all_graphs.append(graph_data)
    
    return all_graphs


class point_type_enum(Enum):
    only_tasks = 1
    tasks_with_self = 2
    everything = 3


def get_available_actions_for_agent(agents_actions):
    """
    Optimized action extraction with better error handling and efficiency.
    """
    # Pre-allocate result list
    available = [(-1, -1, 0)]  # NOOP action first
    
    # Handle different tensor shapes efficiently
    if agents_actions.ndim == 1:
        # Vectorized processing for 1D tensor
        action_values = agents_actions.tolist()
        available.extend([(val, 0, idx + 1) for idx, val in enumerate(action_values)])
    
    elif agents_actions.ndim == 2:
        # Optimized batch processing
        flattened_actions = agents_actions.flatten().tolist()
        available.extend([(val, 0, idx + 1) for idx, val in enumerate(flattened_actions)])
    
    elif agents_actions.ndim == 0:
        # Handle scalar case
        raise ValueError(f"Expected 1D or 2D tensor for agent actions, got scalar: {agents_actions.item()}")
    
    else:
        raise ValueError(f"Unsupported tensor shape: {agents_actions.shape}")
    
    return available


def extract_valid_actions_from_action_space(action_space, env_id=0):
    """
    Optimized action space extraction with caching and better error handling.
    """
    valid_actions = []

    try:
        # Cache action space access
        env_action_space = action_space.spaces[env_id]
        
        # Vectorized action extraction
        for index,discrete_space in enumerate(env_action_space.spaces):
            if hasattr(discrete_space, 'start') and hasattr(discrete_space, 'n'):
                start = discrete_space.start
                n = discrete_space.n
                if start==-1:
                    continue
                valid_actions.append(index)
                # Generate range efficiently
                # action_range = range(start, start + n)
                # valid_actions.extend([action for action in action_range if action != -1])
    
    except (AttributeError, IndexError) as e:
        # More specific error logging
        print(f"Warning: Could not extract actions from action space for env {env_id}: {e}")
        return []
    
    # Remove duplicates efficiently and sort
    return sorted(set(valid_actions))


def observations_to_features_v2(num_parallel_envs: int, observations, point_type, env, only_available_actions, device,shuffle_tasks=False):
    """
    Observation to feature conversion with batching and caching.
    """
    all_features = []
    # Cache device and dtype for efficiency
    dtype = torch.float32
    
    for env_id in range(num_parallel_envs):
        env_features = {}
        
        for agent, obs in observations.items():
            # Optimized valid task indices extraction
            valid_task_indices = []
            
            if only_available_actions:
                try:
                    agent_action_space = env.action_space(agent)
                    # if "2" in agent:
                    #     x=2
                    valid_task_indices = extract_valid_actions_from_action_space(agent_action_space, env_id)
                    # if len(valid_task_indices) > 0:
                    #     print(f"Agent {agent} Env {env_id}: Valid task indices: {valid_task_indices}")
                except Exception as e:
                    print(f"Warning: Could not get action space for agent {agent}: {e}")
                    valid_task_indices = []
            
            # Optimized task feature processing
            tasks = obs['tasks'][env_id]
            task_features = []
            
            if only_available_actions and valid_task_indices:
                # Vectorized task filtering
                if len(tasks) > 0:
                    for task_index in valid_task_indices:
                        if task_index < len(tasks):
                            task = tasks[task_index]
                            
                            # Efficient task conversion
                            if isinstance(task, torch.Tensor):
                                task_list = task.tolist()
                            else:
                                task_list = list(task)
                            
                            # Pad to 4 elements efficiently
                            task_list = (task_list + [-1] * 4)[:4]
                            task_features.append(task_list)
            else:
                # Process all tasks efficiently
                if len(tasks) > 0:
                    for task in tasks:
                        if isinstance(task, torch.Tensor):
                            task_list = task.tolist()
                        else:
                            task_list = list(task)
                        
                        # Pad to 4 elements efficiently
                        task_list = (task_list + [-1] * 4)[:4]
                        task_features.append(task_list)

                
            # # Always add NOOP action at the end
            # task_features.append([0, 0, 0, 0])
            
            # Optimized feature tensor creation
            task_features_tensor = torch.tensor(task_features, dtype=dtype, device=device)
            if shuffle_tasks:
                task_features_tensor=shuffle_task_features(task_features_tensor)
            
            
            # Always add NOOP action at the end
            noop = torch.zeros((1, task_features_tensor.size(1)),
                            dtype=task_features_tensor.dtype,
                            device=task_features_tensor.device)

            # concat along rows
            task_features_tensor = torch.cat((task_features_tensor, noop), dim=0)

            # Efficient self features processing
            self_features = obs['self'][env_id] if (
                point_type == point_type_enum.tasks_with_self.value or 
                point_type == point_type_enum.everything.value
            ) else []
            
            self_features_tensor = _convert_to_tensor(self_features, dtype, device)
            
            # Efficient others features processing
            others_features = obs['others'][env_id] if point_type == point_type_enum.everything.value else []
            others_features_tensor = _convert_to_tensor(others_features, dtype, device)
            
            # Debug information (only if needed)
            # if len(task_features) > 1:  # More than just NOOP
            #     print(f"Agent {agent} Env {env_id}: Task features shape: {task_features_tensor.shape}")
            
            env_features[agent] = {
                'tasks': task_features_tensor,
                'self': self_features_tensor,
                'others': others_features_tensor,
                'valid_task_indices': valid_task_indices,
                'total_tasks': len(task_features)  +1 
            }
        
        all_features.append(env_features)
    
    return all_features



def observations_to_features(num_parallel_envs: int, observations, point_type, env, only_available_actions, device,shuffle_tasks):
    """
    Observation to feature conversion with batching and caching.
    """
    all_features = []
    # Cache device and dtype for efficiency
    dtype = torch.float32
    
    for env_id in range(num_parallel_envs):
        env_features = {}
        
        for agent, obs in observations.items():
            # Optimized valid task indices extraction
            valid_task_indices = []
            
            if only_available_actions:
                try:
                    agent_action_space = env.action_space(agent)
                    # if "2" in agent:
                    #     x=2
                    valid_task_indices = extract_valid_actions_from_action_space(agent_action_space, env_id)
                    # if len(valid_task_indices) > 0:
                    #     print(f"Agent {agent} Env {env_id}: Valid task indices: {valid_task_indices}")
                except Exception as e:
                    print(f"Warning: Could not get action space for agent {agent}: {e}")
                    valid_task_indices = []
            
            # Optimized task feature processing
            tasks = obs['tasks'][env_id]
            task_features = []
            
            if only_available_actions and valid_task_indices:
                # Vectorized task filtering
                if len(tasks) > 0:
                    for task_index in valid_task_indices:
                        if task_index < len(tasks):
                            task = tasks[task_index]
                            
                            # Efficient task conversion
                            if isinstance(task, torch.Tensor):
                                task_list = task.tolist()
                            else:
                                task_list = list(task)
                            
                            # Pad to 4 elements efficiently
                            task_list = (task_list + [-1] * 4)[:4]
                            task_features.append(task_list)
            else:
                # Process all tasks efficiently
                if len(tasks) > 0:
                    for task in tasks:
                        if isinstance(task, torch.Tensor):
                            task_list = task.tolist()
                        else:
                            task_list = list(task)
                        
                        # Pad to 4 elements efficiently
                        task_list = (task_list + [-1] * 4)[:4]
                        task_features.append(task_list)
            
            # Optimized feature tensor creation
            if len(task_features) > 0:
                task_features_tensor = torch.tensor(task_features, dtype=dtype, device=device)
                feat_dim = task_features_tensor.shape[1]
            else:
                task_features_tensor = torch.empty(0, 4, dtype=dtype, device=device)
                feat_dim = 4

            # Shuffle real tasks before appending NOOP (training regularisation)
            # Also shuffle valid_task_indices with the same permutation so that
            # collect_rollout() correctly maps shuffled obs-positions back to env task IDs.
            if shuffle_tasks and task_features_tensor.shape[0] > 1:
                perm = torch.randperm(task_features_tensor.shape[0], device=device)
                task_features_tensor = task_features_tensor[perm]
                if valid_task_indices:
                    valid_task_indices = [valid_task_indices[i] for i in perm.tolist()]

            # Always add NOOP action at the end (after shuffle so it stays last)
            noop = torch.zeros(1, feat_dim, dtype=dtype, device=device)
            task_features_tensor = torch.cat([task_features_tensor, noop], dim=0)

            # Efficient self features processing
            self_features = obs['self'][env_id] if (
                point_type == point_type_enum.tasks_with_self.value or 
                point_type == point_type_enum.everything.value
            ) else []
            
            self_features_tensor = _convert_to_tensor(self_features, dtype, device)
            
            # Efficient others features processing
            others_features = obs['others'][env_id] if point_type == point_type_enum.everything.value else []
            others_features_tensor = _convert_to_tensor(others_features, dtype, device)
            
            # Debug information (only if needed)
            # if len(task_features) > 1:  # More than just NOOP
            #     print(f"Agent {agent} Env {env_id}: Task features shape: {task_features_tensor.shape}")
            
            env_features[agent] = {
                'tasks': task_features_tensor,
                'self': self_features_tensor,
                'others': others_features_tensor,
                'valid_task_indices': valid_task_indices,
                'total_tasks': len(task_features)  +1 
            }
        
        all_features.append(env_features)
    
    return all_features


def extract_wildfire_action_features(env, agents, num_parallel_envs, device,
                                     shuffle_tasks=False):
    """Extract **egocentric, grid-invariant, direction-agnostic** task features.

    Every feature is computed relative to the deciding agent and uses only
    unsigned distances so the pointer network cannot memorise directional
    biases from a small training grid.  A per-task coordination signal tells
    the agent how many *ready* allies can also reach each fire.

    Coordinate systems (important!):
        task_store: [row, col, intensity, fuel, ...]  (from nonzero() on [H, W])
        state.agents: [row, col]  (env convention: [y, x])

    Task features (7 dims):
        [abs_dx_norm, abs_dy_norm, dist_norm, in_range,
         intensity, fuel, agents_in_range]

    Normalization reference: ``attack_range`` from the environment config.
        abs_dx_norm    = |fire_col − agent_col| / attack_range
        abs_dy_norm    = |fire_row − agent_row| / attack_range
        dist_norm      = manhattan_distance  / (2 * attack_range)
        in_range       = 1.0  if chebyshev_distance ≤ attack_range  else 0.0
        agents_in_range= (#other agents within chebyshev attack_range of this
                          fire AND with suppressant > 0) / max(num_agents − 1, 1)

    in_range and agents_in_range use **Chebyshev** distance (max(|dx|,|dy|))
    to match the environment's own range check.

    NOOP sentinel: [0, 0, 0, 0, 0, 0, 0]   (real fires have intensity > 0)

    When ``shuffle_tasks=True``, real task rows (and their action_mapping
    entries) are randomly permuted BEFORE appending NOOP.
    """
    TASK_FEAT_DIM = 7  # <-- single source-of-truth for dimensionality

    action_features_dict = {}
    state = env.state()

    # --- attack range (scalar, shared across agents for now) ---------------
    ar_tensor = env.unwrapped.agent_config.attack_range
    if isinstance(ar_tensor, torch.Tensor):
        attack_range = float(ar_tensor.max().item())  # conservative: use max
    else:
        attack_range = float(ar_tensor)
    attack_range = max(attack_range, 1.0)  # avoid div-by-zero

    # --- agent positions  [num_agents, 2]  state.agents = [row, col] ------
    agents_positions = state.agents.to(device)
    num_agents = agents_positions.shape[0]
    allies_denom = max(num_agents - 1, 1)  # for normalization

    # Pre-compute all agent positions as (row, col)
    all_agent_row_col = []
    for ai in range(num_agents):
        all_agent_row_col.append((
            float(agents_positions[ai, 0].item()),  # row
            float(agents_positions[ai, 1].item()),  # col
        ))

    for agent_idx, agent in enumerate(agents):
        agent_row, agent_col = all_agent_row_col[agent_idx]

        for env_id in range(num_parallel_envs):
            task_indices = env.unwrapped.agent_action_mapping[agent][env_id]

            # Per-env suppressant readiness for all agents
            suppressants = state.suppressants[env_id].squeeze(0)
            agent_ready = [(suppressants[ai].item() > 0) for ai in range(num_agents)]

            action_features = []
            action_mapping = []

            for action_idx, task_id in enumerate(task_indices):
                task_feature = env.unwrapped.task_store[env_id][task_id].float()
                # task_store layout: [row, col, intensity, fuel, ...]
                fire_row = task_feature[0].item()
                fire_col = task_feature[1].item()
                intensity = task_feature[2].item()
                fuel      = task_feature[3].item()

                # Correct axis-aligned deltas (col vs col, row vs row)
                adx = abs(fire_col - agent_col)
                ady = abs(fire_row - agent_row)
                manhattan = adx + ady
                chebyshev = max(adx, ady)

                abs_dx_norm = adx / attack_range
                abs_dy_norm = ady / attack_range
                dist_norm   = manhattan / (2.0 * attack_range)
                # Chebyshev to match the env's own range check
                in_range    = 1.0 if chebyshev <= attack_range else 0.0

                # Count OTHER agents that can fight this fire:
                # (a) within Chebyshev attack_range AND (b) suppressant > 0
                allies_count = 0
                for ai in range(num_agents):
                    if ai == agent_idx:
                        continue
                    if not agent_ready[ai]:
                        continue
                    o_row, o_col = all_agent_row_col[ai]
                    other_cheb = max(abs(o_col - fire_col), abs(o_row - fire_row))
                    if other_cheb <= attack_range:
                        allies_count += 1
                agents_in_range = allies_count / allies_denom

                feat = torch.tensor(
                    [abs_dx_norm, abs_dy_norm, dist_norm, in_range,
                     intensity, fuel, agents_in_range],
                    device=device, dtype=torch.float32,
                )
                action_features.append(feat)
                action_mapping.append((action_idx, 0))

            # Shuffle real tasks (same permutation for features AND mapping)
            if shuffle_tasks and len(action_features) > 1:
                perm = torch.randperm(len(action_features)).tolist()
                action_features = [action_features[i] for i in perm]
                action_mapping = [action_mapping[i] for i in perm]

            # NOOP sentinel — always last, all zeros
            noop_feature = torch.zeros(TASK_FEAT_DIM, device=device, dtype=torch.float32)
            action_features.append(noop_feature)
            action_mapping.append((-1, -1))

            action_features_dict[(agent, env_id)] = {
                'action_features': torch.stack(action_features),
                'action_mapping': action_mapping,
            }

    return action_features_dict

def _convert_to_tensor(features, dtype, device):
    """
    Optimized helper function to convert features to tensors efficiently.
    """
    if isinstance(features, torch.Tensor):
        if features.numel() > 0:
            return features.to(device=device, dtype=dtype, non_blocking=True)
        else:
            return torch.empty(0, dtype=dtype, device=device)
    
    elif isinstance(features, (list, tuple)) and len(features) > 0:
        return torch.tensor(features, dtype=dtype, device=device)
    
    else:
        return torch.empty(0, dtype=dtype, device=device)


# Additional optimization utilities
def batch_process_observations(observations_list, device):
    """
    Batch process multiple observations for better GPU utilization.
    """
    batched_features = []
    
    with torch.no_grad():
        for obs in observations_list:
            # Process each observation and move to GPU in batches
            processed_obs = {}
            for key, value in obs.items():
                if isinstance(value, torch.Tensor):
                    processed_obs[key] = value.to(device, non_blocking=True)
                else:
                    processed_obs[key] = value
            batched_features.append(processed_obs)
    
    return batched_features


def optimize_graph_memory(graph_data_list):
    """
    Optimize memory usage for graph data structures.
    """
    optimized_graphs = []
    
    for graph_data in graph_data_list:
        # Ensure contiguous memory layout
        optimized_graph = {}
        for key, tensor in graph_data.items():
            if isinstance(tensor, torch.Tensor):
                optimized_graph[key] = tensor.contiguous()
            else:
                optimized_graph[key] = tensor
        
        optimized_graphs.append(optimized_graph)
    
    return optimized_graphs


# Caching utility for frequently accessed action spaces
class ActionSpaceCache:
    """Simple cache for action spaces to avoid repeated computations."""
    
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
    
    def get_valid_actions(self, agent, env_id, action_space):
        """Get cached valid actions or compute and cache them."""
        cache_key = (agent, env_id)
        
        if cache_key not in self.cache:
            if len(self.cache) >= self.max_size:
                # Simple LRU: remove first item
                self.cache.pop(next(iter(self.cache)))
            
            self.cache[cache_key] = extract_valid_actions_from_action_space(action_space, env_id)
        
        return self.cache[cache_key]


# Global cache instance
_action_space_cache = ActionSpaceCache()


def normalize_wildfire_agent_features(self_features, others_features, env):
    """Grid-invariant agent feature normalization for the PLATO actor.

    Replaces absolute (x, y) coordinates with intrinsic resource state and
    range-normalized relative distance so the policy transfers across grid
    sizes without retraining.

    Env self obs  = [x, y, power, suppressant, capacity, equipment, ...] (7-dim)
    Env others obs = [x, y, power, suppressant] (4-dim)

    Output per agent (4 dims):
        [power, suppressant, is_ready, rel_dist_norm]
    where
        power          — fighting power (from obs dim 2)
        suppressant    — current suppressant level (obs dim 3)
        is_ready       — 1.0 if suppressant > 0, else 0.0
        rel_dist_norm  — manhattan(other, self) / attack_range  (0.0 for self)

    All features are grid-invariant.
    """
    import torch

    # --- attack range (scalar) ---
    ar_tensor = env.unwrapped.agent_config.attack_range
    if isinstance(ar_tensor, torch.Tensor):
        attack_range = float(ar_tensor.max().item())
    else:
        attack_range = float(ar_tensor)
    attack_range = max(attack_range, 1.0)

    # --- self agent ---
    if self_features.dim() == 1:
        self_features = self_features.unsqueeze(0)  # [1, D]

    self_x   = self_features[0, 0].item()
    self_y   = self_features[0, 1].item()
    self_pow = self_features[0, 2]
    self_sup = self_features[0, 3]

    self_row = torch.stack([
        self_pow,
        self_sup,
        (self_sup > 0).float(),
        torch.tensor(0.0, device=self_features.device, dtype=self_features.dtype),
    ]).unsqueeze(0)  # [1, 4]

    if others_features.numel() == 0:
        return self_row

    # --- other agents ---
    if others_features.dim() == 1:
        others_features = others_features.unsqueeze(0)

    rows = []
    for i in range(others_features.shape[0]):
        ox = others_features[i, 0].item()
        oy = others_features[i, 1].item()
        o_pow = others_features[i, 2]
        o_sup = others_features[i, 3]
        rel_dist = (abs(ox - self_x) + abs(oy - self_y)) / attack_range

        rows.append(torch.stack([
            o_pow,
            o_sup,
            (o_sup > 0).float(),
            torch.tensor(rel_dist, device=self_features.device, dtype=self_features.dtype),
        ]))

    others_out = torch.stack(rows)  # [num_others, 4]
    return torch.cat([self_row, others_out], dim=0)