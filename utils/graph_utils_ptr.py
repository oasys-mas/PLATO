import tensordict
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
# Observation Vector Helper


import torch.nn.functional as F
def get_available_actions(agents_actions):
    """
    Extracts available (agent_index, action) pairs from a NestedTensor of shape [batch, num_agents].
    Always appends a fallback action (0, -1) at the end.
    
    Args:
        agents_actions: NestedTensor where each entry is a 1D tensor of actions.
        
    Returns:
        List of tuples (agent_index, action), plus (0, -1) fallback.
    """
    available = []
    # If there's a batch dimension, unbind to iterate over each entry
    for tensor_i in agents_actions.unbind(0):
        # tensor_i is a 1D tensor: one action per agent
        for agent_idx, action in enumerate(tensor_i.tolist()):
            available.append((agent_idx, 0))
    # Fallback action
    available.append((-1, -1))
    return available

# Example usage:
# actions = nested_tensor([tensor([0, 1])])
# print(get_available_actions(actions))

def print_wildfire_grid(state):
    fires = state.fires[0]         # shape [rows, cols]
    intensities = state.intensity[0]
    agents_pos = state.agents      # shape [num_agents, 2] - stored as (row, col)
    suppressants = state.suppressants[0]
    
    grid_height, grid_width = fires.shape
    # Create empty grid of strings
    display_grid = [[" . " for _ in range(grid_width)] for _ in range(grid_height)]
    
    # Place fires
    for r in range(grid_height):
        for c in range(grid_width):
            if fires[r, c] > 0:
                display_grid[r][c] = f"F_{int(fires[r,c])+1},i={int(intensities[r,c])}"
    
    # Place agents - state.agents is already (row, col)
    for i, (row, col) in enumerate(agents_pos):
        r, c = int(row), int(col)
        
        # Check bounds before placing
        if 0 <= r < grid_height and 0 <= c < grid_width:
            s = float(suppressants[i])
            display_grid[r][c] = f"A_{i+1},S={int(s)}"
        else:
            print(f"WARNING: Agent {i+1} at position (row={row}, col={col}) -> grid({r}, {c}) is outside bounds {grid_height}x{grid_width}")
    
    # Print table
    print("-" * (grid_width * 10 + 1))
    for row in display_grid:
        row_str = "|".join(f"{cell:^9}" for cell in row)
        print(f"|{row_str}|")
        print("-" * (grid_width * 10 + 1))
        
 

def parse_wildfire_tensordict(tdx, configuration):
    grid_width = configuration.grid_width
    grid_height = configuration.grid_height
    grid_size = grid_width * grid_height
    feat_dim = tdx['self'].shape[-1]
    placeholder = -99.0

    batch_size = tdx['self'].shape[0]
    all_obs = []

    for b in range(batch_size):
        obs_vector = torch.full((1 + grid_size + grid_size, feat_dim), placeholder)

        # Self
        obs_vector[0] = tdx['self'][b]

        # Others
        if 'others' in tdx:
            for other in tdx['others'][b]:
                x, y = int(other[0].item()), int(other[1].item())
                if 0 <= x < grid_width and 0 <= y < grid_height:
                    idx = y * grid_width + x
                    padded_other = F.pad(other, (0, feat_dim - other.shape[0]), value=placeholder)
                    obs_vector[1 + idx] = padded_other

        # Tasks
        if 'tasks' in tdx:
            for task in tdx['tasks'][b]:
                x, y = int(task[0].item()), int(task[1].item())
                if 0 <= x < grid_width and 0 <= y < grid_height:
                    idx = y * grid_width + x
                    padded_task = F.pad(task, (0, feat_dim - task.shape[0]), value=placeholder)
                    obs_vector[1 + grid_size + idx] = padded_task

        all_obs.append(obs_vector.flatten())

    return torch.stack(all_obs)

def parse_wildfire_tensordic41t(tdx: tensordict):
    self_tensor=[] #always 1 tensor
    others_tensor=[] #number of agents -1 in the conf
    tasks_tensor=[] #number of fires, can vary in number
    for key, tensor in tdx[0].items():
        match key:
            case 'self':
                self_tensor=torch.flatten(tensor)
                
            case 'others': # this is a list of tensors
                for t in tensor:#extend for avoid having list of tensors
                    others_tensor.extend(torch.flatten(t))
                

            case 'tasks':
                for f in tensor:
                    tasks_tensor.extend(torch.flatten(f))
    concat_list = [self_tensor]

    if len(others_tensor) > 0:
        concat_list.append(torch.stack(others_tensor))

    if len(tasks_tensor) > 0:
        concat_list.append(torch.stack(tasks_tensor))
    if len(concat_list) > 0:
        flattened_wildfire_tensor = torch.cat(concat_list, dim=0)

        return flattened_wildfire_tensor
    else: 
        return ''
    
# Observation Vector Helper
def populate_observation_vector(agent, obs_dict, configuration):
    cells = configuration.grid_height * configuration.grid_width
    feat_dim = obs_dict[agent]["self"].shape[-1]
    obs_vector = torch.full(((1 + cells + cells) * feat_dim,), fill_value=-99.0)

    # self features
    self_feats = obs_dict[agent]["self"].flatten()
    obs_vector[:self_feats.shape[0]] = self_feats

    # tasks
    if "tasks" in obs_dict[agent]:
        tasks_tensor = obs_dict[agent]["tasks"]
        for t in range(min(tasks_tensor.shape[0], cells)):
            task_feats = tasks_tensor[t].flatten()
            start = (1 + t) * feat_dim
            obs_vector[start:start + task_feats.shape[0]] = task_feats

    # others
    if "others" in obs_dict[agent]:
        others_tensor = obs_dict[agent]["others"]
        for o in range(min(others_tensor.shape[0], cells)):
            other_feats = others_tensor[o].flatten()
            start = (1 + cells + o) * feat_dim
            obs_vector[start:start + other_feats.shape[0]] = other_feats

    return obs_vector

def draw_observation_graph(edge_index, edge_weights, node_labels, num_agents, num_tasks):
    G = nx.Graph()
    for i in range(num_agents):
        G.add_node(node_labels[i], type="agent", bipartite=0)
    for j in range(num_tasks):
        G.add_node(node_labels[num_agents + j], type="task", bipartite=1)

    src, dst = edge_index
    for i in range(edge_index.shape[1]):
        u = node_labels[src[i].item()]
        v = node_labels[dst[i].item()]
        G.add_edge(u, v, weight=edge_weights[i].item())

    pos = {}
    for i in range(num_agents):
        pos[node_labels[i]] = (i, 1)
    for j in range(num_tasks):
        pos[node_labels[num_agents + j]] = (j, 0)

    edge_labels = nx.get_edge_attributes(G, 'weight')
    plt.figure(figsize=(max(8, num_agents + num_tasks), 4))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=["skyblue" if G.nodes[n]["type"] == "agent" else "lightcoral" for n in G.nodes],
        node_size=1200, font_weight='bold', edge_color='gray'
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Global Undirected Graph with Weights")
    plt.axis('off')
    plt.show()


def compute_agent_graph_from_task_attention(attn_weights):
    """
    Compute agent-agent edge weights using task attention.
    
    Args:
        attn_weights (Tensor): shape [T, A], where T = number of tasks, A = number of agents
    
    Returns:
        agent_agent_matrix (Tensor): [A, A] softmax-normalized edge weights
    """
    # Step 1: Outer product for each task contribution to agent-agent edges
    # α_k^T @ α_k [A, A] for each task
    T, A = attn_weights.shape
    M = torch.zeros(A, A, device=attn_weights.device)

    for k in range(T):
        alpha_k = attn_weights[k].unsqueeze(1)  # [A, 1]
        M += alpha_k @ alpha_k.T                # [A, A] += [A, 1] x [1, A]

    # Step 2: Softmax row-wise (per-agent)
    M_soft = torch.softmax(M, dim=-1)  # [A, A]

    return M_soft
def convert_actions_to_tensor(actions, device):
    """
    Convert batched (task_idx, action_type) to Tensor per agent.

    Args:
        actions (dict): agent_name -> Tensor [num_envs, 2]
        device (torch.device): target device

    Returns:
        dict: agent_name -> Tensor [num_envs, 2]
    """
    converted = {}
    for agent, action_tensor in actions.items():
        if isinstance(action_tensor, torch.Tensor):
            converted[agent] = action_tensor.to(device)
        else:
            raise ValueError(f"Action for {agent} is not a tensor: {action_tensor}")
    return converted

def action_info(actions_dict):
    """Count NOOP vs fight actions across all agents and environments
    
    Args:
        actions_dict: Dictionary with format {'firefighter_1': tensor([[0, 0]]), ...}
        
    Returns:
        tuple: (noop_counts, fight_counts)
    """
    noop_counts = 0
    fight_counts = 0
    
    for agent, actions in actions_dict.items():
        # actions is a tensor with shape [num_envs, 2] 
        for env_idx in range(actions.shape[0]):
            action = actions[env_idx]
            if action[1].item() == -1:  # NOOP action
                noop_counts += 1
            else:
                fight_counts += 1
    
    return noop_counts, fight_counts


def get_episode_fire_counts(initial_state, last_state,episode_burnouts,episode_putouts,NUM_PARALLEL_ENVS):
    # Extract fire tensors - shape should be [batch_size, height, width] 
    initial_fires = initial_state.fires
    final_fires = last_state.fires
    
    # Count total fires across all environments
    initial_fire_count = (initial_fires > 0).sum().item() /NUM_PARALLEL_ENVS
    final_fire_count = (final_fires > 0).sum().item() /NUM_PARALLEL_ENVS
    new_ignited=(episode_putouts+episode_burnouts+final_fire_count)-initial_fire_count
    episode_total_fires=(episode_putouts+episode_burnouts+final_fire_count)
    return episode_total_fires,new_ignited

def episode_infoes(initial_state, next_state):
    """Calculate fire statistics between two states across all environments
    
    This properly tracks fire dynamics step-by-step:
    Example: Start with 10 fires, 3 get extinguished, 2 new fires ignite -> End with 9 fires
    - Initial: 10, New: 2, Extinguished: 3, Final: 9
    - Verification: 10 + 2 - 3 = 9 
    
    Args:
        initial_state: WildfireState at start of rollout/step
        next_state: WildfireState after rollout/step
        
    Returns:
        tuple: (initial_fires, new_fires, final_fires, extinguished_fires, suppressant_used)
    """
    # Extract fire tensors - shape should be [batch_size, height, width] 
    initial_fires = initial_state.fires
    final_fires = next_state.fires
    
    # Count total fires across all environments
    initial_fire_count = (initial_fires > 0).sum().item()
    final_fire_count = (final_fires > 0).sum().item()
    
    # NEW FIRES: Cells that went from NO fire (0) -> FIRE (>0) 
    # This captures actual new ignitions, regardless of what else happened
    new_fires_mask = (initial_fires == 0) & (final_fires > 0)
    new_fires = new_fires_mask.sum().item()
    
    # EXTINGUISHED FIRES: Cells that went from FIRE (>0) -> NO fire (0)
    # This captures fires that were put out or burned out
    extinguished_mask = (initial_fires > 0) & (final_fires == 0) 
    extinguished_fires = extinguished_mask.sum().item()
    
    # Suppressant used = difference in total suppressant levels
    initial_suppressants = initial_state.suppressants.sum().item()
    final_suppressants = next_state.suppressants.sum().item()
    suppressant_used = max(0, initial_suppressants - final_suppressants)
    
    # VERIFICATION: Check fire conservation
    expected_final = initial_fire_count + new_fires - extinguished_fires
    if expected_final != final_fire_count:
        print(f"FIRE TRACKING ERROR!")
        print(f"   Initial: {initial_fire_count}")
        print(f"   New fires: {new_fires}")
        print(f"   Extinguished: {extinguished_fires}")
        print(f"   Expected final: {expected_final}")
        print(f"   Actual final: {final_fire_count}")
    
    return initial_fire_count, new_fires, final_fire_count, extinguished_fires, suppressant_used
