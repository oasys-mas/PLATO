import tensordict
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
# Observation Vector Helper


import torch.nn.functional as F

# Task state codes
_TASK_STATE_SYMBOLS = {0: "U", 1: "A", 2: "R"}  # Unaccepted / Accepted / Riding
_AGENT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def print_rideshare_grid(state, env_id=0, rewards=None, step=None, agent_names=None):
    """
    Print an ASCII grid of the rideshare environment state for a single parallel env.

    Grid cell legend:
        A1/A2/...   - Agent position (number = 1-based agent index)
        P1u/P2a/... - Passenger pickup location (u=unaccepted, a=accepted, r=riding)
        D1/D2/...   - Passenger dropoff destination
        *           - Multiple items share the cell (details in the summary table below)

    Args:
        state       : RideshareState object (from env.state() or env.unwrapped.state())
        env_id      : Which parallel environment to render (default 0)
        rewards     : Optional dict {agent_name: tensor[parallel_envs]} for this step
        step        : Optional int step number for the header
        agent_names : Optional list of agent name strings (e.g. ['driver_1','driver_2'])
    """
    # ── extract data for this env ────────────────────────────────────────────
    agents_pos = state.agents[env_id]          # [num_agents, 2] as (y, x)
    pax_mask   = state.passengers[:, 0] == env_id
    passengers = state.passengers[pax_mask]    # [N, 11]

    num_agents = agents_pos.shape[0]
    num_pax    = passengers.shape[0]

    # derive grid size from all positions
    all_rows = [int(agents_pos[i, 0]) for i in range(num_agents)]
    all_cols = [int(agents_pos[i, 1]) for i in range(num_agents)]
    if num_pax > 0:
        all_rows += passengers[:, 1].int().tolist() + passengers[:, 3].int().tolist()
        all_cols += passengers[:, 2].int().tolist() + passengers[:, 4].int().tolist()

    grid_h = max(all_rows) + 1 if all_rows else 4
    grid_w = max(all_cols) + 1 if all_cols else 4

    # ── build cell label lists (multiple items can share a cell) ────────────
    cell_labels = [[[] for _ in range(grid_w)] for _ in range(grid_h)]

    # agents
    for i in range(num_agents):
        r, c = int(agents_pos[i, 0]), int(agents_pos[i, 1])
        if 0 <= r < grid_h and 0 <= c < grid_w:
            cell_labels[r][c].append(f"A{i+1}")

    # passengers
    pax_summary = []
    for idx in range(num_pax):
        p = passengers[idx]
        py, px     = int(p[1]), int(p[2])
        dy, dx     = int(p[3]), int(p[4])
        fare       = float(p[5])
        task_state = int(p[6])
        owner      = int(p[7])  # -1 if unowned
        state_sym  = _TASK_STATE_SYMBOLS.get(task_state, "?")
        pid        = idx + 1

        label = f"P{pid}{state_sym.lower()}"
        if 0 <= py < grid_h and 0 <= px < grid_w:
            cell_labels[py][px].append(label)
        if 0 <= dy < grid_h and 0 <= dx < grid_w:
            cell_labels[dy][dx].append(f"D{pid}")

        owner_str = f"A{owner+1}" if owner >= 0 else "none"
        pax_summary.append((pid, py, px, dy, dx, fare, state_sym, owner_str))

    # ── render grid ──────────────────────────────────────────────────────────
    COL_W = 9
    sep = "+" + ("-" * COL_W + "+") * grid_w

    header = f"  Rideshare env={env_id}"
    if step is not None:
        header += f"  step={step}"
    print(header)
    print(sep)

    for r in range(grid_h):
        row_parts = []
        for c in range(grid_w):
            labels = cell_labels[r][c]
            if not labels:
                cell_str = " . "
            elif len(labels) == 1:
                cell_str = labels[0]
            else:
                # truncate to fit, show count
                joined = "/".join(labels)
                if len(joined) <= COL_W - 2:
                    cell_str = joined
                else:
                    cell_str = f"*{len(labels)}items"
            row_parts.append(f"{cell_str:^{COL_W}}")
        print("|" + "|".join(row_parts) + "|")
        print(sep)

    # ── agent positions + rewards summary ───────────────────────────────────
    print()
    print("  Agents:")
    for i in range(num_agents):
        r, c = int(agents_pos[i, 0]), int(agents_pos[i, 1])
        name = (agent_names[i] if agent_names and i < len(agent_names) else f"driver_{i+1}")
        reward_str = ""
        if rewards is not None and name in rewards:
            reward_str = f"  reward={float(rewards[name]):.3f}"
        print(f"    A{i+1} ({name}): pos=({r},{c}){reward_str}")

    # ── passenger summary ───────────────────────────────────────────────────
    if pax_summary:
        print()
        print("  Passengers:  (state: U=unaccepted  A=accepted  R=riding)")
        print(f"  {'ID':>3}  {'pickup':>8}  {'dropoff':>8}  {'fare':>5}  {'state':>5}  {'owner':>6}")
        print(f"  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*6}")
        for pid, py, px, dy, dx, fare, state_sym, owner_str in pax_summary:
            print(f"  P{pid:<2}  ({py},{px}):    ({dy},{dx}):   {fare:>5.1f}  {state_sym:^5}  {owner_str:>6}")
    else:
        print("  No passengers this step.")
    print()


def rideshare_action_info(actions_dict, num_agents):
    """Count rideshare-specific actions per step.
    
    Args:
        actions_dict: Dictionary mapping agent -> list of action tensors
        num_agents: Number of agents
        
    Returns:
        Tuple of (accept_count, pick_count, drop_count, noop_count) normalized by num_agents
    """
    noop_count = 0
    accept_count = 0  # action 0
    pickup_count = 0  # action 1
    dropoff_count = 0  # action 2

    for agent, agent_actions in actions_dict.items():
        for action_tensor in agent_actions:
            action = action_tensor[1] if len(action_tensor) > 1 else action_tensor[0]
            if action == -1:
                noop_count += 1
            elif action == 0:
                accept_count += 1
            elif action == 1:
                pickup_count += 1    
            elif action == 2:
                dropoff_count += 1       

    return accept_count/num_agents, pickup_count/num_agents, dropoff_count/num_agents, noop_count/num_agents


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
    for key, tensor in tdx.items():
        match key:
            case 'self':
                self_tensor=torch.flatten(tensor)
                
            case 'others': # this is a list of tensors
                for t in tensor:#extend for avoid having list of tensors
                    others_tensor.extend(torch.flatten(t))
                

            case 'tasks':
                for f in tensor:
                    tasks_tensor.extend(torch.flatten(f))
    if tasks_tensor:
        flattened_wildfire_tensor = torch.cat((self_tensor, torch.stack(others_tensor),torch.stack(tasks_tensor)), dim=0)                 
    else:
        flattened_wildfire_tensor = torch.cat((self_tensor, torch.stack(others_tensor)), dim=0)    
    return flattened_wildfire_tensor
    
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


def construct_global_observation_graph(observations):
    """
    Constructs a global undirected graph from all agent observations.

    Args:
        observations (dict): agent_name -> (TensorDict obs, ...)

    Returns:
        node_features (Tensor): [num_nodes, feat_dim]
        edge_index (Tensor): [2, num_edges]
        edge_weights (Tensor): [num_edges]
        node_labels (list): names of the nodes
        num_agents (int), num_tasks (int)
    """
    agent_features = []
    task_features = []
    agent_names = []
    edges = []

    task_registry = {}
    for i, (agent_name, obs_tuple) in enumerate(observations.items()):
        obs = obs_tuple[0]
        self_feat = obs["self"].unsqueeze(0)
        agent_features.append(self_feat)
        agent_names.append(agent_name)

        tasks = obs["tasks"]
        for t in range(tasks.shape[0]):
            task_key = tuple(tasks[t].tolist())
            if task_key not in task_registry:
                task_registry[task_key] = len(task_features)
                task_features.append(tasks[t].unsqueeze(0))
            task_idx = task_registry[task_key]
            # Mark as task index for shifting later
            edges.append((i, f"T{task_idx}"))  # agent task
            edges.append((f"T{task_idx}", i))  # task agent

    agent_tensor = torch.cat(agent_features, dim=0)      # [A, d]
    task_tensor = torch.cat(task_features, dim=0)        # [T, d]
    node_features = torch.cat([agent_tensor, task_tensor], dim=0)

    num_agents = agent_tensor.shape[0]
    num_tasks = task_tensor.shape[0]
    task_offset = num_agents

    # Add fully connected agent–agent edges (undirected, no repeats)
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            edges.append((i, j))
            edges.append((j, i))

    # Resolve task-index placeholders and shift them
    final_edges = []
    for src, dst in edges:
        if isinstance(src, str) and src.startswith("T"):
            src = int(src[1:]) + task_offset
        if isinstance(dst, str) and dst.startswith("T"):
            dst = int(dst[1:]) + task_offset
        final_edges.append((src, dst))

    edge_index = torch.tensor(final_edges, dtype=torch.long).T  # [2, E]
    edge_weights = torch.zeros(edge_index.shape[1])             # Init all weights to 0

    node_labels = [f"A{i}:{agent_names[i]}" for i in range(num_agents)] + [f"T{j}" for j in range(num_tasks)]

    return node_features, edge_index, edge_weights, node_labels, num_agents, num_tasks



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
