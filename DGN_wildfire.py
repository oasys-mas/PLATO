# DGN on Wildfire Environment - Fixed Version
import os, sys  
import numpy as np
import torch
import torch.optim as optim
import logging
import tabulate
from colorama import Fore
import gc
import time
from tqdm import trange

# Import your existing DGN components
from baselines.DGN.model import DGN
from baselines.DGN.buffer import ReplayBuffer
from baselines.DGN.config import *
from baselines.DGN.utilities import *


# Import wildfire environment components
from utils.experiment_manager import load_config
from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from utils.graph_utils_ptr import convert_actions_to_tensor, get_available_actions, get_episode_fire_counts
from utils.helper import count_zero_suppressant_agents, get_training_seed, get_validation_seed, resolve_configuration, parse_args, save_checkpoint, set_global_seed, setup_logging, setup_tensorboard_logging
from copy import deepcopy


config = load_config("DGN")
general_config = load_config("General_Config")

if general_config.DB == "postgres":
    from experiment_logger_postgres import ExperimentDB, RunLogger
else:
    from experiment_logger_sqlite import ExperimentDB, RunLogger
    

if general_config.DEVICE != (None or ''):
    DEVICE=general_config.DEVICE
else:    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def init_env(configuration):
    """Initialize wildfire environment"""
    env = wildfire_v0.parallel_env(
        configuration=configuration,
        parallel_envs=1,
        max_steps=general_config.MAX_STEPS,
        device=DEVICE,
        buffer_size=general_config.MAX_STEPS,
        single_seeding=True,
        show_bad_actions=False,
        observe_other_power=True,
        observe_other_suppressant=True,
        override_initialization_check=True
    )
    return action_mapping_wrapper_v0(env)

def pad_observations(observations, max_obs_dim=config.OBS_DIM, env_id=0):
    """Build fixed-size obs vectors"""
    padded_obs = {}
    for agent, obs in observations.items():
        agent_obs_dict = obs[env_id]

        if 'tasks' in agent_obs_dict and agent_obs_dict['tasks'].numel() > 0:
            tasks = agent_obs_dict['tasks'][env_id].float()
        else:
            tasks = torch.empty(0, general_config.FEAT_DIM, device=DEVICE)
            
        self_f = (agent_obs_dict['self'][env_id].float()
                  if 'self' in agent_obs_dict
                  else torch.empty(0, device=DEVICE))
                  
        if 'others' in agent_obs_dict and agent_obs_dict['others'].numel() > 0:
            others = agent_obs_dict['others'][env_id].float()
        else:
            others = torch.empty(0, general_config.FEAT_DIM, device=DEVICE)

        num_tasks = min(tasks.shape[0], general_config.MAX_TASKS) if tasks.numel() > 0 else 0
        num_others = min(others.shape[0], general_config.MAX_TASKS) if others.numel() > 0 else 0

        padded_tasks = torch.full((general_config.MAX_TASKS, general_config.FEAT_DIM), -1.0, device=DEVICE)
        padded_others = torch.full((general_config.MAX_TASKS, general_config.FEAT_DIM), -1.0, device=DEVICE)
        if num_tasks > 0: 
            padded_tasks[:num_tasks] = tasks[:num_tasks]
        if num_others > 0: 
            padded_others[:num_others] = others[:num_others]

        agent_vec = torch.cat([padded_tasks.flatten(), self_f.flatten(), padded_others.flatten()])
        if agent_vec.numel() < max_obs_dim:
            out = torch.zeros(max_obs_dim, device=DEVICE)
            out[:agent_vec.numel()] = agent_vec
        else:
            out = agent_vec[:max_obs_dim]
        padded_obs[agent] = out
    return padded_obs

def create_action_masks(env, agents, observations, infos, env_id=0):
    """Create action masks for wildfire environment"""
    action_masks = {}
    action_mappings = {}

    for agent in agents:
        if infos and agent in infos and "action_mask" in infos[agent]:
            m = infos[agent]["action_mask"][env_id].float().to(DEVICE)
            if m.numel() < general_config.MAX_ACTIONS_PER_AGENT:
                pad = torch.zeros(general_config.MAX_ACTIONS_PER_AGENT - m.numel(), device=DEVICE)
                m = torch.cat([m, pad], dim=0)
            else:
                m = m[:general_config.MAX_ACTIONS_PER_AGENT]
            mask = m
            amap = env.unwrapped.agent_action_mapping[agent]
            action_mappings[agent] = amap[:general_config.MAX_ACTIONS_PER_AGENT]
        else:
            amap = env.unwrapped.agent_action_mapping[agent]
            avail = get_available_actions(amap)
            mask = torch.zeros(general_config.MAX_ACTIONS_PER_AGENT, device=DEVICE)
            if len(avail) > 0:
                count = min(len(avail), general_config.MAX_ACTIONS_PER_AGENT)
                mask[:count] = 1.0
                action_mappings[agent] = avail[:count]
            else:
                mask[-1] = 1.0  # NOOP
                action_mappings[agent] = [(-1, -1)]
        
        mask[-1] = 1.0  # Always allow NOOP
        action_masks[agent] = mask

    return action_masks, action_mappings

def get_wildfire_obs(observations, agents, env_id=0):
    """Convert wildfire observations to DGN format"""
    padded_obs = pad_observations(observations, env_id=env_id)
    obs_list = []
    for agent in agents:
        # Add agent ID like in SMAC
        agent_id = np.zeros(len(agents))
        agent_id[agents.index(agent)] = 1
        agent_obs = torch.cat([padded_obs[agent], torch.tensor(agent_id, device=DEVICE, dtype=torch.float32)])
        obs_list.append(agent_obs.cpu().numpy())
    return np.array(obs_list)

def create_adjacency_matrix(env, agents, comm_flag=1.0):
    """Create adjacency matrix for communication"""
    n_agents = len(agents)
    adj = np.ones((n_agents, n_agents)) * comm_flag + np.eye(n_agents)
    return adj

def evaluate(env, model, agents):
    """Evaluate current policy"""
    total_rewards = []
    total_burnouts = []
    total_putouts = []
    total_lengths = []
    eval_seeds = []
    
    # Set model to eval mode
    model.eval()
    
    with torch.no_grad():
        for eval_episode in range(general_config.VALIDATION_EPISODES):
            seed = get_validation_seed(eval_episode, general_config)
            observations, infos = env.reset(seed=seed)
            eval_seeds.append(seed)
            
            episode_reward = 0
            episode_putouts = 0
            episode_burnouts = 0
            steps = 0
            
            while not torch.all(env.finished):
                obs = get_wildfire_obs(observations, agents, env_id=0)
                adj = create_adjacency_matrix(env, agents, float(config.comm_flag))
                action_masks, action_mappings = create_action_masks(env, agents, observations, infos, env_id=0)
                
                mask = np.array([action_masks[agent].cpu().numpy() for agent in agents])
                
                # Get Q-values (deterministic for evaluation)
                q = model(torch.Tensor(np.array([obs])).to(DEVICE), 
                         torch.Tensor(np.array([adj])).to(DEVICE))[0]
                
                # Select actions via sampling
                env_actions = {}
                for i, agent in enumerate(agents):
                    masked_q = q[i].cpu().detach().clone()
                    masked_q[mask[i] == 0] = float('-inf')
                    probs = torch.softmax(masked_q, dim=-1)
                    a = torch.multinomial(probs, 1).item()
                    
                    # Map to environment action
                    if a >= len(action_mappings[agent]):
                        env_action = (-1, -1)  # NOOP
                    else:
                        env_action = action_mappings[agent][a]
                    env_actions[agent] = torch.tensor(env_action, device=DEVICE).unsqueeze(0)
                
                # Environment step
                observations, reward_dict, term, trunc, infos = env.step(
                    convert_actions_to_tensor(env_actions, DEVICE)
                )
                
                # Process metrics
                episode_burnouts += int(infos.get("burnouts", 0))
                episode_putouts += int(infos.get("putouts", 0))
                
                # Calculate step reward
                reward_per_agent = np.array([r[0].item() if torch.is_tensor(r) else float(r) for r in reward_dict.values()])
                step_reward = np.mean(reward_per_agent)
                episode_reward += step_reward
                
                steps += 1
                
                # Check termination
                if any(t[0].item() if torch.is_tensor(t) else bool(t) for t in term.values()):
                    break
            
            # Store episode metrics
            total_rewards.append(episode_reward)
            total_burnouts.append(episode_burnouts)
            total_putouts.append(episode_putouts)
            total_lengths.append(steps)
    
    # Set model back to train mode
    model.train()
    
    # Calculate averages
    avg_reward = np.mean(total_rewards)
    avg_burnouts = np.mean(total_burnouts)
    avg_putouts = np.mean(total_putouts)
    avg_length = np.mean(total_lengths)
    
    return avg_reward, avg_burnouts, avg_putouts, avg_length, str(eval_seeds)

def test_agent(env, model, agents):
    """Test agent performance"""
    test_r, test_episodes = 0, 0
    
    for episode in range(20):
        observations, infos = env.reset()
        episode_reward = 0
        terminated = False
        
        while not terminated and not torch.all(env.finished):
            obs = get_wildfire_obs(observations, agents, env_id=0)
            adj = create_adjacency_matrix(env, agents)
            action_masks, action_mappings = create_action_masks(env, agents, observations, infos, env_id=0)
            
            mask = np.array([action_masks[agent].cpu().numpy() for agent in agents])
            
            with torch.no_grad():
                q = model(torch.Tensor(np.array([obs])).to(DEVICE), 
                         torch.Tensor(np.array([adj])).to(DEVICE))[0]
            
            actions = []
            env_actions = {}
            for i, agent in enumerate(agents):
                masked_q = q[i].cpu().detach().clone()
                masked_q[mask[i] == 0] = float('-inf')
                probs = torch.softmax(masked_q, dim=-1)
                a = torch.multinomial(probs, 1).item()
                actions.append(a)
                
                if a == (len(mask[i]) - 1):
                    env_action = (-1, -1)
                else:
                    avail = action_mappings[agent]
                    if a < len(avail):
                        env_action = avail[a]
                    else:
                        env_action = (-1, -1)
                env_actions[agent] = torch.tensor(env_action, device=DEVICE).unsqueeze(0)
            
            observations, reward_dict, term, trunc, infos = env.step(
                convert_actions_to_tensor(env_actions, DEVICE)
            )
            
            reward = sum(r[0].item() if torch.is_tensor(r) else float(r) for r in reward_dict.values()) / len(agents)
            episode_reward += reward
            
            terminated = any(t[0].item() if torch.is_tensor(t) else bool(t) for t in term.values())
        
        test_r += episode_reward
        test_episodes += 1
    
    return test_r / test_episodes, test_episodes


def main():
    """Main training loop - FIXED VERSION"""
    args = parse_args()
    args.experiment_name = 'DGN'

    if(general_config.use_global_seed):
        set_global_seed(general_config.Global_Seed,general_config.use_deterministic)
    
    
    # Initialize environment
    configuration = resolve_configuration(args)
    env = init_env(configuration)
    
    # Get environment info
    observations, infos = env.reset()
    agents = env.agents
    n_agents = len(agents)
    n_actions = general_config.MAX_ACTIONS_PER_AGENT
    obs_space = config.OBS_DIM + n_agents
    
    # Initialize DGN components
    buff = ReplayBuffer(capacity, obs_space, n_actions, n_agents)
    model = DGN(n_agents, obs_space, hidden_dim, n_actions).to(DEVICE)
    model_tar = DGN(n_agents, obs_space, hidden_dim, n_actions).to(DEVICE)
    model_tar.load_state_dict(model.state_dict())
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    
    comm_flag = float(config.comm_flag)
    writer, checkpoint_dir = setup_tensorboard_logging(args)
    
    # Setup DB logger
    db = ExperimentDB()    
    experiment_meta = dict(
        run_type="training",
        model_family="DGN",
        model_type="smart",
        name=args.experiment_name,
        variation1="dgn",
        variation2="graph_network",
        grid_size=args.grid_size,
        setting_type=args.setting_type,
        parent_experiment_id=None,
        checkpoint_path=checkpoint_dir,
        notes=None,
    )   

    run_logger = RunLogger(
        writer=writer,
        db=db,
        experiment_meta=experiment_meta,
        log_every=general_config.LOG_EVERY,
    )
    
    logging.info("=== DGN Training on Wildfire Environment ===")
    logging.info(f"Agents: {n_agents}, Actions: {n_actions}, Obs Space: {obs_space}")
    
    # Training variables
    i_episode = 0
    time_step = 0
    test_flag = 0
    epsilon = 0.9
    current_loss = 0.0  # Track actual loss
    
    # Main training loop
    while i_episode < general_config.EPISODES:
        if time_step > max_timestep:
            break
            
        # Decay epsilon
        if i_episode > 100:
            epsilon -= 0.001
            if epsilon < 0.02:
                epsilon = 0.02
        
        i_episode += 1
        episode_steps = 0 
        episode_burnouts = 0
        episode_putouts = 0
        episode_noop_count = 0
        episode_fight_count = 0
        episode_supressant_wait_time = 0

        # Start timer 
        t0 = time.perf_counter()
        
        # Get training seed
        seed = get_training_seed(i_episode, general_config)
        observations, infos = env.reset(seed=seed)
        env_seeds = str(seed)
        initial_state = deepcopy(env.state())
        terminated = False
        episode_reward = 0
        
        # Get initial state
        obs = get_wildfire_obs(observations, agents, env_id=0)
        adj = create_adjacency_matrix(env, agents, comm_flag)
        action_masks, action_mappings = create_action_masks(env, agents, observations, infos, env_id=0)
        mask = np.array([action_masks[agent].cpu().numpy() for agent in agents])
        
        # Episode loop
        while not terminated and not torch.all(env.finished):
            test_flag += 1
            time_step += 1
            episode_steps += 1
            
            # Select actions
            with torch.no_grad():
                q = model(torch.Tensor(np.array([obs])).to(DEVICE), 
                         torch.Tensor(np.array([adj])).to(DEVICE))[0]
            
            actions = []
            env_actions = {}
            for i, agent in enumerate(agents):
                if np.random.rand() < epsilon:
                    avail_actions_ind = np.nonzero(mask[i])[0]
                    a = np.random.choice(avail_actions_ind)
                else:
                    masked_q = q[i].cpu().detach().clone()
                    masked_q[mask[i] == 0] = float('-inf')
                    probs = torch.softmax(masked_q, dim=-1)
                    a = torch.multinomial(probs, 1).item()
                actions.append(a)
                
                # Map to environment action - SIMPLIFIED
                if a >= len(action_mappings[agent]):
                    env_action = (-1, -1)  # NOOP
                    episode_noop_count += 1
                else:
                    env_action = action_mappings[agent][a]
                    if env_action == (-1, -1):
                        episode_noop_count += 1
                    else:
                        episode_fight_count += 1
                env_actions[agent] = torch.tensor(env_action, device=DEVICE).unsqueeze(0)
            
            # Step environment
            observations, reward_dict, term, trunc, infos = env.step(
                convert_actions_to_tensor(env_actions, DEVICE)
            )
            
            # FIXED: Use averaged (shared) rewards to match original buffer expecting scalar
            reward_per_agent = np.array([r[0].item() if torch.is_tensor(r) else float(r) for r in reward_dict.values()])
            # Clip per-agent for stability, then average
            reward_per_agent = np.clip(reward_per_agent, -10, 10)
            reward_avg = np.mean(reward_per_agent)  # Scalar for buffer and logging
            episode_reward += reward_avg
            step_supressant_wait_time=count_zero_suppressant_agents(env.state())

            # Track fire metrics
            episode_burnouts += int(infos.get("burnouts", 0))
            episode_putouts += int(infos.get("putouts", 0))
            episode_supressant_wait_time+=step_supressant_wait_time

            # Get next state
            next_obs = get_wildfire_obs(observations, agents, env_id=0)
            next_adj = create_adjacency_matrix(env, agents, comm_flag)
            action_masks, action_mappings = create_action_masks(env, agents, observations, infos, env_id=0)
            next_mask = np.array([action_masks[agent].cpu().numpy() for agent in agents])
            
            terminated = any(t[0].item() if torch.is_tensor(t) else bool(t) for t in term.values())
            
            # Store experience in buffer - FIXED: scalar reward to match buffer
            buff.add(np.array(obs), actions, reward_avg, np.array(next_obs), adj, next_adj, next_mask, terminated)
            
            # Update state
            obs = next_obs
            adj = next_adj
            mask = next_mask
        
        last_state = env.state()
        episode_total_fires_count, new_fire_ignited_count = get_episode_fire_counts(initial_state, last_state, episode_burnouts, episode_putouts,config.NUM_PARALLEL_ENVS)
        
        # Performance metrics 
        elapsed = time.perf_counter() - t0
        fps = episode_steps / max(elapsed, 1e-9)
        
        # Create  training metrics
        train_metrics = {
            'actor_loss': current_loss,  # Use DQN loss as actor loss
            'critic_loss': current_loss,  # DGN combines actor/critic
            'entropy': 0.0,  # DGN doesn't have explicit entropy
            'approx_kl': 0.0,
            'value_mse': current_loss,
            'explained_variance': 0.0
        }
        
        # Create episode statistics 
        episode_stats = dict(
            reward=episode_reward,
            burnouts=episode_burnouts,
            putouts=episode_putouts,
            episode_total_fires_count=episode_total_fires_count,
            new_fire_ignited_count=new_fire_ignited_count,
            episode_noop_count=episode_noop_count,
            episode_fight_count=episode_fight_count,
            episode_length=episode_steps,
            env_seeds=env_seeds, 
            episode_supressant_wait_time=episode_supressant_wait_time

        )

        # Gradient summary (simplified)
        grad_summary = {
            'actor_vanishing': 0,
            'actor_exploding': 0,
            'critic_vanishing': 0,
            'critic_exploding': 0,
            'actor_max_grad': 0,
            'critic_max_grad': 0,
        }

        # Log episode using RunLogger
        run_logger.log_episode(args.experiment_name,
            episode=i_episode,
            episode_stats=episode_stats,
            train_metrics=train_metrics,
            grad_summary=grad_summary,
            total_actor_grad=0.0, 
            total_critic_grad=0.0,
            elapsed=elapsed,
            fps=fps,
        )
        
        # FIXED: Skip training for first 100 episodes, like SMAC
        if i_episode < 100:
            # Logging for early episodes
            if i_episode % general_config.LOG_EVERY == 0:

                logging.info('\n'+Fore.RED + tabulate.tabulate([
                    ['Episode', i_episode],
                    ['Reward', f"{episode_reward:.3f}"],
                    ['Episode Length', episode_steps],
                    ['Epsilon', f"{epsilon:.3f}"]
                ], headers=['Metric', 'Value'], tablefmt='orgtbl'))
            continue
        
        # FIXED: Training moved outside episode loop, after collecting full episode
        for epoch in range(n_epoch):
            # Get batch from replay buffer
            O, A, R, Next_O, Matrix, Next_Matrix, Next_Mask, D = buff.getBatch(batch_size)
            
            # FIXED: R is scalar per transition (batch_size,); same for all agents
            
            # Convert to tensors
            O_tensor = torch.Tensor(O).to(DEVICE)
            Matrix_tensor = torch.Tensor(Matrix).to(DEVICE)
            Next_O_tensor = torch.Tensor(Next_O).to(DEVICE)
            Next_Matrix_tensor = torch.Tensor(Next_Matrix).to(DEVICE)
            Next_Mask_tensor = torch.Tensor(Next_Mask).to(DEVICE)
            R_tensor = torch.Tensor(R).to(DEVICE)  # (batch_size,)
            
            # Forward pass
            q_values = model(O_tensor, Matrix_tensor)
            
            # Target Q-values
            with torch.no_grad():
                target_q_values = model_tar(Next_O_tensor, Next_Matrix_tensor)
                target_q_values = (target_q_values - 9e15*(1 - Next_Mask_tensor)).max(dim=2)[0]
                target_q_values_np = target_q_values.cpu().data.numpy()
            
            # Compute expected Q-values (following SMAC: shared R for all agents)
            expected_q = q_values.clone().cpu().data.numpy()
            for j in range(batch_size):
                for i in range(n_agents):
                    expected_q[j][i][A[j][i]] = float(R[j][0]) + (1 - float(D[j][0])) * GAMMA * target_q_values_np[j][i]
            
            # Compute loss
            expected_q_tensor = torch.Tensor(expected_q).to(DEVICE)
            loss = (q_values - expected_q_tensor).pow(2).mean()
            current_loss = loss.item()  # Track the actual loss
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            # Soft update target network
            with torch.no_grad():
                for p, p_targ in zip(model.parameters(), model_tar.parameters()):
                    p_targ.data.mul_(tau)
                    p_targ.data.add_((1 - tau) * p.data)
        
        gc.collect()  # Clean up after training
        
        # Run evaluation every VALIDATION_EVERY episodes
        if i_episode % general_config.VALIDATION_EVERY == 0 and i_episode > 0:
            eval_reward, eval_burnouts, eval_putouts, eval_length, eval_seeds = evaluate(env, model, agents)
            

            # Log evaluation using RunLogger
            run_logger.log_eval(
                episode=i_episode,
                eval_reward=eval_reward,
                eval_burnouts=eval_burnouts,
                eval_putouts=eval_putouts,
                eval_length=eval_length,
                eval_seeds=eval_seeds,

                avg_noop= 0
            )
            
            logging.info(f"Evaluation at episode {i_episode}: "
                        f"Reward: {eval_reward:.4f}, "
                        f"Burnouts: {eval_burnouts:.4f}, "
                        f"Putouts: {eval_putouts:.4f}, "
                        f"Length: {eval_length:.2f}, "
                  )
        
        # Logging
        if i_episode % general_config.LOG_EVERY == 0:
    
            logging.info('\n'+Fore.RED + tabulate.tabulate([
                ['Episode', i_episode],
                ['Reward', f"{episode_reward:.3f}"],
                ['Episode Length', episode_steps],
                ['Loss', f"{current_loss:.6f}"],
                ['Epsilon', f"{epsilon:.3f}"],
                ['Episode Seconds', f"{elapsed:.2f}"],
                ['Steps/sec', f"{fps:.2f}"],
            ], headers=['Metric', 'Value'], tablefmt='orgtbl'))
        
        # Testing
        if test_flag > 10000:
            log_r, _ = test_agent(env, model, agents)
            logging.info(f"Test Reward: {log_r:.3f}")
            test_flag = 0
        
        # Save checkpoint
        if i_episode % general_config.SAVE_EVERY == 0:
            save_checkpoint({
                'model': model.state_dict(),
                'model_tar': model_tar.state_dict(),
                'episode': i_episode,
                'reward': episode_reward
            }, checkpoint_dir, i_episode, args.experiment_name)
    
    writer.close()
    logging.info("Training completed")


if __name__ == "__main__":
    main()