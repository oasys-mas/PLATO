import torch
import json
import os
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available - graph visualization disabled")


import csv
import os
import torch

def save_to_csv(csv_path, episode, step, env_id, agent, 
                observation, action, logits, 
                create_headers=False):
    """
    Save observation data to CSV file
    
    Args:
        csv_path: Path to CSV file
        episode: Episode number
        step: Step number
        env_id: Environment ID
        agent: Agent name
        observation: Observation data
        action: Action taken
        logits: Model logits
        create_headers: True for first call to create headers
    """
    
    # Convert tensors to storable format
    if torch.is_tensor(observation):
        obs_str = str(observation.cpu().numpy().tolist())
    else:
        obs_str = str(observation)
    
    if torch.is_tensor(action):
        action_val = action.cpu().item()
    else:
        action_val = action
    
    if torch.is_tensor(logits):
        logits_str = str(logits.cpu().numpy().tolist())
    else:
        logits_str = str(logits)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Write to CSV
    mode = 'w' if create_headers else 'a'
    
    with open(csv_path, mode, newline='') as f:
        writer = csv.writer(f)
        
        if create_headers:
            writer.writerow(['episode', 'step', 'env_id', 'agent', 
                           'observation', 'action', 'logits'])
        
        writer.writerow([episode, step, env_id, agent, 
                        obs_str, action_val, logits_str])

def log_model_architecture(actor, critic, log_dir="network_logs"):
    """Simple model architecture logging - call once at start"""
    os.makedirs(log_dir, exist_ok=True)
    
    arch_info = {}
    
    # Actor info
    actor_params = sum(p.numel() for p in actor.parameters())
    actor_trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    arch_info['actor'] = {
        'total_params': actor_params,
        'trainable_params': actor_trainable,
        'architecture': str(actor)
    }
    
    # Critic info  
    critic_params = sum(p.numel() for p in critic.parameters())
    critic_trainable = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    arch_info['critic'] = {
        'total_params': critic_params,
        'trainable_params': critic_trainable,
        'architecture': str(critic)
    }
    
    # Save to file
    with open(os.path.join(log_dir, 'model_architecture.json'), 'w') as f:
        json.dump(arch_info, f, indent=2, default=str)
    
    print(f"Actor: {actor_params:,} params | Critic: {critic_params:,} params")
    return arch_info

def log_gradient_flow(actor, critic, episode, writer=None):
    """Detailed gradient flow logging - extends your existing grad logging"""
    
    # Actor gradient details
    actor_grad_info = []
    for name, param in actor.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            actor_grad_info.append({
                'layer': name,
                'grad_norm': grad_norm,
                'param_norm': param_norm,
                'ratio': grad_norm / (param_norm + 1e-8)
            })
    
    # Critic gradient details
    critic_grad_info = []
    for name, param in critic.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            critic_grad_info.append({
                'layer': name,
                'grad_norm': grad_norm,
                'param_norm': param_norm,
                'ratio': grad_norm / (param_norm + 1e-8)
            })
    
    # Check for gradient problems
    actor_vanishing = [g for g in actor_grad_info if g['grad_norm'] < 1e-6]
    actor_exploding = [g for g in actor_grad_info if g['grad_norm'] > 1e3]
    critic_vanishing = [g for g in critic_grad_info if g['grad_norm'] < 1e-6]
    critic_exploding = [g for g in critic_grad_info if g['grad_norm'] > 1e3]
    
    # Log to tensorboard if writer provided
    if writer:
        # Per-layer gradients
        for info in actor_grad_info:
            writer.add_scalar(f"PLATO-Debug/actor_grads/{info['layer']}", info['grad_norm'], episode)
        for info in critic_grad_info:
            writer.add_scalar(f"PLATO-Debug/critic_grads/{info['layer']}", info['grad_norm'], episode)
        
        # Problem counts
        writer.add_scalar("PLATO-Debug/actor_vanishing_layers", len(actor_vanishing), episode)
        writer.add_scalar("PLATO-Debug/actor_exploding_layers", len(actor_exploding), episode)
        writer.add_scalar("PLATO-Debug/critic_vanishing_layers", len(critic_vanishing), episode)
        writer.add_scalar("PLATO-Debug/critic_exploding_layers", len(critic_exploding), episode)
    
    # Return summary for your logging table
    gradient_summary = {
        'actor_vanishing': len(actor_vanishing),
        'actor_exploding': len(actor_exploding),
        'critic_vanishing': len(critic_vanishing),
        'critic_exploding': len(critic_exploding),
        'actor_max_grad': max([g['grad_norm'] for g in actor_grad_info]) if actor_grad_info else 0,
        'critic_max_grad': max([g['grad_norm'] for g in critic_grad_info]) if critic_grad_info else 0,
    }
    
    return gradient_summary

def log_network_graph(actor, critic, log_dir="network_logs"):
    """Create visual graphs of network architecture flow"""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available - skipping graph visualization")
        return
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Create actor graph
    _create_network_flow_graph(actor, "Actor", os.path.join(log_dir, "actor_graph.png"))
    
    # Create critic graph  
    _create_network_flow_graph(critic, "Critic", os.path.join(log_dir, "critic_graph.png"))
    
    print(f"Network graphs saved to {log_dir}/")

def _create_network_flow_graph(model, title, save_path):
    """Create a flow graph for a single network"""
    
    # Extract layer info
    layers = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_info = {
                'name': name.split('.')[-1] if '.' in name else name,
                'full_name': name,
                'type': type(module).__name__,
                'params': sum(p.numel() for p in module.parameters())
            }
            
            # Get input/output dimensions
            if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                layer_info['shape'] = f"{module.in_features}{module.out_features}"
            elif hasattr(module, 'hidden_size'):
                layer_info['shape'] = f"hidden={module.hidden_size}"
            elif hasattr(module, 'num_features'):
                layer_info['shape'] = f"features={module.num_features}"
            else:
                layer_info['shape'] = "N/A"
                
            layers.append(layer_info)
    
    if not layers:
        print(f"No layers found for {title}")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(layers) * 0.8)))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(layers) + 1)
    ax.axis('off')
    
    # Title
    ax.text(5, len(layers) + 0.5, f"{title} Network Flow", 
            fontsize=16, fontweight='bold', ha='center')
    
    # Colors for different layer types
    colors = {
        'Linear': '#4CAF50',
        'LSTM': '#2196F3', 
        'GRU': '#FF9800',
        'Conv1d': '#9C27B0',
        'Conv2d': '#9C27B0',
        'ReLU': '#FFC107',
        'Softmax': '#F44336',
        'Dropout': '#607D8B',
        'BatchNorm1d': '#795548',
        'LayerNorm': '#795548'
    }
    
    # Draw layers
    for i, layer in enumerate(layers):
        y_pos = len(layers) - i
        color = colors.get(layer['type'], '#9E9E9E')
        
        # Main layer box
        box = FancyBboxPatch(
            (1, y_pos - 0.3), 6, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1,
            alpha=0.8
        )
        ax.add_patch(box)
        
        # Layer text
        layer_text = f"{layer['name']} ({layer['type']})"
        ax.text(4, y_pos, layer_text, fontsize=10, fontweight='bold', 
                ha='center', va='center', color='white')
        
        # Shape and params info
        info_text = f"{layer['shape']} | {layer['params']:,} params"
        ax.text(8, y_pos, info_text, fontsize=8, ha='left', va='center')
        
        # Arrow to next layer
        if i < len(layers) - 1:
            ax.arrow(4, y_pos - 0.4, 0, -0.3, head_width=0.1, head_length=0.1, 
                    fc='black', ec='black', alpha=0.7)
    
    # Legend
    legend_elements = []
    used_types = set(layer['type'] for layer in layers)
    for layer_type in used_types:
        if layer_type in colors:
            legend_elements.append(mpatches.Patch(color=colors[layer_type], label=layer_type))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Summary stats
    total_params = sum(layer['params'] for layer in layers)
    ax.text(5, 0.3, f"Total Parameters: {total_params:,}", 
            fontsize=12, ha='center', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Network graph saved: {save_path}")

def log_training_flow_graph(log_dir="network_logs"):
    """Create a big picture graph showing how actor, critic, and training flow work together"""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available - skipping training flow graph")
        return
    
    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, "training_flow.png")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, "MAPPO Training Flow", fontsize=20, fontweight='bold', ha='center')
    
    # Define colors
    colors = {
        'environment': '#4CAF50',
        'actor': '#2196F3', 
        'critic': '#FF9800',
        'training': '#F44336',
        'data': '#9C27B0',
        'loss': '#795548'
    }
    
    # Helper function to draw boxes
    def draw_box(x, y, w, h, text, color, text_color='white'):
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=10, fontweight='bold', 
                ha='center', va='center', color=text_color)
    
    # Helper function to draw arrows
    def draw_arrow(x1, y1, x2, y2, text="", offset=0.3):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        if text:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2 + offset
            ax.text(mid_x, mid_y, text, fontsize=8, ha='center', 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # 1. Environment
    draw_box(8, 10, 3, 1, "Environment\n(Wildfire)", colors['environment'])
    
    # 2. Observations
    draw_box(3, 8.5, 2.5, 0.8, "Observations\n(tasks, agents)", colors['data'])
    draw_box(13, 8.5, 2.5, 0.8, "Graph Data\n(centralized)", colors['data'])
    
    # 3. Actor and Critic
    draw_box(3, 6.5, 3, 1.2, "ACTOR\n(Pointer Network)\ntask_features\nactions", colors['actor'])
    draw_box(13, 6.5, 3, 1.2, "CRITIC\n(GNN)\ncent_obs\nvalues", colors['critic'])
    
    # 4. Outputs
    draw_box(3, 4.5, 2.5, 0.8, "Actions\nlog_probs", colors['data'])
    draw_box(13, 4.5, 2.5, 0.8, "State Values", colors['data'])
    
    # 5. Environment interaction
    draw_box(8, 3, 3, 1, "Environment Step\nrewards, done", colors['environment'])
    
    # 6. GAE computation
    draw_box(8, 1.5, 4, 1, "GAE Computation\nadvantages, returns", colors['training'])
    
    # 7. Loss computation
    draw_box(3, 0.5, 3, 0.8, "Actor Loss\n(policy + entropy)", colors['loss'])
    draw_box(13, 0.5, 3, 0.8, "Critic Loss\n(value MSE)", colors['loss'])
    
    # Draw arrows for the flow
    # Environment to observations
    draw_arrow(7, 9.5, 4, 8.8, "obs")
    draw_arrow(9, 9.5, 12, 8.8, "cent_obs")
    
    # Observations to networks
    draw_arrow(3, 8, 3, 7.2, "task_features\nagent_features")
    draw_arrow(13, 8, 13, 7.2, "graph_data")
    
    # Networks to outputs
    draw_arrow(3, 5.8, 3, 5.2, "")
    draw_arrow(13, 5.8, 13, 5.2, "")
    
    # Actions to environment
    draw_arrow(4, 4.5, 7, 3.5, "actions")
    
    # Environment to GAE
    draw_arrow(8, 2.5, 8, 2.2, "rewards\nmasks")
    draw_arrow(12.5, 4.5, 9.5, 2, "values")
    
    # GAE to losses
    draw_arrow(6.5, 1.2, 4.5, 0.8, "advantages")
    draw_arrow(9.5, 1.2, 11.5, 0.8, "returns")
    
    # Gradient flow back
    draw_arrow(3, 0.1, 2.5, 5.8, "∇actor", -0.5)
    draw_arrow(13, 0.1, 13.5, 5.8, "∇critic", -0.5)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['environment'], label='Environment'),
        mpatches.Patch(color=colors['actor'], label='Actor Network'),
        mpatches.Patch(color=colors['critic'], label='Critic Network'),
        mpatches.Patch(color=colors['training'], label='Training Logic'),
        mpatches.Patch(color=colors['data'], label='Data Flow'),
        mpatches.Patch(color=colors['loss'], label='Loss Computation')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    # Add training details box
    details = """MAPPO Training Details:
• Actor: Pointer network (GRU/LSTM/MLP)
• Critic: GNN for centralized value function  
• PPO: Policy gradient with clipping
• GAE: Generalized Advantage Estimation
• Multi-agent: Shared critic, individual actors"""
    
    ax.text(15.5, 3, details, fontsize=9, ha='right', va='center',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training flow graph saved: {save_path}")

def log_gradient_flow_graph(actor, critic, episode, log_dir="network_logs"):
    """Create a graph showing gradient flow through both networks"""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available - skipping gradient flow graph")
        return
    
    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, f"gradient_flow_ep_{episode}.png")
    
    # Get gradient info
    actor_grads = []
    critic_grads = []
    
    for name, param in actor.named_parameters():
        if param.grad is not None:
            actor_grads.append({
                'name': name.split('.')[-2:],  # Last 2 parts of name
                'grad_norm': param.grad.norm().item()
            })
    
    for name, param in critic.named_parameters():
        if param.grad is not None:
            critic_grads.append({
                'name': name.split('.')[-2:],  # Last 2 parts of name  
                'grad_norm': param.grad.norm().item()
            })
    
    if not actor_grads and not critic_grads:
        return
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Actor gradients
    if actor_grads:
        names = ['.'.join(g['name']) for g in actor_grads]
        values = [g['grad_norm'] for g in actor_grads]
        
        bars1 = ax1.bar(range(len(names)), values, color='#2196F3', alpha=0.7)
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Gradient Norm')
        ax1.set_title(f'Actor Gradient Flow (Episode {episode})')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.set_yscale('log')
        
        # Color bars based on gradient magnitude
        for i, (bar, val) in enumerate(zip(bars1, values)):
            if val < 1e-6:
                bar.set_color('#F44336')  # Red for vanishing
            elif val > 1e3:
                bar.set_color('#FF9800')  # Orange for exploding
    
    # Critic gradients
    if critic_grads:
        names = ['.'.join(g['name']) for g in critic_grads]
        values = [g['grad_norm'] for g in critic_grads]
        
        bars2 = ax2.bar(range(len(names)), values, color='#FF9800', alpha=0.7)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_title(f'Critic Gradient Flow (Episode {episode})')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_yscale('log')
        
        # Color bars based on gradient magnitude
        for i, (bar, val) in enumerate(zip(bars2, values)):
            if val < 1e-6:
                bar.set_color('#F44336')  # Red for vanishing
            elif val > 1e3:
                bar.set_color('#FF9800')  # Orange for exploding
    
    # Add gradient health indicators
    ax1.axhline(y=1e-6, color='red', linestyle='--', alpha=0.5, label='Vanishing threshold')
    ax1.axhline(y=1e3, color='orange', linestyle='--', alpha=0.5, label='Exploding threshold')
    ax1.legend()
    
    ax2.axhline(y=1e-6, color='red', linestyle='--', alpha=0.5, label='Vanishing threshold') 
    ax2.axhline(y=1e3, color='orange', linestyle='--', alpha=0.5, label='Exploding threshold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gradient flow graph saved: {save_path}")