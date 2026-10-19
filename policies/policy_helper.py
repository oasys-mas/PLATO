import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# # OBSOLEte TODO
# GPU optimization utility
def enable_actor_optimizations(model):
    """Enable additional optimizations for actor models"""
    if torch.cuda.is_available():
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='reduce-overhead')
                print(f"{model.__class__.__name__} compiled for optimization")
            except:
                print(f" Compilation not available for {model.__class__.__name__}")
    return model


def build_masked_dist_from_logits(logits: torch.Tensor, valid_task_indices, include_noop: bool = True):
    """
    Mask invalid actions in *logit* space and return a Categorical.
    logits: [N+1] (last index is NOOP)
    valid_task_indices: iterable of absolute indices in [0..N-1] that are valid
    """
    T = logits.numel()
    mask_add = torch.full((T,), float("-inf"), device=logits.device, dtype=logits.dtype)

    if valid_task_indices is not None and len(valid_task_indices) > 0:
        vi = torch.as_tensor(valid_task_indices, device=logits.device, dtype=torch.long)
        vi = vi[(vi >= 0) & (vi < T-1)]           # keep only task indices
        mask_add[vi] = 0.0
    if include_noop and T > 0:
        mask_add[-1] = 0.0                       # allow NOOP

    masked_logits = logits + mask_add

    # If everything is -inf, recover (keep NOOP)
    if not torch.isfinite(masked_logits).any():
        masked_logits = torch.full_like(logits, float("-inf"))
        if T > 0:
            masked_logits[-1] = 0.0

    return torch.distributions.Categorical(logits=masked_logits)

def compute_agents_encoding(agent_features):
    """
    Compute agents encoding using self features as primary identifier,
    plus context from other agents via statistics.
    
    IMPORTANT: First row should be the SELF agent, others are context.
    This preserves agent identity for proper action selection.
    
    Args:
        agent_features: torch.Tensor of shape [num_agents, feat_dim]
                       First row = self, remaining rows = others
    
    Returns:
        agents_encoding: torch.Tensor of shape [4 * feat_dim]
                        [self_features, others_mean, others_var, self_position_encoding]
    """
    if agent_features.shape[0] == 0:
        # Handle empty agents case
        feat_dim = agent_features.shape[1] if agent_features.dim() > 1 else 1
        return torch.zeros(4 * feat_dim, device=agent_features.device, dtype=agent_features.dtype)
    
    feat_dim = agent_features.shape[1] if agent_features.dim() > 1 else agent_features.shape[0]
    
    # CRITICAL: First row is SELF - this identifies the agent
    self_feat = agent_features[0]  # [feat_dim]
    
    if agent_features.shape[0] == 1:
        # Only self, no others - use self for all components
        others_mean = torch.zeros_like(self_feat)
        others_var = torch.zeros_like(self_feat)
        position_encoding = self_feat  # Self-relative position
    else:
        # Compute statistics over OTHER agents (not including self)
        others = agent_features[1:]  # [num_others, feat_dim]
        others_mean = torch.mean(others, dim=0)
        others_var = torch.var(others, dim=0, unbiased=False) if others.shape[0] > 1 else torch.zeros_like(self_feat)
        
        # Relative position encoding: how different is self from others?
        position_encoding = self_feat - others_mean
    
    # Concatenate: [self, others_context, variance, relative_position]
    agents_encoding = torch.cat([self_feat, others_mean, others_var, position_encoding])
    
    return agents_encoding
# Batch processing utility
def batch_actor_forward(actor, task_features_batch, agent_features_batch, device):
    """
    Batch processing for multiple agents
    """
    results = []
    
    with torch.no_grad():
        for task_feats, agent_feats in zip(task_features_batch, agent_features_batch):
            task_feats = task_feats.to(device, non_blocking=True)
            agent_feats = agent_feats.to(device, non_blocking=True)
            probs, attention_scores = actor(task_feats, agent_feats)
            results.append((probs, attention_scores))
    
    return results