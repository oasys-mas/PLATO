import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PointerNetworkGRUEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU encoder instead of LSTM
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else 0.1
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize GRU weights"""
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, inputs, hidden_state=None):
        """
        Args:
            inputs: [sequence_length, input_dim] - variable length input sequence
            hidden_state: optional initial hidden state
        Returns:
            encoder_outputs: [sequence_length, hidden_dim] - encoded representations
            hidden_state: [num_layers, 1, hidden_dim] - final hidden state for decoder initialization
        """
        # Add batch dimension for GRU
        inputs = inputs.unsqueeze(0)  # [1, sequence_length, input_dim]
        
        # Encode sequence
        encoder_outputs, hidden_state = self.gru(inputs, hidden_state)
        
        # Remove batch dimension
        encoder_outputs = encoder_outputs.squeeze(0)  # [sequence_length, hidden_dim]
        
        return encoder_outputs, hidden_state

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_dim, attention_type="additive"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_type = attention_type  # "additive" or "dotproduct"

        if attention_type == "additive":
            self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v = nn.Linear(hidden_dim, 1, bias=False)
            self._init_weights()
        elif attention_type == "dotproduct":
            pass  # nothing to initialize

    def _init_weights(self):
        for module in [self.W1, self.W2, self.v]:
            nn.init.uniform_(module.weight, -0.1, 0.1)

    def forward(self, encoder_outputs, decoder_state):
        seq_len = encoder_outputs.shape[0]
        if self.attention_type == "additive":
            decoder_expanded = decoder_state.unsqueeze(0).expand(seq_len, -1)
            energy = torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_expanded))
            attention_scores = self.v(energy).squeeze(-1)
        elif self.attention_type == "dotproduct":
            # decoder_state: [hidden_dim]
            # encoder_outputs: [sequence_length, hidden_dim]
            attention_scores = torch.matmul(encoder_outputs, decoder_state)
        else:
            raise ValueError("Unknown attention type")

        temperature = 2.0
        attention_weights = F.softmax(attention_scores / temperature, dim=0)
        return attention_weights

class PointerNetworkGRUDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=1, attention_type="additive"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False
        )
        self.attention = AttentionMechanism(hidden_dim, attention_type=attention_type)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize GRU weights"""
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, encoder_outputs, initial_hidden):
        """
        Args:
            encoder_outputs: [sequence_length, hidden_dim] - encoder representations
            initial_hidden: [num_layers, 1, hidden_dim] - initial hidden state from encoder
        Returns:
            pointer_probs: [sequence_length] - pointing probabilities
        """
        # Use the final encoder hidden state as decoder state
        decoder_hidden = initial_hidden  # [num_layers, 1, hidden_dim]
        if decoder_hidden.dim() == 3:
            decoder_hidden = decoder_hidden[-1, 0, :]  # Take last layer, remove batch dim
        
        # Compute attention over encoder outputs
        attention_weights = self.attention(encoder_outputs, decoder_hidden)
        
        return attention_weights


class Pointer_GRU_Actor(nn.Module):
    """
    RECURRENT Pointer Network implementation using GRU - minimal changes to support hidden states
    """
    
    def __init__(self, task_feat_dim, agent_feat_dim, hidden_dim, num_layers=1, dropout=0.0, attention_type="additive"):
        super().__init__()
        
        self.task_feat_dim = task_feat_dim
        self.agent_feat_dim = agent_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers  # Store for hidden state initialization
        
        # NOTE: NOOP is now provided as input, not generated internally.
        # The caller must include NOOP as the last task in task_features.

        # Simple agent feature encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Simple task feature encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(task_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # RECURRENT: Memory GRU to process agent context with hidden state
        self.memory_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Pointer Network components
        self.encoder = PointerNetworkGRUEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        self.decoder = PointerNetworkGRUDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            attention_type=attention_type
        )
        self.memory_gru.flatten_parameters()
        self.encoder.gru.flatten_parameters()
        self.decoder.gru.flatten_parameters()
        # Context integration
        self.context_integrator = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize all weights uniformly as in the original paper"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -0.1, 0.1)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.1, 0.1)
    
    def forward(self, task_features, agent_features, hidden_state=None):
        """
        RECURRENT forward pass for GRU actor
        
        IMPORTANT: task_features MUST include NOOP as the last task.
        The actor does NOT generate NOOP internally - it expects NOOP to be provided.
        
        Args:
            task_features: [num_tasks, task_feat_dim] - available tasks (including NOOP as last)
            agent_features: [num_agents, agent_feat_dim] - self + other agents
            hidden_state: tensor - previous hidden state (optional)
        
        Returns:
            probs: [num_tasks] - pointing probabilities (including NOOP)
            next_hidden_state: tensor - updated hidden state for GRU
        """
        num_tasks = task_features.shape[0]
        device = task_features.device
        batch_size = 1  # Single agent inference
        
        # FIXED: Initialize hidden state with proper GRU format
        if hidden_state is None:
            hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        elif isinstance(hidden_state, torch.Tensor):
            # Ensure proper dimensions
            if hidden_state.dim() == 1:  # [hidden_dim]
                hidden_state = hidden_state.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
            elif hidden_state.dim() == 2:  # [num_layers, hidden_dim] or [batch_size, hidden_dim]
                if hidden_state.shape[0] == self.num_layers:  # [num_layers, hidden_dim]
                    hidden_state = hidden_state.unsqueeze(1)  # [num_layers, 1, hidden_dim]
                else:  # [batch_size, hidden_dim]
                    hidden_state = hidden_state.unsqueeze(0)  # [1, batch_size, hidden_dim]
            # else: already in correct format [num_layers, batch_size, hidden_dim]
        else:
            raise ValueError(f"GRU expected tensor hidden state, got {type(hidden_state)}")
        
        # Handle edge cases
        if num_tasks == 0:
            # Should not happen if NOOP is always included, but handle gracefully
            noop_prob = torch.ones(1, device=device, dtype=task_features.dtype)
            return noop_prob, hidden_state
        
        # 1. RECURRENT: Process agent context through memory GRU
        if agent_features.shape[0] > 0:
            encoded_agents = self.agent_encoder(agent_features)  # [num_agents, hidden_dim]
            
            # Use memory GRU to process agent context with hidden state
            agent_input = encoded_agents.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, hidden_dim]
            memory_output, next_hidden_state = self.memory_gru(agent_input, hidden_state)
            agent_context = memory_output.squeeze(0).squeeze(0)  # [hidden_dim]
        else:
            agent_context = torch.zeros(self.hidden_dim, device=device, dtype=task_features.dtype)
            next_hidden_state = hidden_state  # No update if no agent features
        
        # 2. Encode task features (NOOP is already included as last task)
        encoded_tasks = self.task_encoder(task_features)  # [num_tasks, hidden_dim]
        num_total = encoded_tasks.shape[0]
        
        # 3. Integrate agent context into task representations
        agent_context_expanded = agent_context.unsqueeze(0).expand(num_total, -1)
        contextualized_tasks = encoded_tasks + self.context_integrator(agent_context_expanded)
        
        # 4. Encode sequence with Pointer Network encoder
        encoder_outputs, encoder_hidden = self.encoder(contextualized_tasks)
        
        # 5. Decode to get pointing probabilities
        pointer_probs = self.decoder(encoder_outputs, encoder_hidden)
        
        # Sanity check
        if torch.isnan(pointer_probs).any() or torch.isinf(pointer_probs).any() or (pointer_probs < 0).any():
            print("Bad pointer_probs detected:", pointer_probs)
            pointer_probs = torch.ones_like(pointer_probs) / pointer_probs.shape[0]

        # Return probabilities and updated hidden state
        return pointer_probs, next_hidden_state