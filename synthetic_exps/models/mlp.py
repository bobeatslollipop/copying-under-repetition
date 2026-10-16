
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Standard MLP layer using GELU activation function.
    This replaces the standard GPTNeoXMLP in transformer models.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # First linear layer (hidden_size -> intermediate_size)
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        
        if getattr(config, "hidden_act", "gelu") == "gated_relu":
            self.dense_h_to_4h_v = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # Output linear layer (intermediate_size -> hidden_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(config.hidden_dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.dense_h_to_4h.weight)
        nn.init.zeros_(self.dense_h_to_4h.bias)
        
        if hasattr(self, 'dense_h_to_4h_v'):
            nn.init.xavier_uniform_(self.dense_h_to_4h_v.weight)
            nn.init.zeros_(self.dense_h_to_4h_v.bias)
            
        nn.init.xavier_uniform_(self.dense_4h_to_h.weight)
        nn.init.zeros_(self.dense_4h_to_h.bias)
    
    def forward(self, hidden_states):
        """
        Forward pass of Standard GELU MLP.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # First linear transformation with specified activation
        act = getattr(self.config, "hidden_act", "gelu")
        if act == "gelu":
            intermediate = F.gelu(self.dense_h_to_4h(hidden_states))
        elif act == "relu":
            intermediate = F.relu(self.dense_h_to_4h(hidden_states))
        elif act == "gated_relu":
            intermediate = self.dense_h_to_4h_v(hidden_states) * F.relu(self.dense_h_to_4h(hidden_states))
        elif act == "leaky_relu":
            intermediate = F.leaky_relu(self.dense_h_to_4h(hidden_states))
        else:
            intermediate = F.gelu(self.dense_h_to_4h(hidden_states))
        
        # Apply dropout
        intermediate = self.dropout(intermediate)
        
        # Final linear transformation
        output = self.dense_4h_to_h(intermediate)
        
        return output
