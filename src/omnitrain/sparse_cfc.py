import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseCfC(nn.Module):
    """
    Universal Arbitrary Connectome CfC.
    Uses an explicit adjacency matrix to define the graph topology.
    Only the connections specified in the adjacency matrix are evaluated.
    f, g, and h mappings are element-wise to strictly enforce the connectome topology.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, adjacency_matrix: torch.Tensor):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Adjacency matrix: shape (hidden_dim, input_dim + hidden_dim)
        # Mask is 1.0 if connection exists, 0.0 otherwise.
        assert adjacency_matrix.shape == (hidden_dim, input_dim + hidden_dim)
        self.register_buffer('mask', adjacency_matrix.float())
        
        # The main weight matrix for cross-neuron communication
        self.backbone_weight = nn.Parameter(torch.Tensor(hidden_dim, input_dim + hidden_dim))
        self.backbone_bias = nn.Parameter(torch.Tensor(hidden_dim))
        
        # Element-wise parameters for the ODE (keeps the connectome topology strict)
        self.f_weight = nn.Parameter(torch.Tensor(hidden_dim))
        self.f_bias = nn.Parameter(torch.Tensor(hidden_dim))
        
        self.g_weight = nn.Parameter(torch.Tensor(hidden_dim))
        self.g_bias = nn.Parameter(torch.Tensor(hidden_dim))
        
        self.h_weight = nn.Parameter(torch.Tensor(hidden_dim))
        self.h_bias = nn.Parameter(torch.Tensor(hidden_dim))
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self._hsize = hidden_dim

        self._init_weights()

    @property
    def _hsize_prop(self):
        return self._hsize

    def _init_weights(self):
        nn.init.xavier_uniform_(self.backbone_weight)
        nn.init.zeros_(self.backbone_bias)
        
        nn.init.ones_(self.f_weight)
        nn.init.zeros_(self.f_bias)
        
        nn.init.ones_(self.g_weight)
        nn.init.zeros_(self.g_bias)
        
        nn.init.ones_(self.h_weight)
        nn.init.zeros_(self.h_bias)
        
        # Apply mask immediately to weights
        with torch.no_grad():
            self.backbone_weight.mul_(self.mask)

    def forward(self, x: torch.Tensor, times: torch.Tensor):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self._hsize, device=x.device)
        out = []
        
        for t in range(seq_len):
            dt = torch.zeros(batch, 1, device=x.device) if t == 0 else (times[:, t, :] - times[:, t-1, :])
            
            # Apply sparsity mask
            masked_weight = self.backbone_weight * self.mask
            
            x_in = torch.cat([x[:, t, :], h], dim=-1)
            bb = F.linear(x_in, masked_weight, self.backbone_bias)
            bb = torch.tanh(bb)
            
            # Element-wise f, g, h
            f_val = bb * self.f_weight + self.f_bias
            g_val = bb * self.g_weight + self.g_bias
            h_val = bb * self.h_weight + self.h_bias
            
            t_gate = torch.sigmoid(-f_val * dt)
            h = t_gate * torch.tanh(g_val) + (1.0 - t_gate) * torch.tanh(h_val)
            out.append(h.unsqueeze(1))
            
        return self.fc(torch.cat(out, dim=1))
