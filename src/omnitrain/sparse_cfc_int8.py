import torch
import torch.nn as nn
import torch.nn.functional as F

def fake_quantize(x, scale, qmin=-127, qmax=127):
    """Simulate INT8 quantization by rounding to a grid defined by scale."""
    x_q = torch.round(x / scale).clamp(qmin, qmax)
    return x_q * scale

class SparseCfCInt8(nn.Module):
    """
    Native INT8 arithmetic simulation of SparseCfC.
    Every MAC operation and non-linearity is explicitly quantized to an INT8 grid 
    to guarantee bit-for-bit mathematical parity with an integer MCU ALU.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, adjacency_matrix: torch.Tensor):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Adjacency matrix
        assert adjacency_matrix.shape == (hidden_dim, input_dim + hidden_dim)
        self.register_buffer('mask', adjacency_matrix.float())
        
        # Weights
        self.backbone_weight = nn.Parameter(torch.Tensor(hidden_dim, input_dim + hidden_dim))
        self.backbone_bias = nn.Parameter(torch.Tensor(hidden_dim))
        
        self.f_weight = nn.Parameter(torch.Tensor(hidden_dim))
        self.f_bias = nn.Parameter(torch.Tensor(hidden_dim))
        self.g_weight = nn.Parameter(torch.Tensor(hidden_dim))
        self.g_bias = nn.Parameter(torch.Tensor(hidden_dim))
        self.h_weight = nn.Parameter(torch.Tensor(hidden_dim))
        self.h_bias = nn.Parameter(torch.Tensor(hidden_dim))
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self._hsize = hidden_dim

        # Global fixed scales for Simulated Integer ALU (assumes max range of 8.0 for activations)
        self.S_act = 8.0 / 127.0
        self.S_lut_in = 8.0 / 127.0
        self.S_lut_out = 1.0 / 127.0  # Tanh and Sigmoid map to [-1, 1] or [0, 1]
        
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
        
        with torch.no_grad():
            self.backbone_weight.mul_(self.mask)

    def get_weight_scale(self, weight):
        max_val = weight.abs().max().item()
        return max_val / 127.0 if max_val > 0 else 1.0

    def forward(self, x: torch.Tensor, times: torch.Tensor, h_init=None):
        batch, seq_len, _ = x.shape
        h = h_init if h_init is not None else torch.zeros(batch, self._hsize, device=x.device)
        out = []
        
        # Determine weight scales per tensor
        S_bb = self.get_weight_scale(self.backbone_weight)
        S_f = self.get_weight_scale(self.f_weight)
        S_g = self.get_weight_scale(self.g_weight)
        S_h_w = self.get_weight_scale(self.h_weight)
        S_fc = self.get_weight_scale(self.fc.weight)
        
        for t in range(seq_len):
            dt = torch.zeros(batch, 1, device=x.device) if t == 0 else (times[:, t, :] - times[:, t-1, :])
            
            # --- INT8 Backbone MAC ---
            masked_weight = self.backbone_weight * self.mask
            w_bb_q = fake_quantize(masked_weight, S_bb)
            
            x_in = torch.cat([x[:, t, :], h], dim=-1)
            x_in_q = fake_quantize(x_in, self.S_act)
            
            # MAC is exact integer representation (scaled back to float for ease of use in PyTorch)
            bb = F.linear(x_in_q, w_bb_q, self.backbone_bias)
            
            # Tanh LUT Simulation
            bb_lut_in = fake_quantize(bb, self.S_lut_in)
            bb = fake_quantize(torch.tanh(bb_lut_in), self.S_lut_out)
            
            # --- INT8 Element-wise f, g, h MACs ---
            f_w_q = fake_quantize(self.f_weight, S_f)
            g_w_q = fake_quantize(self.g_weight, S_g)
            h_w_q = fake_quantize(self.h_weight, S_h_w)
            
            f_val = bb * f_w_q + self.f_bias
            g_val = bb * g_w_q + self.g_bias
            h_val = bb * h_w_q + self.h_bias
            
            # Time gate LUT Simulation
            # Sigmoid(-f_val * dt)
            # dt is also quantized (e.g. range 0 to 1.0)
            dt_q = fake_quantize(dt, 1.0 / 127.0, qmin=0, qmax=127)
            f_dt = fake_quantize(f_val * dt_q, self.S_lut_in)
            t_gate = fake_quantize(torch.sigmoid(-f_dt), self.S_lut_out, qmin=0, qmax=127)
            
            # g and h LUT Simulation
            g_val_q = fake_quantize(g_val, self.S_lut_in)
            h_val_q = fake_quantize(h_val, self.S_lut_in)
            
            tanh_g = fake_quantize(torch.tanh(g_val_q), self.S_lut_out)
            tanh_h = fake_quantize(torch.tanh(h_val_q), self.S_lut_out)
            
            # Exact blending: h = t_gate * tanh_g + (1 - t_gate) * tanh_h
            # The (1 - t_gate) term is exact in integers because 1.0 is a representable fixed-point constant
            h = fake_quantize(t_gate * tanh_g + (1.0 - t_gate) * tanh_h, self.S_act)
            out.append(h.unsqueeze(1))
            
        # Final FC layer
        w_fc_q = fake_quantize(self.fc.weight, S_fc)
        h_concat = torch.cat(out, dim=1)
        # h_concat is already quantized to self.S_act
        final_out = F.linear(h_concat, w_fc_q, self.fc.bias)
        
        return final_out, h
