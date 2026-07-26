"""
Connectome-Guided Mixed-Precision SparseCfC (QAT-Ready).

Each functional core of the NCP connectome can be independently quantized
to a different precision level (INT4 / INT8 / FP16 / FP32). During training,
fake-quantization with Straight-Through Estimator (STE) gradients enables
Quantization-Aware Training. The ES outer loop evolves the precision genotype.

Cores:
  - sensory:  backbone columns [0 : input_dim]        (LiDAR → hidden)
  - inter:    g_weight, g_bias, h_weight, h_bias       (memory / state)
  - command:  fc output layer                           (hidden → action)
  - timegate: f_weight, f_bias                          (ODE decay rate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────
#  Precision Definitions
# ─────────────────────────────────────────────────────────────────────

PRECISION_LEVELS = {
    0: {'name': 'int4',  'bits': 4,  'qmin': -8,   'qmax': 7},
    1: {'name': 'int8',  'bits': 8,  'qmin': -127, 'qmax': 127},
    2: {'name': 'fp16',  'bits': 16, 'qmin': None,  'qmax': None},
    3: {'name': 'fp32',  'bits': 32, 'qmin': None,  'qmax': None},
}


def fake_quantize(x: torch.Tensor, precision: int, scale: Optional[float] = None) -> torch.Tensor:
    """
    Simulate quantization during training using the Straight-Through Estimator.
    
    For integer types (INT4/INT8): round to grid, clamp, scale back.
    For FP16: cast down and back up (captures rounding errors).
    For FP32: identity (no-op).
    """
    cfg = PRECISION_LEVELS[precision]
    
    if precision == 3:  # FP32 — no quantization
        return x
    
    if precision == 2:  # FP16 — simulate half-precision rounding
        return x.half().float()
    
    # INT4 or INT8 — fixed-point fake quantization
    if scale is None:
        max_val = x.detach().abs().max()
        scale = max_val / cfg['qmax'] if max_val > 0 else 1.0
    
    x_q = torch.round(x / scale).clamp(cfg['qmin'], cfg['qmax'])
    return x_q * scale


class LearnedStepSize(torch.autograd.Function):
    """
    Learned Step Size Quantization (LSQ) — ICLR 2020.
    
    The quantization scale becomes a learnable parameter, allowing the 
    network to jointly optimize weights AND their quantization grid during
    QAT. The gradient of the scale is approximated via STE with a 
    normalization factor of 1/sqrt(Qp * n_elements).
    
    This is the key advantage over FedCFC's naive post-training quantization:
    our ES discovers the precision level, and LSQ optimizes the scale within
    that precision during QAT fine-tuning.
    """
    @staticmethod
    def forward(ctx, x, scale, qmin, qmax, n_elements):
        ctx.save_for_backward(x, scale)
        ctx.other = (qmin, qmax, n_elements)
        x_q = torch.round(x / scale).clamp(qmin, qmax)
        return x_q * scale
    
    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        qmin, qmax, n_elements = ctx.other
        
        x_q = x / scale
        # STE: pass gradient through where x is within quantization range
        below = (x_q < qmin).float()
        above = (x_q > qmax).float()
        between = 1.0 - below - above
        
        grad_x = grad_output * between
        
        # Scale gradient (LSQ formula)
        grad_scale = grad_output * (
            torch.round(x_q).clamp(qmin, qmax) - x_q
        ) * between + grad_output * qmin * below + grad_output * qmax * above
        grad_scale = grad_scale.sum() / max(n_elements ** 0.5, 1.0)
        
        return grad_x, grad_scale, None, None, None


def fake_quantize_lsq(x: torch.Tensor, learned_scale: torch.Tensor, 
                       precision: int) -> torch.Tensor:
    """
    LSQ-based fake quantization with learnable scale parameter.
    Falls back to standard fake_quantize for FP16/FP32.
    """
    cfg = PRECISION_LEVELS[precision]
    
    if precision >= 2:  # FP16 or FP32
        return fake_quantize(x, precision)
    
    return LearnedStepSize.apply(
        x, learned_scale.abs(), 
        float(cfg['qmin']), float(cfg['qmax']),
        x.numel()
    )


def compute_scale(x: torch.Tensor, precision: int) -> float:
    """Compute the quantization scale factor for a tensor at a given precision."""
    cfg = PRECISION_LEVELS[precision]
    if cfg['qmax'] is None:
        return 1.0
    max_val = x.detach().abs().max().item()
    return max_val / cfg['qmax'] if max_val > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────
#  Quantization Genotype
# ─────────────────────────────────────────────────────────────────────

@dataclass
class QuantGenotype:
    """
    4-gene genotype encoding the precision of each functional core.
    Each gene is an integer index into PRECISION_LEVELS.
    """
    sensory:  int = 1   # Default: INT8
    inter:    int = 1   # Default: INT8
    command:  int = 1   # Default: INT8
    timegate: int = 2   # Default: FP16

    def to_list(self) -> list:
        return [self.sensory, self.inter, self.command, self.timegate]

    @classmethod
    def from_list(cls, genes: list) -> 'QuantGenotype':
        return cls(sensory=genes[0], inter=genes[1], command=genes[2], timegate=genes[3])

    def memory_bits(self, input_dim: int, hidden_dim: int, output_dim: int) -> int:
        """Estimate total model memory in bits given dimensions."""
        # Sensory: backbone columns for sensor input
        sensory_params = hidden_dim * input_dim
        # Recurrent backbone: hidden × hidden (not sensory, part of inter)
        recurrent_params = hidden_dim * hidden_dim
        # Inter: g_weight + g_bias + h_weight + h_bias + recurrent backbone
        inter_params = (hidden_dim * 2) + (hidden_dim * 2) + recurrent_params
        # Command: fc layer
        command_params = output_dim * hidden_dim + output_dim
        # Timegate: f_weight + f_bias
        timegate_params = hidden_dim * 2

        bits = 0
        bits += sensory_params  * PRECISION_LEVELS[self.sensory]['bits']
        bits += inter_params    * PRECISION_LEVELS[self.inter]['bits']
        bits += command_params  * PRECISION_LEVELS[self.command]['bits']
        bits += timegate_params * PRECISION_LEVELS[self.timegate]['bits']
        return bits

    def memory_bytes(self, input_dim: int, hidden_dim: int, output_dim: int) -> float:
        return self.memory_bits(input_dim, hidden_dim, output_dim) / 8.0

    def __repr__(self):
        names = [PRECISION_LEVELS[g]['name'] for g in self.to_list()]
        return (f"QuantGenotype(sensory={names[0]}, inter={names[1]}, "
                f"command={names[2]}, timegate={names[3]})")


# ─────────────────────────────────────────────────────────────────────
#  SparseCfCMixed — The Core Module
# ─────────────────────────────────────────────────────────────────────

class SparseCfCMixed(nn.Module):
    """
    Connectome-Guided Mixed-Precision SparseCfC.
    
    Extends SparseCfC with per-core fake quantization for QAT.
    The adjacency matrix defines the NCP topology, and the QuantGenotype
    controls the arithmetic precision of each functional core independently.
    
    During training, fake_quantize injects quantization noise through STE.
    During export, weights are packed at their native bit-width.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        adjacency_matrix: torch.Tensor,
        genotype: Optional[QuantGenotype] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Default genotype: INT8 everywhere, FP16 for timegate
        self.genotype = genotype or QuantGenotype()
        
        # Adjacency matrix: shape (hidden_dim, input_dim + hidden_dim)
        assert adjacency_matrix.shape == (hidden_dim, input_dim + hidden_dim)
        self.register_buffer('mask', adjacency_matrix.float())
        
        # Backbone weight: columns [0:input_dim] = sensory, [input_dim:] = recurrent
        self.backbone_weight = nn.Parameter(torch.Tensor(hidden_dim, input_dim + hidden_dim))
        self.backbone_bias = nn.Parameter(torch.Tensor(hidden_dim))
        
        # Timegate parameters (ODE decay — precision-critical)
        self.f_weight = nn.Parameter(torch.Tensor(hidden_dim))
        self.f_bias = nn.Parameter(torch.Tensor(hidden_dim))
        
        # Inter-neuron state parameters (memory)
        self.g_weight = nn.Parameter(torch.Tensor(hidden_dim))
        self.g_bias = nn.Parameter(torch.Tensor(hidden_dim))
        self.h_weight = nn.Parameter(torch.Tensor(hidden_dim))
        self.h_bias = nn.Parameter(torch.Tensor(hidden_dim))
        
        # Command output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self._hsize = hidden_dim
        
        # ── Learnable Quantization Scales (LSQ) ──
        # Initialized to 1.0, updated via STE during QAT
        self.scale_sensory = nn.Parameter(torch.tensor(1.0))
        self.scale_recurrent = nn.Parameter(torch.tensor(1.0))
        self.scale_timegate = nn.Parameter(torch.tensor(1.0))
        self.scale_inter = nn.Parameter(torch.tensor(1.0))
        self.scale_command = nn.Parameter(torch.tensor(1.0))
        
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
    
    def set_genotype(self, genotype: QuantGenotype):
        """Hot-swap the quantization genotype (used by the ES loop)."""
        self.genotype = genotype
    
    def load_from_fp32(self, fp32_model):
        """
        Load weights from a pre-trained FP32 SparseCfC model.
        This is the starting point for QAT fine-tuning.
        """
        self.backbone_weight.data.copy_(fp32_model.backbone_weight.data)
        self.backbone_bias.data.copy_(fp32_model.backbone_bias.data)
        self.f_weight.data.copy_(fp32_model.f_weight.data)
        self.f_bias.data.copy_(fp32_model.f_bias.data)
        self.g_weight.data.copy_(fp32_model.g_weight.data)
        self.g_bias.data.copy_(fp32_model.g_bias.data)
        self.h_weight.data.copy_(fp32_model.h_weight.data)
        self.h_bias.data.copy_(fp32_model.h_bias.data)
        self.fc.weight.data.copy_(fp32_model.fc.weight.data)
        self.fc.bias.data.copy_(fp32_model.fc.bias.data)
    
    def _quantize_backbone(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Apply mixed-precision fake quantization to the backbone weight matrix.
        Sensory columns (0:input_dim) use sensory precision.
        Recurrent columns (input_dim:) use inter precision.
        """
        # Split into sensory and recurrent partitions
        w_sensory  = weight[:, :self.input_dim]
        w_recurrent = weight[:, self.input_dim:]
        
        # Quantize each partition independently using LSQ
        w_sensory_q  = fake_quantize_lsq(w_sensory,  self.scale_sensory, self.genotype.sensory)
        w_recurrent_q = fake_quantize_lsq(w_recurrent, self.scale_recurrent, self.genotype.inter)
        
        return torch.cat([w_sensory_q, w_recurrent_q], dim=1)
    
    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self._hsize, device=x.device)
        out = []
        
        g = self.genotype
        
        for t in range(seq_len):
            dt = torch.zeros(batch, 1, device=x.device) if t == 0 else (times[:, t, :] - times[:, t-1, :])
            
            # ── Backbone MAC (mixed-precision) ──
            masked_weight = self.backbone_weight * self.mask
            q_weight = self._quantize_backbone(masked_weight)
            
            x_in = torch.cat([x[:, t, :], h], dim=-1)
            bb = F.linear(x_in, q_weight, self.backbone_bias)
            bb = torch.tanh(bb)
            
            # ── Timegate (ODE solver — highest precision) ──
            f_w_q = fake_quantize_lsq(self.f_weight, self.scale_timegate, g.timegate)
            f_b_q = fake_quantize_lsq(self.f_bias,   self.scale_timegate, g.timegate)
            f_val = bb * f_w_q + f_b_q
            
            # ── Inter-neuron state (memory cores) ──
            g_w_q = fake_quantize_lsq(self.g_weight, self.scale_inter, g.inter)
            g_b_q = fake_quantize_lsq(self.g_bias,   self.scale_inter, g.inter)
            h_w_q = fake_quantize_lsq(self.h_weight, self.scale_inter, g.inter)
            h_b_q = fake_quantize_lsq(self.h_bias,   self.scale_inter, g.inter)
            
            g_val = bb * g_w_q + g_b_q
            h_val = bb * h_w_q + h_b_q
            
            # ── ODE Time-Gate ──
            t_gate = torch.sigmoid(-f_val * dt)
            h = t_gate * torch.tanh(g_val) + (1.0 - t_gate) * torch.tanh(h_val)
            out.append(h.unsqueeze(1))
        
        # ── Command output (quantized fc) ──
        fc_w_q = fake_quantize_lsq(self.fc.weight, self.scale_command, g.command)
        fc_b_q = fake_quantize_lsq(self.fc.bias,   self.scale_command, g.command)
        
        return F.linear(torch.cat(out, dim=1), fc_w_q, fc_b_q)
    
    def get_precision_report(self) -> Dict[str, str]:
        """Human-readable report of the current precision configuration."""
        g = self.genotype
        mem = g.memory_bytes(self.input_dim, self.hidden_dim, self.output_dim)
        fp32_mem = QuantGenotype(3,3,3,3).memory_bytes(self.input_dim, self.hidden_dim, self.output_dim)
        savings = (1.0 - mem / fp32_mem) * 100
        
        return {
            'sensory':  PRECISION_LEVELS[g.sensory]['name'],
            'inter':    PRECISION_LEVELS[g.inter]['name'],
            'command':  PRECISION_LEVELS[g.command]['name'],
            'timegate': PRECISION_LEVELS[g.timegate]['name'],
            'memory_bytes': f"{mem:.0f}",
            'fp32_bytes': f"{fp32_mem:.0f}",
            'savings_pct': f"{savings:.1f}%",
        }
