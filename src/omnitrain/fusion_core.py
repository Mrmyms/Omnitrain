import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List, Union
import logging


class CNNProjector(nn.Module):
    """
    High-Fidelity Vision: Processes raw RGB/Depth images.
    Reduces (Batch, C, H, W) -> (Batch, N_tokens, d_model)
    Also provides vision_pool() for Conectoma path that learns a
    spatial reduction instead of destroying information via mean().
    """
    def __init__(self, d_model=512, visual_tokens=64):
        super().__init__()
        grid_size = int(visual_tokens**0.5)
        self.visual_tokens = visual_tokens
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)) 
        )
        self.proj = nn.Linear(64, d_model)
        self.spatial_pool = nn.Linear(visual_tokens * d_model, d_model)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)
    
    def vision_pool(self, x):
        """Reduce (B, C, H, W) -> (B, d_model) via learned spatial pooling.
        Preserves spatial hierarchy unlike naive .mean(dim=1)."""
        tokens = self.forward(x)  # (B, N_tokens, d_model)
        B = tokens.size(0)
        flat = tokens.reshape(B, -1)  # (B, N_tokens * d_model)
        return self.spatial_pool(flat)  # (B, d_model)


class AdaptiveInputProjector(nn.Module):
    """
    Edge-Compatible Modality Projector: Pre-registers all projectors at initialization.
    """
    def __init__(self, d_model: int, input_configs: list = None, default_input_dim: int = 512):
        super().__init__()
        self.d_model = d_model
        self.default_input_dim = default_input_dim
        self.default_proj = nn.Linear(default_input_dim, d_model)
        self.modality_projectors = nn.ModuleDict()
        
        # Pre-register all necessary projectors for ONNX/JIT compatibility
        if input_configs:
            for inp in input_configs:
                m_id = inp.get('id', 'default')
                dim = inp.get('dim', 1)
                key = f"{m_id}_{dim}"
                if key not in self.modality_projectors:
                    self.modality_projectors[key] = nn.Linear(dim, d_model)

    def forward(self, sensor_data: torch.Tensor, modal_id: str = "default") -> torch.Tensor:
        input_dim = sensor_data.size(-1)
        if input_dim == self.default_input_dim and modal_id == "default":
            return self.default_proj(sensor_data)
            
        key = f"{modal_id}_{input_dim}"
        if key in self.modality_projectors:
            return self.modality_projectors[key](sensor_data)
        else:
            raise RuntimeError(f"Unregistered projector requested in forward pass: {key}. This breaks Edge deployment. Add to config.")

class ContinuousTemporalEncoding(nn.Module):
    """
    Projects the arrival time of a sensor pulse into a high-dimensional latent space 
    using a sinusoidal basis.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # omega_k = 1 / 10000^(2k/d_model)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

        self.amplitude = nn.Parameter(torch.ones(d_model))
        self.phase = nn.Parameter(torch.zeros(d_model // 2))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (Batch, 1) or (Batch,)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        sinusoid_inp = torch.einsum('bi,j->bij', t, self.inv_freq)
        sinusoid_inp = sinusoid_inp + self.phase
        emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        emb = emb * self.amplitude
        return emb


class SignalSpatialMixer(nn.Module):
    """
    O(N) Complexity Spatial Fusion with Cumulative Recurrence.
    Replaces traditional O(N^2) Cross-Attention with a Linear Recurrent state.
    
    Mathematical Improvements:
    1. Cumulative Update: S_t = S_{t-1} + (K^T @ V)
    2. Denominator Tracking: Z_t = Z_{t-1} + K
    3. Stable Readout: Y_t = (Q @ S_t) / (Q @ Z_t + epsilon)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        
        # Zero-bias initialization for linear attention stability
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, latents: torch.Tensor, tokens: torch.Tensor, 
                prev_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            latents: (B, N_q, D) - Query sources
            tokens:  (B, N_k, D) - Key/Value sources
            prev_state: (S_prev, Z_prev)
                S_prev: (B, D, D) - Cumulative KV matrix
                Z_prev: (B, D, 1) - Cumulative K vector (denominator state)
        """
        B, D = latents.size(0), latents.size(-1)
        
        # Kernel: ELU+1 ensures non-negativity for stable linear attention
        Q = torch.nn.functional.elu(self.q_proj(latents)) + 1.0  # (B, N_q, D)
        K = torch.nn.functional.elu(self.k_proj(tokens)) + 1.0   # (B, N_k, D)
        V = self.v_proj(tokens)                                    # (B, N_k, D)
        
        # 1. Compute current increments
        # K^T @ V -> (B, D, D)
        curr_kv = torch.bmm(K.transpose(-1, -2), V)
        # sum(K) -> (B, D, 1)
        curr_z = K.sum(dim=1, keepdim=True).transpose(-1, -2)
        
        # 2. Cumulative RNN-style Update
        if prev_state is not None:
            S_prev, Z_prev = prev_state
            S_curr = S_prev + curr_kv
            Z_curr = Z_prev + curr_z
        else:
            S_curr = curr_kv
            Z_curr = curr_z
            
        # 3. Normalized Readout (Prevents explosion without Softmax)
        # numerator = Q @ S_curr -> (B, N_q, D)
        numerator = torch.bmm(Q, S_curr)
        # denominator = Q @ Z_curr -> (B, N_q, 1)
        denominator = torch.bmm(Q, Z_curr) + 1e-4
        
        fused = numerator / denominator
        
        # Neural Gating & Skip Connection
        gate = torch.sigmoid(self.gate(fused))
        
        res = latents + (fused * gate)
        if hasattr(torch.nn.functional, 'layer_norm'):
            out = torch.nn.functional.layer_norm(res, self.norm.normalized_shape, self.norm.weight, self.norm.bias, self.norm.eps)
        else:
            out = self.norm(res)
        
        return out, (S_curr.detach() if not self.training else S_curr.detach().requires_grad_(True), 
                     Z_curr.detach() if not self.training else Z_curr.detach().requires_grad_(True))



class LeCun(nn.Module):
    """LeCun's scaled tanh activation: 1.7159 * tanh(0.666 * x)"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 1.7159 * torch.tanh(0.666 * x)


class BioLiquidCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, 
                 backbone_units: int = 128, backbone_layers: int = 1,
                 continual_learning: bool = False, mode: str = "default", 
                 eta: float = 0.001):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.continual_learning = continual_learning
        self.mode = mode  # default,no_gate, pure
        self.eta = eta
        self.backbone_units = backbone_units
        
        # Affine Sensory Mapping
        self.sensory_w = nn.Parameter(torch.ones(input_size))
        self.sensory_b = nn.Parameter(torch.zeros(input_size))
        
        # 1. State Backbone: Learns Target States (ff1, ff2)
        state_layers = [
            nn.Linear(input_size + hidden_size, backbone_units),
            LeCun(),
        ]
        for _ in range(1, backbone_layers):
            state_layers.append(nn.Linear(backbone_units, backbone_units))
            state_layers.append(LeCun())
        self.backbone_state = nn.Sequential(*state_layers)

        # 2. Time Backbone: Learns the Temporal Gate
        time_layers = [
            nn.Linear(input_size + hidden_size, backbone_units // 2),
            LeCun(),
        ]
        self.backbone_time = nn.Sequential(*time_layers)

        # Two Target State Heads
        self.ff1 = nn.Linear(backbone_units, hidden_size)
        self.ff2 = nn.Linear(backbone_units, hidden_size)
        
        # Learned Time Gate
        self.time_a = nn.Linear(backbone_units // 2, hidden_size)
        self.time_b = nn.Linear(backbone_units // 2, hidden_size)

        # Per-hidden-unit time scale for heterogeneous temporal dynamics
        self.time_scale = nn.Parameter(torch.ones(1, hidden_size))

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        if self.mode == "pure":
            self.w_tau = nn.Parameter(torch.ones(1, hidden_size))
        
        # Persistent Pruning Mask
        self.register_buffer('plastic_pruning_mask', torch.ones(1, backbone_units, hidden_size))

        # Dynamic state for continual learning
        self.w_plastic = None
        
        # Xavier Initialization
        self._init_weights()
        
    def _init_weights(self):
        for name, w in self.named_parameters():
            if w.dim() == 2 and w.requires_grad:
                if "w_tau" in name:
                    nn.init.constant_(w, 1.0)
                elif "time_a" in name:
                    nn.init.normal_(w, std=0.01)
                elif "time_b" in name:
                    nn.init.constant_(w, 0.0)
                else:
                    nn.init.xavier_uniform_(w)
            elif w.dim() == 1 and "time_b" in name:
                # Initialize bias to 0 to prevent early saturation
                nn.init.constant_(w, 0.0)
        
    def reset_plasticity(self):
        if self.continual_learning:
            if self.w_plastic is None:
                self.w_plastic = torch.zeros(1, self.backbone_units, self.hidden_size)
        if self.w_plastic is not None:
            self.w_plastic = torch.zeros_like(self.w_plastic)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        
        # 1. Affine Sensory Mapping
        x_mapped = x * self.sensory_w + self.sensory_b
        
        if self.mode == "pure":
            h_in = torch.zeros_like(h_prev)
        else:
            h_in = h_prev
            
        x_in = torch.cat([x_mapped, h_in], dim=-1)
        
        # 2. Decoupled Feature Extraction
        b_state = self.backbone_state(x_in)
        b_time  = self.backbone_time(x_in)
        
        # We use b_state as the primary feature representation for plasticity
        features = b_state
        
        # --- Continual Learning (Hebbian Plasticity on backbone output) ---
        if self.continual_learning and not self.training:
            if self.w_plastic is None or self.w_plastic.size(0) != B:
                self.w_plastic = torch.zeros(
                    (B, self.backbone_units, self.hidden_size), device=x.device
                )
            
            # Compute plastic activation efficiently using baddbmm
            h_plastic = torch.bmm(features.unsqueeze(1), self.w_plastic).squeeze(1)
            y = self.tanh(self.ff1(features) + h_plastic)
            
            # Autograd-safe Hebbian update
            with torch.no_grad():
                # Oja's Rule: Δw = η(x·y - y²·w)
                dw_hebb = torch.bmm(features.detach().unsqueeze(2), y.detach().unsqueeze(1))
                w_decay = (y.detach() ** 2).unsqueeze(1) * self.w_plastic
                
                delta_w = self.eta * (dw_hebb - w_decay)
                delta_w = delta_w * self.plastic_pruning_mask
                
                self.w_plastic = (self.w_plastic + delta_w).clamp_(-1.0, 1.0)

        elif self.training:
            self.w_plastic = None
        
        # Clamp dt for safety (prevents NaN on negative clock jitter)
        ts = torch.clamp(dt, min=0.0)
        
        # 3. Canonical CfC Mode Selection
        
        ts = ts * torch.abs(self.time_scale)

        if self.mode == "pure":
            
            # (Hasani et al. Eq. 10). Replaces simplified exponential decay
            # with learned sigmoidal gate for full expressivity.
            # Statelessness is preserved via h_in = zeros (line 303).
            ff1 = self.ff1(b_state)
            ff2 = self.ff2(b_state)
            t_interp = self.sigmoid(self.time_a(b_time) * ts + self.time_b(b_time))
            new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        else:
            # Full CfC
            ff1 = self.tanh(self.ff1(b_state))
            ff2 = self.tanh(self.ff2(b_state))
            
            # Sigmoid time-gate
            t_interp = self.sigmoid(self.time_a(b_time) * ts + self.time_b(b_time))
            
            if self.mode == "no_gate":
                new_hidden = ff1 + t_interp * ff2
            else:  # "default" — canonical CfC
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2

        
        return new_hidden


class NCPBackbone(nn.Module):
    def __init__(self, input_dim: int, sensory: int = 12, inter: int = 20, command: int = 8, motor: int = 4, continual_learning: bool = False, eta: float = 0.001):
        super().__init__()
        self.dims = (sensory, inter, command, motor)
        self.motor_layer = nn.Linear(command, motor)
        self.inter_cell = BioLiquidCell(sensory, inter, continual_learning=continual_learning, eta=eta)
        self.command_cell = BioLiquidCell(inter, command, continual_learning=continual_learning, eta=eta)
        
        self.sensory_layer = nn.Linear(input_dim, sensory)
        
    def forward(self, x: torch.Tensor, dt: torch.Tensor, h_prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B = x.size(0)
        h_inter_prev, h_command_prev = h_prev if h_prev else (
            torch.zeros(B, self.dims[1], device=x.device),
            torch.zeros(B, self.dims[2], device=x.device)
        )
        
        s_out = torch.relu(self.sensory_layer(x))
        h_inter = self.inter_cell(s_out, h_inter_prev, dt)
        h_command = self.command_cell(h_inter, h_command_prev, dt)
        m_out = self.motor_layer(h_command)
        
        return m_out, (h_inter, h_command)



class OmniBrainHub(nn.Module):
    """
    Neural Interconnection System.
    Manages multiple specialized NCP brains that communicate via a shared latent bus.
    """
    def __init__(self, input_dim: int, module_configs: Dict[str, dict], continual_learning: bool = False, eta: float = 0.001):
        super().__init__()
        self.brains = nn.ModuleDict()
        self.module_configs = module_configs
        self.input_dim = input_dim

        for b_id, cfg in module_configs.items():
            motor_dim = cfg.get('motor', input_dim)
            self.brains[b_id] = NCPBackbone(
                input_dim=input_dim,
                sensory=cfg.get('sensory', 12),
                inter=cfg.get('inter', 20),
                command=cfg.get('command', 8),
                motor=motor_dim,
                continual_learning=continual_learning,
                eta=eta
            )

    def forward(self, x: torch.Tensor, dt: torch.Tensor, h_states: Dict[str, Tuple]):
        B = x.size(0)
        brain_outputs = {}
        next_states = {}

        for b_id, brain in self.brains.items():
            inputs_from = self.module_configs[b_id].get('inputs_from', [])
            brain_x = x
            for src_id in inputs_from:
                if src_id in brain_outputs:
                    brain_x = brain_x + brain_outputs[src_id]

            out, state = brain(brain_x, dt, h_states.get(b_id))
            brain_outputs[b_id] = out
            next_states[b_id] = state

        final_out = torch.stack(list(brain_outputs.values())).mean(dim=0)
        return final_out, next_states


class LiquidFusionCore(nn.Module):
    """
    OmniTrain Universal Fusion Core v1.1.0 (Hub Enabled).
    Supports Single-Brain, NCP-Structured, and Multi-Brain Hub architectures.
    """
    def __init__(self, n_latents=32, d_model=256, input_dim=512, config: Optional[dict] = None, *args, **kwargs):
        super().__init__()
        self.config = config or {}
        self.n_latents = n_latents
        self.d_model = d_model
        self.input_dim = input_dim

        model_cfg = config.get('model', {}) if config else {}
        v_tokens = model_cfg.get('visual_tokens', 64)

        inputs_cfg = config.get('inputs', []) if config else []
        self.input_projector = AdaptiveInputProjector(d_model=d_model, input_configs=inputs_cfg, default_input_dim=input_dim)
        self.cnn_projector = CNNProjector(d_model=d_model, visual_tokens=v_tokens)

        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model))
        # O(N) SignalSpatialMixer
        self.use_fused_kernels = model_cfg.get('use_fused_kernels', True)
        self.precision = model_cfg.get('precision', 'fp32') # fp32, fp16, int8, nf4
        
        self.spatial_mixer = SignalSpatialMixer(d_model=d_model)
        self._spatial_mixer_uncompiled = self.spatial_mixer  # Keep reference for ONNX export
        if self.use_fused_kernels and hasattr(torch, 'compile'):
            # Optimization: Fused Attention and LayerNorm kernels
            self.spatial_mixer = torch.compile(self.spatial_mixer)

        # Continuous Temporal Encoding (CTE) — per-batch absolute time tracking
        self.temporal_encoder = ContinuousTemporalEncoding(d_model=d_model)
        
        # State buffers (managed via reset_state)
        self._abs_time_buf: Optional[torch.Tensor] = None
        self._last_brain_state: Optional[Dict] = None
        self._last_mixer_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


        conectoma_cfg = model_cfg.get('conectoma')
        hub_cfg = model_cfg.get('hub')
        ncp_cfg = model_cfg.get('ncp')
        continual = model_cfg.get('continual_learning', True) 
        eta = model_cfg.get('plasticity_rate', 0.001)

        if conectoma_cfg:
            self.brain_mode = "conectoma"
            # Build sensory mapping from input config
            sensory_n = conectoma_cfg.get('sensory_n', 4)
            sensory_map = {inp['id']: sensory_n for inp in config.get('inputs', [])}
            self.brain = BioConectomaHub(
                sensory_cfg=sensory_map,
                inter_n=conectoma_cfg.get('wall_n', 32),
                command_n=conectoma_cfg.get('command_n', 16),
                motor_n=d_model,
                d_model=d_model
            )
            # Learned per-latent modulation
            self.conectoma_broadcast = nn.Parameter(
                torch.randn(1, n_latents, d_model) * 0.02 + 1.0
            )
        elif hub_cfg:
            self.brain_mode = "hub"
            self.brain = OmniBrainHub(input_dim=d_model, module_configs=hub_cfg, continual_learning=continual, eta=eta)
        elif ncp_cfg and ncp_cfg.get('enabled', False):
            self.brain_mode = "ncp"
            self.brain = NCPBackbone(
                input_dim=d_model,
                sensory=ncp_cfg.get('sensory', 16),
                inter=ncp_cfg.get('inter', 32),
                command=ncp_cfg.get('command', 12),
                motor=d_model,
                continual_learning=continual,
                eta=eta
            )
        elif model_cfg.get('mixed_memory', False):
            self.brain_mode = "mixed"
            self.brain = MixedMemoryCfC(
                input_size=d_model,
                hidden_size=d_model,
                lstm_ratio=model_cfg.get('lstm_ratio', 0.25),
                mode="default"
            )
        else:
            self.brain_mode = "legacy"
            self.brain = BioLiquidCell(d_model, d_model, continual_learning=continual, eta=eta)

    def reset_state(self, batch_size: Optional[int] = None, device: Optional[torch.device] = None):
        """Reset hidden states and temporal buffers."""
        self._abs_time_buf = None
        self._last_brain_state = None
        self._last_mixer_state = None

        
        # Propagate reset to brain modules (clears plasticity history)
        for module in self.brain.modules():
            if hasattr(module, 'reset_plasticity'):
                module.reset_plasticity()

        if batch_size is not None and device is not None:
            self._abs_time_buf = torch.zeros(batch_size, 1, device=device, dtype=torch.float64)

    def forward(self, sensor_data, dt: torch.Tensor, prev_latents: Optional[torch.Tensor] = None, modal_id: Optional[str] = None, abs_time: Optional[torch.Tensor] = None, is_tokenized: bool = False):
        
        # Sequence Detection
        if dt.dim() == 3:
            return self._sequence_forward(sensor_data, dt, prev_latents, modal_id, is_tokenized=is_tokenized)
        
        if isinstance(sensor_data, dict):
            # Take batch size from the first modality in the dict
            batch_size = next(iter(sensor_data.values())).size(0)
        else:
            batch_size = sensor_data.size(0)
        
        # CTE Absolute Time Tracking
        if abs_time is None:
            dt_col = dt.view(batch_size, 1)
            if self._abs_time_buf is None or self._abs_time_buf.shape[0] != batch_size:
                if self._abs_time_buf is not None and self._abs_time_buf.shape[0] != batch_size:
                    logging.warning(
                        f"[FusionCore] Batch size changed ({self._abs_time_buf.shape[0]} -> {batch_size}), "
                        f"resetting temporal continuity."
                    )
                    # Also reset stateful components that depend on batch size
                    self._last_mixer_state = None
                    self._last_brain_state = None
                # Use float64 for time accumulation
                self._abs_time_buf = torch.zeros(batch_size, 1, device=dt.device, dtype=torch.float64)
            
            # Detach buffer to prevent gradient leakage across steps
            if self.training and self._abs_time_buf.requires_grad:
                self._abs_time_buf = self._abs_time_buf.detach()
                
            self._abs_time_buf = self._abs_time_buf + dt_col.to(torch.float64)
            abs_time = self._abs_time_buf.float()
        elif abs_time.dim() == 1:
            abs_time = abs_time.unsqueeze(-1)
        
        if prev_latents is not None and prev_latents.dim() == 4:
            prev_latents = prev_latents[:, -1]

        if self.brain_mode == "conectoma":
            # Conectoma naturally handles diverse sensor spaces asynchronously.
            # We skip the global token construction and CTE to save massive compute (O(N) tokens -> O(1) module vectors).
            projected_sensors = {}
            if isinstance(sensor_data, dict):
                for m_id, data in sensor_data.items():
                    if data.dim() == 4:
                        projected_sensors[m_id] = self.cnn_projector.vision_pool(data)  # Learned spatial pooling
                    else:
                        projected_sensors[m_id] = self.input_projector(data, m_id)
            else:
                # Single modality input
                if sensor_data.dim() == 4:
                    projected_sensors[modal_id] = self.cnn_projector.vision_pool(sensor_data)
                else:
                    projected_sensors[modal_id] = self.input_projector(sensor_data, modal_id)
            
            # 2. Time Evolution (Asynchronous Hub)
            if dt.dim() == 1: dt = dt.unsqueeze(-1)
            h_out, next_states = self.brain(projected_sensors, dt, self._last_brain_state or {})
            self._last_brain_state = next_states
            
            # Conectoma output uses learned per-latent modulation
            return h_out.unsqueeze(1) * self.conectoma_broadcast

        # ── LEGACY & HUB PATHS (DENSE FUSION) ──
        # 1. Project into semantic tokens
        if is_tokenized:
            tokens = sensor_data
        elif isinstance(sensor_data, dict):
            # Process Dict-based inputs
            fused_tokens = []
            for m_id, data in sensor_data.items():
                if data.dim() == 4: # Video/Vision: (B, C, H, W)
                    fused_tokens.append(self.cnn_projector(data))
                elif data.dim() == 2: # Structured data: (B, dim) -> (B, 1, dim)
                    fused_tokens.append(self.input_projector(data.unsqueeze(1), m_id))
                else:
                    fused_tokens.append(self.input_projector(data, m_id))
            tokens = torch.cat(fused_tokens, dim=1)
        else:
            # Process single input
            if sensor_data.dim() == 4:
                tokens = self.cnn_projector(sensor_data)
            elif sensor_data.dim() == 2:
                tokens = self.input_projector(sensor_data.unsqueeze(1), modal_id)
            else:
                tokens = self.input_projector(sensor_data, modal_id)
            
        # Apply Continuous Temporal Encoding (CTE)
        time_embeddings = self.temporal_encoder(abs_time)
        tokens = tokens + time_embeddings

        # Legacy/NCP/Hub paths (use spatial_mixer)
        latents = self.latents.expand(batch_size, -1, -1)
        # O(N) Spatial Fusion with Cumulative Recurrence
        latents_fused, next_mixer_state = self.spatial_mixer(
            latents, tokens, prev_state=self._last_mixer_state
        )
        self._last_mixer_state = next_mixer_state


        # 2. Time Evolution
        B, N, D = latents_fused.shape
        x_flat = latents_fused.reshape(B * N, D)
        if dt.dim() == 1: dt = dt.unsqueeze(-1)
        dt_flat = dt.view(B, 1).expand(B, N).reshape(B * N, 1)

        if self.brain_mode == "hub":
            h_out, next_states = self.brain(x_flat, dt_flat, self._last_brain_state or {})
            self._last_brain_state = next_states
            h_next = h_out
        elif self.brain_mode == "mixed":
            # MixedMemoryCfC requires (h_cfc, h_lstm, c_lstm)
            if self._last_brain_state is None:
                self._last_brain_state = self.brain.init_state(B * N, x_flat.device)
            
            h_next, next_state = self.brain(x_flat, self._last_brain_state, dt_flat)
            self._last_brain_state = next_state
        elif self.brain_mode == "ncp":
            # If prev_latents is a tuple, it's the full (output, internal_state)
            h_state = None
            if isinstance(prev_latents, tuple):
                h_state = prev_latents[1]
            elif self._last_brain_state is not None:
                h_state = self._last_brain_state
                
            h_next_flat, h_state_next = self.brain(x_flat, dt_flat, h_state)
            self._last_brain_state = h_state_next
            h_next = h_next_flat
        else:
            # Legacy mode: prev_latents is just the hidden state
            h_prev = prev_latents if prev_latents is not None else torch.zeros_like(x_flat)
            if h_prev.dim() == 3: h_prev = h_prev.reshape(B * N, D)
            h_next = self.brain(x_flat, h_prev, dt_flat)

        return h_next.reshape(B, N, D)

    def _sequence_forward(self, sensor_seq, dt_seq, prev_latents, modal_id, is_tokenized=False):
        B, T = dt_seq.shape[:2]
        
        # Support per-timestep modal_ids
        if isinstance(modal_id, (list, tuple)):
            modal_ids = modal_id
            assert len(modal_ids) == T, f"modal_id list length ({len(modal_ids)}) must match T ({T})"
        else:
            modal_ids = [modal_id] * T
        
        # Vectorized Projection and CTE
        # 1. Vectorized Spatial/Semantic Projection
        if is_tokenized:
            tokens_seq = sensor_seq
        elif isinstance(sensor_seq, dict):
            fused_tokens = []
            for m_id, data in sensor_seq.items():
                if data.dim() == 5: # (B, T, C, H, W)
                    flat_data = data.view(B*T, *data.shape[2:])
                    proj = self.cnn_projector(flat_data)
                    fused_tokens.append(proj.view(B, T, *proj.shape[1:]))
                else: # (B, T, dim)
                    flat_data = data.reshape(B*T, *data.shape[2:])
                    proj = self.input_projector(flat_data if flat_data.dim() == 3 else flat_data.unsqueeze(1), m_id)
                    fused_tokens.append(proj.view(B, T, *proj.shape[1:]))
            tokens_seq = torch.cat(fused_tokens, dim=2)
        else:
            if sensor_seq.dim() == 5:
                flat_data = sensor_seq.view(B*T, *sensor_seq.shape[2:])
                proj = self.cnn_projector(flat_data)
                tokens_seq = proj.view(B, T, *proj.shape[1:])
            else:
                # Check if modal_ids are heterogeneous
                unique_ids = set(modal_ids)
                if len(unique_ids) == 1:
                    # Homogeneous: vectorize all at once
                    flat_data = sensor_seq.reshape(B*T, *sensor_seq.shape[2:])
                    proj = self.input_projector(flat_data if flat_data.dim() == 3 else flat_data.unsqueeze(1), modal_ids[0])
                    tokens_seq = proj.view(B, T, *proj.shape[1:])
                else:
                    # Heterogeneous: project each timestep with its own modal_id
                    step_projs = []
                    for t_idx in range(T):
                        step_data = sensor_seq[:, t_idx]  # (B, ...)
                        proj = self.input_projector(
                            step_data.unsqueeze(1) if step_data.dim() == 2 else step_data,
                            modal_ids[t_idx]
                        )
                        step_projs.append(proj)
                    tokens_seq = torch.stack(step_projs, dim=1)
                
        # 2. Vectorized CTE (Continuous Temporal Encoding)
        dt_flat = dt_seq.squeeze(-1) if dt_seq.dim() == 3 else dt_seq
        # Compute cumsum in float64 for precision
        abs_times = torch.cumsum(dt_flat.to(torch.float64), dim=1) # (B, T)
        
        if self._abs_time_buf is not None:
            if self.training and self._abs_time_buf.requires_grad:
                self._abs_time_buf = self._abs_time_buf.detach()
            abs_times = abs_times + self._abs_time_buf
            
        time_emb_seq = self.temporal_encoder(abs_times.float()) # (B, T, D)
        time_emb_seq = time_emb_seq.unsqueeze(2) # (B, T, 1, D)
        tokens_seq = tokens_seq + time_emb_seq
        
        # Sync absolute time buffer with the end of the sequence
        self._abs_time_buf = abs_times[:, -1].unsqueeze(-1).detach()
        
        # 3. Recurrent Time Evolution
        outputs = []
        curr_latents = prev_latents
        for t in range(T):
            tokens = tokens_seq[:, t]
            dt_t = dt_seq[:, t]
            
            # Legacy/NCP/Hub paths (use spatial_mixer)
            latents = self.latents.expand(B, -1, -1)
            latents_fused, next_mixer_state = self.spatial_mixer(
                latents, tokens, prev_state=self._last_mixer_state
            )
            self._last_mixer_state = next_mixer_state
            
            x_flat = latents_fused.reshape(B * self.n_latents, -1)
            dt_flat_t = dt_t.view(B, 1).expand(B, self.n_latents).reshape(B * self.n_latents, 1)
            
            # Brain Step
            if self.brain_mode == "hub":
                h_out, next_states = self.brain(x_flat, dt_flat_t, self._last_brain_state or {})
                self._last_brain_state = next_states
                curr_latents = h_out
            elif self.brain_mode == "mixed":
                if self._last_brain_state is None:
                    self._last_brain_state = self.brain.init_state(B * self.n_latents, x_flat.device)
                curr_latents, next_state = self.brain(x_flat, self._last_brain_state, dt_flat_t)
                self._last_brain_state = next_state
            elif self.brain_mode == "ncp":
                h_state = None
                if isinstance(curr_latents, tuple):
                    h_state = curr_latents[1]
                elif self._last_brain_state is not None:
                    h_state = self._last_brain_state
                curr_latents, h_state_next = self.brain(x_flat, dt_flat_t, h_state)
                self._last_brain_state = h_state_next
            else: # legacy
                h_prev = curr_latents if curr_latents is not None else torch.zeros_like(x_flat)
                if h_prev.dim() == 3: h_prev = h_prev.reshape(B * self.n_latents, -1)
                curr_latents = self.brain(x_flat, h_prev, dt_flat_t)
            
            curr_latents = curr_latents.reshape(B, self.n_latents, -1)
            outputs.append(curr_latents)

        return torch.stack(outputs, dim=1)


FusionCore = LiquidFusionCore

class IrregularTimeManager:
    """
    Computes exact Δt per sensor modality from real timestamps.
    
    This unlocks the key advantage of CfC/LTC over LSTMs: each sensor
    (e.g. LiDAR at 40Hz, camera at 10Hz) gets its own correct Δt,
    so the liquid network can process asynchronous sensor streams natively.

    Usage:
        tm = IrregularTimeManager()
        dt_lidar = tm.get_dt("lidar", sensor_timestamp)
        dt_cam   = tm.get_dt("camera", sensor_timestamp)
    """
    def __init__(self, default_dt: float = 0.01):
        self._last_ts: Dict[str, float] = {}
        self.default_dt = default_dt

    def get_dt(self, modal_id: str, current_ts: float) -> float:
        """Return exact Δt since last reading for this sensor."""
        if modal_id not in self._last_ts:
            dt = self.default_dt
        else:
            dt = current_ts - self._last_ts[modal_id]
            dt = max(dt, 1e-6)  # Prevent dt=0 (division issues in CfC)
        self._last_ts[modal_id] = current_ts
        return dt

    def get_dt_tensor(self, modal_id: str, current_ts: float,
                      batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """Return Δt as a (batch_size,) tensor ready for BioLiquidCell."""
        dt = self.get_dt(modal_id, current_ts)
        return torch.full((batch_size,), dt, dtype=torch.float32, device=device)

    def reset(self, modal_id: Optional[str] = None):
        """Reset time tracking for one or all modalities."""
        if modal_id:
            self._last_ts.pop(modal_id, None)
        else:
            self._last_ts.clear()

class SparseWiring:
    """
    Generates a biologically-inspired sparse connectivity mask.
    Mimics the C. elegans connectome: sensory → inter → command → motor.
    Results in ~60-90% fewer active synapses vs fully-connected layers.
    """
    def __init__(self, units: int, output_dim: int, sparsity: float = 0.5, seed: int = 42):
        self.units = units
        self.output_dim = output_dim
        self.sparsity = sparsity
        torch.manual_seed(seed)
        # Recurrent mask: sparse within the hidden layer
        self.recurrent_mask = (torch.rand(units, units) > sparsity).float()
        self.recurrent_mask.fill_diagonal_(0)  # No self-connections
        # Input mask: all inputs allowed to all neurons
        self.input_mask = torch.ones(1, units)  # Will be broadcast

    @property
    def active_synapses(self) -> int:
        return int(self.recurrent_mask.sum().item())

    @property
    def total_synapses(self) -> int:
        return self.units * self.units


class WiredCfcCell(nn.Module):
    """
    Authentic Neural Circuit Policy (NCP) cell with sparse connectivity.
    
    Uses a fixed SparseWiring mask to enforce the C. elegans-inspired
    topology. Only ~40-50% of recurrent connections are active, giving
    interpretable, parameter-efficient dynamics.
    
    Implements the canonical CfC formula (Hasani et al., 2022) with
    sparse recurrence applied to a shared backbone layer.
    """
    def __init__(self, input_size: int, wiring: SparseWiring, mode: str = "default"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = wiring.units
        self.mode = mode

        self.register_buffer("recurrent_mask", wiring.recurrent_mask)

        # Shared input projection
        self.input_proj = nn.Linear(input_size, wiring.units)

        # Backbone (sparse recurrent weights via masking)
        self.backbone_W = nn.Linear(wiring.units, wiring.units, bias=True)
        self.backbone_act = LeCun()

        # Two target states (canonical CfC)
        self.ff1 = nn.Linear(wiring.units, wiring.units, bias=True)
        self.ff2 = nn.Linear(wiring.units, wiring.units, bias=True)
        
        # Learned time gate
        self.time_a = nn.Linear(wiring.units, wiring.units, bias=True)
        self.time_b = nn.Linear(wiring.units, wiring.units, bias=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # Xavier init
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                nn.init.xavier_uniform_(w)

    def _sparse_recurrent(self, layer: nn.Linear, h: torch.Tensor) -> torch.Tensor:
        """Apply linear layer with the sparse connectivity mask."""
        masked_weight = layer.weight * self.recurrent_mask
        return torch.nn.functional.linear(h, masked_weight, layer.bias)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor,
                dt: torch.Tensor) -> torch.Tensor:
        x_proj = self.input_proj(x)
        x_in = x_proj + h_prev  # Additive merge (sparse version)

        # Sparse backbone
        features = self.backbone_act(self._sparse_recurrent(self.backbone_W, x_in))

        # Canonical CfC
        ts = torch.clamp(dt, min=0.0)
        ff1 = self.tanh(self.ff1(features))
        ff2 = self.tanh(self.ff2(features))
        t_interp = self.sigmoid(self.time_a(features) * ts + self.time_b(features))
        
        if self.mode == "no_gate":
            return ff1 + t_interp * ff2
        else:  # "default" — canonical CfC
            return ff1 * (1.0 - t_interp) + t_interp * ff2


class MixedMemoryCfC(nn.Module):
    """
    Combines a BioLiquidCell (CfC) with a small LSTM auxiliary cell.

    CfC excels at fine-grained temporal dynamics but can suffer from
    vanishing gradients on very long sequences (> 200 steps).
    The LSTM auxiliary provides long-range gradient highways, while
    the CfC retains control of the fast dynamics.
    """
    def __init__(self, input_size: int, hidden_size: int,
                 lstm_ratio: float = 0.25, mode: str = "default"):
        super().__init__()
        self.hidden_size = hidden_size
        lstm_size = max(int(hidden_size * lstm_ratio), 8)
        self.lstm_size = lstm_size

        self.cfc  = BioLiquidCell(input_size, hidden_size, mode=mode)
        self.lstm = nn.LSTMCell(input_size, lstm_size)
        
        # Gated Fusion
        self.mixer = nn.Linear(hidden_size + lstm_size, hidden_size)
        self.gate  = nn.Linear(hidden_size + lstm_size, hidden_size)

    def init_state(self, batch_size: int, device: torch.device):
        h_cfc = torch.zeros(batch_size, self.hidden_size, device=device)
        h_lstm = torch.zeros(batch_size, self.lstm_size, device=device)
        c_lstm = torch.zeros(batch_size, self.lstm_size, device=device)
        return (h_cfc, h_lstm, c_lstm)

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Returns:
            h_out:   (B, hidden_size) — mixed hidden state
            next_state: (h_cfc, h_lstm, c_lstm)
        """
        h_cfc_prev, h_lstm_prev, c_lstm_prev = state
        
        # 1. Liquid Dynamics (CfC Branch)
        h_cfc_next = self.cfc(x, h_cfc_prev, dt)
        
        # 2. Memory Highway (LSTM Branch)
        h_lstm_next, c_lstm_next = self.lstm(x, (h_lstm_prev, c_lstm_prev))
        
        # 3. Gated Fusion
        cat = torch.cat([h_cfc_next, h_lstm_next], dim=-1)
        g = torch.sigmoid(self.gate(cat))
        h_mixed = (1.0 - g) * h_cfc_next + g * self.mixer(cat)
        
        return h_mixed, (h_cfc_next, h_lstm_next, c_lstm_next)


class NCPWiring:
    """
    Biologically-Inspired Static Mask Generator.
    Enforces a strict hierarchy: Sensory -> Wall (Inter) -> Command -> Motor.
    """
    def __init__(self, sensory_neurons: Dict[str, int], inter_n: int, command_n: int, motor_n: int, seed: int = 42):
        self.sensory_neurons = sensory_neurons
        self.total_sensory = sum(sensory_neurons.values())
        self.inter_n = inter_n
        self.command_n = command_n
        self.motor_n = motor_n
        
        gen = torch.Generator().manual_seed(seed)
        
        # 1. Sensory -> Inter (Feedforward)
        self.sens_inter_mask = (torch.rand((self.total_sensory, inter_n), generator=gen) < 0.3).float()
        
        # 2. Inter -> Inter (Recurrent Wall)
        self.inter_inter_mask = (torch.rand((inter_n, inter_n), generator=gen) < 0.2).float()
        self.inter_inter_mask.fill_diagonal_(0)
        
        # 3. Inter -> Command (Feedforward)
        self.inter_comm_mask = (torch.rand((inter_n, command_n), generator=gen) < 0.4).float()
        
        # 4. Command -> Command (Highly Recurrent)
        self.comm_comm_mask = (torch.rand((command_n, command_n), generator=gen) < 0.5).float()
        self.comm_comm_mask.fill_diagonal_(0)
        
        # 5. Command -> Motor (Feedforward)
        self.comm_motor_mask = (torch.rand((command_n, motor_n), generator=gen) < 0.6).float()
        
        # Island Prevention
        def ensure_connectivity(mask):
            # Ensure every row (output) has at least one connection
            rows = mask.sum(dim=1) == 0
            if rows.any():
                mask[rows, torch.randint(0, mask.size(1), (rows.sum(),), generator=gen)] = 1.0
            # Ensure every column (input) has at least one connection
            cols = mask.sum(dim=0) == 0
            if cols.any():
                mask[torch.randint(0, mask.size(0), (cols.sum(),), generator=gen), cols] = 1.0
            return mask
        
        self.sens_inter_mask = ensure_connectivity(self.sens_inter_mask)
        self.inter_inter_mask = ensure_connectivity(self.inter_inter_mask)
        self.inter_comm_mask = ensure_connectivity(self.inter_comm_mask)
        self.comm_comm_mask = ensure_connectivity(self.comm_comm_mask)
        self.comm_motor_mask = ensure_connectivity(self.comm_motor_mask)


class BioConectomaHub(nn.Module):
    """
    High-Fidelity Neural Hub implementing the 'Wall' architecture.
    Features:
    - Asynchronous Sensory Modules (isolated per modality)
    - Central Interneuron Wall (sparse & recurrent)
    - Command decision layer
    - CfC Stable Solver for all dynamics
    """
    def __init__(self, sensory_cfg: Dict[str, int], inter_n: int, command_n: int, motor_n: int, d_model: int):
        super().__init__()
        self.sensory_cfg = sensory_cfg
        self.inter_n = inter_n
        self.command_n = command_n
        self.motor_n = motor_n
        self.d_model = d_model
        
        # 1. Sensory Modules (Isolated)
        self.sensory_modules = nn.ModuleDict()
        self.sensory_offsets = {}
        curr_offset = 0
        for m_id, n in sensory_cfg.items():
            self.sensory_modules[m_id] = BioLiquidCell(d_model, n, mode="default")
            self.sensory_offsets[m_id] = curr_offset
            curr_offset += n
            
        # 2. The Wall (Interneurons)
        self.total_sensory = sum(sensory_cfg.values())
        self.wall_cell = BioLiquidCell(inter_n, inter_n, mode="default")
        
        # 3. Command Layer
        self.command_cell = BioLiquidCell(command_n, command_n, mode="default")
        
        # 4. Motor Output
        self.motor_proj = nn.Linear(command_n, d_model)
        
        # 5. Static Masks (Buffers)
        wiring = NCPWiring(sensory_cfg, inter_n, command_n, motor_n)
        # Register Buffers (for serialization and tracking)
        self.register_buffer("sens_inter_mask", wiring.sens_inter_mask)
        self.register_buffer("inter_inter_mask", wiring.inter_inter_mask)
        self.register_buffer("inter_comm_mask", wiring.inter_comm_mask)
        self.register_buffer("comm_comm_mask", wiring.comm_comm_mask)
        self.register_buffer("comm_motor_mask", wiring.comm_motor_mask)
        
        # Apply True Biological Sparsity to the internal BioLiquidCell weights
        self._apply_true_sparsity(self.wall_cell, self.inter_n, wiring.inter_inter_mask)
        self._apply_true_sparsity(self.command_cell, self.command_n, wiring.comm_comm_mask)
        
    def _apply_true_sparsity(self, cell: BioLiquidCell, input_size: int, recurrence_mask: torch.Tensor):
        # Apply NCP-style sparsity to the canonical CfC cell's linear heads.
        # The backbone processes [input, hidden] → backbone_units.
        # The heads (ff1, ff2, time_a, time_b) operate on backbone_units → hidden_size.
        # We apply recurrence sparsity to the head weights.
        hidden_size = recurrence_mask.size(0)
        backbone_units = cell.backbone_units
        
        # For the backbones' first layers: mask the recurrent (hidden) portion of the input
        for bb in [cell.backbone_state, cell.backbone_time]:
            bb_first = bb[0]  # nn.Linear(input_size + hidden_size, units)
            bb_units = bb_first.weight.size(0)
            bb_in = bb_first.weight.size(1)  # input_size + hidden_size
            
            input_part = torch.ones((bb_units, input_size), device=recurrence_mask.device)
            
            # Repeat the specific recurrence mask rows to fill backbone_units
            repeats = (bb_units + hidden_size - 1) // hidden_size
            rec_part_expanded = recurrence_mask.T.repeat(repeats, 1)[:bb_units]
            
            bb_mask = torch.cat([input_part, rec_part_expanded], dim=1)
            import torch.nn.utils.prune as prune
            prune.custom_from_mask(bb_first, name="weight", mask=bb_mask)

    def forward(self, step_sensors: Dict[str, torch.Tensor], dt: torch.Tensor, h_prev: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # h_prev contains: 'sensory' (B, total_sens), 'wall' (B, inter_n), 'command' (B, command_n)
        B = dt.size(0)
        device = dt.device
        
        h_sens_prev = h_prev.get('sensory', torch.zeros(B, self.total_sensory, device=device))
        h_wall_prev = h_prev.get('wall', torch.zeros(B, self.inter_n, device=device))
        h_comm_prev = h_prev.get('command', torch.zeros(B, self.command_n, device=device))
        
        # 1. Update Sensory Modules (Asynchronous Policy)
        h_sens_next = h_sens_prev.clone()
        for m_id, module in self.sensory_modules.items():
            offset = self.sensory_offsets[m_id]
            n = self.sensory_cfg[m_id]
            h_m_prev = h_sens_prev[:, offset:offset+n]
            
            x_m = step_sensors.get(m_id)
            if x_m is None:
                matching_keys = [k for k in step_sensors.keys() if k.startswith(f"{m_id}_")]
                if matching_keys:
                    # Average multiple sub-modality inputs (e.g. lidar_front, lidar_back)
                    x_m = torch.stack([step_sensors[k] for k in matching_keys]).mean(dim=0)
            
            if x_m is None:
                x_m = torch.zeros(B, module.input_size, device=device)
            
            h_m_next = module(x_m, h_m_prev, dt)
            h_sens_next[:, offset:offset+n] = h_m_next
            
        # 2. Update The Wall (Interneurons)
        # We apply the sparse mask to the sensory inputs flowing into the wall
        # In a real NCP, this would be a sparse linear layer. 
        # Here we simulate it by masking the input to the BioLiquidCell.
        wall_input = h_sens_next @ self.sens_inter_mask
        h_wall_next = self.wall_cell(wall_input, h_wall_prev, dt)
        
        # 3. Update Command Layer
        comm_input = h_wall_next @ self.inter_comm_mask
        h_comm_next = self.command_cell(comm_input, h_comm_prev, dt)
        
        # 4. Final Motor Output
        motor_out = self.motor_proj(h_comm_next)
        
        next_states = {
            'sensory': h_sens_next,
            'wall': h_wall_next,
            'command': h_comm_next
        }
        
        return motor_out, next_states
