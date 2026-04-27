import torch
import torch.nn as nn
from typing import Optional, Dict


class CNNProjector(nn.Module):
    """
    High-Fidelity Vision: Processes raw RGB/Depth images from Isaac Sim.
    Reduces (Batch, C, H, W) -> (Batch, N_tokens, d_model)
    """
    def __init__(self, d_model=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)) # Fixed 64 tokens
        )
        self.proj = nn.Linear(64, d_model)

    def forward(self, x):
        # x: (Batch, 3, H, W)
        x = self.conv(x) # (B, 64, 8, 8)
        x = x.flatten(2).transpose(1, 2) # (B, 64, 64)
        return self.proj(x)


class AdaptiveInputProjector(nn.Module):

    def __init__(self, d_model: int, default_input_dim: int = 512):
        super().__init__()
        self.d_model = d_model
        self.default_input_dim = default_input_dim
        # Default projector for backward compatibility
        self.default_proj = nn.Linear(default_input_dim, d_model)
        # Dynamic per-modality projectors (created on-demand)
        self.modality_projectors = nn.ModuleDict()

    def get_projector(self, input_dim: int, modal_id: str = "default") -> nn.Linear:
        """Get or create a projector for the given modality and input dimension."""
        if input_dim == self.default_input_dim and modal_id == "default":
            return self.default_proj

        key = f"{modal_id}_{input_dim}"
        if key not in self.modality_projectors:
            # Auto-create a new projector for this modality shape
            proj = nn.Linear(input_dim, self.d_model)
            # Initialize with Xavier for stable gradients on new modalities
            nn.init.xavier_uniform_(proj.weight)
            self.modality_projectors[key] = proj

        return self.modality_projectors[key]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Hook to auto-create modality projectors when loading a state dict."""
        for key in list(state_dict.keys()):
            if key.startswith(prefix + "modality_projectors."):
                # Format: modality_projectors.modal_id_dim.weight
                parts = key.replace(prefix + "modality_projectors.", "").split(".")
                if len(parts) >= 2:
                    full_id = parts[0] # modal_id_dim
                    if full_id not in self.modality_projectors:
                        # Infer input_dim from weight shape
                        if "weight" in parts[1]:
                            input_dim = state_dict[key].shape[1]
                            self.modality_projectors[full_id] = nn.Linear(input_dim, self.d_model)
        
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, sensor_data: torch.Tensor, modal_id: str = "default") -> torch.Tensor:
        """
        Project sensor data to d_model using the appropriate projector.

        Args:
            sensor_data: (Batch, N, input_dim) — raw sensor tensor.
            modal_id: Identifier for the sensor modality.

        Returns:
            Projected tokens: (Batch, N, d_model).
        """
        input_dim = sensor_data.size(-1)
        proj = self.get_projector(input_dim, modal_id)
        return proj(sensor_data)


class RecurrentLatentMemory(nn.Module):
    """
    Stateful Latents: Recurrent Latent Memory (RLM) circuit.
    Blends the previous latent state with fresh cross-attention output
    using a learnable gating mechanism (GRU-style).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # GRU-style gating: learns how much to retain vs update
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, fresh_latents: torch.Tensor,
                prev_latents: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Blend fresh latents with memory from the previous step.

        Args:
            fresh_latents: (Batch, n_latents, d_model) — current step output.
            prev_latents:  (Batch, n_latents, d_model) — previous step output, or None.

        Returns:
            Blended latents: (Batch, n_latents, d_model).
        """
        if prev_latents is None:
            return fresh_latents

        # Ensure device and shape alignment
        prev_latents = prev_latents.to(fresh_latents.device)

        # GRU-style gate: decides how much of the memory to retain
        combined = torch.cat([fresh_latents, prev_latents], dim=-1)  # (B, L, 2*d)
        gate = self.gate(combined)  # (B, L, d) values in [0, 1]

        # Blend: gate=1 means "keep fresh", gate=0 means "keep memory"
        blended = gate * fresh_latents + (1 - gate) * prev_latents
        return self.norm(blended)


class FusionCore(nn.Module):
    """
    OmniTrain FusionCore v3.0: Multimodal Transformer with
    Auto-Modality (adaptive input projections) and Stateful Latents
    (recurrent latent memory for temporal continuity).

    Forward signature is tensor-first for zero-overhead high-frequency inference.
    """

    def __init__(self, n_latents=128, d_model=512, n_heads=8, num_layers=3, input_dim=512):
        super().__init__()
        self.n_latents = n_latents
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.input_dim = input_dim

        # Learnable Queries (Latents)
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model))

        # Continuous Temporal Positional Encoding
        self.time_encoding = nn.Linear(1, d_model)

        # Auto-Modality: Input Projectors
        self.input_projector = AdaptiveInputProjector(d_model=d_model, default_input_dim=input_dim)
        self.cnn_projector = CNNProjector(d_model=d_model)

        # Cross-Attention: Sensor tokens -> Latents
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # Stateful Latents: Recurrent Latent Memory
        self.memory = RecurrentLatentMemory(d_model)

        # Reasoning Backbone
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True),
            num_layers=num_layers
        )

        # Legacy compatibility: keep a direct reference for old code paths
        self.token_proj = self.input_projector.default_proj

    def forward(self, sensor_data: torch.Tensor, timestamps: torch.Tensor,
                prev_latents: Optional[torch.Tensor] = None,
                modal_id: str = "default"):
        """
        Multimodal Forward Pass: Fuses sensor data with latent state.
        Supports both raw image data (CNN) and vectorized sensor data (Auto-Modality).
        """

        batch_size = sensor_data.size(0)
        
        # 1. Temporal Encoding (Continuous)
        temporal_encodings = self.time_encoding(timestamps)
        
        # 2. Vision vs Sensor projection
        if len(sensor_data.shape) == 4: # Is Image (B, C, H, W)
            tokens = self.cnn_projector(sensor_data)
        else:
            tokens = self.input_projector(sensor_data, modal_id)
            
        # 3. Fuse temporal information
        tokens_aware = tokens + temporal_encodings

        # 4. Expand latent queries for the batch
        latents = self.latents.expand(batch_size, -1, -1)

        # 5. Cross-Attention: Compress N variable tokens -> fixed latents
        attn_out, _ = self.cross_attn(query=latents, key=tokens_aware, value=tokens_aware)
        latents_fused = self.cross_attn_norm(latents + attn_out)

        # 6. Stateful Latents: Blend with memory from previous step
        latents_with_memory = self.memory(latents_fused, prev_latents)

        # 7. Temporal reasoning over fused latents
        return self.transformer(latents_with_memory)


class CfCCell(nn.Module):
    """
    Closed-form Continuous-time (CfC) Cell.
    A Liquid Neural Network building block that solves the underlying ODE 
    in closed form for incredible speed and stability.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Neural components for drive (f) and projection (g)
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size)
        
        # Time-scale parameters (Liquid aspect)
        self.tau = nn.Parameter(torch.randn(1, hidden_size))
        self.A = nn.Parameter(torch.ones(1, hidden_size))

        self.activation = nn.Tanh()
        
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, input_size) current input.
            h_prev: (Batch, hidden_size) previous liquid state.
            dt: (Batch, 1) time elapsed since last step.
        """
        # Drive: how the input and previous state want to push the system
        drive = self.activation(self.W_in(x) + self.W_rec(h_prev))
        
        # Time-decaying gating: The "Liquid" magic.
        # The flow of time directly alters the gating mechanism.
        time_gate = torch.sigmoid(-dt * (self.A + drive) / torch.exp(self.tau))
        
        # Closed-form ODE update: smooth interpolation between memory and new drive
        h_next = h_prev * time_gate + drive * (1 - time_gate)
        
        return h_next


class LiquidFusionCore(nn.Module):
    """
    OmniTrain Liquid Core v4.0 (MIT CfC Architecture).
    Replaces the discrete Transformer backbone with a Continuous-Time Liquid network.
    Massively reduces parameter count while improving Out-Of-Distribution robustness.
    """
    def __init__(self, n_latents=32, d_model=256, input_dim=512):
        super().__init__()
        self.n_latents = n_latents
        self.d_model = d_model
        self.input_dim = input_dim

        # Auto-Modality Projectors (Same as v3.0)
        self.input_projector = AdaptiveInputProjector(d_model=d_model, default_input_dim=input_dim)
        self.cnn_projector = CNNProjector(d_model=d_model)

        # Cross-Attention to compress spatial/sensor tokens into fixed latents
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # Spatial Self-Attention: Allows latents to share context before liquid evolution
        self.self_attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(d_model)

        # The Liquid Backbone (replaces Transformer and RecurrentLatentMemory)
        self.liquid_cell = CfCCell(d_model, d_model)

        # Legacy compatibility
        self.token_proj = self.input_projector.default_proj

    def forward(self, sensor_data: torch.Tensor, dt: torch.Tensor,
                prev_latents: Optional[torch.Tensor] = None,
                modal_id: str = "default"):
        """
        Args:
            sensor_data: (Batch, N_tokens, input_dim)
            dt: (Batch, 1) time elapsed since last inference in seconds.
            prev_latents: (Batch, n_latents, d_model) previous liquid state.
        """
        batch_size = sensor_data.size(0)
        device = sensor_data.device
        
        # 1. Vision vs Sensor projection
        if len(sensor_data.shape) == 4:
            tokens = self.cnn_projector(sensor_data)
        else:
            tokens = self.input_projector(sensor_data, modal_id)
            
        # 2. Expand latent queries
        latents = self.latents.expand(batch_size, -1, -1)

        # 3. Cross-Attention: Compress N variable tokens -> fixed latents
        attn_out, _ = self.cross_attn(query=latents, key=tokens, value=tokens)
        latents_fused = self.cross_attn_norm(latents + attn_out)

        # 3.5 Spatial Synergy: Latents communicate context before ODE evolution
        self_out, _ = self.self_attn(query=latents_fused, key=latents_fused, value=latents_fused)
        latents_spatial = self.self_attn_norm(latents_fused + self_out)

        # 4. Liquid State Evolution
        if prev_latents is None:
            prev_latents = torch.zeros_like(latents_spatial)

        # Apply CfC across all latents
        B, N, D = latents_spatial.shape
        x_flat = latents_spatial.reshape(B * N, D)
        h_flat = prev_latents.reshape(B * N, D)
        
        # Ensure dt is shaped correctly for the flattened batch (B*N, 1)
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        dt_flat = dt.view(B, 1).expand(B, N).reshape(B * N, 1)
        
        # Evolve the system
        h_next_flat = self.liquid_cell(x_flat, h_flat, dt_flat)
        
        return h_next_flat.reshape(B, N, D)
