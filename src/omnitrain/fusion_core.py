import torch
import torch.nn as nn
from typing import Optional, Dict


class AdaptiveInputProjector(nn.Module):
    """
    Auto-Modality: Dynamically creates and caches per-modality input projections.
    When a new sensor shape is encountered, a new Linear projector is automatically
    registered, eliminating the need to pre-configure input_dim.
    """

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

        # Auto-Modality: Adaptive input projection bank
        self.input_projector = AdaptiveInputProjector(d_model, default_input_dim=input_dim)

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
        Tensor-first forward pass with auto-modality and stateful memory.

        Args:
            sensor_data:  (Batch, N, input_dim) — pre-stacked sensor tensor.
            timestamps:   (Batch, N, 1)         — normalized timestamps.
            prev_latents: (Batch, n_latents, d_model) — optional previous state
                          for temporal continuity. Pass None for stateless mode.
            modal_id:     Modality identifier for auto-projection selection.

        Returns:
            Fused latent representation: (Batch, n_latents, d_model).
            Can be fed back as prev_latents for the next step.
        """
        batch_size = sensor_data.size(0)

        if sensor_data.size(1) == 0:
            return torch.zeros(batch_size, self.n_latents, self.d_model,
                               device=self.latents.device)

        # 1. Temporal Encoding
        temporal_encodings = self.time_encoding(timestamps)  # (B, N, d_model)

        # 2. Auto-Modality: Project raw sensor data using adaptive projector
        tokens = self.input_projector(sensor_data, modal_id)  # (B, N, d_model)

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
