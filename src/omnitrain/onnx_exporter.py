import torch
import torch.nn as nn
import os

# --- MLOps: OmniTrain 1.0 CTMT Architecture (Mock implementation for ONNX Export) ---


class ContinuousTemporalEncoding(nn.Module):
    """Linear encoding of asynchronous timestamps."""

    def __init__(self, d_model):
        super().__init__()
        self.time_proj = nn.Linear(1, d_model)

    def forward(self, timestamps):
        # timestamps: (Batch, N, 1) -> (Batch, N, d_model)
        return self.time_proj(timestamps)


class LatentBottleneck(nn.Module):
    """State compression via Cross-Attention with Learnable Latents."""

    def __init__(self, n_latents, d_model, n_heads=8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sensor_tokens):
        # sensor_tokens: (Batch, N, d_model)
        batch_size = sensor_tokens.size(0)
        query = self.latents.expand(batch_size, -1, -1)

        # Cross-Attention: Q=Latents, K=Sensors, V=Sensors
        attn_out, _ = self.cross_attn(query, sensor_tokens, sensor_tokens)
        return self.norm(query + attn_out)


class OmniTrainONNX(nn.Module):
    """Unified Full Architecture for Graph Compilation."""

    def __init__(self, n_latents=64, d_model=768, n_layers=6):
        super().__init__()
        self.d_model = d_model

        # 1. Front-End: Time Projection and Fusion
        self.time_enc = ContinuousTemporalEncoding(d_model)
        self.token_proj = nn.Linear(512, d_model)  # Ingests 512 raw tokens

        # 2. Bottleneck: Compress N tokens to 64 latents
        self.bottleneck = LatentBottleneck(n_latents, d_model)

        # 3. Backbone: Temporal Reasoning
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. Decoupled Heads: Motor (Regression) and Safety (Classification)
        # Regression (B, 64, d_model) -> Pooling -> Motor Output
        self.motor_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 12)  # e.g., 12 DOF for Thor
        )

        self.safety_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary: Safe / Emergency
        )

    def forward(self, sensor_tokens, timestamps):
        # Input: (Batch, N, 512), (Batch, N, 1)

        # Temporal Encoding and Data Projection
        t_enc = self.time_enc(timestamps)
        s_proj = self.token_proj(sensor_tokens)

        # Continuous Fusion
        fused_input = s_proj + t_enc

        # Latent Compression (N variable -> 64 fixed)
        latents = self.bottleneck(fused_input)

        # Temporal Reasoning
        reasoning = self.backbone(latents)

        # Global Average Pooling over latents for final outputs
        fused_state = torch.mean(reasoning, dim=1)

        # Decoupled Outputs
        motor_control = self.motor_head(fused_state)
        safety_flag = self.safety_head(fused_state)

        return motor_control, safety_flag


# --- MLOps: Universal Export Script ---

def export_omnitrain_to_onnx(output_name="omni_1_0_edge.onnx"):
    model = OmniTrainONNX()
    model.eval()

    # DUMMY INPUTS: (Batch=1, N=100, Features=512)
    # N=100 as baseline, but will be dynamic.
    dummy_sensor_tokens = torch.randn(1, 100, 512)
    dummy_timestamps = torch.randn(1, 100, 1)

    print(f"📦 Exporting OmniTrain 1.0 to {output_name}...")

    torch.onnx.export(
        model,
        (dummy_sensor_tokens, dummy_timestamps),
        output_name,
        export_params=True,
        opset_version=15,  # Support for complex Transformers and Pooling
        do_constant_folding=True,
        input_names=['sensor_tokens', 'timestamps'],
        output_names=['motor_control', 'safety_flag'],
        dynamic_axes={
            'sensor_tokens': {1: 'num_tokens'},  # N is dynamic due to async sensors
            'timestamps': {1: 'num_tokens'},
            'motor_control': {0: 'batch_size'},
            'safety_flag': {0: 'batch_size'}
        }
    )

    if os.path.exists(output_name):
        print(f"✅ Success: Graph compiled. Size: {os.path.getsize(output_name)/1e6:.2f} MB")
    else:
        print("❌ Export failed.")


if __name__ == "__main__":
    export_omnitrain_to_onnx()
