"""
Example: Gesture Wand — OmniTrain Hello World
=============================================
A minimal end-to-end project that teaches OmniTrain by recognizing
hand gestures using a 3-axis IMU (accelerometer on a Raspberry Pi Pico / ESP32).

What this project does:
    1. Records labeled gesture CSV data from the device (via serial_logger)
    2. Trains a Liquid Neural Network (CfC) on the recorded data
    3. Exports the trained brain to a .omnibit file
    4. The microcontroller loads the brain and classifies gestures in real-time

Gestures:
    - "circle"   — rotating wrist in a circle
    - "shake"    — rapid back-and-forth motion
    - "tap"      — quick double-tap on a surface
    - "still"    — no movement (idle)

Run this file with:
    pip install omnitrain
    python train_gesture_wand.py
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from omnitrain.fusion_core import LiquidFusionCore
from omnitrain.esp32_exporter import ESP32Exporter

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────

INPUT_DIM   = 6         # 3-axis accel (ax, ay, az) + 3-axis gyro (gx, gy, gz)
D_MODEL     = 32        # Hidden state size (small for MCU)
N_LATENTS   = 4         # Number of latent vectors
N_CLASSES   = 4         # circle, shake, tap, still
SEQUENCE_LEN = 50       # 50 timesteps per gesture sample (at 50Hz = 1 second)
BATCH_SIZE  = 16
LR          = 1e-3
EPOCHS      = 30
EXPORT_DIR  = "exports/"


GESTURE_LABELS = {0: "still", 1: "circle", 2: "shake", 3: "tap"}


# ─────────────────────────────────────────────
# 2. Synthetic Dataset (replace with real CSV from serial_logger)
# ─────────────────────────────────────────────

def generate_synthetic_gesture_data(n_samples_per_class=100, seq_len=SEQUENCE_LEN):
    """
    Generates synthetic IMU data for 4 gesture classes.
    Replace this with real data from: python -m omnitrain.serial_logger
    """
    X, Y = [], []
    np.random.seed(42)

    for label in range(N_CLASSES):
        for _ in range(n_samples_per_class):
            t = np.linspace(0, 2 * np.pi, seq_len)
            noise = np.random.randn(seq_len, INPUT_DIM) * 0.1

            if label == 0:  # still — near-zero with noise
                sample = noise * 0.05
            elif label == 1:  # circle — sinusoidal in x-y plane
                sample = np.stack([
                    np.sin(t),     np.cos(t),     np.zeros(seq_len),
                    -np.cos(t)*2,  np.sin(t)*2,   np.zeros(seq_len)
                ], axis=1) + noise
            elif label == 2:  # shake — high-frequency oscillation
                sample = np.stack([
                    np.sin(t * 5),  np.zeros(seq_len), np.zeros(seq_len),
                    np.cos(t * 5) * 3, np.zeros(seq_len), np.zeros(seq_len)
                ], axis=1) + noise
            else:  # tap — brief spike then quiet
                sample = noise.copy()
                spike_t = seq_len // 4
                sample[spike_t:spike_t+3, 2] = 5.0  # Z-axis spike

            X.append(sample.astype(np.float32))
            Y.append(label)

    X = torch.tensor(np.array(X))  # [N, T, 6]
    Y = torch.tensor(Y)            # [N]
    return X, Y


# ─────────────────────────────────────────────
# 3. Model: LiquidFusionCore + Classification Head
# ─────────────────────────────────────────────

class GestureWand(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.backbone = LiquidFusionCore(
            n_latents=N_LATENTS,
            d_model=D_MODEL,
            input_dim=INPUT_DIM,
            config={}  # Legacy BioLiquidCell mode
        )
        # Classification head: pools the liquid latents → class logits
        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x_seq, dt_seq):
        """
        x_seq:  [B, T, INPUT_DIM]
        dt_seq: [B, T, 1]
        Returns: [B, N_CLASSES] logits
        """
        self.backbone.reset_state()
        B, T, _ = x_seq.shape

        latent_seq = self.backbone(x_seq, dt_seq)  # [B, T, N, D]
        # Pool: last timestep, mean across latents
        last_latent = latent_seq[:, -1].mean(dim=1)  # [B, D]
        return self.classifier(last_latent)


# ─────────────────────────────────────────────
# 4. Training Loop
# ─────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  OmniTrain Gesture Wand — Hello World Example")
    print("=" * 60)

    # Generate data
    print("\n[1/4] Generating synthetic gesture data...")
    X, Y = generate_synthetic_gesture_data(n_samples_per_class=80)
    n = len(X)
    split = int(n * 0.8)
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val     = X[split:], Y[split:]

    # Fixed dt: 50Hz sensor = 0.02s per step
    dt_train = torch.full((len(X_train), SEQUENCE_LEN, 1), 0.02)
    dt_val   = torch.full((len(X_val),   SEQUENCE_LEN, 1), 0.02)

    print(f"   Train: {len(X_train)} samples | Val: {len(X_val)} samples")

    # Model
    print("\n[2/4] Initializing GestureWand model...")
    model = GestureWand()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print(f"\n[3/4] Training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Mini-batch loop
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        for i in range(0, len(X_train), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            x_b, y_b, dt_b = X_train[idx], Y_train[idx], dt_train[idx]

            logits = model(x_b, dt_b)
            loss = criterion(logits, y_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                logits_val = model(X_val, dt_val)
                acc = (logits_val.argmax(dim=1) == Y_val).float().mean()
            print(f"   Epoch {epoch:3d}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {acc:.1%}")

    # Export
    print("\n[4/4] Exporting brain to .omnibit for microcontroller...")
    Path(EXPORT_DIR).mkdir(exist_ok=True)
    exporter = ESP32Exporter(output_dir=EXPORT_DIR)
    exporter.export(
        model.backbone,
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        output_dim=D_MODEL,
        filename="gesture_wand.omnibit"
    )
    print(f"\n✅ Done! Brain exported to: {EXPORT_DIR}gesture_wand.omnibit")
    print()
    print("Next steps:")
    print("  1. Copy gesture_wand.omnibit to your microcontroller's filesystem")
    print("  2. Flash firmware/gesture_wand_esp32/main.cpp to your ESP32")
    print("  3. Open serial monitor — the robot will classify gestures in real-time!")


if __name__ == "__main__":
    train()
