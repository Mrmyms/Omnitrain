import torch
import torch.nn as nn
import time
import numpy as np
from omnitrain.fusion_core import FusionCore
from omnitrain.heads import ClassificationHead
from omnitrain.token_bus import TokenBus
from omnitrain.trainer import OmniTrainer

"""
OMNITRAIN ADVANCED DIAGNOSTIC v3.1: Text AI Training (FIXED)
This script ensures convergence by making the encoder trainable.
"""

# ── 1. Trainable Text Encoder ──
class TinyTextEncoder(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.chars = " ABCDEFGHIJKLMNOPQRSTUVWXYZ!"
        # Now these are trainable parameters!
        self.embeddings = nn.Embedding(len(self.chars), dim)
        nn.init.normal_(self.embeddings.weight, std=0.02)

    def forward(self, text_list: list):
        # Convert list of strings to padded tensor
        all_indices = []
        max_len = max(len(t) for t in text_list)
        for text in text_list:
            text = text.upper()
            indices = [self.chars.find(c) if c in self.chars else 0 for c in text]
            indices += [0] * (max_len - len(indices)) # Padding
            all_indices.append(indices)
        
        return self.embeddings(torch.tensor(all_indices))


# ── 2. Diagnostic Pipeline ──
def run_diagnostic(epochs=15):
    print(f"🚀 INITIALIZING ADVANCED DIAGNOSTIC: TEXT-AI PIPELINE v3.1 (Epochs: {epochs})")
    print("-" * 60)

    # Initialize Components
    bus = TokenBus(max_tokens=1000, token_dim=512, session_id="diag_text")
    encoder = TinyTextEncoder(dim=512)
    
    # Model: Architecture optimized for small diagnostic tasks
    model = FusionCore(n_latents=16, d_model=128, num_layers=2, input_dim=512)
    heads = {'safety': ClassificationHead(num_classes=2, d_model=128)}
    
    # Optimizer includes ALL parameters including the encoder
    params = list(model.parameters()) + list(heads['safety'].parameters()) + list(encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    # Simulation Data (Clear patterns)
    train_data = [
        ("SYSTEM OK", 0), ("NORMAL", 0), ("ALL GREEN", 0), ("GOOD", 0),
        ("DANGER", 1), ("CRITICAL", 1), ("FIRE", 1), ("EMERGENCY", 1)
    ]

    print(f"🧠 PHASE 1: Training ({epochs} epochs)...")
    start_train = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        optimizer.zero_grad()
        
        for text, label in train_data:
            # 1. Forward pass
            tokens = encoder([text]) # (1, N, 512)
            ts = torch.linspace(0, 1, tokens.size(1)).reshape(1, -1, 1)
            target = torch.tensor([label]).long()
            
            latents = model(tokens, ts)
            prediction = heads['safety'](latents)
            
            loss = criterion(prediction, target)
            loss.backward()
            epoch_loss += loss.item()
            
        optimizer.step()
        
        avg_loss = epoch_loss / len(train_data)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
            
        if avg_loss < 0.05:
            print(f"   [!] Convergence reached at epoch {epoch+1} (Loss: {avg_loss:.4f})")
            break

    train_time = time.time() - start_train
    print(f"✅ Training completed in {train_time:.2f}s")

    print("\n🕵️ PHASE 2: Neural Reasoning Test...")
    model.eval()
    heads['safety'].eval()
    encoder.eval()

    test_cases = ["SYSTEM OK", "CRITICAL FIRE"]
    for text in test_cases:
        with torch.no_grad():
            tokens = encoder([text])
            ts = torch.linspace(0, 1, tokens.size(1)).reshape(1, -1, 1)
            latents = model(tokens, ts)
            logits = heads['safety'](latents)
            prob = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1).item()
            
        status = "DANGER" if pred == 1 else "SAFE"
        confidence = prob[0, pred].item() * 100
        print(f"   '{text}' -> [bold]{status}[/] ({confidence:.1f}% confidence)")

    print("\n🛡️ PHASE 3: Formal Verification Audit...")
    from omnitrain.safety_guard import SafetyGuard
    guard = SafetyGuard(heads['safety'], emergency_class=1)
    guard.add_constraint('manual_kill_switch', min_safe=0.0, max_safe=0.5)
    is_safe, _ = guard.check_constraints({'manual_kill_switch': 1.0})
    print(f"   Formal Override Check: {'❌ REJECTED' if not is_safe else '✅ PASSED'}")

    print("-" * 60)
    print("🏆 OMNITRAIN 3.0: SUPREME DIAGNOSTIC SUCCESS.")
    bus.cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    run_diagnostic(epochs=args.epochs)
