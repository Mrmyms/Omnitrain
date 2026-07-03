import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from omnitrain.fusion_core import BioLiquidCell
from omnitrain.esp32_exporter import ESP32Exporter

class ContinuousCfC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.cell = BioLiquidCell(input_size=input_dim, hidden_size=hidden_dim, backbone_units=32)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, times):
        batch_size, seq_len, _ = x.shape
        hidden = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)
        outputs = []
        for t in range(seq_len):
            xt = x[:, t, :]
            dt = times[:, t, :]
            if t > 0:
                dt = times[:, t, :] - times[:, t-1, :]
            else:
                dt = torch.zeros_like(dt)
            hidden = self.cell(xt, hidden, dt)
            outputs.append(hidden.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return self.fc(outputs)

def train_model(model, X, Y, T, epochs=50):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.HuberLoss()
    
    X_t = torch.FloatTensor(X).unsqueeze(0)
    Y_t = torch.FloatTensor(Y).unsqueeze(0)
    T_t = torch.FloatTensor(T).unsqueeze(0)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t, T_t)
        loss = criterion(pred, Y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
    return model

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Loading CartPole data...")
    X_0 = np.load("data/pendulum_X_0loss.npy")
    Y = np.load("data/pendulum_Y.npy")
    T = np.load("data/pendulum_T.npy")
    
    hidden_dim = 16
    model = ContinuousCfC(4, hidden_dim, 1)
    
    print("Training HIL model...")
    model = train_model(model, X_0, Y, T, epochs=50)
    
    print("Exporting...")
    class DummyModel(nn.Module):
        def __init__(self, brain):
            super().__init__()
            self.brain = brain
    export_model = DummyModel(model.cell)
    
    exporter = ESP32Exporter()
    out_path = exporter.export(
        model=export_model,
        input_dim=4,
        d_model=hidden_dim,
        output_dim=1,
        backbone_units=32,
        filename="hil_model.omnibit"
    )
    print(f"Exported to {out_path}")
