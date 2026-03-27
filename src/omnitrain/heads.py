import torch
import torch.nn as nn


class OutputHead(nn.Module):
    """Base class for OmniTrain output heads."""

    def forward(self, x):
        # Global Average Pooling: flatten latents (Batch, N_latents, d_model) -> (Batch, d_model)
        return torch.mean(x, dim=1)


class ClassificationHead(OutputHead):
    """Classification head for discrete decision outputs (e.g., Safety: Safe/Emergency)."""

    def __init__(self, num_classes, d_model=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = super().forward(x)
        return self.net(x)


class RegressionHead(OutputHead):
    """Regression head for continuous control outputs (e.g., Motor DOF positions)."""

    def __init__(self, output_dim, d_model=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = super().forward(x)
        return self.net(x)
