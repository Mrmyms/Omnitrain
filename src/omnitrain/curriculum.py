import torch
from dataclasses import dataclass
from typing import List

@dataclass
class TrainingPhase:
    name: str
    start_epoch: int
    noise_level: int
    description: str

class CurriculumScheduler:
    """
    Manages the lifecycle of a training process. 
    Allows users to progressively introduce complexity (Noise) to the Liquid Core.
    """
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.phases: List[TrainingPhase] = []
        self.current_epoch = 0
        
    def add_phase(self, name: str, start_epoch: int, noise_level: int, description: str = ""):
        """Register a new training phase."""
        self.phases.append(TrainingPhase(name, start_epoch, noise_level, description))
        # Keep phases sorted chronologically
        self.phases.sort(key=lambda x: x.start_epoch)
        
    def get_current_phase(self) -> TrainingPhase:
        """Returns the active phase based on the current epoch."""
        if not self.phases:
            return TrainingPhase("Default", 0, 0, "No phases defined")
            
        active_phase = self.phases[0]
        for phase in self.phases:
            if self.current_epoch >= phase.start_epoch:
                active_phase = phase
        return active_phase
        
    def step(self):
        """Advance the curriculum scheduler by one epoch."""
        self.current_epoch += 1


class SignalCorruptor:
    """
    Universal Domain Randomization Engine.
    Works on ANY tensor (finance, health, robotics) to make Liquid Networks robust.
    """
    
    @staticmethod
    def apply_dropout(tensor: torch.Tensor, drop_prob: float = 0.1, fill_value: float = 0.0) -> torch.Tensor:
        """Randomly drops signals. Example: Sensor failures, missing market data."""
        
        mask = (torch.rand_like(tensor) > drop_prob).float()
        if drop_prob < 1.0:
            scale = 1.0 / (1.0 - drop_prob)
            return (tensor * mask * scale) + fill_value * (1 - mask)
        return fill_value * (1 - mask)

    @staticmethod
    def apply_gaussian_noise(tensor: torch.Tensor, std: float = 0.1, clamp_min=None, clamp_max=None) -> torch.Tensor:
        """Adds natural variance. Example: Camera static, financial volatility."""
        noise = torch.randn_like(tensor) * std
        noisy = tensor + noise
        if clamp_min is not None or clamp_max is not None:
            noisy = torch.clamp(noisy, min=clamp_min, max=clamp_max)
        return noisy
