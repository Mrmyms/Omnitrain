import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any

class ModalityLatentBuffer:
    """
    Zero-Order Hold (ZOH) buffer for latent space embeddings.
    Retains the last known latent embedding for a modality if no new data arrives.
    """
    def __init__(self, device: Optional[torch.device] = None):
        self.buffer: Dict[str, torch.Tensor] = {}
        self.timestamps: Dict[str, torch.Tensor] = {}
        self.device = device if device is not None else torch.device("cpu")

    def update(self, modal_id: str, latent: torch.Tensor, timestamp: torch.Tensor):
        # Dynamically adapt device to match incoming tensors
        self.device = latent.device
        self.buffer[modal_id] = latent
        self.timestamps[modal_id] = timestamp

    def get_latest(self, modal_id: str, default_shape: Tuple[int, ...]) -> torch.Tensor:
        if modal_id in self.buffer:
            return self.buffer[modal_id]
        # Return zeros if we've never seen this modality
        return torch.zeros(default_shape, device=self.device)
        
    def reset(self):
        self.buffer.clear()
        self.timestamps.clear()

class AsyncSensorAligner(nn.Module):
    """
    Coordinates temporal alignment of multiple asynchronous sensors.
    Calculates individual delta_t for each modality.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        current_latents: Dict[str, torch.Tensor], 
        current_time: torch.Tensor,
        latent_buffer: ModalityLatentBuffer,
        last_times: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            current_latents: newly arrived latents at this step
            current_time: (B, 1) current absolute time
            latent_buffer: the ZOH buffer
            last_times: dict keeping track of when each modality was last updated
        Returns:
            aligned_latents: Dict of latents to use for this step (new or ZOH)
            delta_ts: Dict of time deltas for each modality to evolve (note: global dt is preferred in Hub)
        """
        aligned_latents = {}
        delta_ts = {}

        B = current_time.shape[0]

        all_modalities = set(current_latents.keys()).union(latent_buffer.buffer.keys())

        for m_id in all_modalities:
            if m_id in current_latents:
                latent = current_latents[m_id]
                latent_buffer.update(m_id, latent, current_time)
                aligned_latents[m_id] = latent
            else:
                default_shape = (B, 256) 
                if latent_buffer.buffer:
                    default_shape = next(iter(latent_buffer.buffer.values())).shape
                aligned_latents[m_id] = latent_buffer.get_latest(m_id, default_shape)

            last_t = last_times.get(m_id, torch.zeros_like(current_time))
            dt = current_time - last_t
            dt = torch.clamp(dt, min=0.001, max=1.0)
            delta_ts[m_id] = dt
            
            last_times[m_id] = current_time.clone()

        return aligned_latents, delta_ts

class PerModalityODESolver(nn.Module):
    """
    Independent ODE solver for a specific modality.
    Prevents 'neural double-evolution' by evolving only when necessary.
    NOTE: This class is currently unused in the production code path but preserved
    for future experimental features.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model)
        )
        self.tau = nn.Parameter(torch.ones(1, d_model))

    def forward(self, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if dt.dim() == 1:
            dt = dt.unsqueeze(1)
        
        dh = -h / torch.exp(self.tau) + self.net(h)
        return h + dh * dt
