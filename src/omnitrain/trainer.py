import torch
import torch.nn as nn
import time
import numpy as np
from typing import Optional
from .token_bus import TokenBus


class OmniTrainer:
    """
    OmniTrain training coordinator v3.0.
    Supports stateful training with latent memory carry-over and
    optional FSDP (Fully Sharded Data Parallel) for multi-GPU scaling.
    """

    def __init__(self, bus, model, heads, optimizer, criterion=nn.CrossEntropyLoss(),
                 use_fsdp=False):
        self.bus = bus
        self.model = model
        self.heads = nn.ModuleDict(heads)
        self.optimizer = optimizer
        self.criterion = criterion
        # Stateful Latents: persistent memory across training steps
        self._prev_latents: Optional[torch.Tensor] = None

        # FSDP: Wrap model for distributed training if requested
        if use_fsdp:
            self._setup_fsdp()

    def _setup_fsdp(self):
        """
        Initialize Fully Sharded Data Parallel for multi-GPU training.
        Shards model parameters, gradients, and optimizer states across GPUs.
        """
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision
            import torch.distributed as dist

            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")

            # Mixed precision policy: FP16 for compute, FP32 for safety-critical params
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )

            self.model = FSDP(
                self.model,
                mixed_precision=mp_policy,
                use_orig_params=True,
            )
            self.heads = FSDP(
                self.heads,
                mixed_precision=mp_policy,
                use_orig_params=True,
            )

            # Rebuild optimizer with FSDP-wrapped parameters
            all_params = list(self.model.parameters()) + list(self.heads.parameters())
            self.optimizer = type(self.optimizer)(all_params, **self.optimizer.defaults)

            rank = dist.get_rank()
            print(f"[OmniTrainer] FSDP Active on GPU {rank} / {dist.get_world_size()}")

        except ImportError:
            print("[OmniTrainer] WARN: torch.distributed.fsdp not available. Training on single GPU.")
        except RuntimeError as e:
            print(f"[OmniTrainer] WARN: FSDP init failed ({e}). Falling back to single GPU.")

    def _bus_tokens_to_tensors(self, token_list, device='cpu'):
        """
        Convert a List[Dict] from the TokenBus into pre-allocated tensors
        suitable for the tensor-first FusionCore.forward() signature.

        Returns:
            sensor_data: (1, N, input_dim) tensor
            timestamps:  (1, N, 1) tensor
        """
        if not token_list:
            return None, None

        # Vectorized conversion (single NumPy stack, single Torch copy)
        times = np.array([t['timestamp'] for t in token_list], dtype=np.float64)
        data = np.stack([t['data'] for t in token_list])  # (N, input_dim)

        # Normalize timestamps to [0, 1]
        t_min, t_max = times.min(), times.max()
        t_span = (t_max - t_min) if t_max > t_min else 1.0
        norm_times = ((times - t_min) / t_span).astype(np.float32)

        sensor_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, dim)
        timestamps = torch.tensor(norm_times, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # (1, N, 1)

        return sensor_data, timestamps

    def reset_memory(self):
        """Reset the stateful latent memory (e.g., at start of a new episode)."""
        self._prev_latents = None

    def train_step_from_bus(self, task_id: str, window_size: float = 0.5,
                            stateful: bool = True) -> float:
        """
        Atomic training step with optional stateful latent memory.
        Extracts sensors and labels from a single bus snapshot.

        Args:
            task_id: Head to use for loss computation.
            window_size: Time window (seconds) for the bus snapshot.
            stateful: If True, carry latent state across steps.
        """
        now = time.time()
        # Single snapshot to prevent sensor drift
        all_tokens = self.bus.get_window(now - window_size, now)

        labels = [t for t in all_tokens if t['modal_id'] == 'label_stream']
        if not labels:
            return 0.0

        # Extract class index safely (classification expects 1D target of indices)
        label_val = labels[-1]['data']
        target = torch.tensor([label_val[0]]).long()

        sensor_tokens = [t for t in all_tokens if t['modal_id'] != 'label_stream']
        if not sensor_tokens:
            return 0.0

        # Convert bus tokens to tensors for tensor-first forward pass
        sensor_data, timestamps = self._bus_tokens_to_tensors(sensor_tokens)
        if sensor_data is None:
            return 0.0

        self.model.train()
        self.heads.train()

        # Forward with stateful memory
        prev = self._prev_latents if stateful else None
        latents = self.model(sensor_data, timestamps, prev_latents=prev)

        # Update memory for next step (detach to prevent BPTT explosion)
        if stateful:
            self._prev_latents = latents.detach()

        prediction = self.heads[task_id](latents)

        # Ensure prediction/target alignment (handle sequence dimension if present)
        if prediction.dim() == 3:  # (B, S, C)
            prediction = prediction.view(-1, prediction.size(-1))
            target = target.expand(prediction.size(0))

        # Sync device
        target = target.to(prediction.device)

        loss = self.criterion(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
