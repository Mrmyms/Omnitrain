import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Generator
import yaml
import os
import time

from .fusion_core import LiquidFusionCore
from .heads import ClassificationHead, RegressionHead
from .dataset import OmniLogDataset
from .curriculum import CurriculumScheduler
from .omni_shield import OmniShieldGuard
from .exporter import OmniExporter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()


# ─────────────────────────────────────────────────────────────────────
#  LagrangianSafetyController: Adaptive Safety Weight (Fix #4)
# ─────────────────────────────────────────────────────────────────────

class LagrangianSafetyController:
    """
    Augmented Lagrangian Multiplier for safety-constrained training.

    Problem: A static `barrier_weight` can be dominated by the task loss,
    causing the policy to sacrifice safety for performance.

    Solution: Track constraint violations and adaptively increase λ until
    the constraint h(x) ≥ 0 is satisfied. When satisfied, λ decreases.
    This is the standard primal-dual method for constrained RL.

        λ_{t+1} = max(λ_min, λ_t + lr * max(0, -mean(h(x))))

    Args:
        init_lambda:  Initial safety weight (should be low; controller
                      will increase it if needed).
        lr:           Lagrangian learning rate (dual step size).
        lambda_min:   Minimum allowed weight (prevents collapse to 0).
        lambda_max:   Maximum allowed weight (prevents instability).
    """

    def __init__(
        self,
        init_lambda: float = 0.1,
        lr: float = 0.02,
        lambda_min: float = 0.01,
        lambda_max: float = 10.0,
    ):
        
        self.lam = torch.tensor(init_lambda, dtype=torch.float32)
        self.lr = lr
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def update(self, h_x_mean: torch.Tensor) -> torch.Tensor:
        """
        Update λ based on current mean barrier value without CPU sync.
        """
        if self.lam.device != h_x_mean.device:
            self.lam = self.lam.to(h_x_mean.device)
            
        violation = torch.clamp(-h_x_mean, min=0.0)
        self.lam = torch.clamp(self.lam + self.lr * violation, min=self.lambda_min, max=self.lambda_max)
        return self.lam

    @property
    def value(self) -> torch.Tensor:
        return self.lam


class UniversalTrainer:
    """
    Industrial Universal Trainer for BioLiquid Networks.
    Implements 3-Phase Curriculum with Gradient Synchronization.
    """

    def __init__(
        self,
        core: LiquidFusionCore,
        heads: Dict[str, nn.Module],
        shield: OmniShieldGuard,
        config: dict,
        learning_rate: float = 2e-3,
    ):
        self.core = core
        self.heads = nn.ModuleDict(heads)
        self.shield = shield
        self.config = config
        self.lr = learning_rate

        
        # (e.g., action head is shared between heads and shield)
        all_params = list(core.parameters()) + list(self.heads.parameters())
        all_params += list(shield.state_extractor.parameters())
        all_params += list(shield.barrier.parameters())
        all_params += list(shield.dynamics.parameters())
        
        unique_params = []
        param_ids = set()
        for p in all_params:
            if id(p) not in param_ids:
                unique_params.append(p)
                param_ids.add(id(p))

        
        wd = config.get('training', {}).get('weight_decay', 1e-4)
        self.optimizer = torch.optim.AdamW(unique_params, lr=learning_rate, weight_decay=wd)
        self.criterion_mse = nn.MSELoss()
        self.criterion_ce = nn.CrossEntropyLoss()
        self.history = {'loss': [], 'policy': [], 'safety': [], 'barrier': [], 'lambda': []}

        
        lagr_cfg = config.get('training', {}).get('lagrangian', {})
        self.lagrangian = LagrangianSafetyController(
            init_lambda=lagr_cfg.get('init_lambda', 0.1),
            lr=lagr_cfg.get('lr', 0.02),
            lambda_min=lagr_cfg.get('lambda_min', 0.01),
            lambda_max=lagr_cfg.get('lambda_max', 10.0),
        )

    @classmethod
    def from_config(cls, config_path: str, lr: float = 2e-3) -> 'UniversalTrainer':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        m = config['model']
        input_dim = m.get('input_dim', 512)
        for inp in config.get('inputs', []):
            if 'dim' in inp:
                input_dim = inp['dim']
                break

        core = LiquidFusionCore(
            n_latents=m.get('n_latents', 32),
            d_model=m.get('d_model', 256),
            input_dim=input_dim,
            config=config 
        )

        heads = {}
        for h_cfg in config.get('heads', []):
            h_id = h_cfg['id']
            if h_cfg['type'] == 'regression':
                heads[h_id] = RegressionHead(h_cfg['output_dim'], m['d_model'])
            else:
                heads[h_id] = ClassificationHead(h_cfg['num_classes'], m['d_model'])

        
        action_head_key = config.get('training', {}).get('action_head')
        if not action_head_key:
            action_head_key = next((k for k in heads if 'drive' in k or 'control' in k), list(heads.keys())[0])
        
        shield = OmniShieldGuard.from_config(config, heads[action_head_key], d_model=m['d_model'])

        return cls(core, heads, shield, config, lr)

    def _train_epoch(self, loader: DataLoader, barrier_weight: float = 1.0) -> Dict[str, float]:
        self.core.train()
        self.heads.train()
        self.shield.train()
        metrics = {'policy': 0, 'safety': 0, 'barrier': 0, 'lambda': 0, 'count': 0}

        
        # We initialize as None and let the core handle the first reset.
        self.core.reset_state()

        for batch in loader:
            self.optimizer.zero_grad()

            # Explicit Device Mapping
            device = next(self.core.parameters()).device

            B, T = batch['dt'].shape
            
            
            # Only reset if the batch indicates the start of a trajectory (is_start=True)
            # or if the batch size changed (e.g. last batch in loader).
            force_reset = batch.get('is_start', torch.tensor([False])).any().item()
            
            if force_reset or self.core._abs_time_buf is None or self.core._abs_time_buf.shape[0] != B:
                self.core.reset_state(batch_size=B, device=device)
            else:
                # Truncated BPTT: preserve context but detach from previous graph 
                # to prevent backpropagating through the entire dataset history.
                if self.core._abs_time_buf is not None:
                    self.core._abs_time_buf = self.core._abs_time_buf.detach()
                
                # Internal brain states are detached inside FusionCore during forward 
                # if not handled here. To be safe, we ensure state is detached.
                if hasattr(self.core, '_last_brain_state') and self.core._last_brain_state is not None:
                    # Deep detach of the state dictionary
                    new_state = {}
                    for k, v in self.core._last_brain_state.items():
                        if isinstance(v, tuple):
                            new_state[k] = tuple(item.detach() for item in v)
                        else:
                            new_state[k] = v.detach()
                    self.core._last_brain_state = new_state

            current_abs_time = self.core._abs_time_buf if self.core._abs_time_buf is not None else torch.zeros(B, 1, device=device)
            prev_step_latents = None
            total_sequence_loss = 0


            sequence_h_x = []
            for t in range(T):
                dt_t = batch['dt'][:, t].unsqueeze(1).to(device)
                current_abs_time = current_abs_time + dt_t

                # Group all modalities for the current time step t and move to device
                step_sensors = {m_id: seq[:, t].to(device) for m_id, seq in batch['inputs'].items()}

                # Single core forward pass for all modalities (prevents double-evolution bug)
                current_step_latents = self.core(
                    step_sensors,
                    dt_t.squeeze(1),
                    prev_latents=prev_step_latents,
                    abs_time=current_abs_time
                )

                step_loss = 0
                for h_id, head in self.heads.items():
                    target_key = h_id if h_id in batch['targets'] else 'action'
                    if target_key in batch['targets']:
                        pred = head(current_step_latents)
                        target = batch['targets'][target_key][:, t].to(device)

                        if isinstance(head, RegressionHead):
                            l = self.criterion_mse(pred, target)
                            metrics['policy'] += l.detach()
                        else:
                            l = self.criterion_ce(pred, target)
                            metrics['safety'] += l.detach()
                        step_loss += l

                
                shield_out = self.shield(current_step_latents, sensor_batch=batch['hw_sensors'][:, t].to(device))
                h_x = shield_out['h_x']
                b_loss = shield_out['barrier_loss']
                
                sequence_h_x.append(h_x)

                # Use current lagrangian value for loss, but don't update yet
                
                lagr_loss = b_loss * self.lagrangian.value * barrier_weight
                metrics['barrier'] += lagr_loss.detach()
                metrics['lambda'] += self.lagrangian.value.detach()
                step_loss += lagr_loss

                total_sequence_loss += step_loss
                prev_step_latents = current_step_latents

            
            if T > 1:
                total_sequence_loss = total_sequence_loss / T

            
            if sequence_h_x:
                avg_h_x = torch.stack(sequence_h_x).mean()
                self.lagrangian.update(avg_h_x)

            total_sequence_loss.backward()
            
            for group in self.optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], 1.0)
            
            self.optimizer.step()
            self.shield.barrier._ensure_icnn_constraints()  # Now a no-op, kept for compat
            metrics['count'] += 1

        
        denom = max(1, metrics['count'] * T)
        return {k: (v.item() if isinstance(v, torch.Tensor) else v) / denom for k, v in metrics.items() if k != 'count'}

    def fit(self, csv_path: str, epochs: Optional[int] = None, batch_size: Optional[int] = None) -> Generator[Dict, None, None]:
        """
        Execute the curriculum and yield metrics per epoch for the CLI.
        """
        train_cfg = self.config.get('training', {})
        epochs = epochs or train_cfg.get('epochs', 30)
        batch_size = batch_size or train_cfg.get('batch_size', 16)
        seq_len = train_cfg.get('seq_len', 32)
        
        scheduler = CurriculumScheduler(total_epochs=epochs)
        scheduler.add_phase("Phase 1: Imitation", start_epoch=0, chaos_level=0, description="Behavior Cloning")
        scheduler.add_phase("Phase 2: Safety", start_epoch=int(epochs * 0.5), chaos_level=1, description="CBF Constraint Learning")
        scheduler.add_phase("Phase 3: Chaos", start_epoch=int(epochs * 0.8), chaos_level=2, description="Domain Randomization (OOD)")

        datasets = {
            0: OmniLogDataset(csv_path, self.config, seq_len, chaos_level=0),
            1: OmniLogDataset(csv_path, self.config, seq_len, chaos_level=1),
            2: OmniLogDataset(csv_path, self.config, seq_len, chaos_level=2),
        }

        for epoch in range(epochs):
            phase_info = scheduler.get_current_phase()
            ds = datasets[phase_info.chaos_level]
            
            bw = 0.1 if phase_info.chaos_level == 0 else (1.0 if phase_info.chaos_level == 1 else 2.0)
            
            
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
            m = self._train_epoch(loader, barrier_weight=bw)
            
            
            for k, v in m.items():
                if k in self.history: self.history[k].append(v)
            
            yield {
                'epoch': epoch + 1,
                'total_epochs': epochs,
                'phase': phase_info.name,
                'chaos': phase_info.chaos_level,
                **m
            }
            
            scheduler.step()

        # Final Export
        export_path = f"{self.config.get('project', 'robot')}_final.omni"
        OmniExporter().save(self.core, self.heads, self.config, export_path)
