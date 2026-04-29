import torch
import torch.nn as nn
import os
import yaml
from typing import Optional, Dict, List, Tuple
from torch.utils.data import DataLoader

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.table import Table
from rich.panel import Panel
from rich import box

from .fusion_core import LiquidFusionCore
from .heads import ClassificationHead, RegressionHead
from .omni_shield import OmniShieldGuard
from .curriculum import CurriculumScheduler
from .dataset import OmniLogDataset
from .exporter import OmniExporter

console = Console()

class UniversalTrainer:
    """
    OmniTrain Universal Trainer v2.0 (Dynamic).
    Orchestrates the full 3-phase Curriculum Pipeline using config-driven logic.
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

        all_params = list(core.parameters()) + list(self.heads.parameters())
        all_params += list(shield.state_extractor.parameters())
        all_params += list(shield.barrier.parameters())
        all_params += list(shield.dynamics.parameters())

        self.optimizer = torch.optim.AdamW(all_params, lr=learning_rate)
        self.criterion_mse = nn.MSELoss()
        self.criterion_ce = nn.CrossEntropyLoss()
        self.history = {'loss': [], 'policy': [], 'safety': [], 'barrier': []}

    @classmethod
    def from_config(cls, config_path: str, lr: float = 2e-3) -> 'UniversalTrainer':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        m = config['model']
        # Robust Input Dimension Detection
        input_dim = m.get('input_dim', 512)
        for inp in config.get('inputs', []):
            if 'dim' in inp:
                input_dim = inp['dim']
                break

        # NEW: Pass the full config to LiquidFusionCore to enable Multi-Brain Hub detection
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

        action_head_key = next((k for k in heads if 'drive' in k or 'control' in k), list(heads.keys())[0])
        shield = OmniShieldGuard.from_config(config, heads[action_head_key], d_model=m['d_model'])

        return cls(core, heads, shield, config, lr)

    def _train_epoch(self, loader: DataLoader, barrier_weight: float = 1.0) -> Dict[str, float]:
        self.core.train()
        self.shield.train()
        self.heads.train()

        metrics = {'policy': 0, 'safety': 0, 'barrier': 0, 'count': 0}

        for batch in loader:
            self.optimizer.zero_grad()
            
            B, T = batch['dt'].shape
            prev_step_latents = None
            total_sequence_loss = 0

            for t in range(T):
                dt_t = batch['dt'][:, t]
                current_step_latents = prev_step_latents
                
                for m_id, sensor_seq in batch['inputs'].items():
                    sensor_t = sensor_seq[:, t]
                    current_step_latents = self.core(
                        sensor_t, 
                        dt_t, 
                        modal_id=m_id, 
                        prev_latents=current_step_latents
                    )
                
                step_loss = 0
                for h_id, head in self.heads.items():
                    target_key = h_id if h_id in batch['targets'] else 'action'
                    if target_key in batch['targets']:
                        pred = head(current_step_latents)
                        target = batch['targets'][target_key][:, t]
                        
                        if isinstance(head, RegressionHead):
                            l = self.criterion_mse(pred, target)
                            metrics['policy'] += l.item()
                        else:
                            l = self.criterion_ce(pred, target)
                            metrics['safety'] += l.item()
                        step_loss += l

                state = self.shield.state_extractor(current_step_latents)
                h_x = self.shield.barrier(state)
                b_loss = self.shield.barrier_loss(h_x) * barrier_weight
                metrics['barrier'] += b_loss.item()
                step_loss += b_loss

                total_sequence_loss += step_loss
                prev_step_latents = current_step_latents

            (total_sequence_loss / T).backward()
            torch.nn.utils.clip_grad_norm_(self.core.parameters(), 1.0)
            self.optimizer.step()
            metrics['count'] += 1

        denom = max(1, metrics['count'])
        return {k: v / denom for k, v in metrics.items() if k != 'count'}

    def fit(self, csv_path: str, epochs: int = 50, batch_size: int = 16, seq_len: int = 32):
        # 3-Tier "Chaos" Curriculum using the theoretical Scheduler
        scheduler = CurriculumScheduler(total_epochs=epochs)
        scheduler.add_phase("Phase 1: Imitation", start_epoch=0, chaos_level=0, description="Behavior Cloning")
        scheduler.add_phase("Phase 2: Safety", start_epoch=int(epochs * 0.5), chaos_level=1, description="CBF Constraint Learning")
        scheduler.add_phase("Phase 3: Chaos", start_epoch=int(epochs * 0.8), chaos_level=2, description="Domain Randomization (OOD)")

        # Datasets per chaos level
        datasets = {
            0: OmniLogDataset(csv_path, self.config, seq_len, chaos_level=0),
            1: OmniLogDataset(csv_path, self.config, seq_len, chaos_level=1),
            2: OmniLogDataset(csv_path, self.config, seq_len, chaos_level=2),
        }

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), console=console) as progress:
            task = progress.add_task("Training...", total=epochs)
            for epoch in range(epochs):
                phase_info = scheduler.get_current_phase()
                ds = datasets[phase_info.chaos_level]
                
                # Dynamic barrier weight based on phase
                if phase_info.chaos_level == 0:
                    bw = 0.1
                elif phase_info.chaos_level == 1:
                    bw = 1.0
                else:
                    bw = 2.0

                loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
                m = self._train_epoch(loader, barrier_weight=bw)
                
                progress.update(task, advance=1, description=f"[color(117)]{phase_info.name}[/] Loss: {m['policy']:.4f}")
                if (epoch + 1) % 10 == 0:
                    console.print(f"Epoch {epoch+1} | Policy: {m['policy']:.4f} | Barrier: {m['barrier']:.4f}")
                
                scheduler.step()

        # Final Export
        export_path = f"{self.config.get('project', 'robot')}_final.omni"
        OmniExporter().save(self.core, self.heads, self.config, export_path)
        console.print(f"[bold green]Exported to {export_path}[/bold green]")
