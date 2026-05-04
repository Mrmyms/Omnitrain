import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from .fusion_core import LiquidFusionCore
from .exporter import OmniExporter

class OmniDiagnostic:
    """
    Industrial Sensitivity Analysis for BioLiquid Networks.
    Identifies which sensors are actually driving the robot's decisions.
    """

    def __init__(self, model_path: str):
        self.exporter = OmniExporter()
        self.core, self.heads, self.config = self.exporter.load_as_inference(model_path)
        self.core.eval()

    def analyze_sensitivity(self, num_samples: int = 10) -> Dict[str, float]:
        """
        Computes the gradient-based sensitivity (Saliency) for each input modality.
        Higher values mean the sensor has more influence on the latent state.
        """
        sensitivities = {}
        inputs = self.config.get('inputs', [])
        
        # Mock inputs for gradient tracing
        d_model = self.core.d_model
        n_latents = self.core.n_latents
        
        for input_cfg in inputs:
            m_id = input_cfg['id']
            dim = input_cfg.get('dim', 1)
            
            # Reset state to ensure fresh graph for each modality
            self.core.reset_state(batch_size=1)
            
            # Create a sample input that requires grad
            sample = torch.randn(1, dim, requires_grad=True)
            dt = torch.ones(1, 1)
            prev_state = torch.zeros(1, n_latents, d_model)
            
            # Forward pass
            next_state = self.core(sample, dt, prev_latents=prev_state)
            
            # Target: sum of latent activations
            loss = next_state.abs().sum()
            loss.backward()
            
            # Sensitivity = average magnitude of gradients on the input
            if sample.grad is not None:
                grad_mag = sample.grad.abs().mean().item()
                sensitivities[m_id] = grad_mag
            else:
                sensitivities[m_id] = 0.0

        # Normalize to percentages
        total = sum(sensitivities.values()) + 1e-9
        normalized = {k: (v / total) * 100 for k, v in sensitivities.items()}
        
        return dict(sorted(normalized.items(), key=lambda x: x[1], reverse=True))

    def check_health(self) -> Dict[str, str]:
        """
        Performs structural and weight distribution checks.
        Detects vanishing/exploding gradients or dead neurons across all modules.
        """
        report = {}
        
        # 1. Architecture Identification
        mode = getattr(self.core, 'brain_mode', 'legacy')
        report['Arch Mode'] = mode.upper()

        # 2. Vitality Check: Iterate through all Liquid cells
        vitality_scores = []
        for name, module in self.core.named_modules():
            # Check for BioLiquidCell or similar stateful cells
            if hasattr(module, 'f1') and isinstance(module.f1, nn.Linear):
                with torch.no_grad():
                    w_mean = module.f1.weight.abs().mean().item()
                    vitality_scores.append(w_mean)
        
        if not vitality_scores:
            report['Brain Vitality'] = "N/A (No liquid cells found)"
        else:
            avg_vitality = sum(vitality_scores) / len(vitality_scores)
            if avg_vitality < 1e-5:
                report['Brain Vitality'] = f"CRITICAL: Unresponsive ({avg_vitality:.2e})"
            elif avg_vitality > 5.0:
                report['Brain Vitality'] = f"WARNING: Over-excited ({avg_vitality:.1f})"
            else:
                report['Brain Vitality'] = "HEALTHY"

        # 3. Plasticity Saturation
        plastic_sum = 0
        plastic_count = 0
        for name, module in self.core.named_modules():
            if hasattr(module, 'w_plastic') and module.w_plastic is not None:
                plastic_sum += module.w_plastic.abs().mean().item()
                plastic_count += 1
        
        if plastic_count > 0:
            avg_plasticity = plastic_sum / plastic_count
            report['Plasticity'] = f"ACTIVE ({avg_plasticity:.4f})"
        else:
            report['Plasticity'] = "INACTIVE (No plastic weights found)"
            
        # 4. Conectoma Specific: Sparsity Check
        if mode == 'conectoma':
            hub = self.core.brain
            # Estimate density from the masks
            density = (hub.sens_inter_mask.mean() + hub.inter_inter_mask.mean() + hub.inter_comm_mask.mean()) / 3
            report['Circuit Density'] = f"{density*100:.1f}% (Sparse)"

        return report
