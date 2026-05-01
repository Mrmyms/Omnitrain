import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import copy
from omnitrain.exporter import OmniExporter
from omnitrain.fusion_core import FusionCore


def _rebuild_linear_pruned(linear: nn.Linear, pruning_mask_rows: torch.Tensor) -> nn.Linear:
    """
    Rebuild a Linear layer with structurally pruned (removed) output neurons.
    Only keeps rows where the pruning mask is non-zero.
    Returns a new, smaller nn.Linear.
    """
    kept_indices = pruning_mask_rows.nonzero(as_tuple=True)[0]
    if len(kept_indices) == linear.out_features:
        return linear  # Nothing was pruned

    new_linear = nn.Linear(linear.in_features, len(kept_indices), bias=linear.bias is not None)
    new_linear.weight.data = linear.weight.data[kept_indices]
    if linear.bias is not None:
        new_linear.bias.data = linear.bias.data[kept_indices]
    return new_linear


def apply_omni_pruning(model_path, pruning_ratio=0.3, output_path="omni_2_0_pruned.omni"):
    """
    Apply true structured pruning to the OmniTrain model.
    Removes entire output neurons from Linear layers to produce a genuinely
    smaller and faster model. Skips safety-critical layers per project rules.
    """
    print(f"✂️  Starting Structured Pruning on {model_path} (ratio={pruning_ratio})...")

    # 1. Load the original model
    try:
        core, heads, config = OmniExporter().load_as_inference(model_path)
    except Exception as e:
        print(f"❌ Error loading model for pruning: {e}")
        return

    # Put model in train mode for pruning operations
    core.train()
    pruned_core = copy.deepcopy(core)

    # 2. Identify and prune eligible Linear layers
    layers_pruned = 0
    layers_skipped = 0

    for name, module in list(pruned_core.named_modules()):
        if isinstance(module, nn.Linear):
            # Safety Rule: Never prune safety-critical layers
            if 'safety' in name.lower():
                print(f"   ⛔ Skipping safety-critical layer: {name}")
                layers_skipped += 1
                continue

            # Skip the final projection layers of attention (would break architecture)
            if 'cross_attn' in name or 'out_proj' in name:
                print(f"   ⏭  Skipping attention layer: {name}")
                layers_skipped += 1
                continue

            print(f"   ✂️  Pruning layer: {name} ({module.in_features} → {module.out_features})")

            # Ln Structured Pruning on dim=0 (output neurons)
            # This zeros entire rows based on L1 norm magnitude
            prune.ln_structured(module, name='weight', amount=pruning_ratio, n=1, dim=0)
            prune.remove(module, 'weight')

            # Identify which output neurons survived (row L1 norm > 0)
            row_norms = module.weight.data.abs().sum(dim=1)
            mask = row_norms > 0

            # Rebuild with reduced dimensions
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = dict(pruned_core.named_modules())[parent_name]
            else:
                parent = pruned_core

            new_layer = _rebuild_linear_pruned(module, mask)
            setattr(parent, child_name, new_layer)
            layers_pruned += 1

    print(f"\n   📊 Layers pruned: {layers_pruned} | Layers skipped: {layers_skipped}")

    # 3. Count parameters before/after
    params_before = sum(p.numel() for p in core.parameters())
    params_after = sum(p.numel() for p in pruned_core.parameters())
    reduction = (1 - params_after / params_before) * 100

    print(f"   📊 Parameters: {params_before:,} → {params_after:,} ({reduction:.1f}% reduction)")

    # 4. Save the pruned model
    pruned_core.eval()
    config['pruning_ratio'] = pruning_ratio
    config['optimized'] = True
    config['pruning_type'] = 'structured_channel_removal'

    OmniExporter().save(pruned_core, heads, config, output_path)

    size_old = os.path.getsize(model_path) / 1e6
    size_new = os.path.getsize(output_path) / 1e6
    print(f"   📦 File size: {size_old:.2f} MB → {size_new:.2f} MB")
    print(f"✅ Structured pruning complete. Output: {output_path}")


if __name__ == "__main__":
    if os.path.exists("logic_bot_v2.omni"):
        apply_omni_pruning("logic_bot_v2.omni")
    else:
        print("ℹ  logic_bot_v2.omni not found for pruning test.")


# ─────────────────────────────────────────────────────────────────────
#  NEW: SynapticPruner — Hebbian Unlearning / Plasticity Consolidation
# ─────────────────────────────────────────────────────────────────────

class SynapticPruner:
    """
    Post-training synaptic consolidation using Hebbian Unlearning.

    After training, identifies weak plastic synapses (low activation
    history in w_plastic) and eliminates them — mimicking biological
    synaptic pruning that occurs during sleep/consolidation phases.

    This compresses the adaptive memory of the model and improves
    generalization by removing noisy, underutilized connections.

    Usage:
        pruner = SynapticPruner(threshold=0.01, verbose=True)
        stats = pruner.prune(model)  # model is a BioLiquidCell or LiquidFusionCore
        print(stats)
    """

    def __init__(self, threshold: float = 0.01, verbose: bool = True):
        """
        Args:
            threshold: Synapses with |w_plastic| mean below this are pruned.
            verbose:   Print pruning report per cell.
        """
        self.threshold = threshold
        self.verbose = verbose

    def prune_cell(self, cell) -> dict:
        """
        Prune plastic synapses in a single BioLiquidCell.

        Returns a dict with pruning stats:
            total_synapses, pruned_synapses, sparsity
        """
        from omnitrain.fusion_core import BioLiquidCell
        if not isinstance(cell, BioLiquidCell):
            return {}

        if cell.w_plastic is None:
            if self.verbose:
                print("  [SynapticPruner] No plastic weights to prune (never activated).")
            return {"total_synapses": 0, "pruned_synapses": 0, "sparsity": 0.0}

        with torch.no_grad():
            # Compute importance: mean absolute value across batch dim
            importance = cell.w_plastic.abs().mean(dim=0)  # (In, Out)
            mask = (importance > self.threshold).float()
            pruned = (mask == 0).sum().item()
            total  = mask.numel()

            # Apply mask: zero out weak synapses
            cell.w_plastic = cell.w_plastic * mask.unsqueeze(0)

        sparsity = pruned / total if total > 0 else 0.0
        stats = {"total_synapses": total, "pruned_synapses": int(pruned),
                 "sparsity": round(sparsity, 4)}

        if self.verbose:
            print(f"  [SynapticPruner] {int(pruned)}/{total} synapses pruned "
                  f"({sparsity*100:.1f}% sparsity) | threshold={self.threshold}")
        return stats

    def prune(self, model: nn.Module) -> dict:
        """
        Recursively prune all BioLiquidCells found in the model.

        Args:
            model: Any nn.Module (LiquidFusionCore, NCPBackbone, etc.)
        Returns:
            Aggregated stats across all cells.
        """
        from omnitrain.fusion_core import BioLiquidCell

        total_synapses = 0
        total_pruned = 0
        cells_found = 0

        for name, module in model.named_modules():
            if isinstance(module, BioLiquidCell):
                cells_found += 1
                if self.verbose:
                    print(f"[SynapticPruner] Processing cell: {name}")
                stats = self.prune_cell(module)
                total_synapses += stats.get("total_synapses", 0)
                total_pruned   += stats.get("pruned_synapses", 0)

        overall_sparsity = total_pruned / total_synapses if total_synapses > 0 else 0.0

        summary = {
            "cells_pruned": cells_found,
            "total_synapses": total_synapses,
            "total_pruned": total_pruned,
            "overall_sparsity": round(overall_sparsity, 4),
        }

        if self.verbose:
            print(f"\n[SynapticPruner] DONE — {cells_found} cells, "
                  f"{total_pruned}/{total_synapses} synapses eliminated "
                  f"({overall_sparsity*100:.1f}% overall sparsity)")
        return summary
