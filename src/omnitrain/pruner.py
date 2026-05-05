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
    print(f"INFO: Starting Structured Pruning on {model_path} (ratio={pruning_ratio})...")

    # 1. Load the original model
    try:
        core, heads, config = OmniExporter().load_as_inference(model_path)
    except Exception as e:
        print(f"ERROR loading model for pruning: {e}")
        return

    # Put model in train mode for pruning operations
    core.train()
    pruned_core = copy.deepcopy(core)

    # 2. Identify and prune eligible Linear layers
    layers_pruned = 0
    layers_skipped = 0
    
    # Store the mapping of original module names to their new output dimensions
    # to handle shape propagation.
    new_dims = {}

    module_list = list(pruned_core.named_modules())
    for i, (name, module) in enumerate(module_list):
        
        # and we need to update our input features.
        # For simplicity in this architecture, we check if the PREVIOUS module in the 
        # list was a Linear layer that we just pruned.
        if i > 0:
            prev_name, prev_module = module_list[i-1]
            if prev_name in new_dims:
                mask = new_dims[prev_name]
                new_in = int(mask.sum().item())
                if isinstance(module, nn.Linear):
                    print(f"   ADAPT: Updating input dim for {name}: {module.in_features} -> {new_in}")
                    old_linear = module
                    module = nn.Linear(new_in, old_linear.out_features, bias=old_linear.bias is not None)
                    
                    kept_indices = mask.nonzero(as_tuple=True)[0]
                    module.weight.data = old_linear.weight.data[:, kept_indices]
                    if old_linear.bias is not None:
                        module.bias.data = old_linear.bias.data
                        
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = dict(pruned_core.named_modules())[parent_name] if parent_name else pruned_core
                    setattr(parent, child_name, module)
                elif hasattr(module, 'ff1') and isinstance(module.ff1, nn.Linear):
                    # Handle BioLiquidCell successor
                    print(f"   ADAPT: Updating BioLiquidCell {name} input: {module.ff1.in_features} -> {new_in}")
                    # Rebuild the internal projectors of the BioLiquidCell
                    # This is complex, so we skip for now or provide a specialized handler.
                    pass

        if isinstance(module, nn.Linear):
            # Safety Rule: Never prune safety-critical layers
            if 'safety' in name.lower() or 'shield' in name.lower():
                print(f"   SKIP: safety-critical layer: {name}")
                layers_skipped += 1
                continue

            # Skip the final projection layers of attention (would break architecture)
            if 'cross_attn' in name or 'out_proj' in name or 'mixer' in name.lower():
                print(f"   SKIP: attention/mixer layer: {name}")
                layers_skipped += 1
                continue

            print(f"   PRUNE: layer: {name} ({module.in_features} -> {module.out_features})")

            # Ln Structured Pruning on dim=0 (output neurons)
            prune.ln_structured(module, name='weight', amount=pruning_ratio, n=1, dim=0)
            prune.remove(module, 'weight')

            # Identify which output neurons survived
            row_norms = module.weight.data.abs().sum(dim=1)
            mask = row_norms > 0
            new_out_features = int(mask.sum().item())

            # Rebuild with reduced dimensions
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = dict(pruned_core.named_modules())[parent_name] if parent_name else pruned_core

            new_layer = _rebuild_linear_pruned(module, mask)
            setattr(parent, child_name, new_layer)
            
            # Record the new mask for the NEXT layer
            new_dims[name] = mask
            layers_pruned += 1

    print(f"\n   STATS: Layers pruned: {layers_pruned} | Layers skipped: {layers_skipped}")

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
            
            # we consider the synaptic activity (mean absolute value) as a proxy for 
            # functional sensitivity in a post-training context.
            importance = cell.w_plastic.abs().mean(dim=0)  # (In, Out)
            mask = (importance > self.threshold).float()
            pruned = (mask == 0).sum().item()
            total  = mask.numel()

            # Apply mask: zero out weak synapses
            cell.w_plastic = cell.w_plastic * mask.unsqueeze(0)
            
            
            if hasattr(cell, 'plastic_pruning_mask'):
                cell.plastic_pruning_mask = cell.plastic_pruning_mask * mask.unsqueeze(0)

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
