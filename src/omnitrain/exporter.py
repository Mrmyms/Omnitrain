import torch
import datetime
import os


class OmniExporter:
    """
    Serialization and packaging for OmniTrain models.
    Generates .omni bundle files with session metadata and network states.
    """

    def save(self, model, heads, config, export_path):
        """
        Save a model, its task heads, and metadata into an .omni bundle.
        Includes full architecture specification for deterministic reconstruction.
        """
        # Build architecture manifest from the live model
        architecture = {
            'd_model': model.d_model,
            'n_latents': model.n_latents,
            'n_heads': model.n_heads,
            'num_layers': model.num_layers,
            'input_dim': model.input_dim,
            # v3.0 feature flags
            'has_auto_modality': hasattr(model, 'input_projector'),
            'has_stateful_memory': hasattr(model, 'memory'),
            'heads': {}
        }

        heads_state = {}
        for head_id, head_module in heads.items():
            heads_state[head_id] = head_module.state_dict()
            # Store head architecture for deterministic reconstruction
            if hasattr(head_module, 'net'):
                # Infer head type and dimensions from the network
                first_layer = head_module.net[0]
                last_layer = head_module.net[-1]
                architecture['heads'][head_id] = {
                    'd_model': first_layer.in_features,
                    'output_dim': last_layer.out_features,
                    'type': type(head_module).__name__
                }

        bundle = {
            'model_state': model.state_dict(),
            'heads_state': heads_state,
            'metadata': config,
            'architecture': architecture,
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'version': '3.0-supreme'
        }
        torch.save(bundle, export_path)

    def load_as_inference(self, omni_path):
        """
        Load an .omni bundle and reconstruct the model deterministically
        using the saved architecture manifest.
        Falls back to shape-guessing for legacy v1.0 bundles.
        """
        if not os.path.exists(omni_path):
            raise FileNotFoundError(f"Export bundle {omni_path} not found.")

        # Lazy imports to avoid circular dependencies
        from .fusion_core import FusionCore
        from .heads import ClassificationHead, RegressionHead

        bundle = torch.load(omni_path)
        meta = bundle.get('metadata', {})
        arch = bundle.get('architecture', None)

        # 1. Reconstruct Backbone (metadata-driven or fallback)
        if arch:
            core = FusionCore(
                d_model=arch['d_model'],
                n_latents=arch['n_latents'],
                n_heads=arch.get('n_heads', 8),
                num_layers=arch.get('num_layers', 3),
                input_dim=arch.get('input_dim', 512)
            )
        else:
            # Legacy fallback for v1.0 bundles
            d_model = meta.get('d_model', 512)
            n_latents = meta.get('n_latents', 128)
            core = FusionCore(d_model=d_model, n_latents=n_latents)
            print("[Exporter] WARN: No architecture manifest found. Using legacy reconstruction.")

        core.load_state_dict(bundle['model_state'])
        core.eval()

        # 2. Reconstruct Task Heads (metadata-driven or shape-guessing fallback)
        heads = {}
        head_arch = arch.get('heads', {}) if arch else {}

        for head_id, head_state in bundle.get('heads_state', {}).items():
            try:
                if head_id in head_arch:
                    # Deterministic reconstruction from architecture manifest
                    ha = head_arch[head_id]
                    head_type = ha.get('type', 'ClassificationHead')
                    d_model_head = ha['d_model']
                    output_dim = ha['output_dim']

                    if head_type == 'RegressionHead':
                        h = RegressionHead(output_dim, d_model_head)
                    else:
                        h = ClassificationHead(output_dim, d_model_head)
                else:
                    # Legacy fallback: guess from weight shapes
                    d_model_head = head_state['net.0.weight'].shape[1]
                    num_classes = head_state['net.3.weight'].shape[0]
                    h = ClassificationHead(num_classes, d_model_head)
                    print(f"[Exporter] WARN: Head '{head_id}' reconstructed via shape-guessing (legacy).")

                h.load_state_dict(head_state)
                h.eval()
                heads[head_id] = h
            except Exception as e:
                print(f"[Exporter] ERROR: Could not reconstruct head '{head_id}': {e}")

        return core, heads, meta
