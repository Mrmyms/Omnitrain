import datetime
import os
import yaml
import numpy as np
import torch
import torch.nn as nn


class OmniExporter:
    """
    Serialization and packaging for OmniTrain models.
    Generates .omni bundle files and cross-platform .onnx binaries.
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
            'input_dim': model.input_dim,
            'version': '1.0.0',
            'has_auto_modality': True,
            'has_stateful_memory': True,
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
            'version': '1.0.0'
        }
        torch.save(bundle, export_path)
        print(f"📦 Exported PyTorch bundle to {export_path}")

    def export_to_onnx(self, core, heads, export_path: str):
        """
        Export the full model (core + heads) to a unified ONNX graph.
        Handles recurrent state mapping for the Liquid Brain.
        """
        class UnifiedModel(nn.Module):
            def __init__(self, core, heads):
                super().__init__()
                self.core = core
                self.heads = nn.ModuleDict(heads)
                self.ordered_head_keys = sorted(heads.keys())

            def forward(self, sensor_tokens, dt, prev_state, abs_time):
                # 1. Core Forward
                # In production, sensors are pre-projected into tokens via C++ or a separate ONNX pass.
                # Here we export the core evolution logic.
                h_next = self.core(sensor_tokens, dt, prev_latents=prev_state, abs_time=abs_time)
                
                # 2. Heads Forward
                results = [h_next]
                for h_id in self.ordered_head_keys:
                    results.append(self.heads[h_id](h_next))
                
                return tuple(results)

        import torch.nn.utils.prune as prune
        # 1. Permanently "bake" the sparsity zeros into the raw weight matrix and remove the dynamic hook
        # ONNX tracing cannot handle PyTorch's forward pre-hooks.
        for module in core.modules():
            if hasattr(module, 'weight_orig'):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass

        unified = UnifiedModel(core, heads)
        unified.eval()

        # Input shapes
        batch_size = 1
        n_latents = core.n_latents
        d_model = core.d_model
        
        # Pre-warm projectors to avoid dynamic creation during trace
        with torch.no_grad():
            dummy_tokens = torch.randn(batch_size, 10, d_model)
            dummy_dt = torch.ones(batch_size, 1)
            dummy_state = torch.zeros(batch_size, n_latents, d_model)
            dummy_abs_time = torch.zeros(batch_size, 1)
            unified(dummy_tokens, dummy_dt, dummy_state, dummy_abs_time)

        input_names = ["sensor_tokens", "dt", "prev_state", "abs_time"]
        output_names = ["next_state"] + [f"head_{k}" for k in unified.ordered_head_keys]

        print(f"🚀 Exporting unified ONNX model to {export_path}...")
        # We use strict=False to allow for custom liquid logic that might 
        # use dynamic control flow which ONNX tries to unroll.
        torch.onnx.export(
            unified,
            (dummy_tokens, dummy_dt, dummy_state, dummy_abs_time),
            export_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                'sensor_tokens': {0: 'batch', 1: 'num_tokens'},
                'dt': {0: 'batch'},
                'prev_state': {0: 'batch'},
                'abs_time': {0: 'batch'},
                'next_state': {0: 'batch'}
            },
            opset_version=15
        )
        print(f"✅ ONNX model saved to {export_path}")

    def export_for_qualcomm_snpe(self, core, heads, export_path: str, static_batch: int = 1, static_tokens: int = 10):
        """
        Export the model specifically for Qualcomm SNPE/QNN toolchains.
        Hexagon DSP requires strictly static shapes. This exporter removes dynamic axes.
        """
        # We reuse the same logic but force static shapes
        class UnifiedModel(nn.Module):
            def __init__(self, core, heads):
                super().__init__()
                self.core = core
                self.heads = nn.ModuleDict(heads)
                self.ordered_head_keys = sorted(heads.keys())

            def forward(self, sensor_tokens, dt, prev_state, abs_time):
                h_next = self.core(sensor_tokens, dt, prev_latents=prev_state, abs_time=abs_time)
                results = [h_next]
                for h_id in self.ordered_head_keys:
                    results.append(self.heads[h_id](h_next))
                return tuple(results)

        import torch.nn.utils.prune as prune
        for module in core.modules():
            if hasattr(module, 'weight_orig'):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass

        unified = UnifiedModel(core, heads)
        unified.eval()

        n_latents = core.n_latents
        d_model = core.d_model
        
        with torch.no_grad():
            dummy_tokens = torch.randn(static_batch, static_tokens, d_model)
            dummy_dt = torch.ones(static_batch, 1)
            dummy_state = torch.zeros(static_batch, n_latents, d_model)
            dummy_abs_time = torch.zeros(static_batch, 1)
            unified(dummy_tokens, dummy_dt, dummy_state, dummy_abs_time)

        input_names = ["sensor_tokens", "dt", "prev_state", "abs_time"]
        output_names = ["next_state"] + [f"head_{k}" for k in unified.ordered_head_keys]

        print(f"🚀 Exporting STATIC ONNX model for Qualcomm SNPE to {export_path}...")
        torch.onnx.export(
            unified,
            (dummy_tokens, dummy_dt, dummy_state, dummy_abs_time),
            export_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,  # STRICTLY STATIC SHAPES FOR HEXAGON DSP
            opset_version=15
        )
        print(f"✅ Qualcomm-Optimized ONNX saved to {export_path}")

    def export_for_tensorrt(self, core, heads, export_path: str, max_batch: int = 16, max_tokens: int = 128):
        """
        Export the model specifically for NVIDIA TensorRT (Jetson family).
        TensorRT supports dynamic axes, but requires an Explicit Optimization Profile.
        This function exports the dynamic graph and prints the exact `trtexec` command
        to compile it with FP16 precision.
        """
        class UnifiedModel(nn.Module):
            def __init__(self, core, heads):
                super().__init__()
                self.core = core
                self.heads = nn.ModuleDict(heads)
                self.ordered_head_keys = sorted(heads.keys())

            def forward(self, sensor_tokens, dt, prev_state, abs_time):
                h_next = self.core(sensor_tokens, dt, prev_latents=prev_state, abs_time=abs_time)
                results = [h_next]
                for h_id in self.ordered_head_keys:
                    results.append(self.heads[h_id](h_next))
                return tuple(results)

        import torch.nn.utils.prune as prune
        for module in core.modules():
            if hasattr(module, 'weight_orig'):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass

        unified = UnifiedModel(core, heads)
        unified.eval()

        n_latents = core.n_latents
        d_model = core.d_model
        
        with torch.no_grad():
            dummy_tokens = torch.randn(1, 10, d_model)
            dummy_dt = torch.ones(1, 1)
            dummy_state = torch.zeros(1, n_latents, d_model)
            dummy_abs_time = torch.zeros(1, 1)
            unified(dummy_tokens, dummy_dt, dummy_state, dummy_abs_time)

        input_names = ["sensor_tokens", "dt", "prev_state", "abs_time"]
        output_names = ["next_state"] + [f"head_{k}" for k in unified.ordered_head_keys]

        print(f"🚀 Exporting DYNAMIC ONNX model for NVIDIA TensorRT to {export_path}...")
        torch.onnx.export(
            unified,
            (dummy_tokens, dummy_dt, dummy_state, dummy_abs_time),
            export_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                'sensor_tokens': {0: 'batch', 1: 'num_tokens'},
                'dt': {0: 'batch'},
                'prev_state': {0: 'batch'},
                'abs_time': {0: 'batch'},
                'next_state': {0: 'batch'}
            },
            opset_version=15
        )
        print(f"✅ ONNX exported for TensorRT: {export_path}")
        print(f"⚠️  [JETSON DEPLOYMENT] To compile this engine on your NVIDIA Jetson, run:")
        print(f"trtexec --onnx={export_path} \\")
        print(f"        --saveEngine={export_path.replace('.onnx', '.engine')} \\")
        print(f"        --fp16 \\")
        print(f"        --minShapes=sensor_tokens:1x1x{d_model},dt:1x1,prev_state:1x{n_latents}x{d_model},abs_time:1x1 \\")
        print(f"        --optShapes=sensor_tokens:1x10x{d_model},dt:1x1,prev_state:1x{n_latents}x{d_model},abs_time:1x1 \\")
        print(f"        --maxShapes=sensor_tokens:{max_batch}x{max_tokens}x{d_model},dt:{max_batch}x1,prev_state:{max_batch}x{n_latents}x{d_model},abs_time:{max_batch}x1")

    def load_as_inference(self, omni_path):
        """
        Load an .omni bundle and reconstruct the model deterministically.
        """
        if not os.path.exists(omni_path):
            raise FileNotFoundError(f"Export bundle {omni_path} not found.")

        from .fusion_core import LiquidFusionCore
        from .heads import ClassificationHead, RegressionHead

        bundle = torch.load(omni_path, map_location='cpu')
        meta = bundle.get('metadata', {})
        arch = bundle.get('architecture', {})

        core = LiquidFusionCore(
            d_model=arch.get('d_model', 256),
            n_latents=arch.get('n_latents', 32),
            input_dim=arch.get('input_dim', 512),
            config=meta
        )
        core.load_state_dict(bundle['model_state'])
        core.eval()

        heads = {}
        head_arch = arch.get('heads', {})

        for head_id, head_state in bundle.get('heads_state', {}).items():
            if head_id in head_arch:
                ha = head_arch[head_id]
                head_type = ha.get('type', 'ClassificationHead')
                d_model_head = ha['d_model']
                output_dim = ha['output_dim']

                if head_type == 'RegressionHead':
                    h = RegressionHead(output_dim, d_model_head)
                else:
                    h = ClassificationHead(output_dim, d_model_head)
                
                h.load_state_dict(head_state)
                h.eval()
                heads[head_id] = h

        return core, heads, meta
