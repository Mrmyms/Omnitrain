import datetime
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import logging


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
            ha = getattr(head_module, 'metadata', {})
            if not ha:
                # Deduce output_dim dynamically if missing
                out_dim = getattr(head_module, 'output_dim', None)
                if out_dim is None and hasattr(head_module, 'net'):
                    try:
                        out_dim = head_module.net[-1].out_features
                    except AttributeError:
                        out_dim = 1
                        
                ha = {
                    'd_model': architecture['d_model'],
                    'output_dim': out_dim or 1,
                    'type': type(head_module).__name__
                }
            architecture['heads'][head_id] = ha

        bundle = {
            'model_state': model.state_dict(),
            'heads_state': heads_state,
            'metadata': config,
            'architecture': architecture,
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'version': '2.1.0'
        }
        torch.save(bundle, export_path)
        logging.info(f"📦 Exported PyTorch bundle to {export_path}")

    def export_to_onnx(self, core, heads, export_path: str):
        """
        Export the full model (core + heads) to a unified ONNX graph.
        Handles recurrent state mapping for the Liquid Brain.
        """
        dynamic_axes = {
            'sensor_tokens': {0: 'batch', 1: 'num_tokens'},
            'dt': {0: 'batch'},
            'prev_state': {0: 'batch'},
            'abs_time': {0: 'batch'},
            'next_state': {0: 'batch'}
        }
        for k in heads.keys():
            dynamic_axes[f"head_{k}"] = {0: 'batch'}

        self._trace_and_export(
            core, heads, export_path,
            dynamic_axes=dynamic_axes,
            opset_version=15  # Upgraded opset for better quantization support
        )

    def _trace_and_export(self, core, heads, export_path, dynamic_axes=None, opset_version=15, 
                          static_batch=1, static_tokens=10, use_kv_cache=False):
        """Shared internal engine for ONNX tracing and serialization."""
        
        # KV-Cache Reuse Support
        class UnifiedModel(nn.Module):
            def __init__(self, core, heads, use_kv=False):
                super().__init__()
                self.core = core
                self.heads = nn.ModuleDict(heads)
                self.ordered_head_keys = sorted(heads.keys())
                self.use_kv = use_kv

            def forward(self, sensor_tokens, dt, prev_state, abs_time):
                # Use is_tokenized=True
                h_next = self.core(sensor_tokens, dt, prev_latents=prev_state, abs_time=abs_time, is_tokenized=True)
                
                results = [h_next]
                for h_id in self.ordered_head_keys:
                    results.append(self.heads[h_id](h_next))
                return tuple(results)

        import torch.nn.utils.prune as prune
        for module in core.modules():
            if hasattr(module, 'weight_orig'):
                try: prune.remove(module, 'weight')
                except ValueError: pass

        # Swap torch.compile'd mixer with uncompiled version for ONNX compatibility
        _compiled_mixer = None
        if hasattr(core, '_spatial_mixer_uncompiled') and core.spatial_mixer is not core._spatial_mixer_uncompiled:
            _compiled_mixer = core.spatial_mixer
            core.spatial_mixer = core._spatial_mixer_uncompiled

        unified = UnifiedModel(core, heads, use_kv=use_kv_cache)
        unified.eval()
        n_latents, d_model = core.n_latents, core.d_model
        
        # Static Shape Enforcement
        # Vital for Qualcomm Hexagon and optimized TensorRT engines.
        with torch.no_grad():
            dummy_tokens = torch.randn(static_batch, static_tokens, d_model)
            dummy_dt = torch.ones(static_batch, 1)
            dummy_state = torch.zeros(static_batch, n_latents, d_model)
            dummy_abs_time = torch.zeros(static_batch, 1)
            # Warm up
            unified(dummy_tokens, dummy_dt, dummy_state, dummy_abs_time)

        input_names = ["sensor_tokens", "dt", "prev_state", "abs_time"]
        output_names = ["next_state"] + [f"head_{k}" for k in unified.ordered_head_keys]

        torch.onnx.export(
            unified,
            (dummy_tokens, dummy_dt, dummy_state, dummy_abs_time),
            export_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True, # Static Folding
            verbose=False
        )
        logging.info(f"✅ Model saved to {export_path} (Opset {opset_version}) [Static={dynamic_axes is None}]")

        # Restore compiled mixer if it was swapped
        if _compiled_mixer is not None:
            core.spatial_mixer = _compiled_mixer

    def export_for_qualcomm_snpe(self, core, heads, export_path: str, static_batch: int = 1, static_tokens: int = 10):
        """Export with static shapes for Hexagon DSP."""
        logging.info(f"🚀 Exporting STATIC ONNX for Qualcomm SNPE to {export_path}...")
        self._trace_and_export(
            core, heads, export_path,
            dynamic_axes=None,
            static_batch=static_batch,
            static_tokens=static_tokens,
            opset_version=13
        )

    def convert_onnx_to_dlc(self, onnx_path: str, dlc_path: str) -> bool:
        """Invokes the SNPE SDK to convert the ONNX model to DLC format."""
        import subprocess
        import os
        
        if not os.path.exists(onnx_path):
            logging.error(f"ONNX file not found: {onnx_path}")
            return False
            
        cmd = [
            "snpe-onnx-to-dlc",
            "-i", onnx_path,
            "-o", dlc_path
        ]
        
        logging.info(f"Running Qualcomm SDK Converter: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logging.info(f"✅ DLC successfully generated: {dlc_path}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ SNPE Conversion Failed. Is the SDK in your PATH?")
            logging.error(e.stderr)
            return False
        except FileNotFoundError:
            logging.error(f"❌ 'snpe-onnx-to-dlc' not found. Please install the Qualcomm Neural Processing SDK.")
            return False

    def export_for_tensorrt(self, core, heads, export_path: str, max_batch: int = 16, max_tokens: int = 128):
        """Export with dynamic axes and optimization profile notes."""
        logging.info(f"🚀 Exporting DYNAMIC ONNX for TensorRT to {export_path}...")
        dynamic_axes = {
            'sensor_tokens': {0: 'batch', 1: 'num_tokens'},
            'dt': {0: 'batch'},
            'prev_state': {0: 'batch'},
            'abs_time': {0: 'batch'},
            'next_state': {0: 'batch'}
        }
        for k in heads.keys():
            dynamic_axes[f"head_{k}"] = {0: 'batch'}
            
        self._trace_and_export(core, heads, export_path, dynamic_axes=dynamic_axes, opset_version=13)
        logging.info(f"⚠️  [JETSON] trtexec --onnx={export_path} --fp16 --maxShapes=sensor_tokens:{max_batch}x{max_tokens}x{core.d_model} ...")

    def load_as_inference(self, omni_path):
        """
        Load an .omni bundle and reconstruct the model deterministically.
        """
        if not os.path.exists(omni_path):
            raise FileNotFoundError(f"Export bundle {omni_path} not found.")

        from .fusion_core import LiquidFusionCore
        from .heads import ClassificationHead, RegressionHead

        # SECURITY NOTE: weights_only=False allows pickle deserialization.
        # Only load .omni bundles from trusted sources.
        # TODO: Migrate to safetensors for production deployment.
        bundle = torch.load(omni_path, map_location='cpu', weights_only=False)
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
                    # For ClassificationHead, output_dim == num_classes
                    h = ClassificationHead(num_classes=output_dim, d_model=d_model_head)
                
                h.load_state_dict(head_state)
                h.eval()
                heads[head_id] = h

        return core, heads, meta
