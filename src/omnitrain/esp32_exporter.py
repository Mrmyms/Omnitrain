import struct
import torch
import torch.nn as nn
from pathlib import Path
import yaml
from typing import Dict

class ESP32Exporter:
    """
    Omnitrain to ESP32 binary (.omnibit) exporter.
    Exports network weights into a continuous blob for Zero-Copy access from Flash memory (DROM).
    Includes structured extraction for exact CfC math and OmniShield validation.
    """
    def __init__(self, output_dir="exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.magic = b'OMNI\x04' # Version 4: Ablation Study Support (Arch Flag)
        
    def export(self, model: nn.Module, input_dim: int, d_model: int, output_dim: int, 
               backbone_units: int = 128, heads: Dict[str, nn.Module] = None, config_path: str = None, filename: str = "model.omnibit"):
        out_path = self.output_dir / filename
        
        # Format: list of tuples (name, length, list_of_floats)
        tensor_registry = []
        
        print("[ESP32Exporter] Extracting weights for Zero-Copy export (V3 Structured + TOC)...")
        
        def push_tensor(name, t):
            flat = t.detach().cpu().numpy().flatten().tolist()
            tensor_registry.append((name, len(flat), flat))

        def push_csr_tensor(name, weight_tensor, mask_tensor):
            w = (weight_tensor * mask_tensor).detach().cpu().numpy()
            m = mask_tensor.detach().cpu().numpy()
            values = []
            col_indices = []
            # We store row_ptrs and col_indices as floats using struct unpacking 
            # so they fit in the same float32 array used by the binary format.
            # In C++, we will reinterpret_cast them back to uint32_t.
            row_ptrs = [struct.unpack('f', struct.pack('I', 0))[0]]
            
            for row in range(w.shape[0]):
                for col in range(w.shape[1]):
                    if m[row, col] > 0.5:
                        values.append(float(w[row, col]))
                        col_float = struct.unpack('f', struct.pack('I', col))[0]
                        col_indices.append(col_float)
                row_float = struct.unpack('f', struct.pack('I', len(values)))[0]
                row_ptrs.append(row_float)
                
            tensor_registry.append((f"{name}_val", len(values), values))
            tensor_registry.append((f"{name}_col", len(col_indices), col_indices))
            tensor_registry.append((f"{name}_row", len(row_ptrs), row_ptrs))

        # 1. Input Projector
        if hasattr(model, 'input_projector'):
            proj = model.input_projector.default_proj
            push_tensor('proj_w', proj.weight)
            push_tensor('proj_b', proj.bias)
            
        # 2. Continuous Temporal Encoding (CTE)
        if hasattr(model, 'temporal_encoder'):
            cte = model.temporal_encoder
            push_tensor('cte_inv_freq', cte.inv_freq)
            push_tensor('cte_amp', cte.amplitude)
            push_tensor('cte_phase', cte.phase)
            
        arch_flag = 0 # 0=CfC, 1=GRU, 2=Transformer
        if hasattr(model, 'brain'):
            brain = model.brain
            
            # Helper to extract BioLiquidCell
            def extract_cell(prefix, cell):
                if hasattr(cell, 'sensory_w'):
                    push_tensor(f'{prefix}sensory_w', cell.sensory_w)
                    push_tensor(f'{prefix}sensory_b', cell.sensory_b)
                    
                # Backbone State (iterate all linears)
                for i, layer in enumerate(cell.backbone_state):
                    if isinstance(layer, nn.Linear):
                        push_tensor(f'{prefix}state_w_{i}', layer.weight)
                        push_tensor(f'{prefix}state_b_{i}', layer.bias)
                
                # Backbone Time (iterate all linears)
                for i, layer in enumerate(cell.backbone_time):
                    if isinstance(layer, nn.Linear):
                        push_tensor(f'{prefix}time_w_{i}', layer.weight)
                        push_tensor(f'{prefix}time_b_{i}', layer.bias)
                        
                push_tensor(f'{prefix}ff1_w', cell.ff1.weight)
                push_tensor(f'{prefix}ff1_b', cell.ff1.bias)
                push_tensor(f'{prefix}ff2_w', cell.ff2.weight)
                push_tensor(f'{prefix}ff2_b', cell.ff2.bias)
                push_tensor(f'{prefix}time_a_w', cell.time_a.weight)
                push_tensor(f'{prefix}time_a_b', cell.time_a.bias)
                push_tensor(f'{prefix}time_b_w', cell.time_b.weight)
                push_tensor(f'{prefix}time_b_b', cell.time_b.bias)
                push_tensor(f'{prefix}time_scale', cell.time_scale)

            # Check if NCPBackbone
            if type(brain).__name__ == 'NCPBackbone':
                push_tensor('ncp_sensory_w', brain.sensory_layer.weight)
                push_tensor('ncp_sensory_b', brain.sensory_layer.bias)
                extract_cell('ncp_inter_', brain.inter_cell)
                extract_cell('ncp_cmd_', brain.command_cell)
                push_tensor('ncp_motor_w', brain.motor_layer.weight)
                push_tensor('ncp_motor_b', brain.motor_layer.bias)
            elif type(brain).__name__ == 'GRUBackbone':
                arch_flag = 1
                push_tensor('gru_w_ih', brain.gru.weight_ih)
                push_tensor('gru_w_hh', brain.gru.weight_hh)
                push_tensor('gru_b_ih', brain.gru.bias_ih)
                push_tensor('gru_b_hh', brain.gru.bias_hh)
            elif type(brain).__name__ == 'TransformerBackbone':
                arch_flag = 2
                push_tensor('trf_input_proj_w', brain.input_proj.weight)
                push_tensor('trf_input_proj_b', brain.input_proj.bias)
                push_tensor('trf_wq_w', brain.wq.weight)
                push_tensor('trf_wq_b', brain.wq.bias)
                push_tensor('trf_wk_w', brain.wk.weight)
                push_tensor('trf_wk_b', brain.wk.bias)
                push_tensor('trf_wv_w', brain.wv.weight)
                push_tensor('trf_wv_b', brain.wv.bias)
                push_tensor('trf_wo_w', brain.wo.weight)
                push_tensor('trf_wo_b', brain.wo.bias)
                push_tensor('trf_ffn1_w', brain.ffn1.weight)
                push_tensor('trf_ffn1_b', brain.ffn1.bias)
                push_tensor('trf_ffn2_w', brain.ffn2.weight)
                push_tensor('trf_ffn2_b', brain.ffn2.bias)
                push_tensor('trf_norm1_w', brain.norm1.weight)
                push_tensor('trf_norm1_b', brain.norm1.bias)
                push_tensor('trf_norm2_w', brain.norm2.weight)
                push_tensor('trf_norm2_b', brain.norm2.bias)
            elif type(brain).__name__ in ['ContinuousCfCCell', 'ContinuousCfCFull']:
                arch_flag = 3
                push_tensor('cfc_bb_w', brain.backbone[0].weight)
                push_tensor('cfc_bb_b', brain.backbone[0].bias)
                push_tensor('cfc_f_w', brain.f_head.weight)
                push_tensor('cfc_f_b', brain.f_head.bias)
                push_tensor('cfc_g_w', brain.g_head.weight)
                push_tensor('cfc_g_b', brain.g_head.bias)
                push_tensor('cfc_h_w', brain.h_head.weight)
                push_tensor('cfc_h_b', brain.h_head.bias)
            else:
                # Legacy BioLiquidCell
                extract_cell('legacy_', brain)
        elif type(model).__name__ in ['ContinuousCfCFull', 'ContinuousCfC']:
            arch_flag = 3
            push_tensor('cfc_bb_w', model.backbone[0].weight)
            push_tensor('cfc_bb_b', model.backbone[0].bias)
            push_tensor('cfc_f_w', model.f_head.weight)
            push_tensor('cfc_f_b', model.f_head.bias)
            push_tensor('cfc_g_w', model.g_head.weight)
            push_tensor('cfc_g_b', model.g_head.bias)
            push_tensor('cfc_h_w', model.h_head.weight)
            push_tensor('cfc_h_b', model.h_head.bias)
            
            # Export FC layer as the sole head
            if hasattr(model, 'fc'):
                push_tensor('fc_w', model.fc.weight)
                push_tensor('fc_b', model.fc.bias)
        elif type(model).__name__ == 'SparseCfC':
            arch_flag = 4
            push_csr_tensor('cfc_bb_w', model.backbone_weight, model.mask)
            push_tensor('cfc_bb_b', model.backbone_bias)
            push_tensor('cfc_f_w', model.f_weight)
            push_tensor('cfc_f_b', model.f_bias)
            push_tensor('cfc_g_w', model.g_weight)
            push_tensor('cfc_g_b', model.g_bias)
            push_tensor('cfc_h_w', model.h_weight)
            push_tensor('cfc_h_b', model.h_bias)
            
            if hasattr(model, 'fc'):
                push_tensor('fc_w', model.fc.weight)
                push_tensor('fc_b', model.fc.bias)

        # 4. Heads
        if heads:
            for head_id, head_module in heads.items():
                # Extract all linears dynamically
                for name, module in head_module.named_modules():
                    if isinstance(module, nn.Linear):
                        push_tensor(f'head_{head_id}_{name}_w', module.weight)
                        push_tensor(f'head_{head_id}_{name}_b', module.bias)

        # Flatten weights and prepare TOC
        toc_sizes = [size for name, size, flat in tensor_registry]
        total_weights = sum(toc_sizes)
        all_weights = []
        for name, size, flat in tensor_registry:
            all_weights.extend(flat)

        # Save to Binary Format (.omnibit)
        with open(out_path, "wb") as f:
            # Magic (5) + ArchFlag (1) + Padding (2) = 8 bytes total
            f.write(self.magic + struct.pack('<B', arch_flag) + b'\x00\x00') 
            
            # Dimensions Header (24 bytes)
            num_tensors = len(toc_sizes)
            f.write(struct.pack('<IIIIII', input_dim, d_model, output_dim, backbone_units, total_weights, num_tensors))
            
            # TOC array (num_tensors * 4 bytes)
            if num_tensors > 0:
                f.write(struct.pack(f'<{num_tensors}I', *toc_sizes))
            
            # Weights array
            if total_weights > 0:
                f.write(struct.pack(f'<{total_weights}f', *all_weights))
            
        print(f"[ESP32Exporter] Successfully exported to {out_path}")
        size_kb = out_path.stat().st_size / 1024.0
        print(f"[ESP32Exporter] Flash footprint: {size_kb:.2f} KB ({total_weights} parameters, {num_tensors} tensors)")
        
        if config_path:
            self._export_hw_config(config_path, self.output_dir / "esp32_hw_config.h")

        return out_path
    def _export_hw_config(self, yaml_path, out_header):
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            return
            
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            
        hw = config.get('hardware', {}).get('esp32', {})
        if not hw:
            return
            
        with open(out_header, 'w') as f:
            f.write("#ifndef ESP32_HW_CONFIG_H\n")
            f.write("#define ESP32_HW_CONFIG_H\n\n")
            f.write("// Auto-generated by Omnitrain ESP32Exporter\n\n")
            for key, val in hw.items():
                macro = key.upper()
                f.write(f"#define HW_{macro} {val}\n")
            f.write("\n#endif // ESP32_HW_CONFIG_H\n")
