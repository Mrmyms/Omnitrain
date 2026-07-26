import struct
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import yaml
from typing import Dict, Optional

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
        elif type(model).__name__ == 'SparseCfCMixed':
            # ── Mixed-Precision Export (V5) ──
            # This path uses the new export_mixed method instead of the
            # standard float32 pipeline. We return early.
            return self.export_mixed(model, input_dim, d_model, output_dim,
                                     backbone_units, heads, config_path, filename)
        elif type(model).__name__ == 'DiscreteRNN':
            if isinstance(model.rnn, nn.LSTM):
                arch_flag = 5
                push_tensor('lstm_w_ih', model.rnn.weight_ih_l0)
                push_tensor('lstm_w_hh', model.rnn.weight_hh_l0)
                push_tensor('lstm_b_ih', model.rnn.bias_ih_l0)
                push_tensor('lstm_b_hh', model.rnn.bias_hh_l0)
            elif isinstance(model.rnn, nn.GRU):
                arch_flag = 1
                push_tensor('gru_w_ih', model.rnn.weight_ih_l0)
                push_tensor('gru_w_hh', model.rnn.weight_hh_l0)
                push_tensor('gru_b_ih', model.rnn.bias_ih_l0)
                push_tensor('gru_b_hh', model.rnn.bias_hh_l0)
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
        
        # Auto-generate C-Header for extremely easy usage (no filesystem required)
        c_header_path = out_path.with_suffix('.h')
        with open(out_path, "rb") as f:
            binary_data = f.read()
            
        with open(c_header_path, "w") as f:
            array_name = filename.replace('.', '_').replace('-', '_')
            f.write(f"// Auto-generated by Omnitrain ESP32Exporter\n")
            f.write(f"// Flash footprint: {size_kb:.2f} KB\n")
            f.write(f"#ifndef {array_name.upper()}_H\n")
            f.write(f"#define {array_name.upper()}_H\n\n")
            f.write(f"#include <stddef.h>\n\n")
            f.write(f"const unsigned char {array_name}[] = {{\n    ")
            
            hex_data = [f"0x{b:02x}" for b in binary_data]
            for i in range(0, len(hex_data), 12):
                f.write(", ".join(hex_data[i:i+12]))
                if i + 12 < len(hex_data):
                    f.write(",\n    ")
                    
            f.write(f"\n}};\n")
            f.write(f"const size_t {array_name}_len = {len(binary_data)};\n\n")
            f.write(f"#endif // {array_name.upper()}_H\n")
            
        print(f"[ESP32Exporter] Auto-generated C-Header for direct include: {c_header_path}")
        
        if config_path:
            self._export_hw_config(config_path, self.output_dir / "esp32_hw_config.h")

        return out_path

    # ═══════════════════════════════════════════════════════════════════
    #  Mixed-Precision Export (OMNI V5)
    # ═══════════════════════════════════════════════════════════════════

    def export_mixed(self, model, input_dim, d_model, output_dim,
                     backbone_units=128, heads=None, config_path=None,
                     filename='model_mixed.omnibit'):
        """
        Export a SparseCfCMixed model with per-core precision packing.
        
        Binary layout (V5):
          [0:5]   Magic: OMNI\x05
          [5]     ArchFlag: 6 (SparseCfCMixed)
          [6:8]   Precision map: [sensory, inter, command, timegate] (4 nibbles in 2 bytes)
          [8:32]  Dimensions header (6 × uint32)
          [32:N]  TOC: per-tensor (size_bytes: uint32, dtype_flag: uint8, scale: float32 = 9 bytes each)
          [N:]    Packed weight data
        """
        from .sparse_cfc_mixed import QuantGenotype, PRECISION_LEVELS, compute_scale
        
        out_path = self.output_dir / filename
        g = model.genotype
        magic_v5 = b'OMNI\x05'
        arch_flag = 6  # SparseCfCMixed
        
        # Encode precision map: 4 genes packed into 2 bytes (one nibble each)
        prec_byte0 = (g.sensory & 0x0F) | ((g.inter & 0x0F) << 4)
        prec_byte1 = (g.command & 0x0F) | ((g.timegate & 0x0F) << 4)
        
        # Registry: list of (name, packed_bytes, dtype_flag, scale, n_params)
        # dtype_flag: 0=int4, 1=int8, 2=fp16, 3=fp32
        packed_registry = []
        
        def pack_tensor(name, tensor, precision):
            """Pack a tensor at the specified precision and register it."""
            t = tensor.detach().cpu()
            n_params = t.numel()
            flat = t.numpy().flatten()
            cfg = PRECISION_LEVELS[precision]
            
            if precision == 0:  # INT4 — nibble packing
                scale = compute_scale(t, 0)
                quantized = np.clip(np.round(flat / scale), -8, 7).astype(np.int8)
                # Pack two int4 values per byte
                n_bytes = (n_params + 1) // 2
                packed = bytearray(n_bytes)
                for i in range(0, n_params, 2):
                    lo = int(quantized[i]) & 0x0F
                    hi = (int(quantized[i+1]) & 0x0F) << 4 if i+1 < n_params else 0
                    packed[i // 2] = lo | hi
                packed_registry.append((name, bytes(packed), 0, scale, n_params))
                
            elif precision == 1:  # INT8
                scale = compute_scale(t, 1)
                quantized = np.clip(np.round(flat / scale), -127, 127).astype(np.int8)
                packed = struct.pack(f'{n_params}b', *quantized.tolist())
                packed_registry.append((name, packed, 1, scale, n_params))
                
            elif precision == 2:  # FP16
                half_vals = flat.astype(np.float16)
                packed = half_vals.tobytes()
                packed_registry.append((name, packed, 2, 1.0, n_params))
                
            else:  # FP32
                packed = struct.pack(f'<{n_params}f', *flat.tolist())
                packed_registry.append((name, packed, 3, 1.0, n_params))
        
        def pack_csr_mixed(name, weight_tensor, mask_tensor, precision):
            """Pack CSR sparse backbone at specified precision."""
            w = (weight_tensor * mask_tensor).detach().cpu().numpy()
            m = mask_tensor.detach().cpu().numpy()
            values = []
            col_indices = []
            row_ptrs = [0]
            
            for row in range(w.shape[0]):
                for col in range(w.shape[1]):
                    if m[row, col] > 0.5:
                        values.append(float(w[row, col]))
                        col_indices.append(col)
                row_ptrs.append(len(values))
            
            # Values get quantized at their precision
            val_tensor = torch.tensor(values, dtype=torch.float32)
            pack_tensor(f'{name}_val', val_tensor, precision)
            
            # Indices are always uint32 (structural, not arithmetic)
            col_packed = struct.pack(f'<{len(col_indices)}I', *col_indices)
            packed_registry.append((f'{name}_col', col_packed, 3, 1.0, len(col_indices)))
            row_packed = struct.pack(f'<{len(row_ptrs)}I', *row_ptrs)
            packed_registry.append((f'{name}_row', row_packed, 3, 1.0, len(row_ptrs)))
        
        print(f"[ESP32Exporter] Mixed-Precision Export (V5): {g}")
        
        # ── Pack all parameter groups at their core's precision ──
        
        # Sensory backbone (columns 0:input_dim) — split from full backbone
        bb_full = (model.backbone_weight * model.mask).detach().cpu()
        bb_sensory = bb_full[:, :input_dim]
        bb_recurrent = bb_full[:, input_dim:]
        
        # Pack sensory partition
        pack_tensor('cfc_bb_sensory', bb_sensory, g.sensory)
        # Pack recurrent partition at inter precision
        pack_tensor('cfc_bb_recurrent', bb_recurrent, g.inter)
        # Backbone bias at inter precision
        pack_tensor('cfc_bb_b', model.backbone_bias, g.inter)
        
        # Timegate (ODE solver — critical precision)
        pack_tensor('cfc_f_w', model.f_weight, g.timegate)
        pack_tensor('cfc_f_b', model.f_bias, g.timegate)
        
        # Inter-neuron state (memory)
        pack_tensor('cfc_g_w', model.g_weight, g.inter)
        pack_tensor('cfc_g_b', model.g_bias, g.inter)
        pack_tensor('cfc_h_w', model.h_weight, g.inter)
        pack_tensor('cfc_h_b', model.h_bias, g.inter)
        
        # Command output
        pack_tensor('fc_w', model.fc.weight, g.command)
        pack_tensor('fc_b', model.fc.bias, g.command)
        
        # ── Write binary ──
        num_tensors = len(packed_registry)
        total_data_bytes = sum(len(data) for _, data, _, _, _ in packed_registry)
        
        with open(out_path, 'wb') as f:
            # Header: magic(5) + arch(1) + prec_map(2) = 8 bytes
            f.write(magic_v5 + struct.pack('<B', arch_flag))
            f.write(struct.pack('BB', prec_byte0, prec_byte1))
            
            # Dimensions: 6 × uint32 = 24 bytes
            f.write(struct.pack('<IIIIII', input_dim, d_model, output_dim,
                                backbone_units, total_data_bytes, num_tensors))
            
            # TOC: per tensor → (data_size: uint32, dtype_flag: uint8, scale: float32) = 9 bytes each
            for name, data, dtype_flag, scale, n_params in packed_registry:
                f.write(struct.pack('<IBf', len(data), dtype_flag, scale))
            
            # Weight data
            for name, data, dtype_flag, scale, n_params in packed_registry:
                f.write(data)
        
        size_kb = out_path.stat().st_size / 1024.0
        total_params = sum(n for _, _, _, _, n in packed_registry)
        print(f"[ESP32Exporter] Successfully exported V5 mixed-precision to {out_path}")
        print(f"[ESP32Exporter] Flash footprint: {size_kb:.2f} KB ({total_params} params, {num_tensors} tensors)")
        
        # Precision breakdown
        for name, data, dtype_flag, scale, n_params in packed_registry:
            dtype_name = ['INT4', 'INT8', 'FP16', 'FP32'][dtype_flag]
            print(f"  {name:25s} → {dtype_name:5s} | {len(data):6d} bytes | {n_params} params | scale={scale:.6f}")
        
        # Auto-generate C-Header
        c_header_path = out_path.with_suffix('.h')
        with open(out_path, 'rb') as f:
            binary_data = f.read()
        
        with open(c_header_path, 'w') as f:
            array_name = filename.replace('.', '_').replace('-', '_')
            f.write(f"// Auto-generated by Omnitrain ESP32Exporter (V5 Mixed-Precision)\n")
            f.write(f"// Precision: sensory={PRECISION_LEVELS[g.sensory]['name']}, "
                    f"inter={PRECISION_LEVELS[g.inter]['name']}, "
                    f"command={PRECISION_LEVELS[g.command]['name']}, "
                    f"timegate={PRECISION_LEVELS[g.timegate]['name']}\n")
            f.write(f"// Flash footprint: {size_kb:.2f} KB\n")
            f.write(f"#ifndef {array_name.upper()}_H\n")
            f.write(f"#define {array_name.upper()}_H\n\n")
            f.write(f"#include <stddef.h>\n\n")
            f.write(f"const unsigned char {array_name}[] = {{\n    ")
            
            hex_data = [f"0x{b:02x}" for b in binary_data]
            for i in range(0, len(hex_data), 12):
                f.write(", ".join(hex_data[i:i+12]))
                if i + 12 < len(hex_data):
                    f.write(",\n    ")
            
            f.write(f"\n}};\n")
            f.write(f"const size_t {array_name}_len = {len(binary_data)};\n\n")
            f.write(f"#endif // {array_name.upper()}_H\n")
        
        print(f"[ESP32Exporter] Auto-generated C-Header: {c_header_path}")
        
        if config_path:
            self._export_hw_config(config_path, self.output_dir / 'esp32_hw_config.h')
        
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
