import struct
import torch
import torch.nn as nn
from pathlib import Path
import yaml

class ESP32Exporter:
    """
    Exportador de Omnitrain a formato binario .omnibit para ESP32.
    Exporta los pesos en un blob continuo para acceso Zero-Copy desde memoria Flash (DROM).
    Incluye extracción estructurada para la matemática exacta CfC y OmniShield.
    """
    def __init__(self, output_dir="exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.magic = b'OMNI\x02' # Version 2: Estructurada
        
    def export(self, model: nn.Module, input_dim: int, d_model: int, output_dim: int, 
               backbone_units: int = 128, config_path: str = None, filename: str = "bot_brain.omnibit"):
        out_path = self.output_dir / filename
        weights = []
        
        print("[ESP32Exporter] Extrayendo pesos para exportación Zero-Copy (V2 Estructurada)...")
        
        def push_tensor(t):
            weights.extend(t.detach().cpu().numpy().flatten().tolist())

        # 1. Proyector de Entrada
        if hasattr(model, 'input_projector'):
            proj = model.input_projector.default_proj
            push_tensor(proj.weight)
            push_tensor(proj.bias)
            
        # 2. Codificación Temporal Continua (CTE)
        if hasattr(model, 'temporal_encoder'):
            cte = model.temporal_encoder
            push_tensor(cte.inv_freq)
            push_tensor(cte.amplitude)
            push_tensor(cte.phase)
            
        # 3. BioLiquidCell (Extracción explícita para motor C++)
        # Asumiendo mode="default" y backbone_layers=1
        if hasattr(model, 'brain'):
            brain = model.brain
            if hasattr(brain, 'sensory_w'):
                push_tensor(brain.sensory_w)
                push_tensor(brain.sensory_b)
                
                # Backbone State (Linear 0)
                state_lin = brain.backbone_state[0]
                push_tensor(state_lin.weight)
                push_tensor(state_lin.bias)
                
                # Backbone Time (Linear 0)
                time_lin = brain.backbone_time[0]
                push_tensor(time_lin.weight)
                push_tensor(time_lin.bias)
                
                # FF1 & FF2
                push_tensor(brain.ff1.weight)
                push_tensor(brain.ff1.bias)
                push_tensor(brain.ff2.weight)
                push_tensor(brain.ff2.bias)
                
                # Time Gates
                push_tensor(brain.time_a.weight)
                push_tensor(brain.time_a.bias)
                push_tensor(brain.time_b.weight)
                push_tensor(brain.time_b.bias)
                
                # Time Scale
                push_tensor(brain.time_scale)

        # 4. TODO: Añadir Exportación del regresor y OmniShield

        # Guardado en formato Binario (.omnibit)
        with open(out_path, "wb") as f:
            # Magic + Padding (3 bytes) para asegurar alineación de offsets de floats a 4 bytes
            f.write(self.magic + b'\x00\x00\x00') 
            
            # Dimensions Header (20 bytes)
            # Layout: input_dim, d_model, output_dim, backbone_units, len(weights)
            # Offset actual: 8 bytes. Al sumar 20 bytes -> Offset de floats: 28 bytes (múltiplo de 4)
            f.write(struct.pack('<IIIII', input_dim, d_model, output_dim, backbone_units, len(weights)))
            
            # Weights array
            if len(weights) > 0:
                f.write(struct.pack(f'<{len(weights)}f', *weights))
            
        print(f"[ESP32Exporter] Exportado exitosamente a {out_path}")
        size_kb = out_path.stat().st_size / 1024.0
        print(f"[ESP32Exporter] Tamaño en Flash: {size_kb:.2f} KB ({len(weights)} parámetros)")
        
        if config_path:
            self._export_hw_config(config_path, self.output_dir / "esp32_hw_config.h")

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
            f.write("// Auto-generado por ESP32Exporter de Omnitrain\n\n")
            for key, val in hw.items():
                macro = key.upper()
                f.write(f"#define HW_{macro} {val}\n")
            f.write("\n#endif // ESP32_HW_CONFIG_H\n")
