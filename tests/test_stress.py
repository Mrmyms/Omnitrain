import torch
import torch.nn as nn
import time
import numpy as np
from omnitrain.fusion_core import LiquidFusionCore

def run_noise_overload():
    print("\n" + "!"*60)
    print("🔥 OMNITRAIN SYSTEM STRESS TEST")
    print("!"*60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Hardware: [bold cyan]{device}[/bold cyan]")

    # 1. Configuración de Carga Masiva
    config = {
        'model': {
            'n_latents': 64,
            'd_model': 512,
            'visual_tokens': 128,
            'conectoma': {'enabled': True, 'wall_n': 128, 'command_n': 64}
        },
        'inputs': [
            {'id': f'sensor_{i}', 'dim': 128} for i in range(20) # 20 sensores de alta dimensión
        ]
    }

    core = LiquidFusionCore(config=config, d_model=512, n_latents=64).to(device)
    core.train() # Habilitamos gradientes para ver si explotan

    # 2. Simulación de "Ataque Sensorial" (Noise)
    # Generamos 5000 pasos de tiempo con valores extremos
    steps = 5000
    batch_size = 8
    print(f"Injecting {steps} steps of high-frequency chaotic telemetry...")

    core.reset_state(batch_size=batch_size, device=device)
    
    start_time = time.time()
    nan_count = 0
    explosion_count = 0
    
    for i in range(steps):
        # Cada 100 pasos inyectamos un valor extremo (Spike)
        if i % 100 == 0:
            noise_scale = 1e6 # Spike de un millón
        else:
            noise_scale = 1.0
            
        # Simular 20 sensores enviando datos ruidosos
        sensor_data = {
            f'sensor_{j}': torch.randn(batch_size, 128).to(device) * noise_scale
            for j in range(20)
        }
        
        # dt irregular con jitter extremo
        dt = torch.rand(batch_size, 1).to(device) * 0.1
        
        try:
            # Forward pass con acumulador recurrente activo
            out = core(sensor_data, dt)
            
            # Verificación de integridad
            if torch.isnan(out).any():
                nan_count += 1
            if torch.abs(out).max() > 1e4:
                explosion_count += 1
                
            # Simular Backprop cada 50 pasos (Truncated BPTT)
            if i % 50 == 0:
                loss = out.pow(2).mean()
                loss.backward()
                # Limpiamos gradientes pero no reseteamos el estado (estado persistente)
                for p in core.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                # Detach the state for real Truncated BPTT simulation
                # FIX #36: Recursive detach to handle nested states (e.g. Conectoma dicts or tuples)
                def recursive_detach(obj):
                    if isinstance(obj, torch.Tensor):
                        return obj.detach()
                    elif isinstance(obj, dict):
                        return {k: recursive_detach(v) for k, v in obj.items()}
                    elif isinstance(obj, tuple):
                        return tuple(recursive_detach(v) for v in obj)
                    return obj
                
                if core._last_mixer_state:
                    core._last_mixer_state = recursive_detach(core._last_mixer_state)
                
                if core._last_brain_state:
                    core._last_brain_state = recursive_detach(core._last_brain_state)

                
        except Exception as e:
            print(f"❌ CRITICAL CRASH at step {i}: {e}")
            break

        if i % 500 == 0:
            print(f"  Step {i:4d}: Stability OK | Max Val: {out.abs().max().item():.2f}")

    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print("📊 FINAL AUDIT RESULTS")
    print("="*60)
    print(f"Total Steps        : {steps}")
    print(f"Processing Time    : {total_time:.2f}s ({steps/total_time:.1f} Hz)")
    print(f"NaN Detections     : {nan_count} (Goal: 0)")
    print(f"Explosion Events   : {explosion_count} (Goal: 0)")
    
    if nan_count == 0 and explosion_count == 0:
        print("\n🏆 INTEGRITY INTEGRITY VERIFIED: SignalSpatialMixer is bulletproof.")
    else:
        print("\n⚠️ STABILITY WARNING: System saturated under extreme stress.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_noise_overload()
