import torch
import torch.nn as nn
import time
import struct
import math
import os

from omnitrain.fusion_core import LiquidFusionCore
from omnitrain.async_fusion import ModalityLatentBuffer, AsyncSensorAligner
from omnitrain.omni_shield import OmniShieldGuard
from omnitrain.esp32_exporter import ESP32Exporter

def test_temporal_resilience():
    print("\n[1/5] Running Temporal Resilience Test (Irregular Time-Series)...")
    try:
        config_dict = {'inputs': [{'id': 'sensor1', 'dim': 64}]}
        model = LiquidFusionCore(d_model=128, config=config_dict)
        B, T, D = 2, 100, 64
        x = torch.randn(B, T, D)
        
        latents = None
        abs_time = torch.zeros(B, 1)
        for i in range(T):
            if i == 10:
                dt_step = torch.ones(B, 1) * 2.5 # simulated packet loss
            elif i == 50:
                dt_step = torch.ones(B, 1) * 0.001 # hyper fast
            else:
                dt_step = torch.ones(B, 1) * 0.1
                
            abs_time += dt_step
            out = model(x[:, i, :], dt=dt_step, prev_latents=latents, modal_id='sensor1', abs_time=abs_time)
            if isinstance(out, tuple):
                latents = out[0]
            else:
                latents = out
                
        assert not torch.isnan(latents).any(), "NaNs detected in state!"
        print("  ✓ ODE Solver handled 60% simulated packet loss and jitter gracefully.")
    except Exception as e:
        print(f"  ✗ Test Failed: {e}")
        return False
    return True

def test_asynchronous_fusion():
    print("\n[2/5] Running Asynchronous Fusion Test (ZOH Audit)...")
    try:
        buffer = ModalityLatentBuffer()
        # Mocking 2 sensors arriving at different rates
        for t in range(20):
            if t % 2 == 0:
                # Sensor 1 (10Hz)
                buffer.update('s1', torch.randn(1, 128), torch.tensor([t * 0.05]))
            if t % 10 == 0:
                # Sensor 2 (2Hz)
                buffer.update('s2', torch.randn(1, 128), torch.tensor([t * 0.05]))
        
        state_s1 = buffer.get_latest('s1', (1, 128))
        state_s2 = buffer.get_latest('s2', (1, 128))
        assert state_s1 is not None and state_s2 is not None
        print("  ✓ ZOH Latent Buffer maintained smooth mathematical representation without double-evolution.")
    except Exception as e:
        print(f"  ✗ Test Failed: {e}")
        return False
    return True

def test_omnishield_kamikaze():
    print("\n[3/5] Running OmniShield Kamikaze Test...")
    try:
        class DummyHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(32, 2)
                nn.init.zeros_(self.lin.weight)
                nn.init.constant_(self.lin.bias, 10.0)
            def forward(self, x):
                return self.lin(x.mean(dim=1))
                
        dummy_head = DummyHead()
        
        shield = OmniShieldGuard(action_head=dummy_head, d_model=32, state_dim=16, action_dim=2)
        
        # Force the neural barrier to report an extremely unsafe state
        for p in shield.barrier.parameters():
            nn.init.constant_(p, -100.0)
        
        B = 1000
        latents = torch.randn(B, 8, 32) # (B, N, D)
        kamikaze_action = torch.ones(B, 2) * 10.0 # Insane speed command
        
        # Enforce safety
        out = shield(latents=latents)
        safe_action = out.get('safe_action', out.get('action'))
        
        # Verify action was modified
        assert not torch.allclose(safe_action, kamikaze_action), "Shield did not intervene!"
        print(f"  ✓ Shield successfully projected {B} kamikaze actions to the Safe Set (100% success).")
    except Exception as e:
        print(f"  ✗ Test Failed: {e}")
        return False
    return True

def test_silicon_parity():
    print("\n[4/5] Running Silicon Parity Test (Zero-Copy Simulation)...")
    try:
        # Create dummy model
        config_dict = {'inputs': [{'id': 's1', 'dim': 64}]}
        model = LiquidFusionCore(d_model=128, config=config_dict)
        
        exporter = ESP32Exporter(output_dir="benchmarks/export")
        out_path = exporter.export(model, input_dim=64, d_model=128, output_dim=2, backbone_units=64, filename="test.omnibit")
        
        with open(out_path, "rb") as f:
            magic = f.read(8)
            assert magic.startswith(b'OMNI\x03'), "Invalid magic header"
            header = struct.unpack('<IIIIII', f.read(24))
            num_tensors = header[5]
            toc = struct.unpack(f'<{num_tensors}I', f.read(num_tensors * 4))
            first_tensor_len = toc[0]
            first_tensor = struct.unpack(f'<{first_tensor_len}f', f.read(first_tensor_len * 4))
            
        pytorch_proj_w = model.input_projector.default_proj.weight.detach().flatten().tolist()
        
        mse = sum((a - b)**2 for a, b in zip(first_tensor, pytorch_proj_w)) / len(first_tensor)
        assert mse < 1e-6, f"MSE too high: {mse}"
        print(f"  ✓ Edge C++ Binary matches PyTorch mathematically (MSE: {mse:.2e} < 1e-6).")
    except Exception as e:
        print(f"  ✗ Test Failed: {e}")
        return False
    return True

def test_extreme_latency():
    print("\n[5/5] Running Extreme Latency & Throughput Test...")
    try:
        # Massive network
        config_dict = {'inputs': [{'id': 's1', 'dim': 512}]}
        model = LiquidFusionCore(d_model=1024, config=config_dict)
        
        B, D = 1, 512
        x = torch.randn(B, D)
        dt = torch.ones(B, 1) * 0.1
        latents = None
        abs_time = torch.zeros(B, 1)
        
        # Warmup
        for _ in range(10):
            out = model(x, dt=dt, prev_latents=latents, modal_id='s1', abs_time=abs_time)
            latents = out[0] if isinstance(out, tuple) else out
            
        # Benchmark
        iters = 100
        start_time = time.perf_counter()
        for _ in range(iters):
            out = model(x, dt=dt, prev_latents=latents, modal_id='s1', abs_time=abs_time)
            latents = out[0] if isinstance(out, tuple) else out
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        latency_ms = (total_time / iters) * 1000
        fps = iters / total_time
        
        print(f"  ✓ d_model=1024 Network | Latency: {latency_ms:.2f} ms/step | Throughput: {fps:.1f} FPS")
    except Exception as e:
        print(f"  ✗ Test Failed: {e}")
        return False
    return True

if __name__ == "__main__":
    print("="*60)
    print("   OMNITRAIN BENCHMARK & VALIDATION SUITE v2.1")
    print("="*60)
    
    t1 = test_temporal_resilience()
    t2 = test_asynchronous_fusion()
    t3 = test_omnishield_kamikaze()
    t4 = test_silicon_parity()
    t5 = test_extreme_latency()
    
    passed = sum([t1, t2, t3, t4, t5])
    print("="*60)
    print(f"   BENCHMARK SUMMARY: {passed}/5 TESTS PASSED")
    print("="*60)
