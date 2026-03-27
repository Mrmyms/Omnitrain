import time
import torch
import numpy as np
import onnxruntime as ort
import os
from omnitrain.fusion_core import FusionCore
from omnitrain.exporter import OmniExporter


def run_benchmark():
    print("\n" + "=" * 60)
    print("🚀 OMNITRAIN 2.0 HIGH-FREQUENCY BENCHMARK")
    print("=" * 60)

    # Test Configuration
    batch_size = 1
    num_tokens = 100
    iterations = 200
    burn_in = 20

    # 1. Prepare Mock Data (Tensor-first for PyTorch, dict for ONNX Runtime)
    dummy_sensor_data = torch.randn(batch_size, num_tokens, 512)
    dummy_timestamps = torch.randn(batch_size, num_tokens, 1)

    dummy_input_ort = {
        "sensor_tokens": np.random.randn(batch_size, num_tokens, 512).astype(np.float32),
        "timestamps": np.random.randn(batch_size, num_tokens, 1).astype(np.float32)
    }

    # 2. Load Models
    print("📥 Loading models...")

    # Baseline: PyTorch (Tensor-first FusionCore)
    core = FusionCore(d_model=768, n_latents=64)
    core.eval()

    # Optimized: ONNX FP32
    session_fp32 = ort.InferenceSession("omni_1_0_edge.onnx", providers=['CPUExecutionProvider'])

    # Optimized: ONNX INT8 (Mixed)
    if os.path.exists("omni_2_0_quant.onnx"):
        session_int8 = ort.InferenceSession("omni_2_0_quant.onnx", providers=['CPUExecutionProvider'])
    else:
        session_int8 = None
        print("⚠  INT8 model not found, skipping...")

    def measure(name, func, iter_count):
        print(f"⏱  Benchmarking {name}...")
        latencies = []
        for i in range(iter_count + burn_in):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            if i >= burn_in:
                latencies.append((end - start) * 1000)  # ms

        avg = np.mean(latencies)
        p99 = np.percentile(latencies, 99)
        print(f"   ➤ Avg: {avg:.4f} ms | P99: {p99:.4f} ms | Hz: {1000/avg:.2f}")
        return avg

    # 3. Run Benchmarks
    with torch.no_grad():
        avg_py = measure("Python (PyTorch Baseline)",
                         lambda: core(dummy_sensor_data, dummy_timestamps), iterations)

    avg_ort = measure("C++ Engine (ONNX FP32)",
                      lambda: session_fp32.run(None, dummy_input_ort), iterations)

    if session_int8:
        avg_int8 = measure("C++ Engine (ONNX INT8 Mixed)",
                           lambda: session_int8.run(None, dummy_input_ort), iterations)
        gain = (avg_py / avg_int8)
    else:
        gain = (avg_py / avg_ort)

    print("\n" + "=" * 60)
    print(f"🏆 RESULT: Speed Gain: {gain:.2f}x")
    print(f"🚀 Status: Ready for {1000/min(avg_ort, avg_py):.1f}Hz Real-time Control.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_benchmark()
