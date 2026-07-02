# NVIDIA Jetson Port Guide (CUDA + TensorRT)

## Why Jetson is NOT the same as Raspberry Pi
This is the most important distinction in OmniTrain's platform support:

| Feature | Raspberry Pi 4 | Jetson Nano | Jetson Orin Nano |
| :--- | :--- | :--- | :--- |
| CPU | Cortex-A72 | Cortex-A57 | Cortex-A78 |
| GPU | VideoCore (display only) | **128 CUDA cores** | **1024 CUDA cores** |
| AI Acceleration | ❌ None | ✅ CUDA + TensorRT | ✅ CUDA + TensorRT + DLA |
| Inference (NEON) | ~2.5ms | ~2.5ms | ~0.8ms |
| Inference (TensorRT FP32) | ❌ N/A | **~0.4ms** | **~0.1ms** |
| Inference (TensorRT FP16) | ❌ N/A | **~0.15ms** | **~0.05ms** |
| On-device Training | ❌ Very slow | ✅ Feasible | ✅ Fast |

> **Bottom line:** A Jetson with TensorRT is **6x–50x faster** than a Raspberry Pi for AI inference. If your robot needs real-time visual fusion or >1kHz control loops, you need a Jetson.

---

## Two Inference Modes

### Mode 1: CPU / ARM NEON (Simple, no CUDA setup needed)
Use this if you want to get started quickly. Uses `OmniEngineNEON.hpp` exactly like the Raspberry Pi.

```cpp
#include "hal/jetson/OmniHAL.hpp"
#include "OmniEngineNEON.hpp"   // ARM NEON SIMD

OmniEngineTarget engine;  // ~2.5ms per step (400Hz)
```

### Mode 2: GPU / TensorRT (Recommended for production)
Uses `OmniEngineTensorRT.hpp`. Requires an `.engine` file compiled from ONNX.

```cpp
#include "hal/jetson/OmniHAL.hpp"
#include "OmniEngineTensorRT.hpp"

OmniEngineTensorRT engine;  // ~0.15ms per step (6600Hz in FP16)
```

---

## Full TensorRT Workflow

### Step 1: Export from Python (on your PC)
```python
from omnitrain import LiquidFusionCore
from omnitrain.jetson_exporter import JetsonExporter

model = LiquidFusionCore.load("my_robot.omni")
exporter = JetsonExporter(output_dir="exports/jetson")
exporter.export(model, input_dim=8, d_model=128, output_dim=4)
# → Produces exports/jetson/omni_brain.onnx
```

### Step 2: Compile to TensorRT Engine (on the Jetson)
Copy the `.onnx` to the Jetson and run:
```bash
# Standard FP32 (safe, accurate)
trtexec --onnx=omni_brain.onnx --saveEngine=bot_brain.engine

# FP16 (2x faster, minimal accuracy loss — recommended for most robots)
trtexec --onnx=omni_brain.onnx --saveEngine=bot_brain_fp16.engine --fp16

# INT8 (4x faster, requires calibration data)
trtexec --onnx=omni_brain.onnx --saveEngine=bot_brain_int8.engine --int8
```

### Step 3: C++ Inference
```cpp
#include "hal/jetson/OmniHAL.hpp"
#include "OmniEngineTensorRT.hpp"
#include "OmniShield.hpp"

OmniEngineTensorRT engine;
OmniShieldGuard shield(8, 4);

int main() {
    // Load and map the .engine file
    if (!engine.LoadEngine("bot_brain_fp16.engine")) {
        fprintf(stderr, "Failed to load TRT engine!\n");
        return 1;
    }

    float abs_time = 0.0f;

    while (true) {
        float sensors[8] = { read_all_sensors() };

        // GPU inference: ~0.15ms in FP16
        auto intent = engine.Step(sensors, 0.001f, abs_time);

        bool intervened;
        auto action = shield.Enforce(sensors, intent, intervened);

        drive_actuators(action);
        abs_time += 0.001f;
    }
}
```

### `CMakeLists.txt` for Jetson
```cmake
cmake_minimum_required(VERSION 3.18)
project(omnibot_jetson LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

add_executable(omnibot main.cpp)
target_include_directories(omnibot PRIVATE
    /usr/include/aarch64-linux-gnu
    /path/to/omnitrain/src/cpp_engine/core/include
)
target_link_libraries(omnibot
    nvinfer
    nvinfer_plugin
    cudart
)
target_compile_options(omnibot PRIVATE -O3 -march=native)
```

---

## Supported Jetson Boards

| Board | GPU | TFLOPS (FP16) | Max Model d_model |
| :--- | :--- | :--- | :--- |
| Jetson Nano 4GB | Maxwell 128 | 0.47 | 256 |
| Jetson Orin Nano 8GB | Ampere 1024 | 40.0 | 512+ |
| Jetson Xavier NX | Volta 384 + DLA | 21.0 | 512 |
| Jetson AGX Orin 64GB | Ampere 2048 | 275.0 | 1024+ |

---

## On-Device Sim-to-Real Fine-Tuning
The Jetson's CUDA GPU is fast enough to run incremental fine-tuning directly on the robot:
```bash
# After a field session, fine-tune on the new experience data
python3 fine_tune.py --base-model bot_brain.omni --new-data session_data.csv
# Then re-export and re-compile the TRT engine
```
