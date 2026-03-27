#include "OmniEngine.hpp"
#include <cpu_provider_factory.h>

// Note: For industrial deployment, these headers are conditionally included
// when compiling with GPU support, used here for the fallback logic.
#ifdef USE_CUDA
#include <cuda_provider_factory.h>
#include <tensorrt_provider_factory.h>
#endif

OmniEngine::OmniEngine(const std::string& model_path) {
    // 1. Global Session Optimization
    session_options_.SetIntraOpNumThreads(4); // Optimized for 4-8 core Edge CPUs
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // 2. Hardware Cascade Configuration (The Ollama Effect)
    ConfigureExecutionProviders();

    // 3. Session Instantiation (graph is validated against hardware here)
    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        std::cout << "✔ OmniEngine: Session initialized successfully." << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "✘ Error loading ONNX model: " << e.what() << std::endl;
        throw;
    }
}

void OmniEngine::ConfigureExecutionProviders() {
    /**
     * INDUSTRIAL FALLBACK STRATEGY v3.0:
     * DLA -> TensorRT (GPU) -> CUDA -> CPU
     * We inject execution providers in descending performance-per-watt order.
     * If the driver or hardware is not present, ONNX Runtime throws an exception
     * when adding the provider, which we catch to continue the cascade.
     */

    bool provider_added = false;

    // 0. Ultimate Priority: NVIDIA DLA (Deep Learning Accelerator)
    // Available on Jetson Orin. Runs on dedicated hardware, near-zero power,
    // freeing the GPU for vision/segmentation tasks.
    try {
        OrtTensorRTProviderOptionsV2 dla_options{};
        dla_options.device_id = 0;
        dla_options.trt_fp16_enable = 1;
        dla_options.trt_dla_enable = 1;       // Enable DLA core
        dla_options.trt_dla_core = 0;         // Use DLA core 0
        dla_options.trt_engine_cache_enable = 1;
        dla_options.trt_engine_cache_path = "./dla_cache";
        
        session_options_.AppendExecutionProvider_TensorRT_V2(dla_options);
        std::cout << "🧊 OmniEngine: DLA Execution Provider enabled (near-zero power)." << std::endl;
        provider_added = true;
    } catch (...) {
        std::cout << "ℹ OmniEngine: DLA not available, trying TensorRT GPU..." << std::endl;
    }

    // 1. High Priority: NVIDIA TensorRT (GPU)
    if (!provider_added) {
        try {
            OrtTensorRTProviderOptionsV2 trt_options{};
            trt_options.device_id = 0;
            trt_options.trt_fp16_enable = 1; // FP16 enabled by default for Edge AI
            trt_options.trt_engine_cache_enable = 1; // Engine cache for fast robot boot
            trt_options.trt_engine_cache_path = "./trt_cache";
            
            session_options_.AppendExecutionProvider_TensorRT_V2(trt_options);
            std::cout << "🚀 OmniEngine: TensorRT GPU Execution Provider enabled." << std::endl;
            provider_added = true;
        } catch (...) {
            std::cout << "ℹ OmniEngine: TensorRT not available, trying CUDA..." << std::endl;
        }
    }

    // 2. Second Option: NVIDIA CUDA (Standard)
    if (!provider_added) {
        try {
            OrtCUDAProviderOptionsV2 cuda_options{};
            cuda_options.device_id = 0;
            session_options_.AppendExecutionProvider_CUDA_V2(cuda_options);
            std::cout << "⚡ OmniEngine: CUDA Execution Provider enabled." << std::endl;
            provider_added = true;
        } catch (...) {
            std::cout << "ℹ OmniEngine: CUDA not available, falling back to CPU." << std::endl;
        }
    }

    // 3. Final Fallback: CPU
    if (!provider_added) {
        std::cout << "💻 OmniEngine: Operating in CPU mode (Final Fallback)." << std::endl;
    }
}

void OmniEngine::Forward(const std::vector<float>& sensor_tokens, 
                         const std::vector<float>& timestamps,
                         int64_t batch_size, 
                         int64_t num_tokens) {
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Dynamic shape definitions: [Batch, N, 512] and [Batch, N, 1]
    std::vector<int64_t> tokens_shape = {batch_size, num_tokens, 512};
    std::vector<int64_t> times_shape = {batch_size, num_tokens, 1};

    // Create Tensors (Zero-copy if data comes from SharedMemory in production)
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(sensor_tokens.data()), sensor_tokens.size(), tokens_shape.data(), tokens_shape.size()));
    
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(timestamps.data()), timestamps.size(), times_shape.data(), times_shape.size()));

    // Run Inference
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, 
                                       input_names_.data(), 
                                       input_tensors.data(), 
                                       input_tensors.size(), 
                                       output_names_.data(), 
                                       output_names_.size());

    // Results are in output_tensors[0] (Motor) and output_tensors[1] (Safety)
    // In production, the robot control bus injection logic would go here.
}
