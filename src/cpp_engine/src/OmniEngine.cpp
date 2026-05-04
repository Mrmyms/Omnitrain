#include "OmniEngine.hpp"
#include <cpu_provider_factory.h>
#include <iostream>
#include <numeric>

#ifdef USE_CUDA
#include <cuda_provider_factory.h>
#include <tensorrt_provider_factory.h>
#endif

OmniEngine::OmniEngine(const std::string& model_path) {
    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    ConfigureExecutionProviders();

    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        
        // Auto-detect input/output names from the unified graph
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session_->GetInputCount();
        for (size_t i = 0; i < num_inputs; i++) {
            input_names_.push_back(session_->GetInputName(i, allocator));
        }
        
        size_t num_outputs = session_->GetOutputCount();
        for (size_t i = 0; i < num_outputs; i++) {
            output_names_.push_back(session_->GetOutputName(i, allocator));
        }

        std::cout << "✔ OmniEngine: Loaded unified graph with " << num_outputs << " output heads." << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "✘ Error loading model: " << e.what() << std::endl;
        throw;
    }
}

void OmniEngine::ConfigureExecutionProviders() {
    bool provider_added = false;
#ifdef USE_CUDA
    try {
        OrtTensorRTProviderOptionsV2 trt_options{};
        trt_options.device_id = 0;
        trt_options.trt_fp16_enable = 1;
        session_options_.AppendExecutionProvider_TensorRT_V2(trt_options);
        std::cout << "🚀 OmniEngine: TensorRT GPU Active." << std::endl;
        provider_added = true;
    } catch (...) {}
#endif

    if (!provider_added) {
        std::cout << "💻 OmniEngine: CPU Mode." << std::endl;
    }
}

/**
 * High-Speed Inference Loop
 * Fuses sensor tokens, evolves the liquid state, and produces motor commands.
 */
std::vector<float> OmniEngine::Step(const std::vector<float>& sensor_tokens, 
                                   std::vector<float>& state_buffer,
                                   float dt, 
                                   float& abs_time) {
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Shapes
    int64_t batch_size = 1;
    int64_t num_tokens = sensor_tokens.size() / 512;
    int64_t n_latents = state_buffer.size() / 256;

    std::vector<int64_t> tokens_shape = {batch_size, num_tokens, 512};
    std::vector<int64_t> dt_shape = {batch_size, 1};
    std::vector<int64_t> state_shape = {batch_size, n_latents, 256};
    std::vector<int64_t> time_shape = {batch_size, 1};

    std::vector<float> dt_vec = {dt};
    std::vector<float> time_vec = {abs_time};

    // Create Input Tensors
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(sensor_tokens.data()), sensor_tokens.size(), tokens_shape.data(), tokens_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, dt_vec.data(), dt_vec.size(), dt_shape.data(), dt_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, state_buffer.data(), state_buffer.size(), state_shape.data(), state_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, time_vec.data(), time_vec.size(), time_shape.data(), time_shape.size()));

    // Run
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, 
                                       input_names_.data(), 
                                       input_tensors.data(), 
                                       input_tensors.size(), 
                                       output_names_.data(), 
                                       output_names_.size());

    // 1. Update State (Recurrent connection)
    float* next_state_ptr = output_tensors[0].GetTensorMutableData<float>();
    std::copy(next_state_ptr, next_state_ptr + state_buffer.size(), state_buffer.begin());

    // 2. Update Absolute Time
    abs_time += dt;

    // 3. Extract Main Action (Head 0 after state)
    float* action_ptr = output_tensors[1].GetTensorMutableData<float>();
    auto type_info = output_tensors[1].GetTensorTypeAndShapeInfo();
    size_t action_dim = type_info.GetElementCount();

    std::vector<float> action(action_ptr, action_ptr + action_dim);
    return action;
}
