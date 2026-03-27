#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <iostream>

/**
 * @brief OmniEngine: Universal inference engine with dynamic hardware fallback.
 * Implements the "Ollama Effect" for hardware-agnostic deployment.
 */
class OmniEngine {
public:
    explicit OmniEngine(const std::string& model_path);
    ~OmniEngine() = default;

    /**
     * @brief Execute a dynamic forward pass (inference).
     * @param sensor_tokens Vector of sensor tokens [Batch, N, 512]
     * @param timestamps Vector of timestamps [Batch, N, 1]
     */
    void Forward(const std::vector<float>& sensor_tokens, 
                 const std::vector<float>& timestamps,
                 int64_t batch_size, 
                 int64_t num_tokens);

private:
    void ConfigureExecutionProviders();

    // ONNX Runtime Core Components
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "OmniTrain_Engine"};
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    // Model Metadata
    std::vector<const char*> input_names_ = {"sensor_tokens", "timestamps"};
    std::vector<const char*> output_names_ = {"motor_control", "safety_flag"};
};
