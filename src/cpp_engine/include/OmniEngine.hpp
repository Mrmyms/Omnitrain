#pragma once
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

/**
 * OmniEngine: High-Performance Liquid Neural Inference.
 */
class OmniEngine {
public:
    explicit OmniEngine(const std::string& model_path);

    /**
     * Executes a single step of the Liquid Brain.
     * @param sensor_tokens Input tokens projected from sensors (B, N, D)
     * @param state_buffer  Persistent state buffer (updated in-place)
     * @param dt            Time delta since last step
     * @param abs_time      Accumulated absolute time (updated in-place)
     * @return              The primary action vector (e.g., motor commands)
     */
    std::vector<float> Step(const std::vector<float>& sensor_tokens, 
                           std::vector<float>& state_buffer,
                           float dt, 
                           float& abs_time);

private:
    void ConfigureExecutionProviders();

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "OmniTrain"};
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};
