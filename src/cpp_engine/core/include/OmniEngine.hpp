#ifndef OMNI_ENGINE_HPP
#define OMNI_ENGINE_HPP

#include <vector>
#include <cstdint>
#include <cmath>
#include <cstddef>

// Compile-time configuration: Override OMNI_MAX_DIM in your build system
// to match your model size and conserve SRAM on smaller chips.
// Default: 256 (Suitable for ESP32 / RP2040 / STM32F4 and above)
#ifndef OMNI_MAX_DIM
#define OMNI_MAX_DIM 256
#endif

// OmniEngine: Platform-agnostic Liquid Neural Network inference engine.
// Compatible with: ESP32, RP2040 (Raspberry Pi Pico), STM32, and any C++11 platform.
// Integrates with a platform-specific OmniHAL.hpp for flash memory access.
class OmniEngine {
public:
    OmniEngine() : is_loaded_(false), weights_ptr_(nullptr) {}

    // Load the network from a .omnibit binary blob.
    // The pointer can come from:
    //   - ESP32: A DROM Flash pointer (Zero-Copy)
    //   - RP2040: A static buffer loaded from LittleFS
    //   - STM32: A linker-embedded C-array in Flash
    // Returns true if the binary is valid and dimensions fit within OMNI_MAX_DIM.
    bool Load(const unsigned char* omnibit_data, size_t length);

    // Runs one inference step. Call this at your control loop frequency (e.g., 100Hz).
    // sensors: raw float array of size GetInputDim()
    // dt:      time delta in seconds since last call
    // abs_time: absolute elapsed time in seconds (for Continuous Temporal Encoding)
    // Returns: action vector of size GetOutputDim()
    std::vector<float> Step(const float* sensors, float dt, float abs_time);

    // Dimension getters (available after a successful Load())
    uint32_t GetInputDim()  const { return input_dim_; }
    uint32_t GetModelDim()  const { return d_model_; }
    uint32_t GetOutputDim() const { return output_dim_; }
    bool     IsLoaded()     const { return is_loaded_; }

private:
    bool is_loaded_;

    // Network dimensions (read from .omnibit header)
    uint32_t input_dim_;
    uint32_t d_model_;
    uint32_t output_dim_;
    uint32_t backbone_units_;
    uint32_t total_weights_;

    // Statically allocated SRAM buffers.
    // Size governed by OMNI_MAX_DIM. No heap allocation in the hot loop.
    float latents_[OMNI_MAX_DIM];
    float state_buffer_[OMNI_MAX_DIM];
    float b_state_[OMNI_MAX_DIM];
    float b_time_[OMNI_MAX_DIM];
    float x_in_[OMNI_MAX_DIM * 2]; // input_dim + d_model

    // Raw pointer into the weight data (Zero-Copy on ESP32/STM32, buffered on RP2040)
    const float* weights_ptr_;

    // Exact pointers to BioLiquidCell weight matrices
    const float* sensory_w_;
    const float* sensory_b_;
    const float* state_w_;
    const float* state_b_;
    const float* time_w_;
    const float* time_b_;
    const float* ff1_w_;
    const float* ff1_b_;
    const float* ff2_w_;
    const float* ff2_b_;
    const float* time_a_w_;
    const float* time_a_b_;
    const float* time_b_w_;
    const float* time_b_b_;
    const float* time_scale_;

    // Neural Processing Stages
    void apply_input_projection(const float* sensors);
    void add_temporal_encoding(float abs_time);
    void apply_bio_liquid_cell(float dt);

    // Math primitives (inlined for performance on constrained hardware)
    float lecun_activation(float x) const { return 1.7159f * std::tanh(0.666f * x); }
    float sigmoid(float x)          const { return 1.0f / (1.0f + std::exp(-x)); }
    void  matmul(const float* W, const float* b, const float* x,
                 float* out, int rows, int cols);
};

#endif // OMNI_ENGINE_HPP
