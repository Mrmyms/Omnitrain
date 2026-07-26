#ifndef OMNI_ENGINE_GRU_HPP
#define OMNI_ENGINE_GRU_HPP

#include <cstdint>
#include <cmath>
#include <cstddef>

#ifndef OMNI_MAX_DIM
#define OMNI_MAX_DIM 256
#endif

    // Fallback for Non-ESP32 environments
#ifdef ESP32
#include <esp_attr.h>
#else
#define IRAM_ATTR
#define __restrict
#endif

#ifndef ALIGN16
#define ALIGN16 __attribute__((aligned(16)))
#endif

// GRU Baseline Engine for XIP comparison
class OmniEngineGRU {
public:
    OmniEngineGRU() : is_loaded_(false), weights_ptr_(nullptr) {}

    bool Load(const unsigned char* omnibit_data, size_t length);
    void Step(const float* sensors, float* out_action);

    uint32_t GetInputDim()  const { return input_dim_; }
    uint32_t GetModelDim()  const { return d_model_; }
    uint32_t GetOutputDim() const { return output_dim_; }
    bool     IsLoaded()     const { return is_loaded_; }

private:
    bool is_loaded_;

    uint32_t input_dim_;
    uint32_t d_model_;
    uint32_t output_dim_;
    uint32_t total_weights_;

    ALIGN16 float h_state_[OMNI_MAX_DIM];
    ALIGN16 float x_in_[OMNI_MAX_DIM * 2]; // input_dim + hidden_dim
    ALIGN16 float gates_ih_[3 * OMNI_MAX_DIM];
    ALIGN16 float gates_hh_[3 * OMNI_MAX_DIM];

    const float* weights_ptr_;

    // PyTorch GRU weights are concatenated: W_ir, W_iz, W_in
    // and W_hr, W_hz, W_hn
    const float* w_ih_;
    const float* w_hh_;
    const float* b_ih_;
    const float* b_hh_;
    const float* fc_w_;
    const float* fc_b_;

    // Fast sigmoid: polynomial approximation (~60x faster than std::exp on Xtensa)
    float sigmoid(float x) const {
        if (x > 6.0f) return 1.0f;
        if (x < -6.0f) return 0.0f;
        return 0.5f + x * (0.25f - 0.025f * std::fabs(x));
    }
    float tanh_fast(float x) const { return std::tanh(x); }
    void  matmul(const float* W, const float* b, const float* x, float* out, int rows, int cols);
};

#endif // OMNI_ENGINE_GRU_HPP
