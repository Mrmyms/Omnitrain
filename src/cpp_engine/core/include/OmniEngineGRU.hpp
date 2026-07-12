#ifndef OMNI_ENGINE_GRU_HPP
#define OMNI_ENGINE_GRU_HPP

#include <vector>
#include <cstdint>
#include <cmath>
#include <cstddef>

#ifndef OMNI_MAX_DIM
#define OMNI_MAX_DIM 256
#endif

// GRU Baseline Engine for XIP comparison
class OmniEngineGRU {
public:
    OmniEngineGRU() : is_loaded_(false), weights_ptr_(nullptr) {}

    bool Load(const unsigned char* omnibit_data, size_t length);
    std::vector<float> Step(const float* sensors);

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

    float h_state_[OMNI_MAX_DIM];
    float x_in_[OMNI_MAX_DIM * 2]; // input_dim + hidden_dim

    const float* weights_ptr_;

    // PyTorch GRU weights are concatenated: W_ir, W_iz, W_in
    // and W_hr, W_hz, W_hn
    const float* w_ih_;
    const float* w_hh_;
    const float* b_ih_;
    const float* b_hh_;
    const float* fc_w_;
    const float* fc_b_;

    float sigmoid(float x) const { return 1.0f / (1.0f + std::exp(-x)); }
    float tanh_fast(float x) const { return std::tanh(x); }
    void  matmul(const float* W, const float* b, const float* x, float* out, int rows, int cols);
};

#endif // OMNI_ENGINE_GRU_HPP
