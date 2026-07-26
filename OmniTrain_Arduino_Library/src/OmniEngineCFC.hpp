#ifndef OMNI_ENGINE_CFC_HPP
#define OMNI_ENGINE_CFC_HPP

#include <cstdint>
#include <cmath>
#include <cstddef>

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

class OmniEngineCFC {
public:
    OmniEngineCFC() : is_loaded(false), weights_ptr_(nullptr), arch_flag_(0) {}

    // Loads the weights mapped in Flash memory without copying them to SRAM (Zero-Copy)
    // Returns true if the load was successful.
    bool Load(const unsigned char* omnibit_data, size_t length);

    // Inference loop. Writes the processed action to out_action.
    void Step(const float* sensors, float dt, float abs_time, float* out_action);

    // Getters
    uint32_t GetInputDim() const { return input_dim_; }
    uint32_t GetModelDim() const { return d_model_; }
    uint32_t GetOutputDim() const { return output_dim_; }

private:
    bool is_loaded;
    uint8_t arch_flag_;
    
    // Dimensions
    uint32_t input_dim_;
    uint32_t d_model_;
    uint32_t output_dim_;
    uint32_t backbone_units_;
    uint32_t total_weights_;
    
    // Statically allocated buffers (prevents OOM fragmentation)
    // ALIGN16 guarantees 128-bit bus alignment for ESP32-S3 Data Cache
    ALIGN16 float latents_[256];
    ALIGN16 float state_buffer_[256]; 
    ALIGN16 float b_state_[256];
    ALIGN16 float b_time_[256];
    ALIGN16 float x_in_[256 + 256]; // input_dim + d_model
    
    // Pointers to Flash (DROM)
    const float* weights_ptr_;

    // Dense Matrix Pointers
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

    // Sparse CSR Pointers (For Arch Flag 4)
    const float* bb_val_;
    const uint32_t* bb_col_;
    const uint32_t* bb_row_;
    const float* bb_b_;
    const float* f_w_;
    const float* f_b_;
    const float* g_w_;
    const float* g_b_;
    const float* h_w_;
    const float* h_b_;
    const float* fc_w_;
    const float* fc_b_;

    // Neural Processing Pipeline
    void apply_input_projection(const float* sensors);
    void add_temporal_encoding(float abs_time);
    void apply_bio_liquid_cell(float dt);
    void apply_sparse_cfc(const float* sensors, float dt); // New path for SparseCfC
    
    // Math Utilities
    float lecun_activation(float x) const { return 1.7159f * std::tanh(0.666f * x); }
    float sigmoid(float x) const { return 1.0f / (1.0f + std::exp(-x)); }
    
    // Highly Optimized Math Kernels
    void matmul(const float* __restrict W, const float* __restrict b, const float* __restrict x, float* __restrict out, int rows, int cols);
    void matmul_csr(const float* __restrict val, const uint32_t* __restrict col, const uint32_t* __restrict row_ptr, const float* __restrict b, const float* __restrict x, float* __restrict out, int rows);
};

#endif // OMNI_ENGINE_CFC_HPP
