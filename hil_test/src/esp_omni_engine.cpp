#include "esp_omni_engine.hpp"
#include <cstring>
#include <iostream>
#include <algorithm>

// Force maximum GCC optimizations for the whole file
#pragma GCC optimize ("O3")

bool ESPOmniEngine::Load(const unsigned char* omnibit_data, size_t length) {
    if (length < 28) return false; 
    
    // 1. Verify Magic Bytes 'OMNI'
    if (omnibit_data[0] != 'O' || omnibit_data[1] != 'M' || 
        omnibit_data[2] != 'N' || omnibit_data[3] != 'I') {
        return false;
    }
    
    arch_flag_ = omnibit_data[5]; // Extracts the architecture version (0=CfC, 1=GRU, 4=SparseCfC)

    // 2. Read metadata (24 bytes)
    uint32_t dims[6];
    std::memcpy(dims, omnibit_data + 8, sizeof(dims));
    
    input_dim_ = dims[0];
    d_model_ = dims[1];
    output_dim_ = dims[2];
    backbone_units_ = dims[3];
    total_weights_ = dims[4];
    uint32_t num_tensors = dims[5];

    if (input_dim_ + d_model_ > 512 || d_model_ > 256 || backbone_units_ > 256) {
        return false; // Exceeds static pre-allocation boundaries
    }

    // Initialize hidden state (SRAM)
    std::memset(state_buffer_, 0, sizeof(state_buffer_));
    std::memset(latents_, 0, sizeof(latents_));

    // 3. TOC and Weights parsing
    const uint32_t* toc = reinterpret_cast<const uint32_t*>(omnibit_data + 32);
    weights_ptr_ = reinterpret_cast<const float*>(omnibit_data + 32 + num_tensors * 4);
    
    uint32_t offset = 0;

    if (arch_flag_ == 4) {
        // --- SparseCfC (Arch Flag 4) Parsing ---
        if (num_tensors < 12) return false;
        
        bb_val_ = weights_ptr_ + offset; offset += toc[0];
        // The ESP32Exporter packs integers as floats to keep the binary array uniform. We reinterpret_cast them back.
        bb_col_ = reinterpret_cast<const uint32_t*>(weights_ptr_ + offset); offset += toc[1];
        bb_row_ = reinterpret_cast<const uint32_t*>(weights_ptr_ + offset); offset += toc[2];
        
        bb_b_ = weights_ptr_ + offset; offset += toc[3];
        f_w_ = weights_ptr_ + offset; offset += toc[4];
        f_b_ = weights_ptr_ + offset; offset += toc[5];
        g_w_ = weights_ptr_ + offset; offset += toc[6];
        g_b_ = weights_ptr_ + offset; offset += toc[7];
        h_w_ = weights_ptr_ + offset; offset += toc[8];
        h_b_ = weights_ptr_ + offset; offset += toc[9];
        
        fc_w_ = weights_ptr_ + offset; offset += toc[10];
        fc_b_ = weights_ptr_ + offset; offset += toc[11];
    } else {
        // --- Legacy Dense Parsing ---
        uint32_t proj_offset = (input_dim_ * d_model_) + d_model_;
        offset += proj_offset;
        offset += (d_model_ / 2) * 2 + d_model_; // CTE
        
        sensory_w_ = weights_ptr_ + offset; offset += input_dim_;
        sensory_b_ = weights_ptr_ + offset; offset += input_dim_;
        
        uint32_t in_size = input_dim_ + d_model_;
        state_w_ = weights_ptr_ + offset; offset += backbone_units_ * in_size;
        state_b_ = weights_ptr_ + offset; offset += backbone_units_;
        
        uint32_t half_units = backbone_units_ / 2;
        time_w_ = weights_ptr_ + offset; offset += half_units * in_size;
        time_b_ = weights_ptr_ + offset; offset += half_units;
        
        ff1_w_ = weights_ptr_ + offset; offset += d_model_ * backbone_units_;
        ff1_b_ = weights_ptr_ + offset; offset += d_model_;
        
        ff2_w_ = weights_ptr_ + offset; offset += d_model_ * backbone_units_;
        ff2_b_ = weights_ptr_ + offset; offset += d_model_;
        
        time_a_w_ = weights_ptr_ + offset; offset += d_model_ * half_units;
        time_a_b_ = weights_ptr_ + offset; offset += d_model_;
        
        time_b_w_ = weights_ptr_ + offset; offset += d_model_ * half_units;
        time_b_b_ = weights_ptr_ + offset; offset += d_model_;
        
        time_scale_ = weights_ptr_ + offset;
    }
    
    is_loaded = true;
    return true;
}

// Dense MatMul (IRAM Cached)
void IRAM_ATTR ESPOmniEngine::matmul(const float* __restrict W, const float* __restrict b, const float* __restrict x, float* __restrict out, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float sum = (b != nullptr) ? b[i] : 0.0f;
        const float* W_row = W + (i * cols);
        #pragma GCC unroll 4
        for (int j = 0; j < cols; ++j) {
            sum += x[j] * W_row[j]; 
        }
        out[i] = sum;
    }
}

// Sparse CSR MatMul O(N_nonzero) (IRAM Cached)
// Extreme optimization: Skips all zero-weight synapses automatically!
void IRAM_ATTR ESPOmniEngine::matmul_csr(const float* __restrict val, const uint32_t* __restrict col, const uint32_t* __restrict row_ptr, const float* __restrict b, const float* __restrict x, float* __restrict out, int rows) {
    for (int i = 0; i < rows; ++i) {
        float sum = (b != nullptr) ? b[i] : 0.0f;
        uint32_t row_start = row_ptr[i];
        uint32_t row_end = row_ptr[i+1];
        
        #pragma GCC unroll 4
        for (uint32_t j = row_start; j < row_end; ++j) {
            sum += val[j] * x[col[j]];
        }
        out[i] = sum;
    }
}

// Core Step Function (IRAM Cached)
std::vector<float> IRAM_ATTR ESPOmniEngine::Step(const float* sensors, float dt, float abs_time) {
    if (!is_loaded) {
        return std::vector<float>(output_dim_, 0.0f);
    }
    
    if (arch_flag_ == 4) {
        // --- SparseCfC Path ---
        apply_sparse_cfc(sensors, dt);
        
        // Final Output Generation using FC layer
        std::vector<float> action(output_dim_, 0.0f);
        matmul(fc_w_, fc_b_, state_buffer_, action.data(), output_dim_, d_model_);
        return action;
    } else {
        // --- Legacy Dense Path ---
        std::memset(latents_, 0, d_model_ * sizeof(float));
        apply_input_projection(sensors);
        add_temporal_encoding(abs_time);
        apply_bio_liquid_cell(dt);
        
        std::vector<float> action(output_dim_, 0.0f);
        for (uint32_t i = 0; i < output_dim_ && i < d_model_; ++i) {
            action[i] = state_buffer_[i];
        }
        return action;
    }
}

void IRAM_ATTR ESPOmniEngine::apply_sparse_cfc(const float* sensors, float dt) {
    // 1. Concatenate Sensors + Hidden State -> x_in
    for (uint32_t i = 0; i < input_dim_; ++i) {
        x_in_[i] = sensors[i];
    }
    for (uint32_t i = 0; i < d_model_; ++i) {
        x_in_[input_dim_ + i] = state_buffer_[i];
    }
    
    // 2. CSR Sparse Neural Inference O(N_nonzero)
    matmul_csr(bb_val_, bb_col_, bb_row_, bb_b_, x_in_, b_state_, d_model_);
    
    // Clamp dt to ensure stable temporal dynamics
    float ts = std::max(dt, 0.0f);
    
    // 3. Apply Continuous-Time ODE Gates Element-wise
    for(uint32_t i = 0; i < d_model_; ++i) {
        float bb = std::tanh(b_state_[i]);
        
        float f_val = bb * f_w_[i] + f_b_[i];
        float g_val = bb * g_w_[i] + g_b_[i];
        float h_val = bb * h_w_[i] + h_b_[i];
        
        float t_gate = sigmoid(-f_val * ts);
        
        // Close the ODE loop natively inside SRAM
        state_buffer_[i] = t_gate * std::tanh(g_val) + (1.0f - t_gate) * std::tanh(h_val);
    }
}

void IRAM_ATTR ESPOmniEngine::apply_input_projection(const float* sensors) {
    const float* w = weights_ptr_; 
    const float* b = w + (input_dim_ * d_model_);
    matmul(w, b, sensors, latents_, d_model_, input_dim_);
}

void IRAM_ATTR ESPOmniEngine::add_temporal_encoding(float abs_time) {
    uint32_t proj_offset = (input_dim_ * d_model_) + d_model_;
    const float* inv_freq = weights_ptr_ + proj_offset;
    const float* amplitude = inv_freq + (d_model_ / 2);
    const float* phase = amplitude + (d_model_); 
    
    for (uint32_t i = 0; i < d_model_ / 2; ++i) {
        float arg = abs_time * inv_freq[i] + phase[i];
        latents_[i] += std::sin(arg) * amplitude[i];
        latents_[i + (d_model_ / 2)] += std::cos(arg) * amplitude[i + (d_model_ / 2)];
    }
}

void IRAM_ATTR ESPOmniEngine::apply_bio_liquid_cell(float dt) {
    // Legacy Dense Math Logic
    for (uint32_t i = 0; i < input_dim_; ++i) x_in_[i] = latents_[i] * sensory_w_[i] + sensory_b_[i];
    for (uint32_t i = 0; i < d_model_; ++i) x_in_[input_dim_ + i] = state_buffer_[i];
    
    uint32_t in_size = input_dim_ + d_model_;
    matmul(state_w_, state_b_, x_in_, b_state_, backbone_units_, in_size);
    for(uint32_t i=0; i < backbone_units_; ++i) b_state_[i] = lecun_activation(b_state_[i]);
    
    uint32_t half_units = backbone_units_ / 2;
    matmul(time_w_, time_b_, x_in_, b_time_, half_units, in_size);
    for(uint32_t i=0; i < half_units; ++i) b_time_[i] = lecun_activation(b_time_[i]);
    
    float ff1[256], ff2[256];
    matmul(ff1_w_, ff1_b_, b_state_, ff1, d_model_, backbone_units_);
    matmul(ff2_w_, ff2_b_, b_state_, ff2, d_model_, backbone_units_);
    for(uint32_t i=0; i < d_model_; ++i) { ff1[i] = std::tanh(ff1[i]); ff2[i] = std::tanh(ff2[i]); }
    
    float time_a_out[256], time_b_out[256];
    matmul(time_a_w_, time_a_b_, b_time_, time_a_out, d_model_, half_units);
    matmul(time_b_w_, time_b_b_, b_time_, time_b_out, d_model_, half_units);
    
    float ts = std::max(dt, 0.0f);
    for (uint32_t i = 0; i < d_model_; ++i) {
        float t_scaled = ts * std::abs(time_scale_[i]);
        float t_interp = sigmoid(time_a_out[i] * t_scaled + time_b_out[i]);
        state_buffer_[i] = ff1[i] * (1.0f - t_interp) + t_interp * ff2[i];
    }
}
