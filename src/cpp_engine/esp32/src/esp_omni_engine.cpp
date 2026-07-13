#include "esp_omni_engine.hpp"
#include <cstring>
#include <iostream>
#include <algorithm>

bool ESPOmniEngine::Load(const unsigned char* omnibit_data, size_t length) {
    if (length < 32) {
        return false; 
    }

    // 1. Verify Magic Bytes 'OMNI\x04' (V4)
    if (omnibit_data[0] != 'O' || omnibit_data[1] != 'M' || 
        omnibit_data[2] != 'N' || omnibit_data[3] != 'I' || 
        omnibit_data[4] != 0x04) {
        return false;
    }

    architecture_type_ = omnibit_data[5];

    // 2. Read metadata (24 bytes total: dims[5] + num_tensors)
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

    // 3. Point to the weights array in Flash (Offset = 32 + TOC size)
    uint32_t weights_offset = 32 + (num_tensors * 4);
    weights_ptr_ = reinterpret_cast<const float*>(omnibit_data + weights_offset);
    
    // 4. Strict offset mapping for sequential matrix access
    uint32_t offset = 0;
    
    if (architecture_type_ == 0) { // CfC
        // Input Projector
        offset += (input_dim_ * d_model_) + d_model_;
        // CTE
        offset += (d_model_ / 2) * 2 + d_model_;
        // BioLiquidCell
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
    } else if (architecture_type_ == 1) { // GRU
        gru_w_ih_ = weights_ptr_ + offset; offset += 3 * d_model_ * d_model_;
        gru_w_hh_ = weights_ptr_ + offset; offset += 3 * d_model_ * d_model_;
        gru_b_ih_ = weights_ptr_ + offset; offset += 3 * d_model_;
        gru_b_hh_ = weights_ptr_ + offset; offset += 3 * d_model_;
    } else if (architecture_type_ == 2) { // Transformer
        trf_input_proj_w_ = weights_ptr_ + offset; offset += d_model_ * (2 * d_model_);
        trf_input_proj_b_ = weights_ptr_ + offset; offset += d_model_;
        trf_wq_w_ = weights_ptr_ + offset; offset += d_model_ * d_model_;
        trf_wq_b_ = weights_ptr_ + offset; offset += d_model_;
        trf_wk_w_ = weights_ptr_ + offset; offset += d_model_ * d_model_;
        trf_wk_b_ = weights_ptr_ + offset; offset += d_model_;
        trf_wv_w_ = weights_ptr_ + offset; offset += d_model_ * d_model_;
        trf_wv_b_ = weights_ptr_ + offset; offset += d_model_;
        trf_wo_w_ = weights_ptr_ + offset; offset += d_model_ * d_model_;
        trf_wo_b_ = weights_ptr_ + offset; offset += d_model_;
        trf_ffn1_w_ = weights_ptr_ + offset; offset += backbone_units_ * d_model_;
        trf_ffn1_b_ = weights_ptr_ + offset; offset += backbone_units_;
        trf_ffn2_w_ = weights_ptr_ + offset; offset += d_model_ * backbone_units_;
        trf_ffn2_b_ = weights_ptr_ + offset; offset += d_model_;
        trf_norm1_w_ = weights_ptr_ + offset; offset += d_model_;
        trf_norm1_b_ = weights_ptr_ + offset; offset += d_model_;
        trf_norm2_w_ = weights_ptr_ + offset; offset += d_model_;
        trf_norm2_b_ = weights_ptr_ + offset; offset += d_model_;
    } else if (architecture_type_ == 3) { // Full CfC (No CTE/Proj)
        cfc_bb_w_ = weights_ptr_ + offset; offset += backbone_units_ * (input_dim_ + d_model_);
        cfc_bb_b_ = weights_ptr_ + offset; offset += backbone_units_;
        cfc_f_w_ = weights_ptr_ + offset; offset += d_model_ * backbone_units_;
        cfc_f_b_ = weights_ptr_ + offset; offset += d_model_;
        cfc_g_w_ = weights_ptr_ + offset; offset += d_model_ * backbone_units_;
        cfc_g_b_ = weights_ptr_ + offset; offset += d_model_;
        cfc_h_w_ = weights_ptr_ + offset; offset += d_model_ * backbone_units_;
        cfc_h_b_ = weights_ptr_ + offset; offset += d_model_;
        cfc_fc_w_ = weights_ptr_ + offset; offset += output_dim_ * d_model_;
        cfc_fc_b_ = weights_ptr_ + offset; offset += output_dim_;
    }
    
    is_loaded = true;
    return true;
}

void ESPOmniEngine::matmul(const float* W, const float* b, const float* x, float* out, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float sum = (b != nullptr) ? b[i] : 0.0f;
        for (int j = 0; j < cols; ++j) {
            sum += x[j] * W[i * cols + j]; // Linear Multiplication O(R*C)
        }
        out[i] = sum;
    }
}

std::vector<float> ESPOmniEngine::Step(const float* sensors, float dt, float abs_time) {
    if (!is_loaded) {
        return std::vector<float>(output_dim_, 0.0f);
    }
    
    std::memset(latents_, 0, d_model_ * sizeof(float));
    
    if (architecture_type_ != 3) {
        // Phase 1: Adaptive Projection
        apply_input_projection(sensors);
        
        // Phase 2: Continuous Temporal Encoding (CTE)
        add_temporal_encoding(abs_time);
    }
    
    // Phase 3: Architecture Inference
    if (architecture_type_ == 0) {
        apply_bio_liquid_cell(dt);
    } else if (architecture_type_ == 1) {
        apply_gru_cell();
    } else if (architecture_type_ == 2) {
        apply_transformer_layer();
    } else if (architecture_type_ == 3) {
        apply_cfc_full_cell(sensors, dt);
    }
    
    // Phase 4: Action Generation
    std::vector<float> action(output_dim_, 0.0f);
    if (architecture_type_ == 3) {
        matmul(cfc_fc_w_, cfc_fc_b_, state_buffer_, action.data(), output_dim_, d_model_);
    } else {
        for (uint32_t i = 0; i < output_dim_ && i < d_model_; ++i) {
            action[i] = state_buffer_[i];
        }
    }
    
    return action;
}

void ESPOmniEngine::apply_input_projection(const float* sensors) {
    const float* w = weights_ptr_; 
    const float* b = w + (input_dim_ * d_model_);
    matmul(w, b, sensors, latents_, d_model_, input_dim_);
}

void ESPOmniEngine::add_temporal_encoding(float abs_time) {
    uint32_t proj_offset = (input_dim_ * d_model_) + d_model_;
    const float* inv_freq = weights_ptr_ + proj_offset;
    const float* amplitude = inv_freq + (d_model_ / 2);
    const float* phase = amplitude + (d_model_); 
    
    for (uint32_t i = 0; i < d_model_ / 2; ++i) {
        float arg = abs_time * inv_freq[i] + phase[i];
        
        // PyTorch uses torch.cat([sin, cos], dim=-1), 
        // meaning all sines are concatenated first, followed by all cosines.
        latents_[i] += std::sin(arg) * amplitude[i];
        latents_[i + (d_model_ / 2)] += std::cos(arg) * amplitude[i + (d_model_ / 2)];
    }
}

void ESPOmniEngine::apply_bio_liquid_cell(float dt) {
    // 1. Affine Sensory Mapping (x_mapped = x * sensory_w + sensory_b)
    for (uint32_t i = 0; i < input_dim_; ++i) {
        x_in_[i] = latents_[i] * sensory_w_[i] + sensory_b_[i];
    }
    
    // 2. Concatenate previous state (x_in = cat([x_mapped, h_in]))
    for (uint32_t i = 0; i < d_model_; ++i) {
        x_in_[input_dim_ + i] = state_buffer_[i];
    }
    
    // 3. Backbone State (b_state = LeCun(Linear(x_in)))
    uint32_t in_size = input_dim_ + d_model_;
    matmul(state_w_, state_b_, x_in_, b_state_, backbone_units_, in_size);
    for(uint32_t i=0; i < backbone_units_; ++i) b_state_[i] = lecun_activation(b_state_[i]);
    
    // 4. Backbone Time (b_time = LeCun(Linear(x_in)))
    uint32_t half_units = backbone_units_ / 2;
    matmul(time_w_, time_b_, x_in_, b_time_, half_units, in_size);
    for(uint32_t i=0; i < half_units; ++i) b_time_[i] = lecun_activation(b_time_[i]);
    
    // 5. Compute Targets (FF1, FF2)
    float ff1[256];
    float ff2[256];
    matmul(ff1_w_, ff1_b_, b_state_, ff1, d_model_, backbone_units_);
    matmul(ff2_w_, ff2_b_, b_state_, ff2, d_model_, backbone_units_);
    for(uint32_t i=0; i < d_model_; ++i) {
        ff1[i] = std::tanh(ff1[i]);
        ff2[i] = std::tanh(ff2[i]);
    }
    
    // 6. Compute Time Interpolation Gate
    float time_a_out[256];
    float time_b_out[256];
    matmul(time_a_w_, time_a_b_, b_time_, time_a_out, d_model_, half_units);
    matmul(time_b_w_, time_b_b_, b_time_, time_b_out, d_model_, half_units);
    
    // Clamp dt for safety and apply time_scale
    float ts = std::max(dt, 0.0f);
    
    // 7. Update State (CfC Default Mode)
    for (uint32_t i = 0; i < d_model_; ++i) {
        float t_scaled = ts * std::abs(time_scale_[i]);
        float t_interp = sigmoid(time_a_out[i] * t_scaled + time_b_out[i]);
        state_buffer_[i] = ff1[i] * (1.0f - t_interp) + t_interp * ff2[i];
    }
}

void ESPOmniEngine::apply_gru_cell() {
    float w_ih_out[256 * 3], w_hh_out[256 * 3];
    
    matmul(gru_w_ih_, gru_b_ih_, latents_, w_ih_out, 3 * d_model_, d_model_);
    matmul(gru_w_hh_, gru_b_hh_, state_buffer_, w_hh_out, 3 * d_model_, d_model_);
    
    for (uint32_t i = 0; i < d_model_; ++i) {
        float r_val = sigmoid(w_ih_out[i] + w_hh_out[i]);
        float z_val = sigmoid(w_ih_out[d_model_ + i] + w_hh_out[d_model_ + i]);
        float n_val = std::tanh(w_ih_out[2 * d_model_ + i] + r_val * w_hh_out[2 * d_model_ + i]);
        
        state_buffer_[i] = (1.0f - z_val) * n_val + z_val * state_buffer_[i];
    }
}

void ESPOmniEngine::apply_transformer_layer() {
    for (uint32_t i = 0; i < d_model_; ++i) {
        x_in_[i] = latents_[i];
        x_in_[d_model_ + i] = state_buffer_[i];
    }
    
    float x_proj[256];
    matmul(trf_input_proj_w_, trf_input_proj_b_, x_in_, x_proj, d_model_, 2 * d_model_);
    
    float q[256], k[256], v[256];
    matmul(trf_wq_w_, trf_wq_b_, x_proj, q, d_model_, d_model_);
    matmul(trf_wk_w_, trf_wk_b_, x_proj, k, d_model_, d_model_);
    matmul(trf_wv_w_, trf_wv_b_, x_proj, v, d_model_, d_model_);
    
    // Point-wise self-attention (sequence length 1)
    // context = v
    
    float wo_out[256];
    matmul(trf_wo_w_, trf_wo_b_, v, wo_out, d_model_, d_model_);
    
    float out1[256];
    float mean1 = 0, var1 = 0;
    for(uint32_t i=0; i<d_model_; ++i) {
        out1[i] = x_proj[i] + wo_out[i];
        mean1 += out1[i];
    }
    mean1 /= d_model_;
    for(uint32_t i=0; i<d_model_; ++i) var1 += (out1[i] - mean1) * (out1[i] - mean1);
    var1 /= d_model_;
    float std1 = std::sqrt(var1 + 1e-5f);
    for(uint32_t i=0; i<d_model_; ++i) {
        out1[i] = ((out1[i] - mean1) / std1) * trf_norm1_w_[i] + trf_norm1_b_[i];
    }
    
    float ffn1_out[256];
    matmul(trf_ffn1_w_, trf_ffn1_b_, out1, ffn1_out, backbone_units_, d_model_);
    for(uint32_t i=0; i<backbone_units_; ++i) ffn1_out[i] = std::max(0.0f, ffn1_out[i]); // ReLU
    
    float ffn2_out[256];
    matmul(trf_ffn2_w_, trf_ffn2_b_, ffn1_out, ffn2_out, d_model_, backbone_units_);
    
    float out2[256];
    float mean2 = 0, var2 = 0;
    for(uint32_t i=0; i<d_model_; ++i) {
        out2[i] = out1[i] + ffn2_out[i];
        mean2 += out2[i];
    }
    mean2 /= d_model_;
    for(uint32_t i=0; i<d_model_; ++i) var2 += (out2[i] - mean2) * (out2[i] - mean2);
    var2 /= d_model_;
    float std2 = std::sqrt(var2 + 1e-5f);
    for(uint32_t i=0; i<d_model_; ++i) {
        state_buffer_[i] = ((out2[i] - mean2) / std2) * trf_norm2_w_[i] + trf_norm2_b_[i];
    }
}

void ESPOmniEngine::apply_cfc_full_cell(const float* sensors, float dt) {
    // 1. Prepare input: cat(sensors, state_buffer_)
    for (uint32_t i = 0; i < input_dim_; ++i) {
        x_in_[i] = sensors[i];
    }
    for (uint32_t i = 0; i < d_model_; ++i) {
        x_in_[input_dim_ + i] = state_buffer_[i];
    }
    
    // 2. Backbone computation (bb = Tanh(Linear(x_in)))
    matmul(cfc_bb_w_, cfc_bb_b_, x_in_, b_state_, backbone_units_, input_dim_ + d_model_);
    for(uint32_t i=0; i < backbone_units_; ++i) b_state_[i] = std::tanh(b_state_[i]);
    
    // 3. Compute heads
    float f_out[256], g_out[256], h_out[256];
    matmul(cfc_f_w_, cfc_f_b_, b_state_, f_out, d_model_, backbone_units_);
    matmul(cfc_g_w_, cfc_g_b_, b_state_, g_out, d_model_, backbone_units_);
    matmul(cfc_h_w_, cfc_h_b_, b_state_, h_out, d_model_, backbone_units_);
    
    // 4. State update: t_gate = sigmoid(-f * dt)
    float ts = std::max(dt, 0.0f);
    for (uint32_t i = 0; i < d_model_; ++i) {
        float t_gate = sigmoid(-f_out[i] * ts);
        float g_cand = std::tanh(g_out[i]);
        float h_cand = std::tanh(h_out[i]);
        state_buffer_[i] = t_gate * g_cand + (1.0f - t_gate) * h_cand;
    }
}
