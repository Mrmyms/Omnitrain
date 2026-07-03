#include "OmniEngine.hpp"
#include <cstring>
#include <algorithm>

bool OmniEngine::Load(const unsigned char* omnibit_data, size_t length) {
    if (length < 28) return false;

    // Verify Magic Bytes 'OMNI\x02' (V2 Structured Format)
    if (omnibit_data[0] != 'O' || omnibit_data[1] != 'M' ||
        omnibit_data[2] != 'N' || omnibit_data[3] != 'I' ||
        omnibit_data[4] != 0x02) {
        return false;
    }

    // Read dimension header (20 bytes at offset 8)
    uint32_t dims[5];
    std::memcpy(dims, omnibit_data + 8, sizeof(dims));

    input_dim_      = dims[0];
    d_model_        = dims[1];
    output_dim_     = dims[2];
    backbone_units_ = dims[3];
    total_weights_  = dims[4];

    // Validate dimensions fit within static buffer allocation
    if (d_model_ > OMNI_MAX_DIM || backbone_units_ > OMNI_MAX_DIM ||
        (input_dim_ + d_model_) > OMNI_MAX_DIM * 2) {
        return false; // Model too large for this chip's OMNI_MAX_DIM setting
    }

    // Initialize hidden state to zero
    std::memset(state_buffer_, 0, sizeof(float) * d_model_);
    std::memset(latents_,      0, sizeof(float) * d_model_);

    // Point directly into the binary data at offset 28 (Zero-Copy on DROM/Flash)
    weights_ptr_ = reinterpret_cast<const float*>(omnibit_data + 28);

    // Map each weight matrix pointer via sequential offset arithmetic
    uint32_t offset = 0;

    // Input Projector: W[d_model x input_dim] + b[d_model]
    offset += (input_dim_ * d_model_) + d_model_;

    // Continuous Temporal Encoding (CTE):
    // Exporter saves: inv_freq[d/2] + amplitude[d] + phase[d/2] = 2*d total
    offset += (d_model_ / 2);  // inv_freq
    offset += d_model_;        // amplitude
    offset += (d_model_ / 2);  // phase

    // BioLiquidCell
    uint32_t in_size   = d_model_ + d_model_; // BioLiquidCell input is mapped latent (d_model) + hidden (d_model)
    uint32_t half_units = backbone_units_ / 2;

    sensory_w_ = weights_ptr_ + offset; offset += d_model_;
    sensory_b_ = weights_ptr_ + offset; offset += d_model_;

    state_w_   = weights_ptr_ + offset; offset += backbone_units_ * in_size;
    state_b_   = weights_ptr_ + offset; offset += backbone_units_;

    time_w_    = weights_ptr_ + offset; offset += half_units * in_size;
    time_b_    = weights_ptr_ + offset; offset += half_units;

    ff1_w_     = weights_ptr_ + offset; offset += d_model_ * backbone_units_;
    ff1_b_     = weights_ptr_ + offset; offset += d_model_;

    ff2_w_     = weights_ptr_ + offset; offset += d_model_ * backbone_units_;
    ff2_b_     = weights_ptr_ + offset; offset += d_model_;

    time_a_w_  = weights_ptr_ + offset; offset += d_model_ * half_units;
    time_a_b_  = weights_ptr_ + offset; offset += d_model_;

    time_b_w_  = weights_ptr_ + offset; offset += d_model_ * half_units;
    time_b_b_  = weights_ptr_ + offset; offset += d_model_;

    time_scale_ = weights_ptr_ + offset;

    is_loaded_ = true;
    return true;
}

void OmniEngine::matmul(const float* W, const float* b,
                        const float* x, float* out, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float sum = (b != nullptr) ? b[i] : 0.0f;
        for (int j = 0; j < cols; ++j) {
            sum += x[j] * W[i * cols + j];
        }
        out[i] = sum;
    }
}

std::vector<float> OmniEngine::Step(const float* sensors, float dt, float abs_time) {
    if (!is_loaded_) {
        return std::vector<float>(output_dim_, 0.0f);
    }

    std::memset(latents_, 0, d_model_ * sizeof(float));

    apply_input_projection(sensors);
    add_temporal_encoding(abs_time);
    apply_bio_liquid_cell(dt);

    // Return current hidden state as the action vector
    std::vector<float> action(output_dim_, 0.0f);
    for (uint32_t i = 0; i < output_dim_ && i < d_model_; ++i) {
        action[i] = state_buffer_[i];
    }
    return action;
}

void OmniEngine::apply_input_projection(const float* sensors) {
    const float* w = weights_ptr_;
    const float* b = w + (input_dim_ * d_model_);
    matmul(w, b, sensors, latents_, d_model_, input_dim_);
}

void OmniEngine::add_temporal_encoding(float abs_time) {
    // Offsets must mirror the exact export order in esp32_exporter.py:
    //   push_tensor(cte.inv_freq)   -> shape [d/2]
    //   push_tensor(cte.amplitude)  -> shape [d]
    //   push_tensor(cte.phase)      -> shape [d/2]
    uint32_t proj_offset = (input_dim_ * d_model_) + d_model_;
    const float* inv_freq  = weights_ptr_ + proj_offset;
    const float* amplitude = inv_freq  + (d_model_ / 2);  // offset by d/2
    const float* phase     = amplitude + d_model_;         // offset by d (amplitude is full d_model_)

    // PyTorch cat([sin, cos], dim=-1) layout:
    // latents_[0 .. d/2-1]   += sin(t * inv_freq[i] + phase[i]) * amplitude[i]
    // latents_[d/2 .. d-1]   += cos(t * inv_freq[i] + phase[i]) * amplitude[d/2 + i]
    for (uint32_t i = 0; i < d_model_ / 2; ++i) {
        float arg = abs_time * inv_freq[i] + phase[i];
        latents_[i]                 += std::sin(arg) * amplitude[i];
        latents_[i + (d_model_/2)]  += std::cos(arg) * amplitude[i + (d_model_/2)];
    }
}

void OmniEngine::apply_bio_liquid_cell(float dt) {
    // Affine sensory mapping (mapped latent size is d_model_)
    for (uint32_t i = 0; i < d_model_; ++i) {
        x_in_[i] = latents_[i] * sensory_w_[i] + sensory_b_[i];
    }
    // Concatenate previous state
    for (uint32_t i = 0; i < d_model_; ++i) {
        x_in_[d_model_ + i] = state_buffer_[i];
    }

    uint32_t in_size    = d_model_ + d_model_;
    uint32_t half_units = backbone_units_ / 2;

    // Backbone State
    matmul(state_w_, state_b_, x_in_, b_state_, backbone_units_, in_size);
    for (uint32_t i = 0; i < backbone_units_; ++i) b_state_[i] = lecun_activation(b_state_[i]);

    // Backbone Time
    matmul(time_w_, time_b_, x_in_, b_time_, half_units, in_size);
    for (uint32_t i = 0; i < half_units; ++i) b_time_[i] = lecun_activation(b_time_[i]);

    // Feed-Forward Targets
    float ff1[OMNI_MAX_DIM], ff2[OMNI_MAX_DIM];
    matmul(ff1_w_, ff1_b_, b_state_, ff1, d_model_, backbone_units_);
    matmul(ff2_w_, ff2_b_, b_state_, ff2, d_model_, backbone_units_);
    // Note: tanh for ff1 (h_tilde) and sigmoid for ff2 (g) are applied in the loop

    // Time Interpolation Gate
    float time_a_out[OMNI_MAX_DIM], time_b_out[OMNI_MAX_DIM];
    matmul(time_a_w_, time_a_b_, b_time_, time_a_out, d_model_, half_units);
    matmul(time_b_w_, time_b_b_, b_time_, time_b_out, d_model_, half_units);

    float ts = std::max(dt, 0.0f);

    // CfC Default Mode state update (Hasani 2022)
    for (uint32_t i = 0; i < d_model_; ++i) {
        float h_tilde = std::tanh(ff1[i]);
        float g = sigmoid(ff2[i]);
        float t_scaled  = ts * std::abs(time_scale_[i]);
        float t_interp  = sigmoid(time_a_out[i] * t_scaled + time_b_out[i]);
        float h_prev = state_buffer_[i];
        
        state_buffer_[i] = (1.0f - t_interp) * (g * h_tilde + (1.0f - g) * h_prev) + t_interp * h_prev;
    }
}
