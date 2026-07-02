#include "esp_omni_engine.hpp"
#include <cstring>
#include <iostream>
#include <algorithm>

bool ESPOmniEngine::Load(const unsigned char* omnibit_data, size_t length) {
    if (length < 28) {
        return false; 
    }

    // 1. Verificar Magic Bytes 'OMNI\x02' (V2)
    if (omnibit_data[0] != 'O' || omnibit_data[1] != 'M' || 
        omnibit_data[2] != 'N' || omnibit_data[3] != 'I' || 
        omnibit_data[4] != 0x02) {
        return false;
    }

    // 2. Leer metadata (20 bytes)
    uint32_t dims[5];
    std::memcpy(dims, omnibit_data + 8, sizeof(dims));
    
    input_dim_ = dims[0];
    d_model_ = dims[1];
    output_dim_ = dims[2];
    backbone_units_ = dims[3];
    total_weights_ = dims[4];

    if (input_dim_ + d_model_ > 512 || d_model_ > 256 || backbone_units_ > 256) {
        return false; // Sobrepasa preasignación estática
    }

    // Inicializar estado oculto (SRAM)
    std::memset(state_buffer_, 0, sizeof(state_buffer_));
    std::memset(latents_, 0, sizeof(latents_));

    // 3. Apuntar al arreglo de pesos en Flash (Offset = 28)
    weights_ptr_ = reinterpret_cast<const float*>(omnibit_data + 28);
    
    // 4. Mapeo estricto de offsets para acceso secuencial de matrices
    uint32_t offset = 0;
    
    // Proyector de entrada
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
    
    is_loaded = true;
    return true;
}

void ESPOmniEngine::matmul(const float* W, const float* b, const float* x, float* out, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float sum = (b != nullptr) ? b[i] : 0.0f;
        for (int j = 0; j < cols; ++j) {
            sum += x[j] * W[i * cols + j]; // Multiplicación Lineal O(R*C)
        }
        out[i] = sum;
    }
}

std::vector<float> ESPOmniEngine::Step(const float* sensors, float dt, float abs_time) {
    if (!is_loaded) {
        return std::vector<float>(output_dim_, 0.0f);
    }
    
    std::memset(latents_, 0, d_model_ * sizeof(float));
    
    // Fase 1: Proyección Adaptativa
    apply_input_projection(sensors);
    
    // Fase 2: Codificación Temporal Continua (CTE)
    add_temporal_encoding(abs_time);
    
    // Fase 3: BioLiquidCell Exacta
    apply_bio_liquid_cell(dt);
    
    // Fase 4: Generación de Acción (Usando estado actual)
    std::vector<float> action(output_dim_, 0.0f);
    for (uint32_t i = 0; i < output_dim_ && i < d_model_; ++i) {
        action[i] = state_buffer_[i];
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
        latents_[i] += std::sin(arg) * amplitude[i];
        latents_[i + (d_model_ / 2)] += std::cos(arg) * amplitude[i + (d_model_ / 2)];
    }
}

void ESPOmniEngine::apply_bio_liquid_cell(float dt) {
    // 1. Affine Sensory Mapping (x_mapped = x * sensory_w + sensory_b)
    for (uint32_t i = 0; i < input_dim_; ++i) {
        x_in_[i] = latents_[i] * sensory_w_[i] + sensory_b_[i];
    }
    
    // 2. Concatenar estado previo (x_in = cat([x_mapped, h_in]))
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
    
    // Clamp dt para seguridad y aplicar time_scale
    float ts = std::max(dt, 0.0f);
    
    // 7. Actualizar Estado (CfC Default Mode)
    for (uint32_t i = 0; i < d_model_; ++i) {
        float t_scaled = ts * std::abs(time_scale_[i]);
        float t_interp = sigmoid(time_a_out[i] * t_scaled + time_b_out[i]);
        state_buffer_[i] = ff1[i] * (1.0f - t_interp) + t_interp * ff2[i];
    }
}
