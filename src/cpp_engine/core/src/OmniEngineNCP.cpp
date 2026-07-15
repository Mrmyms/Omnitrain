#include "OmniEngineNCP.hpp"
#include <cstring>
#include <iostream>

bool OmniEngineNCP::Load(const unsigned char* omnibit_data, size_t length) {
    if (length < 32) return false;
    
    // Magic 4 bytes + version 1 byte + arch 1 byte + 2 bytes padding
    if (omnibit_data[0] != 'O' || omnibit_data[1] != 'M' || 
        omnibit_data[2] != 'N' || omnibit_data[3] != 'I' || omnibit_data[4] != 4) {
        return false;
    }
    
    uint8_t arch_flag = omnibit_data[5];
    if (arch_flag != 4) return false; // Must be SparseCfC (NCP-CSR)
    
    const uint32_t* header = reinterpret_cast<const uint32_t*>(omnibit_data + 8);
    input_dim_ = header[0];
    d_model_ = header[1];
    output_dim_ = header[2];
    uint32_t num_tensors = header[5];
    
    if (d_model_ > OMNI_MAX_DIM) return false;
    
    const uint32_t* toc = header + 6;
    const float* weights = reinterpret_cast<const float*>(toc + num_tensors);
    
    if (num_tensors < 11) return false;
    
    size_t offset = 0;
    cfc_bb_val_ = weights + offset; offset += toc[0];
    
    // Indices are packed as floats by python struct unpacking, 
    // we safely reinterpret them as uint32_t for hardware efficiency.
    cfc_bb_col_ = reinterpret_cast<const uint32_t*>(weights + offset); offset += toc[1];
    cfc_bb_row_ = reinterpret_cast<const uint32_t*>(weights + offset); offset += toc[2];
    
    cfc_bb_b_ = weights + offset; offset += toc[3];
    cfc_f_w_ = weights + offset; offset += toc[4];
    cfc_f_b_ = weights + offset; offset += toc[5];
    cfc_g_w_ = weights + offset; offset += toc[6];
    cfc_g_b_ = weights + offset; offset += toc[7];
    cfc_h_w_ = weights + offset; offset += toc[8];
    cfc_h_b_ = weights + offset; offset += toc[9];
    
    if (num_tensors >= 13) {
        fc_w_ = weights + offset; offset += toc[10];
        fc_b_ = weights + offset; offset += toc[11];
    } else {
        fc_w_ = nullptr;
        fc_b_ = nullptr;
    }
    
    for (uint32_t i = 0; i < d_model_; ++i) h_state_[i] = 0.0f;
    is_loaded_ = true;
    return true;
}

void OmniEngineNCP::sparse_mat_vec_mul(const float* val, const uint32_t* col, const uint32_t* row,
                                       const float* x, float* out, int rows) {
    for (int i = 0; i < rows; ++i) {
        float sum = 0.0f;
        uint32_t start = row[i];
        uint32_t end = row[i + 1];
        // Only valid connections are computed (Zero-Multiply skipped)
        for (uint32_t j = start; j < end; ++j) {
            sum += val[j] * x[col[j]];
        }
        out[i] = sum;
    }
}

std::vector<float> OmniEngineNCP::Step(const float* sensors, float dt, float abs_time) {
    if (!is_loaded_) return std::vector<float>(output_dim_, 0.0f);
    
    float x_in[OMNI_MAX_DIM * 2];
    for (uint32_t i = 0; i < input_dim_; ++i) x_in[i] = sensors[i];
    for (uint32_t i = 0; i < d_model_; ++i) x_in[input_dim_ + i] = h_state_[i];
    
    float bb[OMNI_MAX_DIM];
    // Sparse evaluation!
    sparse_mat_vec_mul(cfc_bb_val_, cfc_bb_col_, cfc_bb_row_, x_in, bb, d_model_);
    
    for (uint32_t i = 0; i < d_model_; ++i) {
        bb[i] = std::tanh(bb[i] + cfc_bb_b_[i]);
        
        // Element-wise CfC gating equations
        float f_val = bb[i] * cfc_f_w_[i] + cfc_f_b_[i];
        float g_val = bb[i] * cfc_g_w_[i] + cfc_g_b_[i];
        float h_val = bb[i] * cfc_h_w_[i] + cfc_h_b_[i];
        
        float t_gate = 1.0f / (1.0f + std::exp(f_val * dt)); // Sigmoid(-f_val * dt)
        
        h_state_[i] = t_gate * std::tanh(g_val) + (1.0f - t_gate) * std::tanh(h_val);
    }
    
    std::vector<float> output(output_dim_, 0.0f);
    if (fc_w_) {
        for (uint32_t i = 0; i < output_dim_; ++i) {
            float sum = fc_b_[i];
            for (uint32_t j = 0; j < d_model_; ++j) {
                sum += h_state_[j] * fc_w_[i * d_model_ + j];
            }
            output[i] = sum;
        }
    }
    
    return output;
}
