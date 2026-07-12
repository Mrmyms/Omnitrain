#include "OmniEngineGRU.hpp"
#include <cstring>
#include <algorithm>

bool OmniEngineGRU::Load(const unsigned char* omnibit_data, size_t length) {
    if (length < 28) return false;

    // Verify Magic Bytes
    if (omnibit_data[0] != 'O' || omnibit_data[1] != 'M' ||
        omnibit_data[2] != 'N' || omnibit_data[3] != 'I') {
        return false;
    }

    // Read dimension header
    uint32_t dims[5];
    std::memcpy(dims, omnibit_data + 8, sizeof(dims));

    input_dim_      = dims[0];
    d_model_        = dims[1]; // Hidden dimension
    output_dim_     = dims[2];
    total_weights_  = dims[4];

    if (d_model_ > OMNI_MAX_DIM || (input_dim_ + d_model_) > OMNI_MAX_DIM * 2) {
        return false;
    }

    std::memset(h_state_, 0, sizeof(float) * d_model_);

    // Point directly into the binary data at offset 32
    weights_ptr_ = reinterpret_cast<const float*>(omnibit_data + 32);

    uint32_t offset = 0;
    
    // Layout expected from standard PyTorch GRU state_dict export
    // GRU has 3 gates: reset, update, new
    w_ih_ = weights_ptr_ + offset; offset += 3 * d_model_ * input_dim_;
    w_hh_ = weights_ptr_ + offset; offset += 3 * d_model_ * d_model_;
    b_ih_ = weights_ptr_ + offset; offset += 3 * d_model_;
    b_hh_ = weights_ptr_ + offset; offset += 3 * d_model_;
    fc_w_ = weights_ptr_ + offset; offset += output_dim_ * d_model_;
    fc_b_ = weights_ptr_ + offset;

    is_loaded_ = true;
    return true;
}

void OmniEngineGRU::matmul(const float* W, const float* b,
                            const float* x, float* out, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float sum = (b != nullptr) ? b[i] : 0.0f;
        for (int j = 0; j < cols; ++j) {
            sum += x[j] * W[i * cols + j];
        }
        out[i] = sum;
    }
}

std::vector<float> OmniEngineGRU::Step(const float* sensors) {
    if (!is_loaded_) {
        return std::vector<float>(output_dim_, 0.0f);
    }

    float gates_ih[OMNI_MAX_DIM * 3];
    float gates_hh[OMNI_MAX_DIM * 3];

    // Compute input-to-hidden and hidden-to-hidden gates
    matmul(w_ih_, b_ih_, sensors, gates_ih, 3 * d_model_, input_dim_);
    matmul(w_hh_, b_hh_, h_state_, gates_hh, 3 * d_model_, d_model_);

    // Apply GRU gating (r, z, n)
    for (uint32_t i = 0; i < d_model_; ++i) {
        float r_gate = sigmoid(gates_ih[i] + gates_hh[i]);
        float z_gate = sigmoid(gates_ih[i + d_model_] + gates_hh[i + d_model_]);
        float n_gate = std::tanh(gates_ih[i + 2*d_model_] + r_gate * gates_hh[i + 2*d_model_]);

        h_state_[i] = (1.0f - z_gate) * n_gate + z_gate * h_state_[i];
    }

    // Output projection
    std::vector<float> action(output_dim_, 0.0f);
    matmul(fc_w_, fc_b_, h_state_, action.data(), output_dim_, d_model_);

    return action;
}
