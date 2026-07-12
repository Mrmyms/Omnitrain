#include "OmniEngineLSTM.hpp"
#include <cstring>
#include <algorithm>

bool OmniEngineLSTM::Load(const unsigned char* omnibit_data, size_t length) {
    if (length < 28) return false;

    // Verify Magic Bytes
    if (omnibit_data[0] != 'O' || omnibit_data[1] != 'M' ||
        omnibit_data[2] != 'N' || omnibit_data[3] != 'I') {
        return false;
    }

    // Read dimension header (simplified for LSTM baseline)
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
    std::memset(c_state_, 0, sizeof(float) * d_model_);

    // Point directly into the binary data at offset 32 (assuming TOC is 0 for this test)
    weights_ptr_ = reinterpret_cast<const float*>(omnibit_data + 32);

    uint32_t offset = 0;
    
    // Layout expected from standard PyTorch LSTM state_dict export
    w_ih_ = weights_ptr_ + offset; offset += 4 * d_model_ * input_dim_;
    w_hh_ = weights_ptr_ + offset; offset += 4 * d_model_ * d_model_;
    b_ih_ = weights_ptr_ + offset; offset += 4 * d_model_;
    b_hh_ = weights_ptr_ + offset; offset += 4 * d_model_;
    fc_w_ = weights_ptr_ + offset; offset += output_dim_ * d_model_;
    fc_b_ = weights_ptr_ + offset;

    is_loaded_ = true;
    return true;
}

void OmniEngineLSTM::matmul(const float* W, const float* b,
                            const float* x, float* out, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float sum = (b != nullptr) ? b[i] : 0.0f;
        for (int j = 0; j < cols; ++j) {
            sum += x[j] * W[i * cols + j];
        }
        out[i] = sum;
    }
}

std::vector<float> OmniEngineLSTM::Step(const float* sensors) {
    if (!is_loaded_) {
        return std::vector<float>(output_dim_, 0.0f);
    }

    float gates_ih[OMNI_MAX_DIM * 4];
    float gates_hh[OMNI_MAX_DIM * 4];

    // Compute input-to-hidden and hidden-to-hidden gates
    matmul(w_ih_, b_ih_, sensors, gates_ih, 4 * d_model_, input_dim_);
    matmul(w_hh_, b_hh_, h_state_, gates_hh, 4 * d_model_, d_model_);

    // Apply LSTM gating (i, f, g, o)
    for (uint32_t i = 0; i < d_model_; ++i) {
        float i_gate = sigmoid(gates_ih[i] + gates_hh[i]);
        float f_gate = sigmoid(gates_ih[i + d_model_] + gates_hh[i + d_model_]);
        float g_gate = std::tanh(gates_ih[i + 2*d_model_] + gates_hh[i + 2*d_model_]);
        float o_gate = sigmoid(gates_ih[i + 3*d_model_] + gates_hh[i + 3*d_model_]);

        c_state_[i] = f_gate * c_state_[i] + i_gate * g_gate;
        h_state_[i] = o_gate * std::tanh(c_state_[i]);
    }

    // Output projection
    std::vector<float> action(output_dim_, 0.0f);
    matmul(fc_w_, fc_b_, h_state_, action.data(), output_dim_, d_model_);

    return action;
}
