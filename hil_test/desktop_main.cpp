#include <iostream>
#include <fstream>
#include <vector>
#include "esp_omni_engine.hpp"
#include "model.h"

bool read_binary_floats(const char* filename, std::vector<float>& data) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) return false;
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    data.resize(size / sizeof(float));
    if (file.read((char*)data.data(), size)) return true;
    return false;
}

int main() {
    ESPOmniEngine engine;
    if (!engine.Load(hil_model_omnibit, hil_model_omnibit_len)) {
        std::cerr << "Failed to load OmniBit model" << std::endl;
        return 1;
    }

    std::vector<float> X, Y, T;
    if (!read_binary_floats("../data/X_0_raw.bin", X) ||
        !read_binary_floats("../data/Y_raw.bin", Y) ||
        !read_binary_floats("../data/T_raw.bin", T)) {
        std::cerr << "Failed to read data files" << std::endl;
        return 1;
    }

    int num_samples = X.size() / 4;
    float mse = 0.0f;
    float abs_time = 0.0f;

    for (int i = 0; i < num_samples; ++i) {
        float dt = (i == 0) ? 0.0f : (T[i] - T[i-1]);
        abs_time += dt;
        
        std::vector<float> action = engine.Step(&X[i * 4], dt, abs_time);
        
        if (!action.empty()) {
            float err = action[0] - Y[i];
            mse += err * err;
        }
    }
    
    mse /= num_samples;
    std::cout << "--- C++ Engine Parity Verification ---" << std::endl;
    std::cout << "C++ Engine MSE: " << mse << std::endl;
    std::cout << "PyTorch    MSE: 0.0020" << std::endl;
    std::cout << "Difference    : " << std::abs(mse - 0.0020f) << std::endl;
    return 0;
}
