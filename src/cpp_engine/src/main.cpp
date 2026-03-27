#include "OmniEngine.hpp"
#include <iostream>
#include <vector>

/**
 * @brief Main program for OmniTrain native inference.
 * Demonstrates the C++ engine (OmniEngine) orchestration with dynamic EPs.
 */

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: omni_engine <model_path.onnx>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];

    try {
        std::cout << "--- OmniTrain Native Engine 2.0 ---" << std::endl;
        
        // 1. Engine Initialization (Automatic Fallback enabled)
        OmniEngine engine(model_path);

        // 2. Mock Sensor Data (Simulating 100 async tokens of 512-dim)
        int64_t batch_size = 1;
        int64_t num_tokens = 100;
        std::vector<float> mock_tokens(batch_size * num_tokens * 512, 0.5f);
        std::vector<float> mock_timestamps(batch_size * num_tokens * 1, 0.123f);

        // 3. High-Speed Inference (Zero-GIL)
        std::cout << "🚀 Running inference on " << num_tokens << " tokens..." << std::endl;
        engine.Forward(mock_tokens, mock_timestamps, batch_size, num_tokens);

        std::cout << "✅ Inference cycle completed successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Execution error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
