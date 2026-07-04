#include <Arduino.h>
#include "esp_omni_engine.hpp"

// In a real Zero-Copy deployment, these arrays would be stored in the Flash Memory (DROM).
// Here we simulate loading the binary blobs from SPIFFS/SD or using PROGMEM.
// For the benchmark, we assume the omnibit_data points to a valid memory region.

ESPOmniEngine engine;

// Dummy sensor data (dim 16)
float sensors[16] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                     0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6};

void run_benchmark(const char* arch_name, const unsigned char* model_data, size_t data_len) {
    Serial.printf("\n--- Benchmarking %s ---\n", arch_name);
    
    if (!engine.Load(model_data, data_len)) {
        Serial.println("Failed to load model!");
        return;
    }
    
    Serial.printf("Loaded! Input: %lu, Hidden: %lu\n", engine.GetInputDim(), engine.GetModelDim());
    
    // Warmup
    engine.Step(sensors, 0.1f, 0.1f);
    
    // Measure Latency
    unsigned long start_time = micros();
    int num_runs = 100;
    
    for (int i = 0; i < num_runs; i++) {
        // Increment absolute time and pass dt=0.1s
        engine.Step(sensors, 0.1f, 0.1f * (i + 2)); 
    }
    
    unsigned long end_time = micros();
    float avg_latency_ms = (end_time - start_time) / 1000.0f / num_runs;
    
    Serial.printf("Average Latency: %.3f ms\n", avg_latency_ms);
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("\nStarting Zero-Copy Architecture Ablation Benchmark...");
    
    // TODO: Load the actual binary arrays from Flash here.
    // For example, if you include them via xxd as header files:
    // run_benchmark("CfC (BioLiquidCell)", cfc_ablation_omnibit, cfc_ablation_omnibit_len);
    // run_benchmark("GRU", gru_ablation_omnibit, gru_ablation_omnibit_len);
    // run_benchmark("Transformer", transformer_ablation_omnibit, transformer_ablation_omnibit_len);
}

void loop() {
    delay(10000); // Do nothing
}
