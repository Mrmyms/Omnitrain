#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "OmniEngineCFC.hpp"
#include "OmniEngineGRU.hpp"
#include "OmniEngineLSTM.hpp"
#include "model.h"
#include "USB.h"

// Architecture selection
#define USE_CFC

OmniEngineCFC engine_cfc;
OmniEngineGRU engine_gru;
OmniEngineLSTM engine_lstm;
int active_arch = 0; // 0=CfC/SparseCfC, 1=GRU, 5=LSTM

unsigned long last_time = 0;

// 64KB buffer for dynamically loading weights over Serial
__attribute__((aligned(4))) uint8_t dynamic_model_buffer[65536];
bool using_dynamic_model = false;
size_t dynamic_model_len = 0;

void setup() {
    USB.begin();
    Serial.begin(115200);
    unsigned long start_serial = millis();
    while(!Serial && millis() - start_serial < 2000) {
        delay(10);
    }
    
    Serial.println("ESP32: Booting OmniTrain HIL (Dynamic I/O)...");
    if (!engine_cfc.Load(model_omnibit, model_omnibit_len)) {
        Serial.println("ERR: Failed to load OmniEngine payload.");
        unsigned long err_timer = 0;
        while(1) {
            if (millis() - err_timer > 1000) {
                Serial.println("ERR: Stuck in Load fail loop");
                err_timer = millis();
            }
        }
    }
    active_arch = 0;
    Serial.println("ESP32: OmniEngine loaded successfully.");
    Serial.println("READY");
    last_time = micros();
}

void loop() {
    static unsigned long last_heartbeat = 0;
    if (millis() - last_heartbeat > 1000) {
        Serial.println("HEARTBEAT");
        last_heartbeat = millis();
    }
    
    if (Serial.available()) {
        String input = Serial.readStringUntil('\n');
        input.trim();

        // Check for LOAD command
        if (input.startsWith("LOAD:")) {
            size_t payload_size = input.substring(5).toInt();
            if (payload_size > 0 && payload_size <= sizeof(dynamic_model_buffer)) {
                Serial.print("ACK_LOAD:");
                Serial.println(payload_size);
                
                // Read binary payload
                size_t bytes_read = 0;
                unsigned long start_wait = millis();
                while (bytes_read < payload_size && millis() - start_wait < 5000) {
                    if (Serial.available()) {
                        bytes_read += Serial.readBytes(dynamic_model_buffer + bytes_read, payload_size - bytes_read);
                        start_wait = millis(); // Reset timeout on successful read
                    }
                }
                
                if (bytes_read == payload_size) {
                    uint8_t arch = dynamic_model_buffer[5];
                    bool loaded = false;
                    if (arch == 1) {
                        loaded = engine_gru.Load(dynamic_model_buffer, payload_size);
                        if (loaded) active_arch = 1;
                    } else if (arch == 5) {
                        loaded = engine_lstm.Load(dynamic_model_buffer, payload_size);
                        if (loaded) active_arch = 5;
                    } else {
                        loaded = engine_cfc.Load(dynamic_model_buffer, payload_size);
                        if (loaded) active_arch = 0;
                    }

                    if (loaded) {
                        using_dynamic_model = true;
                        dynamic_model_len = payload_size;
                        Serial.println("LOAD_OK");
                    } else {
                        Serial.println("LOAD_ERR: Parse failed");
                    }
                } else {
                    Serial.println("LOAD_ERR: Timeout");
                }
            } else {
                Serial.println("LOAD_ERR: Invalid size");
            }
            return;
        }
        
        float state_vector[32]; // Max 32 inputs supported for HIL
        int num_inputs = 0;
        int start_idx = 0;
        
        // Parse CSV string into floats
        for (unsigned int i = 0; i < input.length(); i++) {
            if (input[i] == ',' || i == input.length() - 1) {
                int end_idx = (input[i] == ',') ? i : i + 1;
                state_vector[num_inputs++] = input.substring(start_idx, end_idx).toFloat();
                start_idx = i + 1;
                if (num_inputs >= 32) break;
            }
        }
        
        // Reset command hook (if Python sends a single 999.0, clear all memory)
        if (num_inputs == 1 && state_vector[0] == 999.0f) {
            if (using_dynamic_model) {
                uint8_t arch = dynamic_model_buffer[5];
                if (arch == 1) engine_gru.Load(dynamic_model_buffer, dynamic_model_len);
                else if (arch == 5) engine_lstm.Load(dynamic_model_buffer, dynamic_model_len);
                else engine_cfc.Load(dynamic_model_buffer, dynamic_model_len);
            } else {
                engine_cfc.Load(model_omnibit, model_omnibit_len); // Reinitialize engine from scratch
                active_arch = 0;
            }
            Serial.print("F:0.0,0.0\n");
            return;
        }

        int expected_inputs = 0;
        if (active_arch == 1) expected_inputs = engine_gru.GetInputDim() + 1; // 26 + 1 (sim_time ignored by GRU but sent by server dt is included in input_dim)
        else if (active_arch == 5) expected_inputs = engine_lstm.GetInputDim() + 1;
        else expected_inputs = engine_cfc.GetInputDim() + 2;

        if (num_inputs >= expected_inputs) {
            float action[2] = {0.0f, 0.0f};
            int action_size = 0;

            if (active_arch == 1) {
                // dt is already inside state_vector since GRU input_dim is 26
                engine_gru.Step(state_vector, action);
                action_size = engine_gru.GetOutputDim();
            } else if (active_arch == 5) {
                engine_lstm.Step(state_vector, action);
                action_size = engine_lstm.GetOutputDim();
            } else {
                float dt = state_vector[num_inputs - 2];
                float sim_time = state_vector[num_inputs - 1];
                engine_cfc.Step(state_vector, dt, sim_time, action);
                action_size = engine_cfc.GetOutputDim();
            }
            
            // Send forces back to PC Simulation
            Serial.print("F:");
            for (int i = 0; i < action_size; i++) {
                Serial.print(action[i], 4);
                if (i < action_size - 1) Serial.print(",");
            }
            Serial.println();
        } else {
            Serial.print("WARN: Input dim mismatch. Expected ");
            Serial.print(expected_inputs);
            Serial.print(", got ");
            Serial.println(num_inputs);
        }
    }
}
