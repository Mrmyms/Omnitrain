#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "esp_omni_engine.hpp"
#include "model.h"
#include "USB.h"

// Architecture selection
#define USE_CFC

ESPOmniEngine engine;

unsigned long last_time = 0;

void setup() {
    USB.begin();
    Serial.begin(115200);
    unsigned long start_serial = millis();
    while(!Serial && millis() - start_serial < 2000) {
        delay(10);
    }
    
    Serial.println("ESP32: Booting OmniTrain HIL (Dynamic I/O)...");
    if (!engine.Load(model_omnibit, model_omnibit_len)) {
        Serial.println("ERR: Failed to load OmniEngine payload.");
        unsigned long err_timer = 0;
        while(1) {
            if (millis() - err_timer > 1000) {
                Serial.println("ERR: Stuck in Load fail loop");
                err_timer = millis();
            }
        }
    }
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
            engine.Load(model_omnibit, model_omnibit_len); // Reinitialize engine from scratch
            Serial.print("F:0.0,0.0\n");
            return;
        }

        // Only run inference if we received the expected number of inputs + 2 (dt and sim_time)
        if (num_inputs == engine.GetInputDim() + 2) {
            // Extract the simulated physics time provided by the Python Gym HIL server
            float dt = state_vector[num_inputs - 2];
            float sim_time = state_vector[num_inputs - 1];

            // Evaluate CfC Continuous-Time Model
            std::vector<float> action = engine.Step(state_vector, dt, sim_time);
            
            // Send forces back to PC Simulation
            Serial.print("F:");
            for (size_t i = 0; i < action.size(); i++) {
                Serial.print(action[i], 4);
                if (i < action.size() - 1) Serial.print(",");
            }
            Serial.println();
        } else {
            Serial.print("WARN: Input dim mismatch. Expected ");
            Serial.print(engine.GetInputDim());
            Serial.print(", got ");
            Serial.println(num_inputs);
        }
    }
}
