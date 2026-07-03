#include <Arduino.h>
#include "esp_omni_engine.hpp"
#include "model.h"

ESPOmniEngine engine;
bool is_ready = false;

// We expect: [dt (float), x1, x2, x3, x4 (floats)]
// Total 5 floats per packet.

void setup() {
    Serial.begin(115200);
    while (!Serial) { delay(10); }

    is_ready = engine.Load(hil_model_omnibit, hil_model_omnibit_len);

    if (is_ready) {
        Serial.println("OMNI_READY");
    } else {
        Serial.println("OMNI_ERROR");
    }
}

void loop() {
    if (is_ready && Serial.available() >= sizeof(float) * 5) {
        float packet[5];
        Serial.readBytes((char*)packet, sizeof(packet));

        float dt = packet[0];
        const float* sensors = &packet[1];
        
        static float abs_time = 0.0f;
        abs_time += dt;

        std::vector<float> action = engine.Step(sensors, dt, abs_time);
        
        if (!action.empty()) {
            float pred = action[0]; 
            Serial.write((char*)&pred, sizeof(float));
        } else {
            float err = -999.0f;
            Serial.write((char*)&err, sizeof(float));
        }
    }
}
