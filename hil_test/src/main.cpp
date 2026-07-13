#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "esp_omni_engine.hpp"

// Architecture selection
#define USE_CFC

ESPOmniEngine engine;
Adafruit_MPU6050 mpu;

// Motor PWM config
const int MOTOR_PWM_PIN = 18;
const int MOTOR_DIR_PIN = 19;
const int PWM_FREQ = 5000;
const int PWM_RES = 8;
const int PWM_CHANNEL = 0;

float state_vector[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // [pos, vel, angle, ang_vel]
unsigned long last_time = 0;

// Dummy pointer for the .omnibit payload mapped via XIP (DROM)
extern const unsigned char payload_omnibit[];
extern const size_t payload_omnibit_len;

void setup() {
    Serial.begin(115200);
    Wire.begin(21, 22); // I2C pins for ESP32

    if (!mpu.begin()) {
        Serial.println("ERR: Failed to find MPU6050 chip");
        while (1) { delay(10); }
    }
    
    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu.setGyroRange(MPU6050_RANGE_250_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    // Setup PWM for motor
    ledcSetup(PWM_CHANNEL, PWM_FREQ, PWM_RES);
    ledcAttachPin(MOTOR_PWM_PIN, PWM_CHANNEL);
    pinMode(MOTOR_DIR_PIN, OUTPUT);

    // Load payload from Flash (ROM)
    if (!engine.Load(payload_omnibit, payload_omnibit_len)) {
        Serial.println("ERR: Failed to load OmniEngine payload.");
        while(1);
    }
    
    Serial.println("READY");
    last_time = micros();
}

void loop() {
    if (Serial.available()) {
        String input = Serial.readStringUntil('\n');
        
        // Parse simulated CartPole state injected from PC (Hardware-in-the-Loop)
        // Format: "POS,VEL,ANG,ANG_VEL"
        int first_comma = input.indexOf(',');
        int second_comma = input.indexOf(',', first_comma + 1);
        int third_comma = input.indexOf(',', second_comma + 1);
        
        if (first_comma > 0 && second_comma > 0 && third_comma > 0) {
            state_vector[0] = input.substring(0, first_comma).toFloat();
            state_vector[1] = input.substring(first_comma + 1, second_comma).toFloat();
            state_vector[2] = input.substring(second_comma + 1, third_comma).toFloat();
            state_vector[3] = input.substring(third_comma + 1).toFloat();
            
            unsigned long current_time = micros();
            float dt = (current_time - last_time) / 1000000.0f;
            last_time = current_time;

            // --- HARDWARE DUMMY LOAD ---
            // We read the physical I2C MPU6050 here purely to generate physical bus jitter 
            // and interrupt overhead, as described in the paper methodology.
            sensors_event_t a, g, temp;
            mpu.getEvent(&a, &g, &temp);
            // ---------------------------

            // Normalize states using hardcoded dataset statistics
            const float x_mean[4] = {0.00042742f, -0.00004877f, -0.00002502f, -0.00010277f};
            const float x_std[4]  = {0.00894568f,  0.01085484f,  0.00127859f,  0.00267853f};
            
            float norm_state[4];
            for (int i = 0; i < 4; i++) {
                norm_state[i] = (state_vector[i] - x_mean[i]) / x_std[i];
            }

            // Evaluate CfC Continuous-Time Model via Execute-in-Place
            std::vector<float> action = engine.Step(norm_state, dt, current_time / 1000000.0f);
            float force = action[0];
            
            // Actuate physical motor based on computed force
            if (force > 0) {
                digitalWrite(MOTOR_DIR_PIN, HIGH);
            } else {
                digitalWrite(MOTOR_DIR_PIN, LOW);
            }
            int pwm_val = min(255, (int)(abs(force) * 255.0f));
            ledcWrite(PWM_CHANNEL, pwm_val);

            // Send force back to PC Simulation
            Serial.print("F:");
            Serial.println(force, 4);
        }
    }
}
