#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "esp_omni_engine.hpp"

// Physical HIL Test for Inverted Pendulum
// Reads real I2C MPU6050 data and outputs control force via PWM

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
extern const unsigned char cfc_pendulum_omnibit[];
extern const size_t cfc_pendulum_omnibit_len;

void setup() {
    Serial.begin(115200);
    Wire.begin(21, 22); // I2C pins for ESP32

    if (!mpu.begin()) {
        Serial.println("Failed to find MPU6050 chip");
        while (1) { delay(10); }
    }
    
    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu.setGyroRange(MPU6050_RANGE_250_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    // Setup PWM for motor
    ledcSetup(PWM_CHANNEL, PWM_FREQ, PWM_RES);
    ledcAttachPin(MOTOR_PWM_PIN, PWM_CHANNEL);
    pinMode(MOTOR_DIR_PIN, OUTPUT);

    Serial.println("Loading XIP CfC Model...");
    if (!engine.Load(cfc_pendulum_omnibit, cfc_pendulum_omnibit_len)) {
        Serial.println("Failed to load OmniEngine payload.");
        while(1);
    }
    
    Serial.println("HIL System Ready.");
    last_time = micros();
}

void loop() {
    unsigned long current_time = micros();
    float dt = (current_time - last_time) / 1000000.0f;
    last_time = current_time;

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Construct state vector from sensor data
    // (Simplified integration for CartPole)
    state_vector[2] = atan2(a.acceleration.y, a.acceleration.z); // Angle
    state_vector[3] = g.gyro.x; // Angular velocity

    // Evaluate CfC Model in continuous time
    const float* force_prediction = engine.Step(state_vector, dt, current_time / 1000000.0f);
    
    // Actuate motor
    float force = force_prediction[0];
    if (force > 0) {
        digitalWrite(MOTOR_DIR_PIN, HIGH);
    } else {
        digitalWrite(MOTOR_DIR_PIN, LOW);
    }
    
    int pwm_val = min(255, (int)(abs(force) * 255.0f));
    ledcWrite(PWM_CHANNEL, pwm_val);

    // Logging for HIL evaluation
    Serial.printf("T:%.3f, A:%.2f, V:%.2f, F:%.2f\n", dt, state_vector[2], state_vector[3], force);
    
    // ZOH stabilization wait (target 50Hz)
    delay(20);
}
