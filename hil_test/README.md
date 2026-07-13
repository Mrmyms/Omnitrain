# OmniTrain: Hardware-in-the-Loop (ESP32-S3)

This directory contains the C++ embedded inference engine (`OmniEngine`) required to run the Closed-form Continuous-time (CfC) mathematical models natively on an ESP32-S3 using the **Execute-in-Place (XIP)** Zero-Copy architecture.

It also contains the `pc_hil_server.py` script, which implements the literal "Open-Loop Hardware-in-the-Loop" architecture described in the paper by streaming simulated CartPole states over Serial while reading I2C sensor jitter from the physical MPU6050.

## 1. Exporting the PyTorch Model

Before flashing, you must export your trained PyTorch weights into our custom `.omnibit` format.
From the root of the python repository, run:

```bash
python src/omnitrain/esp32_exporter.py --input weights/my_cfc_model.pt --output hil_test/include/bot_brain.omnibit
```
*(This serializes the FP32 weights into a contiguous flat binary payload).*

## 2. Converting Payload to C-Array

To map the `.omnibit` directly into the ESP32-S3's Read-Only Memory (Flash) without dynamic allocation, we embed it as a static C-array. The repository provides a script to do this automatically:

```bash
# In the hil_test directory:
python scripts/bin2c.py include/bot_brain.omnibit > include/model.h
```

## 3. Flashing to ESP32-S3

We use **PlatformIO** for dependency management and flashing. The `platformio.ini` file is already configured for the ESP32-S3 with 80MHz QIO SPI Flash parameters to guarantee maximum data-cache (D-Cache) bandwidth.

Ensure the ESP32-S3 is physically wired to an MPU6050 over I2C (Pins 21 and 22 by default) and an L298N Motor Driver PWM pin on Pin 18 to recreate the paper's interrupt loads.

```bash
# Upload to the ESP32-S3
pio run -e esp32-s3-devkitc-1 -t upload
```

## 4. Running the HIL Serial Injection

To recreate the physical experiments described in Section V.B, you must run the PC-side Python script. This script simulates the pendulum physics, injects 20% packet loss, streams the kinematic states to the ESP32-S3 over Serial, and reads the computed forces back.

```bash
# In the hil_test directory:
pip install pyserial
python pc_hil_server.py --port /dev/cu.usbserial-0001 --baud 115200 --loss 0.20
```

The script will output the true Time-To-Failure (TTF) steps computed natively on the silicon.
