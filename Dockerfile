FROM ubuntu:22.04

# OmniTrain Industrial Reproducibility Container
# Provides a frozen environment for PyTorch training and ESP32 PlatformIO compilation.

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /workspace/OmniTrain

# Install PlatformIO Core for ESP32 compilation
RUN python3 -m pip install -U platformio

# Copy requirements and install Python stack
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Copy the rest of the repository
COPY . .

# Install the omnitrain package in editable mode
RUN python3 -m pip install -e .

# Command to run PiL tests or compile the firmware
CMD ["/bin/bash"]
