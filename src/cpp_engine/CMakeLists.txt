cmake_minimum_required(VERSION 3.17)
project(OmniTrainEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- MLOps: ONNX Runtime Configuration ---

# ARCHITECTURAL NOTE: 
# To achieve the "Ollama Effect" and avoid 'Shared Object Not Found' errors,
# we recommend using the "Shared Providers" feature of ONNX Runtime.
# In this configuration, the TensorRT and CUDA libraries are NOT statically linked
# to the main binary; instead, ONNX Runtime loads them via dlopen() at runtime.
# The binary only depends on 'libonnxruntime.so'.

find_package(onnxruntime REQUIRED)

add_executable(omni_engine 
    src/OmniEngine.cpp
    src/main.cpp
)

target_include_directories(omni_engine PRIVATE 
    ${ONNXRUNTIME_INCLUDE_DIRS}
    include
)

target_link_libraries(omni_engine PRIVATE 
    onnxruntime
)

# --- Deployment Configuration ---
# Make sure to distribute the provider libraries (libonnxruntime_providers_shared.so, 
# libonnxruntime_providers_cuda.so, etc.) in the same directory as the binary or in LD_LIBRARY_PATH.
# This is what allows the main binary 'omni_engine' to be hardware-agnostic
# until the moment of instantiation.
