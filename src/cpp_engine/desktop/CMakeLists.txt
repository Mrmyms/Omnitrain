cmake_minimum_required(VERSION 3.17)
project(OmniTrainEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
