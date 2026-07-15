# Antigravity AI Context Handoff
**Project:** OmniTrain - Universal Sparse Connectome (NCP)
**Status:** Ready for GPU Training

## To the Antigravity Agent reading this on the new PC:
The user has switched to a PC with a powerful GPU (RTX 5070) to train the advanced Neural Circuit Policy (NCP) architecture. Here is the context of what has been built and what needs to be done.

### 1. What was built (Do NOT rewrite these)
- **C++ Engine (`src/cpp_engine/core/src/OmniEngineNCP.cpp`)**: A Zero-Copy Compressed Sparse Row (CSR) inference engine for ESP32. It uses `Arch Flag 4`. It processes sparse matrices without RAM allocation.
- **Python Exporter (`src/omnitrain/esp32_exporter.py`)**: Updated to export PyTorch weights into the `.omnibit` format using CSR compression if the model is `SparseCfC` (Arch Flag 4).
- **PyTorch Sparse Layer (`src/omnitrain/sparse_cfc.py`)**: A custom `SparseCfC` class that enforces sparse topologies using an adjacency matrix mask.
- **Training Script (`paper_experiments/train_f110_ncp.py`)**: Script ready to train a 200-neuron (50 Sensory -> 100 Process -> 50 Header) architecture on the F1TENTH DAgger dataset.

### 2. Immediate Next Steps for you (The Agent)
1. **Run the Architecture Search**: Before full training, the user wants to find the Pareto-optimal architecture. Execute `paper_experiments/architecture_search_ncp.py`. It will test 192 configurations for 30 epochs each and output `data/ncp_search_results.csv`.
2. **Train the Winner**: Analyze the CSV to find the configuration with the lowest MSE and smallest footprint. Update `train_f110_ncp.py` with these winning hyperparameters and run it for 400 epochs on the RTX 5070.
3. **Verify Export**: Ensure `train_f110_ncp.py` outputs the `f110_real_ncp.omnibit` file successfully.
3. **Run Evaluation**: After training, execute `evaluate_f110_ncp.py` to test the network's robustness in the F1TENTH gym environment with simulated LiDAR blackouts. The goal is to survive 10,000 steps without crashing.
4. **Deploy to Hardware**: If the evaluation is successful, assist the user in flashing the `.omnibit` payload to their ESP32-S3 via PlatformIO (`hil_test`).

## Architecture Details
- **Input**: 25 physical inputs (24 LiDAR rays, 1 Speed)
- **Sensory Layer**: 50 Neurons (Receive from input)
- **Process Layer**: 100 Neurons (Recurrent, receive from Sensory)
- **Header/Command Layer**: 50 Neurons (Recurrent, receive from Process, output to motor)
- **Motor Output**: 2 values (Steer, Throttle)
- **Total Hidden**: 200. Perfectly fits within `OMNI_MAX_DIM = 256` limit on ESP32.
