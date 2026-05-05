import numpy as np
import time
import logging
from typing import Tuple, Dict, Optional

# Attempt to load Qualcomm SNPE bindings
try:
    # SNPE python bindings are typically provided by the SDK in PYTHONPATH
    from snpe import snpe
    HAS_SNPE = True
except ImportError:
    HAS_SNPE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class OmniSNPERunner:
    """
    Qualcomm SNPE Runner for OmniTrain.
    
    Optimized for Hexagon DSP / NPU inference on Snapdragon devices.
    Requires the model to be converted to .dlc format using snpe-onnx-to-dlc.
    """
    
    def __init__(self, dlc_path: str, runtime: str = 'DSP', max_batch: int = 1, max_tokens: int = 10, d_model: int = 256):
        self.dlc_path = dlc_path
        self.runtime_target = runtime  # Options: 'CPU', 'GPU', 'DSP', 'AIP'
        self.max_batch = max_batch
        self.max_tokens = max_tokens
        self.d_model = d_model
        
        if HAS_SNPE:
            # Initialize SNPE Context
            self.logger = snpe.DlSystem.ZdlLogger()
            self.logger.SetLogLevel(snpe.DlSystem.LogLevel_t.LOG_WARN)
            
            # Select target runtime
            runtime_t = snpe.DlSystem.Runtime_t.CPU
            if self.runtime_target == 'DSP':
                runtime_t = snpe.DlSystem.Runtime_t.DSP
            elif self.runtime_target == 'GPU':
                runtime_t = snpe.DlSystem.Runtime_t.GPU
            elif self.runtime_target == 'AIP':
                runtime_t = snpe.DlSystem.Runtime_t.AIP_FIXED8_TF
                
            if not snpe.DlSystem.RuntimeList().hasRuntime(runtime_t):
                logging.warning(f"Runtime {self.runtime_target} not available on this device. Falling back to CPU.")
                runtime_t = snpe.DlSystem.Runtime_t.CPU

            # Build SNPE Network
            container = snpe.DlContainer.DlContainer()
            container.load(self.dlc_path)
            
            builder = snpe.DlSystem.SNPEBuilder(container)
            builder.setRuntimeProcessor(runtime_t)
            
            self.engine = builder.build()
            
            # Setup input/output buffers (Using UserBuffer for zero-copy if possible)
            self.input_tensor_map = snpe.DlSystem.TensorMap()
            self.output_tensor_map = snpe.DlSystem.TensorMap()
            
            self.input_names = self.engine.getInputTensorNames()
            self.output_names = self.engine.getOutputTensorNames()
            
        else:
            logging.warning("Qualcomm SNPE bindings not found. Running in SIMULATION mode.")

    def step(self, 
             sensor_tokens: np.ndarray, 
             dt: float, 
             prev_state: np.ndarray, 
             abs_time: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Single inference step on the Snapdragon NPU.
        
        Args:
            sensor_tokens: (Batch, N, d_model)
            dt: float
            prev_state: (Batch, n_latents, d_model)
            abs_time: float
        Returns:
            next_state, action, latency_ms
        """
        start_time = time.perf_counter()
        
        if HAS_SNPE:
            # 1. Prepare inputs (Static Shapes Required)
            # SNPE requires float32 tensors (or int8 if quantized, handled by the container)
            t_sensors = snpe.DlSystem.FloatTensor(sensor_tokens.shape)
            t_sensors[:] = sensor_tokens.ravel()
            self.input_tensor_map.add("sensor_tokens", t_sensors)
            
            t_dt = snpe.DlSystem.FloatTensor((self.max_batch, 1))
            t_dt[0] = dt
            self.input_tensor_map.add("dt", t_dt)
            
            t_state = snpe.DlSystem.FloatTensor(prev_state.shape)
            t_state[:] = prev_state.ravel()
            self.input_tensor_map.add("prev_state", t_state)
            
            t_abs = snpe.DlSystem.FloatTensor((self.max_batch, 1))
            t_abs[0] = abs_time
            self.input_tensor_map.add("abs_time", t_abs)
            
            # 2. Execute on NPU
            self.engine.execute(self.input_tensor_map, self.output_tensor_map)
            
            # 3. Retrieve Outputs
            next_state_t = self.output_tensor_map.getTensor("next_state")
            next_state = np.array(next_state_t).reshape(prev_state.shape)
            
            # Assuming 'head_action' is the first head output for simplicity
            # In a real scenario, we map exactly to the known output names
            action_name = [n for n in self.output_names if "head_" in n][0]
            action_t = self.output_tensor_map.getTensor(action_name)
            action = np.array(action_t)
            
        else:
            # Simulation Mode
            time.sleep(0.005) # Simulate 5ms DSP latency
            next_state = np.zeros_like(prev_state)
            action = np.zeros((self.max_batch, 4)) # Mock 4-dim action

        latency_ms = (time.perf_counter() - start_time) * 1000
        return next_state, action, latency_ms
