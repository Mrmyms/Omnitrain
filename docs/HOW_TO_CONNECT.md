# Connectivity Guide: How to connect sensors to OmniTrain

This guide explains the 4 available methods to feed data (inputs) into the OmniTrain brain. Regardless of the method, all data ends up in the `TokenBus` (Shared Memory) to be processed by the Liquid Neural Network solver.

---

## Prerequisite: YAML Configuration (`config.yaml`)
Before connecting any sensor, you must define its "space" in `config.yaml`. The system reads this file and automatically reserves memory (Shared Memory) for each sensor. The `id` parameter must perfectly match the one you use in your plugin code.

```yaml
inputs:
  - id: "my_sensor_name"
    type: "sensor"       # Options: "sensor" (vector), "vision" (images), "boolean" (flags)
    hz: 10               # Update frequency in Hertz. Crucial for continuous-time calculus (dt).
    dim: 512             # Dimension of the final token. 512 is recommended to standardize the 'Conectoma'.
    range: [0.0, 5.0]    # Physical value range (min, max). Useful for the AI to know the physical boundaries.
    noise: true          # If true, the neural network will apply "Curriculum Dropout" assuming the sensor can fail.
```

---

## Creating a Custom Plugin (Custom Sensors)
If you have a non-standard sensor (e.g., a microphone array or I2C tactile sensors), you can create your own bridge by inheriting from the `ModalityPlugin` base class. The only requirement is that you transform your raw data into a `Numpy` vector of the size defined in `dim`.

```python
import numpy as np
from omnitrain.plugins import ModalityPlugin

class MyCustomSensor(ModalityPlugin):
    def read_raw_data(self):
        # 1. Hardware logic: Read from Serial, I2C, SPI port, etc.
        # EXAMPLE: return spi.read()
        return [2.5, 3.1, 0.4] 

    def encode(self, raw_data):
        # 2. Pre-processing and Normalization
        # The system expects a vector (token) of the exact size defined in the yaml (e.g. 512)
        token = np.zeros(512, dtype='float32')
        
        # Insert data and normalize to the [0, 1] range if possible
        token[0] = raw_data[0] / 5.0
        token[1] = raw_data[1] / 5.0
        token[2] = raw_data[2] / 5.0
        
        return token

# To run it:
# plugin = MyCustomSensor(bus, "my_sensor_name", hz=10)
# plugin.run()
```

---

## 1. Offline / Real Data Method (`plugins.py`)
Ideal for quick testing without hardware or for re-training with recordings.

### Using CSV
If you have data in a table:
1. Make sure the CSV columns match your sensor.
2. Use `CSVModalityPlugin`.

```python
from omnitrain.plugins import CSVModalityPlugin
plugin = CSVModalityPlugin(bus, "my_sensor_name", hz=10, csv_path="data.csv")
plugin.run()
```

### Using Image Folders
To simulate a camera with local photos:
```python
from omnitrain.plugins import ImageFolderPlugin
plugin = ImageFolderPlugin(bus, "my_vision", hz=5, img_dir="./frames")
plugin.run()
```

---

## 2. Robotics Method (ROS 2)
The standard for physical robots (Humble/Iron/Jazzy). OmniTrain uses an internal "Singleton" pattern for the ROS 2 Node, which means you can create dozens of plugins without colliding with the `rclpy` memory manager.

1. **Make sure ROS 2 is in your path** (`source /opt/ros/...`).
2. **If you use custom messages**, make sure to build and source your local ROS 2 workspace.
3. **Launch the Plugins** by instantiating the pre-built ones, or inherit from `ROS2BasePlugin` if you need a message other than Image/LaserScan.

```python
from omnitrain.plugins import ROS2CameraPlugin, ROS2LidarPlugin

# Connect Camera (automatically transforms sensor_msgs/Image to a 512-dim Token)
cam = ROS2CameraPlugin(bus, "front_cam", hz=30, topic_name="/camera/image_raw")
cam.run() # IMPORTANT: run() blocks the thread. If you use multiple plugins, launch them in separate threads or processes.

# Connect Lidar (cleans NaNs/Infs and automatically downsamples to 512-dim)
lidar = ROS2LidarPlugin(bus, "laser_scan", hz=10, topic_name="/scan")
lidar.run()
```

---

## 3. Simulation Method (NVIDIA Isaac Sim)
For training "Digital Twins" and using Reinforcement Learning in Omniverse. This bridge is highly optimized for low RAM consumption (ideal for workstations with 16GB VRAM, like an RTX 5070).

1. Launch the native Isaac Sim Python environment (`python.sh`).
2. The `IsaacOmniBridge` loads the robot, launches the GPU physics simulator, and bridges the virtual sensor data (e.g., `LidarRtx`) directly to the `TokenBus`.
3. It also handles the **closed feedback loop**: it extracts the actions produced by the AI and sends them to the `ArticulationAction` of the motors in the simulation.

```python
from omnitrain.isaac_bridge import IsaacOmniBridge

# token_dim must be identical to the 'dim' in config.yaml
bridge = IsaacOmniBridge(session_id="isaac_train", robot_name="my_robot")
bridge.setup_scene(robot_usd_path="/paths/to/my_robot.usd") 
# The simulation will start transmitting in real-time.
```

---

## 4. Distributed Hardware Method
For systems with dual processors (e.g., Qualcomm for AI + STM32 for Control).

1. **AI Brain**: Processes the neural network and sends "intents".
2. **Action Brain**: Receives the intents and enforces the `OmniShieldGuard`.

```python
from omnitrain.edgecp_bridge import DualBrainRPC, EdgeCPAIBrain, EdgeCPActionBrain

rpc = DualBrainRPC()
ai = EdgeCPAIBrain(rpc)
action = EdgeCPActionBrain(rpc, my_shield)

action.start() # Loop at 1000Hz
ai.start()     # Loop at 30Hz
```

---

## Golden Rules
- **Normalization**: Make sure that in `encode()`, your data always ends up between `0.0` and `1.0` or `-1.0` and `1.0`. Liquid networks are highly scale-sensitive.
- **Asynchrony**: Plugins run in their own threads/processes. Do not block the main ROS thread.
- **Diagnostics**: Use the `/bus` command in the CLI to verify in real-time if the data is arriving at the `TokenBus`.
