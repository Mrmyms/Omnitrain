import numpy as np
import logging
import queue
import time
from typing import Any
from .plugins import ModalityPlugin

class ROS2BasePlugin(ModalityPlugin):
    """
    ROS2 Modality Plugin.
    Uses Best-Effort QoS for high-frequency data and synchronized clock.
    """
    def __init__(self, bus, modal_id, frequency_hz, write_ptr, topic_name: str, msg_type: Any):
        super().__init__(bus, modal_id, frequency_hz, write_ptr=write_ptr)
        self.topic_name = topic_name
        self.msg_type = msg_type
        self.msg_queue = queue.Queue(maxsize=1) 
        
        try:
            from .ros2_bridge import OmniROS2Node, OmniQoS
            self.ros_node = OmniROS2Node()
            self.qos = OmniQoS.SENSOR_DATA
        except ImportError as e:
            logging.error(f"[{modal_id}] Initialization Failed: {e}")
            raise

        self.ros_node.create_subscription(
            self.msg_type,
            self.topic_name,
            self._ros_callback,
            qos=self.qos 
        )
        
        self.ros_node.start_spinning()
        logging.info(f"[{modal_id}] Connected to ROS2 (QoS: Best-Effort) on {topic_name}")

    def _ros_callback(self, msg):
        """Zero-latency callback. Keeps only the latest message."""
        while not self.msg_queue.empty():
            try: self.msg_queue.get_nowait()
            except queue.Empty: break
        self.msg_queue.put_nowait(msg)

    def read_raw_data(self) -> Any:
        try:
            return self.msg_queue.get_nowait()
        except queue.Empty:
            return None


class ROS2CameraPlugin(ROS2BasePlugin):
    """Subscribes to sensor_msgs/Image and encodes to neural tokens."""
    def __init__(self, bus, modal_id, frequency_hz, write_ptr, topic_name: str = "/camera/image_raw"):
        try:
            from sensor_msgs.msg import Image
        except ImportError:
            raise ImportError("sensor_msgs missing.")
        super().__init__(bus, modal_id, frequency_hz, write_ptr, topic_name, msg_type=Image)

    def encode(self, raw_data: Any) -> np.ndarray:
        if raw_data is None: return np.zeros(self.bus.token_dim, dtype='float32')
            
        try:
            import cv2
            
            img_np = np.frombuffer(raw_data.data, dtype=np.uint8).reshape(raw_data.height, raw_data.width, -1)
            if img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Optimized Resize (Must fit into token_dim)
            # 16x16x3 = 768 (Too big for 512 token)
            # Use 12x12x3 = 432 or 16x16 grayscale = 256
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (16, 16)).flatten().astype(np.float32) / 255.0
            token = np.zeros(self.bus.token_dim, dtype='float32')
            token[:len(resized)] = resized
            return token
        except Exception:
            return np.zeros(self.bus.token_dim, dtype='float32')


class ROS2LidarPlugin(ROS2BasePlugin):
    """Subscribes to sensor_msgs/LaserScan."""
    def __init__(self, bus, modal_id, frequency_hz, write_ptr, topic_name: str = "/scan"):
        try:
            from sensor_msgs.msg import LaserScan
        except ImportError:
            raise ImportError("sensor_msgs missing.")
        super().__init__(bus, modal_id, frequency_hz, write_ptr, topic_name, msg_type=LaserScan)

    def encode(self, raw_data: Any) -> np.ndarray:
        if raw_data is None: return np.zeros(self.bus.token_dim, dtype='float32')
            
        ranges = np.array(raw_data.ranges, dtype='float32')
        ranges = np.nan_to_num(ranges, posinf=raw_data.range_max, neginf=0.0)
        
        
        token = np.zeros(self.bus.token_dim, dtype='float32')
        if len(ranges) >= self.bus.token_dim:
            indices = np.linspace(0, len(ranges) - 1, self.bus.token_dim, dtype=int)
            token = ranges[indices] / max(raw_data.range_max, 1.0)
        else:
            token[:len(ranges)] = ranges / max(raw_data.range_max, 1.0)
            
        return token
