import numpy as np
import logging
import queue
from typing import Any
from .plugins import ModalityPlugin

class ROS2BasePlugin(ModalityPlugin):
    """
    Base class for all ROS2 Modality Plugins.
    Manages connection to the OmniROS2Node and handles safe topic subscription.
    """
    def __init__(self, bus, modal_id, frequency_hz, write_ptr, topic_name: str, msg_type: Any):
        super().__init__(bus, modal_id, frequency_hz, write_ptr=write_ptr)
        self.topic_name = topic_name
        self.msg_type = msg_type
        self.msg_queue = queue.Queue(maxsize=5) # Buffer for incoming ROS2 messages
        
        try:
            from .ros2_bridge import OmniROS2Node
            self.ros_node = OmniROS2Node()
        except ImportError as e:
            logging.error(f"[{modal_id}] Initialization Failed: {e}")
            raise

        self.ros_node.create_subscription(
            self.msg_type,
            self.topic_name,
            self._ros_callback,
            qos_profile=10
        )
        
        # Ensure the node is spinning so callbacks fire
        self.ros_node.start_spinning()
        logging.info(f"[{modal_id}] Subscribed to ROS2 Topic: {topic_name}")

    def _ros_callback(self, msg):
        """Asynchronous callback from ROS2 executor. Puts msg in thread-safe queue."""
        try:
            # Keep only the freshest message
            if self.msg_queue.full():
                self.msg_queue.get_nowait() 
            self.msg_queue.put_nowait(msg)
        except queue.Full:
            pass

    def read_raw_data(self) -> Any:
        """Called by the TokenBus worker loop at 'frequency_hz'."""
        try:
            # Non-blocking get. If no new message, return None (bus will handle skip)
            return self.msg_queue.get_nowait()
        except queue.Empty:
            return None


class ROS2CameraPlugin(ROS2BasePlugin):
    """
    Subscribes to sensor_msgs/Image topics.
    Encodes raw pixels into TokenBus-compatible latents.
    """
    def __init__(self, bus, modal_id, frequency_hz, write_ptr, topic_name: str = "/camera/image_raw"):
        try:
            from sensor_msgs.msg import Image
        except ImportError:
            raise ImportError("sensor_msgs is not installed. Run: sudo apt install ros-<distro>-sensor-msgs")
            
        super().__init__(bus, modal_id, frequency_hz, write_ptr, topic_name, msg_type=Image)

    def encode(self, raw_data: Any) -> np.ndarray:
        """
        Convert ROS2 sensor_msgs/Image to a 512-dim token.
        In production, this would use a proper visual backbone (e.g., MobileNet).
        Here we use a fast resizing projection.
        """
        if raw_data is None:
            return np.zeros(512, dtype='float32')
            
        try:
            import cv2
            # Simple conversion from ROS2 Image without cv_bridge dependency
            img_np = np.ndarray(
                shape=(raw_data.height, raw_data.width, 3), 
                dtype=np.uint8, 
                buffer=raw_data.data
            )
            # Resize and flatten to 512 parameters
            resized = cv2.resize(img_np, (16, 32)).flatten()[:512]
        except ImportError:
            # Fallback if cv2 is not installed
            resized = np.zeros(512, dtype='float32')

        token = np.zeros(512, dtype='float32')
        token[:len(resized)] = resized
        return token


class ROS2LidarPlugin(ROS2BasePlugin):
    """
    Subscribes to sensor_msgs/LaserScan topics.
    Encodes distance ranges into TokenBus-compatible latents.
    """
    def __init__(self, bus, modal_id, frequency_hz, write_ptr, topic_name: str = "/scan"):
        try:
            from sensor_msgs.msg import LaserScan
        except ImportError:
            raise ImportError("sensor_msgs is not installed. Run: sudo apt install ros-<distro>-sensor-msgs")
            
        super().__init__(bus, modal_id, frequency_hz, write_ptr, topic_name, msg_type=LaserScan)

    def encode(self, raw_data: Any) -> np.ndarray:
        """
        Convert ROS2 sensor_msgs/LaserScan to a 512-dim token.
        """
        if raw_data is None:
            return np.zeros(512, dtype='float32')
            
        ranges = np.array(raw_data.ranges, dtype='float32')
        
        # Handle infinities/NaNs standard in LiDAR scans
        ranges = np.nan_to_num(ranges, posinf=raw_data.range_max, neginf=0.0)
        
        # Subsample or pad to exactly 512 dimensions
        token = np.zeros(512, dtype='float32')
        if len(ranges) >= 512:
            indices = np.linspace(0, len(ranges) - 1, 512, dtype=int)
            token = ranges[indices]
        else:
            token[:len(ranges)] = ranges
            
        return token
