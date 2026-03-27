import numpy as np
import time
import threading
from .plugins import ModalityPlugin

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

class ROS2ModalityPlugin(ModalityPlugin):
    """
    Standard ROS 2 Bridge for OmniTrain.
    Subscribes to a ROS 2 topic and publishes it to the Industrial C++ Bus.
    """
    def __init__(self, bus, modal_id, frequency_hz, write_ptr, topic_name, msg_type_name="Float32MultiArray"):
        super().__init__(bus, modal_id, frequency_hz, write_ptr)
        self.topic_name = topic_name
        self.msg_type_name = msg_type_name
        self.latest_data = None
        self._lock = threading.Lock()
        
        if not ROS2_AVAILABLE:
            print(f"[ROS2Plugin] ERR: rclpy not found. Install ROS 2 Humble/Iron first.")
            return

        # Initialize ROS 2 Node in a separate thread context
        threading.Thread(target=self._ros_thread, daemon=True).start()

    def _ros_thread(self):
        rclpy.init()
        self.node = Node(f"omni_ingest_{self.modal_id}")
        self.node.create_subscription(
            Float32MultiArray,
            self.topic_name,
            self._msg_callback,
            10
        )
        print(f"[ROS2Plugin] Subscribed to {self.topic_name}")
        rclpy.spin(self.node)

    def _msg_callback(self, msg):
        with self._lock:
            self.latest_data = np.array(msg.data, dtype=np.float32)

    def read_raw_data(self):
        with self._lock:
            return self.latest_data

    def encode(self, raw):
        if raw is None:
            return None
        encoded = np.zeros(512, dtype=np.float32)
        size = min(len(raw), 512)
        encoded[:size] = raw[:size]
        return encoded
