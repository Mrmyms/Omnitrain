import threading
import logging
from typing import Callable, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class OmniROS2Node:
    """
    Singleton-pattern ROS2 Node wrapper for OmniTrain.
    Ensures that rclpy is initialized exactly once per process and handles 
    asynchronous spinning in a background thread to prevent deadlocks
    with Python's multiprocessing.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(OmniROS2Node, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        try:
            import rclpy
            from rclpy.node import Node
            self.rclpy = rclpy
            self.NodeClass = Node
        except ImportError:
            raise ImportError("ROS2 (rclpy) is not installed. OmniROS2Node cannot start. "
                              "Please source your ROS2 workspace or install the rclpy package.")

        self.rclpy.init()
        self.node = self.NodeClass('omnitrain_fusion_hub')
        self.spin_thread = None
        self.running = False
        self._initialized = True
        logging.info("🤖 ROS2 Node 'omnitrain_fusion_hub' initialized successfully.")

    def start_spinning(self):
        """Starts the ROS2 executor in a background thread."""
        if self.running:
            return

        self.running = True
        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()
        logging.info("🤖 ROS2 Executor spinning up...")

    def _spin_loop(self):
        try:
            self.rclpy.spin(self.node)
        except Exception as e:
            logging.error(f"ROS2 Spin Loop died: {e}")
        finally:
            self.running = False

    def create_subscription(self, msg_type: Any, topic: str, callback: Callable, qos_profile: int = 10):
        """Wrapper around node.create_subscription."""
        return self.node.create_subscription(msg_type, topic, callback, qos_profile)

    def shutdown(self):
        """Gracefully shuts down the ROS2 context."""
        if not self._initialized or not self.running:
            return

        self.running = False
        self.node.destroy_node()
        self.rclpy.shutdown()
        
        if self.spin_thread:
            self.spin_thread.join(timeout=2.0)
            
        logging.info("🤖 ROS2 Node shut down cleanly.")
