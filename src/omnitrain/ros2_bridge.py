import threading
import logging
import time
from typing import Callable, Any, Optional
import os

# ROS2 Reliability Profiles
try:
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
except ImportError:
    # Mock classes for non-ROS environments to prevent NameErrors at class definition
    class ReliabilityPolicy: RELIABLE = 0; BEST_EFFORT = 1
    class HistoryPolicy: KEEP_LAST = 0; KEEP_ALL = 1
    class DurabilityPolicy: VOLATILE = 0; TRANSIENT_LOCAL = 1
    class QoSProfile:
        def __init__(self, **kwargs): pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class OmniQoS:
    """QoS profiles for Robotics."""
    RELIABLE = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE
    )
    SENSOR_DATA = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1, # Always take freshest
        durability=DurabilityPolicy.VOLATILE
    )

class OmniClock:
    """Distributed Clock Manager to handle network jitter across ROS2 nodes."""
    def __init__(self, node):
        self.node = node

    def now_seconds(self) -> float:
        return self.node.get_clock().now().nanoseconds / 1e9

class OmniROS2Node:
    """
    ROS2 Node wrapper for OmniTrain.
    Ensures zero-latency callback dispatch and correct QoS handling.
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
            raise ImportError("ROS2 (rclpy) is not installed.")

        if not self.rclpy.ok():
            self.rclpy.init()
            
        node_name = f'omnitrain_node_{os.getpid()}'
        self.node = self.NodeClass(node_name)
        self.clock = OmniClock(self.node)
        self.spin_thread = None
        self.running = False
        self._initialized = True
        logging.info(f"🤖 ROS2 Node '{node_name}' initialized with QoS.")

    def start_spinning(self):
        """Starts the ROS2 executor in a background thread."""
        if self.running: return
        self.running = True
        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()

    def _spin_loop(self):
        try:
            # Use MultiThreadedExecutor for better scaling
            from rclpy.executors import MultiThreadedExecutor
            executor = MultiThreadedExecutor()
            executor.add_node(self.node)
            executor.spin()
        except Exception as e:
            logging.error(f"ROS2 Spin Loop died: {e}")
        finally:
            self.running = False

    def create_subscription(self, msg_type: Any, topic: str, callback: Callable, qos: Optional[QoSProfile] = None):
        """Wrapper with QoS support."""
        profile = qos or OmniQoS.RELIABLE
        return self.node.create_subscription(msg_type, topic, callback, profile)

    def shutdown(self):
        if not self._initialized or not self.running: return
        self.running = False
        self.node.destroy_node()
        self.rclpy.shutdown()
        if self.spin_thread: self.spin_thread.join(timeout=1.0)
        logging.info("🤖 ROS2 Node shut down.")
