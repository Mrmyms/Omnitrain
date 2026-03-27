import time
import numpy as np
from .token_bus import TokenBus

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

class OmniROS2Bridge(Node):
    """
    ROS 2 Bridge for emitting OmniTrain predictions back to the robotics ecosystem.
    """
    def __init__(self, session_id="omni_default", output_topic="/omni/predictions"):
        super().__init__("omni_ros2_bridge")
        self.publisher_ = self.create_publisher(Float32MultiArray, output_topic, 10)
        self.bus = TokenBus(session_id=session_id, create=False)
        self.timer = self.create_timer(0.01, self.timer_callback) # 100Hz emission
        print(f"[OmniROS2] Bridge Active. Publishing to {output_topic}")

    def timer_callback(self):
        now = time.time()
        tokens = self.bus.get_window(now - 0.1, now)
        if not tokens: return
        msg = Float32MultiArray()
        msg.data = tokens[-1]['data'].tolist()
        self.publisher_.publish(msg)

def main():
    if not ROS2_AVAILABLE:
        print("✘ ROS 2 (rclpy) not found.")
        return
    rclpy.init()
    bridge = OmniROS2Bridge()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    bridge.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
