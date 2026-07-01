from omnitrain.robot_registry import SensorSpec
from typing import Any
from typing import Any, Optional
import abc
import logging
import multiprocessing
import numpy as np
import os
import queue
import time



class ModalityPlugin(abc.ABC):
    """
    Base class for all OmniTrain sensor plugins.
    Implements the lifecycle: Read -> Encode -> Publish.
    """

    def __init__(self, bus: Any, modal_id: str, frequency_hz: float, write_ptr: Optional[Any] = None, **kwargs):
        self.bus = bus
        self.modal_id = modal_id
        self.frequency_hz = frequency_hz
        self.write_ptr = write_ptr

    @abc.abstractmethod
    def read_raw_data(self) -> Any:
        pass

    @abc.abstractmethod
    def encode(self, raw_data: Any) -> np.ndarray:
        pass

    @classmethod
    def from_sensor_spec(cls, spec: SensorSpec, bus: Any, write_ptr: Optional[Any] = None) -> 'ModalityPlugin':
        """Factory method to create a generic plugin from a SensorSpec."""
        # For a real system, you'd map spec.type to specific subclasses.
        # Here we return a Dummy plugin that respects the spec's shape.
        return DummyLidarPlugin(bus=bus, modal_id=spec.id, frequency_hz=spec.hz, write_ptr=write_ptr)

    def run(self) -> None:
        period = 1.0 / self.frequency_hz
        while True:
            start = time.perf_counter()
            try:
                raw = self.read_raw_data()
                if raw is not None:
                    tokens = self.encode(raw)
                    # Unified: Always publish with write_ptr (circular buffer atomicity)
                    self.bus.publish(tokens, time.time(), self.modal_id, self.write_ptr)
            except Exception as e:
                print(f"[Plugin:{self.modal_id}] Fault: {e}")

            elapsed = time.perf_counter() - start
            time.sleep(max(0, period - elapsed))


class DummyLidarPlugin(ModalityPlugin):
    """
    Simulated 360 Lidar for testing/benchmarking.
    Generates 512-dim tokens of random ranges.
    """

    def read_raw_data(self) -> Any:
        return np.random.rand(512).astype(np.float32)

    def encode(self, raw_data: Any) -> np.ndarray:
        return raw_data




class CSVModalityPlugin(ModalityPlugin):
    """Plugin that ingests tabular data from a CSV file as sensor tokens."""

    def __init__(self, bus, modal_id, frequency_hz, write_ptr, csv_path, feature_cols=None):
        super().__init__(bus, modal_id, frequency_hz, write_ptr=write_ptr)
        import csv
        self.rows = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            try:
                self.header = next(reader)
            except StopIteration:
                self.header = []
            for row in reader:
                if row:
                    self.rows.append(row)
                    
        self.idx = 0
        self.feature_indices = []
        if feature_cols:
            for col in feature_cols:
                if col in self.header:
                    self.feature_indices.append(self.header.index(col))
        else:
            self.feature_indices = list(range(len(self.header)))
            if self.header and self.header[0] == 'timestamp':
                self.feature_indices = self.feature_indices[1:]

    def read_raw_data(self) -> Any:
        if not self.rows:
            return None
        row_str = self.rows[self.idx % len(self.rows)]
        self.idx += 1
        
        vals = []
        for i in self.feature_indices:
            if i < len(row_str):
                try:
                    vals.append(float(row_str[i]))
                except ValueError:
                    vals.append(0.0)
        return np.array(vals, dtype='float32')

    def encode(self, raw_data: Any) -> np.ndarray:
        base = np.zeros(512, dtype='float32')
        base[:min(512, len(raw_data))] = raw_data[:512]
        return base


class ImageFolderPlugin(ModalityPlugin):
    """Plugin that ingests images from a directory as flattened sensor tokens."""

    def __init__(self, bus, modal_id, frequency_hz, write_ptr, img_dir):
        super().__init__(bus, modal_id, frequency_hz, write_ptr=write_ptr)
        self.img_dir = img_dir
        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))]
        self.idx = 0

    def read_raw_data(self) -> Any:
        # Lazy cv2 import for robust file reading
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required: pip install opencv-python")

        if not self.images:
            return None
        path = os.path.join(self.img_dir, self.images[self.idx % len(self.images)])
        self.idx += 1

        img = cv2.imread(path)
        if img is None:
            print(f"[ImagePlugin:{self.modal_id}] WARN: Could not load {path}")
        return img

    def encode(self, raw_data: Any) -> np.ndarray:
        if raw_data is None:
            return np.zeros(512, dtype='float32')
        import cv2
        resized = cv2.resize(raw_data, (16, 32)).flatten()[:512]
        token = np.zeros(512, dtype='float32')
        token[:len(resized)] = resized
        return token



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
