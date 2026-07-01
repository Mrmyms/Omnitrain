import time
import abc
import numpy as np
import multiprocessing
from typing import Any, Optional
from omnitrain.robot_registry import SensorSpec


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
