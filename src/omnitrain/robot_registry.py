import abc
from dataclasses import dataclass
from typing import Dict, Any, List, Type

@dataclass
class SensorSpec:
    id: str
    dim: int
    hz: float
    type: str = "sensor"
    range: List[float] = None
    noise: bool = False
    
    def __post_init__(self):
        if self.range is None:
            self.range = [0.0, 1.0]

class OmniBaseRobot(abc.ABC):
    """
    Abstract Base Class for all OmniTrain registered robots.
    """
    @abc.abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset the robot to an initial state and return initial sensor readings."""
        pass

    @abc.abstractmethod
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an action and return the next sensor readings."""
        pass

    @abc.abstractmethod
    def get_sensor_specs(self) -> List[SensorSpec]:
        """Return the specifications of the robot's sensors."""
        pass

    def close(self):
        """Cleanup resources."""
        pass

class RobotRegistry:
    """
    Singleton registry for dynamically discovering and instantiating robots.
    """
    _registry: Dict[str, Type[OmniBaseRobot]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(robot_cls: Type[OmniBaseRobot]):
            cls._registry[name] = robot_cls
            return robot_cls
        return decorator

    @classmethod
    def make(cls, name: str, **kwargs) -> OmniBaseRobot:
        if name not in cls._registry:
            raise ValueError(f"Robot '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list(cls) -> List[str]:
        return list(cls._registry.keys())

def register_robot(name: str):
    """Decorator to register a robot class."""
    return RobotRegistry.register(name)

def auto_config(robot: OmniBaseRobot) -> Dict[str, Any]:
    """
    Generate an OmniTrain 'inputs' config block automatically from a robot's SensorSpecs.
    """
    inputs_cfg = []
    for spec in robot.get_sensor_specs():
        inputs_cfg.append({
            'id': spec.id,
            'type': spec.type,
            'hz': spec.hz,
            'dim': spec.dim,
            'range': spec.range,
            'noise': spec.noise
        })
    return inputs_cfg
