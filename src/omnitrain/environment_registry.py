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

class OmniEnvironment(abc.ABC):
    """
    Abstract Base Class for all OmniTrain registered environments.
    """
    @abc.abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset the environment to an initial state and return initial sensor readings."""
        pass

    @abc.abstractmethod
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an action and return the next sensor readings."""
        pass

    @abc.abstractmethod
    def get_sensor_specs(self) -> List[SensorSpec]:
        """Return the specifications of the environment's sensors."""
        pass

    def close(self):
        """Cleanup resources."""
        pass

class EnvironmentRegistry:
    """
    Singleton registry for dynamically discovering and instantiating environments.
    """
    _registry: Dict[str, Type[OmniEnvironment]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(env_cls: Type[OmniEnvironment]):
            cls._registry[name] = env_cls
            return env_cls
        return decorator

    @classmethod
    def make(cls, name: str, **kwargs) -> OmniEnvironment:
        if name not in cls._registry:
            raise ValueError(f"Environment '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list(cls) -> List[str]:
        return list(cls._registry.keys())

def register_environment(name: str):
    """Decorator to register an environment class."""
    return EnvironmentRegistry.register(name)

def auto_config(environment: OmniEnvironment) -> Dict[str, Any]:
    """
    Generate an OmniTrain 'inputs' config block automatically from an environment's SensorSpecs.
    """
    inputs_cfg = []
    for spec in environment.get_sensor_specs():
        inputs_cfg.append({
            'id': spec.id,
            'type': spec.type,
            'hz': spec.hz,
            'dim': spec.dim,
            'range': spec.range,
            'noise': spec.noise
        })
    return inputs_cfg
