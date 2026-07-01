import numpy as np
import torch
from typing import Dict, Any, List
from omnitrain.robot_registry import OmniBaseRobot, SensorSpec, register_robot

@register_robot(name="sim_delivery_bot")
class SimDeliveryBot(OmniBaseRobot):
    """
    Reference implementation of a simulated SafeDelivery_Robot.
    """
    def __init__(self):
        self._step = 0
        self.battery = 1.0
        
    def reset(self) -> Dict[str, Any]:
        self._step = 0
        self.battery = 1.0
        return self._get_obs()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._step += 1
        self.battery = max(0.0, self.battery - 0.001)
        
        # Simulate moving
        # ...
        
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        # Generate simulated sensor readings
        lidar = np.random.uniform(2.0, 10.0, size=(32,)).astype(np.float32)
        
        # Random obstacle occasionally
        if np.random.rand() < 0.1:
            lidar[10:20] = 0.5 
            
        vision = np.random.randn(128).astype(np.float32)
        
        return {
            "lidar_front": torch.tensor(lidar),
            "vision_embed": torch.tensor(vision),
            # battery_state: normalized [0,1] where 0=empty, 1=full
            "battery_state": torch.tensor([self.battery])
        }

    def get_sensor_specs(self) -> List[SensorSpec]:
        return [
            SensorSpec(id="lidar_front", dim=32, hz=10.0, type="sensor", range=[0.0, 10.0], noise=True),
            SensorSpec(id="vision_embed", dim=128, hz=5.0, type="vision", range=[-1.0, 1.0], noise=False),
            SensorSpec(id="battery_state", dim=1, hz=1.0, type="boolean", range=[0.0, 1.0], noise=False)
        ]
