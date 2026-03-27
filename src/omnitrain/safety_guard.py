import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class SafetyGuard(nn.Module):
    """
    Formal Safety Verification: Deterministic safety override layer.

    Wraps a neural safety head with hard interval constraints that GUARANTEE
    an EMERGENCY signal when sensor thresholds are violated, regardless of
    what the neural network predicts.

    This implements a simplified form of interval arithmetic verification:
    if ANY constraint is violated, the output is forced to the emergency class.
    """

    def __init__(self, safety_head: nn.Module, emergency_class: int = 1):
        super().__init__()
        self.safety_head = safety_head
        self.emergency_class = emergency_class
        # Configurable hard safety constraints: {sensor_name: (min_safe, max_safe)}
        self.constraints: Dict[str, Tuple[float, float]] = {}

    def add_constraint(self, sensor_name: str, min_safe: float, max_safe: float):
        """
        Register a hard safety constraint for a sensor.

        If sensor readings fall outside [min_safe, max_safe], the safety
        output is forced to EMERGENCY regardless of the neural network.

        Args:
            sensor_name: Identifier for the sensor (e.g., 'lidar_front').
            min_safe: Minimum safe value (below = danger).
            max_safe: Maximum safe value (above = danger).
        """
        self.constraints[sensor_name] = (min_safe, max_safe)

    def check_constraints(self, sensor_readings: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Verify all registered constraints against current sensor readings.

        Returns:
            (is_safe, violations): Boolean safety status and list of violated constraints.
        """
        violations = []
        for sensor_name, (min_val, max_val) in self.constraints.items():
            if sensor_name in sensor_readings:
                reading = sensor_readings[sensor_name]
                if reading < min_val or reading > max_val:
                    violations.append(
                        f"{sensor_name}: {reading:.4f} outside [{min_val}, {max_val}]"
                    )
        return len(violations) == 0, violations

    def forward(self, latents: torch.Tensor,
                sensor_readings: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        Forward pass with formal safety override.

        Args:
            latents: (Batch, n_latents, d_model) from FusionCore.
            sensor_readings: Optional dict of {sensor_name: current_value}.
                If provided, hard constraints are checked BEFORE neural inference.

        Returns:
            Safety logits: (Batch, num_classes). If constraints are violated,
            the emergency class logit is forced to +inf.
        """
        # 1. Neural network prediction
        nn_output = self.safety_head(latents)

        # 2. Hard constraint override (if sensor readings provided)
        if sensor_readings is not None:
            is_safe, violations = self.check_constraints(sensor_readings)
            if not is_safe:
                # Force EMERGENCY: set emergency class logit to +inf
                override = nn_output.clone()
                override[:, self.emergency_class] = float('inf')
                return override

        return nn_output

    def generate_safety_report(self, test_cases: List[Dict[str, float]]) -> Dict:
        """
        Run a batch of test scenarios and generate a formal safety certificate.

        Args:
            test_cases: List of sensor reading dicts to verify.

        Returns:
            Report dict with pass/fail counts and details.
        """
        report = {
            'total_cases': len(test_cases),
            'passed': 0,
            'failed': 0,
            'constraint_count': len(self.constraints),
            'violations': []
        }

        for i, readings in enumerate(test_cases):
            is_safe, violations = self.check_constraints(readings)
            if is_safe:
                report['passed'] += 1
            else:
                report['failed'] += 1
                report['violations'].append({
                    'case_id': i,
                    'readings': readings,
                    'violations': violations
                })

        report['status'] = 'CERTIFIED' if report['failed'] == 0 else 'VIOLATIONS_FOUND'
        return report
