import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class SafetyGuard(nn.Module):
    """
    Tier 1 Hardware Failsafe — Vectorized hard-limit safety layer.

    This module is the lowest-level, zero-latency safety layer in OmniTrain.
    It performs GPU-accelerated bound checks on raw sensor readings and
    short-circuits neural inference for any violated sample in the batch.

    For the full 3-tier safety system (hardware + CBF + soft penalty),
    see OmniShieldGuard in omni_shield.py.
    """

    def __init__(self, safety_head: nn.Module, num_sensors: int = 0, emergency_class: int = 1):
        super().__init__()
        self.safety_head = safety_head
        self.emergency_class = emergency_class
        self.num_sensors = num_sensors
        self.constraints = {}

        # Vectorized constraints (initialized to pass-through)
        if num_sensors > 0:
            self.register_buffer('min_limits', torch.full((num_sensors,), -float('inf')))
            self.register_buffer('max_limits', torch.full((num_sensors,), float('inf')))

    def add_constraint(self, sensor_id: str, min_safe: float = -float('inf'), max_safe: float = float('inf')):
        """Adds a constraint and rebuilds buffers."""
        self.constraints[sensor_id] = {'min': min_safe, 'max': max_safe}
        self.num_sensors = len(self.constraints)
        
        # Rebuild tensors
        mins = torch.tensor([c['min'] for c in self.constraints.values()], dtype=torch.float32)
        maxs = torch.tensor([c['max'] for c in self.constraints.values()], dtype=torch.float32)
        
        # Use register_buffer with overwrite logic
        if 'min_limits' in self._buffers:
            self._buffers['min_limits'] = mins
            self._buffers['max_limits'] = maxs
        else:
            self.register_buffer('min_limits', mins)
            self.register_buffer('max_limits', maxs)

    def check_constraints(self, sensor_readings: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Non-vectorized check for single-sample validation (used in tests)."""
        violations = []
        for s_id, val in sensor_readings.items():
            if s_id in self.constraints:
                c = self.constraints[s_id]
                if val < c['min'] or val > c['max']:
                    violations.append(s_id)
        return len(violations) == 0, violations

    def generate_safety_report(self, test_cases: List[Dict[str, float]]) -> Dict:
        """Generates a summary report for a set of test cases."""
        passed = 0
        failed = 0
        for case in test_cases:
            ok, _ = self.check_constraints(case)
            if ok:
                passed += 1
            else:
                failed += 1
        return {
            'total_cases': len(test_cases),
            'passed': passed,
            'failed': failed
        }

    def set_limits(self, min_vals: torch.Tensor, max_vals: torch.Tensor):
        """Set limits for all sensors at once using tensors."""
        self.min_limits.copy_(min_vals)
        self.max_limits.copy_(max_vals)

    @classmethod
    def from_config(cls, config: dict, safety_head: nn.Module) -> 'SafetyGuard':
        """
        Factory: Build a SafetyGuard from config.yaml safety_constraints.
        """
        constraints = config.get('safety_constraints', [])
        guard = cls(safety_head, num_sensors=len(constraints))

        if constraints:
            mins = torch.tensor([c['min'] for c in constraints], dtype=torch.float32)
            maxs = torch.tensor([c['max'] for c in constraints], dtype=torch.float32)
            guard.set_limits(mins, maxs)
            # Fill the constraints dict for reporting
            for i, c in enumerate(constraints):
                s_id = c.get('sensor', f"sensor_{i}")
                guard.constraints[s_id] = {'min': c['min'], 'max': c['max']}

        return guard

    def check_batch(self, sensor_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized constraint check on a batch of sensor readings.
        """
        if self.min_limits is None:
            return torch.zeros(sensor_batch.shape[0], dtype=torch.bool, device=sensor_batch.device), None
            
        under = sensor_batch < self.min_limits
        over = sensor_batch > self.max_limits
        violations = under | over
        is_violated = violations.any(dim=1)
        return is_violated, violations

    def forward(self, latents: torch.Tensor,
                sensor_batch: Optional[torch.Tensor] = None,
                sensor_readings: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        Forward pass with hardware failsafe override.
        Supports both vectorized tensor input and single-sample dict input (for tests).
        """
        batch_size = latents.shape[0]
        device = latents.device

        # 1. Determine violations
        is_violated = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        if sensor_batch is not None:
            is_violated, _ = self.check_batch(sensor_batch)
        elif sensor_readings is not None:
            # Single sample override from dict
            ok, _ = self.check_constraints(sensor_readings)
            if not ok:
                is_violated = torch.ones(batch_size, dtype=torch.bool, device=device)

        # 2. Determine output size
        out_features = getattr(self.safety_head, 'out_features', None)
        if out_features is None:
            for m in reversed(list(self.safety_head.modules())):
                if isinstance(m, nn.Linear):
                    out_features = m.out_features
                    break
            if out_features is None:
                out_features = 2

        final_logits = torch.zeros((batch_size, out_features), device=device)

        # 3. Short-circuit
        safe_idx = (~is_violated).nonzero(as_tuple=True)[0]
        unsafe_idx = is_violated.nonzero(as_tuple=True)[0]

        if len(safe_idx) > 0:
            final_logits[safe_idx] = self.safety_head(latents[safe_idx])

        if len(unsafe_idx) > 0:
            # Force emergency class to a very high value (infinity for practical purposes)
            final_logits[unsafe_idx, self.emergency_class] = 1e6

        return final_logits
