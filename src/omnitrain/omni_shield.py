import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────
#  Telemetry: Per-Step Safety Record
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ShieldTelemetry:
    """Immutable record of a single shield evaluation."""
    tier_activated: int = 0               # 0=none, 1=hardware, 2=cbf, 3=soft
    hw_violations: int = 0               # Count of hard-limit violations
    cbf_correction_norm: float = 0.0     # How much the CBF adjusted the action
    barrier_value: float = 0.0           # h(x) — positive is safe
    safety_margin: float = 0.0           # Distance to constraint boundary


# ─────────────────────────────────────────────────────────────────────
#  Neural Barrier Certificate  h(x) : R^n → R
# ─────────────────────────────────────────────────────────────────────

class NeuralBarrier(nn.Module):
    """
    Learned Control Barrier Function (ICNN implementation).
    
    Guarantees that h(x) is convex with respect to x by:
    1. Using non-negative weights for all layers except the first.
    2. Using convex, non-decreasing activation functions (Softplus).
    3. Adding pass-through connections from input to each layer.
    """

    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        
        # Layer 0: Initial projection
        self.w0 = nn.Linear(state_dim, hidden)
        
        # Layer 1: ICNN step
        self.y1_w = nn.Linear(hidden, hidden, bias=True)  # Must be non-negative
        self.x1_w = nn.Linear(state_dim, hidden, bias=False)
        
        # Final Layer
        self.y2_w = nn.Linear(hidden, 1, bias=True)  # Must be non-negative
        self.x2_w = nn.Linear(state_dim, 1, bias=False)
        
        self.softplus = nn.Softplus(beta=5)
        
        # Initialize weights to be positive
        with torch.no_grad():
            nn.init.uniform_(self.y1_w.weight, 0.01, 0.5)
            nn.init.uniform_(self.y2_w.weight, 0.01, 0.5)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        ICNN Forward Pass.
        """
        # Ensure non-negativity of relevant weights during forward
        # (Alternatively, can use constraints/clipping during training)
        with torch.no_grad():
            self.y1_w.weight.clamp_(min=0)
            self.y2_w.weight.clamp_(min=0)
            
        z1 = self.softplus(self.w0(state))
        
        z2 = self.softplus(self.y1_w(z1) + self.x1_w(state))
        
        h = self.y2_w(z2) + self.x2_w(state)
        
        return h.squeeze(-1)


# ─────────────────────────────────────────────────────────────────────
#  State Extractor: Latents → Physical State
# ─────────────────────────────────────────────────────────────────────

class StateExtractor(nn.Module):
    """
    Bridges FusionCore latents (B, N, D) to a physical state vector (B, state_dim).

    This is the missing link: FusionCore produces abstract latent tokens,
    but the CBF needs a concrete physical state (distance, velocity, etc.).
    The extractor learns to decode the relevant physical quantities.
    """

    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # (B, N, D) → (B, D)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, state_dim),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: (Batch, N, d_model) from FusionCore.
        Returns:
            state: (Batch, state_dim) estimated physical state.
        """
        # Pool across latent tokens
        pooled = self.pool(latents.transpose(1, 2)).squeeze(-1)  # (B, D)
        return self.decoder(pooled)


# ─────────────────────────────────────────────────────────────────────
#  Learned Dynamics Model: x_next = f(x, u)
# ─────────────────────────────────────────────────────────────────────

class ResidualDynamics(nn.Module):
    """
    Learns the residual dynamics: x_next = x + dt * f_theta(x, u).

    Residual formulation ensures stability: if the network outputs zero,
    the state doesn't change (identity dynamics as the safe default).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128, dt: float = 0.05):
        super().__init__()
        self.dt = dt
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.Softplus(beta=5),
            nn.Linear(hidden, hidden),
            nn.Softplus(beta=5),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:  (Batch, state_dim)
            action: (Batch, action_dim)
        Returns:
            x_next: (Batch, state_dim)
        """
        xu = torch.cat([state, action], dim=-1)
        return state + self.dt * self.net(xu)


# ─────────────────────────────────────────────────────────────────────
#  OmniShield v2: 3-Tier Unified Safety System
# ─────────────────────────────────────────────────────────────────────

class OmniShieldGuard(nn.Module):
    """
    OmniShield v2: Industrial 3-Tier Safety System.

    Tier 1 — HARDWARE FAILSAFE
        Vectorized hard-limit checks on raw sensor tensors.
        If violated → immediate emergency output, zero NN cost.

    Tier 2 — CBF PROJECTION (Differentiable)
        Learned barrier function h(x) with analytical QP projection.
        Minimally corrects the policy's action to maintain h(x) ≥ 0.
        Full gradient flow back to the policy for end-to-end training.

    Tier 3 — SOFT PENALTY (Training Signal)
        Auxiliary barrier loss encourages the policy to self-correct.
        The policy learns to stay away from the boundary on its own.

    Integration:
        - Consumes FusionCore latents (B, N, D) via StateExtractor.
        - Wraps any action head (RegressionHead / ClassificationHead).
        - Configurable from config.yaml safety_constraints section.
        - Serializable with OmniExporter.
    """

    def __init__(
        self,
        action_head: nn.Module,
        d_model: int,
        state_dim: int,
        action_dim: int,
        num_hw_sensors: int = 0,
        alpha: float = 0.9,
        barrier_loss_weight: float = 0.1,
        dt: float = 0.05,
    ):
        super().__init__()

        # ── Core modules ──
        self.action_head = action_head
        self.state_extractor = StateExtractor(d_model, state_dim)
        self.barrier = NeuralBarrier(state_dim)
        self.dynamics = ResidualDynamics(state_dim, action_dim, dt=dt)

        # ── Config ──
        self.alpha = alpha
        self.barrier_loss_weight = barrier_loss_weight
        self.state_dim = state_dim
        self.action_dim = action_dim

        # ── Tier 1: Hardware limits (vectorized) ──
        self.num_hw_sensors = num_hw_sensors
        if num_hw_sensors > 0:
            self.register_buffer('hw_min', torch.full((num_hw_sensors,), -float('inf')))
            self.register_buffer('hw_max', torch.full((num_hw_sensors,), float('inf')))

        # ── Telemetry ──
        self._last_telemetry: Optional[ShieldTelemetry] = None

    # ─── Configuration API ───────────────────────────────────────────

    def set_hw_limits(self, min_vals: torch.Tensor, max_vals: torch.Tensor):
        """Set hardware failsafe limits for all sensors at once."""
        assert self.num_hw_sensors > 0, "Shield was created with num_hw_sensors=0"
        self.hw_min.copy_(min_vals)
        self.hw_max.copy_(max_vals)

    @classmethod
    def from_config(cls, config: dict, action_head: nn.Module, d_model: int) -> 'OmniShieldGuard':
        """
        Factory: Build an OmniShield from a config.yaml dict.

        Expected config structure:
            safety_constraints:
              - sensor: "lidar_front"
                min: 0.15
                max: 10.0
            model:
              d_model: 256
            heads:
              - id: "drive_control"
                output_dim: 2
        """
        constraints = config.get('safety_constraints', [])
        heads_cfg = config.get('heads', [])

        # Infer action_dim from the drive_control head config
        action_dim = 2  # default
        for h in heads_cfg:
            if h.get('type') == 'regression':
                action_dim = h.get('output_dim', 2)
                break

        # Infer state_dim from input sensors
        inputs = config.get('inputs', [])
        state_dim = max(len(inputs), 4)  # At least 4 for reasonable state space

        num_hw = len(constraints)

        shield = cls(
            action_head=action_head,
            d_model=d_model,
            state_dim=state_dim,
            action_dim=action_dim,
            num_hw_sensors=num_hw,
            alpha=config.get('shield_alpha', 0.9),
            barrier_loss_weight=config.get('barrier_loss_weight', 0.1),
        )

        # Set hardware limits from config
        if num_hw > 0:
            mins = torch.tensor([c['min'] for c in constraints], dtype=torch.float32)
            maxs = torch.tensor([c['max'] for c in constraints], dtype=torch.float32)
            shield.set_hw_limits(mins, maxs)

        return shield

    # ─── Tier 1: Hardware Failsafe ───────────────────────────────────

    def _check_hw_limits(self, sensor_batch: Optional[torch.Tensor]) -> Tuple[torch.Tensor, int]:
        """
        Vectorized hard-limit check.

        Args:
            sensor_batch: (Batch, num_hw_sensors) raw sensor readings.
        Returns:
            is_violated: (Batch,) bool tensor.
            violation_count: total violated constraints across batch.
        """
        if sensor_batch is None or self.num_hw_sensors == 0:
            return None, 0

        under = sensor_batch < self.hw_min
        over = sensor_batch > self.hw_max
        violations = under | over
        is_violated = violations.any(dim=1)
        return is_violated, violations.sum().item()

    # ─── Tier 2: CBF Projection ──────────────────────────────────────

    def _cbf_project(self, u_nn: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable CBF projection.

        Solves:  min ||u - u_nn||²  s.t.  h(f(x,u)) ≥ (1-α)h(x)

        Uses analytical half-space projection (closed-form QP for single
        linear constraint) for real-time performance.

        The key fix from v1: u_nn retains its gradient connection to the
        policy, so the policy learns from the shield's corrections.

        Args:
            u_nn:  (Batch, action_dim) raw policy output.
            state: (Batch, state_dim) estimated physical state.
        Returns:
            u_safe: (Batch, action_dim) projected safe action.
            h_x:   (Batch,) current barrier values.
        """
        # Current safety value
        h_x = self.barrier(state)

        # Forward simulate with current action to get future safety
        x_next = self.dynamics(state, u_nn)
        h_next = self.barrier(x_next)

        # CBF condition: Δh ≥ -α * h(x)  ⟺  h_next - h_x ≥ -α * h_x
        #                ⟺  h_next ≥ (1-α) * h_x
        cbf_violation = (1 - self.alpha) * h_x - h_next  # > 0 means violated

        # Compute Lie derivative: dh_next/du (how action affects future safety)
        if u_nn.requires_grad:
            # Training mode: full gradient flow back to the policy
            lg_h = torch.autograd.grad(
                h_next.sum(), u_nn,
                create_graph=self.training,
                retain_graph=True,
            )[0]
        else:
            # Inference mode: compute analytical derivative locally
            with torch.enable_grad():
                u_var = u_nn.detach().requires_grad_(True)
                x_next_var = self.dynamics(state, u_var)
                h_next_var = self.barrier(x_next_var)
                lg_h = torch.autograd.grad(h_next_var.sum(), u_var)[0]

        # Analytical QP projection onto the safe half-space
        lg_h_norm_sq = (lg_h * lg_h).sum(dim=1, keepdim=True) + 1e-8
        lam = F.relu(cbf_violation).unsqueeze(-1) / lg_h_norm_sq  # (Batch, 1)

        u_safe = u_nn + lam * lg_h  # Minimal correction

        return u_safe, h_x

    # ─── Tier 3: Barrier Loss ────────────────────────────────────────

    def barrier_loss(self, h_x: torch.Tensor) -> torch.Tensor:
        """
        Soft penalty that teaches the policy to stay inside the safe set.

        Uses a hinge-like loss: penalizes when h(x) is near or below zero.
        The policy gradually learns to avoid the boundary entirely.
        """
        # Penalize when barrier value is below a safety margin (0.1)
        margin = 0.1
        return self.barrier_loss_weight * F.relu(margin - h_x).mean()

    # ─── Unified Forward ─────────────────────────────────────────────

    def forward(
        self,
        latents: torch.Tensor,
        sensor_batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        3-Tier Safety Pipeline.

        Args:
            latents:      (Batch, N, d_model) from FusionCore.
            sensor_batch: (Batch, num_hw_sensors) optional raw sensor readings
                          for Tier 1 hardware failsafe.

        Returns:
            Dict with keys:
                'action':       (Batch, action_dim) safe action output.
                'barrier_loss': scalar, auxiliary training loss (Tier 3).
                'h_x':          (Batch,) barrier values for monitoring.
                'tier':         int, highest tier activated.
        """
        batch_size = latents.shape[0]
        device = latents.device
        telemetry = ShieldTelemetry()

        # ── Tier 1: Hardware Failsafe ──
        hw_violated, hw_count = self._check_hw_limits(sensor_batch)
        telemetry.hw_violations = hw_count

        if hw_violated is not None and hw_violated.all():
            # EVERY sample violated → full emergency, skip ALL neural compute
            telemetry.tier_activated = 1
            self._last_telemetry = telemetry
            emergency = torch.zeros(batch_size, self.action_dim, device=device)
            return {
                'action': emergency,
                'barrier_loss': torch.tensor(0.0, device=device),
                'h_x': torch.full((batch_size,), -1.0, device=device),
                'tier': 1,
            }

        # ── Get raw policy action (gradient flows through here) ──
        u_nn = self.action_head(latents)
        if u_nn.requires_grad:
            u_nn.retain_grad()

        # ── Extract physical state from latents ──
        state = self.state_extractor(latents)

        # ── Tier 2: CBF Projection ──
        u_safe, h_x = self._cbf_project(u_nn, state)
        correction_norm = (u_safe - u_nn).norm(dim=1).mean().item()
        telemetry.cbf_correction_norm = correction_norm
        telemetry.barrier_value = h_x.mean().item()
        telemetry.safety_margin = h_x.min().item()

        if correction_norm > 1e-4:
            telemetry.tier_activated = max(telemetry.tier_activated, 2)

        # ── Tier 3: Soft Penalty (only during training) ──
        b_loss = self.barrier_loss(h_x) if self.training else torch.tensor(0.0, device=device)
        if b_loss.item() > 1e-6:
            telemetry.tier_activated = max(telemetry.tier_activated, 3)

        # ── Apply Tier 1 mask: override unsafe samples with zero-action ──
        if hw_violated is not None and hw_violated.any():
            telemetry.tier_activated = 1
            u_safe = u_safe.clone()
            u_safe[hw_violated] = 0.0  # Emergency stop for violated samples
            h_x = h_x.clone()
            h_x[hw_violated] = -1.0

        self._last_telemetry = telemetry

        return {
            'action': u_safe,
            'barrier_loss': b_loss,
            'h_x': h_x,
            'tier': telemetry.tier_activated,
        }

    # ─── Introspection ───────────────────────────────────────────────

    @property
    def telemetry(self) -> Optional[ShieldTelemetry]:
        """Last evaluation's telemetry. Useful for dashboards and logging."""
        return self._last_telemetry

    def safety_report(self, test_states: torch.Tensor) -> Dict:
        """
        Batch evaluate the barrier function on a set of test states.

        Args:
            test_states: (N, state_dim) tensor of physical states to audit.
        Returns:
            Report dict with pass/fail counts and margin statistics.
        """
        with torch.no_grad():
            h_vals = self.barrier(test_states)
            safe_mask = h_vals > 0

        return {
            'total': len(test_states),
            'safe': safe_mask.sum().item(),
            'unsafe': (~safe_mask).sum().item(),
            'min_margin': h_vals.min().item(),
            'mean_margin': h_vals.mean().item(),
            'status': 'CERTIFIED' if safe_mask.all() else 'VIOLATIONS_FOUND',
        }
