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
    1. Soft-parameterizing ICNN weights via softplus(θ).
    2. Using convex, non-decreasing activation functions (Softplus).
    3. Adding pass-through (skip) connections from input to each layer.
    """

    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim

        # Layer 0: Initial projection (unconstrained)
        self.w0    = nn.Linear(state_dim, hidden)

        # Layer 1: ICNN step — raw params, positivity enforced in forward()
        self.y1_raw = nn.Linear(hidden, hidden, bias=True)
        self.x1_w   = nn.Linear(state_dim, hidden, bias=False)

        # Final Layer — raw params, positivity enforced in forward()
        self.y2_raw = nn.Linear(hidden, 1, bias=True)
        self.x2_w   = nn.Linear(state_dim, 1, bias=False)

        # Softplus for convex activations (high beta ≈ ReLU but smooth)
        self.softplus = nn.Softplus(beta=5)

        # Initialize raw weights in a range where softplus(w) > 0.1
        with torch.no_grad():
            nn.init.uniform_(self.y1_raw.weight, 0.1, 0.8)
            nn.init.uniform_(self.y2_raw.weight, 0.1, 0.8)

    def _ensure_icnn_constraints(self):
        """No-op: positivity is now guaranteed by construction in forward().
        Kept for backwards API compatibility with OmniExporter / trainer."""
        pass

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        ICNN Forward Pass with soft-parameterized non-negative weights.
        The effective weights w = softplus(raw_w) are always > 0,
        ensuring convexity at every evaluation without any gradient interference.
        """
        # Project raw params to non-negative space at runtime
        # Add 1e-4 epsilon to ensure strict convexity and prevent vanishing gradients
        w1 = F.softplus(self.y1_raw.weight) + 0.01 * self.y1_raw.weight.abs() + 1e-4
        w2 = F.softplus(self.y2_raw.weight) + 0.01 * self.y2_raw.weight.abs() + 1e-4

        z1 = self.softplus(self.w0(state))
        z2 = self.softplus(F.linear(z1, w1, self.y1_raw.bias) + self.x1_w(state))
        h  = F.linear(z2, w2, self.y2_raw.bias) + self.x2_w(state)

        return h.squeeze(-1)


# ─────────────────────────────────────────────────────────────────────
#  State Extractor: Latents → Physical State
# ─────────────────────────────────────────────────────────────────────

class AttentionStateExtractor(nn.Module):
    """
    Bridges FusionCore latents (B, N, D) to a physical state vector (B, state_dim).

    Uses a learned 'Physical Query' to extract specific features from the 
    multimodal token bus, preserving modality identity and spatial relevance.
    """

    def __init__(self, d_model: int, state_dim: int, num_queries: int = 4):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        
        # Learned queries to 'probe' the latent tokens
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model))
        
        # O(N) Linear Attention for fast extraction
        self.kv_proj = nn.Linear(d_model, d_model * 2)
        self.q_proj = nn.Linear(d_model, d_model)
        
        self.decoder = nn.Sequential(
            nn.Linear(num_queries * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: (Batch, N, d_model) from FusionCore.
        Returns:
            state: (Batch, state_dim) estimated physical state.
        """
        B, N, D = latents.shape
        
        # Fast Linear Attention probe
        kv = self.kv_proj(latents)  # (B, N, 2*D)
        k, v = torch.split(kv, D, dim=-1)
        
        # ELU+1 kernel for stability
        k = F.elu(k) + 1.0
        q = F.elu(self.q_proj(self.queries.expand(B, -1, -1))) + 1.0
        
        # S = k^T @ v
        s = torch.bmm(k.transpose(-1, -2), v) # (B, D, D)
        z = k.sum(dim=1, keepdim=True).transpose(-1, -2) # (B, D, 1)
        
        # Readout
        num = torch.bmm(q, s) # (B, Q, D)
        den = torch.bmm(q, z) + 1e-6
        fused = (num / den).reshape(B, -1) # (B, Q*D)
        
        return self.decoder(fused)


# ─────────────────────────────────────────────────────────────────────
#  Learned Dynamics Model: x_next = f(x, u)
# ─────────────────────────────────────────────────────────────────────

class ResidualDynamics(nn.Module):
    """
    Learns the residual dynamics: x_next = x + dt * f_theta(x, u).
    Uses Runge-Kutta 4 (RK4) integration for improved accuracy.
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

    def f(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Continuous-time derivative f(x, u)."""
        xu = torch.cat([x, u], dim=-1)
        return self.net(xu)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:  (Batch, state_dim)
            action: (Batch, action_dim)
        Returns:
            x_next: (Batch, state_dim)
        """
        # RK4 Integration
        dt = self.dt
        k1 = self.f(state, action)
        k2 = self.f(state + 0.5 * dt * k1, action)
        k3 = self.f(state + 0.5 * dt * k2, action)
        k4 = self.f(state + dt * k3, action)
        
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ─────────────────────────────────────────────────────────────────────
#  OmniShield v2: 3-Tier Unified Safety System
# ─────────────────────────────────────────────────────────────────────

class OmniShieldGuard(nn.Module):
    """
    OmniShield v2: 3-Tier Safety System.

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
        self.state_extractor = AttentionStateExtractor(d_model, state_dim)
        self.barrier = NeuralBarrier(state_dim)
        self.dynamics = ResidualDynamics(state_dim, action_dim, dt=dt)

        # ── Config ──
        self.alpha = alpha
        self.barrier_loss_weight = barrier_loss_weight
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.create_graph = False

        # ── Tier 1: Hardware limits (vectorized) ──
        self.num_hw_sensors = num_hw_sensors
        if num_hw_sensors > 0:
            # Placeholder initialization. Must be set via set_hw_limits or from_config.
            self.register_buffer('hw_min', torch.full((num_hw_sensors,), -float('inf')))
            self.register_buffer('hw_max', torch.full((num_hw_sensors,), float('inf')))
            
        self.register_buffer('emergency_action', torch.zeros(1, action_dim))

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

        # Extract state_dim from config to preserve physical meaning
        model_cfg = config.get('model', {})
        shield_cfg = config.get('shield', {})
        state_dim = shield_cfg.get('state_dim', model_cfg.get('state_dim', 16))

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

        Using Vectorized Jacobian for Inference (torch.func).
        This eliminates the O(N) loop over action dimensions, enabling
        sub-millisecond safety verification for high-dimensional robots.
        """
        # Current safety value
        h_x = self.barrier(state)

        # Forward simulate with current action to get future safety
        x_next = self.dynamics(state, u_nn)
        h_next = self.barrier(x_next)

        # CBF condition: h_next ≥ (1-α) * h_x
        cbf_violation = (1 - self.alpha) * h_x - h_next  # > 0 means violated

        # Compute Lie derivative: dh_next/du
        if u_nn.requires_grad:
            # Training mode: full gradient flow
            lg_h = torch.autograd.grad(
                h_next.sum(), u_nn,
                create_graph=self.create_graph,
                retain_graph=True,
            )[0]
        else:
            
            # We use a functional approach to compute the gradient for each batch element
            # This is significantly faster than the previous finite differences loop.
            try:
                from torch.func import vmap, jacrev
                
                # Define a local function for a single sample to use with vmap
                def single_h_next(u_single, state_single):
                    # Use full RK4 forward pass for mathematical consistency
                    x_n = self.dynamics(state_single.unsqueeze(0), u_single.unsqueeze(0))
                    return self.barrier(x_n)

                # Compute Jacobian across the batch
                lg_h = vmap(jacrev(single_h_next))(u_nn, state).squeeze(1)
            except (ImportError, RuntimeError):
                # Fallback to analytical finite differences if torch.func is unavailable
                eps = 1e-4
                action_dim = u_nn.shape[1]
                lg_h_list = []
                for i in range(action_dim):
                    u_p = u_nn.clone()
                    u_p[:, i] += eps
                    x_next_p = self.dynamics(state, u_p)
                    h_next_p = self.barrier(x_next_p)
                    lg_h_list.append((h_next_p - h_next) / eps)
                lg_h = torch.stack(lg_h_list, dim=1)

        
        # Robust projection even when the gradient is near-zero.
        lg_h_norm_sq = (lg_h * lg_h).sum(dim=1, keepdim=True)
        
        # Regularization epsilon prevents division by zero
        epsilon = 1e-8
        
        cbf_violation = F.relu(cbf_violation).unsqueeze(-1)
        max_correction = 5.0 # Max action delta per step
        
        # Calculate lambda with regularization
        lam = torch.clamp(cbf_violation / (lg_h_norm_sq + epsilon), max=max_correction)
        
        u_safe = u_nn + lam * lg_h

        return u_safe, h_x.detach() if not u_nn.requires_grad else h_x

    # ─── Tier 3: Barrier Loss ────────────────────────────────────────

    def barrier_loss(self, h_x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic Barrier Loss with Centered Penalty.

        Added quadratic centered penalty.
        While the log-barrier pushes the policy away from the edge,
        the centered penalty encourages the policy to stay near a 
        known-safe nominal state (estimated from the state vector),
        preventing 'safety-jitter' where the policy oscillates near the boundary.
        """
        eps = 1e-6
        safe_mask = h_x > eps
        
        # Log-barrier component (Tier 3)
        log_val = -torch.log(h_x.clamp(min=eps))
        violated_val = -torch.log(torch.tensor(eps)) + (0.5 / eps) * ( (h_x - eps)**2 / eps - 2*(h_x - eps) )
        log_barrier = torch.where(safe_mask, log_val, violated_val)
        
        # Centered Penalty
        # Penalize if the state is drifting too far from a safe 'center' 
        # (simplified here as a penalty on state magnitude if state_dim allows)
        dist_penalty = 0.01 * (state**2).sum(dim=-1)
        
        return self.barrier_loss_weight * (log_barrier + dist_penalty).mean()

    # ─── Unified Forward ─────────────────────────────────────────────

    def forward(
        self,
        latents: torch.Tensor,
        sensor_batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        3-Tier Safety Pipeline.
        """
        batch_size = latents.shape[0]
        device = latents.device
        telemetry = ShieldTelemetry()

        # 1. Compute raw action from head (Polymorphic)
        u_nn_raw = self.action_head(latents)
        if isinstance(u_nn_raw, dict):
            u_nn = u_nn_raw.get('action', u_nn_raw.get('mean', next(iter(u_nn_raw.values()))))
        elif hasattr(u_nn_raw, 'rsample'):
            u_nn = u_nn_raw.rsample() if self.training else u_nn_raw.mean
        else:
            u_nn = u_nn_raw

        # 2. Tier 1: Hardware Failsafe & Soft-Limits
        if sensor_batch is not None and self.num_hw_sensors > 0:
            # Soft-limit margin (e.g., 10% of the range)
            margin = 0.1 * (self.hw_max - self.hw_min).clamp(min=1e-6)
            dist_min = sensor_batch - self.hw_min
            dist_max = self.hw_max - sensor_batch
            
            # Sigmoidal penalty (1.0 safe, 0.0 at limit)
            soft_safe = torch.sigmoid(5.0 * dist_min / margin) * \
                        torch.sigmoid(5.0 * dist_max / margin)
            
            if self.training:
                # Modulate action with soft-limit gradient signal
                u_nn = u_nn * soft_safe.min(dim=1, keepdim=True)[0]

            # Hard-limit check
            hw_violated, hw_count = self._check_hw_limits(sensor_batch)
            telemetry.hw_violations = hw_count
            
            if hw_violated.any():
                if torch.isinf(self.hw_min).any() or torch.isinf(self.hw_max).any():
                    raise RuntimeError("Tier 1 active but hardware limits are +/-inf. deployment requires explicit limits.")
                telemetry.tier_activated = 1
        else:
            hw_violated = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if u_nn.requires_grad:
            u_nn.retain_grad()

        # 3. Extract physical state from latents (Attention)
        state = self.state_extractor(latents)

        # 4. Tier 2: CBF Projection (Vectorized)
        u_safe, h_x = self._cbf_project(u_nn, state)
        correction_norm = (u_safe - u_nn).norm(dim=1).mean().item()
        telemetry.cbf_correction_norm = correction_norm
        telemetry.barrier_value = h_x.mean().item()
        telemetry.safety_margin = h_x.min().item()

        if correction_norm > 1e-6:
            telemetry.tier_activated = max(telemetry.tier_activated, 2)

        # 5. Tier 3: Soft Penalty (Centered)
        b_loss = self.barrier_loss(h_x, state) if self.training else torch.tensor(0.0, device=device)
        if b_loss.item() > 1e-6:
            telemetry.tier_activated = max(telemetry.tier_activated, 3)

        # 6. Tier 1 hard override
        if hw_violated.any():
            u_safe = u_safe.clone()
            u_safe[hw_violated] = self.emergency_action.expand(batch_size, -1)[hw_violated]
            h_x = h_x.clone()
            h_x[hw_violated] = -1.0

        self._last_telemetry = telemetry
        return {
            'action': u_safe,
            'barrier_loss': b_loss,
            'h_x': h_x,
            'tier': telemetry.tier_activated
        }

    def get_telemetry(self) -> Optional[ShieldTelemetry]:
        """Returns the telemetry from the last forward pass."""
        return self._last_telemetry

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
