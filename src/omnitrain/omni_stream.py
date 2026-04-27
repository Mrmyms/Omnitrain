"""
OmniStream: Universal Sensor Ingestion Layer.

The "Plug & Play" interface for OmniTrain.
Accepts ANY Python data type and automatically:
  1. Detects its type and structure.
  2. Transforms it into a properly shaped tensor.
  3. Assigns a modal_id for the AdaptiveInputProjector.
  4. Manages timing (dt) transparently.

Supported input types:
  - float / int              → Scalar sensor (battery, temperature)
  - list / tuple             → Vector sensor (IMU, GPS coordinates)
  - dict                     → Multi-sensor bundle (all keys processed)
  - numpy.ndarray            → Direct array (any shape)
  - torch.Tensor             → Pass-through (already correct)
  - str (file path to image) → Vision input via CNN pipeline
  - PIL.Image                → Vision input via CNN pipeline
  - pandas.DataFrame row     → Tabular sensor data
"""

import torch
import numpy as np
import time
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────
#  Type Detection Engine
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DetectedInput:
    """Result of automatic type detection."""
    tensor: torch.Tensor          # (1, N, D) ready for the core
    modal_id: str                 # Routing key for AdaptiveInputProjector
    is_image: bool = False        # Use CNN pipeline instead of linear
    original_type: str = ""       # For telemetry/debugging
    shape_description: str = ""   # Human-readable shape info


class TypeDetector:
    """
    Automatic data type detection and tensor conversion engine.
    Converts any Python object into a (Batch=1, Sequence, Features) tensor.
    """

    @staticmethod
    def detect(data: Any, modal_id: Optional[str] = None) -> DetectedInput:
        """
        Universal detection entry point.
        
        Args:
            data: Any Python object containing sensor data.
            modal_id: Optional override for the modality identifier.
                      If None, auto-generated from the data type/structure.
        
        Returns:
            DetectedInput with tensor ready for FusionCore.
        """
        # ── Already a Tensor ──
        if isinstance(data, torch.Tensor):
            return TypeDetector._from_tensor(data, modal_id)

        # ── NumPy array ──
        if isinstance(data, np.ndarray):
            return TypeDetector._from_numpy(data, modal_id)

        # ── Scalar (int, float, bool) ──
        if isinstance(data, (int, float, bool)):
            return TypeDetector._from_scalar(data, modal_id)

        # ── List or Tuple of numbers ──
        if isinstance(data, (list, tuple)):
            return TypeDetector._from_sequence(data, modal_id)

        # ── Dictionary (multi-sensor bundle) ──
        if isinstance(data, dict):
            return TypeDetector._from_dict(data, modal_id)

        # ── String (file path → image) ──
        if isinstance(data, str):
            return TypeDetector._from_string(data, modal_id)

        # ── PIL Image ──
        try:
            from PIL import Image
            if isinstance(data, Image.Image):
                return TypeDetector._from_pil_image(data, modal_id)
        except ImportError:
            pass

        # ── Pandas DataFrame / Series ──
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                return TypeDetector._from_dataframe(data, modal_id)
            if isinstance(data, pd.Series):
                return TypeDetector._from_series(data, modal_id)
        except ImportError:
            pass

        # ── Fallback: try to convert to float ──
        try:
            val = float(data)
            return TypeDetector._from_scalar(val, modal_id or "unknown")
        except (TypeError, ValueError):
            raise TypeError(
                f"OmniStream cannot process type '{type(data).__name__}'. "
                f"Supported: int, float, list, dict, numpy.ndarray, torch.Tensor, "
                f"PIL.Image, pandas.DataFrame, or image file path (str)."
            )

    # ── Type-Specific Converters ────────────────────────────────────

    @staticmethod
    def _from_scalar(val: Union[int, float, bool], modal_id: Optional[str]) -> DetectedInput:
        t = torch.tensor([[[float(val)]]], dtype=torch.float32)  # (1, 1, 1)
        return DetectedInput(
            tensor=t,
            modal_id=modal_id or "scalar",
            original_type="scalar",
            shape_description="(1, 1, 1)"
        )

    @staticmethod
    def _from_sequence(seq: Union[list, tuple], modal_id: Optional[str]) -> DetectedInput:
        # Flatten nested lists if needed
        flat = TypeDetector._flatten_numeric(seq)
        arr = np.array(flat, dtype=np.float32)
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        auto_id = modal_id or f"vector_{len(flat)}"
        return DetectedInput(
            tensor=t,
            modal_id=auto_id,
            original_type=f"list[{len(flat)}]",
            shape_description=f"(1, 1, {len(flat)})"
        )

    @staticmethod
    def _from_numpy(arr: np.ndarray, modal_id: Optional[str]) -> DetectedInput:
        arr = arr.astype(np.float32)

        # Image detection: 3D array with shape (H, W, C) where C is small
        # Heuristic: spatial dimensions must be large (>16) to avoid sequence confusion
        if arr.ndim == 3 and arr.shape[2] in (1, 3, 4) and arr.shape[0] > 16 and arr.shape[1] > 16:
            return TypeDetector._from_image_array(arr, modal_id)
        
        # Image in CHW format: (C, H, W) where C is small and H, W are large
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[1] > 16 and arr.shape[2] > 16:
            return TypeDetector._from_image_array(arr, modal_id)

        # Ensure 3D: (Batch, Sequence, Features)
        if arr.ndim == 0:
            t = torch.tensor([[[arr.item()]]])
        elif arr.ndim == 1:
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        elif arr.ndim == 2:
            t = torch.from_numpy(arr).unsqueeze(0)  # (1, N, D)
        elif arr.ndim == 3:
            t = torch.from_numpy(arr)  # Already (B, N, D) — do NOT add batch dim
        else:
            # 4D+ : assume image batch
            t = torch.from_numpy(arr)
            return DetectedInput(
                tensor=t, modal_id=modal_id or "image",
                is_image=True, original_type=f"ndarray{list(arr.shape)}",
                shape_description=str(list(t.shape))
            )

        return DetectedInput(
            tensor=t,
            modal_id=modal_id or f"array_{arr.shape[-1]}",
            original_type=f"ndarray{list(arr.shape)}",
            shape_description=str(list(t.shape))
        )

    @staticmethod
    def _from_tensor(t: torch.Tensor, modal_id: Optional[str]) -> DetectedInput:
        # Image detection
        if t.dim() == 4:
            return DetectedInput(
                tensor=t, modal_id=modal_id or "image",
                is_image=True, original_type=f"tensor{list(t.shape)}",
                shape_description=str(list(t.shape))
            )

        if t.dim() == 0:
            t = t.float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif t.dim() == 1:
            t = t.float().unsqueeze(0).unsqueeze(0)
        elif t.dim() == 2:
            t = t.float().unsqueeze(0)

        return DetectedInput(
            tensor=t.float(),
            modal_id=modal_id or f"tensor_{t.shape[-1]}",
            original_type=f"tensor{list(t.shape)}",
            shape_description=str(list(t.shape))
        )

    @staticmethod
    def _from_dict(d: dict, modal_id: Optional[str]) -> DetectedInput:
        """Flatten a dict of numeric values into a single vector."""
        values = []
        keys = []
        for k, v in d.items():
            if isinstance(v, (int, float, bool)):
                values.append(float(v))
                keys.append(k)
            elif isinstance(v, (list, tuple)):
                flat = TypeDetector._flatten_numeric(v)
                values.extend(flat)
                keys.extend([f"{k}_{i}" for i in range(len(flat))])
            elif isinstance(v, np.ndarray):
                flat = v.flatten().tolist()
                values.extend(flat)
                keys.extend([f"{k}_{i}" for i in range(len(flat))])

        if not values:
            raise ValueError("OmniStream: dict contains no numeric values to process.")

        arr = np.array(values, dtype=np.float32)
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, D)

        auto_id = modal_id or "bundle_" + "_".join(list(d.keys())[:3])
        return DetectedInput(
            tensor=t,
            modal_id=auto_id,
            original_type=f"dict[{len(d)} keys]",
            shape_description=f"(1, 1, {len(values)})"
        )

    @staticmethod
    def _from_string(path: str, modal_id: Optional[str]) -> DetectedInput:
        """Interpret a string as an image file path."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"OmniStream: '{path}' is not a valid file path.")

        ext = os.path.splitext(path)[1].lower()
        if ext not in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'):
            raise ValueError(f"OmniStream: '{ext}' is not a supported image format.")

        try:
            from PIL import Image
            img = Image.open(path).convert('RGB')
            return TypeDetector._from_pil_image(img, modal_id or "camera")
        except ImportError:
            raise ImportError("OmniStream: Pillow is required for image loading. Install with: pip install Pillow")

    @staticmethod
    def _from_pil_image(img, modal_id: Optional[str]) -> DetectedInput:
        """Convert a PIL Image to a (1, 3, H, W) tensor."""
        img = img.convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        # (H, W, C) → (C, H, W) → (1, C, H, W)
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return DetectedInput(
            tensor=t,
            modal_id=modal_id or "camera",
            is_image=True,
            original_type=f"PIL.Image({img.size[0]}x{img.size[1]})",
            shape_description=str(list(t.shape))
        )

    @staticmethod
    def _from_image_array(arr: np.ndarray, modal_id: Optional[str]) -> DetectedInput:
        """Convert (H, W, C) or (C, H, W) numpy image to (1, C, H, W) tensor."""
        if arr.shape[2] in (1, 3, 4):
            # (H, W, C) → (C, H, W)
            arr = np.transpose(arr, (2, 0, 1))
        arr = arr / 255.0 if arr.max() > 1.0 else arr
        t = torch.from_numpy(arr).float().unsqueeze(0)
        return DetectedInput(
            tensor=t,
            modal_id=modal_id or "camera",
            is_image=True,
            original_type=f"image_array{list(arr.shape)}",
            shape_description=str(list(t.shape))
        )

    @staticmethod
    def _from_dataframe(df, modal_id: Optional[str]) -> DetectedInput:
        """Convert a pandas DataFrame to a (1, rows, cols) tensor."""
        arr = df.select_dtypes(include=[np.number]).values.astype(np.float32)
        t = torch.from_numpy(arr).unsqueeze(0)  # (1, rows, cols)
        return DetectedInput(
            tensor=t,
            modal_id=modal_id or "dataframe",
            original_type=f"DataFrame({df.shape[0]}x{df.shape[1]})",
            shape_description=str(list(t.shape))
        )

    @staticmethod
    def _from_series(series, modal_id: Optional[str]) -> DetectedInput:
        """Convert a pandas Series to a (1, 1, len) tensor."""
        arr = series.values.astype(np.float32)
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return DetectedInput(
            tensor=t,
            modal_id=modal_id or "series",
            original_type=f"Series({len(series)})",
            shape_description=f"(1, 1, {len(series)})"
        )

    @staticmethod
    def _flatten_numeric(seq) -> List[float]:
        """Recursively flatten nested lists/tuples into a flat list of floats."""
        flat = []
        for item in seq:
            if isinstance(item, (list, tuple)):
                flat.extend(TypeDetector._flatten_numeric(item))
            elif isinstance(item, (int, float, bool)):
                flat.append(float(item))
            elif isinstance(item, np.ndarray):
                flat.extend(item.flatten().tolist())
            else:
                try:
                    flat.append(float(item))
                except (TypeError, ValueError):
                    pass  # Skip non-numeric items
        return flat


# ─────────────────────────────────────────────────────────────────────
#  OmniStream: The Universal Sensor Interface
# ─────────────────────────────────────────────────────────────────────

class OmniStream:
    """
    Universal Sensor Ingestion Layer for OmniTrain.

    Usage:
        stream = OmniStream(core, shield)

        # Send anything — scalars, lists, dicts, images, numpy arrays...
        result = stream.send({"lidar": 1.5, "battery": 0.85})
        result = stream.send([0.3, 0.5, 1.2, 0.8])
        result = stream.send(42.0)
        result = stream.send("camera_frame.jpg")

        # Get safe motor commands
        action = result['action']
    """

    def __init__(self, core, shield=None):
        """
        Args:
            core: LiquidFusionCore or FusionCore instance.
            shield: Optional OmniShieldGuard for safety enforcement.
        """
        self.core = core
        self.shield = shield
        self.detector = TypeDetector()

        # Internal state
        self._prev_latents = None
        self._last_time = time.perf_counter()
        self._step_count = 0
        self._history: List[Dict] = []

    def reset(self):
        """Clear all internal state (liquid memory + timing)."""
        self._prev_latents = None
        self._last_time = time.perf_counter()
        self._step_count = 0
        self._history.clear()

    def send(
        self,
        data: Any,
        modal_id: Optional[str] = None,
        dt: Optional[float] = None,
        hw_sensors: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Send ANY data to the brain.

        Args:
            data:       Any Python object (see module docstring for supported types).
            modal_id:   Optional modality name override. If None, auto-detected.
            dt:         Optional time delta in seconds. If None, auto-computed
                        from wall-clock time since last call.
            hw_sensors: Optional raw sensor values for OmniShield Tier 1 checks.
                        Can be a list, dict, numpy array, or tensor.

        Returns:
            Dict with keys:
                'latents':  (1, n_latents, d_model) current brain state.
                'action':   (1, action_dim) safe motor command (if shield is active).
                'tier':     int, safety tier activated (0=none, 1=hw, 2=cbf).
                'h_x':      float, barrier safety value.
                'dt':       float, time delta used.
                'input':    DetectedInput metadata (for debugging).
        """
        # ── 1. Auto-compute dt ──
        now = time.perf_counter()
        if dt is None:
            dt = now - self._last_time
        self._last_time = now

        dt_tensor = torch.tensor([dt], dtype=torch.float32)

        # ── 2. Detect and transform input ──
        if isinstance(data, dict) and modal_id is None:
            # Multi-sensor dict: process each key as a separate modality
            return self._send_multi(data, dt_tensor, dt, hw_sensors)

        detected = self.detector.detect(data, modal_id)

        # ── 3. Route through the core ──
        with torch.no_grad():
            latents = self.core(
                detected.tensor,
                dt_tensor,
                modal_id=detected.modal_id,
                prev_latents=self._prev_latents,
            )

        self._prev_latents = latents.detach()
        self._step_count += 1

        # ── 4. Safety enforcement ──
        result = self._apply_shield(latents, hw_sensors, dt)

        result['input'] = detected
        return result

    def _send_multi(
        self,
        data_dict: dict,
        dt_tensor: torch.Tensor,
        dt_float: float,
        hw_sensors: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Process a dict as multiple sensor modalities fused sequentially.
        Each key becomes a separate modal_id routed through the core.
        """
        latents = self._prev_latents
        detected_list = []

        for key, value in data_dict.items():
            detected = self.detector.detect(value, modal_id=key)
            detected_list.append(detected)

            with torch.no_grad():
                latents = self.core(
                    detected.tensor,
                    dt_tensor,
                    modal_id=detected.modal_id,
                    prev_latents=latents,
                )

        self._prev_latents = latents.detach()
        self._step_count += 1

        # Build hw_sensors from the dict if not provided
        if hw_sensors is None:
            hw_vals = []
            for key, value in data_dict.items():
                if isinstance(value, (int, float, bool)):
                    hw_vals.append(float(value))
            if hw_vals:
                hw_sensors = hw_vals

        result = self._apply_shield(latents, hw_sensors, dt_float)
        result['input'] = detected_list
        return result

    def _apply_shield(
        self,
        latents: torch.Tensor,
        hw_sensors: Optional[Any],
        dt: float,
    ) -> Dict[str, Any]:
        """Run safety checks if shield is available."""
        result = {
            'latents': latents,
            'action': None,
            'tier': 0,
            'h_x': 0.0,
            'dt': dt,
            'step': self._step_count,
        }

        if self.shield is not None:
            # Convert hw_sensors to tensor if provided
            hw_tensor = None
            if hw_sensors is not None:
                hw_tensor = self._to_hw_tensor(hw_sensors)

            shield_out = self.shield(latents, sensor_batch=hw_tensor)
            result['action'] = shield_out['action']
            result['tier'] = shield_out['tier']
            result['h_x'] = shield_out['h_x'].mean().item()
            result['barrier_loss'] = shield_out['barrier_loss']

        return result

    @staticmethod
    def _to_hw_tensor(hw_data: Any) -> torch.Tensor:
        """Convert any hw_sensors input to a (1, N) tensor."""
        if isinstance(hw_data, torch.Tensor):
            if hw_data.dim() == 1:
                return hw_data.unsqueeze(0)
            return hw_data

        if isinstance(hw_data, np.ndarray):
            t = torch.from_numpy(hw_data.astype(np.float32))
            if t.dim() == 1:
                return t.unsqueeze(0)
            return t

        if isinstance(hw_data, dict):
            vals = [float(v) for v in hw_data.values() if isinstance(v, (int, float, bool))]
            return torch.tensor([vals], dtype=torch.float32)

        if isinstance(hw_data, (list, tuple)):
            return torch.tensor([hw_data], dtype=torch.float32)

        return torch.tensor([[float(hw_data)]], dtype=torch.float32)

    # ── Convenience Properties ──────────────────────────────────────

    @property
    def state(self) -> Optional[torch.Tensor]:
        """Current liquid brain state (latents)."""
        return self._prev_latents

    @property
    def steps(self) -> int:
        """Number of inference steps completed."""
        return self._step_count

    def __repr__(self) -> str:
        core_type = type(self.core).__name__
        shield_status = "Active" if self.shield else "Disabled"
        return (
            f"OmniStream(core={core_type}, shield={shield_status}, "
            f"steps={self._step_count})"
        )
