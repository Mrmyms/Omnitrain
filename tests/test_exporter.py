"""
tests/test_exporter.py
======================
Unit tests for the OmniTrain .omnibit exporter.
Verifies that:
  1. The binary format is correct (magic bytes, header, dimensions)
  2. The exporter produces deterministic outputs
  3. The exported weights can be reloaded and compared

Run:
    cd /path/to/Omnitrain
    pip install -e .
    pytest tests/ -v
"""
import io
import struct
import pytest
import torch
import numpy as np
import sys
import os

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from omnitrain.fusion_core import LiquidFusionCore
from omnitrain.esp32_exporter import ESP32Exporter


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

INPUT_DIM  = 4
D_MODEL    = 16
N_LATENTS  = 2

@pytest.fixture
def small_model():
    """Minimal LiquidFusionCore for fast tests."""
    model = LiquidFusionCore(
        n_latents=N_LATENTS,
        d_model=D_MODEL,
        input_dim=INPUT_DIM,
        config={}  # Legacy brain mode (BioLiquidCell)
    )
    model.eval()
    return model


@pytest.fixture
def exported_omnibit(small_model, tmp_path):
    """Export the small model to a temp .omnibit file."""
    exporter = ESP32Exporter(output_dir=str(tmp_path))
    out_path = exporter.export(small_model, input_dim=INPUT_DIM, d_model=D_MODEL,
                               output_dim=2, backbone_units=8)
    return out_path


# ─────────────────────────────────────────────
# Test 1: Binary Format Integrity
# ─────────────────────────────────────────────

def test_omnibit_magic_bytes(exported_omnibit):
    """The .omnibit file must start with the correct magic bytes."""
    with open(exported_omnibit, 'rb') as f:
        header = f.read(5)
    assert header[:4] == b'OMNI', f"Magic bytes missing, got: {header[:4]!r}"
    assert header[4] == 0x02, f"Expected version 0x02, got: {header[4]:#x}"


def test_omnibit_dimension_header(exported_omnibit):
    """The dimension header must contain 5 uint32 values at offset 8."""
    with open(exported_omnibit, 'rb') as f:
        f.seek(8)
        raw = f.read(5 * 4)  # 5 x uint32
    dims = struct.unpack('<5I', raw)
    input_dim, d_model, output_dim, backbone_units, total_weights = dims

    assert input_dim == INPUT_DIM,  f"input_dim mismatch: {input_dim} != {INPUT_DIM}"
    assert d_model   == D_MODEL,    f"d_model mismatch: {d_model} != {D_MODEL}"
    assert output_dim > 0,          "output_dim must be positive"
    assert backbone_units > 0,      "backbone_units must be positive"
    assert total_weights > 0,       "total_weights must be positive"


def test_omnibit_weight_count(exported_omnibit):
    """Weight section length must match the declared total_weights."""
    with open(exported_omnibit, 'rb') as f:
        content = f.read()

    # total_weights is at offset 28 (5th dim, 0-indexed from offset 8)
    total_weights = struct.unpack('<I', content[24:28])[0]

    # Weights start at byte 28
    weight_bytes = len(content) - 28
    expected_bytes = total_weights * 4  # float32 = 4 bytes

    assert weight_bytes == expected_bytes, (
        f"Weight section size mismatch: got {weight_bytes} bytes, "
        f"expected {expected_bytes} ({total_weights} floats)"
    )


def test_omnibit_not_all_zeros(exported_omnibit):
    """Exported weights should not all be zero (sanity check)."""
    with open(exported_omnibit, 'rb') as f:
        f.seek(28)  # Skip header
        raw = f.read()
    weights = np.frombuffer(raw, dtype=np.float32)
    assert not np.all(weights == 0.0), "All exported weights are zero — likely an export bug"


# ─────────────────────────────────────────────
# Test 2: Determinism
# ─────────────────────────────────────────────

def test_export_is_deterministic(small_model, tmp_path):
    """Exporting the same model twice must produce identical files."""
    exporter = ESP32Exporter(output_dir=str(tmp_path))
    path1 = exporter.export(small_model, input_dim=INPUT_DIM, d_model=D_MODEL,
                            output_dim=2, backbone_units=8, filename="brain_a.omnibit")
    path2 = exporter.export(small_model, input_dim=INPUT_DIM, d_model=D_MODEL,
                            output_dim=2, backbone_units=8, filename="brain_b.omnibit")

    with open(path1, 'rb') as f1, open(path2, 'rb') as f2:
        assert f1.read() == f2.read(), "Two exports of the same model produced different bytes"


# ─────────────────────────────────────────────
# Test 3: step_stateless() correctness
# ─────────────────────────────────────────────

def test_step_stateless_output_shapes(small_model):
    """step_stateless() must return correct shapes."""
    B = 2
    sensors  = torch.zeros(B, INPUT_DIM)
    state_in = torch.zeros(B, D_MODEL)
    dt       = torch.full((B,), 0.01)

    with torch.no_grad():
        action, state_out = small_model.step_stateless(sensors, state_in, dt)

    assert action.shape    == (B, D_MODEL), f"action shape: {action.shape}"
    assert state_out.shape == (B, D_MODEL), f"state_out shape: {state_out.shape}"


def test_step_stateless_state_is_not_zero(small_model):
    """After one step, state_out must not be all zeros (network must activate)."""
    sensors  = torch.randn(1, INPUT_DIM)
    state_in = torch.zeros(1, D_MODEL)
    dt       = torch.tensor([0.01])

    with torch.no_grad():
        action, state_out = small_model.step_stateless(sensors, state_in, dt)

    assert not torch.all(state_out == 0.0), "state_out is all zeros — network is not activating"


def test_step_stateless_state_propagates(small_model):
    """Running two sequential steps with state propagation must change the output."""
    sensors  = torch.randn(1, INPUT_DIM)
    dt       = torch.tensor([0.01])

    with torch.no_grad():
        state = torch.zeros(1, D_MODEL)
        out1, state = small_model.step_stateless(sensors, state, dt)
        out2, state = small_model.step_stateless(sensors, state, dt)

    assert not torch.allclose(out1, out2), (
        "Two sequential steps produced identical output — state is not propagating"
    )


def test_step_stateless_raises_on_ncp_mode():
    """step_stateless() must raise RuntimeError for non-legacy brain modes."""
    model = LiquidFusionCore(
        n_latents=2, d_model=16, input_dim=4,
        config={'model': {'ncp': {'enabled': True, 'sensory': 8, 'inter': 8, 'command': 4}}}
    )
    model.eval()
    sensors  = torch.zeros(1, 4)
    state_in = torch.zeros(1, 16)
    dt       = torch.tensor([0.01])

    with pytest.raises(RuntimeError, match="step_stateless\\(\\) requires brain_mode='legacy'"):
        model.step_stateless(sensors, state_in, dt)
