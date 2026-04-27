"""
OmniStream Validation Test.
Verifies that any Python data type is correctly detected,
transformed, and processed by the Liquid Brain.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

PASS = "✅"
FAIL = "❌"
results = []


def test(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition))
    msg = f"  {status} {name}"
    if detail and not condition:
        msg += f" — {detail}"
    print(msg)


def build_test_brain():
    """Create a minimal Liquid Brain for testing."""
    from omnitrain.fusion_core import LiquidFusionCore
    from omnitrain.heads import RegressionHead
    from omnitrain.omni_shield import OmniShieldGuard

    core = LiquidFusionCore(n_latents=16, d_model=64, input_dim=32)
    core.eval()

    head = RegressionHead(output_dim=2, d_model=64)
    head.eval()

    shield = OmniShieldGuard(
        action_head=head, d_model=64, state_dim=4,
        action_dim=2, num_hw_sensors=2,
    )
    shield.eval()
    shield.set_hw_limits(
        torch.tensor([0.15, 0.0]),
        torch.tensor([10.0, 1.0]),
    )

    return core, shield


def test_type_detector():
    print("\n--- Test 1: Type Detection Engine ---")
    from omnitrain.omni_stream import TypeDetector

    # Scalar
    r = TypeDetector.detect(42.0)
    test("Scalar (float) → (1,1,1)", r.tensor.shape == (1, 1, 1))
    test("Scalar modal_id is 'scalar'", r.modal_id == "scalar")

    # Integer
    r = TypeDetector.detect(7)
    test("Scalar (int) → (1,1,1)", r.tensor.shape == (1, 1, 1))

    # Boolean
    r = TypeDetector.detect(True)
    test("Scalar (bool) → (1,1,1) value=1.0", r.tensor.item() == 1.0)

    # List
    r = TypeDetector.detect([1.0, 2.0, 3.0])
    test("List[3] → (1,1,3)", r.tensor.shape == (1, 1, 3))

    # Nested list
    r = TypeDetector.detect([[1, 2], [3, 4]])
    test("Nested list → flattened (1,1,4)", r.tensor.shape == (1, 1, 4))

    # Numpy 1D
    r = TypeDetector.detect(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    test("numpy 1D(5) → (1,1,5)", r.tensor.shape == (1, 1, 5))

    # Numpy 2D
    r = TypeDetector.detect(np.random.randn(10, 32))
    test("numpy 2D(10,32) → (1,10,32)", r.tensor.shape == (1, 10, 32))

    # Numpy 3D (non-image)
    r = TypeDetector.detect(np.random.randn(2, 10, 64))
    test("numpy 3D(2,10,64) → (2,10,64)", r.tensor.shape == (2, 10, 64))

    # Torch tensor 1D
    r = TypeDetector.detect(torch.tensor([1.0, 2.0]))
    test("Tensor 1D(2) → (1,1,2)", r.tensor.shape == (1, 1, 2))

    # Torch tensor scalar
    r = TypeDetector.detect(torch.tensor(3.14))
    test("Tensor scalar → (1,1,1)", r.tensor.shape == (1, 1, 1))

    # Dict
    r = TypeDetector.detect({"lidar": 1.5, "battery": 0.8, "temp": 25.0})
    test("Dict[3 keys] → (1,1,3)", r.tensor.shape == (1, 1, 3))

    # Dict with mixed values
    r = TypeDetector.detect({"gps": [40.7, -74.0], "speed": 5.0})
    test("Dict mixed → (1,1,3)", r.tensor.shape == (1, 1, 3))

    # Custom modal_id override
    r = TypeDetector.detect(42.0, modal_id="front_sonar")
    test("modal_id override works", r.modal_id == "front_sonar")


def test_omni_stream_basic():
    print("\n--- Test 2: OmniStream Basic Flow ---")
    from omnitrain.omni_stream import OmniStream

    core, shield = build_test_brain()
    stream = OmniStream(core, shield)

    # Scalar input
    result = stream.send(1.5, modal_id="lidar")
    test("Scalar send → has latents", result['latents'] is not None)
    test("Scalar send → latents shape", result['latents'].shape == (1, 16, 64))
    test("Scalar send → has action", result['action'] is not None)
    test("Scalar send → action shape", result['action'].shape == (1, 2))

    # List input
    result = stream.send([0.3, 0.5, 1.2, 0.8, 2.0], modal_id="imu")
    test("List send → latents ok", result['latents'].shape == (1, 16, 64))

    # Numpy input
    result = stream.send(np.random.randn(1, 10, 32))
    test("Numpy send → latents ok", result['latents'].shape == (1, 16, 64))

    # Step counter
    test("Step counter = 3", stream.steps == 3)


def test_omni_stream_dict_fusion():
    print("\n--- Test 3: Multi-Sensor Dict Fusion ---")
    from omnitrain.omni_stream import OmniStream

    core, shield = build_test_brain()
    stream = OmniStream(core, shield)

    # Dict input with multiple sensors
    result = stream.send({
        "lidar": 1.5,
        "battery": 0.85,
    })

    test("Dict fusion → has action", result['action'] is not None)
    test("Dict fusion → latents shape", result['latents'].shape == (1, 16, 64))
    test("Dict fusion → step count = 1", stream.steps == 1)
    test("Dict fusion → multiple inputs detected", isinstance(result['input'], list))
    test("Dict fusion → 2 modalities", len(result['input']) == 2)


def test_omni_stream_safety():
    print("\n--- Test 4: Safety Integration ---")
    from omnitrain.omni_stream import OmniStream

    core, shield = build_test_brain()
    stream = OmniStream(core, shield)

    # Safe reading
    result = stream.send(5.0, modal_id="lidar", hw_sensors=[5.0, 0.5])
    test("Safe reading → tier 0 or 3", result['tier'] in (0, 3))

    # Dangerous reading
    stream.reset()
    result = stream.send(0.05, modal_id="lidar", hw_sensors=[0.05, 0.5])
    test("Danger reading → tier 1 (hardware)", result['tier'] == 1)
    test("Danger → action is zero", torch.allclose(result['action'], torch.zeros(1, 2)))


def test_omni_stream_memory():
    print("\n--- Test 5: Liquid Memory Persistence ---")
    from omnitrain.omni_stream import OmniStream

    core, shield = build_test_brain()
    stream = OmniStream(core, shield)

    # Step 1
    r1 = stream.send(1.0, modal_id="sensor")
    latents_1 = r1['latents'].clone()

    # Step 2 (same input, but memory should affect output)
    r2 = stream.send(1.0, modal_id="sensor")
    latents_2 = r2['latents'].clone()

    different = not torch.allclose(latents_1, latents_2, atol=1e-6)
    test("Liquid memory → consecutive steps differ", different)

    # Reset
    stream.reset()
    test("Reset → step count = 0", stream.steps == 0)
    test("Reset → state is None", stream.state is None)


def test_omni_stream_repr():
    print("\n--- Test 6: Introspection ---")
    from omnitrain.omni_stream import OmniStream

    core, shield = build_test_brain()
    stream = OmniStream(core, shield)

    repr_str = repr(stream)
    test("Repr contains core type", "LiquidFusionCore" in repr_str)
    test("Repr contains shield status", "Active" in repr_str)

    # Without shield
    stream2 = OmniStream(core, shield=None)
    r = stream2.send(1.0, modal_id="test")
    test("No shield → action is None", r['action'] is None)
    test("No shield → tier is 0", r['tier'] == 0)


if __name__ == "__main__":
    print("=" * 60)
    print("🔌 OMNISTREAM UNIVERSAL SENSOR WRAPPER — VALIDATION TEST")
    print("=" * 60)

    test_type_detector()
    test_omni_stream_basic()
    test_omni_stream_dict_fusion()
    test_omni_stream_safety()
    test_omni_stream_memory()
    test_omni_stream_repr()

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"📊 Results: {passed}/{total} tests passed")
    if passed == total:
        print("🏆 ALL TESTS PASSED — OmniStream is fully operational!")
    else:
        print("⚠️  Some tests failed. Review output above.")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)
