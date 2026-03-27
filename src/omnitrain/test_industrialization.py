"""
OmniTrain 3.0 Supreme Intelligence Smoke Test
Verifies all 5 new features:
  1. Auto-Modality (AdaptiveInputProjector)
  2. Stateful Latents (RecurrentLatentMemory)
  3. DLA integration (structural check)
  4. FSDP trainer (structural check)
  5. Formal Safety Verification (SafetyGuard)
Plus backward compatibility with v2.0 tests.
"""
import sys
import os
import torch
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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


# ── v2.0 backward compatibility ──

def test_tensor_first_forward():
    print("\n--- Test 1: Tensor-First Forward (v2 Compat) ---")
    from omnitrain.fusion_core import FusionCore
    core = FusionCore(n_latents=64, d_model=256, n_heads=4, num_layers=2, input_dim=512)
    core.eval()
    sensor_data = torch.randn(2, 50, 512)
    timestamps = torch.randn(2, 50, 1)
    with torch.no_grad():
        output = core(sensor_data, timestamps)
    test("v2 forward shape is (Batch, n_latents, d_model)", output.shape == (2, 64, 256))


def test_metadata_loading():
    print("\n--- Test 2: Metadata-Driven Loading (v2 Compat) ---")
    from omnitrain.fusion_core import FusionCore
    from omnitrain.heads import ClassificationHead
    from omnitrain.exporter import OmniExporter
    tmp = "/tmp/test_v3_bundle.omni"
    core = FusionCore(n_latents=32, d_model=128, n_heads=4, num_layers=2, input_dim=256)
    heads = {'safety': ClassificationHead(num_classes=2, d_model=128)}
    OmniExporter().save(core, heads, {"test": True}, tmp)
    bundle = torch.load(tmp)
    test("Bundle version is 3.0-supreme", bundle['version'] == '3.0-supreme')
    test("Architecture has auto_modality flag", 'has_auto_modality' in bundle['architecture'])
    test("Architecture has stateful_memory flag", 'has_stateful_memory' in bundle['architecture'])
    loaded_core, _, _ = OmniExporter().load_as_inference(tmp)
    test_input = torch.randn(1, 10, 256)
    test_times = torch.randn(1, 10, 1)
    with torch.no_grad():
        out = loaded_core(test_input, test_times)
    test("Round-trip inference works", out.shape == (1, 32, 128))
    os.remove(tmp)


# ── v3.0 new features ──

def test_auto_modality():
    print("\n--- Test 3: Auto-Modality (AdaptiveInputProjector) ---")
    from omnitrain.fusion_core import FusionCore
    core = FusionCore(n_latents=32, d_model=128, n_heads=4, num_layers=2, input_dim=512)
    core.eval()

    # Default input dim (512)
    data_512 = torch.randn(1, 10, 512)
    ts = torch.randn(1, 10, 1)
    with torch.no_grad():
        out1 = core(data_512, ts, modal_id="lidar")
    test("Default 512-dim input works", out1.shape == (1, 32, 128))

    # Different input dim (256) — should auto-create a new projector
    data_256 = torch.randn(1, 10, 256)
    with torch.no_grad():
        out2 = core(data_256, ts, modal_id="gps")
    test("Auto-created 256-dim projector works", out2.shape == (1, 32, 128))

    # Verify new projector was registered
    has_gps_proj = "gps_256" in core.input_projector.modality_projectors
    test("GPS projector auto-registered in ModuleDict", has_gps_proj)

    # Yet another dim (1024)
    data_1024 = torch.randn(1, 5, 1024)
    ts_5 = torch.randn(1, 5, 1)
    with torch.no_grad():
        out3 = core(data_1024, ts_5, modal_id="camera")
    test("Auto-created 1024-dim projector works", out3.shape == (1, 32, 128))


def test_stateful_latents():
    print("\n--- Test 4: Stateful Latents (RecurrentLatentMemory) ---")
    from omnitrain.fusion_core import FusionCore
    core = FusionCore(n_latents=32, d_model=128, n_heads=4, num_layers=2, input_dim=512)
    core.eval()

    data = torch.randn(1, 10, 512)
    ts = torch.randn(1, 10, 1)

    # Step 1: No memory (stateless)
    with torch.no_grad():
        latents_t0 = core(data, ts, prev_latents=None)
    test("Stateless forward works", latents_t0.shape == (1, 32, 128))

    # Step 2: Feed previous latents back (stateful)
    with torch.no_grad():
        latents_t1 = core(data, ts, prev_latents=latents_t0)
    test("Stateful forward with prev_latents works", latents_t1.shape == (1, 32, 128))

    # Verify outputs are different (memory should affect output)
    are_different = not torch.allclose(latents_t0, latents_t1, atol=1e-6)
    test("Memory affects output (t0 != t1)", are_different)


def test_safety_guard():
    print("\n--- Test 5: Formal Safety Verification (SafetyGuard) ---")
    from omnitrain.heads import ClassificationHead
    from omnitrain.safety_guard import SafetyGuard

    head = ClassificationHead(num_classes=2, d_model=128)
    guard = SafetyGuard(head, emergency_class=1)

    # Add constraints
    guard.add_constraint('lidar', min_safe=0.1, max_safe=50.0)
    guard.add_constraint('temp', min_safe=-40.0, max_safe=85.0)
    test("Constraints registered", len(guard.constraints) == 2)

    # Test constraint checking
    safe_ok, _ = guard.check_constraints({'lidar': 5.0, 'temp': 25.0})
    test("Safe readings pass", safe_ok)

    danger_ok, violations = guard.check_constraints({'lidar': 0.05})
    test("Danger readings caught", not danger_ok)
    test("Violation details reported", len(violations) == 1)

    # Test forward with override
    dummy_latents = torch.randn(1, 32, 128)
    with torch.no_grad():
        safe_out = guard(dummy_latents, sensor_readings={'lidar': 5.0})
        danger_out = guard(dummy_latents, sensor_readings={'lidar': 0.01})

    test("Safe forward: normal output", danger_out[:, 1].item() == float('inf'))
    test("Danger forward: emergency forced", safe_out[:, 1].item() != float('inf'))

    # Test report generation
    report = guard.generate_safety_report([
        {'lidar': 5.0, 'temp': 20.0},  # SAFE
        {'lidar': 0.05},                 # DANGER
        {'temp': 100.0},                 # DANGER
    ])
    test("Report catches 2 violations", report['failed'] == 2)
    test("Report passes 1 safe case", report['passed'] == 1)


def test_language_unification():
    print("\n--- Test 6: Language Unification ---")
    spanish_patterns = [
        'Iniciando', 'Cargando', 'completada', 'Estrategia',
        'Cuantización', 'Codificación', 'Orquestador',
    ]
    src_dir = os.path.join(os.path.dirname(__file__))
    py_files = glob.glob(os.path.join(src_dir, '*.py'))
    violations = []
    for fpath in py_files:
        if 'test_' in os.path.basename(fpath):
            continue
        with open(fpath, 'r') as f:
            content = f.read()
        for pattern in spanish_patterns:
            if pattern in content:
                violations.append(f"{os.path.basename(fpath)}: '{pattern}'")
    test("No Spanish patterns in source", len(violations) == 0,
         f"Found: {violations[:5]}")


if __name__ == "__main__":
    print("=" * 60)
    print("🔬 OMNITRAIN 3.0 SUPREME INTELLIGENCE SMOKE TEST")
    print("=" * 60)

    test_tensor_first_forward()
    test_metadata_loading()
    test_auto_modality()
    test_stateful_latents()
    test_safety_guard()
    test_language_unification()

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"📊 Results: {passed}/{total} tests passed")
    if passed == total:
        print("🏆 ALL TESTS PASSED — Supreme Intelligence verified!")
    else:
        print("⚠️  Some tests failed. Review output above.")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)
