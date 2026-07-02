import torch
import torch.nn as nn
from pathlib import Path


class JetsonExporter:
    """
    OmniTrain → NVIDIA Jetson exporter.
    Exports a trained LiquidFusionCore to ONNX format for TensorRT compilation.

    The CfC network is recurrent: it has a persistent hidden state (h) that
    flows from one timestep to the next. For TensorRT, we export a single-step
    function that takes (sensors, state_in, dt) and produces (action, state_out).
    TensorRT then compiles this graph to run optimally on the Jetson GPU/DLA.

    Workflow:
        1. Python: JetsonExporter.export() → produces model.onnx
        2. Shell:  trtexec --onnx=model.onnx --saveEngine=bot_brain.engine [--fp16]
        3. C++:    OmniEngineTensorRT.LoadEngine("bot_brain.engine") → Step()
    """

    def __init__(self, output_dir: str = "exports/jetson"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, model: nn.Module, input_dim: int, d_model: int,
               output_dim: int, filename: str = "omni_brain.onnx",
               use_fp16: bool = False) -> Path:
        """
        Wraps the model step function in a stateless ONNX-compatible form.

        Args:
            model:      Trained LiquidFusionCore (or any CfC-based model).
            input_dim:  Number of sensor inputs.
            d_model:    Hidden state dimension.
            output_dim: Number of action outputs.
            filename:   Output ONNX filename.
            use_fp16:   If True, cast weights to float16 before export.
                        Note: fp16 TensorRT compilation is done via trtexec, not here.
        """
        out_path = self.output_dir / filename
        model.eval()

        # Create a stateless wrapper that exposes state as explicit I/O
        class StatelessStep(nn.Module):
            """
            Wraps model.step() to make the recurrent state explicit.
            ONNX requires all inputs/outputs to be tensors — no hidden state.
            """
            def __init__(self, core):
                super().__init__()
                self.core = core

            def forward(self, sensors: torch.Tensor,
                        state_in: torch.Tensor,
                        dt: torch.Tensor) -> tuple:
                # sensors:  [1, input_dim]
                # state_in: [1, d_model]
                # dt:       [1]   (scalar time delta)
                action, state_out = self.core.step_stateless(sensors, state_in, dt)
                return action, state_out

        wrapper = StatelessStep(model)

        # Dummy inputs (batch=1 for edge deployment)
        dummy_sensors  = torch.zeros(1, input_dim,  dtype=torch.float32)
        dummy_state_in = torch.zeros(1, d_model,    dtype=torch.float32)
        dummy_dt       = torch.tensor([0.01],       dtype=torch.float32)

        print(f"[JetsonExporter] Tracing model for ONNX export...")
        print(f"  Input dim:  {input_dim}")
        print(f"  Model dim:  {d_model}")
        print(f"  Output dim: {output_dim}")

        torch.onnx.export(
            wrapper,
            (dummy_sensors, dummy_state_in, dummy_dt),
            str(out_path),
            opset_version=17,
            input_names=["sensors", "state_in", "dt"],
            output_names=["action", "state_out"],
            # Dynamic axes allow inference on variable batch sizes (for testing)
            dynamic_axes={
                "sensors":   {0: "batch"},
                "state_in":  {0: "batch"},
                "action":    {0: "batch"},
                "state_out": {0: "batch"},
            },
            do_constant_folding=True,  # Fold constant subgraphs for faster TRT compile
        )

        size_kb = out_path.stat().st_size / 1024.0
        print(f"[JetsonExporter] ONNX exported to: {out_path} ({size_kb:.1f} KB)")
        print()
        print("[JetsonExporter] Next steps on the Jetson:")
        print("  # Convert to TensorRT engine (FP32):")
        print(f"  trtexec --onnx={filename} --saveEngine=bot_brain.engine")
        print()
        print("  # For FP16 (2x speedup on Jetson Orin, minor accuracy loss):")
        print(f"  trtexec --onnx={filename} --saveEngine=bot_brain_fp16.engine --fp16")
        print()
        print("  # For INT8 (4x speedup, requires calibration dataset):")
        print(f"  trtexec --onnx={filename} --saveEngine=bot_brain_int8.engine --int8")

        return out_path

    @staticmethod
    def verify_onnx(onnx_path: str):
        """Quick sanity check that the exported ONNX graph is valid."""
        try:
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            print(f"[JetsonExporter] ✅ ONNX model is valid: {onnx_path}")
        except ImportError:
            print("[JetsonExporter] Install onnx to verify: pip install onnx")
        except Exception as e:
            print(f"[JetsonExporter] ❌ ONNX validation failed: {e}")
