import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os


def quantize_omnitrain_mixed(input_model="omni_1_0_edge.onnx", output_model="omni_2_0_quant.onnx", target="default"):
    """
    Apply mixed-precision industrial quantization:
    - Backbone (Transformer/CfC) -> INT8 (Speed/Size)
    - Safety Head -> FP32 (Safety/Fidelity)
    
    Targets:
      - 'default': Standard dynamic quantization for generic edge CPUs.
      - 'qualcomm_hexagon': Strict INT8 rules for SNPE/QNN compatibility.
      - 'jetson_nano_fp16': FP16 pipeline (Maxwell GPU has no INT8 Tensor Cores).
    """

    if not os.path.exists(input_model):
        print(f"❌ Error: Base model not found: {input_model}")
        return

    print(f"💎 Starting Mixed-Precision Quantization on {input_model} (Target: {target})...")

    if target == 'jetson_nano_fp16':
        print(f"⚠️  [JETSON NANO] Skipping INT8 quantization.")
        print(f"NVIDIA Jetson Nano (Maxwell GPU) does NOT have Tensor Cores and does not support hardware INT8.")
        print(f"To optimize for Jetson Nano, you must use TensorRT with FP16 precision.")
        print(f"Run the following command on your Jetson Nano to generate the optimized engine:")
        print(f"\n   trtexec --onnx={input_model} --saveEngine=omni_jetson_fp16.engine --fp16\n")
        return

    model = onnx.load(input_model)

    # Qualcomm Hexagon DSP is highly optimized for INT8/INT16 MatMul and Gemm.
    op_types_to_quantize = ['MatMul', 'Gemm'] if target == 'qualcomm_hexagon' else ['MatMul']

    # Exclude safety heads to preserve exact barrier function math
    nodes_to_exclude = [n.name for n in model.graph.node if 'safety' in n.name.lower()]

    inferred_model = onnx.shape_inference.infer_shapes(model)
    temp_inferred = "temp_inferred.onnx"
    onnx.save(inferred_model, temp_inferred)

    try:
        if target == 'qualcomm_hexagon':
            # SNPE/QNN prefers symmetric quantization for weights
            extra_opts = {
                'WeightSymmetric': True,
                'MatMulConstWeightOnly': False  # Qualcomm can handle dynamic MatMuls better if fully quantized
            }
        else:
            extra_opts = {
                'ForceQuantizeNoInputCheck': True,
                'MatMulConstWeightOnly': True
            }

        quantize_dynamic(
            model_input=temp_inferred,
            model_output=output_model,
            per_channel=True,
            reduce_range=True,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=op_types_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            extra_options=extra_opts
        )
        os.remove(temp_inferred)

        size_old = os.path.getsize(input_model) / 1e6
        size_new = os.path.getsize(output_model) / 1e6

        print(f"✅ Quantization completed successfully.")
        print(f"📊 Reduction: {size_old:.2f} MB → {size_new:.2f} MB ({(1 - size_new/size_old)*100:.1f}%)")

    except Exception as e:
        print(f"❌ Error during quantization: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="omni_1_0_edge.onnx")
    parser.add_argument("--output", default="omni_2_0_quant.onnx")
    parser.add_argument("--target", default="default", choices=["default", "qualcomm_hexagon", "jetson_nano_fp16"])
    args = parser.parse_args()
    
    quantize_omnitrain_mixed(args.input, args.output, args.target)
