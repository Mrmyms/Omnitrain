import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os


def quantize_omnitrain_mixed(input_model="omni_1_0_edge.onnx", output_model="omni_2_0_quant.onnx"):
    """
    Apply mixed-precision industrial quantization:
    - Backbone (Transformer) -> INT8 (Speed/Size)
    - Safety Head -> FP32 (Safety/Fidelity)
    """

    if not os.path.exists(input_model):
        print(f"❌ Error: Base model not found: {input_model}")
        return

    print(f"💎 Starting Mixed-Precision Quantization on {input_model}...")

    # 1. Critical Preservation Strategy (Safety First)
    model = onnx.load(input_model)

    # In robotics architectures, 'MatMul' is the Transformer core (90% of compute).
    # 'Gemm' is typically reserved for the final heads.
    # By quantizing ONLY MatMul, we ensure the safety heads maintain
    # absolute FP32 precision with no risk of drift.

    op_types_to_quantize = ['MatMul']

    # Dynamic scan for double safety
    nodes_to_exclude = [n.name for n in model.graph.node if 'safety' in n.name.lower()]

    inferred_model = onnx.shape_inference.infer_shapes(model)
    temp_inferred = "temp_inferred.onnx"
    onnx.save(inferred_model, temp_inferred)

    try:
        quantize_dynamic(
            model_input=temp_inferred,
            model_output=output_model,
            per_channel=True,
            reduce_range=True,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=op_types_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            extra_options={
                'ForceQuantizeNoInputCheck': True,
                'MatMulConstWeightOnly': True
            }
        )
        os.remove(temp_inferred)

        size_old = os.path.getsize(input_model) / 1e6
        size_new = os.path.getsize(output_model) / 1e6

        print(f"✅ Quantization completed successfully.")
        print(f"📊 Reduction: {size_old:.2f} MB → {size_new:.2f} MB ({(1 - size_new/size_old)*100:.1f}%)")

    except Exception as e:
        print(f"❌ Error during quantization: {e}")


if __name__ == "__main__":
    quantize_omnitrain_mixed()
