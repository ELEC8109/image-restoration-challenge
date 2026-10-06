"""The submission interface, independent of the starter architecture."""
import math

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image


def run_session(session, x):
    y = session.run(["output"], {"input": np.ascontiguousarray(x, dtype=np.float32)})[0]
    if y.shape != x.shape or y.dtype != np.float32 or not np.isfinite(y).all():
        raise ValueError("Expected a finite float32 output with the same shape as input")
    if y.min() < -1e-5 or y.max() > 1+1e-5:
        raise ValueError("Clip the exported model's output to [0, 1]")
    return np.clip(y, 0, 1)


def _stored_model_elements(message):
    """Count storage once, including subgraphs/functions and sparse values/indices.

    Tensor data is never expanded into NumPy arrays just to count it.
    """
    if isinstance(message, onnx.TensorProto):
        if message.data_location == onnx.TensorProto.EXTERNAL or message.external_data:
            raise ValueError("External tensor data is not allowed; submit one self-contained ONNX file")
        return math.prod(message.dims)
    total = 0
    if isinstance(message, onnx.NodeProto) and message.op_type == "Constant" and message.domain in ("", "ai.onnx"):
        for attr in message.attribute:
            if attr.type in (onnx.AttributeProto.FLOAT, onnx.AttributeProto.INT, onnx.AttributeProto.STRING):
                total += 1
            elif attr.type in (onnx.AttributeProto.FLOATS, onnx.AttributeProto.INTS, onnx.AttributeProto.STRINGS):
                total += len(attr.floats) + len(attr.ints) + len(attr.strings)
    for field, value in message.ListFields():
        if field.message_type is not None:
            children = (value,) if hasattr(value, "ListFields") else value
            total += sum(_stored_model_elements(child) for child in children)
    return total


def check(path, spec, announce=True):
    """Return (stored elements, checked CPU session); callers reuse the session."""
    graph = onnx.load(str(path), load_external_data=False)
    default_opsets = [int(import_.version) for import_ in graph.opset_import
                      if import_.domain in ("", "ai.onnx")]
    if default_opsets != [int(spec["opset"])]:
        raise ValueError(f"ONNX opset must be exactly {spec['opset']}; found {default_opsets}")
    stored_elements = _stored_model_elements(graph)
    if stored_elements > spec["max_model_elements"]:
        raise ValueError(f"stored model tensor elements {stored_elements:,} exceed "
                         f"the limit {spec['max_model_elements']:,}")
    shape = (1, 3, spec["image_size"], spec["image_size"])
    for tensors, name in ((graph.graph.input, "input"), (graph.graph.output, "output")):
        if len(tensors) != 1 or tensors[0].name != name:
            raise ValueError(f"Expected exactly one tensor named '{name}'")
        tensor = tensors[0].type.tensor_type
        if tensor.elem_type != onnx.TensorProto.FLOAT or tuple(d.dim_value for d in tensor.shape.dim) != shape:
            raise ValueError(f"Expected one float32 tensor named '{name}' with shape {shape}")
    onnx.checker.check_model(graph)
    if announce:
        print(f"Stored model tensor elements: {stored_elements:,} / {spec['max_model_elements']:,}", flush=True)
        print("Checking CPU inference (zero / one / random)...", flush=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(graph.SerializeToString(), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for x in (np.zeros(shape, np.float32), np.ones(shape, np.float32), rng.random(shape, dtype=np.float32)):
        y = run_session(session, x)
        if not np.allclose(y, run_session(session, x), rtol=1e-5, atol=1e-6):
            raise ValueError("Use deterministic inference (evaluation mode)")
    if announce:
        print(f"ONNX checks passed: opset {spec['opset']}, {shape}, RGB float32", flush=True)
    return stored_elements, session


def score_pair(session, input_path, target_path):
    """Shared PNG -> RGB -> CPU ONNX -> full-image PSNR pipeline."""
    with Image.open(input_path) as im:
        x = np.asarray(im.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)[None] / 255
    with Image.open(target_path) as im:
        target = np.asarray(im.convert("RGB"), dtype=np.float64).transpose(2, 0, 1)[None] / 255
    if x.shape != target.shape:
        raise ValueError(f"Input and target dimensions differ: {input_path.name}")
    scores = []
    for prediction in (x, run_session(session, x)):
        mse = float(np.mean((prediction.astype(np.float64) - target)**2))
        scores.append(float(-10 * np.log10(max(mse, 1e-12))))
    return scores  # Input PSNR, restored PSNR; no quality threshold.
