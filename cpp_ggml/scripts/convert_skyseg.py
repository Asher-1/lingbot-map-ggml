#!/usr/bin/env python3
"""Convert the official skyseg.onnx sky-segmentation model to GGUF.

The official pipeline runs skyseg through onnxruntime (lingbot_map/vis/
sky_segmentation.py). This converter bakes the same network into a GGUF so
the C++ GGML runtime can run it end to end with no onnxruntime dependency:

    python3 convert_skyseg.py <skyseg.onnx> <out.gguf> [--outtype f32|f16|q8_0]

Pipeline: onnx-simplifier constant-folds the dynamic shape arithmetic
(Shape/Gather/Unsqueeze/Cast/Slice all disappear at the fixed 320x320 input),
then the graph is emitted as a flat instruction list stored in the GGUF
metadata (`skyseg.graph`), which src/skyseg.cpp interprets to build the ggml
graph. Weights are stored directly in the ggml im2col layout so inference
needs no runtime permutes:

    conv kernel [O,I,KH,KW] (ONNX)  ->  ggml ne {KW,KH,IC,OC} 4-D (conv_2d),
        or 2-D {K=IC*KH*KW, OC} for q8_0 (mul_mat operand)
    conv input  [N,I,H,W]           ->  ggml ne {IW, IH, I, N} (built at runtime)
    conv output via im2col+mul_mat  ->  ggml ne {OW, OH, O, N}

Quantization applies to weights only (f16 / q8_0); activations stay f32.
"""
import argparse

import gguf
import numpy as np
import onnx
import onnx.numpy_helper as nh


def simplify_model(path: str) -> onnx.ModelProto:
    from onnxsim import simplify
    m = onnx.load(path)
    inp = m.graph.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    assert dims == [1, 3, 320, 320], f"unexpected skyseg input shape {dims}"
    m_simp, ok = simplify(m, overwrite_input_shapes={"input.1": dims})
    if not ok:
        raise RuntimeError("onnx-simplifier failed")
    return onnx.shape_inference.infer_shapes(m_simp)


def quantize_q8_0(x: np.ndarray) -> np.ndarray:
    """GGML block_q8_0: groups of 32, (f16 scale, 32 x int8) = 34 bytes."""
    assert x.dtype == np.float32
    n = x.size
    pad = (-n) % 32
    if pad:
        x = np.concatenate([x.ravel(), np.zeros(pad, np.float32)])
    x = x.reshape(-1, 32)
    amax = np.abs(x).max(axis=1)
    scale = (amax / 127.0).astype(np.float16)
    scale_f = scale.astype(np.float32)
    q = np.clip(np.round(x / np.where(scale_f == 0, 1.0, scale_f)[:, None]),
                -127, 127).astype(np.int8)
    packed = np.zeros((x.shape[0], 34), np.uint8)
    packed[:, 0:2] = scale.view(np.uint8).reshape(-1, 2)
    packed[:, 2:] = q.view(np.uint8)
    return packed.ravel()


class GraphEmitter:
    """Emit flat instructions for src/skyseg.cpp and collect weights."""

    def __init__(self):
        self.lines = []          # instruction lines (one op per line)
        self.weights = {}        # name -> (np.ndarray, dtype tag)
        self.const_nodes = {}    # Constant-node outputs (post-simplify biases)
        self.output = None

    def wname(self, node_name):
        return node_name + ".weight"

    def const_array(self, name):
        """Value of an initializer or a Constant node output (simplified
        graphs often move conv bias into Constant nodes)."""
        if name in self.init:
            return nh.to_array(self.init[name])
        n = self.const_nodes.get(name)
        if n is not None:
            a = {at.name: onnx.helper.get_attribute_value(at) for at in n.attribute}
            v = a.get("value")
            return nh.to_array(v) if v is not None else None
        return None

    def emit_conv(self, n, shapes):
        a = {at.name: onnx.helper.get_attribute_value(at) for at in n.attribute}
        k = a.get("kernel_shape"); s = a.get("strides"); p = a.get("pads")
        d = a.get("dilations", [1, 1])
        assert s in ([1, 1], None) and a.get("group", 1) == 1
        w = self.const_array(n.input[1])                  # [O, I, KH, KW]
        assert w is not None and w.dtype == np.float32
        # F32/F16 keep the native 4-D layout (ggml ne {KW,KH,IC,OC}) for
        # ggml_conv_2d. Q8_0 quantizes along the contiguous axis, so those
        # kernels are stored as a 2-D [OC, K=IC*KH*KW] mul_mat operand
        # (K laid out (ic,kh,kw) — the same order ggml_im2col produces);
        # layers whose K is not a multiple of 32 fall back to f16.
        K_total = int(w.shape[1] * w.shape[2] * w.shape[3])
        if self.outtype == "f32":
            self.weights[self.wname(n.output[0])] = (w.copy(), "f32")
        elif self.outtype == "q8_0" and K_total % 32 == 0:
            self.weights[self.wname(n.output[0])] = (
                w.reshape(w.shape[0], -1).copy(), "q8_0")
        elif self.outtype == "q8_0":
            self.weights[self.wname(n.output[0])] = (
                w.reshape(w.shape[0], -1).copy(), "f16")
        else:  # f16: 2-D mul_mat operand (K layout (ic,kh,kw))
            self.weights[self.wname(n.output[0])] = (
                w.reshape(w.shape[0], -1).copy(), "f16")
        bias = self.const_array(n.input[2]) if len(n.input) > 2 else None
        if bias is not None:
            self.weights[n.output[0] + ".bias"] = (bias.astype(np.float32), "f32")
        shp = shapes.get(n.output[0])
        _, _, oh, ow = shp
        # conv <out> <in> <k> <ph> <pw> <dh> <dw> <oh> <ow> <has_bias>
        self.lines.append("conv {} {} {} {} {} {} {} {} {} {}".format(
            n.output[0], n.input[0], k[0], p[0], p[1], d[0], d[1], oh, ow,
            1 if bias is not None else 0))

    def emit(self, n, shapes):
        op = n.op_type
        o = n.output[0]
        if op == "Conv":
            self.emit_conv(n, shapes)
        elif op == "Relu":
            self.lines.append(f"relu {o} {n.input[0]}")
        elif op == "MaxPool":
            a = {at.name: onnx.helper.get_attribute_value(at) for at in n.attribute}
            assert a.get("kernel_shape") == [2, 2] and a.get("strides") == [2, 2] and a.get("pads") in ([0, 0, 0, 0], None)
            self.lines.append(f"maxpool {o} {n.input[0]}")
        elif op == "Resize":
            a = {at.name: onnx.helper.get_attribute_value(at) for at in n.attribute}
            mode = a.get("mode", b"")
            mode = mode.decode() if isinstance(mode, bytes) else mode
            coord = a.get("coordinate_transformation_mode", b"")
            coord = coord.decode() if isinstance(coord, bytes) else coord
            assert mode == "linear" and coord == "pytorch_half_pixel", (mode, coord)
            _, _, oh, ow = shapes[o]
            self.lines.append(f"resize {o} {n.input[0]} {oh} {ow}")
        elif op == "Concat":
            a = {at.name: onnx.helper.get_attribute_value(at) for at in n.attribute}
            assert a.get("axis") == 1, f"concat axis {a.get('axis')}"
            # ggml_concat is binary; unfold n-ary concat into a chain along C
            prev = n.input[0]
            for k, inp in enumerate(n.input[1:], 1):
                out = o if k == len(n.input) - 1 else f"{o}.cat{k}"
                self.lines.append(f"concat {out} {prev} {inp}")
                prev = out
        elif op == "Add":
            self.lines.append(f"add {o} {n.input[0]} {n.input[1]}")
        elif op == "Sigmoid":
            self.lines.append(f"sigmoid {o} {n.input[0]}")
        else:
            raise RuntimeError(f"unsupported op after simplification: {op}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="skyseg.onnx path")
    ap.add_argument("out", help="output GGUF path")
    ap.add_argument("--outtype", choices=["f32", "f16", "q8_0"], default="f16")
    args = ap.parse_args()

    m = simplify_model(args.model)
    g = m.graph
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in g.value_info}
    shapes.update({i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
                   for i in g.input})
    shapes.update({o.name: [d.dim_value for d in o.type.tensor_type.shape.dim]
                   for o in g.output})
    inits = {i.name: i for i in g.initializer}

    em = GraphEmitter()
    em.init = inits
    em.outtype = args.outtype
    em.const_nodes = {o: n for n in g.node if n.op_type == "Constant"
                      for o in n.output}
    prod = {o: n for n in g.node for o in n.output}
    # topological walk from the input following production order
    done = {"input.1"}
    pending = list(g.node)
    while pending:
        progressed = False
        rest = []
        for n in pending:
            if all(i in done or i in inits for i in n.input):
                em.emit(n, shapes)
                done.add(n.output[0])
                progressed = True
            else:
                rest.append(n)
        pending = rest
        if not progressed:
            raise RuntimeError("cycle / unresolved inputs: " + str(pending[:1]))
    em.output = g.output[0].name

    wtype = {"f32": gguf.GGMLQuantizationType.F32,
             "f16": gguf.GGMLQuantizationType.F16,
             "q8_0": gguf.GGMLQuantizationType.Q8_0}[args.outtype]

    w = gguf.GGUFWriter(args.out, "skyseg")
    w.add_string("skyseg.graph", "\n".join(em.lines))
    w.add_string("skyseg.input", "input.1")
    w.add_string("skyseg.output", em.output)
    w.add_string("skyseg.input_shape", "3,320,320")
    w.add_uint32("skyseg.version", 1)
    n_q8 = n_f16fb = 0
    for name, (arr, tag) in em.weights.items():
        if tag == "f32":
            w.add_tensor(name, np.ascontiguousarray(arr, np.float32))
        elif tag == "f16":
            w.add_tensor(name, np.ascontiguousarray(arr, np.float16),
                         raw_dtype=gguf.GGMLQuantizationType.F16)
            n_f16fb += 1
        else:  # q8_0 2-D [OC, K]
            oc, kdim = arr.shape
            blocks = kdim // 32
            packed = quantize_q8_0(arr).reshape(oc, blocks * 34)
            # raw_shape is the BYTE shape; gguf-py converts it back to the
            # logical (OC, K) shape via quant_shape_from_byte_shape
            w.add_tensor(name, packed, raw_shape=packed.shape,
                         raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            n_q8 += 1
    if em.outtype == "q8_0":
        print(f"q8_0 tensors: {n_q8}, f16 conv_2d fallback: {n_f16fb}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    n_conv = sum(1 for l in em.lines if l.startswith("conv"))
    total = sum(arr.size for arr, _ in em.weights.values())
    print(f"wrote {args.out}: {len(em.lines)} ops ({n_conv} conv), "
          f"{total/1e6:.2f}M params, {args.outtype}")


if __name__ == "__main__":
    main()
