#!/usr/bin/env python3
"""Run the released LingBot-Map checkpoint and dump independent outputs."""
import argparse
import contextlib
from pathlib import Path
import sys
import numpy as np
import torch


def load_gguf_state(path: Path):
    """Decode a LingBot GGUF into the checkpoint state-dict namespace.

    GGML casts its f16 and quantized tensors to f32 before the graph operators
    in this port.  The reference must therefore do the same: this isolates
    graph/operator parity from the intentional information loss of a deployment
    weight format.
    """
    from gguf import GGUFReader, GGMLQuantizationType
    from gguf.quants import dequantize

    reader = GGUFReader(str(path))
    state = {}
    for tensor in reader.tensors:
        qtype = GGMLQuantizationType(tensor.tensor_type)
        values = dequantize(tensor.data, qtype)
        state[tensor.name] = torch.from_numpy(np.asarray(values, dtype=np.float32).copy())
    return state

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("images", type=Path, help=".npy with [B,S,3,H,W] float32 in [0,1]")
    ap.add_argument("output", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--streaming", action="store_true", help="match GCT inference_streaming cache semantics")
    ap.add_argument("--scale-frames", type=int, default=1)
    ap.add_argument("--kv-cache-scale", type=int, default=1)
    ap.add_argument("--kv-cache-window", type=int, default=4)
    ap.add_argument("--keyframe-interval", type=int, default=1,
                    help="official keyframe policy: every N-th streaming frame "
                         "persists its KV (non-keyframes attend and discard); "
                         "demo.py auto-selects ceil(N/320) above 320 frames")
    ap.add_argument("--gguf", type=Path,
                    help="load decoded GGUF weights instead of the f32 checkpoint state")
    ap.add_argument("--autocast-f16", action="store_true",
                    help="run CUDA matmul/conv under f16 autocast to match GGML f16 kernels")
    ap.add_argument("--autocast-bf16", action="store_true",
                    help="run under bf16 autocast to match the official deployment precision")
    ap.add_argument("--allow-tf32", action="store_true",
                    help="allow CUDA TF32 kernels; disabled by default for F32 GGML parity")
    ap.add_argument("--enable-3d-rope", action="store_true",
                    help="build the model with the official demo's 3D (temporal+spatial) global RoPE. "
                         "Off by default because the GGML graph implements the 2D spatial RoPE path; "
                         "use this only to audit the deployment configuration itself")
    ap.add_argument("--dump-global-last", action="store_true",
                    help="write per-frame final global-block features for C++ parity diagnosis")
    ap.add_argument("--dump-stages", action="store_true",
                    help="write DINO and GCT boundary tensors for C++ parity diagnosis")
    ap.add_argument("--dump-camera-stages", action="store_true",
                    help="write only CameraCausalHead boundary tensors for bounded-memory diagnosis")
    ap.add_argument("--dump-camera-internals", action="store_true",
                    help="write CameraCausalHead trunk block-0 operator boundaries")
    ap.add_argument("--dump-dpt-stages", action="store_true",
                    help="write DPT hook outputs from the final streaming frame")
    ap.add_argument("--dump-internals", action="store_true",
                    help="write block-0 internal residual boundaries for diagnosis")
    args = ap.parse_args()
    x = np.load(args.images).astype(np.float32)
    if x.ndim == 4: x = x[None]
    if x.ndim != 5 or x.shape[2] != 3: raise SystemExit("images must have shape [B,S,3,H,W]")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # The C++ GGML graph uses cuBLAS only and its F32 CUDA kernels do not
        # use TF32 or cuDNN.  Keep the independent reference on that same
        # contract, including during reconstruction validation.
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.enabled = False
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from lingbot_map.models.gct_stream import GCTStream
    # The exported checkpoint carries the 518x518 DINO positional grid, so the
    # reference is built at that geometry regardless of the input resolution;
    # inputs of other sizes exercise DINOv2's interpolate_pos_encoding at
    # runtime, exactly like the deployment model. Sizing the model from the
    # input instead would build a mismatched pos_embed and reject the GGUF.
    model = GCTStream(img_size=518, patch_size=14, use_sdpa=True,
                      enable_3d_rope=args.enable_3d_rope, camera_num_iterations=4,
                      kv_cache_scale_frames=args.kv_cache_scale,
                      kv_cache_sliding_window=args.kv_cache_window)
    if args.gguf:
        if not args.gguf.is_file():
            raise SystemExit(f"GGUF checkpoint not found: {args.gguf}")
        state = load_gguf_state(args.gguf)
    else:
        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = raw.get("model", raw)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(device)
    # inference_streaming slices CPU input and uploads one frame at a time.
    # Keeping the entire sequence on CUDA defeats that bounded-memory contract.
    inp = torch.from_numpy(x)
    dpt_raw_holder = {}
    global_last_features = []
    stage_features = {}
    hooks = []
    def capture(name):
        def _hook(_m, _i, o):
            if torch.is_tensor(o):
                value = o.detach().float().cpu()
                if args.dump_dpt_stages:
                    dpt_raw_holder[name] = value
                else:
                    dpt_raw_holder.setdefault(name, value)
        return _hook
    def capture_stage(name):
        def _hook(_m, _i, output):
            if torch.is_tensor(output):
                stage_features[name] = output.detach().float().cpu()
        return _hook
    def capture_stage_input(name):
        def _hook(_m, inp):
            if inp and torch.is_tensor(inp[0]):
                stage_features[name] = inp[0].detach().float().cpu()
        return _hook
    for name, module in model.depth_head.named_modules():
        if name and (name.startswith("projects.") or name.startswith("resize_layers.") or
                     name.startswith("scratch.layer") or name.startswith("scratch.refinenet") or
                     name in ("scratch.output_conv1", "scratch.output_conv2.0", "scratch.output_conv2.2")):
            hooks.append(module.register_forward_hook(capture(name.replace('.', '_'))))
    for i, module in enumerate(model.aggregator.patch_embed.blocks):
        if i == 0:
            def capture_input(_m, inp):
                if inp and torch.is_tensor(inp[0]): dpt_raw_holder.setdefault("dino_block0_input", inp[0].detach().float().cpu())
            hooks.append(module.register_forward_pre_hook(capture_input))
            hooks.append(module.register_forward_hook(capture("dino_block0")))
            if args.dump_stages:
                hooks.append(module.register_forward_pre_hook(capture_stage_input("dino_input")))
                hooks.append(module.register_forward_hook(capture_stage("dino_block0")))
        if args.dump_stages and i != 0:
            hooks.append(module.register_forward_hook(capture_stage(f"dino_block{i}")))
    if args.dump_stages:
        hooks.append(model.aggregator.patch_embed.patch_embed.proj.register_forward_hook(
            capture_stage("dino_patch_conv")))
        for i, module in enumerate(model.aggregator.frame_blocks):
            hooks.append(module.register_forward_hook(capture_stage(f"frame_block{i}")))
            if i == 0:
                hooks.append(module.register_forward_pre_hook(capture_stage_input("frame_input")))
        for i, module in enumerate(model.aggregator.global_blocks):
            hooks.append(module.register_forward_hook(capture_stage(f"global_block{i}")))
    if args.dump_stages or args.dump_camera_stages:
        camera_state = {"iteration": 0, "delta": 0}
        def reset_camera_state(_m, _i):
            camera_state["iteration"] = 0
            camera_state["delta"] = 0
        def capture_camera_block(index):
            def _hook(_m, _i, output):
                if torch.is_tensor(output):
                    stage_features[f"camera_iter{camera_state['iteration']}_block{index}"] = output.detach().float().cpu()
                if index == len(model.camera_head.trunk) - 1:
                    camera_state["iteration"] += 1
            return _hook
        def capture_camera_delta(_m, _i, output):
            if torch.is_tensor(output):
                stage_features[f"camera_iter{camera_state['delta']}_delta"] = output.detach().float().cpu()
                camera_state["delta"] += 1
        hooks.append(model.camera_head.register_forward_pre_hook(reset_camera_state))
        hooks.append(model.camera_head.token_norm.register_forward_hook(capture_stage("camera_token")))
        modulation_state = {"embed": 0, "silu": 0, "mod": 0}
        def capture_camera_modulation(key, stage):
            def _hook(_m, _i, output):
                if torch.is_tensor(output):
                    stage_features[f"camera_iter{modulation_state[key]}_{stage}"] = output.detach().float().cpu()
                    modulation_state[key] += 1
            return _hook
        hooks.append(model.camera_head.embed_pose.register_forward_hook(
            capture_camera_modulation("embed", "embed")))
        hooks.append(model.camera_head.poseLN_modulation[0].register_forward_hook(
            capture_camera_modulation("silu", "embed_silu")))
        hooks.append(model.camera_head.poseLN_modulation[1].register_forward_hook(
            capture_camera_modulation("mod", "mod")))
        hooks.append(model.camera_head.adaln_norm.register_forward_hook(capture_stage("camera_adaln")))
        for i, module in enumerate(model.camera_head.trunk):
            if i == 0:
                hooks.append(module.register_forward_pre_hook(
                    lambda _m, inp: stage_features.__setitem__(
                        f"camera_iter{camera_state['iteration']}_input", inp[0].detach().float().cpu())
                    if inp and torch.is_tensor(inp[0]) else None))
            hooks.append(module.register_forward_hook(capture_camera_block(i)))
        hooks.append(model.camera_head.pose_branch.fc2.register_forward_hook(capture_camera_delta))
    if args.dump_camera_internals:
        block = model.camera_head.trunk[0]
        hooks.append(block.attn.qkv.register_forward_hook(capture_stage("internal_camera_0_qkv")))
        for name, module in (
            ("norm1", block.norm1), ("attn", block.attn), ("ls1", block.ls1),
            ("norm2", block.norm2), ("fc1", block.mlp.fc1), ("gelu", block.mlp.act),
            ("fc2", block.mlp.fc2), ("ls2", block.ls2)):
            hooks.append(module.register_forward_hook(capture_stage(f"internal_camera_0_{name}")))
        hooks.append(block.norm2.register_forward_pre_hook(
            capture_stage_input("internal_camera_0_attn_residual")))
    if args.dump_internals:
        target = 0
        for family, blocks in (("dino", model.aggregator.patch_embed.blocks),
                               ("frame", model.aggregator.frame_blocks),
                               ("global", model.aggregator.global_blocks)):
            if target >= len(blocks):
                continue
            block = blocks[target]
            # qkv is captured separately from the attention module output so
            # a layout/linear-kernel mismatch can be distinguished from the
            # score/softmax implementation.
            hooks.append(block.attn.qkv.register_forward_hook(
                capture_stage(f"internal_{family}_{target}_qkv")))
            for name, module in (
                ("norm1", block.norm1), ("attn", block.attn), ("ls1", block.ls1),
                ("norm2", block.norm2), ("fc1", block.mlp.fc1),
                ("gelu", block.mlp.act), ("fc2", block.mlp.fc2),
                ("ls2", block.ls2)):
                hooks.append(module.register_forward_hook(
                    capture_stage(f"internal_{family}_{target}_{name}")))
            hooks.append(block.norm2.register_forward_pre_hook(
                capture_stage_input(f"internal_{family}_{target}_attn_residual")))
            hooks.append(block.mlp.fc1.register_forward_pre_hook(
                capture_stage_input(f"internal_{family}_{target}_norm2")))
    if args.dump_global_last:
        def capture_global_last(_m, _i, output):
            if torch.is_tensor(output):
                global_last_features.append(output.detach().float().cpu())
        hooks.append(model.aggregator.global_blocks[-1].register_forward_hook(capture_global_last))
    if args.autocast_f16 and device.type != "cuda":
        raise SystemExit("--autocast-f16 requires CUDA")
    if args.autocast_f16 and args.autocast_bf16:
        raise SystemExit("--autocast-f16 and --autocast-bf16 are mutually exclusive")
    if args.autocast_bf16:
        precision_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif args.autocast_f16:
        precision_context = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        precision_context = contextlib.nullcontext()
    with torch.no_grad(), precision_context:
        if args.streaming:
            predictions = model.inference_streaming(
                inp, num_scale_frames=min(args.scale_frames, inp.shape[1]),
                keyframe_interval=args.keyframe_interval, output_device=torch.device("cpu"))
            pose = predictions["pose_enc"].float().numpy()
            depth = predictions["depth"][..., 0].float().numpy()
            depth_conf = predictions["depth_conf"].float().numpy()
            payload = dict(pose_enc=pose, depth=depth, depth_conf=depth_conf)
            if args.dump_global_last:
                payload["global_last"] = torch.cat(global_last_features, dim=0).numpy()
        else:
            inp = inp.to(device)
            norm_inp = (inp - model.aggregator._resnet_mean) / model.aggregator._resnet_std
            norm_flat = norm_inp.view(-1, 3, inp.shape[-2], inp.shape[-1])
            patch_raw = model.aggregator.patch_embed.patch_embed.proj(norm_flat)
            patch_embed = model.aggregator.patch_embed(norm_flat)
            if isinstance(patch_embed, dict): patch_embed = patch_embed["x_norm_patchtokens"]
            aggregated, patch_start = model.aggregator(
                inp, selected_idx=[4, 11, 17, 23], num_frame_for_scale=None,
                sliding_window_size=None, num_frame_per_block=1)
            pose_list = model.camera_head(aggregated, causal_inference=False,
                                          num_frame_for_scale=-1, sliding_window_size=-1,
                                          num_frame_per_block=1)
            depth, depth_conf = model.depth_head(aggregated, inp, patch_start)
            pose = pose_list[-1].detach().float().cpu().numpy()
            if depth.ndim == 5: depth = depth[..., 0]
            payload = dict(pose_enc=pose, depth=depth.detach().float().cpu().numpy(),
                     depth_conf=depth_conf.detach().float().cpu().numpy(),
                     aggregated_last=aggregated[-1].detach().float().cpu().numpy(),
                     aggregated_all=np.stack([t.detach().float().cpu().numpy() for t in aggregated]),
                     patch_embed=patch_embed.detach().float().cpu().numpy(),
                     patch_raw=patch_raw.detach().float().cpu().numpy(),
                     dpt_raw=dpt_raw_holder["scratch_output_conv2_2"].numpy())
    # Intermediate hooks are useful while auditing the non-stream graph, but
    # serialising them for a real streaming sequence needlessly turns a small
    # parity reference into a multi-gigabyte artifact.
    if not args.streaming:
        payload.update({k: v.numpy() for k, v in dpt_raw_holder.items()})
    if args.dump_stages or args.dump_camera_stages:
        payload.update({k: v.numpy() for k, v in stage_features.items()})
    if args.dump_internals or args.dump_camera_internals:
        payload.update({k: v.numpy() for k, v in stage_features.items()})
    if args.dump_dpt_stages:
        payload.update({f"dpt_{k}": v.numpy() for k, v in dpt_raw_holder.items()})
    np.savez(args.output, **payload)
    for hook in hooks: hook.remove()
    source = f"gguf={args.gguf}" if args.gguf else f"checkpoint={args.checkpoint}"
    if args.autocast_f16:
        source += " autocast=f16"
    print(f"wrote {args.output}: pose={pose.shape} depth={depth.shape} device={device} {source}")

if __name__ == "__main__": main()
