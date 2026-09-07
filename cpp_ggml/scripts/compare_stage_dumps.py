#!/usr/bin/env python3
"""Compare matching C++ GGML and independent PyTorch stage dump tensors."""
import argparse
from pathlib import Path

import numpy as np


# C++ stage names deliberately describe graph boundaries.  PyTorch hook names
# follow the original module path, so keep the translation explicit and auditable.
STAGE_KEYS = {
    "dino.input": "dino_input",
    "dino.block0": "dino_block0",
    "dino.block23": "dino_block23",
    "frame.input": "frame_input",
    "frame.block0": "frame_block0",
    "frame.block23": "frame_block23",
    "global.block0": "global_block0",
    "global.block23": "global_block23",
    "dpt.project0": "dpt_projects_0",
    "dpt.project1": "dpt_projects_1",
    "dpt.project2": "dpt_projects_2",
    "dpt.project3": "dpt_projects_3",
    "dpt.resize0": "dpt_resize_layers_0",
    "dpt.resize1": "dpt_resize_layers_1",
    "dpt.resize2": "dpt_resize_layers_2",
    "dpt.resize3": "dpt_resize_layers_3",
    "dpt.rn0": "dpt_scratch_layer1_rn",
    "dpt.rn1": "dpt_scratch_layer2_rn",
    "dpt.rn2": "dpt_scratch_layer3_rn",
    "dpt.rn3": "dpt_scratch_layer4_rn",
    "dpt.refine3": "dpt_scratch_refinenet4",
    "dpt.refine2": "dpt_scratch_refinenet3",
    "dpt.refine1": "dpt_scratch_refinenet2",
    "dpt.refine0": "dpt_scratch_refinenet1",
    "dpt.output_conv1": "dpt_scratch_output_conv1",
    "dpt.output_conv2_0": "dpt_scratch_output_conv2_0",
    "dpt.logits": "dpt_scratch_output_conv2_2",
}
for _i in range(24):
    STAGE_KEYS.setdefault(f"dino.block{_i}", f"dino_block{_i}")
    STAGE_KEYS.setdefault(f"frame.block{_i}", f"frame_block{_i}")
    STAGE_KEYS.setdefault(f"global.block{_i}", f"global_block{_i}")
STAGE_KEYS["camera.token"] = "camera_token"
STAGE_KEYS["camera.adaln"] = "camera_adaln"
for _iteration in range(4):
    for _stage in ("embed", "embed_silu", "mod"):
        STAGE_KEYS[f"camera.iter{_iteration}.{_stage}"] = f"camera_iter{_iteration}_{_stage}"
    STAGE_KEYS[f"camera.iter{_iteration}.input"] = f"camera_iter{_iteration}_input"
    STAGE_KEYS[f"camera.iter{_iteration}.delta"] = f"camera_iter{_iteration}_delta"
    for _block in range(4):
        STAGE_KEYS[f"camera.iter{_iteration}.block{_block}"] = f"camera_iter{_iteration}_block{_block}"

DEFAULT_STAGES = (
    ["dino.input"] + [f"dino.block{i}" for i in range(24)] +
    ["frame.input"] + [f"frame.block{i}" for i in range(24)] +
    [f"global.block{i}" for i in range(24)] +
    ["dpt.project0", "dpt.project1", "dpt.project2", "dpt.project3",
     "dpt.resize0", "dpt.resize1", "dpt.resize2", "dpt.resize3",
     "dpt.rn0", "dpt.rn1", "dpt.rn2", "dpt.rn3",
     "dpt.refine3", "dpt.refine2", "dpt.refine1", "dpt.refine0",
     "dpt.output_conv1", "dpt.output_conv2_0", "dpt.logits"] +
    ["camera.token"] +
    ["camera.adaln"] +
    [f"camera.iter{_iteration}.{_stage}" for _iteration in range(4) for _stage in ("embed", "embed_silu", "mod")] +
    [f"camera.iter{_iteration}.input" for _iteration in range(4)] +
    [f"camera.iter{_iteration}.block{_block}" for _iteration in range(4) for _block in range(4)] +
    [f"camera.iter{_iteration}.delta" for _iteration in range(4)]
)


def metrics(cpp: np.ndarray, torch: np.ndarray):
    delta = cpp - torch
    absolute = np.abs(delta)
    if cpp.size > 1 and np.std(cpp) and np.std(torch):
        corr = float(np.corrcoef(cpp, torch)[0, 1])
    else:
        corr = float("nan")
    return (
        float(np.mean(absolute)),
        float(np.sqrt(np.mean(delta * delta))),
        float(np.quantile(absolute, 0.99)),
        float(np.max(absolute)),
        corr,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Report layer boundaries from LINGBOT_DUMP_STAGES and --dump-stages.")
    parser.add_argument("cpp_prefix", type=Path,
                        help="prefix passed as LINGBOT_DUMP_STAGES, for example /tmp/cpp-stage")
    parser.add_argument("torch_npz", type=Path)
    parser.add_argument("--frame", type=int, required=True,
                        help="zero-based inference index selected by LINGBOT_DUMP_AT")
    parser.add_argument("--only", action="append", choices=sorted(STAGE_KEYS),
                        help="compare only this C++ stage; repeatable")
    args = parser.parse_args()

    reference = np.load(args.torch_npz)
    stages = args.only or DEFAULT_STAGES
    print("stage\tshape\tmae\trmse\tp99_abs\tmax_abs\tcorr")
    missing = []
    for stage in stages:
        cpp_path = Path(f"{args.cpp_prefix}.{stage}.{args.frame}")
        torch_key = STAGE_KEYS[stage]
        if not cpp_path.is_file() or torch_key not in reference:
            missing.append(f"{stage} (cpp={cpp_path.is_file()} torch={torch_key in reference})")
            continue
        cpp = np.fromfile(cpp_path, dtype=np.float32)
        torch = np.asarray(reference[torch_key], dtype=np.float32).reshape(-1)
        if cpp.size != torch.size:
            raise SystemExit(
                f"{stage}: element count mismatch cpp={cpp.size} torch={torch.size}; "
                "the selected graph boundary is not equivalent")
        mae, rmse, p99, max_abs, corr = metrics(cpp, torch)
        print(f"{stage}\t{cpp.size}\t{mae:.8g}\t{rmse:.8g}\t{p99:.8g}\t{max_abs:.8g}\t{corr:.8g}")
    if missing:
        raise SystemExit("missing stage dumps: " + "; ".join(missing))


if __name__ == "__main__":
    main()
