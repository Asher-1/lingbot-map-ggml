#!/usr/bin/env bash
# One-command GGML 3D reconstruction GUI, callable from inside cpp_ggml/.
#
# Thin entry point so the C++ subproject has its own launcher; the actual logic
# (interpreter probe, GGUF/build resolution, first-run cmake build) lives in the
# repository-root run_gui.sh, which also serves the PyTorch engine so both GUIs
# stay on one code path.
#
#   bash cpp_ggml/scripts/run_gui.sh                                # courthouse, all frames
#   bash cpp_ggml/scripts/run_gui.sh --frames 40                    # quick look
#   bash cpp_ggml/scripts/run_gui.sh --backend CUDA0 --build cpp_ggml/build-cuda
#
# For the official PyTorch engine on the same scene and profile:
#   bash run_gui.sh --engine pytorch
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/run_gui.sh" --engine ggml "$@"
