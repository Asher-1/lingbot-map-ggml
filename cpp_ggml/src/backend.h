#pragma once

#include <string>

struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

namespace lingbot {

class backend {
public:
    // vulkan_fast: opt out of the exact scalar F32 Vulkan paths (non-parity).
    // vulkan_integer_dot: enable the exact-integer dp4a q8 GEMM variants.
    // vulkan_fa_coopmat: flash mode only — keep the GEMMs scalar but let the
    //   flash-attention dispatch use the coopmat1 f32acc pipeline, whose P
    //   hi/lo split in the shader holds 1e-4 parity at ~1.7x the scalar FA
    //   speed (validated in validation_report.md).
    // All three drive the GGML_VK_* environment bridge below — ggml v0.21
    // only exposes these toggles through the environment, so the setenv here
    // is an implementation detail of this bridge, driven by explicit options.
    bool init(const std::string & name, int threads,
              bool vulkan_fast = false, bool vulkan_integer_dot = false,
              bool vulkan_fa_coopmat = false);
    void release();
    ggml_backend_t get() const { return backend_; }
    const std::string & error() const { return error_; }

private:
    ggml_backend_t backend_ = nullptr;
    std::string error_;
};

} // namespace lingbot
