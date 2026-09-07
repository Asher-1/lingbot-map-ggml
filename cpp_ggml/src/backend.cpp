#include "backend.h"

#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cstdlib>

namespace lingbot {

bool backend::init(const std::string & name, int threads,
                   bool vulkan_fast, bool vulkan_integer_dot,
                   bool vulkan_fa_coopmat) {
    release();
    // The Vulkan cooperative/fp16 kernels use reduced-precision accumulation.
    // GCT compounds that error across 24 streaming blocks, so all modes except
    // flash's FA disable them and run the exact scalar F32 paths. The coopmat
    // experiments of 2026-09-13 are recorded in validation_report.md
    // ("Vulkan speed optimization"): the coopmat GEMMs round activations to
    // F16, the NV coopmat2 FA is TF32-class, and the coopmat1 FA only holds
    // 1e-4 parity with the P hi/lo split (vulkan_fa_coopmat, enabled for the
    // flash mode: measured 286-frame pose/depth in validation_report.md).
    //
    // The GGML_VK_* calls below are an implementation detail of driving
    // ggml v0.21 (which only exposes these toggles through the environment);
    // they are driven by the explicit backend options. Every toggle is set
    // to its target state on EVERY init (setenv or unsetenv), so repeated
    // in-process init calls (mode or model switches) cannot inherit a
    // previous call's environment — the target state is a pure function of
    // the current options.
    if (name.rfind("Vulkan", 0) == 0 && !vulkan_fast) {
        if (vulkan_fa_coopmat) {
            unsetenv("GGML_VK_DISABLE_COOPMAT");       // coopmat1 FA must enumerate
            setenv("GGML_VK_FA_COOPMAT_ONLY", "1", 1); // GEMMs stay scalar
        } else {
            setenv("GGML_VK_DISABLE_COOPMAT", "1", 1);
            unsetenv("GGML_VK_FA_COOPMAT_ONLY");
        }
        setenv("GGML_VK_DISABLE_COOPMAT2", "1", 1);    // NV coopmat2: TF32-class
        if (vulkan_integer_dot) {
            unsetenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT");
        } else {
            setenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "1", 1);
        }
        setenv("GGML_VK_DISABLE_F16", "1", 1);
    }
    ggml_backend_load_all();
    if (name == "cpu") {
        backend_ = ggml_backend_cpu_init();
        if (backend_) ggml_backend_cpu_set_n_threads(backend_, threads);
    } else {
        backend_ = ggml_backend_init_by_name(name.c_str(), nullptr);
    }
    if (!backend_) {
        error_ = "requested ggml backend is unavailable: " + name;
        return false;
    }
    return true;
}

void backend::release() {
    if (backend_) ggml_backend_free(backend_);
    backend_ = nullptr;
    error_.clear();
}

} // namespace lingbot
