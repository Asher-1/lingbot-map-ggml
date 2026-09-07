#include "lingbot_map.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s MODEL.gguf [backend] [iters] [H] [W]\n", argv[0]); return 2; }
    const std::string backend = argc > 2 ? argv[2] : "cpu";
    const int iters = argc > 3 ? std::atoi(argv[3]) : 5;
    const int h = argc > 4 ? std::atoi(argv[4]) : 518;
    const int w = argc > 5 ? std::atoi(argv[5]) : 518;
    // This is a stateless one-frame benchmark. Avoid staging the global and
    // camera KV tensors through host memory when every iteration resets them.
    lingbot::model_options options;
    options.disable_kv_cache = true;
    std::vector<float> image((size_t)3 * h * w, 0.5f);
    lingbot::model model;
    if (!model.load(argv[1], backend, 4, options)) { std::fprintf(stderr, "load failed: %s\n", model.error().c_str()); return 1; }
    lingbot::output out;
    for (int i = 0; i < 2; ++i) { model.reset_cache(); if (!model.infer(image.data(), 1, 1, 3, h, w, &out)) return 1; }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        model.reset_cache();
        if (!model.infer(image.data(), 1, 1, 3, h, w, &out)) return 1;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
    std::printf("RESULT engine=ggml backend=%s precision=%s median_ms=%.4f frames_per_s=%.4f\n", backend.c_str(), model.info().weight_type.c_str(), ms, 1000.0 / ms);
    return 0;
}
