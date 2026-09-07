// Standalone sky-segmentation runner for the native ggml skyseg runtime.
//
//   skyseg-cli MODEL.gguf IMAGE backend [out_prefix] [iters]
//
// IMAGE is a PNG/JPG (decoded with stb) or a .f32 raw RGB interleaved dump
// (width/height taken from --size WxH). The raw sky-probability map
// (320*320 floats) is written to OUT_PREFIX.f32 so scripts can diff it
// against the onnxruntime baseline bit by bit; a visual mask PNG and timing
// statistics are printed alongside.
#include "skyseg.h"

#include "backend.h"

// Implementations come from tools/stb_impl.cpp; here we only need the API.
#include "stb_image.h"
#include "stb_image_write.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s MODEL.gguf IMAGE backend [out_prefix] [iters] [--size WxH]\n",
                     argv[0]);
        return 2;
    }
    const std::string model = argv[1], image = argv[2], backend_name = argv[3];
    const std::string prefix = argc > 4 ? argv[4] : "skyseg_out";
    const int iters = argc > 5 ? std::atoi(argv[5]) : 10;

    int force_w = 0, force_h = 0;
    for (int i = 6; i < argc - 1; ++i)
        if (std::strcmp(argv[i], "--size") == 0)
            std::sscanf(argv[i + 1], "%dx%d", &force_w, &force_h);

    int w = 0, h = 0, channels = 0;
    std::vector<float> rgb;
    if (force_w > 0) {
        w = force_w; h = force_h;
        rgb.resize(static_cast<size_t>(w) * h * 3);
        FILE * f = std::fopen(image.c_str(), "rb");
        if (!f || std::fread(rgb.data(), 4, rgb.size(), f) != rgb.size()) {
            std::fprintf(stderr, "failed to read raw f32 RGB: %s\n", image.c_str());
            return 1;
        }
        std::fclose(f);
    } else {
        unsigned char * pixels = stbi_load(image.c_str(), &w, &h, &channels, 3);
        if (!pixels) {
            std::fprintf(stderr, "failed to decode image: %s\n", image.c_str());
            return 1;
        }
        rgb.resize(static_cast<size_t>(w) * h * 3);
        for (size_t i = 0; i < rgb.size(); ++i)
            rgb[i] = pixels[i] / 255.0f;
        stbi_image_free(pixels);
    }
    std::printf("image %s: %dx%d\n", image.c_str(), w, h);

    lingbot::backend be;
    if (!be.init(backend_name, 4)) {
        std::fprintf(stderr, "backend init failed: %s\n", be.error().c_str());
        return 1;
    }

    lingbot::skyseg seg;
    if (!seg.load(model, be.get())) {
        std::fprintf(stderr, "skyseg load failed: %s\n", seg.error().c_str());
        return 1;
    }

    std::vector<float> map;
    if (!seg.infer(rgb.data(), h, w, map)) {
        std::fprintf(stderr, "skyseg infer failed: %s\n", seg.error().c_str());
        return 1;
    }

    // timing
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (!seg.infer(rgb.data(), h, w, map)) {
            std::fprintf(stderr, "skyseg infer failed: %s\n", seg.error().c_str());
            return 1;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    std::printf("RESULT backend=%s model=%s avg_ms=%.3f iters=%d\n",
                backend_name.c_str(), model.c_str(), ms, iters);

    double mn = 1e30, mx = -1e30, sum = 0;
    int sky_pixels = 0;
    for (float v : map) {
        mn = std::min(mn, static_cast<double>(v));
        mx = std::max(mx, static_cast<double>(v));
        sum += v;
        if (v > 0.5f) ++sky_pixels;
    }
    std::printf("map min=%.4f max=%.4f mean=%.4f sky_frac(>0.5)=%.4f\n",
                mn, mx, sum / map.size(),
                static_cast<double>(sky_pixels) / map.size());

    const std::string f32_path = prefix + ".f32";
    FILE * f = std::fopen(f32_path.c_str(), "wb");
    if (f) { std::fwrite(map.data(), 4, map.size(), f); std::fclose(f); }
    std::printf("wrote %s (%d floats)\n", f32_path.c_str(), 320 * 320);

    // grayscale mask preview (u8, 320x320)
    std::vector<unsigned char> gray(map.size());
    for (size_t i = 0; i < map.size(); ++i)
        gray[i] = static_cast<unsigned char>(std::lround(
            std::min(1.f, std::max(0.f, map[i])) * 255.f));
    const std::string png_path = prefix + ".png";
    stbi_write_png(png_path.c_str(), 320, 320, 1, gray.data(), 320);
    std::printf("wrote %s\n", png_path.c_str());
    return 0;
}
