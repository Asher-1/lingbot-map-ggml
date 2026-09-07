#include "lingbot_map.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>

#include "skyseg.h"

#include "stb_image.h"
#include "stb_image_write.h"
#include "backend.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <vector>

int main(int argc, char ** argv) {
    // Named flags are extracted first so the positional contract below stays
    // unchanged: MODEL.gguf IMAGE.bin [backend] [H] [W] [out] [frames] [dir].
    // All former LINGBOT_* environment switches are explicit options now.
    lingbot::model_options options;
    std::vector<char *> positional;
    positional.push_back(argv[0]); // keep argv[0] at the front
    std::string f16_err;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next_value = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", flag); std::exit(2); }
            return argv[++i];
        };
        if (a == "--kv-f16") {
            const std::string v = next_value("--kv-f16");
            if (v == "none") options.kv_f16 = lingbot::kv_f16_mode::none;
            else if (v == "strict") options.kv_f16 = lingbot::kv_f16_mode::strict;
            else if (v == "flash") options.kv_f16 = lingbot::kv_f16_mode::flash;
            else { std::fprintf(stderr, "invalid --kv-f16 '%s' (none|strict|flash)\n", v.c_str()); return 2; }
        } else if (a == "--kv-scale") options.kv_cache_scale = std::atoi(next_value("--kv-scale"));
        else if (a == "--kv-window") options.kv_cache_window = std::atoi(next_value("--kv-window"));
        else if (a == "--kv-total") options.kv_total_frames = std::atoi(next_value("--kv-total"));
        else if (a == "--scale-frames") options.num_scale_frames = std::atoi(next_value("--scale-frames"));
        else if (a == "--no-kv-resident") options.kv_resident = false;
        else if (a == "--disable-kv-cache") options.disable_kv_cache = true;
        else if (a == "--force-f32-weights") options.force_f32_weights = true;
        else if (a == "--no-dpt-pos") options.use_dpt_pos = false;
        else if (a == "--vulkan-fast") options.vulkan_fast = true;
        else if (a == "--vulkan-integer-dot") options.vulkan_integer_dot = true;
        else if (a == "--repeat") options.repeat_run = true;
        else if (a == "--debug-kv") options.debug_kv = true;
        else if (a == "--debug-infer") options.debug_infer = true;
        else if (a == "--profile") options.profile = true;
        else if (a == "--flat-off") options.flat_off = true;
        else if (a == "--dump-internal") options.dump_internal = true;
        else if (a == "--dump-stages") options.dump_stages = true, options.dump_stages_dir = next_value("--dump-stages");
        else if (a == "--dump-camera-stages") options.dump_camera_stages = true, options.dump_camera_stages_dir = next_value("--dump-camera-stages");
        else if (a == "--dump-features") options.dump_features_dir = next_value("--dump-features");
        else if (a == "--dump-block") options.dump_block = next_value("--dump-block");
        else if (a == "--dump-at") options.dump_at = std::atoi(next_value("--dump-at"));
        else if (a.rfind("--", 0) == 0) { std::fprintf(stderr, "unknown flag: %s\n", a.c_str()); return 2; }
        else positional.push_back(argv[i]);
    }
    if (positional.size() < 2) {
        std::fprintf(stderr,
            "usage: %s MODEL.gguf IMAGE_FLOAT32.bin [backend] [H] [W] [output.bin] [frames] [stream-out-dir] [skyseg.gguf] [mask-cache-dir] [viz-dir]\n"
            "options:\n"
            "  --kv-f16 none|strict|flash   persistent-cache attention mode (default none)\n"
            "  --kv-scale N / --kv-window N / --kv-total N / --scale-frames N\n"
            "  --no-kv-resident / --disable-kv-cache / --force-f32-weights / --no-dpt-pos\n"
            "  --vulkan-fast / --vulkan-integer-dot / --repeat\n"
            "  --debug-kv / --debug-infer / --profile / --flat-off\n"
            "  --dump-internal / --dump-stages DIR / --dump-camera-stages DIR / --dump-features DIR / --dump-block N / --dump-at N\n",
            positional.empty() ? argv[0] : positional[0]);
        return 2;
    }
    argv = positional.data();
    argc = static_cast<int>(positional.size());
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf IMAGE_FLOAT32.bin [backend] [H] [W] [output.bin] [frames] [stream-out-dir]\n", argv[0]);
        return 2;
    }
    const int h = argc > 4 ? std::atoi(argv[4]) : 518;
    const int w = argc > 5 ? std::atoi(argv[5]) : 518;
    const std::string backend = argc > 3 ? argv[3] : "cpu";
    const int requested_frames = argc > 7 ? std::atoi(argv[7]) : 0;
    std::ifstream in(argv[2], std::ios::binary | std::ios::ate);
    if (!in) { std::fprintf(stderr, "cannot open input: %s\n", argv[2]); return 2; }
    const size_t bytes = static_cast<size_t>(in.tellg()); in.seekg(0);
    if (bytes % sizeof(float) != 0) { std::fprintf(stderr, "input is not float32\n"); return 2; }
    std::vector<float> image(bytes / sizeof(float)); in.read(reinterpret_cast<char *>(image.data()), bytes);
    const size_t frame_values = static_cast<size_t>(3) * h * w;
    if (frame_values == 0 || image.size() % frame_values != 0) { std::fprintf(stderr, "input size does not match [B,F,3,%d,%d]\n", h, w); return 2; }
    const int frames = requested_frames > 0 ? requested_frames : static_cast<int>(image.size() / frame_values);
    // Device-resident cache capacity hint: the total streamed-frame count.
    if (options.kv_total_frames <= 0) options.kv_total_frames = frames;
    if (frames <= 0 || image.size() != frame_values * static_cast<size_t>(frames)) { std::fprintf(stderr, "input contains %zu values, expected exactly %d frame(s)\n", image.size(), frames); return 2; }
    lingbot::model model;
    if (!model.load(argv[1], backend, 4, options)) { std::fprintf(stderr, "load failed: %s\n", model.error().c_str()); return 1; }

    // Optional sky segmentation: argv[9] is a skyseg GGUF (see
    // scripts/convert_skyseg.py), argv[10] an optional mask-cache dir
    // (PNG per frame, {0,255}, reused across runs like the official
    // <folder>_sky_masks cache), argv[11] an optional visualization dir
    // (official-style original | mask | overlay panel). Every frame's RGB
    // image runs through the
    // native sky-segmentation network BEFORE reconstruction; sky pixels get
    // their depth confidence zeroed, exactly matching the official viewer's
    // --mask_sky semantics (non-sky confidence > 0.1 is kept) but with no
    // onnxruntime dependency.
    std::vector<unsigned char> sky_keep;   // per frame: 1 = keep pixel
    lingbot::skyseg skyseg;
    const char * skyseg_path = argc > 9 ? argv[9] : nullptr;
    if (skyseg_path && *skyseg_path) {
        lingbot::backend seg_be;
        if (!seg_be.init(backend, 4)) { std::fprintf(stderr, "skyseg backend failed: %s\n", seg_be.error().c_str()); return 1; }
        if (!skyseg.load(skyseg_path, seg_be.get())) { std::fprintf(stderr, "skyseg load failed: %s\n", skyseg.error().c_str()); return 1; }
        sky_keep.resize(static_cast<size_t>(frames) * h * w, 1);
        std::vector<float> hwc(static_cast<size_t>(3) * h * w);
        std::vector<float> map;
        const char * mask_dir = argc > 10 ? argv[10] : nullptr;
        const char * vis_dir = argc > 11 ? argv[11] : nullptr;
        if (vis_dir && *vis_dir) { std::filesystem::create_directories(vis_dir); }
        if (mask_dir && *mask_dir) { std::filesystem::create_directories(mask_dir); }
        for (int f = 0; f < frames; ++f) {
            const float * chw = image.data() + static_cast<size_t>(f) * frame_values;
            for (int y = 0; y < h; ++y)
                for (int xx = 0; xx < w; ++xx)
                    for (int c = 0; c < 3; ++c)
                        hwc[(static_cast<size_t>(y) * w + xx) * 3 + c] =
                            chw[(static_cast<size_t>(c) * h + y) * w + xx];
            // mask PNG cache (official <folder>_sky_masks semantics): reuse
            // a same-size cached mask instead of re-running the network
            char mask_png[1024] = "", vis_png[1024] = "";
            if (mask_dir && *mask_dir)
                std::snprintf(mask_png, sizeof(mask_png), "%s/frame_%04d.png", mask_dir, f);
            if (vis_dir && *vis_dir)
                std::snprintf(vis_png, sizeof(vis_png), "%s/frame_%04d.png", vis_dir, f);
            std::vector<unsigned char> u8map(static_cast<size_t>(h) * w, 0);
            bool have_mask = false;
            {
                int iw = 0, ih = 0, nc = 0;
                unsigned char * png = mask_png[0] ? stbi_load(mask_png, &iw, &ih, &nc, 1) : nullptr;
                if (png && iw == w && ih == h) {
                    // cached PNG is {0,255} non-sky conf: normalize back to
                    // the "resized u8 == 0" lattice the keep test expects
                    for (size_t i = 0; i < u8map.size(); ++i) u8map[i] = png[i] ? 0 : 255;
                    have_mask = true;
                }
                if (png) stbi_image_free(png);
            }
            if (!have_mask) {
                if (!skyseg.infer(hwc.data(), h, w, map)) { std::fprintf(stderr, "skyseg infer failed: %s\n", skyseg.error().c_str()); return 1; }
                // Official postprocess (sky_segmentation.py run_skyseg +
                // segment_sky_from_array): min-max normalize, truncate to
                // u8, resize, and keep only pixels whose resized u8 value
                // is 0 - the uint8 clip in the official code collapses
                // every other value to non-sky=0.
                float mn = map[0], mx = map[0];
                for (float v : map) { mn = std::min(mn, v); mx = std::max(mx, v); }
                const float denom = std::max(mx - mn, 1e-8f);
                std::vector<unsigned char> small(320 * 320);
                for (size_t i = 0; i < small.size(); ++i)
                    small[i] = (unsigned char)((map[i] - mn) * 255.0f / denom);  // truncate
                // resize the u8 lattice with the bit-exact cv2 u8 INTER_LINEAR
                // semantics of the official segment_sky_from_array
                lingbot::skyseg_resize_mask_u8(small.data(), 320, 320,
                                               u8map.data(), w, h);
                // cache PNG carries the official {0,255} non-sky conf
                // (255 = keep), interoperable with <folder>_sky_masks
                if (mask_png[0]) {
                    std::vector<unsigned char> keep_png(u8map.size());
                    for (size_t i = 0; i < u8map.size(); ++i)
                        keep_png[i] = u8map[i] == 0 ? 255 : 0;
                    stbi_write_png(mask_png, w, h, 1, keep_png.data(), w);
                }
            }
            unsigned char * keep = sky_keep.data() + static_cast<size_t>(f) * h * w;
            for (size_t i = 0; i < u8map.size(); ++i)
                keep[i] = u8map[i] == 0 ? 1 : 0;
            if (vis_png[0]) {
                // original | mask | overlay, as in the official
                // _save_sky_mask_visualization (sky tinted [255,64,64] @ 0.65)
                const int pw = 3 * w;
                std::vector<unsigned char> panel(static_cast<size_t>(pw) * h * 3, 255);
                for (int y = 0; y < h; ++y)
                    for (int xx = 0; xx < w; ++xx) {
                        const size_t i = static_cast<size_t>(y) * w + xx;
                        const unsigned char r = (unsigned char)std::lround(hwc[i * 3 + 0] * 255.f);
                        const unsigned char g = (unsigned char)std::lround(hwc[i * 3 + 1] * 255.f);
                        const unsigned char b = (unsigned char)std::lround(hwc[i * 3 + 2] * 255.f);
                        unsigned char * o = &panel[(static_cast<size_t>(y) * pw + xx) * 3];
                        o[0] = r; o[1] = g; o[2] = b;                       // original
                        unsigned char * m = &panel[(static_cast<size_t>(y) * pw + w + xx) * 3];
                        const unsigned char mv = keep[i] ? 255 : 0;         // mask (non-sky conf)
                        m[0] = m[1] = m[2] = mv;
                        unsigned char * v = &panel[(static_cast<size_t>(y) * pw + 2 * w + xx) * 3];
                        if (keep[i]) { v[0] = r; v[1] = g; v[2] = b; }
                        else {
                            v[0] = (unsigned char)std::lround(r * 0.35f + 255.f * 0.65f);
                            v[1] = (unsigned char)std::lround(g * 0.35f + 64.f * 0.65f);
                            v[2] = (unsigned char)std::lround(b * 0.35f + 64.f * 0.65f);
                        }
                    }
                stbi_write_png(vis_png, pw, h, 3, panel.data(), pw * 3);
            }
        }
        std::printf("SKYSEG model=%s frames=%d\n", skyseg_path, frames);
    }
    auto apply_sky = [&](lingbot::output & o) {
        if (sky_keep.empty()) return;
        for (int f = 0; f < frames; ++f) {
            const unsigned char * keep = sky_keep.data() + static_cast<size_t>(f) * h * w;
            float * conf = o.depth_conf.data() + static_cast<size_t>(f) * h * w;
            for (size_t i = 0; i < static_cast<size_t>(h) * w; ++i)
                if (!keep[i]) conf[i] = 0.0f;
        }
    };
    // Optional per-frame streaming output: when a directory is given, every
    // completed frame is written to DIR/frame_XXXX.bin (LBF3: pose 9 + depth
    // + depth_conf + c2w 16 + intrinsics 4) and a "FRAME i/n" line is flushed
    // to stdout, so a caller can render frames as they are produced.
    const char * stream_dir = argc > 8 ? argv[8] : nullptr;
    if (stream_dir && *stream_dir) {
        model.set_frame_callback([&](int idx, const lingbot::output & f) {
            char path[1024];
            std::snprintf(path, sizeof(path), "%s/frame_%04d.bin", stream_dir, idx);
            std::ofstream file(path, std::ios::binary);
            if (!file) { std::fprintf(stderr, "cannot open stream frame: %s\n", path); return false; }
            const uint32_t magic = 0x4c424633; // LBF3
            const uint32_t idx32 = static_cast<uint32_t>(idx);
            const uint32_t fh = static_cast<uint32_t>(f.height), fw = static_cast<uint32_t>(f.width);
            file.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
            file.write(reinterpret_cast<const char *>(&idx32), sizeof(idx32));
            file.write(reinterpret_cast<const char *>(&fh), sizeof(fh));
            file.write(reinterpret_cast<const char *>(&fw), sizeof(fw));
            file.write(reinterpret_cast<const char *>(f.pose_enc.data()), f.pose_enc.size() * sizeof(float));
            file.write(reinterpret_cast<const char *>(f.depth.data()), f.depth.size() * sizeof(float));
            std::vector<float> masked_conf;
            const float * conf_ptr = f.depth_conf.data();
            if (!sky_keep.empty()) {
                // mask only THIS frame: idx is the global frame index
                masked_conf = f.depth_conf;
                const unsigned char * keep = sky_keep.data() + static_cast<size_t>(idx) * h * w;
                for (size_t i = 0; i < masked_conf.size(); ++i)
                    if (!keep[i]) masked_conf[i] = 0.0f;
                conf_ptr = masked_conf.data();
            }
            file.write(reinterpret_cast<const char *>(conf_ptr), f.depth_conf.size() * sizeof(float));
            file.write(reinterpret_cast<const char *>(f.c2w.data()), f.c2w.size() * sizeof(float));
            file.write(reinterpret_cast<const char *>(f.intrinsics.data()), f.intrinsics.size() * sizeof(float));
            std::printf("FRAME %d/%d\n", idx + 1, frames);
            std::fflush(stdout);
            return true;
        });
    }
    lingbot::output out;
    if (!model.infer(image.data(), 1, frames, 3, h, w, &out)) { std::fprintf(stderr, "infer failed: %s\n", model.error().c_str()); return 1; }
    if (options.repeat_run) {
        lingbot::output streamed;
        if (!model.infer(image.data(), 1, frames, 3, h, w, &streamed)) { std::fprintf(stderr, "streamed infer failed: %s\n", model.error().c_str()); return 1; }
        std::printf("STREAM depth_mean=%.8g pose_delta=%.8g cache=enabled\n", streamed.depth.empty() ? 0.0 : streamed.depth[0],
            streamed.pose_enc.empty() || out.pose_enc.empty() ? 0.0 : streamed.pose_enc[0] - out.pose_enc[0]);
    }
    apply_sky(out);
    std::printf("RESULT backend=%s frames=%d image=%dx%d depth_mean=%.8g pose_values=%zu\n", backend.c_str(), frames, h, w, out.depth.empty() ? 0.0 : out.depth[0], out.pose_enc.size());
    if (argc > 6) {
        std::ofstream dump(argv[6], std::ios::binary);
        if (!dump) { std::fprintf(stderr, "cannot open output: %s\n", argv[6]); return 2; }
        const uint32_t magic = 0x4c424f31; // LBO1
        const uint32_t n_pose = static_cast<uint32_t>(out.pose_enc.size());
        const uint32_t n_depth = static_cast<uint32_t>(out.depth.size());
        dump.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
        dump.write(reinterpret_cast<const char *>(&n_pose), sizeof(n_pose));
        dump.write(reinterpret_cast<const char *>(&n_depth), sizeof(n_depth));
        dump.write(reinterpret_cast<const char *>(out.pose_enc.data()), out.pose_enc.size() * sizeof(float));
        dump.write(reinterpret_cast<const char *>(out.depth.data()), out.depth.size() * sizeof(float));
        std::ofstream post(std::string(argv[6]) + ".post", std::ios::binary);
        if (!post) { std::fprintf(stderr, "cannot open postprocess output: %s.post\n", argv[6]); return 2; }
        const uint32_t post_magic = 0x4c425032; // LBP2: C2W, intrinsics, depth confidence
        const uint32_t n_c2w = static_cast<uint32_t>(out.c2w.size());
        const uint32_t n_intrinsics = static_cast<uint32_t>(out.intrinsics.size());
        const uint32_t n_confidence = static_cast<uint32_t>(out.depth_conf.size());
        post.write(reinterpret_cast<const char *>(&post_magic), sizeof(post_magic));
        post.write(reinterpret_cast<const char *>(&n_c2w), sizeof(n_c2w));
        post.write(reinterpret_cast<const char *>(&n_intrinsics), sizeof(n_intrinsics));
        post.write(reinterpret_cast<const char *>(&n_confidence), sizeof(n_confidence));
        post.write(reinterpret_cast<const char *>(out.c2w.data()), out.c2w.size() * sizeof(float));
        post.write(reinterpret_cast<const char *>(out.intrinsics.data()), out.intrinsics.size() * sizeof(float));
        post.write(reinterpret_cast<const char *>(out.depth_conf.data()), out.depth_conf.size() * sizeof(float));
    }
    return 0;
}
