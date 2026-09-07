// FA (ggml_flash_attn_ext, PREC_F32) vs hand-written F32 attention at
// streaming shapes, on identical inputs — isolates kernel precision from
// integration bugs in the model's flash branch.
//
//   g++ -O2 -std=c++17 -I <ggml>/include -I <ggml>/src fattn_stream_test.cpp \
//       -o fattn -L <build>/src -L <build>/src/ggml-cuda \
//       -lggml -lggml-base -lggml-cpu -lggml-cuda
//   ./fattn CUDA0 [S_k] [S_q]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <fstream>
#include <vector>

static std::vector<float> g_ref;

static void run_graph(ggml_backend_t be, ggml_context * ctx, ggml_tensor * out, int tag) {
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, out);
    if (!ggml_gallocr_alloc_graph(ga, g)) {
        std::fprintf(stderr, "graph %d: alloc failed\n", tag);
        std::exit(1);
    }
    ggml_backend_graph_compute(be, g);
    g_ref.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, g_ref.data(), 0, g_ref.size() * sizeof(float));
    ggml_gallocr_free(ga);
}

int main(int argc, char ** argv) {
    const char * backend_name = argc > 1 ? argv[1] : "CUDA0";
    const int64_t S_k = argc > 2 ? atoll(argv[2]) : 8613;
    const int64_t S_q = argc > 3 ? atoll(argv[3]) : 783;
    const int64_t heads = 16, hd = 64;
    const float scale = 1.0f / std::sqrt((float)hd);

    ggml_backend_t be = ggml_backend_init_by_name(backend_name, nullptr);
    if (!be) { std::fprintf(stderr, "backend %s unavailable\n", backend_name); return 1; }
    std::printf("backend: %s  S_k=%lld S_q=%lld\n", ggml_backend_name(be),
                (long long)S_k, (long long)S_q);

    ggml_init_params ip = { 512ull * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, heads, S_q);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, heads, S_k);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, heads, S_k);
    ggml_backend_buffer_t inbuf = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_get_default_buffer_type(be));
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> hq(ggml_nelements(q)), hk(ggml_nelements(k)), hv(ggml_nelements(v));
    for (auto & x : hq) x = dist(rng);
    for (auto & x : hk) x = dist(rng);
    for (auto & x : hv) x = dist(rng);
    ggml_backend_tensor_set(q, hq.data(), 0, hq.size() * sizeof(float));
    ggml_backend_tensor_set(k, hk.data(), 0, hk.size() * sizeof(float));
    ggml_backend_tensor_set(v, hv.data(), 0, hv.size() * sizeof(float));
    {
        std::ofstream f("/tmp/fst_q.bin", std::ios::binary); f.write((const char*)hq.data(), hq.size()*4);
        std::ofstream g("/tmp/fst_k.bin", std::ios::binary); g.write((const char*)hk.data(), hk.size()*4);
        std::ofstream h("/tmp/fst_v.bin", std::ios::binary); h.write((const char*)hv.data(), hv.size()*4);
    }

    // ---- graph 1: hand-written F32 reference (with intermediate dumps) ----
    ggml_tensor * s_pre_t = nullptr, * s_post_t = nullptr;
    {
        ggml_tensor * k_t = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // {hd, S_k, H}
        ggml_tensor * q_t = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // {hd, S_q, H}
        ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // {S_k, hd, H} (mul_mat operand as in model.cpp)
        ggml_tensor * s = ggml_mul_mat(ctx, k_t, q_t);                        // {S_k, S_q, H}
        ggml_mul_mat_set_prec(s, GGML_PREC_F32);
        s = ggml_scale(ctx, s, scale);
        s_pre_t = ggml_cont(ctx, s);
        s = ggml_soft_max_inplace(ctx, s);
        s_post_t = ggml_cont(ctx, s);
        ggml_tensor * out_t = ggml_mul_mat(ctx, v_t, s);                      // {hd, S_q, H}
        ggml_mul_mat_set_prec(out_t, GGML_PREC_F32);
        ggml_tensor * out = ggml_cont(ctx, ggml_permute(ctx, out_t, 0, 2, 1, 3));
        // one gallocr, all roots expanded: the dumps stay alive to graph end
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);
        ggml_build_forward_expand(g, s_pre_t);
        ggml_build_forward_expand(g, s_post_t);
        if (!ggml_gallocr_alloc_graph(ga, g)) { std::fprintf(stderr, "alloc1 failed\n"); return 1; }
        ggml_backend_graph_compute(be, g);
        g_ref.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, g_ref.data(), 0, g_ref.size() * sizeof(float));
        std::vector<float> pre(ggml_nelements(s_pre_t)), post(ggml_nelements(s_post_t));
        ggml_backend_tensor_get(s_pre_t, pre.data(), 0, pre.size()*4);
        ggml_backend_tensor_get(s_post_t, post.data(), 0, post.size()*4);
        std::ofstream f1("/tmp/fst_spre.bin", std::ios::binary); f1.write((const char*)pre.data(), pre.size()*4);
        std::ofstream f2("/tmp/fst_spost.bin", std::ios::binary); f2.write((const char*)post.data(), post.size()*4);
        ggml_gallocr_free(ga);
    }
    std::vector<float> ref = g_ref;

    // ---- graph 2: FA kernel, PREC_F32, F16 K/V (model's flash configuration) ----
    {
        ggml_tensor * qf = ggml_permute(ctx, q, 0, 2, 1, 3);
        ggml_tensor * kf = ggml_permute(ctx, ggml_cast(ctx, k, GGML_TYPE_F16), 0, 2, 1, 3);
        ggml_tensor * vf = ggml_permute(ctx, ggml_cast(ctx, v, GGML_TYPE_F16), 0, 2, 1, 3);
        ggml_tensor * fa = ggml_flash_attn_ext(ctx, qf, kf, vf, nullptr, scale, 0.f, 0.f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        ggml_tensor * out = ggml_cont(ctx, fa);
        run_graph(be, ctx, out, 2);
    }
    std::vector<float> fa = g_ref;

    // ---- graph 3: FA again — determinism probe ----
    {
        ggml_tensor * qf = ggml_permute(ctx, q, 0, 2, 1, 3);
        ggml_tensor * kf = ggml_permute(ctx, ggml_cast(ctx, k, GGML_TYPE_F16), 0, 2, 1, 3);
        ggml_tensor * vf = ggml_permute(ctx, ggml_cast(ctx, v, GGML_TYPE_F16), 0, 2, 1, 3);
        ggml_tensor * fa = ggml_flash_attn_ext(ctx, qf, kf, vf, nullptr, scale, 0.f, 0.f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        ggml_tensor * out = ggml_cont(ctx, fa);
        run_graph(be, ctx, out, 3);
    }

    auto stats = [](const char * name, std::vector<float> & x) {
        double mn = 1e30, mx = -1e30, sm = 0;
        for (float v : x) { mn = std::min(mn, (double)v); mx = std::max(mx, (double)v); sm += v; }
        std::printf("%s: n=%zu min %.4g max %.4g mean %.4g\n", name, x.size(), mn, mx, sm / x.size());
    };
    stats("manual", ref);
    stats("fa    ", fa);
    {
        std::ofstream f("/tmp/fst_manual.bin", std::ios::binary); f.write((const char*)ref.data(), ref.size()*4);
        std::ofstream g("/tmp/fst_fa.bin", std::ios::binary); g.write((const char*)fa.data(), fa.size()*4);
    }
    double max1 = 0, sum1 = 0, max2 = 0;
    for (size_t i = 0; i < fa.size(); ++i) {
        const double d = std::fabs(fa[i] - ref[i]);
        max1 = std::max(max1, d);
        sum1 += d * d;
        max2 = std::max(max2, (double)std::fabs(fa[i] - g_ref[i]));
    }
    const double rmse = std::sqrt(sum1 / (double)fa.size());
    std::printf("FA vs manual : max %.4g  rmse %.4g   (FA determinism: %s)\n",
                max1, rmse, max2 == 0.0 ? "bit-stable" : "NONDETERMINISTIC");
    ggml_backend_buffer_free(inbuf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;
}
