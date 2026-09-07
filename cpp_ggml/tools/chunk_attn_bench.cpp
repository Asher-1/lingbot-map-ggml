// Layer-level benchmark of the chunked attention used by model.cpp
// (persistent F16 KV cache, hand-written F32 path), plus the fused-softmax
// variant (ggml_soft_max_ext_inplace) that merges the per-chunk ggml_scale
// into the softmax kernel.
//
// Shapes mirror the 518x378 working point: hd=64, heads=16, S_q tokens per
// streaming frame, S_k cached+current keys. Every variant allocates its own
// gallocr and computes the SAME graph repeatedly (one graph per variant, as
// model.cpp does per frame), so no cross-size galloc re-reserve is involved.
//
//   unchunked   : mul_mat(k,q) -> scale -> softmax -> mul_mat(v, scores)
//   chunked     : the model.cpp loop over [hd, qc, heads] query chunks,
//                 each chunk ggml_cont-materialized, results ggml_concat-ed
//   chunked-fs  : chunked with ggml_soft_max_ext_inplace(scale fused)
//
// Build (CUDA):
//   g++ -O2 -std=c++17 -I third_party/ggml/include -I third_party/ggml/src \
//       tools/chunk_attn_bench.cpp -o /tmp/cab \
//       -L build-cuda/ggml/src -L build-cuda/ggml/src/ggml-cuda \
//       -lggml -lggml-base -lggml-cpu -lggml-cuda
// Run: /tmp/cab CUDA0 [S_k] [S_q] [chunk] [iters]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

double bench(ggml_backend_t backend, ggml_gallocr_t galloc, ggml_cgraph * g,
             int iters, std::vector<float> * out_sample = nullptr,
             ggml_tensor * out_tensor = nullptr, size_t sample_n = 0) {
    if (!ggml_gallocr_alloc_graph(galloc, g)) {
        std::fprintf(stderr, "graph alloc failed\n");
        std::exit(1);
    }
    // warmup + correctness sample on the first iteration
    ggml_backend_graph_compute(backend, g);
    if (out_sample && out_tensor && sample_n) {
        out_sample->resize(sample_n);
        ggml_backend_tensor_get(out_tensor, out_sample->data(), 0, sample_n * 4);
    }
    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, g);
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

} // namespace

int main(int argc, char ** argv) {
    const char * backend_name = argc > 1 ? argv[1] : "CUDA0";
    const int64_t S_k = argc > 2 ? atoll(argv[2]) : 20000;
    const int64_t S_q = argc > 3 ? atoll(argv[3]) : 1000;
    const int64_t q_chunk = argc > 4 ? atoll(argv[4]) : 64;
    const int iters = argc > 5 ? atoi(argv[5]) : 20;
    const bool skip_unchunked = argc > 6 && atoi(argv[6]) != 0;
    const bool skip_chunked = argc > 7 && atoi(argv[7]) != 0;
    const bool skip_fused = argc > 8 && atoi(argv[8]) != 0;
    const int64_t hd = 64, heads = 16;

    ggml_backend_t backend = ggml_backend_init_by_name(backend_name, nullptr);
    if (!backend) { std::fprintf(stderr, "backend %s not found\n", backend_name); return 1; }
    std::printf("backend=%s S_k=%lld S_q=%lld hd=%lld heads=%lld chunk=%lld iters=%d\n",
        ggml_backend_name(backend), (long long)S_k, (long long)S_q,
        (long long)hd, (long long)heads, (long long)q_chunk, iters);
    const ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    struct ggml_init_params ip = { 512ull * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    ggml_tensor * k_src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, heads, S_k, 1);
    ggml_tensor * v_src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, heads, S_k, 1);
    ggml_tensor * q_src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, heads, S_q, 1);
    ggml_backend_buffer_t inbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    {
        std::vector<float> kh(hd * heads * S_k), vh(hd * heads * S_k), qh(hd * heads * S_q);
        for (auto & x : kh) x = dist(rng);
        for (auto & x : vh) x = dist(rng);
        for (auto & x : qh) x = dist(rng);
        ggml_backend_tensor_set(k_src, kh.data(), 0, kh.size() * 4);
        ggml_backend_tensor_set(v_src, vh.data(), 0, vh.size() * 4);
        ggml_backend_tensor_set(q_src, qh.data(), 0, qh.size() * 4);
    }
    const float scale = 1.0f / std::sqrt((float)hd);

    // model.cpp layouts: k [hd, S_k, heads] via permute(0,2,1,3)+cont,
    // v [hd, S_k, heads] via permute(1,2,0,3)+cont, q [hd, S_q, heads]
    ggml_tensor * k = ggml_cont(ctx, ggml_permute(ctx, k_src, 0, 2, 1, 3));
    ggml_tensor * v = ggml_cont(ctx, ggml_permute(ctx, v_src, 1, 2, 0, 3));
    ggml_tensor * q = ggml_cont(ctx, ggml_permute(ctx, q_src, 0, 2, 1, 3));
    // k/v/q are graph nodes recomputed inside every variant graph

    // ---- variant 1: unchunked (fits only for small S_k) ----
    double un_ms = -1;
    std::vector<float> ref_sample;
    if (!skip_unchunked) {
        ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        scores = ggml_scale(ctx, scores, scale);
        scores = ggml_soft_max_inplace(ctx, scores);
        ggml_tensor * out = ggml_mul_mat(ctx, v, scores);
        ggml_mul_mat_set_prec(out, GGML_PREC_F32);
        out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3)); // [hd, heads, S_q]
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);
        ggml_gallocr_t galloc = ggml_gallocr_new(buft);
        un_ms = bench(backend, galloc, g, iters, &ref_sample, out, 1024);
        ggml_gallocr_free(galloc);
        std::printf("unchunked            : %8.2f ms\n", un_ms);
    }

    // helper: build the chunked graph (mode 0 = scale+softmax_inplace,
    // mode 1 = soft_max_ext_inplace with fused scale)
    auto build_chunked = [&](int mode, ggml_tensor ** out_t, ggml_tensor ** concat_t) {
        int64_t n_chunks = (S_q + q_chunk - 1) / q_chunk;
        ggml_tensor * out = nullptr;
        for (int64_t c = 0; c < n_chunks; ++c) {
            const int64_t q0 = c * q_chunk;
            const int64_t qc = std::min(q_chunk, S_q - q0);
            ggml_tensor * q_c = ggml_cont(ctx, ggml_view_4d(ctx, q, hd, qc, q->ne[2], q->ne[3],
                q->nb[1], q->nb[2], q->nb[3], q0 * q->nb[1]));
            ggml_tensor * scores = ggml_mul_mat(ctx, k, q_c);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            if (mode == 0) {
                scores = ggml_scale(ctx, scores, scale);
                scores = ggml_soft_max_inplace(ctx, scores);
            } else {
                scores = ggml_soft_max_ext_inplace(ctx, scores, nullptr, scale, 0.0f);
            }
            ggml_tensor * out_c = ggml_mul_mat(ctx, v, scores);
            ggml_mul_mat_set_prec(out_c, GGML_PREC_F32);
            out = out ? ggml_concat(ctx, out, out_c, 1) : out_c;
        }
        *concat_t = out;
        ggml_tensor * final_out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
        *out_t = final_out;
    };

    // ---- variant 2: chunked, scale + softmax_inplace (model.cpp current) ----
    double ch_ms = -1;
    std::vector<float> ch_sample;
    if (!skip_chunked) {
        ggml_tensor * out_t = nullptr, * concat_t = nullptr;
        build_chunked(0, &out_t, &concat_t);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out_t);
        ggml_gallocr_t galloc = ggml_gallocr_new(buft);
        ch_ms = bench(backend, galloc, g, iters, &ch_sample, out_t, 1024);
        ggml_gallocr_free(galloc);
        std::printf("chunked              : %8.2f ms\n", ch_ms);
    }

    // ---- variant 3: chunked with fused softmax (soft_max_ext_inplace) ----
    double fs_ms = -1;
    std::vector<float> fs_sample;
    if (!skip_fused) {
        ggml_tensor * out_t = nullptr, * concat_t = nullptr;
        build_chunked(1, &out_t, &concat_t);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out_t);
        ggml_gallocr_t galloc = ggml_gallocr_new(buft);
        fs_ms = bench(backend, galloc, g, iters, &fs_sample, out_t, 1024);
        ggml_gallocr_free(galloc);
        std::printf("chunked-fused-softmax: %8.2f ms\n", fs_ms);
    }

    if (!ref_sample.empty() && !ch_sample.empty()) {
        double e = 0;
        for (size_t i = 0; i < ref_sample.size(); ++i)
            e = std::max(e, (double)std::fabs(ch_sample[i] - ref_sample[i]));
        std::printf("max_err chunked vs unchunked: %.3e\n", e);
    }
    if (!ref_sample.empty() && !fs_sample.empty()) {
        double e = 0;
        for (size_t i = 0; i < ref_sample.size(); ++i)
            e = std::max(e, (double)std::fabs(fs_sample[i] - ref_sample[i]));
        std::printf("max_err fused   vs unchunked: %.3e\n", e);
    }
    if (!ch_sample.empty() && !fs_sample.empty()) {
        double e = 0;
        for (size_t i = 0; i < ch_sample.size(); ++i)
            e = std::max(e, (double)std::fabs(ch_sample[i] - fs_sample[i]));
        std::printf("max_err fused   vs chunked: %.3e\n", e);
    }
    if (un_ms > 0 && ch_ms > 0 && fs_ms > 0)
        std::printf("summary: chunked/unchunked = %.3f  fused/chunked = %.3f\n",
            ch_ms / un_ms, fs_ms / ch_ms);
    else if (ch_ms > 0 && fs_ms > 0)
        std::printf("summary: fused/chunked = %.3f\n", fs_ms / ch_ms);
    else if (ch_ms > 0 && un_ms > 0)
        std::printf("summary: chunked/unchunked = %.3f\n", ch_ms / un_ms);

    ggml_backend_buffer_free(inbuf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
