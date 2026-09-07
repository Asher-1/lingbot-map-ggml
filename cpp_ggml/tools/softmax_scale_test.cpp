// Same-graph comparison of the two softmax forms used by model.cpp's
// chunked attention:
//   explicit: ggml_scale(scores, s) -> ggml_soft_max_inplace
//   fused:    ggml_soft_max_ext_inplace(scores, nullptr, s, 0)
// Both consume identical mul_mat outputs inside one graph, so any
// difference is a numerics/semantics difference of the ops themselves,
// independent of graph allocation.
//
// Run: /tmp/sst CPU | /tmp/sst CUDA0 | /tmp/sst Vulkan0 [S_k S_q heads]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    const char * backend_name = argc > 1 ? argv[1] : "CPU";
    const int64_t S_k = argc > 2 ? atoll(argv[2]) : 128;
    const int64_t S_q = argc > 3 ? atoll(argv[3]) : 8;
    const int64_t heads = argc > 4 ? atoll(argv[4]) : 2;
    const int64_t hd = 64;
    const float scale = 1.0f / std::sqrt((float)hd);

    ggml_backend_t backend = ggml_backend_init_by_name(backend_name, nullptr);
    if (!backend) { std::fprintf(stderr, "backend not found\n"); return 1; }
    std::printf("backend=%s S_k=%lld S_q=%lld heads=%lld\n",
        ggml_backend_name(backend), (long long)S_k, (long long)S_q, (long long)heads);
    const ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    struct ggml_init_params ip = { 64ull*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    ggml_tensor * k_src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, heads, S_k);
    ggml_tensor * q_src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, heads, S_q);
    ggml_backend_buffer_t inbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    {
        std::vector<float> kh(hd*heads*S_k), qh(hd*heads*S_q);
        for (auto & x : kh) x = dist(rng);
        for (auto & x : qh) x = dist(rng);
        ggml_backend_tensor_set(k_src, kh.data(), 0, kh.size()*4);
        ggml_backend_tensor_set(q_src, qh.data(), 0, qh.size()*4);
    }
    // layout [hd, S, heads] for both, matching model.cpp
    ggml_tensor * k = ggml_cont(ctx, ggml_permute(ctx, k_src, 0, 2, 1, 3));
    ggml_tensor * q1 = ggml_cont(ctx, ggml_permute(ctx, q_src, 0, 2, 1, 3));
    ggml_tensor * q2 = ggml_cont(ctx, ggml_permute(ctx, q_src, 0, 2, 1, 3));

    // explicit path
    ggml_tensor * s1 = ggml_mul_mat(ctx, k, q1);
    ggml_mul_mat_set_prec(s1, GGML_PREC_F32);
    s1 = ggml_scale(ctx, s1, scale);
    s1 = ggml_soft_max_inplace(ctx, s1);
    // fused path
    ggml_tensor * s2 = ggml_mul_mat(ctx, k, q2);
    ggml_mul_mat_set_prec(s2, GGML_PREC_F32);
    s2 = ggml_soft_max_ext_inplace(ctx, s2, nullptr, scale, 0.0f);

    // full pipeline in the SAME graph: unchunked attention vs the
    // model.cpp chunked form with fused softmax (q_c cont + per-chunk
    // scale/softmax/mul_mat + concat + final permute/cont)
    ggml_tensor * out_u = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_permute(ctx, k_src, 1, 2, 0, 3)), s1);
    ggml_mul_mat_set_prec(out_u, GGML_PREC_F32);
    out_u = ggml_cont(ctx, ggml_permute(ctx, out_u, 0, 2, 1, 3));
    const int64_t q_chunk = 400;
    ggml_tensor * out_c_all = nullptr;
    for (int64_t q0 = 0; q0 < S_q; q0 += q_chunk) {
        const int64_t qc = q_chunk < (S_q - q0) ? q_chunk : (S_q - q0);
        ggml_tensor * q_c = ggml_cont(ctx, ggml_view_3d(ctx, q2, hd, qc, heads,
            q2->nb[1], q2->nb[2], q0 * q2->nb[1]));
        ggml_tensor * sc = ggml_mul_mat(ctx, k, q_c);
        ggml_mul_mat_set_prec(sc, GGML_PREC_F32);
        sc = ggml_soft_max_ext_inplace(ctx, sc, nullptr, scale, 0.0f);
        ggml_tensor * oc = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_permute(ctx, k_src, 1, 2, 0, 3)), sc);
        ggml_mul_mat_set_prec(oc, GGML_PREC_F32);
        out_c_all = out_c_all ? ggml_concat(ctx, out_c_all, oc, 1) : oc;
    }
    ggml_tensor * out_c_fin = ggml_cont(ctx, ggml_permute(ctx, out_c_all, 0, 2, 1, 3));

    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, s1);
    ggml_build_forward_expand(g, s2);
    ggml_build_forward_expand(g, out_u);
    ggml_build_forward_expand(g, out_c_fin);
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(galloc, g)) { std::fprintf(stderr, "alloc failed\n"); return 1; }
    ggml_backend_graph_compute(backend, g);

    const int64_t n = S_k * S_q * heads;
    std::vector<float> a(n), b(n), u(n), c(n);
    ggml_backend_tensor_get(s1, a.data(), 0, n*4);
    ggml_backend_tensor_get(s2, b.data(), 0, n*4);
    ggml_backend_tensor_get(out_u, u.data(), 0, n*4);
    ggml_backend_tensor_get(out_c_fin, c.data(), 0, n*4);
    double max_abs = 0, sum2 = 0; int64_t worst = 0;
    for (int64_t i = 0; i < n; ++i) {
        double d = std::fabs((double)a[i] - b[i]);
        if (d > max_abs) { max_abs = d; worst = i; }
        sum2 += d*d;
    }
    std::printf("explicit vs fused softmax: rmse=%.3e max_abs=%.3e at [%lld,%lld,%lld] (a=%.6f b=%.6f)\n",
        std::sqrt(sum2/n), max_abs, worst % S_k, (worst / S_k) % S_q, worst / (S_k*S_q),
        a[worst], b[worst]);
    double max_up = 0, max_uc = 0;
    for (int64_t i = 0; i < n; ++i) {
        max_up = std::max(max_up, (double)std::fabs(u[i] - c[i]));
        max_uc = std::max(max_uc, (double)std::fabs(a[i] - b[i]));
    }
    std::printf("unchunked attn vs chunked+fused attn (same graph): max_abs=%.3e\n", max_up);

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(inbuf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
