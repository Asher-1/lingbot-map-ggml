// Minimal reproduction: ggml_mul_mat with a strided-b query chunk.
//
// The chunked attention in lingbot-map's model.cpp views the contiguous
// [hd, S_q, heads] query as [hd, qc, heads] chunks whose ne2 stride spans
// the full S_q, and feeds that strided-b tensor straight into ggml_mul_mat:
//   src0 = km  [hd, S_k, heads]      (contiguous)
//   src1 = qv  [hd, qc, heads, 1]    (view into qm; nb[2] = S_q*hd*ts)
// This program checks s[k, l, h] = sum_d km[d, k, h] * qv[d, l, h] against
// a host-computed reference for the first chunk (q0 = 0), repeating the
// alloc/compute cycle several times on the same gallocr to expose
// nondeterminism.
//
// Build (CUDA):
//   g++ -O2 -std=c++17 -I third_party/ggml/include -I third_party/ggml/src \
//       tools/mul_mat_strided_b_test.cpp -o /tmp/mmsb \
//       -L build-cuda/ggml/src -L build-cuda/ggml/src/ggml-cuda \
//       -lggml -lggml-base -lggml-cpu -lggml-cuda
// Run: /tmp/mmsb CPU   |   /tmp/mmsb CUDA0   |   /tmp/mmsb Vulkan0
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    const char * backend_name = argc > 1 ? argv[1] : "CPU";
    const int64_t hd = 64, heads = 16, S_q = 1370, S_k = 20000, qc = 64;

    ggml_backend_t backend = ggml_backend_init_by_name(backend_name, nullptr);
    if (!backend) { std::fprintf(stderr, "backend %s not found\n", backend_name); return 1; }
    std::printf("backend: %s\n", ggml_backend_name(backend));
    const ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    struct ggml_init_params ip = { 256ull*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    ggml_tensor * k_src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, heads, S_k);
    ggml_tensor * q_src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, heads, S_q, 1);
    ggml_backend_buffer_t inbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);

    std::vector<float> kh(hd*heads*S_k), qh(hd*heads*S_q);
    for (auto & x : kh) x = dist(rng);
    for (auto & x : qh) x = dist(rng);
    ggml_backend_tensor_set(k_src, kh.data(), 0, kh.size()*4);
    ggml_backend_tensor_set(q_src, qh.data(), 0, qh.size()*4);

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);

    // host reference: s[k, l, h] = sum_d kh[d + hd*(h + heads*k)] * qh[d + hd*(h + heads*l)]
    const int64_t kref = 4, lref = 4;
    float ref[16];
    for (int64_t h = 0; h < 2; ++h)
        for (int64_t l = 0; l < lref; ++l)
            for (int64_t k = 0; k < kref; ++k) {
                double acc = 0;
                for (int64_t d = 0; d < hd; ++d)
                    acc += (double)kh[d + hd*(h + heads*k)] * (double)qh[d + hd*(h + heads*l)];
                ref[h*lref*kref + l*kref + k] = (float)acc;
            }

    // km/qm created ONCE and shared across multiple galloc graphs
    // (this mirrors the fa_min_test A7 loop and is the failing pattern)
    ggml_tensor * km = ggml_cont(ctx, ggml_permute(ctx, k_src, 0, 2, 1, 3)); // [hd, S_k, heads]
    ggml_tensor * qm = ggml_cont(ctx, ggml_permute(ctx, q_src, 0, 2, 1, 3)); // [hd, S_q, heads, 1]

    // graph-size alternation: run one large graph first so the gallocr must
    // re-reserve (shrink) for the small chunk graphs that follow
    {
        ggml_tensor * qm_big = ggml_view_4d(ctx, qm, hd, 342, heads, 1,
            qm->nb[1], qm->nb[2], qm->nb[3], 0);
        ggml_tensor * s_big = ggml_mul_mat(ctx, km, qm_big); // [S_k, 342, heads] ~438 MB
        ggml_mul_mat_set_prec(s_big, GGML_PREC_F32);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, s_big);
        if (!ggml_gallocr_alloc_graph(galloc, g)) {
            std::fprintf(stderr, "big graph alloc failed\n");
            return 1;
        }
        ggml_backend_graph_compute(backend, g);
    }

    for (int iter = 0; iter < 3; ++iter) {
        for (int64_t q0 : {0, 64, 128, 640}) {
        ggml_tensor * qv = ggml_view_4d(ctx, qm, hd, qc, heads, 1,
            qm->nb[1], qm->nb[2], qm->nb[3], q0 * qm->nb[1]);
        ggml_tensor * s = ggml_mul_mat(ctx, km, qv); // [S_k, qc, heads]
        ggml_mul_mat_set_prec(s, GGML_PREC_F32);
        ggml_tensor * qv_c = ggml_cont(ctx, qv);
        ggml_tensor * s2 = ggml_mul_mat(ctx, km, qv_c); // second output in the same graph
        ggml_mul_mat_set_prec(s2, GGML_PREC_F32);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, s);
        ggml_build_forward_expand(g, s2);
        if (!ggml_gallocr_alloc_graph(galloc, g)) {
            std::fprintf(stderr, "iter %d: alloc failed\n", iter);
            return 1;
        }
        ggml_backend_graph_compute(backend, g);

        double max_err = 0;
        for (int64_t h = 0; h < 2; ++h)
            for (int64_t l = 0; l < lref; ++l) {
                float got[4];
                ggml_backend_tensor_get(s, got, (h*S_k*qc + l*S_k)*4, kref*4);
                for (int64_t k = 0; k < kref; ++k) {
                    // reference against the chunk starting at token q0+l
                    double acc = 0;
                    for (int64_t d = 0; d < hd; ++d)
                        acc += (double)kh[d + hd*(h + heads*k)] * (double)qh[d + hd*(h + heads*(q0+l))];
                    max_err = std::max(max_err, std::fabs((double)got[k] - acc));
                }
            }
        std::printf("iter %d q0=%3lld: mul_mat strided-b [64,%lld,16]x[64,%lld,16,1] max_err=%.3e -> %s\n",
            iter, (long long)q0, (long long)S_k, (long long)qc, max_err, max_err < 1e-2 ? "PASS" : "FAIL");
        // second output (mul_mat on the cont copy) in the same graph
        max_err = 0;
        for (int64_t h = 0; h < 2; ++h)
            for (int64_t l = 0; l < lref; ++l) {
                float got[4];
                ggml_backend_tensor_get(s2, got, (h*S_k*qc + l*S_k)*4, kref*4);
                for (int64_t k = 0; k < kref; ++k) {
                    double acc = 0;
                    for (int64_t d = 0; d < hd; ++d)
                        acc += (double)kh[d + hd*(h + heads*k)] * (double)qh[d + hd*(h + heads*(q0+l))];
                    max_err = std::max(max_err, std::fabs((double)got[k] - acc));
                }
            }
        std::printf("iter %d q0=%3lld: mul_mat cont-copy  [64,%lld,16]x[64,%lld,16,1] max_err=%.3e -> %s\n",
            iter, (long long)q0, (long long)S_k, (long long)qc, max_err, max_err < 1e-2 ? "PASS" : "FAIL");
        }
    }

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(inbuf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
