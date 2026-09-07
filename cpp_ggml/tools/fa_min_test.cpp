// Minimal isolation test for the model.cpp F16+flash attention usage.
// Compares ggml_flash_attn_ext (as called by model.cpp) against the
// hand-written F32 softmax path on identical inputs.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct ctx_handle { ggml_context * ctx; };

static void compare(const char * tag, const std::vector<float> & a, const std::vector<float> & b) {
    double max_abs = 0, sum2 = 0; size_t worst = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > max_abs) { max_abs = d; worst = i; }
        sum2 += ((double)a[i]-b[i])*((double)a[i]-b[i]);
    }
    std::printf("%-28s rmse=%.3e max_abs=%.3e (fa=%.6f manual=%.6f)\n",
        tag, std::sqrt(sum2/a.size()), max_abs, a[worst], b[worst]);
}

int main(int argc, char ** argv) {
    const char * backend_name = argc > 1 ? argv[1] : "CUDA0";
    const int64_t hd = 64, heads = 16, tokens = 1370, batch = 1;
    const int64_t dim = hd * heads;

    ggml_backend_t backend = ggml_backend_init_by_name(backend_name, nullptr);
    if (!backend) { std::fprintf(stderr, "backend %s not found\n", backend_name); return 1; }
    std::printf("backend: %s\n", ggml_backend_name(backend));
    const ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    struct ggml_init_params ip = { 256ull*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    ggml_tensor * q_src  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, heads, tokens, batch);
    ggml_tensor * k16_src = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hd, heads, tokens, batch);
    ggml_tensor * v16_src = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hd, heads, tokens, batch);
    ggml_tensor * k32_src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, heads, tokens, batch);
    ggml_tensor * v32_src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, heads, tokens, batch);
    // A7 cache-side keys: S_k > S_q to mirror the streaming cache shape
    const int64_t k_tokens = 20000;
    ggml_tensor * kbig_src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, heads, k_tokens);
    ggml_backend_buffer_t inbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);

    std::vector<float> qh(hd*heads*tokens), kh(hd*heads*tokens), vh(hd*heads*tokens);
    for (auto & x : qh) x = dist(rng);
    for (auto & x : kh) x = dist(rng);
    for (auto & x : vh) x = dist(rng);
    std::vector<ggml_fp16_t> k16h(hd*heads*tokens), v16h(hd*heads*tokens);
    for (size_t i = 0; i < kh.size(); ++i) k16h[i] = ggml_fp32_to_fp16(kh[i]);
    for (size_t i = 0; i < vh.size(); ++i) v16h[i] = ggml_fp32_to_fp16(vh[i]);
    ggml_backend_tensor_set(q_src,  qh.data(),  0, qh.size()*4);
    ggml_backend_tensor_set(k16_src, k16h.data(), 0, k16h.size()*2);
    ggml_backend_tensor_set(v16_src, v16h.data(), 0, v16h.size()*2);
    ggml_backend_tensor_set(k32_src, k16h.empty() ? nullptr : (void*)kh.data(), 0, kh.size()*4);
    ggml_backend_tensor_set(v32_src, (void*)vh.data(), 0, vh.size()*4);
    std::vector<float> kb(hd*heads*k_tokens);
    {
        std::mt19937 rng2(77);
        std::uniform_real_distribution<float> dist2(-2.0f, 2.0f);
        for (auto & x : kb) x = dist2(rng2);
    }
    ggml_backend_tensor_set(kbig_src, kb.data(), 0, kb.size()*4);

    const float scale = 1.0f / std::sqrt((float)hd);
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);

#ifndef A7_ONLY
    // ---- ground truth B: manual F32 softmax path on F16-rounded K/V ----
    std::vector<float> outb(dim*tokens);
    {
        auto * qm = ggml_cont(ctx, ggml_permute(ctx, q_src, 0, 2, 1, 3));
        auto * km = ggml_cont(ctx, ggml_permute(ctx, k32_src, 0, 2, 1, 3));
        auto * vm = ggml_cont(ctx, ggml_permute(ctx, v32_src, 1, 2, 0, 3));
        auto * scores = ggml_mul_mat(ctx, km, qm);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        scores = ggml_scale(ctx, scores, scale);
        scores = ggml_soft_max(ctx, scores);
        auto * outm = ggml_mul_mat(ctx, vm, scores);
        ggml_mul_mat_set_prec(outm, GGML_PREC_F32);
        outm = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, outm, 0, 2, 1, 3)), dim, tokens, batch);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, outm);
        ggml_gallocr_alloc_graph(galloc, g);
        ggml_backend_graph_compute(backend, g);
        ggml_backend_tensor_get(outm, outb.data(), 0, outb.size()*4);
    }

    // ---- A1: FA strided F16 K/V (current model.cpp) ----
    {
        auto * qf = ggml_permute(ctx, q_src, 0, 2, 1, 3);
        auto * kf = ggml_permute(ctx, k16_src, 0, 2, 1, 3);
        auto * vf = ggml_permute(ctx, v16_src, 0, 2, 1, 3);
        auto * fa = ggml_flash_attn_ext(ctx, qf, kf, vf, nullptr, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        auto * out = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, fa, 0, 2, 1, 3)), dim, tokens, batch);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);
        ggml_gallocr_alloc_graph(galloc, g);
        ggml_backend_graph_compute(backend, g);
        std::vector<float> outa(dim*tokens);
        ggml_backend_tensor_get(out, outa.data(), 0, outa.size()*4);
        compare("A1 FA strided F16 KV", outa, outb);
    }

    // ---- A4: FA strided F16 K/V, output consumed directly (fix candidate) ----
    {
        auto * qf = ggml_permute(ctx, q_src, 0, 2, 1, 3);
        auto * kf = ggml_permute(ctx, k16_src, 0, 2, 1, 3);
        auto * vf = ggml_permute(ctx, v16_src, 0, 2, 1, 3);
        auto * fa = ggml_flash_attn_ext(ctx, qf, kf, vf, nullptr, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        // fa is already [hd, heads, S_q, batch] — the token-major layout the
        // manual path materializes with its final permute+cont.
        auto * out = ggml_reshape_3d(ctx, fa, dim, tokens, batch);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);
        ggml_gallocr_alloc_graph(galloc, g);
        ggml_backend_graph_compute(backend, g);
        std::vector<float> outa(dim*tokens);
        ggml_backend_tensor_get(out, outa.data(), 0, outa.size()*4);
        compare("A4 FA direct reshape", outa, outb);
    }

    // ---- A2: FA with contiguous K/V (cont before permute) ----
    {
        auto * qf = ggml_cont(ctx, ggml_permute(ctx, q_src, 0, 2, 1, 3));
        auto * kf = ggml_cont(ctx, ggml_permute(ctx, k16_src, 0, 2, 1, 3));
        auto * vf = ggml_cont(ctx, ggml_permute(ctx, v16_src, 0, 2, 1, 3));
        auto * fa = ggml_flash_attn_ext(ctx, qf, kf, vf, nullptr, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        auto * out = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, fa, 0, 2, 1, 3)), dim, tokens, batch);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);
        ggml_gallocr_alloc_graph(galloc, g);
        ggml_backend_graph_compute(backend, g);
        std::vector<float> outa(dim*tokens);
        ggml_backend_tensor_get(out, outa.data(), 0, outa.size()*4);
        compare("A2 FA contig F16 KV", outa, outb);
    }

    // ---- A3: FA strided F32 K/V ----
    {
        auto * qf = ggml_permute(ctx, q_src, 0, 2, 1, 3);
        auto * kf = ggml_permute(ctx, k32_src, 0, 2, 1, 3);
        auto * vf = ggml_permute(ctx, v32_src, 0, 2, 1, 3);
        auto * fa = ggml_flash_attn_ext(ctx, qf, kf, vf, nullptr, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        auto * out = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, fa, 0, 2, 1, 3)), dim, tokens, batch);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);
        ggml_gallocr_alloc_graph(galloc, g);
        ggml_backend_graph_compute(backend, g);
        std::vector<float> outa(dim*tokens);
        ggml_backend_tensor_get(out, outa.data(), 0, outa.size()*4);
        compare("A3 FA strided F32 KV", outa, outb);
    }

    // ---- A5: FA F32 unrounded K/V + direct reshape vs manual F32 unrounded
    //         (pure kernel agreement on identical F32 values) ----
    {
        auto * qf = ggml_permute(ctx, q_src, 0, 2, 1, 3);
        auto * kf = ggml_permute(ctx, k32_src, 0, 2, 1, 3);
        auto * vf = ggml_permute(ctx, v32_src, 0, 2, 1, 3);
        auto * fa = ggml_flash_attn_ext(ctx, qf, kf, vf, nullptr, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        auto * out = ggml_reshape_3d(ctx, fa, dim, tokens, batch);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);
        ggml_gallocr_alloc_graph(galloc, g);
        ggml_backend_graph_compute(backend, g);
        std::vector<float> outa(dim*tokens);
        ggml_backend_tensor_get(out, outa.data(), 0, outa.size()*4);
        compare("A5 FA F32 == manual F32", outa, outb);
    }

    // ---- A6: FA F16 K/V vs manual on the SAME F16-rounded values ----
    {
        // rebuild ground truth with F16-rounded values upcast to F32
        auto * km16 = ggml_cont(ctx, ggml_permute(ctx, k16_src, 0, 2, 1, 3));
        km16 = ggml_cast(ctx, km16, GGML_TYPE_F32);
        auto * vm16 = ggml_cont(ctx, ggml_permute(ctx, v16_src, 1, 2, 0, 3));
        vm16 = ggml_cast(ctx, vm16, GGML_TYPE_F32);
        auto * qm = ggml_cont(ctx, ggml_permute(ctx, q_src, 0, 2, 1, 3));
        auto * scores = ggml_mul_mat(ctx, km16, qm);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        scores = ggml_scale(ctx, scores, scale);
        scores = ggml_soft_max(ctx, scores);
        auto * outm = ggml_mul_mat(ctx, vm16, scores);
        ggml_mul_mat_set_prec(outm, GGML_PREC_F32);
        outm = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, outm, 0, 2, 1, 3)), dim, tokens, batch);
        ggml_cgraph * g2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(g2, outm);
        ggml_gallocr_alloc_graph(galloc, g2);
        ggml_backend_graph_compute(backend, g2);
        std::vector<float> outb16(dim*tokens);
        ggml_backend_tensor_get(outm, outb16.data(), 0, outb16.size()*4);

        auto * qf = ggml_permute(ctx, q_src, 0, 2, 1, 3);
        auto * kf = ggml_permute(ctx, k16_src, 0, 2, 1, 3);
        auto * vf = ggml_permute(ctx, v16_src, 0, 2, 1, 3);
        auto * fa = ggml_flash_attn_ext(ctx, qf, kf, vf, nullptr, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        auto * out = ggml_reshape_3d(ctx, fa, dim, tokens, batch);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);
        ggml_gallocr_alloc_graph(galloc, g);
        ggml_backend_graph_compute(backend, g);
        std::vector<float> outa(dim*tokens);
        ggml_backend_tensor_get(out, outa.data(), 0, outa.size()*4);
        compare("A6 FA F16 == manual F16up", outa, outb16);
    }
#endif // A7_ONLY

    // ---- A7: strided-b query chunk handling (model.cpp chunked attention) ----
    // model.cpp views the contiguous [hd, S_q, heads] query as [hd, qc, heads]
    // chunks whose ne2 stride spans the full S_q, i.e. the [D, S_k, H] x
    // [D, qc, H, B] shape. Per (qc, chunk), in separate single-output graphs
    // on identical device inputs, this checks:
    //   (a) the ggml_cont copy of the strided view against host data,
    //   (b) mul_mat on the strided view against a host-computed reference,
    //   (c) mul_mat on the ggml_cont copy against the same reference.
    // A correct backend must pass all three for every qc.
    {
        const int64_t S_k = k_tokens;
        auto * km = ggml_cont(ctx, ggml_permute(ctx, kbig_src, 0, 2, 1, 3)); // [hd, S_k, heads]
        auto * qm = ggml_cont(ctx, ggml_permute(ctx, q_src, 0, 2, 1, 3)); // [hd, S_q, heads, 1]

        // sanity: km/qm device contents and one full-width mul_mat row
        {
            auto * qm_big = ggml_view_4d(ctx, qm, hd, 342, heads, 1,
                qm->nb[1], qm->nb[2], qm->nb[3], 0);
            auto * s_full = ggml_mul_mat(ctx, km, qm_big); // [S_k, 342, heads] ~438 MB
            ggml_mul_mat_set_prec(s_full, GGML_PREC_F32);
            ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, km);
            ggml_build_forward_expand(g, qm);
            ggml_build_forward_expand(g, s_full);
            if (!ggml_gallocr_alloc_graph(galloc, g)) {
                std::fprintf(stderr, "A7 sanity: alloc failed\n");
                return 1;
            }
            ggml_backend_graph_compute(backend, g);
            float buf[4];
            ggml_backend_tensor_get(km, buf, 0, 16);
            std::printf("A7 sanity km[0..3]=%g,%g,%g,%g (expect %g,%g,%g,%g)\n",
                buf[0], buf[1], buf[2], buf[3], kb[0], kb[1], kb[2], kb[3]);
            ggml_backend_tensor_get(qm, buf, 0, 16);
            std::printf("A7 sanity qm[0..3]=%g,%g,%g,%g (expect %g,%g,%g,%g)\n",
                buf[0], buf[1], buf[2], buf[3], qh[0], qh[1], qh[2], qh[3]);
            ggml_backend_tensor_get(s_full, buf, 0, 16);
            float ref4[4];
            for (int64_t k = 0; k < 4; ++k) {
                double acc = 0;
                for (int64_t d = 0; d < hd; ++d)
                    acc += (double)kb[d + hd*heads*k] * (double)qh[d];
                ref4[k] = (float)acc;
            }
            std::printf("A7 sanity s_full[0..3,0,0]=%g,%g,%g,%g (expect %g,%g,%g,%g)\n",
                buf[0], buf[1], buf[2], buf[3], ref4[0], ref4[1], ref4[2], ref4[3]);
        }

        for (int64_t chunk_q : {64, 80, 96, 128, 160, 256, 512}) {
            int64_t bad_a = 0, chunks = 0;
            double max_err_b = 0, max_err_c = 0;
            for (int64_t q0 = 0; q0 + chunk_q <= tokens; q0 += chunk_q) {
                auto * qv = ggml_view_4d(ctx, qm, hd, chunk_q, heads, 1,
                    qm->nb[1], qm->nb[2], qm->nb[3], q0 * qm->nb[1]);
                auto * qv_c = ggml_cont(ctx, qv);
                auto * s_strided = ggml_mul_mat(ctx, km, qv);
                ggml_mul_mat_set_prec(s_strided, GGML_PREC_F32);
                auto * s_cont = ggml_mul_mat(ctx, km, qv_c);
                ggml_mul_mat_set_prec(s_cont, GGML_PREC_F32);
                ++chunks;

                // (a) pure-copy graph
                {
                    ggml_cgraph * g = ggml_new_graph(ctx);
                    ggml_build_forward_expand(g, qv_c);
                    if (!ggml_gallocr_alloc_graph(galloc, g)) {
                        std::fprintf(stderr, "A7(a): alloc failed q0=%lld qc=%lld\n", (long long)q0, (long long)chunk_q);
                        return 1;
                    }
                    ggml_backend_graph_compute(backend, g);
                    std::vector<float> qvc(hd*chunk_q*heads);
                    ggml_backend_tensor_get(qv_c, qvc.data(), 0, qvc.size()*4);
                    // qv_c[d, l, h] must equal q_src[d, h, q0+l] = qh[d + hd*(h + heads*(q0+l))]
                    for (int64_t h = 0; h < heads; ++h)
                        for (int64_t l = 0; l < chunk_q; ++l)
                            for (int64_t d = 0; d < hd; ++d)
                                if (qvc[d + hd*(l + chunk_q*h)] != qh[d + hd*(h + heads*(q0+l))])
                                    ++bad_a;
                }

                // (b) strided mul_mat, (c) cont mul_mat — first 4 key rows vs host
                {
                    ggml_cgraph * g = ggml_new_graph(ctx);
                    ggml_build_forward_expand(g, s_strided);
                    ggml_build_forward_expand(g, s_cont);
                    if (!ggml_gallocr_alloc_graph(galloc, g)) {
                        std::fprintf(stderr, "A7(bc): alloc failed q0=%lld qc=%lld\n", (long long)q0, (long long)chunk_q);
                        return 1;
                    }
                    ggml_backend_graph_compute(backend, g);
                    const int64_t kref = 4;
                    float row[kref], ref[kref];
                    for (int64_t h = 0; h < heads; ++h)
                        for (int64_t l = 0; l < chunk_q; ++l) {
                            for (int64_t k = 0; k < kref; ++k) {
                                double acc = 0;
                                for (int64_t d = 0; d < hd; ++d)
                                    acc += (double)kb[d + hd*(h + heads*k)] *
                                           (double)qh[d + hd*(h + heads*(q0+l))];
                                ref[k] = (float)acc;
                            }
                            // s[k, l, h] lives at k + l*S_k + h*S_k*chunk_q
                            ggml_backend_tensor_get(s_strided, row, (h*S_k*chunk_q + l*S_k)*4, kref*4);
                            for (int64_t k = 0; k < kref; ++k)
                                max_err_b = std::max(max_err_b, (double)std::fabs(row[k] - ref[k]));
                            ggml_backend_tensor_get(s_cont, row, (h*S_k*chunk_q + l*S_k)*4, kref*4);
                            for (int64_t k = 0; k < kref; ++k)
                                max_err_c = std::max(max_err_c, (double)std::fabs(row[k] - ref[k]));
                        }
                }
            }
            std::printf("A7 qc=%4lld: cont-copy bad=%lld/%lld mul_mat-strided max_err=%.3e mul_mat-cont max_err=%.3e (%lld chunks)\n",
                (long long)chunk_q, (long long)bad_a, (long long)(chunks*hd*chunk_q*heads),
                max_err_b, max_err_c, (long long)chunks);
        }
    }

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(inbuf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
