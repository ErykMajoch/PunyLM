#pragma once
#include <cstddef>
#include <cstdint>

// The only symbols nvcc must provide!

#ifdef PUNYLM_ENABLE_CUDA
namespace punylm::backend::cuda {

    // Memory
    float* device_malloc(std::size_t n_floats);
    void device_free(float* p);
    void copy_h2d(float* dst_dev, const float* src_host, std::size_t n);
    void copy_d2h(float* dst_host, const float* src_dev, std::size_t n);
    void memset_zero(float* dev, std::size_t n);

    // Elementwise and BLAS
    void axpy(float* y, const float* x, float a, std::size_t n); // y += a * x
    void scale(float* out, const float* x, float a, std::size_t n); // out = a * x
    void add(float* out, const float* a, const float* b, std::size_t n);
    void mul(float* out, const float* a, const float* b, std::size_t n);
    void silu_fwd(float* out, const float* x, std::size_t n);
    void silu_bwd(float* dx, const float* x, const float* dout, std::size_t n);

    void matmul(float* C, const float* A, const float* B, int m, int k, int n); // Row major C[m,n] = A[m,k] * B[k,n]

    void softmax_lastdim_fwd(float* out, const float* x, int rows, int cols);
    void softmax_lastdim_bwd(float* dx, const float* y, const float* dout, int rows, int cols);

    void rmsnorm_fwd(float* out, const float* x, const float* w, int rows, int cols, float eps);
    void rmsnorm_bwd(float* dx, float* dw, const float* x, const float* w, const float* dout, int rows, int cols, float eps);

    void embedding_fwd(float* out, const float* table, const int* ids, int n_ids, int dim);
    void embedding_bwd(float* dtable, const int* ids, const float* dout, int n_ids, int dim);

    void cross_entropy_fwd(float* loss, float* probs, const float* logits, const int* targets, int rows, int vocab);
    void cross_entropy_bwd(float* dlogits, const float* probs, const int* targets, int rows, int vocab, float scale);

    void adamw_step(float* p, float* m, float* v, const float* g, int n, float lr, float beta1, float beta2, float eps, float wd, float bias1, float bias2);

} // namespace punylm::backend::cuda
#endif
