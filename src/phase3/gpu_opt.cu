#include "gpu_opt.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

static constexpr int Cin = 3;
static constexpr int C1 = 256;
static constexpr int C2 = 128;
static constexpr int C3 = 128;
static constexpr int C4 = 256;
static constexpr int Cout = 3;

__device__ __forceinline__ int idx4(int n, int c, int h, int w, int C, int H, int W)
{
    return ((n * C + c) * H + h) * W + w;
}
static inline int divUp(int a, int b) { return (a + b - 1) / b; }

__constant__ float c_w1[C1 * Cin * 3 * 3];

static void malloc_async(void **p, size_t bytes, cudaStream_t s)
{
#if CUDART_VERSION >= 11020
    CUDA_CHECK(cudaMallocAsync(p, bytes, s));
#else
    (void)s;
    CUDA_CHECK(cudaMalloc(p, bytes));
#endif
}
static void free_async(void *p, cudaStream_t s)
{
#if CUDART_VERSION >= 11020
    CUDA_CHECK(cudaFreeAsync(p, s));
#else
    (void)s;
    CUDA_CHECK(cudaFree(p));
#endif
}

__global__ void maxpool2x2_fwd(const float *__restrict__ x, float *__restrict__ y,
                               int N, int C, int H, int W)
{
    int h2 = blockIdx.y * blockDim.y + threadIdx.y;
    int w2 = blockIdx.x * blockDim.x + threadIdx.x;
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc - n * C;
    int H2 = H >> 1, W2 = W >> 1;
    if (n >= N || h2 >= H2 || w2 >= W2)
        return;
    int h = h2 * 2, w = w2 * 2;
    float m = x[idx4(n, c, h, w, C, H, W)];
    m = fmaxf(m, x[idx4(n, c, h, w + 1, C, H, W)]);
    m = fmaxf(m, x[idx4(n, c, h + 1, w, C, H, W)]);
    m = fmaxf(m, x[idx4(n, c, h + 1, w + 1, C, H, W)]);
    y[idx4(n, c, h2, w2, C, H2, W2)] = m;
}

__global__ void upsample2x2_nn_fwd(const float *__restrict__ x, float *__restrict__ y,
                                   int N, int C, int H, int W)
{
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc - n * C;
    int H2 = H * 2, W2 = W * 2;
    if (n >= N || h >= H2 || w >= W2)
        return;
    y[idx4(n, c, h, w, C, H2, W2)] = x[idx4(n, c, h >> 1, w >> 1, C, H, W)];
}

__global__ void mse_loss_and_grad_batch(
    const float *__restrict__ out,
    const float *__restrict__ target,
    float *__restrict__ g_out,
    float *__restrict__ loss_out,
    int total)
{
    __shared__ float sh[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float v = 0.f;
    if (i < total)
    {
        float d = out[i] - target[i];
        g_out[i] = (2.f / (float)total) * d;
        v = d * d;
    }
    sh[tid] = v;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(loss_out, sh[0] / (float)total);
}

__global__ void accumulate_epoch_loss(const float *__restrict__ batch_loss,
                                      float *__restrict__ epoch_loss)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        atomicAdd(epoch_loss, batch_loss[0]);
    }
}

__global__ void relu_bwd(const float *__restrict__ act,
                         const float *__restrict__ g_out,
                         float *__restrict__ g_in,
                         int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
        g_in[i] = (act[i] > 0.f) ? g_out[i] : 0.f;
}

__global__ void upsample2x2_nn_bwd(const float *__restrict__ g_y,
                                   float *__restrict__ g_x,
                                   int N, int C, int H, int W)
{
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc - n * C;
    if (n >= N || h >= H || w >= W)
        return;
    int H2 = H * 2, W2 = W * 2;
    int oh = h * 2, ow = w * 2;
    float s = 0.f;
    s += g_y[idx4(n, c, oh, ow, C, H2, W2)];
    s += g_y[idx4(n, c, oh, ow + 1, C, H2, W2)];
    s += g_y[idx4(n, c, oh + 1, ow, C, H2, W2)];
    s += g_y[idx4(n, c, oh + 1, ow + 1, C, H2, W2)];
    g_x[idx4(n, c, h, w, C, H, W)] = s;
}

__global__ void maxpool2x2_bwd(const float *__restrict__ x,
                               const float *__restrict__ y,
                               const float *__restrict__ g_y,
                               float *__restrict__ g_x,
                               int N, int C, int H, int W)
{
    int h2 = blockIdx.y * blockDim.y + threadIdx.y;
    int w2 = blockIdx.x * blockDim.x + threadIdx.x;
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc - n * C;
    int H2 = H >> 1, W2 = W >> 1;
    if (n >= N || h2 >= H2 || w2 >= W2)
        return;

    int h = h2 * 2, w = w2 * 2;
    float out = y[idx4(n, c, h2, w2, C, H2, W2)];
    float gx = g_y[idx4(n, c, h2, w2, C, H2, W2)];

    int base = idx4(n, c, h, w, C, H, W);
    float v00 = x[base], v01 = x[base + 1], v10 = x[base + W], v11 = x[base + W + 1];

    g_x[base] = (v00 == out) ? gx : 0.f;
    g_x[base + 1] = (v01 == out) ? gx : 0.f;
    g_x[base + W] = (v10 == out) ? gx : 0.f;
    g_x[base + W + 1] = (v11 == out) ? gx : 0.f;
}

template <int TILE_H, int TILE_W, bool USE_CONST_W1>
__global__ void conv3x3_bias_relu_fwd_opt(
    const float *__restrict__ x,
    const float *__restrict__ w,
    const float *__restrict__ b,
    float *__restrict__ y,
    int N, int Cin_, int H, int W, int Cout_)
{
    int nz = blockIdx.z;
    int n = nz / Cout_;
    int oc = nz - n * Cout_;
    if (n >= N)
        return;

    int out_h0 = blockIdx.y * TILE_H;
    int out_w0 = blockIdx.x * TILE_W;

    int th = threadIdx.y;
    int tw = threadIdx.x;

    extern __shared__ float sh[];
    int SH_H = TILE_H + 2;
    int SH_W = TILE_W + 2;

    float acc = 0.f;

    for (int ic = 0; ic < Cin_; ++ic)
    {
        for (int lh = th; lh < SH_H; lh += blockDim.y)
        {
            int ih = out_h0 + lh - 1;
            bool in_h = (ih >= 0 && ih < H);
            for (int lw = tw; lw < SH_W; lw += blockDim.x)
            {
                int iw = out_w0 + lw - 1;
                bool in_w = (iw >= 0 && iw < W);
                float v = 0.f;
                if (in_h && in_w)
                    v = x[idx4(n, ic, ih, iw, Cin_, H, W)];
                sh[lh * SH_W + lw] = v;
            }
        }
        __syncthreads();

        int oh = out_h0 + th;
        int ow = out_w0 + tw;
        if (oh < H && ow < W)
        {
            int base = th * SH_W + tw;
            float ww[9];

            if constexpr (USE_CONST_W1)
            {
                int off = ((oc * Cin_ + ic) * 9);
#pragma unroll
                for (int k = 0; k < 9; ++k)
                    ww[k] = c_w1[off + k];
            }
            else
            {
                const float *src = w + ((oc * Cin_ + ic) * 9);
#pragma unroll
                for (int k = 0; k < 9; ++k)
                    ww[k] = src[k];
            }

            acc += sh[base + 0 * SH_W + 0] * ww[0];
            acc += sh[base + 0 * SH_W + 1] * ww[1];
            acc += sh[base + 0 * SH_W + 2] * ww[2];
            acc += sh[base + 1 * SH_W + 0] * ww[3];
            acc += sh[base + 1 * SH_W + 1] * ww[4];
            acc += sh[base + 1 * SH_W + 2] * ww[5];
            acc += sh[base + 2 * SH_W + 0] * ww[6];
            acc += sh[base + 2 * SH_W + 1] * ww[7];
            acc += sh[base + 2 * SH_W + 2] * ww[8];
        }
        __syncthreads();
    }

    int oh = out_h0 + th;
    int ow = out_w0 + tw;
    if (oh < H && ow < W)
    {
        acc += b[oc];
        if (acc < 0.f)
            acc = 0.f;
        y[idx4(n, oc, oh, ow, Cout_, H, W)] = acc;
    }
}

template <int TILE_H, int TILE_W>
__global__ void conv3x3_bias_fwd_opt(const float *x, const float *w, const float *b, float *y,
                                     int N, int Cin_, int H, int W, int Cout_)
{
    int nz = blockIdx.z;
    int n = nz / Cout_;
    int oc = nz - n * Cout_;
    if (n >= N)
        return;

    int out_h0 = blockIdx.y * TILE_H;
    int out_w0 = blockIdx.x * TILE_W;
    int th = threadIdx.y;
    int tw = threadIdx.x;

    extern __shared__ float sh[];
    int SH_H = TILE_H + 2;
    int SH_W = TILE_W + 2;

    float acc = 0.f;

    for (int ic = 0; ic < Cin_; ++ic)
    {
        for (int lh = th; lh < SH_H; lh += blockDim.y)
        {
            int ih = out_h0 + lh - 1;
            bool in_h = (ih >= 0 && ih < H);
            for (int lw = tw; lw < SH_W; lw += blockDim.x)
            {
                int iw = out_w0 + lw - 1;
                bool in_w = (iw >= 0 && iw < W);
                sh[lh * SH_W + lw] = (in_h && in_w) ? x[idx4(n, ic, ih, iw, Cin_, H, W)] : 0.f;
            }
        }
        __syncthreads();

        int oh = out_h0 + th;
        int ow = out_w0 + tw;
        if (oh < H && ow < W)
        {
            int base = th * SH_W + tw;
            const float *ww = w + ((oc * Cin_ + ic) * 9);
#pragma unroll
            for (int k = 0; k < 9; ++k)
            {
                acc += sh[base + (k / 3) * SH_W + (k % 3)] * ww[k];
            }
        }
        __syncthreads();
    }

    int oh = out_h0 + th;
    int ow = out_w0 + tw;
    if (oh < H && ow < W)
    {
        acc += b[oc];
        y[idx4(n, oc, oh, ow, Cout_, H, W)] = acc;
    }
}

__global__ void conv3x3_dx_naive(
    const float *__restrict__ w,
    const float *__restrict__ g_y,
    float *__restrict__ g_x,
    int N, int Cin_, int H, int W, int Cout_)
{
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int nic = blockIdx.z;
    int n = nic / Cin_;
    int ic = nic - n * Cin_;
    if (n >= N || ih >= H || iw >= W)
        return;

    float acc = 0.f;
    for (int oc = 0; oc < Cout_; ++oc)
    {
        for (int kh = 0; kh < 3; ++kh)
        {
            for (int kw = 0; kw < 3; ++kw)
            {
                int oh = ih - (kh - 1);
                int ow = iw - (kw - 1);
                if (oh >= 0 && oh < H && ow >= 0 && ow < W)
                {
                    float gy = g_y[idx4(n, oc, oh, ow, Cout_, H, W)];
                    float ww = w[((oc * Cin_ + ic) * 9) + kh * 3 + kw];
                    acc += gy * ww;
                }
            }
        }
    }
    g_x[idx4(n, ic, ih, iw, Cin_, H, W)] = acc;
}

template <int TILE_H, int TILE_W>
__global__ void conv3x3_bwd_dwdb_tiled(
    const float *__restrict__ x,
    const float *__restrict__ g_y,
    float *__restrict__ g_w,
    float *__restrict__ g_b,
    int N, int Cin_, int H, int W, int Cout_)
{
    int oc = blockIdx.z;
    int ic = blockIdx.y;

    int tilesW = (W + TILE_W - 1) / TILE_W;
    int tile_id = blockIdx.x;
    int tile_h = tile_id / tilesW;
    int tile_w = tile_id - tile_h * tilesW;

    int oh0 = tile_h * TILE_H;
    int ow0 = tile_w * TILE_W;

    int th = threadIdx.y;
    int tw = threadIdx.x;
    int oh = oh0 + th;
    int ow = ow0 + tw;

    float dw[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    float db = 0.f;

    if (oh < H && ow < W)
    {
        for (int n = 0; n < N; ++n)
        {
            float gy = g_y[idx4(n, oc, oh, ow, Cout_, H, W)];
            db += gy;
#pragma unroll
            for (int kh = 0; kh < 3; ++kh)
            {
#pragma unroll
                for (int kw = 0; kw < 3; ++kw)
                {
                    int ih = oh + kh - 1;
                    int iw = ow + kw - 1;
                    float xv = 0.f;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        xv = x[idx4(n, ic, ih, iw, Cin_, H, W)];
                    dw[kh * 3 + kw] += gy * xv;
                }
            }
        }
    }

    __shared__ float sh_db[8][16];
    __shared__ float sh_dw[9][8][16];

    sh_db[th][tw] = db;
#pragma unroll
    for (int k = 0; k < 9; ++k)
        sh_dw[k][th][tw] = dw[k];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tw < s)
        {
            sh_db[th][tw] += sh_db[th][tw + s];
#pragma unroll
            for (int k = 0; k < 9; ++k)
                sh_dw[k][th][tw] += sh_dw[k][th][tw + s];
        }
        __syncthreads();
    }

    for (int s = blockDim.y / 2; s > 0; s >>= 1)
    {
        if (tw == 0 && th < s)
        {
            sh_db[th][0] += sh_db[th + s][0];
#pragma unroll
            for (int k = 0; k < 9; ++k)
                sh_dw[k][th][0] += sh_dw[k][th + s][0];
        }
        __syncthreads();
    }

    if (th == 0 && tw == 0)
    {
        atomicAdd(&g_b[oc], sh_db[0][0]);
        float *gw = g_w + ((oc * Cin_ + ic) * 9);
#pragma unroll
        for (int k = 0; k < 9; ++k)
            atomicAdd(&gw[k], sh_dw[k][0][0]);
    }
}

__global__ void sgd_update(float *w, float *g, float lr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        w[i] -= lr * g[i];
        g[i] = 0.f;
    }
}
__global__ void sgd_update_bias(float *b, float *g, float lr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        b[i] -= lr * g[i];
        g[i] = 0.f;
    }
}

void AEWeightsDev::alloc(cudaStream_t s)
{
    auto A = [&](float **p, size_t bytes)
    { malloc_async((void **)p, bytes, s); };
    auto Z = [&](void *p, size_t bytes)
    { CUDA_CHECK(cudaMemsetAsync(p, 0, bytes, s)); };

    size_t n_w1 = C1 * Cin * 9, n_b1 = C1;
    size_t n_w2 = C2 * C1 * 9, n_b2 = C2;
    size_t n_w3 = C3 * C3 * 9, n_b3 = C3;
    size_t n_w4 = C4 * C3 * 9, n_b4 = C4;
    size_t n_w5 = Cout * C4 * 9, n_b5 = Cout;

    A(&w1, n_w1 * sizeof(float));
    A(&b1, n_b1 * sizeof(float));
    A(&w2, n_w2 * sizeof(float));
    A(&b2, n_b2 * sizeof(float));
    A(&w3, n_w3 * sizeof(float));
    A(&b3, n_b3 * sizeof(float));
    A(&w4, n_w4 * sizeof(float));
    A(&b4, n_b4 * sizeof(float));
    A(&w5, n_w5 * sizeof(float));
    A(&b5, n_b5 * sizeof(float));

    A(&gw1, n_w1 * sizeof(float));
    A(&gb1, n_b1 * sizeof(float));
    A(&gw2, n_w2 * sizeof(float));
    A(&gb2, n_b2 * sizeof(float));
    A(&gw3, n_w3 * sizeof(float));
    A(&gb3, n_b3 * sizeof(float));
    A(&gw4, n_w4 * sizeof(float));
    A(&gb4, n_b4 * sizeof(float));
    A(&gw5, n_w5 * sizeof(float));
    A(&gb5, n_b5 * sizeof(float));

    Z(gw1, n_w1 * sizeof(float));
    Z(gb1, n_b1 * sizeof(float));
    Z(gw2, n_w2 * sizeof(float));
    Z(gb2, n_b2 * sizeof(float));
    Z(gw3, n_w3 * sizeof(float));
    Z(gb3, n_b3 * sizeof(float));
    Z(gw4, n_w4 * sizeof(float));
    Z(gb4, n_b4 * sizeof(float));
    Z(gw5, n_w5 * sizeof(float));
    Z(gb5, n_b5 * sizeof(float));
}

void AEWeightsDev::free(cudaStream_t s)
{
    free_async(w1, s);
    free_async(b1, s);
    free_async(w2, s);
    free_async(b2, s);
    free_async(w3, s);
    free_async(b3, s);
    free_async(w4, s);
    free_async(b4, s);
    free_async(w5, s);
    free_async(b5, s);

    free_async(gw1, s);
    free_async(gb1, s);
    free_async(gw2, s);
    free_async(gb2, s);
    free_async(gw3, s);
    free_async(gb3, s);
    free_async(gw4, s);
    free_async(gb4, s);
    free_async(gw5, s);
    free_async(gb5, s);
}

void AEBuffersDev::alloc(int N, cudaStream_t s)
{
    auto A = [&](float **p, size_t n)
    { malloc_async((void **)p, n * sizeof(float), s); };

    size_t n_x = (size_t)N * Cin * 32 * 32;
    size_t n_c1 = (size_t)N * C1 * 32 * 32;
    size_t n_p1 = (size_t)N * C1 * 16 * 16;
    size_t n_c2 = (size_t)N * C2 * 16 * 16;
    size_t n_lat = (size_t)N * C2 * 8 * 8;
    size_t n_c3 = (size_t)N * C3 * 8 * 8;
    size_t n_u1 = (size_t)N * C3 * 16 * 16;
    size_t n_c4 = (size_t)N * C4 * 16 * 16;
    size_t n_u2 = (size_t)N * C4 * 32 * 32;
    size_t n_out = (size_t)N * Cout * 32 * 32;

    A(&x, n_x);
    A(&c1, n_c1);
    A(&p1, n_p1);
    A(&c2, n_c2);
    A(&lat, n_lat);
    A(&c3, n_c3);
    A(&u1, n_u1);
    A(&c4, n_c4);
    A(&u2, n_u2);
    A(&out, n_out);

    A(&g_out, n_out);
    A(&g_u2, n_u2);
    A(&g_c4, n_c4);
    A(&g_u1, n_u1);
    A(&g_c3, n_c3);
    A(&g_lat, n_lat);
    A(&g_c2, n_c2);
    A(&g_p1, n_p1);
    A(&g_c1, n_c1);
    A(&g_x, n_x);

    malloc_async((void **)&d_loss, sizeof(float), s);
    malloc_async((void **)&d_epoch_loss, sizeof(float), s);

    CUDA_CHECK(cudaMemsetAsync(d_loss, 0, sizeof(float), s));
    CUDA_CHECK(cudaMemsetAsync(d_epoch_loss, 0, sizeof(float), s));
}

void AEBuffersDev::free(cudaStream_t s)
{
    free_async(x, s);
    free_async(c1, s);
    free_async(p1, s);
    free_async(c2, s);
    free_async(lat, s);
    free_async(c3, s);
    free_async(u1, s);
    free_async(c4, s);
    free_async(u2, s);
    free_async(out, s);

    free_async(g_out, s);
    free_async(g_u2, s);
    free_async(g_c4, s);
    free_async(g_u1, s);
    free_async(g_c3, s);
    free_async(g_lat, s);
    free_async(g_c2, s);
    free_async(g_p1, s);
    free_async(g_c1, s);
    free_async(g_x, s);

    free_async(d_loss, s);
    free_async(d_epoch_loss, s);
}

void Phase3Engine::init(const AEParams &params)
{
    p = params;
    batchBytes = (size_t)p.batch * Cin * 32 * 32 * sizeof(float);

    for (int i = 0; i < 2; ++i)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&sH2D[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sCompute[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&eH2D[i]));
        CUDA_CHECK(cudaMallocHost((void **)&h_batchPinned[i], batchBytes));
    }

    w.alloc(sCompute[0]);
    b.alloc(p.batch, sCompute[0]);
}

void Phase3Engine::shutdown()
{
    sync_all();
    b.free(sCompute[0]);
    w.free(sCompute[0]);
    for (int i = 0; i < 2; ++i)
    {
        if (h_batchPinned[i])
            CUDA_CHECK(cudaFreeHost(h_batchPinned[i]));
        CUDA_CHECK(cudaEventDestroy(eH2D[i]));
        CUDA_CHECK(cudaStreamDestroy(sH2D[i]));
        CUDA_CHECK(cudaStreamDestroy(sCompute[i]));
    }
}

void Phase3Engine::sync_all() { CUDA_CHECK(cudaDeviceSynchronize()); }

void Phase3Engine::reset_epoch_loss()
{
    CUDA_CHECK(cudaMemsetAsync(b.d_epoch_loss, 0, sizeof(float), sCompute[0]));
}

float Phase3Engine::get_epoch_loss_avg_sync(int steps_per_epoch)
{
    float h = 0.f;
    CUDA_CHECK(cudaMemcpyAsync(&h, b.d_epoch_loss, sizeof(float),
                               cudaMemcpyDeviceToHost, sCompute[0]));
    CUDA_CHECK(cudaStreamSynchronize(sCompute[0]));
    return h / (float)steps_per_epoch;
}

static void launch_conv_relu(cudaStream_t s,
                             const float *x, const float *w, const float *b, float *y,
                             int N, int Cin_, int H, int W, int Cout_, bool useConstW1)
{
    constexpr int TILE_H = 8, TILE_W = 16;
    dim3 block(TILE_W, TILE_H);
    dim3 grid(divUp(W, TILE_W), divUp(H, TILE_H), N * Cout_);
    size_t shmem = (TILE_H + 2) * (TILE_W + 2) * sizeof(float);

    if (useConstW1)
    {
        conv3x3_bias_relu_fwd_opt<TILE_H, TILE_W, true><<<grid, block, shmem, s>>>(x, w, b, y, N, Cin_, H, W, Cout_);
    }
    else
    {
        conv3x3_bias_relu_fwd_opt<TILE_H, TILE_W, false><<<grid, block, shmem, s>>>(x, w, b, y, N, Cin_, H, W, Cout_);
    }
}

static void launch_conv_norelu(cudaStream_t s,
                               const float *x, const float *w, const float *b, float *y,
                               int N, int Cin_, int H, int W, int Cout_)
{
    constexpr int TILE_H = 8, TILE_W = 16;
    dim3 block(TILE_W, TILE_H);
    dim3 grid(divUp(W, TILE_W), divUp(H, TILE_H), N * Cout_);
    size_t shmem = (TILE_H + 2) * (TILE_W + 2) * sizeof(float);
    conv3x3_bias_fwd_opt<TILE_H, TILE_W><<<grid, block, shmem, s>>>(x, w, b, y, N, Cin_, H, W, Cout_);
}

void Phase3Engine::train_step_async(const float *x_batch_host, int bufIdx)
{
    std::memcpy(h_batchPinned[bufIdx], x_batch_host, batchBytes);
    CUDA_CHECK(cudaMemcpyAsync(b.x, h_batchPinned[bufIdx], batchBytes,
                               cudaMemcpyHostToDevice, sH2D[bufIdx]));
    CUDA_CHECK(cudaEventRecord(eH2D[bufIdx], sH2D[bufIdx]));
    CUDA_CHECK(cudaStreamWaitEvent(sCompute[bufIdx], eH2D[bufIdx], 0));

    cudaStream_t s = sCompute[bufIdx];
    int N = p.batch;

    CUDA_CHECK(cudaMemsetAsync(b.d_loss, 0, sizeof(float), s));
    launch_conv_relu(s, b.x, w.w1, w.b1, b.c1, N, Cin, 32, 32, C1, false);

    {
        dim3 block(16, 16), grid(divUp(16, 16), divUp(16, 16), N * C1);
        maxpool2x2_fwd<<<grid, block, 0, s>>>(b.c1, b.p1, N, C1, 32, 32);
    }

    launch_conv_relu(s, b.p1, w.w2, w.b2, b.c2, N, C1, 16, 16, C2, false);

    {
        dim3 block(16, 16), grid(divUp(8, 16), divUp(8, 16), N * C2);
        maxpool2x2_fwd<<<grid, block, 0, s>>>(b.c2, b.lat, N, C2, 16, 16);
    }

    launch_conv_relu(s, b.lat, w.w3, w.b3, b.c3, N, C3, 8, 8, C3, false);

    {
        dim3 block(16, 16), grid(divUp(16, 16), divUp(16, 16), N * C3);
        upsample2x2_nn_fwd<<<grid, block, 0, s>>>(b.c3, b.u1, N, C3, 8, 8);
    }

    launch_conv_relu(s, b.u1, w.w4, w.b4, b.c4, N, C3, 16, 16, C4, false);

    {
        dim3 block(16, 16), grid(divUp(32, 16), divUp(32, 16), N * C4);
        upsample2x2_nn_fwd<<<grid, block, 0, s>>>(b.c4, b.u2, N, C4, 16, 16);
    }

    launch_conv_norelu(s, b.u2, w.w5, w.b5, b.out, N, C4, 32, 32, Cout);

    int total = N * Cout * 32 * 32;
    mse_loss_and_grad_batch<<<divUp(total, 256), 256, 0, s>>>(b.out, b.x, b.g_out, b.d_loss, total);

    accumulate_epoch_loss<<<1, 32, 0, s>>>(b.d_loss, b.d_epoch_loss);

    constexpr int TH = 8, TW = 16;
    dim3 blockDW(TW, TH);

    auto launch_dwdb = [&](const float *x, const float *gy, float *gw, float *gb,
                           int CinL, int HL, int WL, int CoutL)
    {
        int tilesW = divUp(WL, TW);
        int tilesH = divUp(HL, TH);
        int gridX = tilesW * tilesH;
        dim3 gridDW(gridX, CinL, CoutL);
        conv3x3_bwd_dwdb_tiled<TH, TW><<<gridDW, blockDW, 0, s>>>(x, gy, gw, gb, N, CinL, HL, WL, CoutL);
    };

    auto launch_dx = [&](const float *w, const float *gy, float *gx,
                         int CinL, int HL, int WL, int CoutL)
    {
        dim3 block2d(16, 16);
        dim3 grid2d(divUp(WL, 16), divUp(HL, 16), N * CinL);
        conv3x3_dx_naive<<<grid2d, block2d, 0, s>>>(w, gy, gx, N, CinL, HL, WL, CoutL);
    };

    CUDA_CHECK(cudaMemsetAsync(w.gw5, 0, (size_t)Cout * C4 * 9 * sizeof(float), s));
    CUDA_CHECK(cudaMemsetAsync(w.gb5, 0, (size_t)Cout * sizeof(float), s));
    launch_dwdb(b.u2, b.g_out, w.gw5, w.gb5, C4, 32, 32, Cout);
    launch_dx(w.w5, b.g_out, b.g_u2, C4, 32, 32, Cout);

    {
        dim3 block(16, 16), grid(divUp(16, 16), divUp(16, 16), N * C4);
        upsample2x2_nn_bwd<<<grid, block, 0, s>>>(b.g_u2, b.g_c4, N, C4, 16, 16);
    }
    relu_bwd<<<divUp(N * C4 * 16 * 16, 256), 256, 0, s>>>(b.c4, b.g_c4, b.g_c4, N * C4 * 16 * 16);

    CUDA_CHECK(cudaMemsetAsync(w.gw4, 0, (size_t)C4 * C3 * 9 * sizeof(float), s));
    CUDA_CHECK(cudaMemsetAsync(w.gb4, 0, (size_t)C4 * sizeof(float), s));
    launch_dwdb(b.u1, b.g_c4, w.gw4, w.gb4, C3, 16, 16, C4);
    launch_dx(w.w4, b.g_c4, b.g_u1, C3, 16, 16, C4);

    {
        dim3 block(16, 16), grid(divUp(8, 16), divUp(8, 16), N * C3);
        upsample2x2_nn_bwd<<<grid, block, 0, s>>>(b.g_u1, b.g_c3, N, C3, 8, 8);
    }
    relu_bwd<<<divUp(N * C3 * 8 * 8, 256), 256, 0, s>>>(b.c3, b.g_c3, b.g_c3, N * C3 * 8 * 8);

    CUDA_CHECK(cudaMemsetAsync(w.gw3, 0, (size_t)C3 * C3 * 9 * sizeof(float), s));
    CUDA_CHECK(cudaMemsetAsync(w.gb3, 0, (size_t)C3 * sizeof(float), s));
    launch_dwdb(b.lat, b.g_c3, w.gw3, w.gb3, C3, 8, 8, C3);
    launch_dx(w.w3, b.g_c3, b.g_lat, C3, 8, 8, C3);

    {
        dim3 block(16, 16), grid(divUp(8, 16), divUp(8, 16), N * C2);
        maxpool2x2_bwd<<<grid, block, 0, s>>>(b.c2, b.lat, b.g_lat, b.g_c2, N, C2, 16, 16);
    }
    relu_bwd<<<divUp(N * C2 * 16 * 16, 256), 256, 0, s>>>(b.c2, b.g_c2, b.g_c2, N * C2 * 16 * 16);

    CUDA_CHECK(cudaMemsetAsync(w.gw2, 0, (size_t)C2 * C1 * 9 * sizeof(float), s));
    CUDA_CHECK(cudaMemsetAsync(w.gb2, 0, (size_t)C2 * sizeof(float), s));
    launch_dwdb(b.p1, b.g_c2, w.gw2, w.gb2, C1, 16, 16, C2);
    launch_dx(w.w2, b.g_c2, b.g_p1, C1, 16, 16, C2);

    {
        dim3 block(16, 16), grid(divUp(16, 16), divUp(16, 16), N * C1);
        maxpool2x2_bwd<<<grid, block, 0, s>>>(b.c1, b.p1, b.g_p1, b.g_c1, N, C1, 32, 32);
    }
    relu_bwd<<<divUp(N * C1 * 32 * 32, 256), 256, 0, s>>>(b.c1, b.g_c1, b.g_c1, N * C1 * 32 * 32);

    CUDA_CHECK(cudaMemsetAsync(w.gw1, 0, (size_t)C1 * Cin * 9 * sizeof(float), s));
    CUDA_CHECK(cudaMemsetAsync(w.gb1, 0, (size_t)C1 * sizeof(float), s));
    launch_dwdb(b.x, b.g_c1, w.gw1, w.gb1, Cin, 32, 32, C1);
    launch_dx(w.w1, b.g_c1, b.g_x, Cin, 32, 32, C1);

    auto updW = [&](float *W, float *G, int n)
    { sgd_update<<<divUp(n, 256), 256, 0, s>>>(W, G, p.lr, n); };
    auto updB = [&](float *B, float *G, int n)
    { sgd_update_bias<<<divUp(n, 256), 256, 0, s>>>(B, G, p.lr, n); };

    updW(w.w5, w.gw5, Cout * C4 * 9);
    updB(w.b5, w.gb5, Cout);
    updW(w.w4, w.gw4, C4 * C3 * 9);
    updB(w.b4, w.gb4, C4);
    updW(w.w3, w.gw3, C3 * C3 * 9);
    updB(w.b3, w.gb3, C3);
    updW(w.w2, w.gw2, C2 * C1 * 9);
    updB(w.b2, w.gb2, C2);
    updW(w.w1, w.gw1, C1 * Cin * 9);
    updB(w.b1, w.gb1, C1);
}

void Phase3Engine::extract_features(const float *x_host_contig, float *feat_host_contig, int Ntotal)
{
    int N = p.batch;
    size_t inBytes = (size_t)N * Cin * 32 * 32 * sizeof(float);
    size_t featBytes = (size_t)N * C2 * 8 * 8 * sizeof(float);

    float *h_inPinned = nullptr;
    float *h_featPinned = nullptr;
    CUDA_CHECK(cudaMallocHost((void **)&h_inPinned, inBytes));
    CUDA_CHECK(cudaMallocHost((void **)&h_featPinned, featBytes));

    CUDA_CHECK(cudaMemcpyToSymbol(c_w1, w.w1, sizeof(c_w1)));

    for (int i = 0; i < Ntotal; i += N)
    {
        int curN = std::min(N, Ntotal - i);
        std::memset(h_inPinned, 0, inBytes);
        std::memcpy(h_inPinned, x_host_contig + (size_t)i * 3072, (size_t)curN * 3072 * sizeof(float));

        CUDA_CHECK(cudaMemcpyAsync(b.x, h_inPinned, inBytes, cudaMemcpyHostToDevice, sCompute[0]));

        // encoder only
        launch_conv_relu(sCompute[0], b.x, w.w1, w.b1, b.c1, N, Cin, 32, 32, C1, true);
        {
            dim3 block(16, 16), grid(divUp(16, 16), divUp(16, 16), N * C1);
            maxpool2x2_fwd<<<grid, block, 0, sCompute[0]>>>(b.c1, b.p1, N, C1, 32, 32);
        }
        launch_conv_relu(sCompute[0], b.p1, w.w2, w.b2, b.c2, N, C1, 16, 16, C2, false);
        {
            dim3 block(16, 16), grid(divUp(8, 16), divUp(8, 16), N * C2);
            maxpool2x2_fwd<<<grid, block, 0, sCompute[0]>>>(b.c2, b.lat, N, C2, 16, 16);
        }

        CUDA_CHECK(cudaMemcpyAsync(h_featPinned, b.lat, featBytes, cudaMemcpyDeviceToHost, sCompute[0]));
        CUDA_CHECK(cudaStreamSynchronize(sCompute[0]));
        std::memcpy(feat_host_contig + (size_t)i * 8192, h_featPinned, (size_t)curN * 8192 * sizeof(float));
    }

    CUDA_CHECK(cudaFreeHost(h_inPinned));
    CUDA_CHECK(cudaFreeHost(h_featPinned));
}

void Phase3Engine::save_to_file(const std::string &filename, cudaStream_t s)
{
    struct WeightEntry
    {
        float *dev_ptr;
        size_t count;
    };

    std::vector<WeightEntry> entries = {
        {w.w1, (size_t)C1 * Cin * 9}, {w.b1, (size_t)C1}, {w.w2, (size_t)C2 * C1 * 9}, {w.b2, (size_t)C2}, {w.w3, (size_t)C3 * C3 * 9}, {w.b3, (size_t)C3}, {w.w4, (size_t)C4 * C3 * 9}, {w.b4, (size_t)C4}, {w.w5, (size_t)Cout * C4 * 9}, {w.b5, (size_t)Cout}};

    std::ofstream os(filename, std::ios::binary);
    if (!os.is_open())
    {
        printf("[Error] Could not open file %s for writing\n", filename.c_str());
        return;
    }

    for (auto &entry : entries)
    {
        size_t bytes = entry.count * sizeof(float);
        std::vector<float> host_buf(entry.count);

        CUDA_CHECK(cudaMemcpyAsync(host_buf.data(), entry.dev_ptr, bytes, cudaMemcpyDeviceToHost, s));
        cudaStreamSynchronize(s);

        os.write(reinterpret_cast<const char *>(host_buf.data()), bytes);
    }
    os.close();
    printf("[System] Weights successfully saved to: %s\n", filename.c_str());
}

void Phase3Engine::load_from_file(const std::string &filename, cudaStream_t s)
{
    std::ifstream is(filename, std::ios::binary | std::ios::ate);
    if (!is.is_open())
        return;

    size_t fileSize = is.tellg();
    is.seekg(0, std::ios::beg);

    std::vector<float> full_host_buf(fileSize / sizeof(float));
    is.read(reinterpret_cast<char *>(full_host_buf.data()), fileSize);
    is.close();

    struct WeightEntry
    {
        float *dev_ptr;
        size_t count;
    };
    std::vector<WeightEntry> entries = {
        {w.w1, (size_t)C1 * Cin * 9}, {w.b1, (size_t)C1}, {w.w2, (size_t)C2 * C1 * 9}, {w.b2, (size_t)C2}, {w.w3, (size_t)C3 * C3 * 9}, {w.b3, (size_t)C3}, {w.w4, (size_t)C4 * C3 * 9}, {w.b4, (size_t)C4}, {w.w5, (size_t)Cout * C4 * 9}, {w.b5, (size_t)Cout}};

    size_t offset = 0;
    for (auto &entry : entries)
    {
        CUDA_CHECK(cudaMemcpyAsync(entry.dev_ptr, full_host_buf.data() + offset,
                                   entry.count * sizeof(float), cudaMemcpyHostToDevice, s));
        offset += entry.count;
    }
    cudaStreamSynchronize(s);
    printf("[System] Weights successfully loaded from: %s\n", filename.c_str());
}

void Phase3Engine::forward_only(int N, cudaStream_t s)
{
    launch_conv_relu(s, b.x, w.w1, w.b1, b.c1, N, Cin, 32, 32, C1, false);

    {
        dim3 block(16, 16), grid(divUp(16, 16), divUp(16, 16), N * C1);
        maxpool2x2_fwd<<<grid, block, 0, s>>>(b.c1, b.p1, N, C1, 32, 32);
    }

    launch_conv_relu(s, b.p1, w.w2, w.b2, b.c2, N, C1, 16, 16, C2, false);

    {
        dim3 block(16, 16), grid(divUp(8, 16), divUp(8, 16), N * C2);
        maxpool2x2_fwd<<<grid, block, 0, s>>>(b.c2, b.lat, N, C2, 16, 16);
    }

    launch_conv_relu(s, b.lat, w.w3, w.b3, b.c3, N, C3, 8, 8, C3, false);

    {
        dim3 block(16, 16), grid(divUp(16, 16), divUp(16, 16), N * C3);
        upsample2x2_nn_fwd<<<grid, block, 0, s>>>(b.c3, b.u1, N, C3, 8, 8);
    }

    launch_conv_relu(s, b.u1, w.w4, w.b4, b.c4, N, C3, 16, 16, C4, false);

    {
        dim3 block(16, 16), grid(divUp(32, 16), divUp(32, 16), N * C4);
        upsample2x2_nn_fwd<<<grid, block, 0, s>>>(b.c4, b.u2, N, C4, 16, 16);
    }

    launch_conv_norelu(s, b.u2, w.w5, w.b5, b.out, N, C4, 32, 32, Cout);
}