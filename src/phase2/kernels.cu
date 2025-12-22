// kernels.cu - forward and backward naive kernels + wrappers
#include "kernels.cuh"
#include "gpu_autoencoder.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>

// ---------------- Forward kernels ----------------

__global__ void conv2d_forward_naive_kernel(const float* __restrict__ input,
                                            const float* __restrict__ weights,
                                            const float* __restrict__ bias,
                                            float* __restrict__ output,
                                            int inC, int inH, int inW, int outC, int k)
{
    int outH = inH;
    int outW = inW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outC * outH * outW;
    if (idx >= total) return;

    int w = idx % outW;
    int tmp = idx / outW;
    int h = tmp % outH;
    int oc = tmp / outH;

    float sum = 0.0f;
    int pad = k/2;
    for (int ic = 0; ic < inC; ++ic) {
        const float* in_ch = input + ic * (inH * inW);
        for (int kh = 0; kh < k; ++kh) {
            int ih = h + (kh - pad);
            for (int kw = 0; kw < k; ++kw) {
                int iw = w + (kw - pad);
                float val = 0.0f;
                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                    val = in_ch[ih * inW + iw];
                }
                int widx = oc * (inC * k * k) + ic * (k * k) + kh * k + kw;
                sum += val * weights[widx];
            }
        }
    }
    if (bias) sum += bias[oc];
    output[idx] = sum;
}

__global__ void relu_inplace_kernel(float* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = x[i];
        x[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void maxpool2x2_forward_kernel(const float* in, float* out, int* max_idx,
                                          int C, int H, int W)
{
    int outH = H / 2;
    int outW = W / 2;
    int total = C * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int w = idx % outW;
    int tmp = idx / outW;
    int h = tmp % outH;
    int c = tmp / outH;

    int inBase = c * (H * W);
    int h0 = h * 2;
    int w0 = w * 2;
    float best = -1e10f;
    int bi = 0;
    for (int dh = 0; dh < 2; ++dh) {
        for (int dw = 0; dw < 2; ++dw) {
            int ih = h0 + dh;
            int iw = w0 + dw;
            float v = in[inBase + ih * W + iw];
            int li = dh * 2 + dw;
            if (v > best) { best = v; bi = li; }
        }
    }
    out[idx] = best;
    max_idx[idx] = bi;
}

__global__ void upsample_nn_forward_kernel(const float* in, float* out, int inC, int inH, int inW) {
    int outH = inH * 2;
    int outW = inW * 2;
    int total = inC * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int w = idx % outW;
    int tmp = idx / outW;
    int h = tmp % outH;
    int c = tmp / outH;

    int ih = h / 2;
    int iw = w / 2;
    const float* in_ch = in + c * (inH * inW);
    out[idx] = in_ch[ih * inW + iw];
}

__global__ void mse_loss_and_grad_kernel(const float* pred, const float* target, float* grad, float* loss_accum, int N) {
    extern __shared__ float ssum[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float local = 0.0f;
    if (idx < N) {
        float d = pred[idx] - target[idx];
        grad[idx] = 2.0f * d / (float)N;
        local = d * d / (float)N;
    } else {
        local = 0.0f;
    }
    ssum[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(loss_accum, ssum[0]);
}

// ---------------- Backward kernels ----------------

// ReLU backward: grad_in = grad_out * (act > 0)
__global__ void relu_backward_kernel(const float* grad_out, const float* act, float* grad_in, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        grad_in[i] = (act[i] > 0.0f) ? grad_out[i] : 0.0f;
    }
}

// Maxpool 2x2 backward: place grad_out back to grad_in according to max_idx
__global__ void maxpool2x2_backward_kernel(const float* grad_out, float* grad_in, const int* max_idx,
                                           int C, int H, int W)
{
    int outH = H / 2;
    int outW = W / 2;
    int total = C * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int w = idx % outW;
    int tmp = idx / outW;
    int h = tmp % outH;
    int c = tmp / outH;

    int inBase = c * (H * W);
    int h0 = h * 2;
    int w0 = w * 2;
    int which = max_idx[idx];
    int dh = which / 2;
    int dw = which % 2;
    int ih = h0 + dh;
    int iw = w0 + dw;
    int inIdx = inBase + ih * W + iw;
    // grad_in location is unique for this out element, atomic not necessary but safe
    atomicAdd(&grad_in[inIdx], grad_out[idx]);
}

// Upsample backward: sum 2x2 block from grad_out into grad_in
__global__ void upsample_backward_kernel(const float* grad_out, float* grad_in, int inC, int inH, int inW) {
    int inIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalIn = inC * inH * inW;
    if (inIdx >= totalIn) return;
    int iw = inIdx % inW;
    int tmp = inIdx / inW;
    int ih = tmp % inH;
    int ic = tmp / inH;
    int outH = inH * 2, outW = inW * 2;
    int base_out = ic * (outH * outW);
    int out_h0 = ih * 2;
    int out_w0 = iw * 2;
    float acc = 0.0f;
    acc += grad_out[base_out + (out_h0  )*outW + (out_w0  )];
    acc += grad_out[base_out + (out_h0  )*outW + (out_w0+1)];
    acc += grad_out[base_out + (out_h0+1)*outW + (out_w0  )];
    acc += grad_out[base_out + (out_h0+1)*outW + (out_w0+1)];
    grad_in[inIdx] = acc;
}

// Conv weight grad: each thread handles one weight index (oc,ic,kh,kw) and loops over spatial to accumulate
// **atomicAdd** to grad_weights because kernel may be called concurrently for multiple samples.
__global__ void conv2d_weight_grad_naive_kernel(const float* input, const float* grad_out,
                                                float* grad_weights, float* grad_bias,
                                                int inC, int inH, int inW, int outC, int k)
{
    int totalW = outC * inC * k * k;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalW) {
        // compute bias accumulation for a separate range: handled below
        return;
    }
    int tmp = idx;
    int kw = tmp % k; tmp /= k;
    int kh = tmp % k; 
    tmp /= k; // careful: previous line bug-prone; easier recompute cleanly below
    // Recompute properly:
    tmp = idx;
    kw = tmp % k; tmp /= k;
    kh = tmp % k; tmp /= k;
    int ic = tmp % inC; tmp /= inC;
    int oc = tmp;

    float acc = 0.0f;
    int pad = k/2;
    for (int h = 0; h < inH; ++h) {
        for (int w = 0; w < inW; ++w) {
            int ih = h + (kh - pad);
            int iw = w + (kw - pad);
            float inVal = 0.0f;
            if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                inVal = input[ic * (inH * inW) + ih * inW + iw];
            }
            float gout = grad_out[oc * (inH * inW) + h * inW + w];
            acc += inVal * gout;
        }
    }
    atomicAdd(&grad_weights[idx], acc);

    // compute bias contributions for oc threads (we'll let first outC threads compute bias to avoid extra kernel)
    // but simpler: have separate kernel below to accumulate bias.
}

// Simpler bias accumulation kernel: each thread per oc
__global__ void conv2d_bias_grad_kernel(const float* grad_out, float* grad_bias, int outC, int H, int W) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= outC) return;
    float acc = 0.0f;
    int total = H * W;
    const float* gout_ch = grad_out + oc * total;
    for (int i = 0; i < total; ++i) acc += gout_ch[i];
    atomicAdd(&grad_bias[oc], acc);
}

// Conv input grad: each thread computes one input pixel (ic, h, w)
__global__ void conv2d_input_grad_naive_kernel(const float* grad_out, const float* weights,
                                               float* grad_input,
                                               int inC, int inH, int inW, int outC, int k)
{
    int inIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalIn = inC * inH * inW;
    if (inIdx >= totalIn) return;
    int iw = inIdx % inW;
    int tmp = inIdx / inW;
    int ih = tmp % inH;
    int ic = tmp / inH;
    float acc = 0.0f;
    int pad = k/2;
    for (int oc = 0; oc < outC; ++oc) {
        for (int kh = 0; kh < k; ++kh) {
            for (int kw = 0; kw < k; ++kw) {
                int oh = ih - (kh - pad);
                int ow = iw - (kw - pad);
                if (oh >= 0 && oh < inH && ow >= 0 && ow < inW) {
                    float gout = grad_out[oc * (inH * inW) + oh * inW + ow];
                    int widx = oc * (inC * k * k) + ic * (k * k) + kh * k + kw;
                    float wv = weights[widx];
                    acc += gout * wv;
                }
            }
        }
    }
    grad_input[inIdx] = acc;
}

// Update weights on device: w[idx] -= lr * (g[idx] / batch_size); then zero g[idx]
__global__ void update_weights_on_device_kernel(float* weights, float* grads, int N, float lr, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = grads[i] / (float)batch_size;
    weights[i] -= lr * g;
    grads[i] = 0.0f;
}
__global__ void update_bias_on_device_kernel(float* bias, float* grads, int N, float lr, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = grads[i] / (float)batch_size;
    bias[i] -= lr * g;
    grads[i] = 0.0f;
}

// ---------------- wrappers ----------------

void launch_conv2d_forward_naive(const float* input, const float* weights, const float* bias,
                                 float* output, int inC, int inH, int inW, int outC, int k)
{
    int total = outC * inH * inW;
    int block = 256;
    int grid = (total + block - 1) / block;
    conv2d_forward_naive_kernel<<<grid, block>>>(input, weights, bias, output, inC, inH, inW, outC, k);
    CHECK(cudaGetLastError());
}

void launch_relu_inplace(float* x, int N) {
    int block = 256;
    int grid = (N + block - 1) / block;
    relu_inplace_kernel<<<grid, block>>>(x, N);
    CHECK(cudaGetLastError());
}

void launch_maxpool2x2_forward(const float* in, float* out, int* max_idx, int C, int H, int W) {
    int total = C * (H/2) * (W/2);
    int block = 256;
    int grid = (total + block - 1) / block;
    maxpool2x2_forward_kernel<<<grid, block>>>(in, out, max_idx, C, H, W);
    CHECK(cudaGetLastError());
}

void launch_upsample_nn_forward(const float* in, float* out, int inC, int inH, int inW) {
    int total = inC * (inH*2) * (inW*2);
    int block = 256;
    int grid = (total + block - 1) / block;
    upsample_nn_forward_kernel<<<grid, block>>>(in, out, inC, inH, inW);
    CHECK(cudaGetLastError());
}

void launch_mse_loss_and_grad(const float* pred, const float* target, float* grad, float* loss_accum, int N) {
    int block = 256;
    int grid = (N + block - 1) / block;
    size_t shmem = block * sizeof(float);
    mse_loss_and_grad_kernel<<<grid, block, shmem>>>(pred, target, grad, loss_accum, N);
    CHECK(cudaGetLastError());
}

// Backward wrappers

void launch_relu_backward(const float* grad_out, const float* act, float* grad_in, int N) {
    int block = 256;
    int grid = (N + block - 1) / block;
    relu_backward_kernel<<<grid, block>>>(grad_out, act, grad_in, N);
    CHECK(cudaGetLastError());
}

void launch_maxpool2x2_backward(const float* grad_out, float* grad_in, const int* max_idx, int C, int H, int W) {
    int total = C * (H/2) * (W/2);
    int block = 256;
    int grid = (total + block - 1) / block;
    maxpool2x2_backward_kernel<<<grid, block>>>(grad_out, grad_in, max_idx, C, H, W);
    CHECK(cudaGetLastError());
}

void launch_upsample_backward(const float* grad_out, float* grad_in, int inC, int inH, int inW) {
    int total = inC * inH * inW;
    int block = 256;
    int grid = (total + block - 1) / block;
    upsample_backward_kernel<<<grid, block>>>(grad_out, grad_in, inC, inH, inW);
    CHECK(cudaGetLastError());
}

void launch_conv2d_weight_grad_naive(const float* input, const float* grad_out,
                                     float* grad_weights, float* grad_bias,
                                     int inC, int inH, int inW, int outC, int k) {
    int totalW = outC * inC * k * k;
    int block = 256;
    int gridW = (totalW + block - 1) / block;
    conv2d_weight_grad_naive_kernel<<<gridW, block>>>(input, grad_out, grad_weights, grad_bias, inC, inH, inW, outC, k);
    CHECK(cudaGetLastError());
    // bias grads
    int gridB = (outC + block - 1) / block;
    conv2d_bias_grad_kernel<<<gridB, block>>>(grad_out, grad_bias, outC, inH, inW);
    CHECK(cudaGetLastError());
}

void launch_conv2d_input_grad_naive(const float* grad_out, const float* weights,
                                    float* grad_input, int inC, int inH, int inW, int outC, int k) {
    int totalIn = inC * inH * inW;
    int block = 256;
    int grid = (totalIn + block - 1) / block;
    conv2d_input_grad_naive_kernel<<<grid, block>>>(grad_out, weights, grad_input, inC, inH, inW, outC, k);
    CHECK(cudaGetLastError());
}

void launch_update_weights_on_device(float* weights, float* grads, int N, float lr, int batch_size) {
    int block = 256;
    int grid = (N + block - 1) / block;
    update_weights_on_device_kernel<<<grid, block>>>(weights, grads, N, lr, batch_size);
    CHECK(cudaGetLastError());
}

void launch_update_bias_on_device(float* bias, float* grads, int N, float lr, int batch_size) {
    int block = 256;
    int grid = (N + block - 1) / block;
    update_bias_on_device_kernel<<<grid, block>>>(bias, grads, N, lr, batch_size);
    CHECK(cudaGetLastError());
}
