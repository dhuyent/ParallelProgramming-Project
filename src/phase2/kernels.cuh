#pragma once
// kernels.cuh - declarations for naive kernels (forward + backward)
// No extern "C" — C++ usage only.

#include <cuda_runtime.h>

#ifndef KERNELS_CUH
#define KERNELS_CUH

// Simple CUDA check if not defined elsewhere
#ifndef CHECK
#include <cstdio>
inline void CHECK(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::fprintf(stderr,"CUDA Error: %s %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) std::exit(code);
    }
}
#define CHECK(x) CHECK((x), __FILE__, __LINE__)
#endif

// -------------------- Forward wrappers --------------------
// conv forward: input (inC*H*W), weights(outC*inC*k*k), bias(outC) -> output(outC*H*W)
void launch_conv2d_forward_naive(const float* input, const float* weights, const float* bias,
                                 float* output, int inC, int inH, int inW, int outC, int k);

// relu in-place
void launch_relu_inplace(float* x, int N);

// maxpool 2x2 forward + indices
void launch_maxpool2x2_forward(const float* in, float* out, int* max_idx, int C, int H, int W);

// upsample nearest neighbour forward (scale=2)
void launch_upsample_nn_forward(const float* in, float* out, int inC, int inH, int inW);

// mse loss + grad
void launch_mse_loss_and_grad(const float* pred, const float* target, float* grad, float* loss_accum, int N);

// -------------------- Backward wrappers --------------------
// ReLU backward: grad_out, act -> grad_in
void launch_relu_backward(const float* grad_out, const float* act, float* grad_in, int N);

// MaxPool 2x2 backward: grad_out (C,H/2,W/2), max_idx -> grad_in (C,H,W)
void launch_maxpool2x2_backward(const float* grad_out, float* grad_in, const int* max_idx, int C, int H, int W);

// Upsample backward: grad_out (C, H*2, W*2) -> grad_in (C, H, W) (sum of 4 outputs)
void launch_upsample_backward(const float* grad_out, float* grad_in, int inC, int inH, int inW);

// Conv weight-grad: input (inC,H,W), grad_out (outC,H,W)
// accumulates into grad_weights (outC*inC*k*k) and grad_bias (outC)
// Uses atomicAdd to safely accumulate when called per-sample in a batch.
void launch_conv2d_weight_grad_naive(const float* input, const float* grad_out,
                                     float* grad_weights, float* grad_bias,
                                     int inC, int inH, int inW, int outC, int k);

// Conv input-grad: grad_out (outC,H,W), weights -> grad_input (inC,H,W)
void launch_conv2d_input_grad_naive(const float* grad_out, const float* weights,
                                    float* grad_input, int inC, int inH, int inW, int outC, int k);

// Weight update on device: weight -= lr * (grad / batch_size), then zero grad
void launch_update_weights_on_device(float* weights, float* grads, int N, float lr, int batch_size);

// Bias update on device
void launch_update_bias_on_device(float* bias, float* grads, int N, float lr, int batch_size);

#endif // KERNELS_CUH
