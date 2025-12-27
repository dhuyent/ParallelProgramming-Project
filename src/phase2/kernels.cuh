#pragma once
#include <cuda_runtime.h>

#ifndef KERNELS_CUH
#define KERNELS_CUH

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

void launch_conv2d_forward_naive(const float* input, const float* weights, const float* bias,
                                 float* output, int inC, int inH, int inW, int outC, int k);

void launch_relu_inplace(float* x, int N);

void launch_maxpool2x2_forward(const float* in, float* out, int* max_idx, int C, int H, int W);

void launch_upsample_nn_forward(const float* in, float* out, int inC, int inH, int inW);

void launch_mse_loss_and_grad(const float* pred, const float* target, float* grad, float* loss_accum, int N);

void launch_relu_backward(const float* grad_out, const float* act, float* grad_in, int N);

void launch_maxpool2x2_backward(const float* grad_out, float* grad_in, const int* max_idx, int C, int H, int W);

void launch_upsample_backward(const float* grad_out, float* grad_in, int inC, int inH, int inW);
.
void launch_conv2d_weight_grad_naive(const float* input, const float* grad_out,
                                     float* grad_weights, float* grad_bias,
                                     int inC, int inH, int inW, int outC, int k);

void launch_conv2d_input_grad_naive(const float* grad_out, const float* weights,
                                    float* grad_input, int inC, int inH, int inW, int outC, int k);

void launch_update_weights_on_device(float* weights, float* grads, int N, float lr, int batch_size);

void launch_update_bias_on_device(float* bias, float* grads, int N, float lr, int batch_size);

#endif // KERNELS_CUH
