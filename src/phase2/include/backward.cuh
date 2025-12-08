#pragma once
// backward.cuh - backward pass orchestration and device SGD update wrappers
// depends on kernels.cuh and GPUAutoencoder declaration

#include "kernels.cuh"
#include "gpu_autoencoder.cuh"

#ifndef BACKWARD_CUH
#define BACKWARD_CUH

// Run backward for single sample (assumes forward stored activations into net's device buffers)
// This function accumulates gradients into net.d_g_* arrays (device) — do not reset them here if you
// want to accumulate across a batch. Caller should zero d_g_* before batch.
void backward_single_device(GPUAutoencoder& net);

// Apply SGD update on device: for each weight/bias, weight -= lr * (grad / batch_size)
// After update, gradient buffers will be zeroed (done in kernels).
void sgd_update_on_device(GPUAutoencoder& net, int batch_size, float lr);

#endif // BACKWARD_CUH
