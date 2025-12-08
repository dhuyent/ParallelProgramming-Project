#pragma once
// forward.cuh - forward pass declarations that use GPUAutoencoder device buffers and kernels
// forward_single_device: runs encoder+decoder for one sample (device buffers), computes loss
// forward_batch_host: copies a batch of host images to device, runs forward per-sample, returns average loss

#include "gpu_autoencoder.cuh" // your device buffers class
#include "kernels.cuh"

#ifndef FORWARD_CUH
#define FORWARD_CUH

// Forward for a single sample (device pointers for input and target must be ready)
// Returns loss (host float)
float forward_single_device(GPUAutoencoder& net, float* d_input_sample, float* d_target_sample);

// Forward for a host batch: h_inputs is array of host pointers to float image data (NCHW flattened).
// Copies each sample to device (net.d_input), runs forward_single_device, accumulates loss.
// Returns average loss over batch.
float forward_batch_on_host(GPUAutoencoder& net, float** h_inputs, float** h_targets, int start_idx, int batch_size, int n_samples_total);

#endif // FORWARD_CUH
