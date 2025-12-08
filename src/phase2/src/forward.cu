// forward.cu - forward pass implementations using kernels.cu wrappers and GPUAutoencoder device buffers
#include "forward.cuh"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>

// forward_single_device: uses net.d_input (or accepts d_input_sample pointer) and writes prediction into net.d_out
// It expects net device activations allocated and weights on device.
float forward_single_device(GPUAutoencoder& net, float* d_input_sample, float* d_target_sample) {
    // Encoder
    // conv1: (3,32,32) -> (256,32,32)
    launch_conv2d_forward_naive(d_input_sample, net.d_w_conv1, net.d_b_conv1, net.d_act1, 3, 32, 32, 256, GPUAutoencoder::KERNEL);
    CHECK(cudaDeviceSynchronize());
    launch_relu_inplace(net.d_act1, 256*32*32);
    CHECK(cudaDeviceSynchronize());

    // pool1 -> (256,16,16)
    launch_maxpool2x2_forward(net.d_act1, net.d_pool1, net.d_pool1_idx, 256, 32, 32);
    CHECK(cudaDeviceSynchronize());

    // conv2 -> (128,16,16)
    launch_conv2d_forward_naive(net.d_pool1, net.d_w_conv2, net.d_b_conv2, net.d_act2, 256, 16, 16, 128, GPUAutoencoder::KERNEL);
    CHECK(cudaDeviceSynchronize());
    launch_relu_inplace(net.d_act2, 128*16*16);
    CHECK(cudaDeviceSynchronize());

    // pool2 -> latent (128,8,8)
    launch_maxpool2x2_forward(net.d_act2, net.d_latent, net.d_pool2_idx, 128, 16, 16);
    CHECK(cudaDeviceSynchronize());

    // Decoder
    // dec1 conv (128->128) on latent
    launch_conv2d_forward_naive(net.d_latent, net.d_w_dec1, net.d_b_dec1, net.d_dec_act1, 128, 8, 8, 128, GPUAutoencoder::KERNEL);
    CHECK(cudaDeviceSynchronize());
    launch_relu_inplace(net.d_dec_act1, 128*8*8);
    CHECK(cudaDeviceSynchronize());

    // upsample to 16x16
    launch_upsample_nn_forward(net.d_dec_act1, net.d_up1, 128, 8, 8);
    CHECK(cudaDeviceSynchronize());

    // dec2 conv (128->256)
    launch_conv2d_forward_naive(net.d_up1, net.d_w_dec2, net.d_b_dec2, net.d_dec_act2, 128, 16, 16, 256, GPUAutoencoder::KERNEL);
    CHECK(cudaDeviceSynchronize());
    launch_relu_inplace(net.d_dec_act2, 256*16*16);
    CHECK(cudaDeviceSynchronize());

    // upsample to 32x32
    launch_upsample_nn_forward(net.d_dec_act2, net.d_up2, 256, 16, 16);
    CHECK(cudaDeviceSynchronize());

    // dec3 conv -> output (3,32,32)
    launch_conv2d_forward_naive(net.d_up2, net.d_w_dec3, net.d_b_dec3, net.d_out, 256, 32, 32, 3, GPUAutoencoder::KERNEL);
    CHECK(cudaDeviceSynchronize());

    // compute loss + grad
    int total_out = 3 * 32 * 32;
    CHECK(cudaMemset(net.d_grad_out, 0, total_out * sizeof(float)));
    // allocate device scalar loss_accum on stack and zero it
    float* d_loss_accum = nullptr;
    CHECK(cudaMalloc(&d_loss_accum, sizeof(float)));
    CHECK(cudaMemset(d_loss_accum, 0, sizeof(float)));
    launch_mse_loss_and_grad(net.d_out, d_target_sample, net.d_grad_out, d_loss_accum, total_out);
    CHECK(cudaDeviceSynchronize());
    float h_loss = 0.0f;
    CHECK(cudaMemcpy(&h_loss, d_loss_accum, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_loss_accum));

    return h_loss;
}

// forward_batch_on_host: since activations are per-sample in net, we loop samples in batch,
// copy each host->device into net.d_input then call forward_single_device (per-sample).
float forward_batch_on_host(GPUAutoencoder& net, float** h_inputs, float** h_targets, int start_idx, int batch_size, int n_samples_total) {
    int end = start_idx + batch_size;
    if (end > n_samples_total) end = n_samples_total;
    int actual_bs = end - start_idx;
    size_t s_input = (size_t)3*32*32 * sizeof(float);
    float batch_loss = 0.0f;
    for (int i = start_idx; i < end; ++i) {
        // copy host input to net.d_input (device)
        CHECK(cudaMemcpy(net.d_input, h_inputs[i], s_input, cudaMemcpyHostToDevice));
        // copy host target to a temporary device buffer (we can reuse net.d_out_target if provided; but keep it simple)
        // we'll reuse net.d_out (prediction) for output, so create a device buffer for target
        float* d_target = nullptr;
        CHECK(cudaMalloc(&d_target, s_input));
        CHECK(cudaMemcpy(d_target, h_targets[i], s_input, cudaMemcpyHostToDevice));

        float sample_loss = forward_single_device(net, net.d_input, d_target);
        batch_loss += sample_loss;

        CHECK(cudaFree(d_target));
    }
    return batch_loss / (float)actual_bs;
}
