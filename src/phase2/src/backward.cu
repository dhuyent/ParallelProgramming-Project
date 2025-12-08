// backward.cu - orchestrate backward pass using kernels and update weights on device
#include "backward.cuh"
#include <cuda_runtime.h>
#include <cstdio>

// Sequence (mirror of forward):
// Given: net.d_grad_out contains gradient of loss wrt output (or forward produced d_grad_out)
// We implement same order as forward but reversed, using device buffers inside net.

void backward_single_device(GPUAutoencoder& net) {
    // Note: this function assumes net.d_grad_out has been set (by MSE kernel in forward)
    // Steps (decoder -> encoder):
    // dec3 (conv 256->3): weight grads from d_up2 and d_grad_out, input grad -> d_grad_up2
    launch_conv2d_weight_grad_naive(net.d_up2, net.d_grad_out, net.d_g_w_dec3, net.d_g_b_dec3, 256, 32, 32, 3, GPUAutoencoder::KERNEL);
    launch_conv2d_input_grad_naive(net.d_grad_out, net.d_w_dec3, net.d_grad_up2, 256, 32, 32, 3, GPUAutoencoder::KERNEL);

    // upsample backward: grad_dec_act2 (16x16) = upsample_backward(d_grad_up2)
    launch_upsample_backward(net.d_grad_up2, net.d_grad_dec_act2, 256, 16, 16);

    // relu backward on dec_act2 (was activated) -> grad_dec_act2 uses dec_act2 as activation
    launch_relu_backward(net.d_grad_dec_act2, net.d_dec_act2, net.d_grad_dec_act2, 256*16*16);

    // dec2 conv (input d_up1 -> out d_dec_act2): weight grad and input grad
    launch_conv2d_weight_grad_naive(net.d_up1, net.d_grad_dec_act2, net.d_g_w_dec2, net.d_g_b_dec2, 128, 16, 16, 256, GPUAutoencoder::KERNEL);
    launch_conv2d_input_grad_naive(net.d_grad_dec_act2, net.d_w_dec2, net.d_grad_up1, 128, 16, 16, 256, GPUAutoencoder::KERNEL);

    // upsample backward to dec_act1
    launch_upsample_backward(net.d_grad_up1, net.d_grad_dec_act1, 128, 8, 8);

    // relu backward on dec_act1
    launch_relu_backward(net.d_grad_dec_act1, net.d_dec_act1, net.d_grad_dec_act1, 128*8*8);

    // dec1 conv (latent -> dec_act1)
    launch_conv2d_weight_grad_naive(net.d_latent, net.d_grad_dec_act1, net.d_g_w_dec1, net.d_g_b_dec1, 128, 8, 8, 128, GPUAutoencoder::KERNEL);
    launch_conv2d_input_grad_naive(net.d_grad_dec_act1, net.d_w_dec1, net.d_grad_latent, 128, 8, 8, 128, GPUAutoencoder::KERNEL);

    // Now encoder side: unpool pool2 (latent gradient ->grad_act2)
    // net.d_grad_latent is grad at latent (128,8,8), need to propagate to act2 (128,16,16)
    // Use maxpool2x2_backward with indices net.d_pool2_idx -> result add into net.d_grad_act2
    CHECK(cudaMemset(net.d_grad_act2, 0, (size_t)128*16*16*sizeof(float)));
    launch_maxpool2x2_backward(net.d_grad_latent, net.d_grad_act2, net.d_pool2_idx, 128, 16, 16);

    // relu backward for act2 (using net.d_act2)
    launch_relu_backward(net.d_grad_act2, net.d_act2, net.d_grad_act2, 128*16*16);

    // conv2 (encoder) weight grad: input = net.d_pool1 (256,16,16), grad_out = net.d_grad_act2
    launch_conv2d_weight_grad_naive(net.d_pool1, net.d_grad_act2, net.d_g_w_conv2, net.d_g_b_conv2, 256, 16, 16, 128, GPUAutoencoder::KERNEL);
    // conv2 input grad -> net.d_grad_pool1
    launch_conv2d_input_grad_naive(net.d_grad_act2, net.d_w_conv2, net.d_grad_pool1, 256, 16, 16, 128, GPUAutoencoder::KERNEL);

    // unpool pool1 -> grad_act1 (256,32,32)
    CHECK(cudaMemset(net.d_grad_act1, 0, (size_t)256*32*32*sizeof(float)));
    launch_maxpool2x2_backward(net.d_grad_pool1, net.d_grad_act1, net.d_pool1_idx, 256, 32, 32);

    // relu backward for act1
    launch_relu_backward(net.d_grad_act1, net.d_act1, net.d_grad_act1, 256*32*32);

    // conv1 weight grad: input = net.d_input, grad_out = net.d_grad_act1
    launch_conv2d_weight_grad_naive(net.d_input, net.d_grad_act1, net.d_g_w_conv1, net.d_g_b_conv1, 3, 32, 32, 256, GPUAutoencoder::KERNEL);

    // conv1 input grad not needed for autoencoder training (unless you want to propagate further)
    // Done: gradients accumulated into device grad buffers d_g_*
}

// sgd_update_on_device: update all weights & biases on device in-place using grads and zero grads
void sgd_update_on_device(GPUAutoencoder& net, int batch_size, float lr) {
    // conv1 w + b
    launch_update_weights_on_device(net.d_w_conv1, net.d_g_w_conv1, (int)net.h_w_conv1.size(), lr, batch_size);
    launch_update_bias_on_device(net.d_b_conv1, net.d_g_b_conv1, (int)net.h_b_conv1.size(), lr, batch_size);

    // conv2
    launch_update_weights_on_device(net.d_w_conv2, net.d_g_w_conv2, (int)net.h_w_conv2.size(), lr, batch_size);
    launch_update_bias_on_device(net.d_b_conv2, net.d_g_b_conv2, (int)net.h_b_conv2.size(), lr, batch_size);

    // dec1
    launch_update_weights_on_device(net.d_w_dec1, net.d_g_w_dec1, (int)net.h_w_dec1.size(), lr, batch_size);
    launch_update_bias_on_device(net.d_b_dec1, net.d_g_b_dec1, (int)net.h_b_dec1.size(), lr, batch_size);

    // dec2
    launch_update_weights_on_device(net.d_w_dec2, net.d_g_w_dec2, (int)net.h_w_dec2.size(), lr, batch_size);
    launch_update_bias_on_device(net.d_b_dec2, net.d_g_b_dec2, (int)net.h_b_dec2.size(), lr, batch_size);

    // dec3
    launch_update_weights_on_device(net.d_w_dec3, net.d_g_w_dec3, (int)net.h_w_dec3.size(), lr, batch_size);
    launch_update_bias_on_device(net.d_b_dec3, net.d_g_b_dec3, (int)net.h_b_dec3.size(), lr, batch_size);

    // ensure kernel finishes before returning
    CHECK(cudaDeviceSynchronize());
}
