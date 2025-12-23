#include "cpu_layers.h"
#include <algorithm>
#include <cmath>
#include<random>

// Conv2D 
Conv2D::Conv2D(int in_c, int out_c, int k, int s, int p) : in_c(in_c), out_c(out_c), k_size(k), stride(s), pad(p), input_cache_ptr(nullptr) {
    init_weights();
    accum_grad_w.assign(weights.size(), 0.0f);
    accum_grad_b.assign(biases.size(), 0.0f);
}

void Conv2D::init_weights() {
    default_random_engine gen(42);
    normal_distribution<float> dist(0.0f, sqrt(2.0f / (float)(in_c * k_size * k_size)));

    int total = out_c * in_c * k_size * k_size;
    weights.resize(total);
    biases.resize(out_c);

    for (int i = 0; i < total; ++i) weights[i] = dist(gen);
    fill(biases.begin(), biases.end(), 0.0f);
}

Tensor Conv2D::forward(const Tensor& input) {
    input_cache_ptr = &input;

    const int in_h = input.height;
    const int in_w = input.width;

    const int out_h = (in_h + 2 * pad - k_size) / stride + 1;
    const int out_w = (in_w + 2 * pad - k_size) / stride + 1;

    Tensor output(out_c, out_h, out_w);

    const int inHW  = in_h * in_w;
    const int outHW = out_h * out_w;
    const int kHW   = k_size * k_size;

    const float* inData = input.data.data();
    float* outData      = output.data.data();

    for (int oc = 0; oc < out_c; ++oc) {
        const int w_base_oc = oc * (in_c * kHW);
        const float b = biases[oc];

        for (int oy = 0; oy < out_h; ++oy) {
            const int in_y_origin  = oy * stride - pad;
            const int out_row_base = oc * outHW + oy * out_w;

            for (int ox = 0; ox < out_w; ++ox) {
                const int in_x_origin = ox * stride - pad;
                float sum = b;

                for (int ic = 0; ic < in_c; ++ic) {
                    const int w_base_ic  = w_base_oc + ic * kHW;
                    const int in_base_ic = ic * inHW;

                    for (int ky = 0; ky < k_size; ++ky) {
                        const int in_y = in_y_origin + ky;
                        if ((unsigned)in_y >= (unsigned)in_h) continue;

                        const int in_row_base = in_base_ic + in_y * in_w;
                        const int w_row_base  = w_base_ic  + ky * k_size;

                        for (int kx = 0; kx < k_size; ++kx) {
                            const int in_x = in_x_origin + kx;
                            if ((unsigned)in_x >= (unsigned)in_w) continue;

                            sum += inData[in_row_base + in_x] * weights[w_row_base + kx];
                        }
                    }
                }

                outData[out_row_base + ox] = sum;
            }
        }
    }
    return output;
}

Tensor Conv2D::backward(const Tensor& grad_out) {
    const Tensor& input = *input_cache_ptr;

    const int in_h  = input.height;
    const int in_w  = input.width;
    const int out_h = grad_out.height;
    const int out_w = grad_out.width;

    Tensor grad_in(in_c, in_h, in_w);
    fill(grad_in.data.begin(), grad_in.data.end(), 0.0f);

    const int inHW  = in_h * in_w;
    const int outHW = out_h * out_w;
    const int kHW   = k_size * k_size;

    const float* inData = input.data.data();
    const float* gOut   = grad_out.data.data();
    float* gIn          = grad_in.data.data();

    for (int oc = 0; oc < out_c; ++oc) {
        const int w_base_oc     = oc * (in_c * kHW);
        const int g_out_base_oc = oc * outHW;

        for (int oy = 0; oy < out_h; ++oy) {
            const int in_y_origin    = oy * stride - pad;
            const int g_out_row_base = g_out_base_oc + oy * out_w;

            for (int ox = 0; ox < out_w; ++ox) {
                const float g = gOut[g_out_row_base + ox];
                accum_grad_b[oc] += g;

                const int in_x_origin = ox * stride - pad;

                for (int ic = 0; ic < in_c; ++ic) {
                    const int w_base_ic  = w_base_oc + ic * kHW;
                    const int in_base_ic = ic * inHW;

                    for (int ky = 0; ky < k_size; ++ky) {
                        const int in_y = in_y_origin + ky;
                        if ((unsigned)in_y >= (unsigned)in_h) continue;

                        const int in_row_base = in_base_ic + in_y * in_w;
                        const int w_row_base  = w_base_ic  + ky * k_size;

                        for (int kx = 0; kx < k_size; ++kx) {
                            const int in_x = in_x_origin + kx;
                            if ((unsigned)in_x >= (unsigned)in_w) continue;

                            const int in_idx = in_row_base + in_x;
                            const int w_idx  = w_row_base + kx;

                            accum_grad_w[w_idx] += inData[in_idx] * g;
                            gIn[in_idx]          += weights[w_idx] * g;
                        }
                    }
                }
            }
        }
    }
    return grad_in;
}

void Conv2D::update_weights(float lr) {
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= lr * accum_grad_w[i];
        accum_grad_w[i] = 0.0f;
    }
    for (size_t i = 0; i < biases.size(); ++i) {
        biases[i] -= lr * accum_grad_b[i];
        accum_grad_b[i] = 0.0f;
    }
}

// ReLU
Tensor ReLU::forward(const Tensor& input) {
    input_cache = input;
    Tensor out = input;
    for (float& v : out.data) 
        v = max(0.0f, v);
    return out;
}

Tensor ReLU::backward(const Tensor& grad_out) {
    Tensor grad_in = grad_out;
    for (size_t i = 0; i < grad_in.data.size(); ++i) {
        if (input_cache.data[i] <= 0.0f) 
            grad_in.data[i] = 0.0f;
    }
    return grad_in;
}

// Max Pooling 2x2
MaxPool2D::MaxPool2D(int size, int s) : pool_size(size), stride(s), input_shape(0,0,0) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    input_shape = Tensor(input.channels, input.height, input.width);

    const int C = input.channels;
    const int H = input.height;
    const int W = input.width;

    const int out_h = (H - pool_size) / stride + 1;
    const int out_w = (W - pool_size) / stride + 1;

    Tensor output(C, out_h, out_w);
    max_indices.assign((size_t)C * out_h * out_w, -1);

    const float* inData = input.data.data();
    float* outData      = output.data.data();

    const int inHW  = H * W;
    const int outHW = out_h * out_w;

    for (int c = 0; c < C; ++c) {
        const int in_base_c  = c * inHW;
        const int out_base_c = c * outHW;

        for (int oy = 0; oy < out_h; ++oy) {
            const int h_start = oy * stride;

            for (int ox = 0; ox < out_w; ++ox) {
                const int w_start = ox * stride;

                float max_val = -1e30f;
                int max_idx   = -1;

                for (int ky = 0; ky < pool_size; ++ky) {
                    const int iy = h_start + ky;
                    const int in_row = in_base_c + iy * W;

                    for (int kx = 0; kx < pool_size; ++kx) {
                        const int ix = w_start + kx;
                        const int idx = in_row + ix;
                        const float v = inData[idx];
                        if (v > max_val) { max_val = v; max_idx = idx; }
                    }
                }

                const int out_idx = out_base_c + oy * out_w + ox;
                outData[out_idx] = max_val;
                max_indices[out_idx] = max_idx;
            }
        }
    }
    return output;
}

Tensor MaxPool2D::backward(const Tensor& grad_out) {
    const int C = input_shape.channels;
    const int H = input_shape.height;
    const int W = input_shape.width;

    Tensor grad_in(C, H, W);
    fill(grad_in.data.begin(), grad_in.data.end(), 0.0f);

    for (size_t i = 0; i < max_indices.size(); ++i) {
        int idx = max_indices[i];
        if (idx >= 0) grad_in.data[(size_t)idx] += grad_out.data[i];
    }
    return grad_in;
}
// UpSample 2x2
UpSample2D::UpSample2D(int s) : scale(s), input_shape(0,0,0) {}

Tensor UpSample2D::forward(const Tensor& input) {
    input_shape = Tensor(input.channels, input.height, input.width);

    const int C = input.channels;
    const int H = input.height;
    const int W = input.width;

    const int out_h = H * scale;
    const int out_w = W * scale;

    Tensor output(C, out_h, out_w);

    const float* inData = input.data.data();
    float* outData      = output.data.data();

    const int inHW  = H * W;
    const int outHW = out_h * out_w;

    for (int c = 0; c < C; ++c) {
        const int in_base_c  = c * inHW;
        const int out_base_c = c * outHW;

        for (int iy = 0; iy < H; ++iy) {
            for (int ix = 0; ix < W; ++ix) {
                const float v = inData[in_base_c + iy * W + ix];
                const int oy0 = iy * scale;
                const int ox0 = ix * scale;

                for (int dy = 0; dy < scale; ++dy) {
                    const int out_row = out_base_c + (oy0 + dy) * out_w + ox0;
                    for (int dx = 0; dx < scale; ++dx) outData[out_row + dx] = v;
                }
            }
        }
    }
    return output;
}

Tensor UpSample2D::backward(const Tensor& grad_out) {
    const int C = input_shape.channels;
    const int H = input_shape.height;
    const int W = input_shape.width;

    Tensor grad_in(C, H, W);
    fill(grad_in.data.begin(), grad_in.data.end(), 0.0f);

    const float* gOut = grad_out.data.data();

    const int out_h = grad_out.height;
    const int out_w = grad_out.width;

    const int inHW  = H * W;
    const int outHW = out_h * out_w;

    for (int c = 0; c < C; ++c) {
        const int in_base_c  = c * inHW;
        const int out_base_c = c * outHW;

        for (int iy = 0; iy < H; ++iy) {
            for (int ix = 0; ix < W; ++ix) {
                const int oy0 = iy * scale;
                const int ox0 = ix * scale;

                float sum = 0.0f;
                for (int dy = 0; dy < scale; ++dy) {
                    const int out_row = out_base_c + (oy0 + dy) * out_w + ox0;
                    for (int dx = 0; dx < scale; ++dx) sum += gOut[out_row + dx];
                }

                grad_in.data[in_base_c + iy * W + ix] += sum;
            }
        }
    }
    return grad_in;
}

// MSE Loss
float mse_loss(const Tensor& pred, const Tensor& target) {
    float sum = 0.0f;
    for (size_t i = 0; i < pred.data.size(); ++i) {
        const float d = pred.data[i] - target.data[i];
        sum += d * d;
    }
    return sum / (float)pred.data.size();
}

Tensor mse_loss_grad(const Tensor& pred, const Tensor& target) {
    Tensor grad = pred;
    const float scale = 2.0f / (float)pred.data.size();

    for (size_t i = 0; i < grad.data.size(); ++i) {
        grad.data[i] = scale * (pred.data[i] - target.data[i]);
    }
    return grad;
}