#include "cpu_layers.h"
#include <cmath>
#include <algorithm>

// --- Helper Functions ---
// Truy cập mảng 1D như 4D (out, in, h, w) cho weights
inline float& w_at(vector<float>& w, int o, int i, int kh, int kw, int in_c, int k_sz) {
    return w[o * (in_c * k_sz * k_sz) + i * (k_sz * k_sz) + kh * k_sz + kw];
}

// --- Conv2D Implementation ---
Conv2D::Conv2D(int in_c, int out_c, int k, int s, int p) : in_c(in_c), out_c(out_c), k_size(k), stride(s), pad(p), input_cache(0,0,0) {
    init_weights();
}

void Conv2D::init_weights() {
    // He Initialization
    default_random_engine generator;
    normal_distribution<float> distribution(0.0, sqrt(2.0 / (in_c * k_size * k_size)));
    
    int total_weights = out_c * in_c * k_size * k_size;
    weights.resize(total_weights);
    grad_w.resize(total_weights);
    for(int i=0; i<total_weights; ++i) weights[i] = distribution(generator);

    biases.resize(out_c, 0.0f);
    grad_b.resize(out_c, 0.0f);
}

Tensor Conv2D::forward(const Tensor& input) {
    input_cache = input;
    int out_h = (input.height + 2 * pad - k_size) / stride + 1;
    int out_w = (input.width + 2 * pad - k_size) / stride + 1;
    Tensor output(out_c, out_h, out_w);

    // Naive Convolution implementation
    for (int o = 0; o < out_c; ++o) {
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                float sum = biases[o];
                int in_y_origin = y * stride - pad;
                int in_x_origin = x * stride - pad;

                for (int i = 0; i < in_c; ++i) {
                    for (int ky = 0; ky < k_size; ++ky) {
                        for (int kx = 0; kx < k_size; ++kx) {
                            int in_y = in_y_origin + ky;
                            int in_x = in_x_origin + kx;

                            if (in_y >= 0 && in_y < input.height && in_x >= 0 && in_x < input.width) {
                                float val = input.at(i, in_y, in_x);
                                float w = weights[o * (in_c * k_size * k_size) + 
                                                  i * (k_size * k_size) + 
                                                  ky * k_size + kx];
                                sum += val * w;
                            }
                        }
                    }
                }
                output.at(o, y, x) = sum;
            }
        }
    }
    return output;
}

Tensor Conv2D::backward(const Tensor& grad_output, float learning_rate) {
    Tensor grad_input(in_c, input_cache.height, input_cache.width);
    
    // Reset gradients
    fill(grad_w.begin(), grad_w.end(), 0.0f);
    fill(grad_b.begin(), grad_b.end(), 0.0f);

    int out_h = grad_output.height;
    int out_w = grad_output.width;

    // Tính gradient 
    for (int o = 0; o < out_c; ++o) {
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                float g = grad_output.at(o, y, x);
                grad_b[o] += g;

                int in_y_origin = y * stride - pad;
                int in_x_origin = x * stride - pad;

                for (int i = 0; i < in_c; ++i) {
                    for (int ky = 0; ky < k_size; ++ky) {
                        for (int kx = 0; kx < k_size; ++kx) {
                            int in_y = in_y_origin + ky;
                            int in_x = in_x_origin + kx;

                            if (in_y >= 0 && in_y < input_cache.height && in_x >= 0 && in_x < input_cache.width) {
                                // dL/dw = input * grad_output
                                int w_idx = o * (in_c * k_size * k_size) + 
                                            i * (k_size * k_size) + 
                                            ky * k_size + kx;
                                grad_w[w_idx] += input_cache.at(i, in_y, in_x) * g;

                                // dL/dx = weight * grad_output (Accumulate to input grad)
                                grad_input.at(i, in_y, in_x) += weights[w_idx] * g;
                            }
                        }
                    }
                }
            }
        }
    }

    // Update weights (SGD) 
    for (size_t i = 0; i < weights.size(); ++i) weights[i] -= learning_rate * grad_w[i];
    for (size_t i = 0; i < biases.size(); ++i) biases[i] -= learning_rate * grad_b[i];

    return grad_input;
}

// --- ReLU Implementation --- 
Tensor ReLU::forward(const Tensor& input) {
    input_cache = input;
    Tensor output = input;
    for (float& val : output.data) {
        val = max(0.0f, val);
    }
    return output;
}

Tensor ReLU::backward(const Tensor& grad_output, float learning_rate) {
    Tensor grad_input = grad_output;
    for (size_t i = 0; i < grad_input.data.size(); ++i) {
        if (input_cache.data[i] <= 0) {
            grad_input.data[i] = 0.0f;
        }
    }
    return grad_input;
}

// --- MaxPool Implementation --- 
MaxPool2D::MaxPool2D(int size, int s) : pool_size(size), stride(s), input_shape(0,0,0) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    input_shape = Tensor(input.channels, input.height, input.width); // Chỉ lưu shape
    int out_h = (input.height - pool_size) / stride + 1;
    int out_w = (input.width - pool_size) / stride + 1;
    Tensor output(input.channels, out_h, out_w);
    max_indices.resize(output.data.size()); // Lưu index để backward

    for (int c = 0; c < input.channels; ++c) {
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                float max_val = -1e9;
                int max_idx = -1;
                
                int h_start = y * stride;
                int w_start = x * stride;

                for (int kh = 0; kh < pool_size; ++kh) {
                    for (int kw = 0; kw < pool_size; ++kw) {
                        int cur_idx = c * input.height * input.width + (h_start + kh) * input.width + (w_start + kw);
                        if (input.data[cur_idx] > max_val) {
                            max_val = input.data[cur_idx];
                            max_idx = cur_idx;
                        }
                    }
                }
                output.at(c, y, x) = max_val;
                max_indices[c * out_h * out_w + y * out_w + x] = max_idx;
            }
        }
    }
    return output;
}

Tensor MaxPool2D::backward(const Tensor& grad_output, float learning_rate) {
    Tensor grad_input(input_shape.channels, input_shape.height, input_shape.width);
    // Fill 0
    fill(grad_input.data.begin(), grad_input.data.end(), 0.0f);

    for (size_t i = 0; i < max_indices.size(); ++i) {
        int idx = max_indices[i];
        grad_input.data[idx] += grad_output.data[i];
    }
    return grad_input;
}

// --- Upsample Implementation ---
UpSample2D::UpSample2D(int scale) : scale(scale), input_shape(0,0,0) {}

Tensor UpSample2D::forward(const Tensor& input) {
    input_shape = Tensor(input.channels, input.height, input.width);
    Tensor output(input.channels, input.height * scale, input.width * scale);

    for (int c = 0; c < output.channels; ++c) {
        for (int y = 0; y < output.height; ++y) {
            for (int x = 0; x < output.width; ++x) {
                // Nearest neighbor: lấy giá trị từ pixel gốc
                output.at(c, y, x) = input.at(c, y / scale, x / scale);
            }
        }
    }
    return output;
}

Tensor UpSample2D::backward(const Tensor& grad_output, float learning_rate) {
    Tensor grad_input(input_shape.channels, input_shape.height, input_shape.width);
    fill(grad_input.data.begin(), grad_input.data.end(), 0.0f);

    for (int c = 0; c < grad_output.channels; ++c) {
        for (int y = 0; y < grad_output.height; ++y) {
            for (int x = 0; x < grad_output.width; ++x) {
                // Cộng dồn gradient về pixel gốc
                grad_input.at(c, y / scale, x / scale) += grad_output.at(c, y, x);
            }
        }
    }
    return grad_input;
}

// --- Loss Functions --- 
float mse_loss(const Tensor& pred, const Tensor& target) {
    float sum = 0.0f;
    for (size_t i = 0; i < pred.data.size(); ++i) {
        float diff = pred.data[i] - target.data[i];
        sum += diff * diff;
    }
    return sum / pred.data.size();
}

Tensor mse_loss_grad(const Tensor& pred, const Tensor& target) {
    Tensor grad(pred.channels, pred.height, pred.width);
    float n = (float)pred.data.size();
    for (size_t i = 0; i < pred.data.size(); ++i) {
        grad.data[i] = 2.0f * (pred.data[i] - target.data[i]) / n;
    }
    return grad;
}