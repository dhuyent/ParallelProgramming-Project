#pragma once
#include "../utils.h"

// Convolution
class Conv2D {
public:
    int in_c, out_c, k_size, stride, pad;
    vector<float> weights, biases;
    
    // Tích lũy gradient cho cả batch
    vector<float> accum_grad_w; 
    vector<float> accum_grad_b;
    
    Tensor input_cache;

    Conv2D(int in, int out, int k=3, int s=1, int p=1);
    void init_weights();
    
    // Xóa gradient cũ đầu mỗi batch
    void clear_grads();
    
    void update_weights(float lr); 

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_out); 
};

// ReLU
class ReLU {
public:
    Tensor input_cache;
    ReLU() : input_cache(0,0,0) {}
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_out);
};

// Max Pooling 2x2
class MaxPool2D {
public:
    int pool_size, stride;
    vector<int> max_indices;
    Tensor input_shape;
    MaxPool2D(int size=2, int s=2);
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_out);
};

// UpSample 2x2
class UpSample2D {
public:
    int scale;
    Tensor input_shape;
    UpSample2D(int s=2);
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_out);
};

// MSE Loss
float mse_loss(const Tensor& pred, const Tensor& target);
Tensor mse_loss_grad(const Tensor& pred, const Tensor& target);