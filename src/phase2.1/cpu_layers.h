#include "../utils.h"

// 1. Convolution: 3x3 kernel, padding, stride [cite: 248]
class Conv2D {
public:
    int in_c, out_c, k_size, stride, pad;
    vector<float> weights, biases;
    vector<float> grad_w, grad_b;
    Tensor input_cache;

    Conv2D(int in, int out, int k=3, int s=1, int p=1);
    void init_weights();
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_out, float lr);
};

// 2. ReLU: max(0, x) [cite: 249]
class ReLU {
public:
    Tensor input_cache;
    ReLU() : input_cache(0,0,0) {}
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_out, float lr);
};

// 3. MaxPool: 2x2 [cite: 250]
class MaxPool2D {
public:
    int pool_size, stride;
    vector<int> max_indices; // backprop
    Tensor input_shape;

    MaxPool2D(int size=2, int s=2);
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_out, float lr);
};

// 4. Upsample: Nearest Neighbor 
class UpSample2D {
public:
    int scale;
    Tensor input_shape;

    UpSample2D(int s=2);
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_out, float lr);
};

// Loss Function
float mse_loss(const Tensor& pred, const Tensor& target);
Tensor mse_loss_grad(const Tensor& pred, const Tensor& target);