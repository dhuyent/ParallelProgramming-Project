#pragma once
#include "cpu_layers.h"

class Autoencoder {
public:
    // Layers
    Conv2D *c1, *c2, *c3, *c4, *c5;
    ReLU *r1, *r2, *r3, *r4;
    MaxPool2D *p1, *p2;
    UpSample2D *u1, *u2;

    Autoencoder();
    ~Autoencoder();

    Tensor forward(const Tensor& x);
    void backward(const Tensor& grad);
    void update(float lr);

    vector<float> extract_features(const Tensor& x);
    
    void save_weights(const string& filepath);
    void load_weights(const string& filepath);

private:
    // Cache toàn bộ activation cần cho backward
    Tensor x_c1{1,1,1}, x_r1{1,1,1}, x_p1{1,1,1};
    Tensor x_c2{1,1,1}, x_r2{1,1,1}, x_p2{1,1,1};
    Tensor x_c3{1,1,1}, x_r3{1,1,1}, x_u1{1,1,1};
    Tensor x_c4{1,1,1}, x_r4{1,1,1}, x_u2{1,1,1};
};