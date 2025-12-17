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

    Tensor forward(Tensor x);
    void backward(Tensor grad);
    
    // Update weights cho toàn bộ mạng
    void update(float lr);

    vector<float> extract_features(Tensor x);
    
    // Lưu/Load weights
    void save_weights(const string& filepath);
    void load_weights(const string& filepath);
};