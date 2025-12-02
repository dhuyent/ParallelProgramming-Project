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
    ~Autoencoder(); // Cleanup memory

    Tensor forward(Tensor x);
    void backward(Tensor grad, float lr);
    
    // Trích xuất đặc trưng (cho Phase 4 SVM) 
    vector<float> extract_features(Tensor x);
    
    // Lưu trọng số
    void save_weights(const string& filepath);
};