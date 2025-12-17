#include "autoencoder.h"
#include <fstream>
#include <iostream>

Autoencoder::Autoencoder() {
    // Encoder
    c1 = new Conv2D(3, 256, 3, 1, 1); 
    r1 = new ReLU();
    p1 = new MaxPool2D(2, 2);         

    c2 = new Conv2D(256, 128, 3, 1, 1); 
    r2 = new ReLU();
    p2 = new MaxPool2D(2, 2);         

    // Decoder 
    c3 = new Conv2D(128, 128, 3, 1, 1); 
    r3 = new ReLU();
    u1 = new UpSample2D(2);           

    c4 = new Conv2D(128, 256, 3, 1, 1); 
    r4 = new ReLU();
    u2 = new UpSample2D(2);           

    c5 = new Conv2D(256, 3, 3, 1, 1); 
}

Autoencoder::~Autoencoder() {
    delete c1; delete r1; delete p1;
    delete c2; delete r2; delete p2;
    delete c3; delete r3; delete u1;
    delete c4; delete r4; delete u2;
    delete c5;
}

Tensor Autoencoder::forward(Tensor x) {
    x = p1->forward(r1->forward(c1->forward(x)));
    x = p2->forward(r2->forward(c2->forward(x)));
    x = u1->forward(r3->forward(c3->forward(x)));
    x = u2->forward(r4->forward(c4->forward(x)));
    x = c5->forward(x);
    return x;
}

void Autoencoder::backward(Tensor grad) {
    grad = c5->backward(grad);
    grad = c4->backward(r4->backward(u2->backward(grad)));
    grad = c3->backward(r3->backward(u1->backward(grad)));
    grad = c2->backward(r2->backward(p2->backward(grad)));
    grad = c1->backward(r1->backward(p1->backward(grad)));
}

// Update và clear grad
void Autoencoder::update(float lr) {
    c1->update_weights(lr); c1->clear_grads();
    c2->update_weights(lr); c2->clear_grads();
    c3->update_weights(lr); c3->clear_grads();
    c4->update_weights(lr); c4->clear_grads();
    c5->update_weights(lr); c5->clear_grads();
}

vector<float> Autoencoder::extract_features(Tensor x) {
    x = p1->forward(r1->forward(c1->forward(x)));
    x = p2->forward(r2->forward(c2->forward(x)));
    return x.data; 
}

// Lưu weights
void Autoencoder::save_weights(const string& filepath) {
    ofstream out(filepath, ios::binary);
    if (!out.is_open()) return;

    auto save_conv = [&](Conv2D* l) {
        out.write((char*)l->weights.data(), l->weights.size()*sizeof(float));
        out.write((char*)l->biases.data(), l->biases.size()*sizeof(float));
    };
    save_conv(c1); save_conv(c2); save_conv(c3); save_conv(c4); save_conv(c5);
    out.close();
    cout << "Model saved to " << filepath << endl;
}

// Load weights
void Autoencoder::load_weights(const string& filepath) {
    ifstream in(filepath, ios::binary);
    if (!in.is_open()) { cerr << "Err loading " << filepath << endl; return; }

    auto load_conv = [&](Conv2D* l) {
        in.read((char*)l->weights.data(), l->weights.size()*sizeof(float));
        in.read((char*)l->biases.data(), l->biases.size()*sizeof(float));
    };
    load_conv(c1); load_conv(c2); load_conv(c3); load_conv(c4); load_conv(c5);
    in.close();
    cout << "Model loaded from " << filepath << endl;
}