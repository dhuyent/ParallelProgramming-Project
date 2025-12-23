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

Tensor Autoencoder::forward(const Tensor& x) {
    x_c1 = c1->forward(x);
    x_r1 = r1->forward(x_c1);
    x_p1 = p1->forward(x_r1);

    x_c2 = c2->forward(x_p1);
    x_r2 = r2->forward(x_c2);
    x_p2 = p2->forward(x_r2);

    x_c3 = c3->forward(x_p2);
    x_r3 = r3->forward(x_c3);
    x_u1 = u1->forward(x_r3);

    x_c4 = c4->forward(x_u1);
    x_r4 = r4->forward(x_c4);
    x_u2 = u2->forward(x_r4);

    Tensor out = c5->forward(x_u2);
    return out;
}

void Autoencoder::backward(const Tensor& grad) {
    Tensor g = grad;

    g = c5->backward(g);

    g = u2->backward(g);
    g = r4->backward(g);
    g = c4->backward(g);

    g = u1->backward(g);
    g = r3->backward(g);
    g = c3->backward(g);

    g = p2->backward(g);
    g = r2->backward(g);
    g = c2->backward(g);

    g = p1->backward(g);
    g = r1->backward(g);
    g = c1->backward(g);
}

void Autoencoder::update(float lr) {
    c1->update_weights(lr); 
    c2->update_weights(lr); 
    c3->update_weights(lr); 
    c4->update_weights(lr);
    c5->update_weights(lr); 
}

vector<float> Autoencoder::extract_features(const Tensor& x) {
    Tensor y = c1->forward(x);
    y = r1->forward(y);
    y = p1->forward(y);

    y = c2->forward(y);
    y = r2->forward(y);
    y = p2->forward(y);

    return y.data;
}

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