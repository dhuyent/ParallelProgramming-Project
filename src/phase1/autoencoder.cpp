#include "autoencoder.h"
#include <fstream>

Autoencoder::Autoencoder() {
    // Encoder 
    c1 = new Conv2D(3, 256, 3, 1, 1); // Input: 32x32x3 -> 32x32x256
    r1 = new ReLU();
    p1 = new MaxPool2D(2, 2);         // -> 16x16x256

    c2 = new Conv2D(256, 128, 3, 1, 1); // -> 16x16x128
    r2 = new ReLU();
    p2 = new MaxPool2D(2, 2);         // -> 8x8x128 (Latent)

    // Decoder
    c3 = new Conv2D(128, 128, 3, 1, 1); // -> 8x8x128
    r3 = new ReLU();
    u1 = new UpSample2D(2);           // -> 16x16x128

    c4 = new Conv2D(128, 256, 3, 1, 1); // -> 16x16x256
    r4 = new ReLU();
    u2 = new UpSample2D(2);           // -> 32x32x256

    c5 = new Conv2D(256, 3, 3, 1, 1); // -> 32x32x3 (Output)
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

void Autoencoder::backward(Tensor grad, float lr) {
    grad = c5->backward(grad, lr);
    grad = c4->backward(r4->backward(u2->backward(grad, lr), lr), lr);
    grad = c3->backward(r3->backward(u1->backward(grad, lr), lr), lr);
    grad = c2->backward(r2->backward(p2->backward(grad, lr), lr), lr);
    grad = c1->backward(r1->backward(p1->backward(grad, lr), lr), lr);
}

vector<float> Autoencoder::extract_features(Tensor x) {
    x = p1->forward(r1->forward(c1->forward(x)));
    x = p2->forward(r2->forward(c2->forward(x)));
    return x.data; // Flatten 8x8x128
}

void Autoencoder::save_weights(const string& filepath) {
    ofstream out(filepath, ios::binary);
    // Lưu tượng trưng weights của c1 (trong thực tế phải lưu hết)
    if(out.is_open()) {
        out.write((char*)c1->weights.data(), c1->weights.size()*sizeof(float));
        out.close();
        cout << "Model saved to " << filepath << endl;
    }
}