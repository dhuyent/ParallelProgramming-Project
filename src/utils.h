#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <random>

using namespace std;

// Kích thước ảnh CIFAR-10 [cite: 31]
const int IMG_WIDTH = 32;
const int IMG_HEIGHT = 32;
const int IMG_CHANNELS = 3;

// Cấu trúc Tensor 3D (Channels, Height, Width)
struct Tensor {
    vector<float> data;
    int channels;
    int height;
    int width;

    Tensor(int c, int h, int w) : channels(c), height(h), width(w) {
        data.resize(c * h * w, 0.0f);
    }
    
    // Helper truy cập an toàn
    float& at(int c, int h, int w_idx) {
        return data[c * height * width + h * width + w_idx];
    }
    const float& at(int c, int h, int w_idx) const {
        return data[c * height * width + h * width + w_idx];
    }
};