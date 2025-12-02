#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>

using namespace std;

// Kích thước ảnh CIFAR-10
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
    
    float& at(int c, int h, int w_idx) {
        return data[c * height * width + h * width + w_idx];
    }
    const float& at(int c, int h, int w_idx) const {
        return data[c * height * width + h * width + w_idx];
    }
};

// Timer cho CPU 
struct CpuTimer {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;

    void Start() { start_time = std::chrono::high_resolution_clock::now(); }
    void Stop() { end_time = std::chrono::high_resolution_clock::now(); }

    float ElapsedSeconds() {
        return std::chrono::duration<float>(end_time - start_time).count();
    }
};