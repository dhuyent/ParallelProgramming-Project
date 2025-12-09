#ifndef SVM_WRAPPER_H
#define SVM_WRAPPER_H

#include <vector>
#include <string>
#include "svm.h" // Include thư viện LIBSVM

class SVMClassifier {
public:
    SVMClassifier();
    ~SVMClassifier();

    // Cấu hình tham số (C, Gamma)
    void set_parameters(double C, double gamma);

    // Huấn luyện model
    // features: Mảng 2 chiều [num_samples][num_features]
    // labels: Mảng 1 chiều [num_samples]
    void train(const std::vector<std::vector<float>>& features, const std::vector<int>& labels);

    // Dự đoán 1 mẫu
    double predict(const std::vector<float>& feature);

    // Lưu/Load model
    void save_model(const std::string& filename);
    bool load_model(const std::string& filename);

    // Đánh giá độ chính xác
    double evaluate(const std::vector<std::vector<float>>& features, const std::vector<int>& labels);

private:
    struct svm_parameter param; // Tham số SVM
    struct svm_problem prob;    // Dữ liệu train
    struct svm_model* model;    // Model sau khi train
    
    // Vùng nhớ đệm để giữ dữ liệu cho svm_problem trỏ tới
    std::vector<struct svm_node*> x_space; 
    std::vector<struct svm_node> data_pool; 
};

#endif