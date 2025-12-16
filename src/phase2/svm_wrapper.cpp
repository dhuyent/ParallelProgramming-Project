#include "svm_wrapper.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

// Include header
#include <thundersvm/model/svc.h> 
#include <thundersvm/dataset.h>
#include <thundersvm/util/metric.h>

SVMClassifier::SVMClassifier() {
    auto svc_ptr = std::make_shared<SVC>();
    model = std::static_pointer_cast<void>(svc_ptr);
    C_param = 10.0f;
    gamma_param = 0.0f;
}

SVMClassifier::~SVMClassifier() {}

std::shared_ptr<SVC> get_model(std::shared_ptr<void> model_void) {
    return std::static_pointer_cast<SVC>(model_void);
}

void SVMClassifier::set_parameters(double C, double gamma) {
    C_param = (float)C;
    gamma_param = (float)gamma;
}

void SVMClassifier::train(const std::vector<std::vector<float>>& features, const std::vector<int>& labels) {
    if (features.empty()) return;

    int n_samples = features.size();
    int n_features = features[0].size();

    std::vector<float> dense_data;
    dense_data.reserve(n_samples * n_features);
    for(const auto& row : features) {
        dense_data.insert(dense_data.end(), row.begin(), row.end());
    }
    
    std::vector<float> float_labels(n_samples);
    for(int i=0; i<n_samples; ++i) float_labels[i] = (float)labels[i];

    DataSet dataset;
    dataset.load_from_dense(n_samples, n_features, dense_data.data(), float_labels.data());

    SvmParam param_struct;
    param_struct.svm_type = SvmParam::C_SVC; 
    param_struct.kernel_type = SvmParam::RBF; 
    param_struct.C = C_param;
    if (gamma_param > 0) param_struct.gamma = gamma_param;
    else param_struct.gamma = 1.0 / n_features; 
    
    auto svc = get_model(model);
    std::cout << "[ThunderSVM] Training on GPU (Batch Mode)...\n";
    svc->train(dataset, param_struct);
    std::cout << "[ThunderSVM] Training finished.\n";
}

std::vector<double> SVMClassifier::predict_batch(const std::vector<std::vector<float>>& features) {
    if(features.empty()) return {};
    
    int n_samples = features.size();
    int n_features = features[0].size();

    // 1. Chuyển đổi dữ liệu sang dạng phẳng (Flatten)
    std::vector<float> dense_data;
    dense_data.reserve(n_samples * n_features);
    for(const auto& row : features) {
        dense_data.insert(dense_data.end(), row.begin(), row.end());
    }

    // 2. Tạo Dataset (Labels giả vì ta đang predict)
    std::vector<float> dummy_labels(n_samples, 0.0f);
    DataSet dataset;
    dataset.load_from_dense(n_samples, n_features, dense_data.data(), dummy_labels.data());

    // 3. Dự đoán toàn bộ một lần (Batch Size lớn để tận dụng GPU)
    auto svc = get_model(model);
    return svc->predict(dataset.instances(), 1000); // Batch size 1000
}

void SVMClassifier::save_model(const std::string& filename) {
    auto svc = get_model(model);
    svc->save_to_file(filename);
    std::cout << "[ThunderSVM] Model saved to " << filename << "\n";
}

bool SVMClassifier::load_model(const std::string& filename) {
    try {
        auto svc = get_model(model);
        svc->load_from_file(filename);
        return true;
    } catch (...) {
        return false;
    }
}

double SVMClassifier::evaluate(const std::vector<std::vector<float>>& features, const std::vector<int>& labels) {
    // Hàm này đã dùng Batch rồi, rất tốt
    std::vector<double> preds = predict_batch(features);
    int correct = 0;
    for(size_t i=0; i<preds.size(); ++i) {
        if((int)preds[i] == labels[i]) correct++;
    }
    return (double)correct / preds.size() * 100.0;
}