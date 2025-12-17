#ifndef SVM_WRAPPER_H
#define SVM_WRAPPER_H

#include <vector>
#include <string>
#include <memory>

class SVMClassifier {
public:
    SVMClassifier();
    ~SVMClassifier();

    void set_parameters(double C, double gamma);
    void train(const std::vector<std::vector<float>>& features, const std::vector<int>& labels);
    
    // --- MỚI: DỰ ĐOÁN THEO LÔ (SONG SONG) ---
    std::vector<double> predict_batch(const std::vector<std::vector<float>>& features);

    void save_model(const std::string& filename);
    bool load_model(const std::string& filename);
    double evaluate(const std::vector<std::vector<float>>& features, const std::vector<int>& labels);

private:
    std::shared_ptr<void> model; 
    float C_param;
    float gamma_param;
};

#endif