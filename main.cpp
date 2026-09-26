#include "model_cpp/model.hpp"
#include "draw.hpp"
#include <iomanip>
#include <cmath>

Matrix softmax(Matrix& in){
    // in: (10, 1)
    float sum = 0;
    for (int i = 0; i < 10; i++) sum += expf(in.get({i, 0}));
    Matrix result(10, 1);
    for (int i = 0; i < 10; i++) result.set({i, 0}, expf(in.get({i, 0})) / sum);
    return result;
}
 
int argmax(Matrix& in){
    // in: (10, 1)
    int ans = 0;
    for (int i = 1; i < 10; i++){
        if (in.get({i, 0}) > in.get({ans, 0})) ans = i;
    }
    return ans;
}

int main(){
    HDRModel model;
    std::cout << "Enter the absolute path of the weights file: ";
    std::string path; std::cin >> path;
    for (int i = 0; i < 4; i++){
        model.LoadWeights(i, path + "/net." + std::to_string(2 * i) + ".weight.txt", path + "/net." + std::to_string(2 * i) + ".bias.txt");
    }
    Matrix img = GetImage();
    auto c = model.forward(img);
    c = softmax(c);
    // Show distribution
    std::cout << std::string(50, '-') << "\n";
    for (int i = 0; i < 10; i++){
        std::cout << i << " (~" << std::setw(2) << std::setfill(' ') << int(c.get({i, 0}) * 100) << "%): ";
        std::cout << std::string(int(c.get({i, 0}) * 100), '#');
        std::cout << "\n";
    }
    std::cout << std::string(50, '-') << "\n";
    std::cout << "Predict: " << argmax(c);
    system("pause>nul");
    return 0;
}
