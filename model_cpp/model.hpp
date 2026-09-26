#include <fstream>
#include <string>
#include "../matrix/matrix.hpp"

void ReLU(Matrix& x){
    int r = x.shape()[0];
    for (int i = 0; i < r; i++){
        x.set({i, 0}, x.get({i, 0}) > 0 ? x.get({i, 0}) : 0);
    }
}

class HDRModel{
    public:
        std::vector<Matrix> _weights;
        std::vector<Matrix> _bias;
        HDRModel() {
            _weights.resize(4); _bias.resize(4);
            _weights[0] = Matrix(256, 784);
            _weights[1] = Matrix(128, 256);
            _weights[2] = Matrix(64, 128);
            _weights[3] = Matrix(10, 64);
            _bias[0] = Matrix(256, 1);
            _bias[1] = Matrix(128, 1);
            _bias[2] = Matrix(64, 1);
            _bias[3] = Matrix(10, 1);
        }

        void LoadWeights(int idx, std::string weight_path, std::string bias_path){
            std::ifstream fin_w(weight_path);
            std::ifstream fin_b(bias_path);

            if (!fin_w || !fin_b){
                throw std::invalid_argument("Failed to open file");
            }
            for (size_t i = 0; i < size_t(_weights[idx]._row * _weights[idx]._col) && fin_w >> _weights[idx]._data[i]; ++i);
            for (size_t i = 0; i < size_t(_bias[idx]._row * _bias[idx]._col) && fin_b >> _bias[idx]._data[i]; ++i);
            std::cout << "Weights " << idx <<  " Loading successful.\n";
        }

        Matrix forward(Matrix x) {
            x.flatten();  // (28, 28) -> (784, 1)
            for (int i = 0; i < 4; i++){
                x = _weights[i] * x + _bias[i];
                ReLU(x);
            }
            return x;
        }
};