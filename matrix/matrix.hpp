#pragma once
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include "accelerate.hpp"

class Matrix{
    friend class HDRModel;
    private:
        std::vector<float> _data;
        int _row;
        int _col;
        void _isValidPlace(std::vector<int> coordinate) const {
            if (coordinate.size() != 2 || coordinate[0] < 0 || coordinate[1] < 0 || coordinate[0] >= _row || coordinate[1] >= _col) {
                throw std::invalid_argument("Not a valid coordinate");
            }
        }
    public:
        // Generator
        Matrix () : _row(1), _col(1), _data(1) {};
        Matrix (int r, int c) : _row(r), _col(c) {
            if (r <= 0 || c <= 0){
                throw std::invalid_argument("Not a valid shape");
            }
            _data.resize(r * c);
        };
        Matrix (const Matrix& target){
            _row = target.shape()[0];
            _col = target.shape()[1];
            _data = target._data;
        }

        // Prop
        std::vector<int> shape() const {
            std::vector<int> sp = {_row, _col};
            return sp;
        }
        void show() const {
            for (int r = 0; r < _row; r++){
                for (int c = 0; c < _col; c++){
                    std::cout << _data[r * _col + c];
                    if (c != _col - 1) std::cout << ", ";
                }
                std::cout << "\n";
            }
        }

        // Flatten
        void flatten() {_row *= _col; _col = 1;}

        // Reshape
        void reshape(const std::vector<int>& new_shape){
            if (new_shape[0] * new_shape[1] != _row * _col){
                throw std::invalid_argument("Dimension Error");
            }
            _row = new_shape[0]; _col = new_shape[1];
        }
        
        // R/W
        void set(const std::vector<int>& coordinate, const float& new_value) {
            _isValidPlace(coordinate);
            _data[coordinate[0] * _col + coordinate[1]] = new_value;
        }
        float get(const std::vector<int>& coordinate){
            _isValidPlace(coordinate);
            return _data[coordinate[0] * _col + coordinate[1]];
        }
        void set_from_vector(const std::vector<float>& src){
            if (src.size() != _col * _row){
                throw std::invalid_argument("Dimension Error");
            }
            for (int i = 0; i < src.size(); i++) _data[i] = src[i];
        }

        // Operators
        void operator=(const Matrix& other) {
            _data = other._data;
            _row = other._row; _col = other._col;
        }

        Matrix operator+(const Matrix& other) const {
            if (shape() != other.shape()){
                throw std::invalid_argument("Dimension Error");
            }
            Matrix n_matrix(*this);
            vec_add(_data.data(), other._data.data(), n_matrix._data.data(), _row * _col, 8, true);
            return n_matrix;
        }

        Matrix operator-(const Matrix& other) const {
            if (shape() != other.shape()){
                throw std::invalid_argument("Dimension Error");
            }
            Matrix n_matrix(*this);
            vec_add(_data.data(), other._data.data(), n_matrix._data.data(), _row * _col, 8, false);
            return n_matrix;
        }

        Matrix operator*(const Matrix& other) const {
            if (_col != other._row){
                throw std::invalid_argument("Dimension Error");
            }
            Matrix n_matrix(_row, other._col);
            std::fill(n_matrix._data.begin(), n_matrix._data.end(), 0.0f);
            matmul_avx2(_data.data(), other._data.data(), n_matrix._data.data(), _row, _col, other._col);
            return n_matrix;
        }
};