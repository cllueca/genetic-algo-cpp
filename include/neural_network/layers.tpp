#pragma once

#include <neural_network/layers.hpp>

#include <iostream>
#include <random>

namespace neural_network::layers {

// ─────────────────────────────────────────────
// Random helpers
// ─────────────────────────────────────────────

template<typename T>
T random_value(){
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<T> dist(T(-1), T(1));
    return static_cast<T>(dist(gen));
}

// ─────────────────────────────────────────────
// HiddenLayer
// ─────────────────────────────────────────────

template<typename T>
HiddenLayer<T>::HiddenLayer(size_t r, size_t c, activations::Activation act)
    : rows(r), cols(c), data(r * c), bias(c), activation(act)
{
    randomize();
}

template<typename T>
inline T& HiddenLayer<T>::operator()(size_t r, size_t c) {
    return data[r * cols + c];
}

template<typename T>
inline const T& HiddenLayer<T>::operator()(size_t r, size_t c) const {
    return data[r * cols + c];
}

template<typename T>
void HiddenLayer<T>::randomize() {
    for (auto& x : data) x = random_value<T>();
    for (auto& x : bias) x = random_value<T>();
}

template<typename T>
void HiddenLayer<T>::apply_activation(std::vector<T>& v) const {
    if (activation == activations::Activation::Softmax) {
        activations::softmax_inplace(v);
    } else {
        for (auto& x : v) x = activation_fn(activation, x);
    }
}

template<typename T>
void HiddenLayer<T>::print_memory_usage() const {
    std::cout << "Type size: " << sizeof(T) << " bytes\n";
    std::cout << "Matrix memory: " << data.size() * sizeof(T) << " bytes\n";
    std::cout << "Bias memory:   " << bias.size() * sizeof(T) << " bytes\n";
}

template<typename T>
void HiddenLayer<T>::show_weights() const {
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c)
            std::cout << (*this)(r, c) << ' ';
        std::cout << '\n';
    }
    std::cout << "bias: ";
    for (auto& b : bias) std::cout << b << ' ';
    std::cout << "\n\n";
}

} // namespace neural_network