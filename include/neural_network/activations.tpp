#pragma once

#include <neural_network/activations.hpp>

#include <algorithm>
#include <cmath>

namespace neural_network::activations {

template<typename T>
T activation_fn(Activation act, T x) {
    switch (act) {
        case Activation::Linear:    return x;
        case Activation::ReLU:      return x > T(0) ? x : T(0);
        case Activation::LeakyReLU: return x > T(0) ? x : T(0.01) * x;
        case Activation::Sigmoid:   return T(1) / (T(1) + std::exp(-x));
        case Activation::Tanh:      return std::tanh(x);
        case Activation::Swish:     return x / (T(1) + std::exp(-x));
        case Activation::Softmax:   return x;   // Handled at the vector level; fall back to identity here.
    }
    return x;
}

template<typename T>
void softmax_inplace(std::vector<T>& v) {
    T max_val = *std::max_element(v.begin(), v.end());  // numerical stability
    T sum = T(0);
    for (auto& x : v) { x = std::exp(x - max_val); sum += x; }
    for (auto& x : v) { x /= sum; }
}

} // namespace neural_network::activations