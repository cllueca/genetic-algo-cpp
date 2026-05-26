#pragma once

#include <vector>

namespace neural_network::activations {

// ─────────────────────────────────────────────
// Activation functions
// ─────────────────────────────────────────────

enum class Activation {
    Linear,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    Swish,
    Softmax,
};

template<typename T>
T activation_fn(Activation act, T x);

template<typename T>
void softmax_inplace(std::vector<T>& v);

} // namespace neural_network::activations

#include "activations.tpp"