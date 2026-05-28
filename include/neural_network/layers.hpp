#pragma once

#include <vector>
#include <cstddef>

#include "neural_network/activations.hpp"

namespace neural_network::layers {

template<typename T>
T random_value();

template<typename T>
class DenseLayer {
public:
    size_t rows, cols;
    std::vector<T> data;   // weights: shape (rows, cols), row-major
    std::vector<T> bias;   // one bias per output neuron, length = cols
    activations::Activation activation;

public:
    DenseLayer(
        size_t r,
        size_t c,
        activations::Activation act = activations::Activation::ReLU
    );

    inline T& operator()(size_t r, size_t c);

    inline const T& operator()(size_t r, size_t c) const;

    void randomize();

    void apply_activation(std::vector<T>& v) const;

    void print_memory_usage() const;

    void show_weights() const;
};

} // namespace neural_network

#include "layers.tpp"