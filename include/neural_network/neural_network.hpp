#pragma once

#include <vector>
#include <cstddef>

#include "layers.hpp"
#include "activations.hpp"

namespace neural_network {

template<typename T>
class NeuralNetwork {
public:
    size_t input_size;
    size_t output_size;
    size_t num_hidden_layers;

    std::vector<layers::HiddenLayer<T>> hidden_layers;
    layers::HiddenLayer<T> output_layer;

public:
    NeuralNetwork(
        size_t in_size,
        size_t out_size,
        const std::vector<size_t>& hidden_sizes,
        const std::vector<activations::Activation>& hidden_activations,
        activations::Activation output_activation =
            activations::Activation::Softmax
    );

    std::vector<T> forward(const std::vector<T>& input) const;

    std::vector<T> get_genome() const;

    void set_genome(const std::vector<T>& g);

    size_t total_weights() const;

    void print_architecture() const;

    void print_architechture_detail() const;
};

} // namespace neural_network

#include "neural_network.tpp"