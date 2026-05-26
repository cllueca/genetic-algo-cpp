#pragma once

#include <neural_network/neural_network.hpp>

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <utility>

namespace neural_network {

template<typename T>
NeuralNetwork<T>::NeuralNetwork(
    size_t in_size,
    size_t out_size,
    const std::vector<size_t>& hidden_sizes,
    const std::vector<activations::Activation>& hidden_activations,
    activations::Activation output_activation
)
    : input_size(in_size),
      output_size(out_size),
      num_hidden_layers(hidden_sizes.size()),
      output_layer(
          hidden_sizes.empty() ? in_size : hidden_sizes.back(),
          out_size,
          output_activation
      )
{
    if (hidden_sizes.size() != hidden_activations.size()) {
        throw std::invalid_argument("The number of activations and hidden layers must be the same");
    }

    if (input_size == 0 || output_size == 0) {
        throw std::invalid_argument("Input/output size must be > 0");
    }

    size_t previous_size = input_size;

    for (size_t i = 0; i < hidden_sizes.size(); ++i) {
        if (hidden_sizes[i] == 0) {
            throw std::invalid_argument("Hidden layer sizes must be > 0");
        }

        hidden_layers.emplace_back(
            previous_size,
            hidden_sizes[i],
            hidden_activations[i]
        );

        previous_size = hidden_sizes[i];
    }
}


// ─────────────────────────────────────────────
// Forward pass
// ─────────────────────────────────────────────
//
// Each layer is a weight matrix W of shape (in, out).
// For a row-input vector x (length = in):
//   output[j] = sum_i( x[i] * W(i, j) + bias[j] )   for j in 0..out
// Then activation is applied element-wise.
//
template<typename T>
std::vector<T> NeuralNetwork<T>::forward(
    const std::vector<T>& input
) const {
    if (input.size() != input_size)
        throw std::invalid_argument("Input vector size mismatch");

    std::vector<T> current = input;

    // Hidden layers
    for (const auto& layer : hidden_layers) {
        std::vector<T> next(layer.cols, T(0));
        for (size_t j = 0; j < layer.cols; ++j) {
            for (size_t i = 0; i < layer.rows; ++i)
                next[j] += current[i] * layer(i, j);
            next[j] += layer.bias[j];
        }
        layer.apply_activation(next);
        current = std::move(next);
    }

    // Output layer
    std::vector<T> out(output_layer.cols, T(0));

    for (size_t j = 0; j < output_layer.cols; ++j) {
        for (size_t i = 0; i < output_layer.rows; ++i)
            out[j] += current[i] * output_layer(i, j);
        out[j] += output_layer.bias[j];
    }
    output_layer.apply_activation(out);

    return out;
}


// ─────────────────────────────────────────────
// Genome helpers
// ─────────────────────────────────────────────
//
// Layout per layer:  [ weights (rows*cols) | bias (cols) ]
// Order: hidden_layers[0], hidden_layers[1], ..., output_layer
//
// Flatten all weights into a single vector  (hidden layers first, then output)
template<typename T>
std::vector<T> NeuralNetwork<T>::get_genome() const {
    std::vector<T> g;
    g.reserve(total_weights());
    for (const auto& layer : hidden_layers) {
        g.insert(g.end(), layer.data.begin(), layer.data.end());
        g.insert(g.end(), layer.bias.begin(), layer.bias.end());
    }
    g.insert(
        g.end(),
        output_layer.data.begin(),
        output_layer.data.end()
    );
    g.insert(
        g.end(),
        output_layer.bias.begin(),
        output_layer.bias.end()
    );
    return g;
}

// Load a flat genome back into the network weights
template<typename T>
void NeuralNetwork<T>::set_genome(const std::vector<T>& g) {

    if (g.size() != total_weights()) {
        throw std::invalid_argument("Genome size does not match network weight count");
    }

    size_t offset = 0;
    auto copy_chunk = [&](std::vector<T>& dest) {
        std::copy(
            g.begin() + offset,
            g.begin() + offset + dest.size(),
            dest.begin()
        );
        offset += dest.size();
    };

    for (auto& layer : hidden_layers) {
        copy_chunk(layer.data);
        copy_chunk(layer.bias);
    }

    copy_chunk(output_layer.data);
    copy_chunk(output_layer.bias);
}

// Total number of trainable weights
template<typename T>
size_t NeuralNetwork<T>::total_weights() const {
    size_t n = 0;
    for (const auto& layer : hidden_layers)
        n += layer.data.size() + layer.bias.size();
    n += output_layer.data.size() + output_layer.bias.size();
    return n;
}


// ─────────────────────────────────────────────
// Diagnostics
// ─────────────────────────────────────────────
template<typename T>
void NeuralNetwork<T>::print_architecture() const {
    std::cout << "Input size: " << input_size << '\n';

    for (size_t i = 0; i < hidden_layers.size(); ++i) {
        std::cout << "Hidden Layer " << i+1
                << ": " << hidden_layers[i].rows
                << " x " << hidden_layers[i].cols
                << "  (+" << hidden_layers[i].cols << " bias)\n";
    }
    std::cout << "Output Layer: "
                << output_layer.rows << " x " << output_layer.cols
                << "  (+" << output_layer.cols << " bias)\n";
    std::cout << "Total weights (incl. bias): " << total_weights() << '\n';
}

template<typename T>
void NeuralNetwork<T>::print_architechture_detail() const {
    std::cout << "==== NEURAL NETWORK ARCHITECHTURE DETAIL ====" << '\n';
    std::cout << "Input size: " << input_size << '\n';
    for (size_t l = 0; l < hidden_layers.size(); l++) {
        std::cout << std::string(40, '=') << '\n';
        std::cout << "\tHidden Layer " << l+1 << "\n\n";
        hidden_layers[l].print_memory_usage();
        std::cout << std::string(40, '=') << "\n\t\t↓\n";
        hidden_layers[l].show_weights();
    }
    std::cout << std::string(40, '=') << '\n';
    std::cout << "\tOutput layer" << "\n\n";
    output_layer.print_memory_usage();
    std::cout << std::string(40, '=') << "\n\t\t↓\n";
    output_layer.show_weights();
}

} // namespace neural_network