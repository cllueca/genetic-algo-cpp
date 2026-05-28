#pragma once

#include <neural_network/neural_network.hpp>
#include <neural_network/activations.hpp>
#include <problems/price_direction.hpp>
#include <vector>
#include <cmath>

namespace nn = neural_network;
namespace pd = data::price_direction;
using Activation = nn::activations::Activation;

namespace ga {

template<typename T>
std::vector<nn::NeuralNetwork<T>> initialize_population(
    int candidates, int input_len, int output_len,
    std::vector<size_t> layers_dim, std::vector<Activation> layers_activations,
    Activation output_activation
);

template<typename T>
float evaluate_fitness(
    const nn::NeuralNetwork<T>& network,
    const std::vector<pd::StockPrice>& data
);

template<typename T>
void run(
    int candidates, int input_len, int output_len,
    std::vector<size_t> layers_dim, std::vector<Activation> layers_activations,
    Activation output_activation, const std::vector<pd::StockPrice>& data
);

} // namespace ga

#include "ga.tpp"