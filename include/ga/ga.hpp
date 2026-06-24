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
    class GeneticAlgorithm {
    public:
        // Network topology — stored so the GA can replenish dead candidates
        int input_len;
        int output_len;
        std::vector<size_t>     layers_dim;
        std::vector<Activation> layers_activations;
        Activation              output_activation;

        std::vector<nn::NeuralNetwork<T>> population;

    public:
        GeneticAlgorithm(
            int candidates,
            int input_len,
            int output_len,
            std::vector<size_t>     layers_dim,
            std::vector<Activation> layers_activations,
            Activation              output_activation
        );

        void run(
            int iterations,
            const std::vector<pd::StockPrice>& data
        );

    private:
        float evaluate_fitness(
            const nn::NeuralNetwork<T>& network,
            const std::vector<pd::StockPrice>& data
        ) const;

        // Spawn `count` randomly-initialised networks and append them to `dest`
        void replenish(std::vector<nn::NeuralNetwork<T>>& dest, int count) const;

        // Print the top-n best and worst scores from a sorted score vector
        void print_score_summary(
            const std::vector<float>& sorted_scores,
            size_t n
        ) const;
    };

} // namespace ga

#include "ga.tpp"