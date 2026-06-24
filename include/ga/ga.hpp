#pragma once

#include <neural_network/neural_network.hpp>
#include <neural_network/activations.hpp>
#include <problems/price_direction.hpp>
#include <ga/thread_pool.hpp>
#include <vector>
#include <cmath>

namespace nn = neural_network;
namespace pd = data::price_direction;
using Activation = nn::activations::Activation;

namespace ga {

    template<typename T>
    struct IndividualFitness {
        size_t index;
        float fitness;
    };

    template<typename T>
    struct ElitismResponse {
        std::vector<nn::NeuralNetwork<T>> survivors;
        int removed;
    };

    template<typename T>
    class GeneticAlgorithm {
    public:
        // Network topology — stored so the GA can replenish dead candidates
        int input_len;
        int output_len;
        float mutation_prob;
        float elitism_pct;
        std::vector<size_t>     layers_dim;
        std::vector<Activation> layers_activations;
        Activation              output_activation;

        std::vector<nn::NeuralNetwork<T>> population;

    public:
        GeneticAlgorithm(
            int candidates,
            int input_len,
            int output_len,
            float mutation_prob,
            float elitism_pct,
            std::vector<size_t>     layers_dim,
            std::vector<Activation> layers_activations,
            Activation              output_activation
        );

        void run(
            int iterations,
            const std::vector<pd::StockPrice>& data
        );

    private:
        // Fixed-size thread pool — created once, reused every generation
        ThreadPool pool_;

        void initialize_population(std::vector<nn::NeuralNetwork<T>>& dest, int count) const;

        std::vector<IndividualFitness<T>> get_population_fitness(
            const std::vector<nn::NeuralNetwork<T>>& population,
            const std::vector<pd::StockPrice>& data
        );

        float evaluate_fitness(
            const nn::NeuralNetwork<T>& network,
            const std::vector<pd::StockPrice>& data
        ) const;

        ElitismResponse<T> get_elite_individuals(
            std::vector<IndividualFitness<T>>& individuals
        );

        void replenish_population(
            std::vector<nn::NeuralNetwork<T>>& elite_members,
            int count
        );

        // Print the top-n best and worst scores from a sorted score vector
        void print_score_summary(
            const std::vector<float>& sorted_scores,
            size_t n
        ) const;
    };

} // namespace ga

#include "ga.tpp"