#pragma once

#include <ga/ga.hpp>
#include <algorithm>
#include <iostream>

namespace ga {

    // ─────────────────────────────────────────────
    // Constructor — builds the initial population
    // ─────────────────────────────────────────────

    template<typename T>
    GeneticAlgorithm<T>::GeneticAlgorithm(
        int candidates,
        int input_len,
        int output_len,
        std::vector<size_t>     layers_dim,
        std::vector<Activation> layers_activations,
        Activation              output_activation
    )
        : input_len(input_len),
        output_len(output_len),
        layers_dim(std::move(layers_dim)),
        layers_activations(std::move(layers_activations)),
        output_activation(output_activation),
        pool_()   // spins up hardware_concurrency() workers
    {
        // Initialize population
        replenish(population, candidates);
    }


    // ─────────────────────────────────────────────
    // run — main evolution loop
    // ─────────────────────────────────────────────

    template<typename T>
    void GeneticAlgorithm<T>::run(
        int iterations,
        const std::vector<pd::StockPrice>& data
    ) {
        for (int round = 0; round < iterations; ++round) {

            // ── Evaluate population via the thread pool ───────────────────────────
            // Each individual gets its own task. Workers pull from the queue as they
            // finish — at most hardware_concurrency() tasks run at once.
            std::vector<float> scores(population.size());

            for (size_t i = 0; i < population.size(); ++i)
                pool_.enqueue([&, i]() {
                    scores[i] = evaluate_fitness(population[i], data);
                });

            pool_.wait_all();   // block until every individual has been scored

            // ── Compute mean ──────────────────────────────────────────────────────
            float total = 0.0f;
            for (float s : scores) total += s;
            float mean = total / static_cast<float>(population.size());

            // ── Print summary ─────────────────────────────────────────────────────
            std::vector<float> sorted_scores = scores;
            std::sort(sorted_scores.begin(), sorted_scores.end());

            std::cout << "Round " << round + 1 << " / " << iterations << '\n';
            std::cout << "Candidates: " << population.size() << '\n';
            print_score_summary(sorted_scores, 3);
            std::cout << "Scores mean: " << mean << '\n';

            // ── Cull below-mean candidates ────────────────────────────────────────
            std::vector<nn::NeuralNetwork<T>> survivors;
            survivors.reserve(population.size());

            for (size_t i = 0; i < population.size(); ++i)
                if (scores[i] >= mean)
                    survivors.push_back(std::move(population[i]));

            int removed = static_cast<int>(population.size()) - static_cast<int>(survivors.size());
            std::cout << "Removed " << removed << " candidates, replenishing...\n";

            // ── Replenish with fresh random candidates ────────────────────────────
            replenish(survivors, removed);
            population = std::move(survivors);

            std::cout << "\n================================================\n\n";
        }
    }


    // ─────────────────────────────────────────────
    // evaluate_fitness
    // ─────────────────────────────────────────────

    template<typename T>
    float GeneticAlgorithm<T>::evaluate_fitness(
        const nn::NeuralNetwork<T>& network,
        const std::vector<pd::StockPrice>& data
    ) const {
        constexpr float eps = 1e-7f;
        float total_loss = 0.0f;

        for (const auto& sample : data) {
            float p = network.forward(sample.to_features())[0];
            float y = sample.direction ? 1.0f : 0.0f;

            p = std::clamp(p, eps, 1.0f - eps);
            total_loss += -(y * std::log(p) + (1.0f - y) * std::log(1.0f - p));
        }

        // Negate so higher = better (GA maximises fitness)
        return -(total_loss / static_cast<float>(data.size()));
    }


    // ─────────────────────────────────────────────
    // replenish — append `count` fresh networks
    // ─────────────────────────────────────────────

    template<typename T>
    void GeneticAlgorithm<T>::replenish(
        std::vector<nn::NeuralNetwork<T>>& dest,
        int count
    ) const {
        dest.reserve(dest.size() + count);
        for (int i = 0; i < count; ++i)
            dest.emplace_back(
                input_len, output_len,
                layers_dim, layers_activations,
                output_activation
            );
    }


    // ─────────────────────────────────────────────
    // print_score_summary
    // ─────────────────────────────────────────────

    template<typename T>
    void GeneticAlgorithm<T>::print_score_summary(
        const std::vector<float>& sorted_scores,
        size_t n
    ) const {
        size_t total = sorted_scores.size();
        size_t count = std::min(n, total);

        std::cout << "Top " << count << " best scores:\n";
        for (size_t i = 0; i < count; ++i)
            std::cout << "  " << sorted_scores[total - 1 - i] << '\n';

        std::cout << "Top " << count << " worst scores:\n";
        for (size_t i = 0; i < count; ++i)
            std::cout << "  " << sorted_scores[i] << '\n';
    }

} // namespace ga
