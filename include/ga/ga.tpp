#pragma once

#include <ga/ga.hpp>
#include <thread>

namespace ga {

template<typename T>
std::vector<nn::NeuralNetwork<T>> initialize_population(
    int candidates, int input_len, int output_len,
    std::vector<size_t> layers_dim, std::vector<Activation> layers_activations,
    Activation output_activation
) {
    std::vector<nn::NeuralNetwork<T>> population;
    population.reserve(candidates);

    for (int i = 0; i < candidates; i++)
        population.emplace_back(
            input_len, output_len,
            layers_dim, layers_activations,
            output_activation
        );

    return population;
}

template<typename T>
float evaluate_fitness(
    const nn::NeuralNetwork<T>& network,
    const std::vector<pd::StockPrice>& data
) {
    constexpr float eps = 1e-7f;  // avoid log(0)
    float total_loss = 0.0f;

    for (const auto& sample : data) {
        float p = network.forward(sample.to_features())[0];  // Hardcoded for the current problem (classification with one output)
        float y = sample.direction ? 1.0f : 0.0f;

        p = std::clamp(p, eps, 1.0f - eps);
        total_loss += -(y * std::log(p) + (1.0f - y) * std::log(1.0f - p));
    }

    // Negate so higher = better, ebcause GA maximizes fitness. 0 (-0.0000001 because of clamp) = perfect score
    return -(total_loss / static_cast<float>(data.size()));
}

template<typename T>
void run(
    int candidates, int input_len, int output_len,
    std::vector<size_t> layers_dim, std::vector<Activation> layers_activations,
    Activation output_activation, const std::vector<pd::StockPrice>& data
) {
    std::vector<nn::NeuralNetwork<T>> population = initialize_population<T>(
        candidates, input_len, output_len, layers_dim, layers_activations, output_activation
    );

    for (int round = 0; round < 3; round++) {
        // Evaluate population
        std::vector<float> scores(population.size());
        std::vector<std::thread> threads;

        for (size_t i = 0; i < population.size(); ++i)
            threads.emplace_back([&, i]() {
                scores[i] = evaluate_fitness(population[i], data);
            });

        for (auto& t : threads) t.join();

        // Get scores mean
        float total_scores = 0.0f;
        for (float score: scores)
            total_scores += score;
        float mean = total_scores / static_cast<float>(population.size());

        // Get top n best and worst scores to display them
        std::vector<float> sorted_scores = scores;
        std::sort(sorted_scores.begin(), sorted_scores.end());

        size_t n = sorted_scores.size();
        size_t count = std::min<size_t>(3, n);

        std::cout << "Candidates: " << n << '\n';

        std::cout << "Top " << count << " best scores:\n";
        for (size_t i = 0; i < count; ++i)
            std::cout << "  " << sorted_scores[n - 1 - i] << '\n';

        std::cout << "Top " << count << " worst scores:\n";
        for (size_t i = 0; i < count; ++i)
            std::cout << "  " << sorted_scores[i] << '\n';

        std::cout << "Scores mean: " << mean << '\n';

        // Remove candidates with fitness evaluation below the mean
        std::vector<nn::NeuralNetwork<T>> survivors;
        for (size_t i = 0; i < population.size(); ++i)
            if (scores[i] >= mean)
                survivors.push_back(std::move(population[i]));

        int removed = static_cast<int>(population.size()) - static_cast<int>(survivors.size());
        std::cout << "Removed " << removed << " candidates, replenishing...\n";

        // New random replacements
        auto newcomers = initialize_population<T>(
            removed, input_len, output_len, layers_dim, layers_activations, output_activation
        );
        for (auto& n : newcomers)
            survivors.push_back(std::move(n));

        population = std::move(survivors);

        std::cout << "\n================================================\n\n";
    }
}

} // namespace ga