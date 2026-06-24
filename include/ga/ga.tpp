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
        float mutation_prob,
        float elitism_pct,
        std::vector<size_t>     layers_dim,
        std::vector<Activation> layers_activations,
        Activation              output_activation
    )
        : input_len(input_len),
        output_len(output_len),
        mutation_prob(mutation_prob),
        elitism_pct(elitism_pct),
        layers_dim(std::move(layers_dim)),
        layers_activations(std::move(layers_activations)),
        output_activation(output_activation),
        pool_()   // spins up hardware_concurrency() workers
    {
        // Initialize population
        initialize_population(population, candidates);
    }


    // ─────────────────────────────────────────────
    // run — main evolution loop
    // ─────────────────────────────────────────────

    template<typename T>
    void GeneticAlgorithm<T>::run(
        int iterations,
        const std::vector<pd::StockPrice>& data
    ) {
        // ── Evaluate population via the thread pool ───────────────────────────
        std::vector<IndividualFitness<T>> individuals = get_population_fitness(population, data);

        for (const auto& ind : individuals)
            std::cout << ind.fitness << " ";
        std::cout << '\n';

        for (int round = 0; round < iterations; ++round) {

            ElitismResponse elitism_selection = get_elite_individuals(individuals);

            std::cout << "Elites fitness: ";

            for (const auto& ind : elitism_selection.survivors)
                std::cout << evaluate_fitness(ind, data) << " ";
            std::cout << '\n';

            std::cout
            << "Selected "
            << population.size() - elitism_selection.removed
            << " elite candidate(s), replenishing "
            << elitism_selection.removed
            << "...\n";

            std::cout << population.size() << '\n';
            replenish_population(elitism_selection.survivors, elitism_selection.removed);

            std::cout << population.size() << '\n';

            individuals = get_population_fitness(population, data);

            for (const auto& ind : individuals)
                std::cout << ind.fitness << " ";
            std::cout << '\n';

            std::cout << "\n================================================\n\n";
        }
    }


    // ─────────────────────────────────────────────
    // Initialize population
    // ─────────────────────────────────────────────

    template<typename T>
    void GeneticAlgorithm<T>::initialize_population(
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
    // evaluate_fitness
    // ─────────────────────────────────────────────

    template<typename T>
    std::vector<IndividualFitness<T>> GeneticAlgorithm<T>::get_population_fitness(
        const std::vector<nn::NeuralNetwork<T>> &population,
        const std::vector<pd::StockPrice>& data
    ) {
        // Each individual gets its own task. Workers pull from the queue as they
        // finish — at most hardware_concurrency() tasks run at once.
        std::vector<IndividualFitness<T>> individuals(population.size());

        for (size_t i = 0; i < population.size(); ++i)
            pool_.enqueue([&, i]() {
                individuals[i].index = i;
                individuals[i].fitness = evaluate_fitness(population[i], data);
            });

        pool_.wait_all();   // block until every individual has been scored

        return individuals;
    }

    template<typename T>
    float GeneticAlgorithm<T>::evaluate_fitness(
        const nn::NeuralNetwork<T>& network,
        const std::vector<pd::StockPrice>& data
    ) const {
        constexpr float eps = 1e-7f;  // avoid log(0)
        float total_loss = 0.0f;

        for (const auto& sample : data) {
            float p = network.forward(sample.to_features())[0];  // Hardcoded for the current problem (classification with one output)
            float y = sample.direction ? 1.0f : 0.0f;

            p = std::clamp(p, eps, 1.0f - eps);
            total_loss += -(y * std::log(p) + (1.0f - y) * std::log(1.0f - p));
        }

        // Negate so higher = better (GA maximises fitness). 0 (-0.0000001 because of clamp) = perfect score
        return -(total_loss / static_cast<float>(data.size()));
    }


    // ─────────────────────────────────────────────
    // Get elite individuals
    // ─────────────────────────────────────────────

    template<typename T>
    ElitismResponse<T> GeneticAlgorithm<T>::get_elite_individuals(
        std::vector<IndividualFitness<T>>& individuals
    ) {
        // Get how many elite members to keep
        size_t elite_count = static_cast<size_t>(
            population.size() * elitism_pct
        );
        // In case the % of elites and population size causes the elite count to be 0, this line sets it to 1
        elite_count = std::max<size_t>(1, elite_count);

        // Order individuals so the best fitness scores are first on the list
        std::partial_sort(
            individuals.begin(),
            individuals.begin() + elite_count,
            individuals.end(),
            [](const auto& a, const auto& b) {
                return a.fitness > b.fitness;
            }
        );

        // Get the survivors (top performers)
        std::vector<nn::NeuralNetwork<T>> survivors;
        survivors.reserve(population.size());

        for (size_t i = 0; i < elite_count; ++i)
        {
            survivors.push_back(
                std::move(
                    population[individuals[i].index]
                )
            );
        }

        // Get the number of individuals that need to be generated to replenish the population
        int removed =
            static_cast<int>(population.size()) -
            static_cast<int>(survivors.size());

        return ElitismResponse(survivors, removed);
    }


    // ─────────────────────────────────────────────
    // Replenish population
    // ─────────────────────────────────────────────
    template<typename T>
    void GeneticAlgorithm<T>::replenish_population(
        std::vector<nn::NeuralNetwork<T>>& elite_members,
        int count
    ) {
        elite_members.reserve(elite_members.size() + count);

        for (int i = 0; i < count; ++i)
            elite_members.emplace_back(
                input_len, output_len,
                layers_dim, layers_activations,
                output_activation
            );

        population = std::move(elite_members);
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
