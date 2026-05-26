
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <functional>

#include <neural_network/activations.hpp>
#include <neural_network/neural_network.hpp>
using Activation = neural_network::activations::Activation;
namespace nn = neural_network;



int main(){
    // 3 inputs → hidden(4, ReLU) → hidden(4, Tanh) → 2 outputs (Softmax)
    nn::NeuralNetwork<float> nn(
        3, 2,
        {4, 4},
        {Activation::ReLU, Activation::Tanh},
        Activation::Softmax
    );

    nn.print_architecture();
    nn.print_architechture_detail();

    std::vector<float> input = {0.5f, -0.3f, 1.2f};
    std::vector<float> output = nn.forward(input);

    std::cout << "\nForward pass output: ";
    for (float v : output) std::cout << v << ' ';
    std::cout << '\n';

    // GA genome round-trip
    auto genome = nn.get_genome();
    std::cout << "Genome size: " << genome.size() << " weights\n";

    // Mutate one gene
    genome[0] += 0.1f;
    nn.set_genome(genome);

    std::cout << "Mutation applied and loaded back OK.\n";

    nn.print_architechture_detail();

    return 0;
}