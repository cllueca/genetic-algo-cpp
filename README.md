# Neural Network + Genetic Algorithm from scratch in C++

---

## Overview

This project implements a simple feedforward neural network entirely from scratch in C++ with no external machine learning libraries. It is designed for experimentation with neural network inference and genetic algorithm–style weight optimization.

It may not be the most efficient way of solving the given problem, but this is just an excuse to kill some free time while coming back to GA (I enjoyed them a lot when I first got introduced to them), learning C++ and reminiscing AI maths — even though I am not implementing backprop :)

---

## Features

* Fully templated neural network (`float`, `double`, etc.)
* Feedforward architecture with bias terms per layer
* Multiple activation functions: Linear, ReLU, Leaky ReLU, Sigmoid, Tanh, Swish, Softmax
* Arbitrary hidden layer configuration
* Genome encoding / decoding for genetic algorithms
* Softmax implemented with numerical stability
* Debug-friendly architecture and weight printing

---

## Architecture

The network is composed of an input vector, N hidden fully-connected layers, and an output layer. Each layer holds a weight matrix and a bias vector. The forward pass computes:

```
output[j] = activation( Σ (input[i] * W[i][j]) + bias[j] )
```

Softmax is applied across the full output vector rather than element-wise.

---

## Genetic Algorithm Support

Instead of gradient-based learning, weights are manipulated directly as a flat genome vector, making mutation and crossover straightforward:

```cpp
auto genome = nn.get_genome();   // export
genome[i] += mutation_value;     // mutate
nn.set_genome(genome);           // reload
```

---

## Example Usage

```cpp
NeuralNetwork<float> nn(
    3, 2,
    {4, 4},
    {Activation::ReLU, Activation::Tanh},
    Activation::Softmax
);

std::vector<float> input = {0.5f, -0.3f, 1.2f};
auto output = nn.forward(input);

for (float v : output)
    std::cout << v << " ";
```

---

## Limitations

* ❌ No backpropagation
* ❌ No batching
* ❌ No GPU acceleration

---

## Future Improvements

* Implement crossover operators for GA
* Add serialization to file
* Add a full training loop with GA population management
* Optimize matrix operations

---

## Purpose

Educational and experimental — focused on understanding neural network internals and exploring evolutionary optimization in C++.

---

## Run (simple)

```bash
g++ main.cpp -o main
./main
```

---

## TODO's
- Start implementing GA
- Define problem to solve (classification - is stock price going up or down?)
- Some nice TUI?? --> Distant future