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
* Stock price direction prediction problem (binary classification)
* Technical indicator computation from raw NAV data (EMA, RSI, momentum, volatility)

---

## Architecture

The network is composed of an input vector, N hidden fully-connected layers, and an output layer. Each layer holds a weight matrix and a bias vector. The forward pass computes:

```
output[j] = activation( Σ (input[i] * W[i][j]) + bias[j] )
```

Softmax is applied across the full output vector rather than element-wise.

---

## Problem: Stock Price Direction

The current use case is binary classification — predicting whether a fund's NAV will go up or down the next trading day. Input data is a CSV with `Date` and `NAV` columns:

```
Date,NAV
2006-06-07T00:00:00.000Z,78.275
2006-06-08T00:00:00.000Z,76.922
...
```

Seven technical features are derived from the raw prices and fed into the network:

| Feature | Description |
|---|---|
| `nav / ema20 - 1` | Distance from 20-day EMA (short-term trend) |
| `nav / ema50 - 1` | Distance from 50-day EMA (medium-term trend) |
| `nav / ema200 - 1` | Distance from 200-day EMA (long-term trend) |
| `momentum5` | Rate of change over 5 days: `nav[T] / nav[T-5] - 1` |
| `momentum20` | Rate of change over 20 days: `nav[T] / nav[T-20] - 1` |
| `rsi14` | 14-period RSI, normalised to [0, 1] |
| `volatility20` | 20-day rolling std-dev of log-returns |

The label is `direction = true` if the next day's NAV is higher. The network outputs a single Sigmoid-activated value, where > 0.5 means "UP".

The first 199 entries (warm-up period for EMA-200) and the last entry (no future label) are dropped from the dataset.

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
// Load and preprocess stock data
auto prices = pd::load_data("data/IE0031786142.csv");

// Build network: 7 inputs → 3 hidden layers → 1 sigmoid output
nn::NeuralNetwork<float> nn(
    7, 1,
    {300, 150, 50},
    {Activation::ReLU, Activation::ReLU, Activation::Tanh},
    Activation::Sigmoid
);

// Run a forward pass on one sample
std::vector<float> input = prices[450].to_features();
std::vector<float> output = nn.forward(input);

// output[0] > 0.5 → predict UP, else DOWN
bool predicted_up = output[0] > 0.5f;
bool actual_up    = prices[450].direction;
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

> **Inside root folder**
```bash
mkdir build
cd build
cmake ..
cmake --build .
./genetic_algo
```

---

## TODO's
- Start implementing GA
- ✅ Define problem to solve (classification — is stock price going up or down?)
- Some nice TUI?? --> Distant future