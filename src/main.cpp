
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <functional>


// ─────────────────────────────────────────────
//  Activation functions
// ─────────────────────────────────────────────
enum class Activation {
    Linear,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    Swish,       // x * sigmoid(x)  – smooth, good for hidden layers
    Softmax,     // applied to the whole output vector, not element-wise
};

template<typename T>
T activation_fn(Activation act, T x) {
    switch (act) {
        case Activation::Linear:    return x;
        case Activation::ReLU:      return x > T(0) ? x : T(0);
        case Activation::LeakyReLU: return x > T(0) ? x : T(0.01) * x;
        case Activation::Sigmoid:   return T(1) / (T(1) + std::exp(-x));
        case Activation::Tanh:      return std::tanh(x);
        case Activation::Swish:     return x / (T(1) + std::exp(-x));
        case Activation::Softmax:   return x;   // Handled at the vector level; fall back to identity here.
    }
    return x;
}

// Softmax over a whole vector (in-place)
template<typename T>
void softmax_inplace(std::vector<T>& v) {
    T max_val = *std::max_element(v.begin(), v.end());  // numerical stability
    T sum = T(0);
    for (auto& x : v) { x = std::exp(x - max_val); sum += x; }
    for (auto& x : v) { x /= sum; }
}


// ─────────────────────────────────────────────
//  Random helpers
// ─────────────────────────────────────────────
template<typename T>
T random_value(){
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    return static_cast<T>(dist(gen));
}


// ─────────────────────────────────────────────
//  HiddenLayer  (weight matrix + its activation)
// ─────────────────────────────────────────────
template<typename T>
class HiddenLayer {
public:
    size_t rows, cols;
    std::vector<T> data;   // weights: shape (rows, cols), row-major
    std::vector<T> bias;   // one bias per output neuron, length = cols
    Activation activation;

public:
    HiddenLayer(size_t r, size_t c, Activation act = Activation::ReLU)
        : rows(r), cols(c), data(r * c), bias(c), activation(act)
    {
        randomize();
    }

    inline T& operator()(size_t r, size_t c)             { return data[r * cols + c]; }
    inline const T& operator()(size_t r, size_t c) const { return data[r * cols + c]; }

    void randomize() {
        for (auto& x : data) x = random_value<T>();
        for (auto& x : bias) x = random_value<T>();
    }

    void apply_activation(std::vector<T>& v) const {
        if (activation == Activation::Softmax) {
            softmax_inplace(v);
        } else {
            for (auto& x : v) x = activation_fn(activation, x);
        }
    }

    void print_memory_usage() const {
        std::cout << "Type size: " << sizeof(T) << " bytes\n";
        std::cout << "Matrix memory: " << data.size() * sizeof(T) << " bytes\n";
        std::cout << "Bias memory:   " << bias.size() * sizeof(T) << " bytes\n";
    }

    void show_weights() const {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c)
                std::cout << (*this)(r, c) << ' ';
            std::cout << '\n';
        }
        std::cout << "bias: ";
        for (auto& b : bias) std::cout << b << ' ';
        std::cout << "\n\n";
    }
};


// ─────────────────────────────────────────────
//  NeuralNetwork
// ─────────────────────────────────────────────
template<typename T>
class NeuralNetwork {
public:
    size_t input_size;
    size_t output_size;
    size_t num_hidden_layers;

    std::vector<HiddenLayer<T>> hidden_layers;
    HiddenLayer<T> output_layer;

    NeuralNetwork(
        size_t in_size,
        size_t out_size,
        const std::vector<size_t>&     hidden_sizes,
        const std::vector<Activation>& hidden_activations,
        Activation output_activation = Activation::Softmax
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
        if (hidden_sizes.size() != hidden_activations.size())
            throw std::invalid_argument("The number of activations and hidden layers must be the same");

        if (input_size == 0 || output_size == 0)
            throw std::invalid_argument("Input/output size must be > 0");

        size_t previous_size = input_size;
        for (size_t i = 0; i < hidden_sizes.size(); ++i) {
            if (hidden_sizes[i] == 0)
                throw std::invalid_argument("Hidden layer sizes must be > 0");
            hidden_layers.emplace_back(previous_size, hidden_sizes[i], hidden_activations[i]);
            previous_size = hidden_sizes[i];
        }
    }

    // ── Forward pass ────────────────────────────────────────────────────────
    //
    // Each layer is a weight matrix W of shape (in, out).
    // For a row-input vector x (length = in):
    //   output[j] = sum_i( x[i] * W(i, j) + bias[j] )   for j in 0..out
    // Then activation is applied element-wise.
    //
    std::vector<T> forward(const std::vector<T>& input) const {
        if (input.size() != input_size)
            throw std::invalid_argument("Input vector size mismatch");

        std::vector<T> current = input;

        // Hidden layers
        for (const auto& layer : hidden_layers) {
            std::vector<T> next(layer.cols, T(0));
            for (size_t j = 0; j < layer.cols; ++j) {
                for (size_t i = 0; i < layer.rows; ++i)
                    next[j] += current[i] * layer(i, j);
                next[j] += layer.bias[j];          // <-- bias added here
            }
            layer.apply_activation(next);
            current = std::move(next);
        }

        // Output layer
        std::vector<T> out(output_layer.cols, T(0));
        for (size_t j = 0; j < output_layer.cols; ++j) {
            for (size_t i = 0; i < output_layer.rows; ++i)
                out[j] += current[i] * output_layer(i, j);
            out[j] += output_layer.bias[j];        // <-- bias added here
        }
        output_layer.apply_activation(out);

        return out;
    }


    // ── Genome helpers (for GA) ──────────────────────────────────────────────
    //
    // Layout per layer:  [ weights (rows*cols) | bias (cols) ]
    // Order: hidden_layers[0], hidden_layers[1], ..., output_layer
    //
    // Flatten all weights into a single vector  (hidden layers first, then output)
    std::vector<T> get_genome() const {
        std::vector<T> g;
        g.reserve(total_weights());
        for (const auto& layer : hidden_layers) {
            g.insert(g.end(), layer.data.begin(), layer.data.end());
            g.insert(g.end(), layer.bias.begin(), layer.bias.end());
        }
        g.insert(g.end(), output_layer.data.begin(), output_layer.data.end());
        g.insert(g.end(), output_layer.bias.begin(), output_layer.bias.end());
        return g;
    }

    // Load a flat genome back into the network weights
    void set_genome(const std::vector<T>& g) {
        if (g.size() != total_weights())
            throw std::invalid_argument("Genome size does not match network weight count");

        size_t offset = 0;
        auto copy_chunk = [&](std::vector<T>& dest) {
            std::copy(g.begin() + offset, g.begin() + offset + dest.size(), dest.begin());
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
    size_t total_weights() const {
        size_t n = 0;
        for (const auto& layer : hidden_layers)
            n += layer.data.size() + layer.bias.size();
        n += output_layer.data.size() + output_layer.bias.size();
        return n;
    }


    // ── Diagnostics ─────────────────────────────────────────────────────────
    void print_architecture() const {
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

    void print_architechture_detail() const {
        std::cout << "==== NEURAL NETWORK ARCHITECHTURE DETAIL ====" << '\n';
        std::cout << "Input size: " << input_size << '\n';
        for (size_t l = 0; l < hidden_layers.size(); l++) {
            std::cout << "  Hidden Layer " << l+1 << "\n\t↓\n";
            hidden_layers[l].show_weights();
        }
        std::cout << "  Output layer" << '\n';
        output_layer.show_weights();
    }
};


int main(){

    /***************************

        LAYER

    ***************************/
    HiddenLayer<double> fp64_matrix(3, 4);
    std::cout << "\nFP64:\n";
    fp64_matrix.print_memory_usage();

    HiddenLayer<float> fp32_matrix(3, 4);
    std::cout << "FP32:\n";
    fp32_matrix.print_memory_usage();

    std::cout << fp32_matrix.data.size() << std::endl;

    // 3 inputs → hidden(4, ReLU) → hidden(4, Tanh) → 2 outputs (Softmax)
    NeuralNetwork<float> nn(
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