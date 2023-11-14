# Import necessary modules
from python import Python
let math = Python.import_module("math")
let random = Python.import_module("random")

# Define a simple neural network structure
struct NeuralNetwork:
    weights: List[List[float]]
    biases: List[float]

# Function to initialize a neural network with random weights and biases
fn initialize_neural_network(input_size: Int, hidden_size: Int, output_size: Int) -> NeuralNetwork:
    let weights = [[random.random() for _ in range(hidden_size)] for _ in range(input_size)]
    let biases = [random.random() for _ in range(hidden_size)]

    return NeuralNetwork(weights=weights, biases=biases)

# Function to perform a simple forward pass in the neural network
fn forward_pass(nn: NeuralNetwork, inputs: List[float]) -> List[float]:
    let hidden_layer = [sum(nn.weights[i][j] * inputs[i] for i in range(len(inputs))) + nn.biases[j] for j in range(len(nn.biases))]
    let output = [math.sigmoid(val) for val in hidden_layer]

    return output

# Example usage
fn main():
    let input_size = 2
    let hidden_size = 3
    let output_size = 1

    # Initialize neural network
    let neural_net = initialize_neural_network(input_size, hidden_size, output_size)

    # Input data
    let input_data = [0.5, 0.8]

    # Perform forward pass
    let output_data = forward_pass(neural_net, input_data)

    # Print the result
    print("Output:", output_data)
