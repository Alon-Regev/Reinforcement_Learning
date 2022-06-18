import numpy as np

class NeuralNetwork():
    def __init__(self, layers_sizes: list, activation_functions: list = None):
        self.layers_sizes = layers_sizes
        self.layer_count = len(layers_sizes) - 1
        # initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(self.layer_count):
            self.weights.append(np.random.rand(layers_sizes[i], layers_sizes[i + 1]) - 0.5)
            self.biases.append(np.random.rand(layers_sizes[i + 1]) - 0.5)
        self.activation_functions = activation_functions[0]
        self.activation_derivatives = activation_functions[1]

    def get_output(self, inputs: list):
        """
        method returns the output of the neural network.
        input:
            inputs: list of inputs
        return: list of outputs
        """
        # front propagation
        out = inputs
        for i in range(self.layer_count):
            out = self.activation_functions[i](out.dot(self.weights[i]) + self.biases[i])
        return out

    def mutate(self):
        """
        method mutates the current parameters of the neural network.
        input: None
        output: None
        """
        mutation_rate = np.random.choice([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
        for i in range(self.layer_count):
            self.weights[i] += (np.random.rand(self.layers_sizes[i], self.layers_sizes[i + 1]) - 0.5) * mutation_rate
            self.biases[i] += (np.random.rand(self.layers_sizes[i + 1]) - 0.5) * mutation_rate

    def save(self, path: str) -> None:
        """
        method saves the current neural network to a file.
        input:
            path: path to save the file to
        return: None
        """
        np.savez(path, weights=self.weights, biases=self.biases)
        
    def load(self, path: str) -> None:
        """
        method loads a neural network from a file.
        input:
            path: path to load the file from
        return: None
        """
        data = np.load(path, allow_pickle=True)
        self.weights = data['weights']
        self.biases = data['biases']

# activation function definition
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x) * sigmoid(1 - x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
def softmax_derivative(x):
    return softmax(x) * (1 - softmax(x))
