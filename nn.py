import numpy as np
from collections import namedtuple

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    t = sigmoid(z)
    return t * (1 - t)

def identity(z):
    return z

def identity_derivative(z):
    return np.ones(z.shape)

ACTIVATION_FUNCTIONS = {
    'identity': (identity, identity_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
}

def compute_error(a, y):
    m = y.shape[1]   # number of training examples
    return 0.5 / m * np.linalg.norm(a - y, 'fro') ** 2

NodeValues = namedtuple('NodeValues', ['z', 'a'])
Layer = namedtuple('Layer', ['weights', 'biases', 'activation_fun', 'activation_deriv'])

class NeuralNetwork:

    def __init__(self, layer_config):
        unit_counts = [layer[0] for layer in layer_config]
        self.weights = [None] + [np.random.randn(m, n) for n, m in zip(unit_counts[:-1], unit_counts[1:])]
        self.biases = [None] + [np.random.randn(m, 1) for m in unit_counts[1:]]
        self.activation_functions = [None] + [ACTIVATION_FUNCTIONS[c[1]] for c in layer_config[1:]]

    def evaluate(self, a):
        values = [NodeValues(None, a)]
        for weights, biases, g in list(zip(self.weights, self.biases, self.activation_functions))[1:]:
            z = np.dot(weights, a) + biases
            a = g[0](z)
            values.append(NodeValues(z, a))
        return values

    def backprop(self, values, y):
        m = y.shape[1]
        delta_weights = []
        delta_biases = []

        L = len(values) - 1
        # l = L, L - 1, ..., 1
        for l in range(L, 0, -1):
            if l == L:
                # special case for output layer
                delta_a = (values[L].a - y) / m
            else:
                # delta_z is the delta z for layer l + 1
                delta_a = np.dot(self.weights[l + 1].T, delta_z)

            # element-wise multiplication
            delta_z = delta_a * self.activation_functions[l][1](values[l].z)

            delta_w = np.dot(delta_z, values[l - 1].a.T)
            delta_b = np.sum(delta_z, axis=1, keepdims=True)

            delta_weights.append(delta_w)
            delta_biases.append(delta_b)

        delta_weights.reverse()
        delta_biases.reverse()

        return delta_weights, delta_biases

    def train(self, xs, ys, epochs, learning_rate):
        for epoch in range(epochs):
            values = self.evaluate(xs)
            error = compute_error(values[-1].a, ys)
            print("Epoch {}: error = {}".format(epoch, error))
            delta_weights, delta_biases = self.backprop(values, ys)
            for k in range(0, len(delta_weights)):
                self.weights[k + 1] -= learning_rate * delta_weights[k]
                self.biases[k + 1] -= learning_rate * delta_biases[k]
        print(values[-1].a)
        print(ys)

if __name__ == "__main__":
    np.random.seed(seed=0)
    nn = NeuralNetwork([(2, ), (3, 'sigmoid'), (2, 'identity')])
    xs = np.array([
        [1.0, 0.0],
        [2.0, 0.0]
    ])
    ys = np.array([
        [1.0, 0.5],
        [3.0, 2.0]
    ])
    # values = nn.evaluate(xs)
    # print(values)
    # print(compute_error(values[-1].a, ys))
    # print(nn.backprop(values, ys))
    nn.train(xs, ys, 20, 0.1)
