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

def compute_error(a, y):
    m = y.shape[1]   # number of training examples
    return 0.5 / m * np.linalg.norm(a - y, 'fro') ** 2

NodeValues = namedtuple('NodeValues', ['z', 'a'])
Layer = namedtuple('Layer', ['weights', 'biases', 'activation_fun', 'activation_deriv'])

class NeuralNetwork:

    def __init__(self, unit_counts):
        self.layers = [None] + [
            Layer(
                np.random.randn(unit_counts[k], unit_counts[k - 1]),
                np.random.randn(unit_counts[k], 1),
                sigmoid if k < len(unit_counts) - 1 else identity,
                sigmoid_derivative if k < len(unit_counts) - 1 else identity_derivative
            )
            for k in range(1, len(unit_counts))
        ]

    def evaluate(self, a):
        values = [NodeValues(None, a)]
        for layer in self.layers[1:]:
            z = np.dot(layer.weights, a) + layer.biases
            a = layer.activation_fun(z)
            values.append(NodeValues(z, a))
        return values

    def backprop(self, values, y):
        m = y.shape[1]
        delta_weights = []
        delta_biases = []

        L = len(self.layers) - 1
        # l = L, L - 1, ..., 1
        for l in range(L, 0, -1):
            if l == L:
                # special case for output layer
                delta_a = (values[L].a - y) / m
            else:
                # delta_z is the delta z for layer l + 1
                delta_a = np.dot(self.layers[l + 1].weights.T, delta_z)

            # element-wise multiplication
            delta_z = delta_a * self.layers[l].activation_deriv(values[l].z)

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
            self.layers[1:] = [
                Layer(
                    layer.weights - learning_rate * delta_w,
                    layer.biases - learning_rate * delta_b,
                    layer.activation_fun,
                    layer.activation_deriv
                )
                for layer, delta_w, delta_b in zip(self.layers[1:], delta_weights, delta_biases)
            ]
        print(values[-1].a)
        print(ys)

if __name__ == "__main__":
    np.random.seed(seed=0)
    nn = NeuralNetwork([2, 5, 2])
    xs = np.array([
        [1.0, 0.0],
        [2.0, 0.0]
    ])
    ys = np.array([
        [1.0, 0.5],
        [3.0, 2.0]
    ])
    # values = nn.feedforward(xs)
    # print(values)
    # print(compute_error(values[-1].a, ys))
    # print(nn.backprop(values, ys))
    nn.train(xs, ys, 100, 0.1)
