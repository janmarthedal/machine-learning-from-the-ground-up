import numpy as np
from collections import namedtuple

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    t = sigmoid(z)
    return t * (1 - t)

IDENTITY_ACTIVATION = (lambda z: z, lambda z: np.ones(z.shape))
SIGMOID_ACTIVATION = (sigmoid, sigmoid_prime)

def compute_error(a, y):
    m = y.shape[1]   # number of training examples
    return 0.5 / m * np.linalg.norm(a - y, 'fro') ** 2

NodeValues = namedtuple('NodeValues', ['z', 'a'])

class Layer:
    def __init__(self, input_count, output_count, activation):
        self.weights = np.random.randn(output_count, input_count)
        self.biases = np.random.randn(output_count, 1)
        self.activation = activation
    def evaluate(self, a):
        z = np.dot(self.weights, a) + self.biases
        a = self.activation[0](z)
        return z, a
    def increment_weights(self, delta_weights, delta_biases):
        self.weights += delta_weights
        self.biases += delta_biases

class NeuralNetwork:

    def __init__(self, layer_config):
        unit_counts = [layer[0] for layer in layer_config]
        activations = [layer[1] for layer in layer_config[1:]]
        self.layers = [Layer(m, n, g) for m, n, g in zip(unit_counts[:-1], unit_counts[1:], activations)]

    def evaluate(self, a):
        values = [NodeValues(None, a)]
        for layer in self.layers:
            z, a = layer.evaluate(a)
            values.append(NodeValues(z, a))
        return values

    def backprop(self, values, y):
        m = y.shape[1]
        L = len(self.layers)
        dJdWs = []
        dJdbs = []

        # l = L, L - 1, ..., 1
        for l in range(L, 0, -1):
            if l == L:
                # special case for output layer
                # dJda^l = (a^L - y) / m
                dJda = (values[L].a - y) / m
            else:
                # dJda^l = W^(l+1)^T * dJdz^(l+1)
                dJda = np.dot(self.layers[l + 1 - 1].weights.T, dJdz)

            # numpy '*' does element-wise multiplication
            # dJdz^l = dJda^l * g^l'(z^l)
            dJdz = dJda * self.layers[l - 1].activation[1](values[l].z)

            # dJdW^l = dJdz^l * a^(l-1)^T
            dJdW = np.dot(dJdz, values[l - 1].a.T)
            # dJdb^l = dJdz^l * [1, 1, ..., 1]
            dJdb = np.sum(dJdz, axis=1, keepdims=True)

            dJdWs.append(dJdW)
            dJdbs.append(dJdb)

        dJdWs.reverse()
        dJdbs.reverse()

        return dJdWs, dJdbs

    def train(self, xs, ys, epochs, learning_rate):
        for epoch in range(epochs):
            values = self.evaluate(xs)
            error = compute_error(values[-1].a, ys)
            print("Epoch {}: error = {}".format(epoch, error))
            dJdWs, dJdbs = self.backprop(values, ys)
            for layer, dJdW, dJdb in zip(self.layers, dJdWs, dJdbs):
                layer.increment_weights(-learning_rate * dJdW, -learning_rate * dJdb)
        print(values[-1].a)
        print(ys)

if __name__ == "__main__":
    np.random.seed(seed=0)
    nn = NeuralNetwork([(2, ), (3, SIGMOID_ACTIVATION), (2, IDENTITY_ACTIVATION)])
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
