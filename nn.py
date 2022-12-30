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
        self.weights = [np.random.randn(m, n) for n, m in zip(unit_counts[:-1], unit_counts[1:])]
        self.biases = [np.random.randn(m, 1) for m in unit_counts[1:]]
        self.activations = [ACTIVATION_FUNCTIONS[c[1]] for c in layer_config[1:]]

    def evaluate(self, a):
        values = [NodeValues(None, a)]
        for w, b, g in zip(self.weights, self.biases, self.activations):
            z = np.dot(w, a) + b
            a = g[0](z)
            values.append(NodeValues(z, a))
        return values

    def backprop(self, values, y):
        m = y.shape[1]
        L = len(self.weights)
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
                dJda = np.dot(self.weights[l + 1 - 1].T, dJdz)

            # '*' does element-wise multiplication
            # dJdz^l = dJda^l * g^l'(z^l)
            dJdz = dJda * self.activations[l - 1][1](values[l].z)

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
            delta_weights, delta_biases = self.backprop(values, ys)
            for k in range(0, len(delta_weights)):
                self.weights[k] -= learning_rate * delta_weights[k]
                self.biases[k] -= learning_rate * delta_biases[k]
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
