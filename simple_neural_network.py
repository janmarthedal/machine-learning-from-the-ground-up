import numpy as np
from collections import namedtuple
from np_utils import compute_error

def dummy_hook(epoch, error):
    pass

NodeValues = namedtuple('NodeValues', ['z', 'a'])

class Layer:
    def __init__(self, input_count, output_count, activation):
        self.W = 0.2 * np.random.randn(output_count, input_count)
        self.b = 0.2 * np.random.randn(output_count, 1)
        self.g = activation

class SimpleNeuralNetwork:

    def __init__(self, input_count, layer_params):
        unit_counts = [input_count] + [layer[0] for layer in layer_params]
        activations = [layer[1] for layer in layer_params]
        self.layers = [Layer(m, n, g) for m, n, g in zip(unit_counts[:-1], unit_counts[1:], activations)]

    def evaluate(self, a):
        values = [NodeValues(None, a)]
        for layer in self.layers:
            z = np.dot(layer.W, a) + layer.b
            a = layer.g[0](z)
            values.append(NodeValues(z, a))
        return values

    # compute gradient using back propagation
    def compute_gradient(self, values, y):
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
                dJda = np.dot(self.layers[l + 1 - 1].W.T, dJdz)

            # numpy '*' does element-wise multiplication
            # dJdz^l = dJda^l * g^l'(z^l)
            dJdz = dJda * self.layers[l - 1].g[1](values[l].z)

            # dJdW^l = dJdz^l * a^(l-1)^T
            dJdW = np.dot(dJdz, values[l - 1].a.T)
            # dJdb^l = dJdz^l * [1 1 ... 1]^T
            dJdb = np.sum(dJdz, axis=1, keepdims=True)

            dJdWs.append(dJdW)
            dJdbs.append(dJdb)

        dJdWs.reverse()
        dJdbs.reverse()

        return dJdWs, dJdbs

    def train(self, xs, ys, epochs, learning_rate, callback=dummy_hook):
        values = self.evaluate(xs)
        error = compute_error(values[-1].a, ys)
        callback(0, error)
        for epoch in range(epochs):
            dJdWs, dJdbs = self.compute_gradient(values, ys)
            for layer, dJdW, dJdb in zip(self.layers, dJdWs, dJdbs):
                layer.W -= learning_rate * dJdW
                layer.b -= learning_rate * dJdb
            values = self.evaluate(xs)
            error = compute_error(values[-1].a, ys)
            callback(epoch + 1, error)
        return values[-1].a
