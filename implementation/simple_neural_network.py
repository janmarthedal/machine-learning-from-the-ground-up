import numpy as np
from collections import namedtuple

# Compute error/loss function
def compute_error(A, Y):
    m = Y.shape[1]   # number of training examples
    return 0.5 * np.linalg.norm(A - Y, 'fro') ** 2 / m

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    t = sigmoid(x)
    return t * (1 - t)

# Activation functions and their derivatives
SIGMOID_ACTIVATION = (sigmoid, sigmoid_prime)
IDENTITY_ACTIVATION = (lambda x: x, lambda x: np.ones(x.shape))
RELU_ACTIVATION = (lambda x: (x >= 0) * x, lambda x: x >= 0)

# Helper class to store values of Z and A
NodeValues = namedtuple('NodeValues', ['Z', 'A'])

class Layer:
    def __init__(self, input_count, output_count, activation):
        self.W = 0.2 * np.random.randn(output_count, input_count)
        self.b = 0.2 * np.random.randn(output_count, 1)
        self.g = activation

class SimpleNeuralNetwork:

    # input_count is the number of input units
    # layer_params is a list of tuples (unit_count, activation)
    def __init__(self, input_count, layer_params):
        unit_counts = [input_count] + [layer[0] for layer in layer_params]
        activations = [layer[1] for layer in layer_params]
        self.layers = [Layer(m, n, g) for m, n, g in zip(unit_counts[:-1], unit_counts[1:], activations)]

    # Compute output of each layer
    def evaluate(self, A):
        values = [NodeValues(None, A)]
        for layer in self.layers:
            Z = np.dot(layer.W, A) + layer.b
            A = layer.g[0](Z)
            values.append(NodeValues(Z, A))
        return values

    # Compute gradient using back propagation
    def compute_gradient(self, values, Y):
        m = Y.shape[1]
        L = len(self.layers)
        dWs = []
        dbs = []

        # l = L, L - 1, ..., 1
        for l in range(L, 0, -1):
            if l == L:
                # special case for output layer
                # dA^l = (A^L - Y) / m
                dA = (values[L].A - Y) / m
            else:
                # dA^l = W^(l+1)^T * dZ^(l+1)
                dA = np.dot(self.layers[l].W.T, dZ)

            # numpy '*' does element-wise multiplication
            # dZ^l = dA^l * g^l'(Z^l)
            dZ = dA * self.layers[l - 1].g[1](values[l].Z)

            # dW^l = dZ^l * A^(l-1)^T
            dW = np.dot(dZ, values[l - 1].A.T)
            # db^l = dZ^l * [1 1 ... 1]^T
            db = np.sum(dZ, axis=1, keepdims=True)

            dWs.append(dW)
            dbs.append(db)

        dWs.reverse()
        dbs.reverse()

        return dWs, dbs

    # Train the neural network
    def train(self, Xs, Ys, epochs, learning_rate, callback):
        values = self.evaluate(Xs)
        error = compute_error(values[-1].A, Ys)
        callback(0, error)
        for epoch in range(epochs):
            dWs, dbs = self.compute_gradient(values, Ys)
            for layer, dW, db in zip(self.layers, dWs, dbs):
                layer.W -= learning_rate * dW
                layer.b -= learning_rate * db
            values = self.evaluate(Xs)
            error = compute_error(values[-1].A, Ys)
            callback(epoch + 1, error)
        return values[-1].A

def iteration_hook(epoch, error):
    if epoch % 100 == 0:
        print("epoch: {}, error: {}".format(epoch, error))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # produce the same initial weights every time
    np.random.seed(seed=0)
    nn = SimpleNeuralNetwork(1, [(20, SIGMOID_ACTIVATION), (1, IDENTITY_ACTIVATION)])
    Xs = np.arange(-5, 5, 0.1).reshape(1, -1)
    Ys = np.sin(Xs)
    output = nn.train(Xs, Ys, 20000, 0.1, iteration_hook)
    plt.plot(Xs.ravel(), Ys.ravel(), 'b', label='target')
    plt.plot(Xs.ravel(), output.ravel(), 'r', label='NN output')
    plt.legend(loc='upper right')
    plt.show()
