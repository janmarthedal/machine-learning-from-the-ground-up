import numpy as np
from simple_neural_network import SimpleNeuralNetwork
from activation import SIGMOID_ACTIVATION, IDENTITY_ACTIVATION

if __name__ == "__main__":
    np.random.seed(seed=0)
    nn = SimpleNeuralNetwork([(2, ), (3, SIGMOID_ACTIVATION), (2, IDENTITY_ACTIVATION)])
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
