import numpy as np
import matplotlib.pyplot as plt
from simple_neural_network import SimpleNeuralNetwork
from activation import SIGMOID_ACTIVATION, IDENTITY_ACTIVATION

# produce the same initialization values every time
np.random.seed(seed=0)

def iteration_hook(epoch, error):
    print("epoch: {}, error: {}".format(epoch, error))

def test1():
    nn = SimpleNeuralNetwork(2, [(3, SIGMOID_ACTIVATION), (2, IDENTITY_ACTIVATION)])
    xs = np.array([
        [1.0, 0.0],
        [2.0, 0.0]
    ])
    ys = np.array([
        [1.0, 0.5],
        [3.0, 2.0]
    ])
    output = nn.train(xs, ys, 20, 0.1, iteration_hook)
    print("output:\n", output)
    print("target:\n", ys)

def simple_linear_regression():
    nn = SimpleNeuralNetwork(1, [(1, IDENTITY_ACTIVATION)])
    np.random.seed(seed=1)
    xs = np.array([np.linspace(1, 5, 10)])
    ys = 0.5 * xs + 1.0 + 0.1 * np.random.randn(*xs.shape)
    output = nn.train(xs, ys, 50, 0.01, iteration_hook)
    print("output:\n", output)
    print("target:\n", ys)
    plt.scatter(xs, ys)
    plt.plot([xs[0][0], xs[0][-1]], [output[0][0], output[0][-1]], color="red")
    plt.show()

if __name__ == "__main__":
    # test1()
    simple_linear_regression()
