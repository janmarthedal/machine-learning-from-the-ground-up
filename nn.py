import numpy as np
import matplotlib.pyplot as plt
from load_mnist import load_mnist_data
from np_utils import to_categorical
from simple_neural_network import SimpleNeuralNetwork
from activation import RELU_ACTIVATION, SIGMOID_ACTIVATION, IDENTITY_ACTIVATION

# produce the same initialization values every time
np.random.seed(seed=1)

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

def mnist_iteration_hook(epoch, error, nn, x_test, y_test):
    if epoch % 10 == 0:
        print("epoch: {}, error: {}".format(epoch, error))
        y_predict = nn.evaluate(x_test)[-1].a
        # print("Prediction: {}".format(y0))
        # print("Target: {}".format(y_test[0:5]))
        correct = np.argmax(y_predict, axis=0) == np.argmax(y_test, axis=0)
        print("correct: {}/{} ({}%)".format(np.sum(correct), correct.size, round(100.0 * np.sum(correct) / correct.size)))

# 2000 training samples, 200 test samples, 0.005 learning rate
#   4300 iterations, 75% accuracy
#   9780 iterations, 80% accuracy

def mnist_test():
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    x_train = x_train.reshape(-1, 28 * 28).T / 255.0
    y_train = to_categorical(y_train, 10).T
    x_test = x_test.reshape(-1, 28 * 28).T / 255.0
    y_test = to_categorical(y_test, 10).T

    # x_train = x_train[:, 0:2000]
    # y_train = y_train[:, 0:2000]
    # x_test = x_test[:, 0:200]
    # y_test = y_test[:, 0:200]

    print("x_train.shape: {}".format(x_train.shape))
    print("y_train.shape: {}".format(y_train.shape))
    print("x_test.shape: {}".format(x_test.shape))
    print("y_test.shape: {}".format(y_test.shape))
    nn = SimpleNeuralNetwork(784, [(300, SIGMOID_ACTIVATION), (10, IDENTITY_ACTIVATION)])
    _ = nn.train(x_train, y_train, 10000, 0.01, lambda epoch, error: mnist_iteration_hook(epoch, error, nn, x_test, y_test))

if __name__ == "__main__":
    # test1()
    # simple_linear_regression()
    mnist_test()
