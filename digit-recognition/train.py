import numpy as np
from load_mnist import load_mnist_data
from np_utils import to_categorical
from simple_neural_network import SimpleNeuralNetwork
from activation import SIGMOID_ACTIVATION, IDENTITY_ACTIVATION

# produce the same initialization values every time
np.random.seed(seed=1)

def mnist_iteration_hook(epoch, error, nn, x_test, y_test):
    if epoch % 10 == 0:

        y_predict = nn.evaluate(x_test)[-1].a
        correct = np.argmax(y_predict, axis=0) == np.argmax(y_test, axis=0)

        # print("epoch: {}, error: {}".format(epoch, error))
        # print("correct: {}/{} ({:.1f}%)".format(np.sum(correct), correct.size, 100.0 * np.sum(correct) / correct.size))
        print("{}, {}, {}".format(epoch, error, np.sum(correct)))

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    x_train = x_train.reshape(-1, 28 * 28).T / 255.0
    y_train = to_categorical(y_train, 10).T
    x_test = x_test.reshape(-1, 28 * 28).T / 255.0
    y_test = to_categorical(y_test, 10).T

    print("x_train.shape: {}".format(x_train.shape))
    print("y_train.shape: {}".format(y_train.shape))
    print("x_test.shape: {}".format(x_test.shape))
    print("y_test.shape: {}".format(y_test.shape))

    nn = SimpleNeuralNetwork(784, [(300, SIGMOID_ACTIVATION), (10, IDENTITY_ACTIVATION)])
    _ = nn.train(x_train, y_train, 10000, 0.03, lambda epoch, error: mnist_iteration_hook(epoch, error, nn, x_test, y_test))
