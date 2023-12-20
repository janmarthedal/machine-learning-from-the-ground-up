import numpy as np

# np.seterr(all='raise')

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    t = sigmoid(z)
    return t * (1 - t)

def relu(x):
    return (x >= 0) * x

def relu_prime(x):
    return x >= 0

IDENTITY_ACTIVATION = (lambda z: z, lambda z: np.ones(z.shape))
SIGMOID_ACTIVATION = (sigmoid, sigmoid_prime)
RELU_ACTIVATION = (relu, relu_prime)
