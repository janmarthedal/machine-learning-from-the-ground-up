import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    t = sigmoid(z)
    return t * (1 - t)

IDENTITY_ACTIVATION = (lambda z: z, lambda z: np.ones(z.shape))
SIGMOID_ACTIVATION = (sigmoid, sigmoid_prime)
