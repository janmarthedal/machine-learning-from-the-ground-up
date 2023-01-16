import numpy as np

def compute_error(a, y):
    m = y.shape[1]   # number of training examples
    return 0.5 * np.linalg.norm(a - y, 'fro') ** 2 / m

# Inspired by Keras, https://github.com/keras-team/keras/blob/2d183db0372e5ac2a686608cb9da0a9bd4319764/keras/utils/np_utils.py#L9
def to_categorical(y, num_classes=None):
    assert y.ndim == 1
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.size
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical
