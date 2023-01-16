import numpy as np
import matplotlib.pyplot as plt
import gzip

def load_mnist_data():
    def load(path, offset):
        with open(path, 'rb') as f:
            data = f.read()
        # https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html
        # "This should be safe in general, but it may make sense to copy the result
        # when the original object is mutable or untrusted."
        return np.frombuffer(gzip.decompress(data), dtype=np.uint8, offset=offset).copy()
    x_train = load('./mnist/train-images-idx3-ubyte.gz', 16).reshape(-1, 28, 28)
    y_train = load('./mnist/train-labels-idx1-ubyte.gz', 8)
    assert x_train.shape[0] == y_train.shape[0]
    x_test = load('./mnist/t10k-images-idx3-ubyte.gz', 16).reshape(-1, 28, 28)
    y_test = load('./mnist/t10k-labels-idx1-ubyte.gz', 8)
    assert x_test.shape[0] == y_test.shape[0]
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    num_row = 5
    num_col = 10
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(num_row * num_col):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(x_train[i], cmap='gray_r')
        ax.set_title('Digit: {}'.format(y_train[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()
