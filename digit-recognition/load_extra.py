import numpy as np
import matplotlib.pyplot as plt

def load_pgm_digit(path):
    with open(path, 'r') as f:
        lines = f.read().strip().split('\n')
    assert lines[0] == 'P2'
    assert lines[1].startswith('#')
    assert lines[2] == '28 28'
    assert lines[3] == '255'
    assert len(lines) == 28 * 28 + 4
    return np.array(list(map(lambda x: 255 - int(x), lines[4:]))).reshape(28, 28)

def load_extra_data():
    x_extra = np.zeros((10, 28, 28))
    y_extra = list(range(0, 10))
    for i in y_extra:
        x_extra[i] = load_pgm_digit('./extra/{}.pgm'.format(i))
    return x_extra, np.array(y_extra)

if __name__ == '__main__':
    x_extra, y_extra = load_extra_data()

    num_row = 2
    num_col = 5
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(num_row * num_col):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(x_extra[i], cmap='gray_r')
        ax.set_title('Digit: {}'.format(y_extra[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()
