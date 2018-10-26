import numpy as np
from matplotlib import pyplot as plt

__all__ = ['conv2d',
           'decode_chip_from_indices',
           'show_chips',
           'add_grid']


def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


def decode_chip_from_indices(indices):
    """
    Generate a matrix representing a chip from indices format
    Parameters
    ----------
    indices : list
        a list of (i, j) index representing the colored pixels

    Returns
    -------
    mat
    """
    c = np.array(indices)  # shape : [k, 2]
    assert np.alltrue(np.min(c, axis=0) == 0)
    mat = np.zeros(np.max(c, axis=0) + 1, dtype=int)
    mat[c[:, 0], c[:, 1]] = 1
    return mat


def show_chips(mats, rows, cols, size=2):
    """
    Plot chips in a figure.
    Parameters
    ----------
    mats : list
        a list of matrices representing chips
    rows : int
        the number of rows
    cols : int
        the number of columns
    size : int
        the size of each chip

    Returns
    -------
    canvas, fig, ax

    """
    w, h = np.max([m.shape for m in mats], axis=0)  # width and height
    # empty canvas
    canvas = np.zeros([(w + 1) * rows + 1, (h + 1) * cols + 1])
    for i, mat in enumerate(mats):
        row_i, col_i = i // cols, i % cols
        x0, y0 = row_i * (w + 1) + 1, col_i * (h + 1) + 1
        x1 = x0 + mat.shape[0]
        y1 = y0 + mat.shape[1]
        canvas[x0:x1, y0:y1] += mat * mat.sum()  # color chips

    figsize = [cols * size, rows * size]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # The color map "Set2" has 8 colors
    cmap = plt.get_cmap('Set2')
    ax.imshow(np.ma.masked_equal(canvas, 0), cmap=cmap, vmin=1, vmax=cmap.N)

    # Generate Grid
    add_grid(ax, canvas)
    # Add texts
    for i in range(rows):
        for j in range(cols):
            y = (i + 1) * (w + 1) - w // 2 + 1
            x = j * (h + 1) + h // 2 + 1
            k = i * cols + j
            ax.text(x, y, f'{k}')

    fig.tight_layout()
    return canvas, fig, ax


def add_grid(ax, img):
    m, n = img.shape
    # Generate Grid
    # Hide major labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Minor ticks
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, m, 1), minor=True)
    # Grid
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    # Hide ticks
    ax.tick_params(axis='both', which='both', bottom=False, left=False, right=False, top=False)
