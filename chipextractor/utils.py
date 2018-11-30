import numpy as np
from matplotlib import pyplot as plt


def conv2d(img, kernel):
    s = kernel.shape + tuple(np.subtract(img.shape, kernel.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    submat = strd(img, shape=s, strides=img.strides * 2)
    return np.einsum('ij,ijkl->kl', kernel, submat)


def show_gray_img(img, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.imshow(img, cmap='gray')


def consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def fill_blanks(arr, max_margin=1, verbose=1):
    n = len(arr)
    filled_arr = arr.copy()
    last_true_idx = -1
    triggered_times = 0
    for i in range(n):
        if arr[i]:
            if 1 < i - last_true_idx <= max_margin + 1:
                filled_arr[last_true_idx:i] = True
                triggered_times += 1
            last_true_idx = i
    if verbose > 0 and triggered_times > 0:
        print(f"Function fill_blanks triggered {triggered_times} times!")
    return filled_arr

