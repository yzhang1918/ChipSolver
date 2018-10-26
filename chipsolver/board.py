import numpy as np

from matplotlib import pyplot as plt

from chipsolver.utils import *
from chipsolver._const import CHIPS


def show_all_chips():
    _1, fig, _2 = show_chips([decode_chip_from_indices(c) for c in CHIPS], 5, 8)
    return fig


class Chip:

    def __init__(self, indices):
        self.indices = indices
        self.states = self._get_all_states(decode_chip_from_indices(indices))

    def size(self):
        return len(self.indices)

    @staticmethod
    def _get_all_states(mat):
        """
        Rotate the chip to get all states.
        """
        states = []
        codes = set()
        for k in range(4):
            mat = np.rot90(mat, k)
            code = (mat.tostring(), mat.shape)
            if code in codes:
                continue
            codes.add(code)
            states.append(mat)
        return states

    def plot_all_states(self):
        _1, fig, _2 = show_chips(self.states, 1, 4, 2)
        return fig


class Board:

    def __init__(self, rows, cols, mask_idx=None):
        masked_board = np.ma.masked_equal(np.ones([rows, cols], dtype=int), 0)
        if mask_idx is not None:
            masked_board[mask_idx] = np.ma.masked
        index_board = masked_board.copy()
        index_board[~np.ma.getmaskarray(masked_board)] = np.arange(index_board.count(), dtype=int)
        self.masked_board = masked_board
        self.index_board = index_board
        self._unravel_indices = self._init_unravel_indices(index_board)
        self.shape = masked_board.shape
        self.block_state_mats = self.get_all_state_mats()

    @staticmethod
    def _init_unravel_indices(index_board):
        # Prepare unravel index
        row_mat = np.ma.zeros(index_board.shape, dtype=int)
        row_mat[:, :] = np.arange(row_mat.shape[0])[:, None]
        row_mat.mask = index_board.mask
        col_mat = np.ma.zeros(index_board.shape, dtype=int)
        col_mat[:, :] = np.arange(col_mat.shape[1])[None, :]
        col_mat.mask = index_board.mask
        row_index = row_mat.compressed()
        col_index = col_mat.compressed()
        _unravel_indices = np.stack([row_index, col_index]).T
        return _unravel_indices

    def size(self):
        return self.masked_board.count()

    def unravel_index(self, indices):
        idx = self._unravel_indices[indices]
        idx = tuple(idx.T)
        return idx

    def create_canvas(self, chips_blocks):
        canvas = np.zeros(self.shape + (len(chips_blocks),), dtype=int)
        for i, blocks in enumerate(chips_blocks):
            idx = self.unravel_index(blocks)
            canvas[:, :, i][idx] += 1
        return canvas

    def is_valid_solution(self, chips_blocks):
        canvas = self.create_canvas(chips_blocks)
        flag = np.alltrue(canvas.sum(axis=-1) <= 1)
        return flag

    def plot_solution(self, chips_blocks, figsize=None):
        if figsize is None:
            figsize = (5, 5)
        canvas = self.create_canvas(chips_blocks)
        w = np.arange(len(chips_blocks)) + 1
        img = (canvas * w[None, None, :]).sum(axis=-1)
        img = np.ma.masked_array(img, self.index_board.mask)
        cmap = plt.get_cmap('gist_rainbow')
        cmap.set_bad('white')
        cmap.set_under('black')
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        add_grid(ax, img)
        ax.imshow(img, cmap=cmap, vmin=1)
        return fig

    def find_chip_valid_states(self, chip):
        states = []
        board_mat = self.masked_board.filled()
        for kernel in chip.states:
            ret = conv2d(board_mat, kernel) == chip.size()
            m, n = kernel.shape
            for i, j in zip(*np.where(ret)):
                covered_blocks = self.index_board[i:i + m, j:j + n][kernel == 1]
                states.append(covered_blocks)
        if len(states):
            states = np.stack(states)
            assert np.ma.count_masked(states) == 0
            states = list(states.data)
        return states

    def get_all_state_mats(self):
        block_state_mats = []
        for _, chip_indices in enumerate(CHIPS):
            chip = Chip(chip_indices)
            block_state_mapping = self.find_chip_valid_states(chip)
            mapping_mat = np.zeros([self.size(), len(block_state_mapping)])
            for i, idx in enumerate(block_state_mapping):
                mapping_mat[:, i][idx] += 1
            block_state_mats.append(mapping_mat)
        return block_state_mats
