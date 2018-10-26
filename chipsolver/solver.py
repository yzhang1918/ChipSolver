import numpy as np
import cvxpy as cvx


class ChipSolver:

    def __init__(self, board, chips_pool, up=None, lo=None, weights=None, solver=None):
        self.board = board
        self.chips_pool = chips_pool
        if up is None:
            up = np.array([1000, 1000, 1000, 1000])
        self.up = up
        self.lo = lo
        self.weights = weights
        self.solver = solver
        # prepare matrices for MIP
        self._A, self._M, self._H = self.prepare_matrices()

        self.is_solved = False
        self.selected_chips = []
        self._solution = []

    def prepare_matrices(self):
        type_ids, A = zip(*self.chips_pool)
        A = np.stack(A)  # attrs
        # M : block-state;  H : chip-state
        M = [self.board.block_state_mats[i] for i in type_ids]
        H = block_diag(*[np.ones(x.shape[1], dtype=int) for x in M])  # shape : [n_chips, n_states]
        M = np.concatenate(M, axis=1)  # shape : [n_blocks, n_states]
        return A, M, H

    def solve(self):
        prob, objval, x = mip_solve(A=self._A,
                                    M=self._M,
                                    H=self._H,
                                    up=self.up,
                                    lo=self.lo,
                                    weights=self.weights,
                                    solver=self.solver)
        assert prob.status == 'optimal'
        self.is_solved = True
        chip_idx = np.where(self._H @ x)[0]
        self.selected_chips = [self.chips_pool[i] for i in chip_idx]
        # show results
        selected_states = x > 0
        solution = []
        for line in self._M.T[np.where(selected_states)[0]]:
            solution.append(np.where(line)[0])
        self._solution = solution
        attrs = self._A.T @ self._H @ x
        return self.selected_chips, objval, attrs

    def show(self):
        assert self.is_solved
        fig = self.board.plot_solution(self._solution)
        return fig


def mip_solve(A, M, H, up, lo=None, weights=None, solver=None):
    n_chips, n_attrs = A.shape
    n_blocks, n_states = M.shape
    if lo is None:
        lo = np.zeros([n_attrs], dtype=int)
    if weights is None:
        weights = np.ones([n_attrs], dtype=int)
    # variables
    x = cvx.Variable(n_states, boolean=True)
    y = cvx.Variable(4, nonneg=True)

    s = A.T @ H @ x
    obj_val = weights @ y
    obj = cvx.Minimize(obj_val)
    constraints = [
        H @ x <= 1,
        M @ x <= 1,
        s + y >= up,
        s >= lo
    ]
    prob = cvx.Problem(obj, constraints)
    obj_v = prob.solve(solver=solver)
    return prob, obj_v, x.value


def block_diag(*arr):
    m = len(arr)
    n = np.sum(len(x) for x in arr)
    mat = np.zeros((m, n), dtype=int)
    c = 0
    for i, x in enumerate(arr):
        mat[i, c:c+len(x)] = x
        c += len(x)
    return mat
