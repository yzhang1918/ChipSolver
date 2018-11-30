import numpy as np
import pandas as pd
from collections import namedtuple
import argparse

from chipsolver import *


def load_board(name):
    if name.startswith('BGM-71'):
        star = int(name[-1])
        if star == 5:
            mask_idx = None
        elif star == 4:
            mask_idx = list(zip(*[[4, 5], [5, 3], [5, 4], [5, 5]]))
        else:
            raise NotImplementedError
        board = Board(6, 6, mask_idx)
    elif name.startswith('AGS-30'):
        mat = np.array([[0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 1, 1, 0, 0]
                        ])
        star = int(name[-1])
        if star == 5:
            board = Board(8, 8, np.where(mat == 0))
        else:
            raise NotImplementedError
    elif name.startswith('2B14'):
        mat = np.array([[0, 0, 1, 0, 0, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 0, 0, 1, 0, 0]
                       ])
        star = int(name[-1])
        if star == 5:
            board = Board(6, 8, np.where(mat == 0))
        else:
            raise NotImplementedError
    elif name.startswith('Ex1'):
        mat = np.array([[1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 0, 0, 1, 1],
                        [1, 1, 0, 0, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        ])
        board = Board(6, 6, np.where(mat == 0))
    elif name.startswith('Ex2'):
        mat = np.array([[0, 0, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 0, 0],
                        ])
        board = Board(6, 6, np.where(mat == 0))
    elif name.startswith('Ex3'):
        mat = np.array([[1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 1, 1, 1],
                        [1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1],
                        ])
        board = Board(6, 6, np.where(mat == 0))
    elif name.startswith('Ex4'):
        mat = np.array([[1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 1, 1, 1],
                        [1, 0, 0, 0, 1, 1],
                        [1, 1, 0, 0, 0, 1],
                        [1, 1, 1, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1],
                        ])
        board = Board(6, 6, np.where(mat == 0))
    else:
        raise NotImplementedError
    return board


class ConsoleLogger:

    def __init__(self):
        pass

    def log(self, msg):
        print(msg)


def solve(board, chips_pool, args):
    logger = ConsoleLogger()
    solver = ChipSolver(board, chips_pool,
                        up=args.upper,
                        lo=args.lower,
                        weights=args.weights,
                        solver=args.solver)
    logger.log(f'Board Size: {board.size()}')
    selected_chips, fval, total_attrs = solver.solve()
    logger.log(f'*fval = {fval}')
    logger.log('Optimal Chips:')
    for chip in selected_chips:
        attrs = ', '.join([f'{x:3d}' for x in chip.attrs])
        logger.log(f'|- ID:{chip.ID:>3d} Type-{chip.type_id:0>2d}   [{attrs}]')
    total_attrs = ', '.join([f'{x:3.0f}' for x in total_attrs])
    logger.log(f'|- Total     [{total_attrs}]')

    fig = solver.show()
    fig.savefig('solution.png')


def load_chips_from_csv(fname):
    df = pd.read_csv(fname)
    df['type'] = df['type'].fillna(method='ffill').astype(int)
    # df['star'] = df['star'].fillna(method='ffill').astype(int)
    df = df.fillna(0)
    return df


def get_chip_pool_from_df(df):
    chips_pool = []
    ChipInfo = namedtuple('ChipInfo', 'ID type_id attrs')
    for i, (type_id, pw, sa, pr, rl, level, *_) in enumerate(df.itertuples(False, False)):
        chips_pool.append(ChipInfo(i, type_id, np.array([pw, sa, pr, rl])))
    return chips_pool


def main():
    parser = argparse.ArgumentParser(description='ChipSolver v0.1')
    parser.add_argument('-f', '--file', help='the csv file of chips')
    parser.add_argument('-p', '--preset', help='the preset')
    parser.add_argument('-w', '--weights', help='the weights of four attributes',
                        nargs=4, type=int, metavar=('w1', 'w2', 'w3', 'w4'))
    parser.add_argument('-u', '--upper', help='the upper bound of the board',
                        nargs=4, type=int, metavar=('u1', 'u2', 'u3', 'u4'))
    parser.add_argument('-l', '--lower', help='the lower bound of the plan',
                        nargs=4, type=int, metavar=('l1', 'l2', 'l3', 'l4'))
    parser.add_argument('-s', '--solver', help='the solver to use')
    parser.add_argument('-c', '--showchips', help='show all types of chips',
                        action='store_true', default=False)

    args = parser.parse_args()

    if args.showchips:
        fig = show_all_chips()
        fig.savefig('chip_type.pdf')
    else:
        df = load_chips_from_csv(args.file)
        chips_pool = get_chip_pool_from_df(df)
        board = load_board(args.preset)

        solve(board, chips_pool, args)


if __name__ == '__main__':
    main()
