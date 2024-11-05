import numpy as np
import random

MOVE_MAP = {
    'U': (np.array([0,  3,  6,  7,  8,  5,  2,  1,  20, 23, 26, 47, 50, 53, 29, 32, 35, 38, 41, 44]),
          np.array([6,  7,  8,  5,  2,  1,  0,  3,  47, 50, 53, 29, 32, 35, 38, 41, 44, 20, 23, 26])),
    'D': (np.array([9,  10, 11, 14, 17, 16, 15, 12, 18, 21, 24, 45, 48, 51, 27, 30, 33, 36, 39, 42]),
          np.array([15, 12, 9,  10, 11, 14, 17, 16, 36, 39, 42, 18, 21, 24, 45, 48, 51, 27, 30, 33])),
    'R': (np.array([29, 32, 35, 34, 33, 30, 27, 28, 15, 16, 17, 51, 52, 53, 6,  7,  8,  38, 37, 36]),
          np.array([27, 28, 29, 32, 35, 34, 33, 30, 38, 37, 36, 15, 16, 17, 51, 52, 53, 6,  7,  8])),
    'L': (np.array([26, 25, 24, 21, 18, 19, 20, 23, 2, 1, 0, 47, 46, 45, 11, 10, 9, 42, 43, 44]),
          np.array([20, 23, 26, 25, 24, 21, 18, 19, 42, 43, 44, 2, 1, 0, 47, 46, 45, 11, 10, 9])),
    'F': (np.array([47, 50, 53, 52, 51, 48, 45, 46, 0, 3, 6, 29, 28, 27, 17, 14, 11, 24, 25, 26]),
          np.array([45, 46, 47, 50, 53, 52, 51, 48, 24, 25, 26, 0, 3, 6, 29, 28, 27, 17, 14, 11])),
    'B': (np.array([38, 41, 44, 43, 42, 39, 36, 37, 2, 5, 8, 35, 34, 33, 15, 12, 9, 18, 19, 20]),
          np.array([36, 37, 38, 41, 44, 43, 42, 39, 35, 34, 33, 15, 12, 9, 18, 19, 20, 2, 5, 8])),
}
FAST_MAPR = np.zeros((6, 3, 20), dtype=int)
FAST_MAPL = np.zeros((6, 20), dtype=int)
OP2NUM = {'U': 0, 'D': 1, 'R': 2, 'L': 3, 'F': 4, 'B': 5}
for move in ['U', 'D', 'R', 'L', 'F', 'B']:
    op = OP2NUM[move]
    cubeidx = np.arange(54)
    FAST_MAPL[op, :] = MOVE_MAP[move][0]
    for times in [1, 2, 3]:
        cubeidx[MOVE_MAP[move][0]] = cubeidx[MOVE_MAP[move][1]]
        FAST_MAPR[op][times - 1] = cubeidx[MOVE_MAP[move][0]]
    assert (FAST_MAPR[op][0] == MOVE_MAP[move][1]).all()


class Cube:
    """
    A class representing a cube state.
    Initial color:

            0 0 0
            0 Y 0
            0 0 0

    2 2 2   5 5 5   3 3 3  4 4 4
    2 B 2   5 R 5   3 G 3  4 O 4
    2 2 2   5 5 5   3 3 3  4 4 4

            1 1 1
            1 W 1
            1 1 1

    Exact Indexes

              2  5  8
              1  4  7
              0  3  6
    20 23 26  47 50 53  29 32 35  38 41 44
    19 22 25  46 49 52  28 31 34  37 40 43
    18 21 24  45 48 51  27 30 33  36 39 42
              11 14 17
              10 13 16
              9  12 15

    Allowed moves (Half-Turn Metric, HTM):
        L,  R,  U,  D,  F,  B
        L', R', U', D', F', B'
        L2, R2, U2, D2, F2, B2
    """

    def __init__(self):
        self.reset()

    def get_state(self):
        return self.state

    def reset(self):
        self.state = np.arange(54)//9

    def is_solved(self):
        return (self.state == np.arange(54)//9).all()

    def __move(self, move, times=1):
        for _ in range(times):
            self.state[MOVE_MAP[move][0]] = self.state[MOVE_MAP[move][1]]

    def apply(self, moves):
        moves = moves.split()
        for i in moves:
            if "'" in i:
                self.__move(i[0], 3)
            elif "2" in i:
                self.__move(i[0], 2)
            else:
                self.__move(i[0])


class VecCube(Cube):
    def apply(self, moves):
        moves = moves.split()
        for i in moves:
            move = OP2NUM[i[0]]
            if "'" in i:
                times = 3
            elif "2" in i:
                times = 2
            else:
                times = 1
            self.state[FAST_MAPL[move]] = self.state[FAST_MAPR[move][times-1]]


if __name__ == '__main__':
    # This should be able to test all valid moves.
    # cube = Cube()
    # cube.apply("L R U D F B D' F' B' L' R' U' L2 R2 U2 D2 F2 B2")
    cube = VecCube()
    cube.apply("L R U D F B D' F' B' L' R' U' L2 R2 U2 D2 F2 B2")
    assert (cube.get_state() == np.array(
        [3, 5, 3, 4, 0, 5, 2, 4, 2, 5, 5, 4, 0, 1, 0, 5, 4, 4, 0, 4, 4, 3,
         2, 1, 0, 1, 5, 0, 5, 5, 2, 3, 1, 0, 1, 4, 3, 3, 1, 2, 4, 2, 2, 0,
         1, 2, 2, 1, 3, 5, 3, 3, 0, 1])).all()
