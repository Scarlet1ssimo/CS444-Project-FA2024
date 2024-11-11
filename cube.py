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
# Move X is equivalent to state[FAST_MAPL[X]] = state[FAST_MAPR[X][*]], where * is turn type
# FAST_MAPL[*]: the indexes of the cubies that will be moved
# FAST_MAPR[*][0]: clockwise turn
# FAST_MAPR[*][1]: double turn
# FAST_MAPR[*][2]: counter-clockwise turn
OP2NUM = {'U': 0, 'D': 1, 'R': 2, 'L': 3, 'F': 4, 'B': 5}
for move in ['U', 'D', 'R', 'L', 'F', 'B']:
    op = OP2NUM[move]
    cubeidx = np.arange(54)
    FAST_MAPL[op, :] = MOVE_MAP[move][0]
    for times in [1, 2, 3]:
        cubeidx[MOVE_MAP[move][0]] = cubeidx[MOVE_MAP[move][1]]
        FAST_MAPR[op][times - 1] = cubeidx[MOVE_MAP[move][0]]
    assert (FAST_MAPR[op][0] == MOVE_MAP[move][1]).all()


class QTM:
    """
    Character based representation of QTM (Quarter Turn Metric) cube.
    Allowed moves:
    U   D   R,  L,  F   B
    U'  D'  R'  L'  F'  B'
    God's number is 26 in QTM.
    """
    MOVES = ["U", "D", "R", "L", "F", "B",
             "U'", "D'", "R'", "L'", "F'", "B'"]

    def isSameFace(a, b):
        return a[0] == b[0]

    def isCancel(a, b):
        return a == b+"'" or a+"'" == b

    def isParallel(a, b):
        return (a[0] in ['U', 'D'] and b[0] in ['U', 'D']) or \
            (a[0] in ['R', 'L'] and b[0] in ['R', 'L']) or \
            (a[0] in ['F', 'B'] and b[0] in ['F', 'B'])

    def isOpposite(a, b):
        return QTM.isParallel(a, b) and not QTM.isSameFace(a, b)

    def getCancel(a):
        return a[0] if a[-1] == "'" else a+"'"

    def move(state, move):
        """
        Accept a move produced by this mode and apply it to the state.
        """
        QTM.apply(state, move)

    def apply(state, move: str):
        """
        Accept a string of move and apply it to the state.
        """
        move_face = OP2NUM[move[0]]
        if "'" in move:
            turn_type = 2
        elif "2" in move:
            turn_type = 1
        else:
            turn_type = 0
        state[FAST_MAPL[move_face]] = state[FAST_MAPR[move_face][turn_type]]

    MOVES_NO_CANCEL = {}
    for i in MOVES:
        MOVES_NO_CANCEL[i] = []
        for j in MOVES:
            if not isCancel(i, j):  # cannot cancel out
                MOVES_NO_CANCEL[i].append(j)


class QTMIX(QTM):
    """
    Index based representation of QTM (Quarter Turn Metric) cube.
    U   D   R   L   F   B
    U'  D'  R'  L'  F'  B'
    Numbered from 0 to 11
    """
    def isSameFace(a, b):
        return a % 6 == b % 6

    def isCancel(a, b):
        return a == b+6 or a+6 == b

    def isParallel(a, b):
        return (a//2) % 3 == (b//2) % 3

    def isOpposite(a, b):
        return QTMIX.isParallel(a, b) and not QTMIX.isSameFace(a, b)

    def getCancel(a):
        # [0,6)->[7,12)
        # [7,12)->[0,6)
        return a+6 if a < 6 else a-6
    MOVES = [i for i in range(12)]
    NUM2CHAR = {i: j for i, j in enumerate(QTM.MOVES)}
    CHAR2NUM = {j: i for i, j in enumerate(QTM.MOVES)}
    MOVES_NO_CANCEL = []
    for i in QTM.MOVES:
        MOVES_NO_CANCEL.append([])
        for j in QTM.MOVES:
            if not QTM.isCancel(i, j):  # cannot cancel out
                MOVES_NO_CANCEL[-1].append(CHAR2NUM[j])
    FAST_MAPR_IX = np.zeros((12, 20), dtype=int)
    for i in range(6):
        for j in range(2):
            for k in range(20):
                FAST_MAPR_IX[i+j*6][k] = FAST_MAPR[i][j*2][k]
    

    def move(state, moveIx: int):
        """
        Accept a move produced by this mode and apply it to the state.
        """
        move = moveIx % 6
        state[FAST_MAPL[move]] = state[QTMIX.FAST_MAPR_IX[moveIx]]


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

    Allowed moves in Half-Turn Metric (HTM):
        U,  D,  R,  L,  F,  B
        U2, D2, R2, L2, F2, B2
        U', D', R', L', F', B'
    Labeled from 0 to 17.
    God's number is 20 in HTM and 26 in QTM.
    """

    def __init__(self, mode=QTMIX):
        self.reset()
        if isinstance(mode, str):
            if mode == "QTM":
                self.mode = QTM
            elif mode == "QTMIX":
                self.mode = QTMIX
            else:
                raise ValueError("Invalid mode.")
            return
        self.mode = mode

    def get_state(self):
        return self.state

    def reset(self):
        self.state = np.arange(54)//9

    def is_solved(self):
        return (self.state == np.arange(54)//9).all()

    def move(self, move):
        self.mode.move(self.state, move)

    def apply(self, moves):
        if isinstance(moves, list):
            for i in moves:
                self.mode.apply(self.state, i)
        elif isinstance(moves, str):
            for i in moves.split():
                self.mode.apply(self.state, i)

    def scrambler(self, scramble_length=26):
        """
        In our scheme, we will not generate sequence like UDDU or UDDU'
        """
        mode = self.mode
        if mode in [QTMIX, QTM]:  # FIXME: Should we implement in mode class?
            while True:
                # Reset the cube state, scramble, and return cube state and scramble moves
                self.reset()
                scramble = []
                face = []
                for i in range(scramble_length):
                    if i > 1:
                        sec_last_move = scramble[-2]
                        last_move = scramble[-1]
                        while True:
                            move = random.choice(
                                mode.MOVES_NO_CANCEL[last_move])
                            # seq like UUDD or U'D' we don't allow any of UU' again
                            if len(face) > 1:
                                sec_last_face = face[-2]
                                last_face = face[-1]
                                if mode.isOpposite(sec_last_face, last_face) and mode.isSameFace(move, sec_last_face):
                                    continue

                            # seq like UU we don't allow U or U'
                            if move == last_move == sec_last_move:
                                continue
                            break
                        if not mode.isSameFace(move, last_move):
                            face.append(move)
                    elif i == 1:
                        last_move = scramble[-1]
                        move = random.choice(
                            mode.MOVES_NO_CANCEL[scramble[0]])
                        if not mode.isSameFace(move, last_move):
                            face.append(move)
                    else:
                        move = random.choice(mode.MOVES)
                        face.append(move)

                    mode.move(self.state, move)
                    scramble.append(move)
                    yield self.state, move
        else:
            # TODO: HTM is inherently more efficent than QTM. We will implement this later.
            raise NotImplementedError("HTM not supported.")


if __name__ == '__main__':
    # Testing turn mode
    for i in range(12):
        for j in range(12):
            ii = QTMIX.NUM2CHAR[i]
            jj = QTMIX.NUM2CHAR[j]
            assert QTM.isSameFace(ii, jj) == QTMIX.isSameFace(i, j)
            assert QTM.isCancel(ii, jj) == QTMIX.isCancel(i, j)
            assert QTM.isParallel(ii, jj) == QTMIX.isParallel(i, j)
            assert QTM.isOpposite(ii, jj) == QTMIX.isOpposite(i, j)

    # This should be able to test all valid moves.
    for i in range(6):
        for j in range(2):
            for k in range(20):
                print(QTMIX.FAST_MAPR_IX.shape)
                print(QTMIX.FAST_MAPR_IX[i+j*6][k])
                print(FAST_MAPR[i][j*2][k])
                assert QTMIX.FAST_MAPR_IX[i+j*6][k] == FAST_MAPR[i][j*2][k]

    cube = Cube(QTMIX)
    cube.apply("L R U D F B D' F' B' L' R' U' L2 R2 U2 D2 F2 B2")
    assert (cube.get_state() == np.array(
        [3, 5, 3, 4, 0, 5, 2, 4, 2, 5, 5, 4, 0, 1, 0, 5, 4, 4, 0, 4, 4, 3,
         2, 1, 0, 1, 5, 0, 5, 5, 2, 3, 1, 0, 1, 4, 3, 3, 1, 2, 4, 2, 2, 0,
         1, 2, 2, 1, 3, 5, 3, 3, 0, 1])).all()

    cube = Cube(QTMIX)
    S = cube.scrambler(26)
    s = []
    for i in range(26):
        _, move = next(S)
        s.append(move)
    # print(' '.join(s))
    for i in reversed(s):
        i = cube.mode.getCancel(i)
        cube.move(i)
    assert cube.is_solved()
