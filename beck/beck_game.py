import sys
sys.path.append('..')

# Global imports
from scipy.ndimage import convolve
import numpy as np

# Local imports
from game import Game

class Beck(Game):
    def __init__(self, m=4, n=9, k=4):
        self.m, self.n, self.k = m, n, k
        self.valid_wins = [
            np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]),
            np.array([
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0]
            ]),
            np.array([[1, 1, 1, 1]]),
            np.array([[1],
                    [1],
                    [1],
                    [1]])
        ]
        self.players = [1, 2]

    def get_initial_state(self):
        return np.zeros((self.m, self.n))

    def get_next_state(self, state, player, action):
        state[action // self.m, action % self.m] = player
        return state, 2 - (player%2)

    def get_actions_size(self):
        return self.m * self.n

    def get_allowed_actions(self, state, player):
        return (state == 0).ravel()
        # allowed_coordinates = np.argwhere(state == 0)
        # return [x[0] * self.m + x[1] for x in allowed_coordinates]

    def get_is_terminal_state(self, state, player):
        if (state != 0).sum() == self.m * self.n:
            return True

        is_terminal_state = False
        for p in self.players:
            filtered_board = (state == p).astype(int)
            for win in self.valid_wins:
                if self.k in convolve(
                    filtered_board,
                    win,
                    mode='constant',
                    cval=0.0,
                    origin=0
                ):
                    is_terminal_state = True
                    break
            if is_terminal_state:
                break

        return is_terminal_state

    def get_result(self, state, player):
        winner = None

        for p in self.players:
            filtered_board = (state == p).astype(int)
            for win in self.valid_wins:
                if self.k in convolve(
                    filtered_board,
                    win,
                    mode='constant',
                    cval=0.0,
                    origin=0
                ):
                    winner = p
                    break
            if winner is not None:
                break

        if winner is not None:
            return 1 if winner == player else -1

        if (state != 0).sum() == self.m * self.n:
            return 0

        return None

    def get_canonical_form(self, state, player):
        if player == 2:
            ones, twos = state == 1, state == 2
            state[ones] = 2
            state[twos] = 1
        return state

    def get_symmetries(self, state, pi):
        pi_board = np.reshape(pi, (self.m, self.n))
        return [
            (state, pi),
            (np.rot90(state, 2), np.rot90(pi_board, 2).ravel()),
            (np.flip(state, axis=0), np.flip(pi_board, axis=0).ravel()),
            (np.flip(state, axis=1), np.flip(pi_board, axis=1).ravel())
        ]

    def get_hash_of_state(self, state):
        return hash(state.tostring)
