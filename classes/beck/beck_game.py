from __future__ import print_function
import sys
sys.path.append('..')
from game import Game
# from beck.beck_logic import Board
import numpy as np
from scipy.ndimage import convolve

class BeckGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    @staticmethod
    def getSquarePiece(piece):
        return BeckGame.square_content[piece]

    def __init__(self, m, n, k):
        assert k <= m and k <= n, "n-in-a-row must fit on the board all four ways!"
        self.m = m
        self.n = n
        self.k = k
        self.valid_wins = [
            np.identity(k),
            np.rot90(np.identity(k)),
            np.ones((1,k)),
            np.ones((k,1))
        ]

    def getInitBoard(self):
        # return initial board (numpy board)
        pieces = [None]*self.m
        for i in range(self.m):
            pieces[i] = [0]*self.n
        return np.array(pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.m, self.n)

    def getActionSize(self):
        # return number of actions
        return self.m*self.n

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        # if action == self.n*self.n:    < --- no passing
        #     return (board, -player)
        x, y = (int(action/self.n), action%self.n)
        new_board = np.copy(board)
        new_board[x][y] = player
        return (new_board, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = (board == 0).flatten().astype(int)
        return np.array(valids)

    def getGameEnded(self, board, player):        
        """
        Input:
            board: current board
            player: current player (1 or -1)
        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        for p in [player, -player]:
            filtered_board = (board == p).astype(int)
            for win in self.valid_wins:
                if self.k in convolve(
                    filtered_board,
                    win,
                    mode='constant',
                    cval=0.0,
                    origin=0
                ):
                    return (p * player)

        if (board != 0).sum() == self.m * self.n:
            return 0.0001
        
        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.m*self.n)  # 1 for pass
        pi_board = np.reshape(pi, (self.m, self.n))
        return [
            (board, pi),
            (np.rot90(board, 2), np.rot90(pi_board, 2).ravel()),
            (np.flip(board, axis=0), np.flip(pi_board, axis=0).ravel()),
            (np.flip(board, axis=1), np.flip(pi_board, axis=1).ravel())
        ]

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    @staticmethod
    def display(board):
        m = board.shape[0]
        n = board.shape[1]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(m):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(BeckGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")