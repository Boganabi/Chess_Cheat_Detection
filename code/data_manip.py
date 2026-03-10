
import pandas as pd
from chess import pgn
import io
import chess
import chess.engine
import numpy as np
import sys

class DataManip:

    def __init__(self):
        self.STOCKFISH_PATH = "stockfish.exe"
        np.set_printoptions(threshold=sys.maxsize) # so that csv output doesnt have ...

    def add_labels(self):
        df = pd.read_csv("../datasets/l_Games.csv")
        # df["white_cheat"] = df.apply(lambda x: 1 if x["Liste cheat white"] != "0" else 0, axis=1)
        # df["black_cheat"] = df.apply(lambda x: 1 if x["Liste cheat black"] != "0" else 0, axis=1)
        df["cheat_code"] = df.apply(lambda x: self.get_cheat_code(x[["Liste cheat white", "Liste cheat black"]]), axis=1)
        df.to_csv("../datasets/l_Games.csv", index=False)

    def get_cheat_code(self, cols) -> int:
        white, black = cols
        if(white != "0" and black != "0"):
            # both cheated, so return 3
            return 3
        if(white == "0" and black != "0"):
            # black cheated, return 2
            return 1
        if(white != "0" and black == "0"):
            # white cheated, return 1
            return 1
        # no one cheated, return 0
        return 0

    def encode_games(self):
        df = pd.read_csv("../datasets/l_Games.csv")
        # df[["matrix_game", "eval", "CPL"]] = df.apply(lambda x: self.get_matrix_from_game(x["Game"]), axis=1)
        df["matrix_game"] = df.apply(lambda x: self.get_matrix_from_game(x["Game"]), axis=1)
        df.to_csv("../datasets/l_Games.csv", index=False)
    
    def get_matrix_from_game(self, pgn_game: str):
        # converts a given PGN format game into a matrix, also returns centipawn loss and evaluation at each move
        # need to flatten the final game matrix so it can go into fit()
        matricies = []
        # evals = []
        # cpls = []
        # with chess.engine.SimpleEngine.popen_uci(self.STOCKFISH_PATH) as engine:
        game = pgn.read_game(io.StringIO(pgn_game))
        board = chess.Board()

        # starting position
        matricies.append(self._board_to_matrix(board))
        # info = engine.analyse(board, chess.engine.Limit(time=0.1)) # give it 0.1 seconds to analyze
        # score = info["score"].white().score()
        # evals.append(score)
        # cpls.append(0)

        # iterate through moves
        for move in game.mainline_moves():
            board.push(move)
            matricies.append(self._board_to_matrix(board))

            # get new engine info
            # info = engine.analyse(board, chess.engine.Limit(time=0.1)) # give it 0.1 seconds to analyze
            # new_score = info["score"].white().score()

            # if(new_score != None): # None means the game is over
            #     # new centipawn loss calculation
            #     cpls.append(score - new_score)
            #     score = new_score

        matrix_game = np.array(matricies)
        matrix_game = str(matrix_game.flatten()).replace(' ', '').replace('[', '').replace(']', '').replace('\n', '') # removes all spaces, brackets, and newlines
        return matrix_game#, evals, cpls

    def _board_to_matrix(self, board: chess.Board):
        # convert board position to an 8x8 matrix
        matrix = np.zeros((8,8), dtype=int)
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                matrix[i // 8, i % 8] = piece.piece_type * (1 if piece.color == chess.WHITE else 10)
        return matrix


if __name__ == "__main__":
    dm = DataManip()
    # dm.add_labels()
    dm.encode_games()