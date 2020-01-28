# Global imports

# Local imports
from ai_player import AIPlayer
from beck.beck_display import BeckDisplay
from beck.beck_game import BeckGame
from beck.beck_model import RandomBeckModel
from mcts import MCTS
from stage import Stage

game = BeckGame(m=4, n=9, k=4)
mcts_args = {
    'parallel_threads': 1,
    'cpuct': 1,
    'mcts_iterations': 20
}

model1, model2 = RandomBeckModel(game), RandomBeckModel(game)
mcts1, mcts2 = MCTS(game, model1, mcts_args), MCTS(game, model1, mcts_args)
players = [AIPlayer(mcts1, True), AIPlayer(mcts2, True)]

display = BeckDisplay(game, ['Rebecca', 'Rebecca'])
stage = Stage(players, game, display)

stage.execute()

import time
time.sleep(5)