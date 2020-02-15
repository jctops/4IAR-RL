import multiprocessing as mp
import numpy as np
import os

from ai_player import AIPlayer
from mcts import MCTS

class SelfplayProcess(mp.Process):
    def __init__(self, gpu_id, game, lazy_model_class, model_args, mcts_args, selfplay_args):
        mp.Process.__init__(self)
        self.gpu_id = gpu_id
        self.game = game
        self.lazy_model_class = lazy_model_class
        self.model_args = model_args
        self.mcts_args = mcts_args
        self.selfplay_args = selfplay_args

    def update_model_weights(self, weights):
        pass

    def generate_training_examples(self):
        state = self.game.get_initial_state()
        current_player = 1
        turn_count = 0

        while not self.game.get_is_terminal_state(state):
            canonical_state = self.game.get_canonical_form(state, current_player)
            if turn_count >= self.selfplay_args['DETERMINISTIC_PLAY']:
                probs = self.search.get_probs(canonical_state, temperature=0)
            else:
                probs = self.search.get_probs(canonical_state, temperature=1)
            action = np.argmax(probs)
            state, current_player = self.game.get_next_state(state, current_player, action)

        result = self.game.get_result(state, current_player)
        return result


    def run(self):
        np.random.seed()
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(self.gpu_id)

        self.model = self.lazy_model_class(*self.model_args)
        self.search = MCTS(self.game, self.model, self.mcts_args)

