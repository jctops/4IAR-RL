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


    def execute_episode_with_example_storing(self):
        state = self.game.get_initial_state()
        current_player = 1
        turn_count = 0
        training_examples = []

        while not self.game.get_is_terminal_state(state):
            canonical_state = self.game.get_canonical_form(state, current_player)
            if turn_count >= self.selfplay_args['DETERMINISTIC_PLAY']:
                probs = self.searches[current_player - 1].get_probs(canonical_state, temperature=0)
            else:
                probs = self.searches[current_player - 1].get_probs(canonical_state, temperature=1)
            symmetries = self.game.get_symmetries(canonical_state, probs)
            for s, p in symmetries:
                training_examples.append([s, p, current_player])
            action = np.argmax(probs)
            previous_player = current_player
            state, current_player = self.game.get_next_state(state, current_player, action)

        result = self.game.get_result(state, current_player)
        training_examples = [(x[0], x[1], result*((-1)**x[2] != previous_player)) for x in training_examples]
        
        return training_examples

    def generate_training_examples(self):
        training_examples = []
        for _ in range(self.selfplay_args['GAMES_PER_SUBMISSION']):
            training_examples += execute_episode_with_example_storing()

        return training_examples

    def run(self):
        np.random.seed()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)#'{}'.format()

        self.model = self.lazy_model_class(*self.model_args)
        self.searches = [MCTS(self.game, self.model, self.mcts_args) for _ in range(len(self.game.players))]

