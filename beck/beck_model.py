import sys
sys.path.append('..')

# Global imports
import numpy as np

# Local imports
from model import Model
from beck.nnet_construction import build_residual_cnn

class RandomBeckModel(Model):
    def __init__(self, game):
        self.game = game

    def train(self, examples):
        pass

    def predict(self, state):
        p = np.random.random(36)
        p[~self.game.get_allowed_actions(state)] = 0
        return p / p.sum(), np.random.random()

    def predict_batch(self, state):
        pass

    def save_checkpoint(self, folder, filename):
        pass

    def load_checkpoint(self, folder, filename):
        pass


def sigmoid(y):
    return 1.0 / (1.0 + np.exp(-y))

class NnetBeckModel(Model):
    def __init__(self, game, nnet_args):
        self.game = game
        self.nnet = build_residual_cnn(nnet_args)

    def train(self, examples):
        pass

    def predict(self, state):
        p, v = self.eval_net(state)
        return p, v[0]

    def predict_batch(self, state):
        pass    # TODO

    def save_checkpoint(self, folder, filename):
        pass    # TODO

    def load_checkpoint(self, folder, filename):
        pass    # TODO

    @staticmethod
    def state_to_nnet_input(state):
        nnet_input = np.zeros((1, 3, *state.shape))
        nnet_input[0][0] = (state == 1)
        nnet_input[0][1] = (state == 2)
        is_player_ones_turn = 1 - np.sum(state != 0)
        nnet_input[0][2] = np.full(state.shape, is_player_ones_turn)
        return nnet_input

    @staticmethod
    def states_to_nnet_input(states):
        nnet_input = np.zeros((len(states), 3, *states[0].board.shape))
        for i in range(len(states)):
            state = states[i]
            nnet_input[i][0] = (state == 1)
            nnet_input[i][1] = (state == 2)
            is_player_ones_turn = 1 - np.sum(state != 0)
            nnet_input[i][2] = np.full(state.shape, is_player_ones_turn)
        return nnet_input
    
    def get_policy_from_logits(self, logits, state):
        probs = sigmoid(logits)
        probs[~self.game.get_allowed_actions(state)] = 0
        probs = probs / np.sum(probs)
        return probs
    
    def eval_net(self, state):
        nnet_input = self.state_to_nnet_input(state)
        nnet_output = self.nnet.predict(nnet_input)
        value = nnet_output[0][0]
        policy = self.get_policy_from_logits(nnet_output[1][0], state)
        return policy, value 
    
    def get_policy(self, state):
        return self.eval_net(state)[0]
    
    def get_value(self, state):
        return self.eval_net(state)[1]