import sys
sys.path.append('..')

# Global imports
import numpy as np

# Local imports
from model import Model

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