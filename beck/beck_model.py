import sys
sys.path.append('..')

# Local imports
from model import Model

class BeckModel(Model):
    def __init__(self, game):
        pass

    def train(self, examples):
        pass

    def predict(self, state):
        pass

    def predict_batch(self, state):
        pass

    def save_checkpoint(self, folder, filename):
        pass

    def load_checkpoint(self, folder, filename):
        pass