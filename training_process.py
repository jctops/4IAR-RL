from collections import deque
import multiprocessing as mp
import numpy as np
import os

from memory import Memory
from stage import execute_episode

class TrainingProcess(mp.Process):
    def __init__(self, gpu_id, game, lazy_model_class, model_args, ingress, egress, kill_ingress, training_args):
        mp.Process.__init__(self)
        self.gpu_id = gpu_id
        self.game = game
        self.lazy_model_class = lazy_model_class
        self.model_args = model_args
        self.ingress = ingress
        self.egress = egress
        self.kill_ingress = kill_ingress
        self.training_args = training_args
        self.examples = Memory(self.training_args['MAX_MEMORY_SIZE'])

    def receive_examples(self):
        received_count = 0
        while not self.ingress.empty() and (received_count < self.training_args['MAX_EXAMPLES_PER_RECEIVE']):
            memories = self.ingress.get()
            self.examples.store_memories(memories)
            received_count += 1

    def output_weights(self):
        for _ in self.training_args['NUM_OF_SELFPLAY_PROCESSES']:
            self.egress.put("weights")

    def train(self):
        training_sample = self.examples.sample_memories(self.training_args['TRAINING_SAMPLE_SIZE'])
        self.model.train(training_sample)

    def eval(self):
        games_with_rookie_first = int(self.training_args['EVAL_GAMES'] / 2)
        games_with_champ_first = self.training_args['EVAL_GAMES'] - games_with_rookie_first
        wins, draws, losses = 0, 0, 0
        for _ in range(games_with_rookie_first):
            result = execute_episode([self.rookie, self.champ], self.game)
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1
        for _ in range(games_with_champ_first):
            result = execute_episode([self.rookie, self.champ], self.game)
            if result == -1:
                wins += 1
            elif result == 1:
                losses += 1
            else:
                draws += 1
        
        if wins + losses == 0 or wins / (wins + losses) < self.training_args['PROMOTION_THRESHOLD']:
            print('Rookie not promoted, get back to training')
        else:
            print('Rookie promoted!')

    def run(self):
        np.random.seed()
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(self.gpu_id)

        self.model = self.lazy_model_class(*self.model_args)

        self.champ = self.model
        self.rookie = self.model

        can_train = False

        while True:
            for _ in range(self.training_args['TRAINING_ROUNDS_PER_EVAL']):
                if not self.kill_ingress.empty():
                    break
                self.receive_examples()
                if can_train:
                    self.train()
            if can_train:
                self.eval()

        # exit