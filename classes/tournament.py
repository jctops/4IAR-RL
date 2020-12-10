import datetime
import numpy as np
import os

participants_dir = '/scratch/jt3974/tournaments/tournament_2'
results_dir = '/scratch/jt3974/tournaments/results/tournament_2'

participants = sorted([x[12:] for x in os.listdir(participants_dir)])

iters = {p: sorted(list(set([
        int(x.split('.')[0].split('_')[1]) \
        for x in os.listdir(os.path.join(participants_dir, 'checkpoints_' + p)) \
        if x.startswith('checkpoint') and x != 'checkpoint' and not x.endswith('examples')
    ]))) for p in participants
}

print(iters)
# print(datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(participants_dir, participants[0], f'checkpoint_{35}.pth.tar.index'))))

import pandas as pd

from arena import Arena
from beck.beck_game import BeckGame as Game
from beck.beck_nnet import NNetWrapper as nn
from beck.beck_players import HumanBeckPlayer, RandomPlayer, GreedyBeckPlayer
from mcts import MCTS
from utils import *

def get_args(participants_dir, participant, ix, mcts_sims, cpuct):
    return dotdict({
        'tempThreshold': 15,
        'numMCTSSims': mcts_sims,
        'cpuct': cpuct,
        'load_model': True,
        'load_folder_file': (os.path.join(participants_dir, 'checkpoints_' + participant), f'checkpoint_{ix}.pth.tar'),
    })

def get_player(game, participants_dir, participant_iter):
    if participant_iter == 'human':
        player = HumanBeckPlayer(game)
        return lambda x: player.play(x)
    elif participant_iter == 'random':
        player = RandomPlayer(game)
        return lambda x: player.play(x)
    elif participant_iter == 'greedy':
        player = GreedyBeckPlayer(game)
        return lambda x: player.play(x)
    else:
        participant, ix = participant_iter.split(';')
        mcts_sims = int(participant.split('_')[0][4:])
        cpuct = int(participant.split('_')[1][5:])
        args = get_args(participants_dir, participant, ix, mcts_sims, cpuct)
        nnet = nn(game)
        if args.load_model:
            # log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        nmcts = MCTS(game, nnet, args)
        return lambda x: np.argmax(nmcts.getActionProb(x, temp=0))

participant_iters = [f'{k};{x}' for k, v in iters.items() for x in v] + ['random', 'greedy', 'human']
print(f'{len(participant_iters)} participant iterations!')


def play_game(game, participants_dir, participant_iter_1, participant_iter_2):
    print('\n')
    print(f'Game beginning! {participant_iter_1} v/s {participant_iter_2}...')
    player1 = get_player(game, participants_dir, participant_iter_1)
    player2 = get_player(game, participants_dir, participant_iter_2)
    show_game = (participant_iter_1 == 'human') or (participant_iter_2 == 'human')
    display = game.display if show_game else None
    arena = Arena(player1, player2, game, display=display)
    return arena.playGame(verbose=show_game)

def play_human_games():
    results_df = pd.DataFrame(index=participant_iters, columns=participant_iters)
    results_name = 'vs_human'
    human_vs_iters_black = [z for y in {k: [f'{k};{x}' for x in v if (x % 2 == 1) or (x == v[-1])] for k, v in iters.items()}.values() for z in y] + ['random', 'greedy']
    human_vs_iters_white = [z for y in {k: [f'{k};{x}' for x in v if (x % 2 == 0) or (x == v[-1])] for k, v in iters.items()}.values() for z in y] + ['random', 'greedy']

    g = Game(4,9,4)
    for opponent in human_vs_iters_black:
        results_df.loc[opponent, 'human'] = play_game(g, participants_dir, opponent, 'human')
        results_df.to_csv(os.path.join(results_dir, results_name + '.csv'))
    for opponent in human_vs_iters_white:
        results_df.loc['human', opponent] = play_game(g, participants_dir, 'human', opponent)
        results_df.to_csv(os.path.join(results_dir, results_name + '.csv'))

    print('Done!')

def printl(*args, flush=True, **kwargs):
    time_str = f'[{datetime.datetime.today()}]'
    print(time_str, flush=flush, *args, **kwargs)

def play_ai_round_robin():
    printl('Starting round robin!')
    results_name = 'round_robin'
    # results_df = pd.DataFrame(index=participant_iters, columns=participant_iters)
    try:
        results_df = pd.read_csv(os.path.join(results_dir, results_name + '.csv'), index_col=0)
    except:
        results_df = pd.DataFrame(index=participant_iters, columns=participant_iters)
    
    g = Game(4,9,4)
    for p1 in participant_iters:
        for p2 in participant_iters:
            if (p1 == p2) or (p1 == 'human') or (p2 == 'human') or (not pd.isnull(results_df.loc[p1, p2])):
                printl(f'Skipping {p1} v/s {p2}...')
            else:
                printl(f'{p1} v/s {p2}!')
                results_df.loc[p1, p2] = play_game(g, participants_dir, p1, p2)
                results_df.to_csv(os.path.join(results_dir, results_name + '.csv'))


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--play_as_human', action='store_true')
parser.add_argument('--play_ai_round_robin', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    if args.play_as_human:
        play_human_games()
    if args.play_ai_round_robin:
        play_ai_round_robin()



import multiprocessing as mp