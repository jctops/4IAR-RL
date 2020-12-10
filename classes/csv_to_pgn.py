import os
import pandas as pd

### This script takes a CSV of tournament results and converts
### them to minimal chess notation so that BayesELO can calculate
### ELO ratings. 

participants_dir = '/scratch/jt3974/tournaments/tournament_1'
results_dir = '/scratch/jt3974/tournaments/results/tournament_1'

human = pd.read_csv(os.path.join(results_dir, 'vs_human.csv'), index_col=0)
round_robin = pd.read_csv(os.path.join(results_dir, 'round_robin.csv'), index_col=0)

results = human.fillna(0) + round_robin.fillna(0)

with open(os.path.join(results_dir, 'pgn.pgn'), 'w+') as f:
    print(os.path.join(results_dir, 'pgn.pgn'))
    for p in results.index:
        for q in results.columns:
            outcome = results.loc[p, q]
            if (outcome == 0):# or ('mcts80' in p) or ('mcts80' in q):
                continue
            elif outcome == 1:
                outcome = '1-0'
            elif outcome == -1:
                outcome = '0-1'
            else:
                outcome = '1/2-1/2'
            f.write(f'[White "{p}"][Black "{q}"][Result "{outcome}"] 1. c4 Nf6\n')