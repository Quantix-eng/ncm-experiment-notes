"""Cell-local observables bounded to level9. No solution or future labels."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import CELL_POS, GROUPS, observed_counts, probability_features, prompt_cells

UNITS = np.array([[GROUPS[r], GROUPS[9+c], GROUPS[18+(r//3)*3+c//3]] for r in range(9) for c in range(9)])
PEERS = np.array([[ [j for j in unit if j != i] for unit in UNITS[i]] for i in range(81)])


def visible_features(prefix, trace, raw_logits, level=9):
    given = prompt_cells(prefix)
    b = len(given)
    coords = np.array([[i//9, i%9, (i//27)*3+(i%9)//3] for i in range(81)])
    position = np.concatenate([np.eye(9)[coords[:,j]] for j in range(3)], axis=1)
    parts = [np.broadcast_to(position, (b,81,27))]
    previous = None
    for k in range(level):
        probs, ids = trace['decoder_probs'][:,k,CELL_POS], trace['decoded_ids'][:,k,CELL_POS]
        cells = ids[:,:,None,None]
        same = ((ids[:,PEERS] == cells) & (cells>=1) & (cells<=9)).sum(-1)/8
        changed = np.zeros((b,81)) if previous is None else (ids != previous).astype(float)
        parts += [probs, np.eye(12)[ids], ((ids<1)|(ids>9))[...,None].astype(float), same, changed[...,None]]
        previous = ids
    clues = (given[:,UNITS] != 0).sum(-1)/9
    parts.append(clues)
    current = trace['decoder_probs'][:,level-1,CELL_POS]
    neighborhood = current[:,PEERS]
    parts += [neighborhood.mean(3).reshape(b,81,-1), neighborhood.std(3).reshape(b,81,-1), raw_logits[:,CELL_POS]/10]
    entropy, margin, maximum = probability_features(current, 12)
    blank = given == 0
    global_parts = []
    for values in (entropy, margin, maximum):
        selected = values[blank].reshape(b,51)
        global_parts += [selected.mean(1), selected.std(1)]
    invalid, separators, clues_bad, pairs, valid = observed_counts(trace['decoded_ids'][:,level-1],given)
    global_features = np.column_stack([*global_parts,invalid/81,separators/8,clues_bad/30,pairs/324,valid])
    parts.append(np.broadcast_to(global_features[:,None,:],(b,81,global_features.shape[1])))
    result = np.concatenate(parts,axis=2)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite or future-contaminated cell features')
    return result
