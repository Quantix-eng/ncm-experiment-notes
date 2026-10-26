"""Check solution multiplicity of official TRAINING puzzles, no held-out access."""
import json
from pathlib import Path
import time
from confirmation_corpus import exact_cover_count
import numpy as np
from features import prompt_cells
from run_pilot import ROOT,file_hash,load_boards,write_json


def main():
    split=json.loads((ROOT/'audit/metacognition/split-v0.2.json').read_text())
    puzzles=load_boards(ROOT/'data/sudoku/hard/puzzles.pickle')
    selected=split['partition_indices']['train'];start=time.perf_counter();counts={0:[],1:[],2:[]}
    for index in selected:
        cells=prompt_cells(np.array([puzzles[index]]))[0].tolist()
        counts[exact_cover_count(cells)].append(index)
    result=dict(status='complete',scope='official_training_only',n=len(selected),count_capped_at=2,
        groups_by_solution_count={str(k):v for k,v in counts.items()},counts={str(k):len(v)for k,v in counts.items()},
        source_sha256=file_hash(__file__),solver_sha256=file_hash(Path(__file__).with_name('confirmation_corpus.py')),
        wall_seconds=time.perf_counter()-start)
    write_json(ROOT/'audit/metacognition/correction-v0.1/training-uniqueness.json',result)
    print(json.dumps({k:v for k,v in result.items()if k!='groups_by_solution_count'}))

if __name__=='__main__':main()
