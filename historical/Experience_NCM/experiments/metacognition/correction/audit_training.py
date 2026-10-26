"""Describe correction labels on the frozen TRAINING groups only."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import CELL_POS, prompt_cells
from run_pilot import ROOT, file_hash, load_boards, write_json


def main():
    directory = ROOT / 'audit/metacognition/train-v0.2'
    report = json.loads((directory / 'report.json').read_text())
    split = json.loads((ROOT / 'audit/metacognition/split-v0.2.json').read_text())
    solutions = load_boards(ROOT / 'data/sudoku/hard/solutions.pickle')
    selected = split['partition_indices']['train']
    solutions = {i: np.asarray(solutions[i])[CELL_POS] for i in selected}
    records = []
    for batch in report['batches']:
        path = directory / batch['file']
        assert file_hash(path) == batch['sha256']
        with np.load(path, allow_pickle=False) as trace:
            blank = prompt_cells(trace['prefix']) == 0
            gold = np.stack([solutions[i] for i in trace['sample_indices']])
            now = trace['decoded_ids'][:, 8, CELL_POS]
            final = trace['decoded_ids'][:, 11, CELL_POS]
            errors = blank & (now != gold)
            corrected = errors & (final == gold)
            persistent = errors & (final != gold)
            new_errors = blank & (now == gold) & (final != gold)
            for j, index in enumerate(trace['sample_indices']):
                records.append(dict(index=int(index), current_errors=int(errors[j].sum()),
                    corrected=int(corrected[j].sum()), persistent=int(persistent[j].sum()),
                    new_final_errors=int(new_errors[j].sum()), final_success=bool(not trace['failure'][j])))
    assert [r['index'] for r in records] == selected
    result = dict(status='complete', scope='training_only_no_model_fit_no_held_out_read', primary_level=9,
        n=1000, extraction_report_sha256=file_hash(directory / 'report.json'), records=records,
        totals={key: sum(r[key] for r in records) for key in ['current_errors','corrected','persistent','new_final_errors']},
        groups_with_current_errors=sum(r['current_errors']>0 for r in records),
        mixed_outcome_groups=sum(r['corrected']>0 and r['persistent']>0 for r in records),
        all_current_errors_corrected_groups=sum(r['current_errors']>0 and r['persistent']==0 for r in records),
        no_current_error_corrected_groups=sum(r['current_errors']>0 and r['corrected']==0 for r in records))
    out = ROOT / 'audit/metacognition/correction-v0.1'
    out.mkdir(exist_ok=True)
    write_json(out / 'training-label-audit.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))

if __name__ == '__main__': main()
