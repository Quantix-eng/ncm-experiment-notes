"""Reject a prefix-cache candidate immediately if it changes any bit."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from ncm_mac.sudoku import SudokuModel, load_boards
from ncm_mac.cached import generate_cached

torch.set_num_threads(4)
report = []
for difficulty in ["easy", "medium", "hard"]:
    model = SudokuModel(ROOT / "artifacts/sudoku" / difficulty)
    boards = load_boards(ROOT / "data/sudoku" / difficulty / "puzzles.pickle")
    prefix = model.prefix(boards[:8])
    base = model.predict(prefix)
    start = time.perf_counter()
    codes, z = generate_cached(model.net, prefix)
    with torch.inference_mode():
        logits = model.vqvae.decoder(z)
    row = dict(difficulty=difficulty, codes_equal=torch.equal(base.codes, codes),
               reconstruction_equal=torch.equal(base.reconstruction, z), logits_equal=torch.equal(base.logits, logits),
               answer_mismatches=int((base.ids != logits.argmax(-1)).any(-1).sum()),
               max_abs_logit_error=(base.logits - logits).abs().max().item(),
               cached_seconds=time.perf_counter() - start)
    report.append(row)
    print(json.dumps(row), flush=True)
    (ROOT / "audit/sudoku-cache-probe.json").write_text(json.dumps(report, indent=2) + "\n")
