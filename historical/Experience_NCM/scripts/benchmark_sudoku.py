"""Small CPU sweep; records bitwise differences before selecting a runtime."""
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from ncm_mac.sudoku import SudokuModel, load_boards


def main():
    torch.set_num_threads(4)
    model = SudokuModel(ROOT / "artifacts/sudoku/easy")
    boards = load_boards(ROOT / "data/sudoku/easy/puzzles.pickle")
    results = []
    for batch in [1, 8, 32]:
        prefix = model.prefix(boards[:batch])
        for threads in [4, 1, 2]:
            torch.set_num_threads(threads)
            original = model.predict(prefix, optimized=False)
            optimized = model.predict(prefix)
            equal = all(torch.equal(x, y) for x, y in [(original.codes, optimized.codes),
                         (original.reconstruction, optimized.reconstruction), (original.logits, optimized.logits)])
            timings = {"original": [], "optimized": []}
            for rep in range(3):
                for name in (["original", "optimized"] if rep % 2 == 0 else ["optimized", "original"]):
                    t = time.perf_counter()
                    model.predict(prefix, optimized=name == "optimized")
                    timings[name].append(time.perf_counter() - t)
            row = dict(batch_size=batch, threads=threads, bitwise_equal=equal, seconds=timings,
                       original_median=statistics.median(timings["original"]),
                       optimized_median=statistics.median(timings["optimized"]))
            row["boards_per_second"] = batch / row["optimized_median"]
            results.append(row)
            print(json.dumps(row), flush=True)
            (ROOT / "audit/sudoku-cpu-sweep.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
