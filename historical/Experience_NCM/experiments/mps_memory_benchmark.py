"""Paired CPU/MPS timings with explicit Metal cache release included in time."""
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from ncm_mac.sudoku import SudokuModel, load_boards
from ncm_mac.metal_experiment import install_area_pool
from sudoku_mps import predict


def main():
    torch.set_num_threads(4)
    ck = ROOT / "artifacts/sudoku/medium"
    cpu, gpu = SudokuModel(ck), SudokuModel(ck)
    install_area_pool(gpu.vqvae.vq)
    gpu.vqvae.to("mps"); gpu.net.to("mps")
    boards = load_boards(ROOT / "data/sudoku/medium/puzzles.pickle")
    results = {}
    for batch in [8,32]:
        prefix = cpu.prefix(boards[:batch]); mp = prefix.to("mps")
        predict(cpu,prefix); predict(gpu,mp); torch.mps.synchronize(); torch.mps.empty_cache()
        times = dict(cpu=[],mps=[])
        memory = []
        for i in range(3):
            for device in (["cpu","mps"] if i%2==0 else ["mps","cpu"]):
                if device=="mps":torch.mps.synchronize()
                start=time.perf_counter()
                predict(cpu if device=="cpu" else gpu, prefix if device=="cpu" else mp)
                if device=="mps":
                    torch.mps.synchronize()
                    before=torch.mps.driver_allocated_memory()
                    torch.mps.empty_cache(); torch.mps.synchronize()
                    memory.append(dict(driver_before_clear=before,driver_after_clear=torch.mps.driver_allocated_memory()))
                times[device].append(time.perf_counter()-start)
        results[str(batch)] = dict(seconds=times,memory=memory,
            cpu_median=statistics.median(times["cpu"]),mps_median=statistics.median(times["mps"]))
        (ROOT / "audit/sudoku-mps-memory.json").write_text(json.dumps(results,indent=2)+"\n")
        print(batch,results[str(batch)],flush=True)


if __name__ == "__main__":main()
