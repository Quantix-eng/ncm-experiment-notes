"""Separate interpolation rounding from other MPS backend differences."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from ncm_mac.metal_experiment import AreaPool1d, install_area_pool
from ncm_mac.sudoku import SudokuModel, load_boards
from sudoku_mps import capture, distribution_stats


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    report = dict(pool_random_mps=[], cpu_adapter_control={})
    x = torch.randn(8, 256, 89)
    mx = x.to("mps")
    for out in [2,4,6,9,13,18,26,37,50,67,89]:
        actual = AreaPool1d(89, out).to("mps")(mx).cpu()
        expected = torch.nn.functional.interpolate(x, size=out, mode="area")
        report["pool_random_mps"].append(dict(output_length=out,
            max_abs_error=(actual-expected).abs().max().item()))
    for difficulty in ["easy","medium","hard"]:
        ck = ROOT / "artifacts/sudoku" / difficulty
        base, candidate = SudokuModel(ck), SudokuModel(ck)
        install_area_pool(candidate.vqvae.vq)
        boards = load_boards(ROOT / "data/sudoku" / difficulty / "puzzles.pickle")[:8]
        prefix = base.prefix(boards)
        bc, bl, bh = capture(base, prefix)
        cc, cl, _ = capture(candidate, prefix)
        _, _, ch = capture(candidate, prefix, bc)
        stats = [distribution_stats(x,y) for x,y in zip(bh,ch)]
        report["cpu_adapter_control"][difficulty] = dict(
            changed_code_boards=int((bc!=cc).any(-1).sum()),
            changed_answer_boards=int((bl.argmax(-1)!=cl.argmax(-1)).any(-1).sum()),
            generator_max_abs_probability=max(x["max_abs_probability"] for x in stats),
            decoder=distribution_stats(bl,cl))
    (ROOT / "audit/sudoku-mps-pooling-control.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
