"""Isolated FP32 MPS study against CPU commit 254bb49, no fallback."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from ncm_mac.cached import generate_cached
from ncm_mac.sudoku import SudokuModel, load_boards, metrics, verify_assets
from ncm_mac.metal_experiment import AreaPool1d, install_area_pool


@torch.inference_mode()
def predict(model, prefix):
    codes, z = generate_cached(model.net, prefix)
    logits = model.vqvae.decoder(z)
    return codes, logits


def capture(model, prefix, forced_codes=None):
    outputs, handles = [], []
    offset = 0
    for k, length in enumerate(model.net.levels):
        forced = None if forced_codes is None else forced_codes[:, offset:offset + length]
        def hook(module, args, out, forced=forced):
            outputs.append(out.detach())
            if forced is not None:
                return torch.full_like(out, -torch.inf).scatter(-1, forced.unsqueeze(-1), 0.0)
        handles.append(model.net.lm_heads[k].register_forward_hook(hook))
        offset += length
    try:
        codes, logits = predict(model, prefix)
        return codes, logits, outputs
    finally:
        for handle in handles:
            handle.remove()


def distribution_stats(a, b):
    # Compare probabilities in CPU float64, so the measuring softmax adds no
    # extra CPU/MPS backend discrepancy. Inference itself remains FP32.
    a, b = a.cpu().double(), b.cpu().double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError("Non-finite reference or experimental logits")
    pa, pb = a.softmax(-1), b.softmax(-1)
    diff = (pa - pb).abs()
    kl = (pa * (a.log_softmax(-1) - b.log_softmax(-1))).sum(-1)
    return dict(max_abs_logit=float((a - b).abs().max()), max_abs_probability=float(diff.max()),
                mean_abs_probability=float(diff.mean()), max_total_variation=float(diff.sum(-1).max() / 2),
                max_kl_cpu_to_mps=float(kl.max().clamp_min(0)),
                argmax_disagreements=int((a.argmax(-1) != b.argmax(-1)).sum()),
                positions=a.shape[0] * a.shape[1])


def timed(model, prefix, device, repeats):
    sync = torch.mps.synchronize if device == "mps" else lambda: None
    for _ in range(2):
        predict(model, prefix)
    sync()
    times = []
    for _ in range(repeats):
        sync(); start = time.perf_counter()
        predict(model, prefix)
        sync(); times.append(time.perf_counter() - start)
    return dict(seconds=times, median_seconds=statistics.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--skip-benchmark", action="store_true", help="Fidelity-only extension at another batch size")
    ap.add_argument("--output", type=Path, default=ROOT / "audit/sudoku-mps.json")
    a = ap.parse_args()
    if a.n < 1 or a.batch_size < 1 or a.repeats < 1:
        ap.error("Counts must be positive")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("CPU fallback must be disabled")
    if os.environ.get("PYTORCH_MPS_FAST_MATH", "0") != "0":
        raise RuntimeError("Fast math must be disabled")
    for path in ["src/ncm_mac/cached.py", "src/ncm_mac/sudoku.py", "src/ncm_mac/_vendor/models/common.py",
                 "src/ncm_mac/_vendor/models/next_concept.py", "src/ncm_mac/_vendor/models/quant.py",
                 "src/ncm_mac/_vendor/models/vae.py", "src/ncm_mac/_vendor/models/vqvae.py"]:
        original = subprocess.check_output(["git", "show", "254bb49:" + path], cwd=ROOT)
        if original != (ROOT / path).read_bytes():
            raise RuntimeError(f"CPU ground-truth source changed: {path}")
    torch.set_num_threads(4)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this execution context")
    report = dict(ground_truth_commit="254bb49", torch=torch.__version__, macos=platform.mac_ver()[0],
                  arch=platform.machine(), dtype="float32", threads=4, n_per_task=a.n, batch_size=a.batch_size,
                  fallback=False, fast_math=False, teacher_forcing=True, tasks={})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for difficulty in ["easy", "medium", "hard"]:
        verify_assets(ROOT, difficulty, include_data=True)
        cpu = SudokuModel(ROOT / "artifacts/sudoku" / difficulty)
        gpu = SudokuModel(ROOT / "artifacts/sudoku" / difficulty)
        install_area_pool(gpu.vqvae.vq)
        gpu.vqvae.to("mps"); gpu.net.to("mps")
        dd = ROOT / "data/sudoku" / difficulty
        boards, refs = load_boards(dd / "puzzles.pickle"), load_boards(dd / "solutions.pickle")
        indices = [i * (len(boards) - 1) // max(a.n - 1, 1) for i in range(a.n)]
        boards, refs = [boards[i] for i in indices], [refs[i] for i in indices]
        cpu_answers, gpu_answers, stats = [], [], []
        changed_code_boards = 0
        for start in range(0, a.n, a.batch_size):
            prefix = cpu.prefix(boards[start:start + a.batch_size]); mp = prefix.to("mps")
            cc, cl, ch = capture(cpu, prefix)
            mc, ml = predict(gpu, mp)
            _, _, mh = capture(gpu, mp, cc.to("mps"))
            for level, (x, y) in enumerate(zip(ch, mh)):
                stats.append(dict(level=level, **distribution_stats(x, y)))
            stats.append(dict(level="decoder_free_running", **distribution_stats(cl, ml)))
            # Also compare decoder on exactly the CPU codes, separately from
            # divergences amplified by a different discrete trajectory.
            with torch.inference_mode():
                conditioned = gpu.vqvae.decoder(gpu.vqvae.vq.idx_to_z_q(cc.to("mps")))
            stats.append(dict(level="decoder_same_codes", **distribution_stats(cl, conditioned)))
            changed_code_boards += int((cc != mc.cpu()).any(-1).sum())
            cpu_answers.extend(cl.argmax(-1).tolist()); gpu_answers.extend(ml.argmax(-1).cpu().tolist())
            print(f"{difficulty} {min(start+a.batch_size,a.n)}/{a.n}", flush=True)
        measurements = {}
        for batch in ([] if a.skip_benchmark else [1, 8, 32]):
            prefix = cpu.prefix(boards[:min(batch, len(boards))]); mp = prefix.to("mps")
            measurements[str(len(prefix))] = dict(cpu=timed(cpu, prefix, "cpu", a.repeats),
                                                  mps=timed(gpu, mp, "mps", a.repeats))
        task = dict(indices=indices, changed_code_boards=changed_code_boards,
                    changed_answer_boards=sum(x != y for x, y in zip(cpu_answers, gpu_answers)),
                    changed_answer_tokens=sum(sum(x != y for x, y in zip(p, q)) for p, q in zip(cpu_answers,gpu_answers)),
                    cpu_metrics=metrics(cpu_answers, refs, boards), mps_metrics=metrics(gpu_answers, refs, boards),
                    probability_comparisons=stats, benchmark=measurements,
                    mps_current_bytes=torch.mps.current_allocated_memory(), mps_driver_bytes=torch.mps.driver_allocated_memory())
        # Optional comparison with the immutable CPU8 responses from 254bb49.
        reference_path = ROOT / f"audit/sudoku-parity-{difficulty}-predictions.json"
        if reference_path.exists() and a.n == 128:
            baseline = json.loads(subprocess.check_output(
                ["git", "show", "254bb49:audit/sudoku-parity.json"], cwd=ROOT))
            answers = json.loads(reference_path.read_text())
            actual_hash = hashlib.sha256(json.dumps(answers, separators=(",", ":")).encode()).hexdigest()
            if actual_hash != baseline["tasks"][difficulty]["predictions_sha256"]:
                raise RuntimeError("CPU8 reference predictions do not match 254bb49")
            if indices != baseline["tasks"][difficulty]["official_sample_indices"]:
                raise RuntimeError("Different sample indices from CPU8 reference")
            task["changed_mps_answer_boards_vs_committed_cpu8"] = sum(x != y for x,y in zip(answers,gpu_answers))
            task["changed_cpu_answer_boards_vs_committed_cpu8"] = sum(x != y for x,y in zip(answers,cpu_answers))
        report["tasks"][difficulty] = task
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        print(difficulty, "changed answers", task["changed_answer_boards"], "benchmarks", measurements, flush=True)
        del cpu, gpu, prefix, mp, cc, cl, ch, mc, ml, mh, conditioned
        gc.collect(); torch.mps.empty_cache()


if __name__ == "__main__":
    main()
