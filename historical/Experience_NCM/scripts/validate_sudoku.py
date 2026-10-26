"""Compare every code/logit/answer against the unmodified pinned upstream code."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "upstream-ncm"))
import torch
from models.next_concept import NextConceptTransformer as OriginalGenerator
from models.vqvae import VQVAE as OriginalVQVAE
from ncm_mac.sudoku import SudokuModel, load_boards, metrics, verify_assets


def original_model(ck, cfg):
    vq = OriginalVQVAE(**cfg.tokenizer.vqvae).eval()
    sd = torch.load(ck / "tokenizer.pt", map_location="cpu", weights_only=True, mmap=True)
    vq.load_state_dict(sd["model"], strict=True)
    del sd
    net = OriginalGenerator(vq, cfg.model).eval()
    sd = torch.load(ck / "generator.pt", map_location="cpu", weights_only=True, mmap=True)
    net.load_state_dict(sd["model"], strict=True)
    return vq, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--difficulty", choices=["all", "easy", "medium", "hard"], default="all")
    ap.add_argument("--output", type=Path, default=ROOT / "audit/sudoku-parity.json")
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    actual_commit = subprocess.check_output(["git", "-C", str(ROOT / "upstream-ncm"), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != "c70274273f489abc73832b2b98584a8483b6b203":
        raise RuntimeError("Reference must be the pinned upstream commit")
    if subprocess.check_output(["git", "-C", str(ROOT / "upstream-ncm"), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Reference source must be unmodified")
    if a.n < 0 or a.batch_size <= 0 or a.threads <= 0:
        ap.error("Invalid sample count, batch size, or thread count")
    torch.set_num_threads(a.threads)
    report = dict(upstream_commit="c70274273f489abc73832b2b98584a8483b6b203", torch=torch.__version__,
                  python=platform.python_version(), macos=platform.mac_ver()[0], arch=platform.machine(),
                  threads=a.threads, batch_size=a.batch_size, dtype="float32", backend="cpu",
                  optimization="prefix-cache and reused reconstruction accumulator",
                  guarantee="Observed bitwise parity on listed official cases; no cross-backend guarantee",
                  tasks={})
    for difficulty in (["easy", "medium", "hard"] if a.difficulty == "all" else [a.difficulty]):
        verify_assets(ROOT, difficulty, include_data=True)
        dd, ck = ROOT / "data/sudoku" / difficulty, ROOT / "artifacts/sudoku" / difficulty
        puzzles, solutions = load_boards(dd / "puzzles.pickle"), load_boards(dd / "solutions.pickle")
        indices = list(range(len(puzzles)))
        if a.n:
            # Evenly span the official corpus, including its first and last boards.
            n = min(a.n, len(puzzles))
            indices = [i * (len(puzzles) - 1) // max(n - 1, 1) for i in range(n)]
            puzzles, solutions = [puzzles[i] for i in indices], [solutions[i] for i in indices]
        model = SudokuModel(ck)
        vq, net = original_model(ck, model.config)

        @torch.no_grad()
        def reference(prefix):
            codes = net.generate(None, prefix, temperature=0.0, top_k=0, top_p=1.0,
                                 cfg_scale=0.0, cfg_null_eos=None)
            _, levels = vq.vq.idx_to_zhats(codes, return_zlevels=True)
            z = levels[-1]
            logits = vq.decoder(z)
            return codes, z, logits, logits.argmax(-1)

        prefix = model.prefix(puzzles[:a.batch_size])
        reference(prefix)
        model.predict(prefix)
        # Check repaired full-history path as well as optimized final-only decoding.
        ref = reference(prefix)
        repaired = model.predict(prefix, optimized=False)
        for label, x, y in zip(["codes", "reconstruction", "logits", "ids"], ref,
                                [repaired.codes, repaired.reconstruction, repaired.logits, repaired.ids]):
            if not torch.equal(x, y):
                raise AssertionError(f"{difficulty}: repaired {label} differs")
        preds, original_times, optimized_times = [], [], []
        for i, start in enumerate(range(0, len(puzzles), a.batch_size)):
            prefix = model.prefix(puzzles[start:start + a.batch_size])
            # Alternate order to reduce systematic thermal/cache ordering bias.
            if i % 2:
                t = time.perf_counter(); opt = model.predict(prefix); optimized_times.append(time.perf_counter() - t)
                t = time.perf_counter(); ref = reference(prefix); original_times.append(time.perf_counter() - t)
            else:
                t = time.perf_counter(); ref = reference(prefix); original_times.append(time.perf_counter() - t)
                t = time.perf_counter(); opt = model.predict(prefix); optimized_times.append(time.perf_counter() - t)
            for label, x, y in zip(["codes", "reconstruction", "logits", "ids"], ref,
                                    [opt.codes, opt.reconstruction, opt.logits, opt.ids]):
                if not torch.equal(x, y):
                    raise AssertionError(f"{difficulty} start={start}: {label} differs, max abs={(x-y).abs().max().item()}")
            preds.extend(opt.ids.tolist())
            if i % 10 == 0:
                print(f"{difficulty}: {len(preds)}/{len(puzzles)} bitwise equal", flush=True)
        report["tasks"][difficulty] = dict(
            all_codes_equal=True, all_reconstructions_equal=True, all_logits_equal=True, all_answers_equal=True,
            repaired_path_equal_on_first_batch=True, metrics=metrics(preds, solutions, puzzles),
            official_sample_indices=indices,
            original_seconds=sum(original_times), optimized_seconds=sum(optimized_times),
            speedup=sum(original_times) / sum(optimized_times),
            original_batch_seconds=original_times, optimized_batch_seconds=optimized_times,
            predictions_sha256=hashlib.sha256(json.dumps(preds, separators=(",", ":")).encode()).hexdigest())
        report["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        a.output.with_name(a.output.stem + f"-{difficulty}-predictions.json").write_text(json.dumps(preds))
        print(json.dumps({difficulty: {k: v for k, v in report["tasks"][difficulty].items() if "batch_seconds" not in k}}), flush=True)
        del reference, vq, net, model, ref, opt, repaired
        gc.collect()


if __name__ == "__main__":
    main()
