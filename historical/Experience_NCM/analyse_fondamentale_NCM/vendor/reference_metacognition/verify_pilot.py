"""Verify saved evidence independently, without loading a model or held-out data."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from features import build_features

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directory = ROOT / "audit/metacognition/pilot-v0.2"
    report = json.loads((directory / "report.json").read_text())
    split_path = ROOT / "audit/metacognition/split-v0.2.json"
    split = json.loads(split_path.read_text())
    protocol = json.loads((ROOT / "experiments/metacognition/protocol-v0.2.json").read_text())
    assert report["status"] == "complete"
    assert sha(split_path) == report["split_sha256"]
    for path, fingerprint in report["source_files"].items():
        assert sha(ROOT / path) == fingerprint, path
    for name, metadata in report["run_artifacts"].items():
        assert sha(directory / name) == metadata["sha256"], name
    with np.load(directory / "traces-32.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    with np.load(directory / "full-traces-2.npz", allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    assert arrays["sample_indices"].tolist() == split["pilot_indices"]
    assert set(arrays["sample_indices"]).issubset(split["partition_indices"]["train"])
    assert all(np.isfinite(a).all() for a in arrays.values())
    assert np.allclose(arrays["code_probs"].sum(-1), 1, atol=1e-12, rtol=0)
    assert np.allclose(arrays["decoder_probs"].sum(-1), 1, atol=1e-12, rtol=0)
    offset = 0
    for k, length in enumerate(protocol["level_lengths"]):
        h = torch.from_numpy(raw["hidden"][:, offset:offset + length]).double()
        pooled = torch.cat([h.mean(1), h.std(1, correction=0)], dim=-1).numpy()
        assert np.array_equal(pooled, arrays["hidden_summary"][:2, k]), k
        cp = torch.from_numpy(raw["code_logits"][:, offset:offset + length]).double().softmax(-1).numpy()
        assert np.array_equal(cp, arrays["code_probs"][:2, offset:offset + length]), k
        dp = torch.from_numpy(raw["decoder_logits"][:, k]).double().softmax(-1).numpy()
        assert np.array_equal(dp, arrays["decoder_probs"][:2, k]), k
        offset += length
    verified_levels = []
    for level in [3, 6, 9, 11]:
        before = build_features(arrays["prefix"], arrays, level, protocol["level_lengths"])
        poisoned = {key: value.copy() for key, value in arrays.items()}
        boundary = sum(protocol["level_lengths"][:level])
        poisoned["code_probs"][:, boundary:] = np.nan
        poisoned["codes"][:, boundary:] = 99
        for name in ("decoder_probs", "hidden_summary", "reconstruction_relative_change"):
            poisoned[name][:, level:] = np.nan
        poisoned["decoded_ids"][:, level:] = 99
        after = build_features(arrays["prefix"], poisoned, level, protocol["level_lengths"])
        assert all(np.array_equal(before[key], after[key]) for key in before), level
        verified_levels.append(level)
    result = dict(status="passed", report_sha256=sha(directory / "report.json"),
                  verifier_sha256=sha(Path(__file__)),
                  checks=["source and artifact hashes", "training-only selection", "finite trace arrays",
                          "normalized probabilities", "raw two-sample pooling and probabilities bitwise replay",
                          "future states cannot alter requested-level features"],
                  temporal_boundaries_tested=verified_levels,
                  model_execution=False, held_out_traces_read=False)
    (directory / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
