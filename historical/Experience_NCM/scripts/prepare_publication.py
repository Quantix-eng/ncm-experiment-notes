"""Export a minimal English release; keep the research archive local."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "publication"
NAME = "ncm-mac-0.2.0"
FLAGS = ["all_codes_equal", "all_reconstructions_equal", "all_logits_equal", "all_answers_equal"]


def complete_full_report(report):
    tasks = report.get("tasks", {})
    return set(tasks) == {"easy", "medium", "hard"} and all(
        t.get("official_sample_indices") == list(range(2000))
        and all(t.get(k) is True for k in FLAGS) for t in tasks.values())


def main():
    files = ["README.md", "LICENSE", "CITATION.cff", "pyproject.toml", "assets.lock.json",
             "scripts/download_sudoku.py", "scripts/validate_sudoku.py"]
    files += [str(p.relative_to(ROOT)) for p in (ROOT / "src/ncm_mac").rglob("*.py")
              if "__pycache__" not in p.parts and p.name != "metal_experiment.py"]
    files = sorted(set(files))
    stage = DEST / NAME
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for name in files:
        source = ROOT / name
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"Expected regular source file: {name}")
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(ROOT / "audit/requirements-lock.txt", stage / "requirements.txt")
    (stage / ".gitignore").write_text(
        ".venv/\n__pycache__/\n*.py[cod]\n*.egg-info/\n.DS_Store\n"
        "artifacts/\ndata/\nupstream-ncm/\nbuild/\ndist/\n"
        "results*.json\nparity*.json\n")
    evidence = dict(
        description="Recorded CPU FP32 comparisons with unmodified, pinned upstream; no universal equivalence claim.",
        upstream_commit="c70274273f489abc73832b2b98584a8483b6b203",
        measured_hardware="MacBook Pro M1, 8 GB unified memory",
        sampled_parity=json.loads((ROOT / "audit/sudoku-parity.json").read_text()),
        benchmark=json.loads((ROOT / "audit/sudoku-summary.json").read_text()),
        benchmark_scope="384 boards; loading and warmup excluded; timings are not an end-to-end application speedup.",
        full_corpus_status="Not yet certified complete; sampled validation above is complete.")
    full = ROOT / "audit/sudoku-parity-full.json"
    if full.exists():
        report = json.loads(full.read_text())
        if complete_full_report(report):
            evidence["full_corpus_parity"] = report
            evidence["full_corpus_status"] = "Passed: all 6000 official boards, default single-process CPU path."
            evidence["full_corpus_timing_note"] = "Concurrent preparation tasks; use the separate sampled benchmark for speed claims."
    (stage / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    hashes = {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(stage.rglob("*")) if p.is_file()}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    manifest = dict(version="0.2.0", source_commit=commit, working_tree_dirty=dirty,
                    upstream_commit=evidence["upstream_commit"],
                    note="Curated source export; hashes identify its exact files. Research history is retained separately.",
                    files=hashes)
    (stage / "PUBLICATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = DEST / (NAME + ".tar.gz")
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(stage, arcname=NAME)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    (DEST / "SHA256SUMS").write_text(f"{sha}  {archive.name}\n")
    print(json.dumps(dict(folder=str(stage), archive=str(archive), sha256=sha,
                          files=len(hashes) + 1, source_commit=commit, dirty=dirty,
                          full_corpus_status=evidence["full_corpus_status"]), indent=2))


if __name__ == "__main__":
    main()
