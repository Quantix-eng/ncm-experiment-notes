"""Download only pinned Sudoku assets, verifying content before atomic rename."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path, entry):
    raw = path.read_bytes()
    if len(raw) != entry["bytes"]:
        return False
    if entry["sha256"]:
        return hashlib.sha256(raw).hexdigest() == entry["sha256"]
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == entry["git_oid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", choices=["all", "easy", "medium", "hard"], default="all")
    a = ap.parse_args()
    entries = json.loads((ROOT / "assets.lock.json").read_text())
    entries = [e for e in entries if a.difficulty == "all" or e["path"].startswith(f"sudoku/{a.difficulty}/")]
    if shutil.disk_usage(ROOT).free < sum(e["bytes"] for e in entries) + 1024**3:
        raise RuntimeError("Keep at least 1 GiB free beyond download size")

    def download(e):
        dest = ROOT / ("artifacts" if e["kind"] == "model" else "data") / e["path"]
        if dest.is_file() and digest(dest, e):
            return str(dest.relative_to(ROOT)) + " verified (existing)"
        dest.parent.mkdir(parents=True, exist_ok=True)
        prefix = "datasets/" if e["kind"] == "dataset" else ""
        url = f"https://huggingface.co/{prefix}{e['repo']}/resolve/{e['revision']}/{e['path']}"
        part = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as src, part.open("wb") as out:
            shutil.copyfileobj(src, out, length=1024 * 1024)
        if not digest(part, e):
            raise RuntimeError(f"Integrity check failed: {part}")
        part.replace(dest)
        return str(dest.relative_to(ROOT)) + " verified"

    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(download, entries):
            print(result, flush=True)


if __name__ == "__main__":
    main()
