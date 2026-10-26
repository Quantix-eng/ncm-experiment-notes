"""Inventory official NCM releases using metadata/configs only, never weights."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from omegaconf import OmegaConf
from ncm_mac._vendor.models.next_concept import NextConceptTransformer
from ncm_mac._vendor.models.vqvae import VQVAE


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def main():
    repo = "hazyresearch/ncm-checkpoints"
    metadata = json.loads(fetch(f"https://huggingface.co/api/models/{repo}"))
    revision = metadata["sha"]
    tree = json.loads(fetch(f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&limit=1000"))
    paths = {entry["path"] for entry in tree if entry["type"] == "file"}
    rows = []
    for name in ("sudoku/easy", "sudoku/medium", "sudoku/hard", "tinystories",
                 "xsum", "wmt14", "lm1b/small", "lm1b/medium", "lm1b/large", "owt"):
        row = {"name": name, "generator_available": f"{name}/generator.pt" in paths,
               "source": f"https://huggingface.co/{repo}/tree/{revision}/{name}"}
        config_path = f"{name}/config.yaml"
        if config_path in paths:
            raw = fetch(f"https://huggingface.co/{repo}/resolve/{revision}/{config_path}")
            cfg = OmegaConf.create(raw.decode())
            # Meta tensors retain shapes without allocating parameter storage.
            if bool(OmegaConf.select(cfg, "model.ctx_encoder.enabled", default=False)) and OmegaConf.select(cfg, "model.ctx_encoder.pretrained"):
                raise ValueError("Pretrained context encoder requires a separate count")
            with contextlib.redirect_stdout(io.StringIO()), torch.device("meta"):
                vq = VQVAE(**cfg.tokenizer.vqvae)
                net = NextConceptTransformer(vq, cfg.model)
            row.update(generator_parameters=sum(p.numel() for p in net.parameters()),
                       tokenizer_parameters=sum(p.numel() for p in vq.parameters()),
                       levels=len(vq.vq.levels), target_tokens=int(cfg.tokenizer.vqvae.context_length),
                       config_sha256=hashlib.sha256(raw).hexdigest())
            row["total_parameters"] = row["generator_parameters"] + row["tokenizer_parameters"]
            row["tokenizer_path"] = str((Path(name) / cfg.checkpoints.tokenizer).as_posix())
            # Resolve ../ for shared LM1B tokenizer.
            import posixpath
            row["tokenizer_path"] = posixpath.normpath(row["tokenizer_path"])
            row["tokenizer_available"] = row["tokenizer_path"] in paths
            del net, vq
        else:
            row["files"] = sorted(p for p in paths if p.startswith(name + "/"))
        rows.append(row)
    report = {"date": "2026-10-22", "repository": repo, "revision": revision,
              "method": "Parameter counts from official configurations instantiated on PyTorch meta device using pinned local port; buffers excluded; no weights downloaded or checkpoint compatibility test",
              "code_upstream_commit": "c70274273f489abc73832b2b98584a8483b6b203",
              "metadata": metadata, "tree": tree, "models": rows}
    out = ROOT / "audit/ncm-model-inventory.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"revision": revision, "models": rows}, indent=2))


if __name__ == "__main__":
    main()
