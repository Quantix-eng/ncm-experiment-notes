"""Small, offline TinyStories audit; preserves the upstream model and decoding."""
import argparse
import gc
import json
import platform
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream-ncm"))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer
from tokenization.bpe_tokenizer import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--area-cpu", action="store_true", help="Explicit CPU fallback for MPS area interpolation")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    if a.area_cpu:
        import models.quant as quant
        original = quant.F
        class FunctionalProxy:
            def __getattr__(self, name):
                return getattr(original, name)
            def interpolate(self, x, *args, **kwargs):
                if x.device.type == "mps" and kwargs.get("mode") == "area":
                    return original.interpolate(x.cpu(), *args, **kwargs).to(x.device)
                return original.interpolate(x, *args, **kwargs)
        quant.F = FunctionalProxy()
    torch.set_num_threads(4)
    ck = ROOT / "artifacts/tinystories"
    cfg = OmegaConf.load(ck / "config.yaml")
    report = dict(device=a.device, torch=torch.__version__, python=platform.python_version(),
                  macos=platform.mac_ver()[0], arch=platform.machine(), batch_size=a.batch_size,
                  threads=torch.get_num_threads(), mps_built=torch.backends.mps.is_built(),
                  mps_available=torch.backends.mps.is_available(), seed=42,
                  protocol="FP32, upstream sampling temperature=1 top_k=0 top_p=1; 1 warmup")
    report["area_interpolation_on_cpu"] = a.area_cpu
    if a.device == "mps" and not report["mps_available"]:
        raise RuntimeError("MPS unavailable in this execution context")
    t0 = time.perf_counter()
    # Tensor-only loading; BPE pickles are the pinned, SHA-verified upstream files.
    sd = torch.load(ck / "tokenizer.pt", map_location="cpu", weights_only=True, mmap=True)
    vqvae = VQVAE(**cfg.tokenizer.vqvae).eval()
    vqvae.load_state_dict(sd["model"], strict=True)
    del sd
    net = NextConceptTransformer(vqvae, cfg.model).eval()
    sd = torch.load(ck / "generator.pt", map_location="cpu", weights_only=True, mmap=True)
    net.load_state_dict(sd["model"], strict=True)
    l0p = sd["inference"]["level0_unigram"].float().cpu().clone()
    report["generator_checkpoint_keys"] = list(sd)
    del sd
    gc.collect()
    report["parameters"] = {"generator": sum(p.numel() for p in net.parameters()),
                            "vqvae": sum(p.numel() for p in vqvae.parameters())}
    report["tensor_bytes_cpu"] = sum(t.numel() * t.element_size() for m in (vqvae, net)
                                     for t in list(m.parameters()) + list(m.buffers()))
    report["levels"] = list(vqvae.vq.levels)
    for key, file in [("vocab_file_path", "bpe_vocab_8192.pickle"),
                      ("merges_file_path", "bpe_merges_8192.pickle")]:
        cfg.tokenizer.pre_tokenizer[key] = str(ck / file)
    bpe = BPETokenizer.from_config(cfg.tokenizer.pre_tokenizer)
    # vqvae is held in an ordinary tuple by net: BOTH must be moved explicitly.
    vqvae.to(a.device)
    net.to(a.device)
    report["load_seconds"] = time.perf_counter() - t0

    def sync():
        if a.device == "mps":
            torch.mps.synchronize()

    samples, timings = [], []
    with torch.inference_mode():
        for i in range(a.repeats + 1):
            torch.manual_seed(42 + max(i - 1, 0))
            l0 = torch.multinomial(l0p, a.batch_size, replacement=True).view(a.batch_size, 1).to(a.device)
            prefix = torch.full((a.batch_size, 1), bpe.eos_token_id, device=a.device, dtype=torch.long)
            sync()
            start = time.perf_counter()
            codes = net.generate(l0, prefix, temperature=1.0, top_k=0, top_p=1.0)
            _, zlevels = vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
            logits = vqvae.decoder(zlevels[-1])
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("Non-finite decoder logits")
            ids = logits.argmax(-1)
            sync()
            elapsed = time.perf_counter() - start
            if i:
                timings.append(elapsed)
                for row in ids.cpu().tolist():
                    if bpe.eos_token_id in row:
                        row = row[:row.index(bpe.eos_token_id)]
                    samples.append(bpe.decode(row, skip_special_tokens=True))
            else:
                report["warmup_seconds"] = elapsed
        zhats, _ = vqvae.vq.idx_to_zhats(codes, return_zlevels=False), None
        report["zhats_alias_confirmed"] = len({id(z) for z in zhats[0]}) == 1
        report["code_shape"] = list(codes.shape)
        report["decoded_shape"] = list(ids.shape)
    report["generation_seconds"] = timings
    report["samples"] = samples
    report["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if a.device == "mps":
        report["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        report["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    Path(a.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
