"""Development-only, no training; safe checkpoint tensors and head-level swaps."""
from pathlib import Path
import functools
import hashlib
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
META = json.loads((HERE / "sources/benchmark_metadata.json").read_text())
CASES = {x["case_id"]: x for x in META["cases"]}


def less(a, b):
    try:
        return (a > b) - (a < b)
    except TypeError:
        return (str(a) > str(b)) - (str(a) < str(b))


def vocabulary(case):
    # The published Tracr BasisDirection comparator, NOT fitted to predictions.
    return sorted(CASES[str(case)]["vocab"] + ["BOS", "PAD"],
                  key=functools.cmp_to_key(less))


def oracle(case, seq):
    if case == 8:
        return seq
    if case == 33:
        return [int(len(x) % 2 == 0) for x in seq]
    raise ValueError("Reserved tasks are not executable in this pilot")


def generate():
    result = []
    for case in [8, 33]:
        rng = np.random.default_rng(20261025 + case)
        vocab = CASES[str(case)]["vocab"]
        used = set()
        for i in range(128):
            while True:
                n = int(rng.integers(3, 10))
                seq = [vocab[int(j)] for j in rng.integers(0, len(vocab), n)]
                if tuple(seq) not in used:
                    used.add(tuple(seq))
                    break
            result.append({"id": f"dev-{case}-{i:03d}", "case": case,
                           "tokens": seq, "target_ids": oracle(case, seq)})
    return result


class Model:
    def __init__(self, case):
        self.case = case
        self.cfg = CASES[str(case)]["transformer_cfg"]
        assert self.cfg["normalization_type"] is None
        assert not self.cfg["parallel_attn_mlp"]
        assert self.cfg["act_fn"] == "gelu"
        self.w = torch.load(HERE / f"sources/models/{case}/ll_model.pth",
                            weights_only=True, map_location="cpu")

    def attention(self, x, layer):
        p = f"blocks.{layer}.attn."
        w = self.w
        q, k, v = [torch.einsum("bpd,hdm->bphm", x, w[p + "W_" + c])
                   + w[p + "b_" + c] for c in ["Q", "K", "V"]]
        scores = torch.einsum("bqhm,bkhm->bhqk", q, k)
        scores /= math.sqrt(self.cfg["d_head"])
        if self.cfg["attention_dir"] == "causal":
            mask = w[p + "mask"][:x.shape[1], :x.shape[1]].bool()
            scores = torch.where(mask[None, None], scores, w[p + "IGNORE"])
        pattern = scores.softmax(-1)
        z = torch.einsum("bhqk,bkhm->bqhm", pattern, v)
        heads = torch.einsum("bphm,hmd->bphd", z, w[p + "W_O"])
        return heads.sum(2) + w[p + "b_O"], heads

    def mlp(self, x, layer):
        p = f"blocks.{layer}.mlp."
        return F.gelu(x @ self.w[p + "W_in"] + self.w[p + "b_in"]) @ self.w[p + "W_out"] + self.w[p + "b_out"]

    def decode(self, x):
        return x @ self.w["unembed.W_U"] + self.w["unembed.b_U"]

    def native(self, ids):
        x = self.w["embed.W_E"][ids] + self.w["pos_embed.W_pos"][:ids.shape[1]]
        cache = {}
        for l in range(self.cfg["n_layers"]):
            cache[(l, "pre")] = x.clone()
            a, h = self.attention(x, l)
            cache[(l, "attn")] = a.clone()
            cache[(l, "heads")] = h.clone()
            x = x + a
            cache[(l, "mid")] = x.clone()
            m = self.mlp(x, l)
            cache[(l, "mlp")] = m.clone()
            x = x + m
        return self.decode(x), cache

    def suffix(self, mid, cut):
        x = mid
        contributions = []
        m = self.mlp(x, cut)
        contributions.append(m)
        x = x + m
        for l in range(cut + 1, self.cfg["n_layers"]):
            a, _ = self.attention(x, l)
            contributions.append(a)
            x = x + a
            m = self.mlp(x, l)
            contributions.append(m)
            x = x + m
        return self.decode(x), contributions

    def replay(self, mid, contributions):
        x = mid
        for c in contributions:
            x = x + c
        return self.decode(x)


def encode(row):
    vocab = vocabulary(row["case"])
    m = {v: i for i, v in enumerate(vocab)}
    return torch.tensor([[m["BOS"]] + [m[t] for t in row["tokens"]]
                         + [m["PAD"]] * (9 - len(row["tokens"]))])


def mean_loss(logits, row):
    t = torch.tensor(row["target_ids"])
    z = logits[0, 1:len(t) + 1]
    return float(F.cross_entropy(z, t)), bool(torch.equal(z.argmax(-1), t))


@torch.inference_mode()
def run():
    started = time.perf_counter()
    seal = json.loads((HERE / "results/protocol_frozen.json").read_text())
    for name, digest in seal["sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, name
    rows = json.loads((HERE / "data/dev.json").read_text())
    outputs = []; native_rows = []; arrays = {}
    for case in [8, 33]:
        subset = [r for r in rows if r["case"] == case]
        model = Model(case)
        natives = [model.native(encode(r)) for r in subset]
        for i, row in enumerate(subset):
            z, cache = natives[i]
            loss, exact = mean_loss(z, row)
            native_rows.append({"id": row["id"], "case": case, "loss": loss,
                                "exact": exact})
            arrays[row["id"] + "-native"] = z.numpy()
            for cut in [0, 1]:
                # All attention heads are absent from the two published circuits.
                head = 0
                mid = cache[(cut, "mid")]
                y00, t0 = model.suffix(mid, cut)
                assert torch.equal(y00, z), "native suffix parity"
                donors = [(i + 1) % 128, (i + 2) % 128]
                # Donors may have different unpadded lengths; the entire length-10
                # head activation is exchanged, as the declared intervention unit.
                mids = [mid + (natives[d][1][(cut, "heads")][:, :, head]
                               - cache[(cut, "heads")][:, :, head])
                        for d in donors]
                y11, td = model.suffix(mids[0], cut)
                _, tw = model.suffix(mids[1], cut)
                conditions = {"Y00": y00,
                              "Y10": model.replay(mids[0], t0),
                              "Y11": y11,
                              "Y01": model.replay(mid, td),
                              "Y1wrong": model.replay(mids[0], tw)}
                assert torch.equal(model.replay(mid, t0), z), "sham parity"
                record = {"id": row["id"], "case": case, "cut": cut,
                          "head": head, "donors": [subset[d]["id"] for d in donors],
                          "conditions": {k: dict(zip(["loss", "exact"], mean_loss(v, row)))
                                         for k, v in conditions.items()}}
                losses = {k: v["loss"] for k, v in record["conditions"].items()}
                record["contrasts"] = {
                    "D": losses["Y10"] - losses["Y00"],
                    "R": losses["Y10"] - losses["Y11"],
                    "I": losses["Y10"] + losses["Y01"] - losses["Y11"] - losses["Y00"],
                    "S": losses["Y1wrong"] - losses["Y11"]}
                record["oracle_claim"] = "unchanged_sequence_at_valid_positions"
                record["oracle_agreement"] = bool(torch.equal(
                    y11[0, 1:len(row["tokens"]) + 1].argmax(-1),
                    torch.tensor(row["target_ids"])))
                outputs.append(record)
                for key, value in conditions.items():
                    arrays[row["id"] + f"-L{cut}H0-" + key] = value.numpy()
    np.savez_compressed(HERE / "results/development_logits.npz", **arrays)
    summary = {"native": native_rows, "interventions": outputs,
               "elapsed_seconds": time.perf_counter() - started,
               "reserved_tasks_executed": [], "device": "CPU", "dtype": "FP32",
               "threads": torch.get_num_threads(), "torch_version": torch.__version__,
               "numpy_version": np.__version__}
    (HERE / "results/development.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"native_exact": sum(x["exact"] for x in native_rows),
                      "native_total": len(native_rows),
                      "interventions": len(outputs),
                      "oracle_agreement": sum(x["oracle_agreement"] for x in outputs),
                      "elapsed_seconds": summary["elapsed_seconds"]}))


if __name__ == "__main__":
    run()
