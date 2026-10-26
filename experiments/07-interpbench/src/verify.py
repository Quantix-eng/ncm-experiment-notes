"""Independent NumPy FP64 native calculation, no reserved-task evaluations."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def forward(ids, weights):
    w = {k: v.double().numpy() for k, v in weights.items()}
    x = w["embed.W_E"][ids] + w["pos_embed.W_pos"][:len(ids)]
    for layer in [0, 1]:
        p = f"blocks.{layer}.attn."
        q, k, v = [np.einsum("pd,hdm->phm", x, w[p + "W_" + c]) + w[p + "b_" + c]
                   for c in ["Q", "K", "V"]]
        scores = np.einsum("qhm,khm->hqk", q, k)
        scores = np.where(w[p + "mask"][None].astype(bool), scores, w[p + "IGNORE"])
        ex = np.exp(scores - scores.max(-1, keepdims=True))
        pattern = ex / ex.sum(-1, keepdims=True)
        z = np.einsum("hqk,khm->qhm", pattern, v)
        x = x + np.einsum("phm,hmd->pd", z, w[p + "W_O"]) + w[p + "b_O"]
        p = f"blocks.{layer}.mlp."
        y = x @ w[p + "W_in"] + w[p + "b_in"]
        y = 0.5 * y * (1 + np.vectorize(math.erf)(y / math.sqrt(2)))
        x = x + y @ w[p + "W_out"] + w[p + "b_out"]
    return x @ w["unembed.W_U"] + w["unembed.b_U"]


def run():
    data = json.loads((HERE / "data/dev.json").read_text())
    results = json.loads((HERE / "results/development.json").read_text())
    logits = np.load(HERE / "results/development_logits.npz")
    stats = []; max_difference = 0.; all_exact = 0
    for case in [8, 33]:
        w = torch.load(HERE / f"sources/models/{case}/ll_model.pth", map_location="cpu", weights_only=True)
        # Independent explicit encoders derived from source, no runtime imports.
        vocab = list(range(11)) + ["BOS", "PAD"] if case == 8 else [
            "BOS", "J", "LB", "PAD", "TPSI", "V", "b", "no", "oCLrZaW", "poiVg"]
        index = {v: i for i, v in enumerate(vocab)}
        for row in [r for r in data if r["case"] == case]:
            ids = [index["BOS"]] + [index[v] for v in row["tokens"]] + [index["PAD"]] * (9-len(row["tokens"]))
            expected = forward(np.array(ids), w)
            actual = logits[row["id"] + "-native"][0]
            difference = float(np.max(np.abs(expected-actual)))
            max_difference = max(max_difference, difference)
            assert np.allclose(expected, actual, atol=2e-4, rtol=2e-5), (row["id"], difference)
            assert np.array_equal(expected.argmax(-1), actual.argmax(-1))
            all_exact += 1
    for case in [8, 33]:
        native = [r for r in results["native"] if r["case"] == case]
        for cut in [0, 1]:
            obs = [r for r in results["interventions"] if r["case"] == case and r["cut"] == cut]
            for r in obs:
                l = {k: v["loss"] for k,v in r["conditions"].items()}
                expected = dict(D=l["Y10"]-l["Y00"], R=l["Y10"]-l["Y11"],
                                I=l["Y10"]+l["Y01"]-l["Y11"]-l["Y00"], S=l["Y1wrong"]-l["Y11"])
                assert expected == r["contrasts"]
            stats.append({"case":case,"cut":cut,"families":len(obs),
                          "native_exact":sum(r["exact"] for r in native),
                          "exact":{k:sum(r["conditions"][k]["exact"] for r in obs)
                                   for k in ["Y00","Y10","Y11","Y01","Y1wrong"]},
                          "oracle_agreement":sum(r["oracle_agreement"] for r in obs),
                          "mean_contrasts":{k:float(np.mean([r["contrasts"][k] for r in obs]))
                                            for k in ["D","R","I","S"]}})
    seal=json.loads((HERE/'results/protocol_frozen.json').read_text())
    for p,h in seal['sha256'].items():assert hashlib.sha256((HERE/p).read_bytes()).hexdigest()==h,p
    out = {"native_numpy_parity_count":all_exact,"max_abs_logits_fp32_vs_fp64":max_difference,
           "argmax_parity":True,"frozen_hashes_unchanged":True,"cells":stats,
           "official_transformerlens_parity":False,"independent_scientific_replication":False}
    (HERE / "results/verification.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))


if __name__ == "__main__":run()
