"""CPU pilot. Previous campaigns are read-only; only this directory is written."""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = D.parent
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLCONFIGDIR"] = str(D / ".cache/matplotlib")
sys.path.insert(0, str(ROOT / "scripts"))
from common import CELL, SEP, SudokuModel, np, torch, write

Q = json.loads((D / "protocol.json").read_text())
OLD = ROOT / "recherche_positive"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_paths():
    return ([D / "protocol.json", D / "experiment.py", D / "verify.py"]
            + [OLD / "results/discovery_tables.json", OLD / "results/hard_readout.npz", OLD / "data/fresh.json"]
            + sorted((ROOT / "artifacts/sudoku/hard").glob("*"))
            + sorted((ROOT / "vendor/ncm_mac").rglob("*.py"))
            + [ROOT / "scripts/common.py", ROOT / "protocol.json"])


def freeze():
    path = D / "results/protocol_frozen.json"
    assert not path.exists(), "Refusing to replace local freeze"
    write(path, dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     files={str(p.relative_to(ROOT)): sha(p) for p in frozen_paths() if p.is_file()},
                     note="Local freeze before new interventions; existing corpus; not independent preregistration."))


def check_freeze():
    f = json.loads((D / "results/protocol_frozen.json").read_text())
    for name, digest in f["files"].items():
        assert sha(ROOT / name) == digest, name
    return f


def choose_pairs(prefix):
    rng = np.random.default_rng(Q["seed"])
    pairs = []
    for p in prefix:
        blank = np.flatnonzero(p[1:-1][CELL] == 0)
        eligible = {s: [] for s in Q["strata"]}
        for i, a in enumerate(blank):
            ra, ca = divmod(int(a), 9)
            for b in blank[i + 1:]:
                rb, cb = divmod(int(b), 9)
                block = (ra // 3, ca // 3) == (rb // 3, cb // 3)
                flags = [ra == rb and abs(ca-cb) == 1,
                         ra == rb and abs(ca-cb) >= 3,
                         ca == cb and not block,
                         block and ra != rb and ca != cb,
                         ra != rb and ca != cb and not block and abs(int(CELL[a])-int(CELL[b])) > 2]
                for s, flag in zip(Q["strata"], flags):
                    if flag:
                        eligible[s].append((int(a), int(b)))
        assert all(eligible.values()), "Missing stratum; do not replace source"
        pairs.append([eligible[s][rng.integers(len(eligible[s]))] for s in Q["strata"]])
    return np.asarray(pairs)


def contribution(model, codes):
    vq = model.vqvae.vq
    return vq._apply_quant_resi(vq.embeddings[11](codes), 11)


@torch.inference_mode()
def decode_edits(model, z, codes, edits, batch_size):
    out = []
    for lo in range(0, len(edits), batch_size):
        part = edits[lo:lo+batch_size]
        c = codes.unsqueeze(0).repeat(len(part), 1)
        for j, edit in enumerate(part):
            for position, code in edit:
                c[j, position] = int(code)
        original = codes.unsqueeze(0).repeat(len(part), 1)
        delta = contribution(model, c) - contribution(model, original)
        out.append(model.vqvae.decoder(z.unsqueeze(0) + delta).argmax(-1).numpy())
    return np.concatenate(out)


def make_edits(pair, target, alphabet):
    a, b = CELL[pair]
    singles = [[[(int(pos), int(alphabet[d-1]))] for d in digits]
               for pos, digits in zip((a, b), target)]
    joint = [[(int(a), int(alphabet[da-1])), (int(b), int(alphabet[db-1]))]
             for da in target[0] for db in target[1]]
    return singles, joint


def measure(a):
    native, pairs, targets = a["native"], a["pairs"], a["targets"]
    n, s = pairs.shape[:2]
    hit = np.zeros((n, s, 8, 8, 2), bool)
    single_clean = np.zeros((n, s, 2, 8), bool)
    clean = np.zeros((n, s, 8, 8), bool)
    collateral = np.zeros((n, s, 8, 8), int)
    format_change = np.zeros_like(collateral, bool)
    format_invalid = np.zeros_like(collateral, bool)
    unchanged = np.zeros_like(hit)
    for i in range(n):
        for k in range(s):
            pos = CELL[pairs[i, k]]
            pred = a["joint"][i, k].reshape(8, 8, 89)
            ideal = np.broadcast_to(native[i], (8, 8, 89)).copy()
            ideal[:, :, pos[0]] = targets[i, k, 0, :, None]
            ideal[:, :, pos[1]] = targets[i, k, 1, None, :]
            hit[i, k, :, :, 0] = pred[:, :, pos[0]] == targets[i, k, 0, :, None]
            hit[i, k, :, :, 1] = pred[:, :, pos[1]] == targets[i, k, 1, None, :]
            unchanged[i, k] = pred[:, :, pos] == native[i, pos]
            clean[i, k] = np.all(pred == ideal, -1)
            other = np.setdiff1d(CELL, pos)
            collateral[i, k] = (pred[:, :, other] != native[i, other]).sum(-1)
            format_change[i, k] = np.any(pred[:, :, SEP] != native[i, SEP], -1)
            format_invalid[i, k] = np.any(pred[:, :, SEP] != 10, -1)
            for side in range(2):
                expected = np.tile(native[i], (8, 1))
                expected[:, pos[side]] = targets[i, k, side]
                single_clean[i, k, side] = np.all(a["single"][i, k, side] == expected, -1)
    matched = single_clean[:, :, 0, :, None] & single_clean[:, :, 1, None, :]
    both = hit.all(-1)
    loss = matched & ~clean
    gain = ~matched & clean
    arrays = dict(both=both, one=(hit.sum(-1) == 1), neither=(hit.sum(-1) == 0),
                  clean=clean, matched=matched, loss=loss, gain=gain,
                  collateral=(collateral > 0), format_change=format_change,
                  format_invalid=format_invalid, both_unchanged=unchanged.all(-1))
    rng = np.random.default_rng(Q["seed"])
    ix = rng.integers(0, n, (Q["bootstrap"], n))

    def rate(v):
        per_source = v.reshape(n, -1).mean(1)
        return dict(count=int(v.sum()), total=int(v.size), rate=float(v.mean()),
                    ci95=np.quantile(per_source[ix].mean(1), [.025, .975]).tolist())

    def retention(mask, joint):
        numerator = (mask & joint).reshape(n, -1).sum(1)
        denominator = mask.reshape(n, -1).sum(1)
        den = denominator[ix].sum(1)
        boot = numerator[ix].sum(1)[den > 0] / den[den > 0]
        return dict(count=int(numerator.sum()), total=int(denominator.sum()),
                    rate=float(numerator.sum()/denominator.sum()),
                    ci95=np.quantile(boot, [.025, .975]).tolist())

    summary = {key: rate(v) for key, v in arrays.items()}
    summary["single_clean"] = rate(single_clean)
    summary["retention"] = retention(matched, clean)
    net_loss = matched.astype(float) - clean.astype(float)
    summary["net_composition_loss"] = rate(net_loss)
    summary["by_stratum"] = [{"stratum": name, **{key: rate(v[:, k:k+1]) for key, v in arrays.items()},
                               "retention": retention(matched[:, k:k+1], clean[:, k:k+1])}
                              for k, name in enumerate(Q["strata"])]
    equal_targets = targets[:, :, 0, :, None] == targets[:, :, 1, None, :]
    summary["by_equal_targets"] = [dict(equal=flag, trials=int((equal_targets == flag).sum()),
                                      both=int((both & (equal_targets == flag)).sum())) for flag in (False, True)]
    summary["sources_with_loss"] = int(loss.reshape(n, -1).any(1).sum())
    summary["sources_with_collateral"] = int(arrays["collateral"].reshape(n, -1).any(1).sum())
    summary["zero_event_source_wilson_upper95"] = float(1.959963984540054**2/(n+1.959963984540054**2))
    summary["decision"] = ("strong_composition_degradation" if summary["retention"]["rate"] < .95
                           or summary["net_composition_loss"]["rate"] > .05 else "no_composition_veto_at_tested_scope")
    return summary


@torch.inference_mode()
def run():
    check_freeze()
    assert not (D / "results/interventions.npz").exists(), "Use a new directory for a new run"
    start = time.monotonic()
    old = np.load(OLD / "results/hard_readout.npz")
    indices = np.arange(Q["source_indices"]["start"], Q["source_indices"]["stop_exclusive"])
    prefix, native = old["prefix"][indices], old["native"][indices]
    pairs = choose_pairs(prefix)
    alphabet = np.array(json.loads((OLD / "results/discovery_tables.json").read_text())["hard"]["alphabet_by_digit"])
    model = SudokuModel(ROOT / "artifacts/sudoku/hard")
    codes, zs = [], []
    for lo in range(0, len(prefix), 16):
        pred = model.predict(torch.tensor(prefix[lo:lo+16]))
        ids = indices[lo:lo+16]
        assert np.array_equal(pred.codes.numpy(), old["codes"][ids])
        assert np.array_equal(pred.reconstruction.numpy(), old["zfinal"][ids])
        assert np.array_equal(model.vqvae.decoder(pred.reconstruction).argmax(-1).numpy(), native[lo:lo+16])
        codes.append(pred.codes[:, -89:].clone())
        zs.append(pred.reconstruction.clone())
    codes, zs = torch.cat(codes), torch.cat(zs)
    single = np.zeros((len(prefix), 5, 2, 8, 89), np.int64)
    joint = np.zeros((len(prefix), 5, 64, 89), np.int64)
    targets = np.zeros((len(prefix), 5, 2, 8), np.int64)
    for i in range(len(prefix)):
        sham = decode_edits(model, zs[i], codes[i], [[]], 1)[0]
        assert np.array_equal(sham, native[i])
        for k, pair in enumerate(pairs[i]):
            targets[i, k] = [[d for d in range(1, 10) if d != native[i, CELL[c]]] for c in pair]
            singles, edits = make_edits(pair, targets[i, k], alphabet)
            for side in range(2):
                single[i, k, side] = decode_edits(model, zs[i], codes[i], singles[side], Q["decoder_batch_size"])
            joint[i, k] = decode_edits(model, zs[i], codes[i], edits, Q["decoder_batch_size"])
        print(json.dumps(dict(stage="composition", done=i+1, total=len(prefix))), flush=True)
    arrays = dict(indices=indices, prefix=prefix, native=native, pairs=pairs, targets=targets,
                  single=single, joint=joint, codes=codes.numpy(), zfinal=zs.numpy(), alphabet=alphabet)
    np.savez_compressed(D / "results/interventions.npz", **arrays)
    summary = measure(arrays)
    summary.update(n_sources=len(indices), n_pairs=int(np.prod(pairs.shape[:2])),
                   environment=dict(python=platform.python_version(), torch=torch.__version__,
                                    numpy=np.__version__, platform=platform.platform(),
                                    device="cpu", dtype="float32", threads=torch.get_num_threads()),
                   utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   elapsed_seconds=time.monotonic()-start, native_parity_sources=len(indices), sham_sources=len(indices),
                   frozen_manifest_sha256=sha(D / "results/protocol_frozen.json"))
    write(D / "results/summary.json", summary)
    check_freeze()
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--freeze"]:
        freeze()
    elif not sys.argv[1:]:
        run()
    else:
        raise SystemExit("Usage: experiment.py [--freeze]")
