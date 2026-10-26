"""Separate scalar audit and singleton replay; not independent peer review."""
from experiment import D, Q, ROOT, CELL, SEP, SudokuModel, np, torch, write, check_freeze, sha


@torch.inference_mode()
def main():
    check_freeze()
    a = np.load(D / "results/interventions.npz")
    q = __import__("json").loads((D / "results/summary.json").read_text())
    model = SudokuModel(ROOT / "artifacts/sudoku/hard")
    counts = dict(both=0, one=0, neither=0, clean=0, matched=0, loss=0, gain=0, collateral=0,
                  format_change=0, format_invalid=0, both_unchanged=0, single_clean=0)
    source_losses = set()
    source_collateral = set()
    source_counts = []
    replay = 0
    for i in range(len(a["indices"])):
        per_source = {key: 0 for key in counts}
        native = a["native"][i]
        for k, pair in enumerate(a["pairs"][i]):
            pos = CELL[pair]
            ok = np.zeros((2, 8), bool)
            for side in range(2):
                for t in range(8):
                    ideal = native.copy()
                    ideal[pos[side]] = a["targets"][i, k, side, t]
                    ok[side, t] = np.array_equal(a["single"][i, k, side, t], ideal)
                    per_source["single_clean"] += int(ok[side, t])
            for ta in range(8):
                for tb in range(8):
                    out = a["joint"][i, k, ta*8+tb]
                    ideal = native.copy()
                    ideal[pos] = [a["targets"][i, k, 0, ta], a["targets"][i, k, 1, tb]]
                    hits = sum(int(out[p] == ideal[p]) for p in pos)
                    clean = np.array_equal(out, ideal)
                    matched = bool(ok[0, ta] and ok[1, tb])
                    collateral = any(out[p] != native[p] for p in CELL if p not in pos)
                    values = dict(both=hits == 2, one=hits == 1, neither=hits == 0, clean=clean,
                                  matched=matched, loss=matched and not clean, gain=clean and not matched,
                                  collateral=collateral, format_change=any(out[SEP] != native[SEP]),
                                  format_invalid=any(out[SEP] != 10), both_unchanged=all(out[pos] == native[pos]))
                    for key, value in values.items():
                        per_source[key] += int(value)
                    if matched and not clean:
                        source_losses.add(i)
                    if collateral:
                        source_collateral.add(i)
            # Replay first source using singleton calls and direct code replacement.
            if i == 0:
                c0 = torch.tensor(a["codes"][i:i+1])
                z = torch.tensor(a["zfinal"][i:i+1])
                vq = model.vqvae.vq
                original = vq._apply_quant_resi(vq.embeddings[-1](c0), 11)
                jobs = []
                for side in range(2):
                    for t in range(8):
                        jobs.append(([(pos[side], a["targets"][i, k, side, t])], a["single"][i, k, side, t]))
                for ta in range(8):
                    for tb in range(8):
                        jobs.append(([(pos[0], a["targets"][i, k, 0, ta]),
                                      (pos[1], a["targets"][i, k, 1, tb])], a["joint"][i, k, ta*8+tb]))
                for changes, expected in jobs:
                    c = c0.clone()
                    for p, digit in changes:
                        c[0, p] = int(a["alphabet"][digit-1])
                    r = vq._apply_quant_resi(vq.embeddings[-1](c), 11)
                    output = model.vqvae.decoder(z + (r-original)).argmax(-1).numpy()[0]
                    assert np.array_equal(output, expected), (k, changes)
                    replay += 1
        for key, value in per_source.items():
            counts[key] += value
        source_counts.append(per_source)
    for key, count in counts.items():
        assert q[key]["count"] == count, (key, count, q[key])
    assert q["sources_with_loss"] == len(source_losses)
    assert q["sources_with_collateral"] == len(source_collateral)
    assert q["retention"]["count"] == counts["matched"]-counts["loss"]
    assert q["retention"]["total"] == counts["matched"]
    assert q["both"]["total"] == 32*5*64
    # Independently recompute the clustered bootstrap from scalar counts.
    ix = np.random.default_rng(Q["seed"]).integers(0, 32, (Q["bootstrap"], 32))
    for key in counts:
        per = np.array([v[key] for v in source_counts])/(80 if key == "single_clean" else 320)
        assert np.allclose(np.quantile(per[ix].mean(1), [.025, .975]), q[key]["ci95"], atol=1e-14)
    m = np.array([v["matched"] for v in source_counts])
    loss = np.array([v["loss"] for v in source_counts])
    ratio = (m-loss)[ix].sum(1)/m[ix].sum(1)
    assert np.allclose(np.quantile(ratio, [.025, .975]), q["retention"]["ci95"], atol=1e-14)
    check_freeze()
    write(D / "results/verification.json", dict(status="passed", scalar_counts=counts,
          singleton_replays=replay, replay_sources=1, replay_tokens=89,
          bootstrap_unit="source puzzle", frozen_inputs_unchanged=True,
          observations_sha256=sha(D / "results/interventions.npz"),
          summary_sha256=sha(D / "results/summary.json"),
          limitation="Same research agent; software audit, not independent replication or scientific certification."))
    print("Audit passed: scalar metrics, clustered intervals, frozen hashes, 400 singleton replays.")


if __name__ == "__main__":
    main()
