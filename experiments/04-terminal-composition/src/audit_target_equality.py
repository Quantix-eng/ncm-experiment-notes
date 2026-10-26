"""Post-hoc stratification: different requested digits versus equal digits.

No new neural inference, endpoint selection or change to the frozen decision.
"""
import json
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent


def main():
    with np.load(D / "results/interventions.npz") as archive:
        a = {k: archive[k] for k in archive.files}
    q = json.loads((D / "results/summary.json").read_text())
    out = []
    for k, stratum in enumerate(q["by_stratum"]):
        counts = {flag: dict(trials=0, matched=0, retained=0, loss=0, collateral=0)
                  for flag in ("same", "different")}
        for i in range(32):
            pos = np.array([int(c)+int(c)//9 for c in a["pairs"][i, k]])
            native = a["native"][i]
            isolated = np.zeros((2, 8), bool)
            for side in range(2):
                for t in range(8):
                    expected = native.copy()
                    expected[pos[side]] = a["targets"][i, k, side, t]
                    isolated[side, t] = np.array_equal(expected, a["single"][i, k, side, t])
            other = [x for x in range(89) if x % 10 != 9 and x not in pos]
            for ta in range(8):
                for tb in range(8):
                    da, db = a["targets"][i, k, 0, ta], a["targets"][i, k, 1, tb]
                    c = counts["same" if da == db else "different"]
                    c["trials"] += 1
                    expected = native.copy()
                    expected[pos] = [da, db]
                    pred = a["joint"][i, k, ta*8+tb]
                    matched = bool(isolated[0, ta] and isolated[1, tb])
                    clean = np.array_equal(pred, expected)
                    c["matched"] += int(matched)
                    c["retained"] += int(matched and clean)
                    c["loss"] += int(matched and not clean)
                    c["collateral"] += int(np.any(pred[other] != native[other]))
        for metric in ("matched", "loss", "collateral"):
            assert sum(c[metric] for c in counts.values()) == stratum[metric]["count"]
        for c in counts.values():
            c["retention"] = c["retained"]/c["matched"]
        out.append(dict(stratum=stratum["stratum"], by_equal_target=counts))
    result = dict(status="passed", scope="Post-hoc descriptive breakdown; no new inference or change to frozen decision", by_stratum=out)
    (D / "results/posthoc_target_equality_audit.json").write_text(json.dumps(result, indent=2))
    print("Post-hoc target equality audit passed.")


if __name__ == "__main__":
    main()
