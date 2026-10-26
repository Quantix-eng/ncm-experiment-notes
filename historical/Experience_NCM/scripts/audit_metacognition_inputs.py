"""Audit archived outputs for metacognition design; runs no model inference."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ncm_mac.sudoku import load_boards, verify_assets


def digest(row):
    return hashlib.sha256(bytes(row)).hexdigest()


def cells(row):
    return [v for v in row if v not in (10, 11)]


def valid(grid):
    if len(grid) != 81:
        return False
    groups = ([grid[i:i + 9] for i in range(0, 81, 9)] +
              [grid[j::9] for j in range(9)] +
              [[grid[(br + i) * 9 + bc + j] for i in range(3) for j in range(3)]
               for br in (0, 3, 6) for bc in (0, 3, 6)])
    return all(sorted(group) == list(range(1, 10)) for group in groups)


def main():
    source = ROOT / "audit/sudoku-parity-full.json"
    report = {"scope": "Archived official test predictions; no new inference or uniqueness proof",
              "reference_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "tasks": {}, "cross_difficulty_overlap": {}}
    prior = {}
    for difficulty in ("easy", "medium", "hard"):
        verify_assets(ROOT, difficulty, include_data=True)
        dd = ROOT / "data/sudoku" / difficulty
        puzzles = load_boards(dd / "puzzles.pickle")
        refs = load_boards(dd / "solutions.pickle")
        pp = ROOT / "audit" / f"sudoku-parity-full-{difficulty}-predictions.json"
        preds = json.loads(pp.read_text())
        archived = json.loads(source.read_text())["tasks"][difficulty]
        assert archived["official_sample_indices"] == list(range(len(puzzles)))
        assert len(preds) == len(puzzles) == len(refs)
        assert hashlib.sha256(json.dumps(preds, separators=(",", ":")).encode()).hexdigest() == archived["predictions_sha256"]
        exact, accepted, disagreement, clue_breaks, malformed = 0, 0, 0, 0, 0
        grammar_anomalies = []
        for index, (puzzle, ref, pred) in enumerate(zip(puzzles, refs, preds)):
            g, target, given = cells(pred), cells(ref), cells(puzzle)
            assert len(given) == 81 and valid(target)
            assert all(v == 0 or v == target[i] for i, v in enumerate(given))
            correct = pred == ref
            clues_ok = len(g) == 81 and all(v == 0 or v == g[i] for i, v in enumerate(given))
            # Check exact output grammar independently of the existing evaluator.
            grammar = len(pred) == 89 and all(
                (v == 10 if i % 10 == 9 else 1 <= v <= 9)
                for i, v in enumerate(pred))
            accept = grammar and valid(g) and clues_ok
            exact += correct
            accepted += accept
            disagreement += correct != accept
            clue_breaks += not clues_ok
            malformed += not grammar
            if not grammar:
                grammar_anomalies.append({"index": index, "length": len(pred),
                    "invalid_positions": [i for i, v in enumerate(pred)
                        if not (v == 10 if i % 10 == 9 else 1 <= v <= 9)],
                    "invalid_values": [v for i, v in enumerate(pred)
                        if not (v == 10 if i % 10 == 9 else 1 <= v <= 9)]})
        assert exact == archived["metrics"]["solved"]
        ph, sh = {digest(p) for p in puzzles}, {digest(s) for s in refs}
        report["tasks"][difficulty] = {
            "n": len(preds), "exact_successes": exact, "failures": len(preds) - exact,
            "external_verifier_acceptances": accepted,
            "external_verifier_vs_exact_disagreements": disagreement,
            "clue_violations": clue_breaks, "malformed_outputs": malformed,
            "grammar_anomalies": grammar_anomalies,
            "unique_puzzles": len(ph), "unique_solutions": len(sh),
            "clue_counts": sorted({sum(v != 0 for v in cells(p)) for p in puzzles}),
            "predictions_file_sha256": hashlib.sha256(pp.read_bytes()).hexdigest()}
        for other, (oph, osh, other_refs) in prior.items():
            report["cross_difficulty_overlap"][f"{other}/{difficulty}"] = {
                "identical_prompts": len(ph & oph), "identical_solutions": len(sh & osh),
                "same_solution_at_same_index": sum(a == b for a, b in zip(refs, other_refs))}
        prior[difficulty] = ph, sh, refs
    out = ROOT / "audit/metacognition-input-audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
