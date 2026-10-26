"""Technical corpus pilot: randomized backtracking, independent exact cover.

No NCM scoring, no uniform-distribution or exhaustive-isomorphism claim.
"""
import argparse
import hashlib
import json
import random
import time

from features import CELL_POS
from run_pilot import ROOT, file_hash, load_boards, write_json

UNITS = [[r * 9 + c for c in range(9)] for r in range(9)] + [[r * 9 + c for r in range(9)] for c in range(9)] + [
    [(r + dr) * 9 + c + dc for dr in range(3) for dc in range(3)] for r in (0, 3, 6) for c in (0, 3, 6)]
PEERS = [set().union(*(u for u in UNITS if i in u)) - {i} for i in range(81)]


def backtracking(board, limit=2, rng=None):
    board, found = list(board), []
    if any(len([board[i] for i in u if board[i]]) != len({board[i] for i in u if board[i]}) for u in UNITS):
        return found

    def search():
        best, choices = None, None
        for i, value in enumerate(board):
            if value:
                continue
            candidates = sorted(set(range(1, 10)) - {board[j] for j in PEERS[i]})
            if not candidates:
                return
            if choices is None or len(candidates) < len(choices):
                best, choices = i, candidates
        if best is None:
            found.append(board.copy())
            return
        if rng:
            rng.shuffle(choices)
        for value in choices:
            board[best] = value
            search()
            if len(found) >= limit:
                break
        board[best] = 0

    search()
    return found


def exact_cover_count(board, limit=2):
    columns = {i: set() for i in range(324)}
    row_columns = {}
    for r in range(9):
        for c in range(9):
            for d in ([board[r * 9 + c]] if board[r * 9 + c] else range(1, 10)):
                row = (r * 9 + c) * 9 + d - 1
                keys = (r * 9 + c, 81 + r * 9 + d - 1, 162 + c * 9 + d - 1,
                        243 + (r // 3 * 3 + c // 3) * 9 + d - 1)
                row_columns[row] = keys
                for key in keys:
                    columns[key].add(row)
    count = 0

    def search(active):
        nonlocal count
        if not active:
            count += 1
            return
        key = min(active, key=lambda c: (len(active[c]), c))
        for row in sorted(active[key]):
            covered = row_columns[row]
            conflicts = set().union(*(active[c] for c in covered))
            search({c: rows - conflicts for c, rows in active.items() if c not in covered})
            if count >= limit:
                return

    search(columns)
    return count


def known_transform_key(board):
    """Digit relabeling plus eight square symmetries, not all Sudoku isomorphisms."""
    variants = []
    for reflect in (False, True):
        current = [board[r * 9 + (8 - c if reflect else c)] for r in range(9) for c in range(9)]
        for _ in range(4):
            mapping, normalized = {}, []
            for digit in current:
                if digit not in mapping:
                    mapping[digit] = len(mapping) + 1
                normalized.append(mapping[digit])
            variants.append(bytes(normalized))
            current = [current[(8 - c) * 9 + r] for r in range(9) for c in range(9)]
    return hashlib.sha256(min(variants)).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    assert args.count > 0
    directory = ROOT / "audit/metacognition/confirmation-generator-pilot"
    directory.mkdir(exist_ok=False)
    solutions = load_boards(ROOT / "data/sudoku/hard/solutions.pickle")
    excluded = {known_transform_key([solution[i] for i in CELL_POS]) for solution in solutions}
    rng, rows, rejected = random.Random(args.seed), [], []
    start, attempts = time.perf_counter(), 0
    while len(rows) < args.count:
        attempts += 1
        if attempts > args.count * 100:
            raise RuntimeError("Preset generation attempt cap reached")
        solution = backtracking([0] * 81, limit=1, rng=rng)[0]
        key = known_transform_key(solution)
        if key in excluded:
            rejected.append(dict(attempt=attempts, reason="known_solution_transformation"))
            continue
        puzzle = solution.copy()
        positions = list(range(81))
        rng.shuffle(positions)
        for position in positions:
            old, puzzle[position] = puzzle[position], 0
            if len(backtracking(puzzle)) != 1:
                puzzle[position] = old
            if sum(bool(v) for v in puzzle) == 30:
                break
        if sum(bool(v) for v in puzzle) != 30:
            rejected.append(dict(attempt=attempts, reason="greedy_removal_stopped_above_30_clues"))
            continue
        assert exact_cover_count(puzzle) == 1
        assert backtracking(puzzle, limit=1)[0] == solution
        excluded.add(key)
        rows.append(dict(attempt=attempts, puzzle=puzzle, solution=solution, known_transform_key=key))
    destination = directory / "puzzles.json"
    write_json(destination, rows)
    write_json(directory / "report.json", dict(status="complete", seed=args.seed, n=len(rows), attempts=attempts,
        rejections=rejected, wall_seconds=time.perf_counter() - start, source_sha256=file_hash(__file__),
        puzzles_sha256=file_hash(destination), nm_clues=30, independent_exact_cover_uniqueness_verified=True,
        ncm_inference=False, purpose="technical generator pilot, not scored confirmation",
        caveat="Randomized backtracking plus greedy clue removal is not uniform and may differ from official distribution; no exhaustive isomorphism or pretraining-contamination exclusion"))
    print(json.dumps(dict(status="complete", n=len(rows), attempts=attempts, seconds=time.perf_counter() - start)))


if __name__ == "__main__":
    main()
