"""Small algebra check with synthetic embeddings; no NCM checkpoint is loaded."""
import json
import math
import random


def windows(length=89, count=9):
    return [(length*j//count, math.ceil(length*(j+1)/count)) for j in range(count)]


def serialized_grid(mapping):
    rows = [[mapping[(r*3+r//3+c) % 9] for c in range(9)] for r in range(9)]
    return [v for r, row in enumerate(rows) for v in row + ([10] if r < 8 else [])]


def main():
    rng = random.Random(20261026)
    embeddings = [[rng.uniform(-1, 1) for _ in range(8)] for _ in range(11)]
    positions = [[rng.uniform(-1, 1) for _ in range(8)] for _ in range(89)]
    baseline = None
    largest = 0.0
    for _ in range(128):
        mapping = rng.sample(list(range(1, 10)), 9)
        tokens = serialized_grid(mapping)
        pooled = [[math.fsum(embeddings[tokens[i]][k]+positions[i][k]
                             for i in range(lo, hi))/(hi-lo)
                   for k in range(8)] for lo, hi in windows()]
        if baseline is None:
            baseline = pooled
        largest = max(largest, max(abs(x-y) for a,b in zip(pooled, baseline) for x,y in zip(a,b)))
    assert largest < 1e-12
    # An unequal weighting can distinguish one order-changing swap.
    tokens = serialized_grid(list(range(1, 10)))
    changed = tokens.copy(); changed[0], changed[1] = changed[1], changed[0]
    lo, hi = windows()[0]
    def weighted(t):
        return [math.fsum((1+(2*(i-lo)/(hi-lo-1)-1))*(embeddings[t[i]][k]+positions[i][k])
                          for i in range(lo,hi))/(hi-lo) for k in range(8)]
    assert any(abs(a-b) > 1e-12 for a,b in zip(weighted(tokens), weighted(changed)))
    print(json.dumps({'passed': True, 'examples': 128, 'max_difference': largest,
                      'scope': 'raw additive pooling identity with synthetic embeddings; not a checkpoint rerun'}, indent=2))


if __name__ == '__main__':
    main()
