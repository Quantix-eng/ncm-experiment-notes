"""Sudoku-only inference; no rounding, mixed precision, repair, or answer search."""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import pickle

import torch
from omegaconf import OmegaConf

from ncm_mac._vendor.models.next_concept import NextConceptTransformer
from ncm_mac._vendor.models.vqvae import VQVAE
from ncm_mac._vendor.tokenization.sudoku_tokenizer import SudokuTokenizer
from ncm_mac.cached import generate_cached


class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Only plain data accepted, not {module}.{name}")


def load_boards(path):
    """Official inputs are lists of token lists; forbid pickle globals."""
    obj = _DataOnlyUnpickler(io.BytesIO(Path(path).read_bytes())).load()
    if not isinstance(obj, list) or not obj:
        raise ValueError("Expected a nonempty list of Sudoku boards")
    for row in obj:
        if not isinstance(row, list) or not all(type(x) is int and 0 <= x <= 11 for x in row):
            raise ValueError("Expected lists of Sudoku token IDs (0..11)")
    return obj


def verify_assets(root, difficulty, *, include_data=False):
    root = Path(root)
    entries = json.loads((root / "assets.lock.json").read_text())
    selected = [e for e in entries if e["path"].startswith(f"sudoku/{difficulty}/")
                and (include_data or e["kind"] == "model")]
    if len(selected) != (5 if include_data else 3):
        raise ValueError("Incomplete asset manifest")
    for e in selected:
        path = root / ("artifacts" if e["kind"] == "model" else "data") / e["path"]
        raw = path.read_bytes()
        actual = (hashlib.sha256(raw).hexdigest() if e["sha256"] else
                  hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest())
        if len(raw) != e["bytes"] or actual != (e["sha256"] or e["git_oid"]):
            raise ValueError(f"Integrity failure: {path}")


@dataclass
class Prediction:
    codes: torch.Tensor
    reconstruction: torch.Tensor
    logits: torch.Tensor
    ids: torch.Tensor


class SudokuModel:
    def __init__(self, checkpoint_dir):
        ck = Path(checkpoint_dir)
        self.config = OmegaConf.load(ck / "config.yaml")
        if not str(self.config.task).startswith("sudoku-"):
            raise ValueError("This runner supports Sudoku checkpoints only")
        g = self.config.generation
        if float(g.temperature) != 0 or float(g.cfg_scale) != 0:
            raise ValueError("Expected deterministic official Sudoku decoding")
        self.tokenizer = SudokuTokenizer()
        self.vqvae = VQVAE(**self.config.tokenizer.vqvae).eval()
        sd = torch.load(ck / "tokenizer.pt", map_location="cpu", weights_only=True, mmap=True)
        self.vqvae.load_state_dict(sd["model"], strict=True)
        del sd
        self.net = NextConceptTransformer(self.vqvae, self.config.model).eval()
        sd = torch.load(ck / "generator.pt", map_location="cpu", weights_only=True, mmap=True)
        self.net.load_state_dict(sd["model"], strict=True)
        del sd
        self.vqvae.requires_grad_(False)
        self.net.requires_grad_(False)

    def prefix(self, boards):
        p = int(self.config.model.prefix_len)
        # Official Sudoku inputs are already BOS-delimited, exactly 91 tokens.
        # Reject malformed data rather than silently inventing padding for PAD=None.
        if not boards or any(len(row) != p for row in boards):
            raise ValueError(f"Expected a nonempty batch of {p}-token official prompts")
        return torch.tensor(boards, dtype=torch.long)

    @torch.inference_mode()
    def predict(self, prefix, *, optimized=True, prefix_cache=True):
        if prefix.device.type != "cpu" or prefix.dtype != torch.long:
            raise ValueError("Validated execution uses CPU integer prompts and FP32 weights")
        if optimized and prefix_cache:
            codes, z = generate_cached(self.net, prefix)
        elif optimized:
            codes, z = self.net.generate(None, prefix, temperature=0.0, top_k=0, top_p=1.0,
                                        cfg_scale=0.0, cfg_null_eos=None, return_reconstruction=True)
        else:
            codes = self.net.generate(None, prefix, temperature=0.0, top_k=0, top_p=1.0,
                                      cfg_scale=0.0, cfg_null_eos=None)
            _, levels = self.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
            z = levels[-1]
        logits = self.vqvae.decoder(z)
        return Prediction(codes, z, logits, logits.argmax(-1))


def metrics(predictions, solutions, puzzles):
    if not (len(predictions) == len(solutions) == len(puzzles)) or not predictions:
        raise ValueError("Need equally sized nonempty predictions/references/prompts")
    tk = SudokuTokenizer()
    def text(row):
        return tk.decode(row, skip_special_tokens=True)
    def cells(row):
        return [v for v in row if v != 10]
    exact = cell_correct = valid = clues_ok = malformed = 0
    for pred, ref, prompt in zip(predictions, solutions, puzzles):
        p, r = cells(pred), cells(ref)
        if len(r) != 81:
            raise ValueError("Malformed reference: expected 81 cells")
        if len(p) != 81:
            # An invalid model answer is a failure, not an evaluation crash.
            malformed += 1
            continue
        exact += text(pred) == text(ref)
        cell_correct += sum(x == y for x, y in zip(p, r))
        clues = cells([v for v in prompt if v != 11])
        clues_ok += len(clues) == 81 and all(v == 0 or v == p[i] for i, v in enumerate(clues))
        groups = ([p[i:i + 9] for i in range(0, 81, 9)] + [p[j::9] for j in range(9)] +
                  [[p[(br + i) * 9 + bc + j] for i in range(3) for j in range(3)]
                   for br in (0, 3, 6) for bc in (0, 3, 6)])
        valid += all(sorted(group) == list(range(1, 10)) for group in groups)
    n = len(predictions)
    return dict(n=n, solved=exact, solved_rate=exact / n, cell_accuracy=cell_correct / (81 * n),
                valid_grids=valid, valid_rate=valid / n, clues_preserved=clues_ok,
                clues_preserved_rate=clues_ok / n, malformed_outputs=malformed)
