"""Passive observation of the existing cached CPU path, with guaranteed cleanup."""
import time

import numpy as np
import torch


@torch.inference_mode()
def instrumented_prediction(model, prefix):
    net, vq = model.net, model.vqvae.vq
    lengths = list(net.levels)
    hidden, logits = [], {}
    handles = []
    expected_shapes = [prefix.shape[1], *lengths]

    def block_hook(module, args, output):
        call = len(hidden)
        if call >= len(expected_shapes) or output.shape[1:] != (expected_shapes[call], 512):
            raise AssertionError("Unexpected block call or target/prefix shape")
        # Keep a sentinel for prefill; do not observe its values.
        hidden.append(None if call == 0 else output.detach().clone())

    def head_hook(k):
        def capture(module, args, output):
            if k in logits or output.shape[1:] != (lengths[k], 16):
                raise AssertionError("Unexpected head call")
            logits[k] = output.detach().clone()
        return capture

    t0 = time.perf_counter()
    try:
        handles.append(net.transformer.blocks[7].register_forward_hook(block_hook))
        for k in range(len(lengths)):
            handles.append(net.lm_heads[k].register_forward_hook(head_hook(k)))
        prediction = model.predict(prefix)
    finally:
        for handle in handles:
            handle.remove()
    generation_decode_seconds = time.perf_counter() - t0
    if len(hidden) != len(lengths) + 1 or len(logits) != len(lengths):
        raise AssertionError("Missing generation observations")
    if any(block.attn.caching or block.attn.cached_k is not None for block in net.transformer.blocks):
        raise AssertionError("Cache not cleared")
    hidden = hidden[1:]
    offset = 0
    for k, length in enumerate(lengths):
        if not torch.equal(logits[k].argmax(-1), prediction.codes[:, offset:offset + length]):
            raise AssertionError("Observed code logits not aligned with generated codes")
        offset += length
    t1 = time.perf_counter()
    snapshots, = vq.idx_to_zhats(prediction.codes, return_zlevels=False)
    if len({z.data_ptr() for z in snapshots}) != len(lengths):
        raise AssertionError("Aliased reconstruction history")
    if not torch.equal(snapshots[-1], prediction.reconstruction):
        raise AssertionError("Offline reconstruction does not equal generation accumulator")
    # Independent explicit accumulation verifies every level, not just the last.
    running = torch.zeros_like(snapshots[-1])
    offset = 0
    for k, length in enumerate(lengths):
        emb = vq.embeddings[k](prediction.codes[:, offset:offset + length])
        if k < len(lengths) - 1:
            emb = vq._up_to_full(emb, k, lengths[-1])
        running.add_(vq._apply_quant_resi(emb, k))
        if not torch.equal(running, snapshots[k]):
            raise AssertionError(f"Incorrect intermediate reconstruction at {k + 1}")
        offset += length
    reconstruction_seconds = time.perf_counter() - t1
    decoder_logits, decoder_probs, decoded, decoder_seconds = [], [], [], []
    summaries, relative = [], []
    for k, snapshot in enumerate(snapshots):
        t = time.perf_counter()
        dl = model.vqvae.decoder(snapshot)
        decoder_seconds.append(time.perf_counter() - t)
        if k == len(lengths) - 1 and not torch.equal(dl, prediction.logits):
            raise AssertionError("Final decoder logits changed on replay")
        decoder_logits.append(dl)
        decoder_probs.append(dl.double().softmax(-1).numpy())
        decoded.append(dl.argmax(-1).numpy().astype(np.uint8))
        h = hidden[k].double()
        summaries.append(torch.cat([h.mean(1), h.std(1, correction=0)], dim=-1).numpy())
        old = torch.zeros_like(snapshot) if k == 0 else snapshots[k - 1]
        num = (snapshot.double() - old.double()).flatten(1).norm(dim=1)
        den = old.double().flatten(1).norm(dim=1).clamp_min(1e-12)
        # No previous reconstruction exists at level 1; encode zero explicitly.
        relative.append(np.zeros(len(prefix)) if k == 0 else (num / den).numpy())
    trace = dict(codes=prediction.codes.numpy().copy(),
                 code_probs=np.concatenate([logits[k].double().softmax(-1).numpy() for k in range(len(lengths))], axis=1),
                 decoder_probs=np.stack(decoder_probs, axis=1), decoded_ids=np.stack(decoded, axis=1),
                 hidden_summary=np.stack(summaries, axis=1),
                 reconstruction_relative_change=np.stack(relative, axis=1))
    raw = dict(hidden=hidden, code_logits=[logits[k] for k in range(len(lengths))],
               reconstruction=snapshots, decoder_logits=decoder_logits)
    timing = dict(instrumented_generation_and_final_decode_seconds=generation_decode_seconds,
                  reconstruction_and_history_checks_seconds=reconstruction_seconds,
                  intermediate_decoder_seconds=decoder_seconds)
    return prediction, trace, raw, timing
