"""Prefix-only KV cache for the released deterministic Sudoku architecture."""
import torch


@torch.inference_mode()
def generate_cached(net, prefix):
    vqvae = net.vqvae_local[0]
    vq = vqvae.vq
    cfg = net.config
    if not (net.query_level0 and net.use_rope and cfg.raw_token_prefix and
            not net.l0_ar and cfg.use_level_encoding and cfg.attn_pattern == "block_diagonal" and
            not net.use_ctx_encoder and not net.use_post_upsample_conv and net.prefix_pad_id is None):
        raise ValueError("Prefix caching currently supports the released Sudoku architecture only")
    b, p = prefix.shape
    prefix_h = net.transformer.wte(vqvae.encoder.wte(prefix))
    positions = net.transformer.wle(net.lvl_1L)
    z_hat = torch.zeros(b, net.levels[-1], vqvae.embed_dim, device=prefix.device)
    rope = (net.rope_cos_gen_0, net.rope_sin_gen_0)
    prefix_mask = torch.zeros(1, 1, p, p, device=prefix.device)
    net._set_kv_caching(True)
    try:
        for block in net.transformer.blocks:
            prefix_h = block(prefix_h, attn_mask=prefix_mask, rope_cos_sin=rope)
        prefix_kv = [(block.attn.cached_k, block.attn.cached_v) for block in net.transformer.blocks]
        codes = []
        offset = 0
        for k, length in enumerate(net.levels):
            if k == 0:
                x = net.build_l0_query_input(b, prefix_h.dtype)
            else:
                emb = vq.embeddings[k - 1](codes[-1])
                z_hat, x = vq.get_next_autoregressive_input(k - 1, z_hat, emb)
            x = net.transformer.wte(x) + positions[:, offset:offset + length, :]
            mask = torch.zeros(1, 1, length, p + length, device=prefix.device)
            rope_k = (getattr(net, f"rope_cos_gen_{k}"), getattr(net, f"rope_sin_gen_{k}"))
            for block, (pk, pv) in zip(net.transformer.blocks, prefix_kv):
                block.attn.cached_k, block.attn.cached_v = pk, pv
                x = block(x, attn_mask=mask, rope_cos_sin=rope_k)
            logits = net.lm_heads[k](net.transformer.ln_f(x))
            codes.append(logits.argmax(-1))
            offset += length
        z_hat.add_(vq._apply_quant_resi(vq.embeddings[-1](codes[-1]), len(net.levels) - 1))
        return torch.cat(codes, dim=1), z_hat
    finally:
        net._set_kv_caching(False)
