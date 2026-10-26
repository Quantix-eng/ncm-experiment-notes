"""Regression tests for intermediate history and unchanged final reconstruction."""
from pathlib import Path
import unittest
import torch
from omegaconf import OmegaConf
from ncm_mac._vendor.models.vqvae import VQVAE
from ncm_mac._vendor.models.next_concept import NextConceptTransformer
from ncm_mac.sudoku import metrics
from ncm_mac.cached import generate_cached

ROOT = Path(__file__).resolve().parents[1]


class Representations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(42)
        # Small synthetic model exercises the same multiscale quantizer, no downloads.
        config = OmegaConf.create(dict(
            clss="multiscale_residual_vector_quantizer", decay=0.99, epsilon=1e-5,
            commitment_cost=0.25, levels=dict(use_manual_levels=True, manual_levels=[1, 2, 5, 9]),
            quant_resi=dict(enabled=True, ratio=0.5, share_mode=0, learnable_ratio=False)))
        cls.vq = VQVAE(in_vocab_size=12, context_length=9, embed_dim=8, vqvae_vocab_size=[16] * 4,
                       n_layers=1, vector_quantizer_config=config, compression_factor=1).eval().vq

    def test_intermediate_snapshots_and_final(self):
        vq = self.vq
        idx = torch.randint(0, 16, (2, sum(vq.levels)))
        with torch.inference_mode():
            snapshots, levels = vq.idx_to_zhats(idx)
            self.assertEqual(len({z.data_ptr() for z in snapshots}), len(vq.levels))
            running = torch.zeros_like(snapshots[0])
            offset = 0
            for k, length in enumerate(vq.levels):
                z = vq.embeddings[k](idx[:, offset:offset + length])
                if k < len(vq.levels) - 1:
                    z = vq._up_to_full(z, k, vq.levels[-1])
                running.add_(vq._apply_quant_resi(z, k))
                self.assertTrue(torch.equal(snapshots[k], running), f"Wrong history at level {k}")
                offset += length
            self.assertTrue(torch.equal(vq.idx_to_z_q(idx), levels[-1]))
            self.assertTrue(torch.equal(snapshots[-1], levels[-1]))
            before = snapshots[-1].clone()
            snapshots[0].zero_()
            self.assertTrue(torch.equal(snapshots[-1], before))
            self.assertTrue(torch.equal(levels[-1], before))
            no_levels = vq.idx_to_zhats(idx, return_zlevels=False)
            self.assertEqual(len(no_levels), 1)
            self.assertTrue(torch.equal(no_levels[0][-1], before))

    def test_metrics_count_malformed_output_as_failure(self):
        result = metrics([[1] * 80], [[1] * 81], [[0] * 91])
        self.assertEqual(result["malformed_outputs"], 1)
        self.assertEqual(result["solved"], 0)
        self.assertEqual(result["valid_grids"], 0)

    def test_reuse_generation_accumulator_preserves_codes_and_final_tensor(self):
        cfg = OmegaConf.create(dict(n_layer=2, n_head=2, n_embd=16, bias=True, dropout=0.0,
            vocab_size=[16] * 4, query_level0=True, predict_level0=False,
            context_length=12, prefix_len=3, raw_token_prefix=True, attn_pattern="block_diagonal",
            use_positional_encoding=True, use_level_encoding=True, use_rope=True, use_qk_norm=True))
        vqvae = VQVAE(in_vocab_size=12, context_length=9, embed_dim=8, vqvae_vocab_size=[16] * 4,
            n_layers=1, compression_factor=1,
            vector_quantizer_config=OmegaConf.create(dict(clss="multiscale_residual_vector_quantizer",
                decay=0.99, epsilon=1e-5, commitment_cost=0.25,
                levels=dict(use_manual_levels=True, manual_levels=[1, 2, 5, 9]),
                quant_resi=dict(enabled=True, ratio=0.5, share_mode=0, learnable_ratio=False)))).eval()
        net = NextConceptTransformer(vqvae, cfg).eval()
        with torch.inference_mode():
            for batch in [1, 3]:
                prefix = torch.randint(0, 12, (batch, 3))
                original_codes = net.generate(None, prefix, temperature=0.0)
                codes, z = net.generate(None, prefix, temperature=0.0, return_reconstruction=True)
                self.assertTrue(torch.equal(codes, original_codes))
                self.assertTrue(torch.equal(z, vqvae.vq.idx_to_z_q(original_codes)))
                self.assertTrue(torch.equal(vqvae.decoder(z), vqvae.decode(original_codes)))
                cached_codes, cached_z = generate_cached(net, prefix)
                self.assertTrue(torch.equal(cached_codes, original_codes))
                self.assertTrue(torch.equal(cached_z, z))
                self.assertTrue(all(not block.attn.caching and block.attn.cached_k is None
                                    for block in net.transformer.blocks))


if __name__ == "__main__":
    unittest.main()
