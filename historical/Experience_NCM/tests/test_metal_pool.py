import unittest
import torch
from ncm_mac.metal_experiment import AreaPool1d, install_area_pool


class AreaPoolTests(unittest.TestCase):
    def test_all_sudoku_bin_boundaries(self):
        # Basis vectors validate every bin and weighting, including overlaps.
        basis = torch.eye(89).unsqueeze(0)
        for out in [2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89]:
            actual = AreaPool1d(89, out)(basis)
            expected = torch.nn.functional.interpolate(basis, size=out, mode="area")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_random_values_and_constants(self):
        torch.manual_seed(42)
        x = torch.randn(2, 256, 89)
        for out in [2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89]:
            torch.testing.assert_close(AreaPool1d(89, out)(x),
                torch.nn.functional.interpolate(x, size=out, mode="area"), rtol=1e-5, atol=5e-7)
            self.assertTrue(torch.equal(AreaPool1d(89, out)(torch.ones_like(x)), torch.ones(2,256,out)))

    def test_adapter_is_instance_local(self):
        class Quantizer(torch.nn.Module):
            levels = [1,2,89]
            def get_next_autoregressive_input(self, *args):
                return "original"
        reference, experiment = Quantizer(), Quantizer()
        install_area_pool(experiment)
        self.assertEqual(reference.get_next_autoregressive_input(), "original")
        self.assertIs(reference.get_next_autoregressive_input.__func__, Quantizer.get_next_autoregressive_input)
        self.assertFalse(hasattr(reference,"metal_area_pools"))
        self.assertTrue(hasattr(experiment,"metal_area_pools"))
