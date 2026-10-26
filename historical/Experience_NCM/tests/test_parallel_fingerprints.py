import unittest
import torch
from ncm_mac.sudoku import Prediction
from ncm_mac.parallel import prediction_fingerprints


class FingerprintTests(unittest.TestCase):
    def test_one_ulp_is_detected(self):
        a=Prediction(torch.tensor([[1,2]]),torch.ones(1,2,4),torch.ones(1,2,12),torch.zeros(1,2,dtype=torch.long))
        b=Prediction(a.codes,a.reconstruction,a.logits.clone(),a.ids)
        b.logits[0,0,0]=torch.nextafter(b.logits[0,0,0],torch.tensor(float('inf')))
        ah,bh=prediction_fingerprints(a),prediction_fingerprints(b)
        self.assertEqual(ah[:2],bh[:2])
        self.assertNotEqual(ah[2],bh[2])

    def test_shape_is_part_of_identity(self):
        a=Prediction(torch.tensor([[1,2]]),torch.ones(1,2,4),torch.ones(1,2,12),torch.zeros(1,2,dtype=torch.long))
        b=Prediction(a.codes.reshape(2,1),a.reconstruction,a.logits,a.ids)
        self.assertNotEqual(prediction_fingerprints(a)[0],prediction_fingerprints(b)[0])
