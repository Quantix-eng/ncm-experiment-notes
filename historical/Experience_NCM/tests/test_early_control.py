"""Time boundary and abstention semantics for the isolated early-control engine."""
from pathlib import Path
import importlib.util
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/metacognition/early_control'))
spec=importlib.util.spec_from_file_location('early_control_study',ROOT/'experiments/metacognition/early_control/study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)

class EarlyControlTests(unittest.TestCase):
    def test_future_code_arrays_rejected_before_prompt_access(self):
        with self.assertRaises(AssertionError):
            study.code_features(None,np.zeros((2,322),dtype=int),np.zeros((2,322,16)))
        with self.assertRaises(AssertionError):
            study.code_features(None,np.zeros((2,116),dtype=int),np.zeros((2,322,16)))

    def test_keep_none_performs_no_tail_and_reports_abstentions(self):
        with patch.object(study,'begin',return_value=({},{})),patch.object(study,'rows',return_value=[{}]*8),patch.object(study,'finish',side_effect=AssertionError('no tail allowed')):
            answers,scores,timing=study.run_policy(None,torch.zeros((16,91)),np.zeros(16,dtype=int),'none',scripted=np.zeros(16,dtype=bool))
        self.assertEqual(answers,{})
        self.assertEqual(timing['retained'],0)
        self.assertEqual(timing['rejected'],16)
        self.assertEqual(timing['tail_batches'],0)
        self.assertEqual(timing['tail_seconds'],0)

if __name__=='__main__':unittest.main()
