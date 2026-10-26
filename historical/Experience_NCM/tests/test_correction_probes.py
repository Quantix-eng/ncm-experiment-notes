"""Regression tests for the cell target's leakage and pseudoreplication traps."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/metacognition/correction'))
from cell_features import visible_features
from learning import fit_logistic, grouped_metrics, puzzle_weights
from test_metacognition_pilot import synthetic_trace


class CorrectionProbes(unittest.TestCase):
    def test_cell_features_do_not_read_future_levels(self):
        prefix,trace=synthetic_trace()
        logits=np.zeros((1,89,12))
        before=visible_features(prefix,trace,logits)
        self.assertEqual(before.shape,(1,81,388))
        trace['decoder_probs'][:,9:]=np.nan
        trace['decoded_ids'][:,9:]=99
        trace['hidden_summary'][:]=np.nan
        after=visible_features(prefix,trace,logits)
        self.assertTrue(np.array_equal(before,after))

    def test_global_group_offset_cannot_help_within_puzzle_ranking(self):
        y=np.array([1,0,1,0]);group=np.array([0,0,1,1]);score=np.array([2.,1.,1.,2.])
        self.assertEqual(grouped_metrics(y,score,group)['within_puzzle_auc'],.5)
        shifted=score+np.array([100.,100.,-100.,-100.])
        self.assertEqual(grouped_metrics(y,shifted,group)['within_puzzle_auc'],.5)
        self.assertEqual(grouped_metrics(y,np.zeros(4),group)['within_puzzle_auc'],.5)

    def test_weighting_is_invariant_to_duplicate_cells_of_one_puzzle(self):
        torch.set_num_threads(2)
        settings=dict(lr=1.,max_iter=500,tolerance_grad=1e-7,tolerance_change=1e-9,line_search_fn='strong_wolfe')
        x=np.array([[-2.],[2.],[-.5],[.5]])
        y=np.array([0,1,0,1]);group=np.array([0,0,1,1])
        a=fit_logistic(x,y,group,1.,settings)
        duplicate=np.array([0,1,0,1,2,3])
        b=fit_logistic(x[duplicate],y[duplicate],group[duplicate],1.,settings)
        self.assertTrue(np.allclose(a['weight'],b['weight'],atol=1e-6))
        w=puzzle_weights(group[duplicate])
        self.assertAlmostEqual(w[group[duplicate]==0].sum(),.5)

    def test_macro_metric_excludes_uniform_groups_but_retains_them_in_pooled_metrics(self):
        y=np.array([1,0,0,0]);group=np.array([0,0,1,1]);scores=np.array([2.,1.,1.,2.])
        result=grouped_metrics(y,scores,group)
        self.assertEqual(result['mixed_groups'],1)
        self.assertEqual(result['within_puzzle_auc'],1.)
        self.assertEqual(result['n'],4)

if __name__=='__main__':unittest.main()
