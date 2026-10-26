"""Boundaries and held-out invariance for the new native residual readout."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[1]/'experiments/metacognition/dynamics'
sys.path.insert(0,str(HERE))
from native import native_features,reconstruct
from study import prepare,transform
from verify import direct_auc

class NativeDynamicsTests(unittest.TestCase):
    def test_future_codes_rejected_before_model_access(self):
        with self.assertRaises(AssertionError):native_features(None,torch.zeros((2,322),dtype=torch.long))
        with self.assertRaises(AssertionError):reconstruct(None,torch.zeros((2,322),dtype=torch.long),9)

    def test_independent_auc_ties_and_orientation(self):
        y=np.array([False,False,True,True]);s=np.array([0.,1.,1.,2.])
        self.assertEqual(direct_auc(y,s),.875)
        self.assertEqual(direct_auc(y,-s),.125)

    def test_heldout_values_and_future_labels_do_not_change_fitted_transforms(self):
        torch.set_num_threads(2);rng=np.random.default_rng(19);n=12*51
        data=dict(group=np.repeat(np.arange(12),51),cell=np.tile(np.arange(51),12).astype('uint8'),
            visible=rng.normal(size=(n,388)),static=rng.normal(size=(n,256)).astype('float32'),
            delta=rng.normal(size=(n,256)).astype('float32'),amplitude=rng.normal(size=(n,3)),
            current_error=np.ones(n,dtype=bool),persistent=rng.integers(2,size=n).astype(bool))
        train,valid=np.arange(9),np.arange(9,12)
        matrices,t,v,saved=prepare(data,train,valid)
        for name,(a,b)in matrices.items():
            np.testing.assert_allclose(transform(saved,data,name,t),a,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(transform(saved,data,name,v),b,rtol=1e-12,atol=1e-12)
        poisoned={k:value.copy()for k,value in data.items()}
        heldout=np.isin(data['group'],valid)
        for k in ['visible','static','delta','amplitude']:poisoned[k][heldout]*=1000
        poisoned['persistent']=~poisoned['persistent']
        _,_,_,after=prepare(poisoned,train,valid)
        for k in saved:np.testing.assert_array_equal(saved[k],after[k])

if __name__=='__main__':unittest.main()
