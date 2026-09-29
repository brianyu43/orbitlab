import unittest
import torch
from object_state_readout import set_loss,discrete_states,measure,attribute_cost


class ReadoutChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(401);torch.set_num_threads(2)
        self.target=torch.zeros(4,4,16)
        for i in range(2):
            self.target[:,i,i]=1;self.target[:,i,4+i]=1;self.target[:,i,10]=(-.3 if i==0 else .3)
            self.target[:,i,12]=.75;self.target[:,i,13]=1;self.target[:,i,15]=1

    def test_set_loss_does_not_depend_on_slot_or_gt_order(self):
        raw=torch.randn(4,5,16,requires_grad=True);a,match=set_loss(raw,self.target)
        b,_=set_loss(raw[:,[4,2,0,3,1]],self.target[:,[2,1,3,0]])
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6);a.backward()
        self.assertTrue(torch.isfinite(raw.grad).all());self.assertGreater(float(raw.grad.abs().sum()),0.)

    def test_perfect_predictions_misses_and_extra_objects(self):
        raw=torch.zeros(4,5,16);raw[...,15]=-10
        for i in range(2):
            raw[:,i]=self.target[:,i];raw[:,i,:10]*=10;raw[:,i,15]=10
        rows,_=measure(raw,self.target);self.assertTrue(all(v['full_scene_success'] for v in rows))
        raw[:,2,15]=10;rows,_=measure(raw,self.target);self.assertTrue(all(not v['full_scene_success'] and v['predicted_objects']==3 for v in rows))
        raw[...,15]=-10;rows,_=measure(raw,self.target);self.assertTrue(all(v['matched_fraction']==0 and v['shape_correct']==0 for v in rows))


if __name__=='__main__':unittest.main()
