import unittest
import torch
from dynamics_models import Transition,rotate_state,rotate_vec,rotate_context,rotate_motion,inertial_next,encode_states


class ModelChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(770);torch.set_num_threads(2)
        self.s=torch.randn(5,4,12);self.s[...,-1]=torch.tensor([1.,0.,1.,0.]);self.s*=self.s[...,-1:]
        self.a=torch.randn(5,4,2)*self.s[...,-1:];self.c=torch.randn(5,3)

    def nonzero_model(self,kind):
        model=Transition(kind)
        torch.nn.init.normal_(model.net[-1].weight,std=.1);torch.nn.init.normal_(model.net[-1].bias,std=.1)
        return model

    def test_joint_and_wrong_condition_group_actions(self):
        for kind in ['joint_exact','wrong_exact']:
            model=self.nonzero_model(kind);result=model(self.s,self.a,self.c)
            for k in [1,2,3]:
                c=rotate_context(self.c,k) if kind=='joint_exact' else self.c
                actual=model(rotate_state(self.s,k),rotate_vec(self.a,k),c)
                torch.testing.assert_close(actual,rotate_motion(result,k),atol=2e-6,rtol=2e-5)

    def test_permutation_empty_slots_and_real_gradients(self):
        p=[2,1,3,0]
        for kind in ['object','interaction','augmented','wrong_exact','joint_exact','relaxed']:
            model=self.nonzero_model(kind);result=model(self.s,self.a,self.c)
            torch.testing.assert_close(model(self.s[:,p],self.a[:,p],self.c),result[:,p],atol=1e-6,rtol=1e-5)
            self.assertEqual(float(result[:,[1,3]].detach().abs().max()),0.)
            loss=(result-self.s[...,:4]).square().mean();loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()),0.)

    def test_common_initial_motion_and_static_encoding(self):
        for kind in ['flat','object','interaction','augmented','wrong_exact','joint_exact','relaxed']:
            torch.testing.assert_close(Transition(kind)(self.s,self.a,self.c),inertial_next(self.s,self.a))
        raw=torch.zeros(2,4,7);raw[:,0]=torch.tensor([31.5,-31.5,3.,-3.,4.,2.,1.])
        code=encode_states(raw);self.assertEqual(tuple(code.shape),(2,4,12))
        torch.testing.assert_close(code[:,0,:5],torch.tensor([[1.,-1.,1.,-1.,.5]]).expand(2,-1))
        self.assertEqual(float(code[:,1:].abs().sum()),0.)


if __name__=='__main__':unittest.main()
