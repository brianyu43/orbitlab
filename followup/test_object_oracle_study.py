import unittest
import torch.nn.functional as F
from common import torch, np
from object_world import ObjectState, state_slots, edit
from object_tasks import command, analytic_step
from object_oracle_study import Predictor, permute_slots, factor_checks


class OracleStudyChecks(unittest.TestCase):
    def test_equal_active_parameter_budget_and_object_permutation(self):
        flat,shared=Predictor('flat'),Predictor('object')
        self.assertEqual(sum(p.numel() for p in flat.parameters()),34991)
        self.assertEqual(sum(p.numel() for p in shared.parameters()),34991)
        x=torch.randn(3,4,16);mask=F.one_hot(torch.tensor([0,1,3]),4).float();c=torch.randn(3,13)
        perm=torch.tensor([2,0,3,1])
        torch.testing.assert_close(shared(x[:,perm],mask[:,perm],c),shared(x,mask,c)[:,perm])
        for model in [flat,shared]:
            loss=model(x,mask,c).square().mean();loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters()))

    def test_analytic_update_matches_renderer_factors(self):
        ss=(ObjectState(0,1,2,20,25,7,3),ObjectState(1,3,0,44,37,6,1))
        for op,value in [('color',5),('rotate',1),('rotate',3),('translate',[5,-3])]:
            for tid in [0,1]:
                x=torch.from_numpy(state_slots(ss))[None];mask=F.one_hot(torch.tensor([tid]),4).float();c=torch.from_numpy(command(op,value))[None]
                y=torch.from_numpy(state_slots(edit(ss,tid,op,value)))[None]
                torch.testing.assert_close(analytic_step(x,mask,c),y,atol=1e-6,rtol=1e-6)

    def test_permutation_includes_empty_slots_and_preserves_target_alignment(self):
        x=torch.zeros(512,4,16);x[:,:2,15]=1;x[:,0,0]=1;y=x.clone();y[:,0,0]=4
        m=torch.zeros(512,4);m[:,0]=1
        bx,by,bm=permute_slots(x,y,m,torch.Generator().manual_seed(13))
        self.assertTrue((bm.sum(0)>0).all());self.assertTrue((bx[:,:,15].sum(0)>0).all())
        torch.testing.assert_close((by-bx)[:,:,0],3*bm)

    def test_factor_checks_detect_target_and_preservation_errors(self):
        ss=(ObjectState(0,0,1,20,20,6,0),ObjectState(1,2,4,40,40,7,2))
        y=torch.from_numpy(state_slots(ss))[None];pred=y.clone()
        self.assertTrue(factor_checks(pred,y).all())
        pred[0,1,10]+=2/31.5
        self.assertFalse(factor_checks(pred,y)[0,1,2])
        pred[0,2,15]=.8
        self.assertFalse(factor_checks(pred,y)[0,2,6])


if __name__=='__main__':unittest.main()
