import unittest
from common import torch
from object_perception_v2 import LinearImageSlots, balanced_loss


class LinearPilotChecks(unittest.TestCase):
    def test_balanced_loss_is_independent_of_black_padding(self):
        target=torch.zeros(1,3,8,8);target[:,:,2:4,2:4]=1;pred=torch.zeros_like(target)
        self.assertAlmostEqual(float(balanced_loss(pred,target)),.5)
        big=torch.zeros(1,3,16,16);big[:,:,:8,:8]=target
        self.assertAlmostEqual(float(balanced_loss(torch.zeros_like(big),big)),.5)
        self.assertEqual(float(balanced_loss(target,target)),0.)

    def test_linear_rgb_has_gradient_beyond_sigmoid_saturation(self):
        model=LinearImageSlots('slot');x=torch.zeros(1,3,64,64);x[:,:,15:20,15:20]=1
        with torch.no_grad():model.decoder[-1].bias[:3].fill_(-20)
        out=model(x,torch.randn(1,5,32));loss=balanced_loss(out['image'],x);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue((model.decoder[-1].bias.grad[:3].abs()>1).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=='__main__':torch.set_num_threads(2);unittest.main()
