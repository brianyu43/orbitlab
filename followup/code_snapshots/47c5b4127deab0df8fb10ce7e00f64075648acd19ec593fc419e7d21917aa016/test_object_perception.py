import unittest
from common import torch, np
from object_perception import SlotAttention, ImageSlots, segmentation_metrics


class PerceptionChecks(unittest.TestCase):
    def test_attention_permutations_and_finite_gradients(self):
        torch.manual_seed(3);model=SlotAttention(16);features=torch.randn(2,15,16);eps=torch.randn(2,5,16)
        z,a=model(features,eps);ip=torch.randperm(15);sp=torch.tensor([2,4,0,1,3])
        zp,_=model(features[:,ip],eps[:,sp]);torch.testing.assert_close(zp,z[:,sp],atol=2e-6,rtol=2e-5)
        (z.square().mean()).backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_image_only_shapes_mask_mixture_and_slot_permutation(self):
        torch.manual_seed(1);x=torch.rand(1,3,64,64);eps=torch.randn(1,5,32);perm=torch.tensor([3,0,4,2,1])
        for kind in ['flat','slot']:
            model=ImageSlots(kind);out=model(x,eps)
            self.assertEqual(out['image'].shape,x.shape);self.assertEqual(out['slots'].shape,(1,5,32))
            torch.testing.assert_close(out['masks'].sum(1),torch.ones(1,1,64,64))
            if kind=='slot':
                other=model(x,eps[:,perm]);torch.testing.assert_close(out['image'],other['image'],atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(other['masks'],out['masks'][:,perm],atol=2e-6,rtol=2e-5)

    def test_segmentation_perfect_permutation_and_merge_failure(self):
        gt=np.zeros((16,16),np.int64);gt[2:5,2:5]=1;gt[10:13,10:13]=2
        good=np.zeros_like(gt);good[gt==1]=4;good[gt==2]=3
        m=segmentation_metrics(gt,good);self.assertEqual(m['foreground_ari'],1.);self.assertEqual(m['matched_visible_iou'],1.)
        merged=good.copy();merged[gt>0]=4;m=segmentation_metrics(gt,merged)
        self.assertEqual(m['foreground_ari'],0.);self.assertLess(m['matched_visible_iou'],.6)


if __name__=='__main__':torch.set_num_threads(2);unittest.main()
