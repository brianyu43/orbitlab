"""Check scientific invariants used by the new measurements."""
import unittest
from common import torch, o, r, state_digest, original_orbits, render_fresh
from flow_ablation import rotate_batch

torch.set_num_threads(4)


class MeasurementChecks(unittest.TestCase):
    def test_rotation_applied_to_noise_and_target_preserves_flow_path(self):
        gen=torch.Generator().manual_seed(4)
        z=torch.randn(16,4,16,generator=gen);eps=torch.randn(16,4,16,generator=gen)
        t=torch.rand(16,1,1,generator=gen);k=torch.arange(16)%4
        torch.testing.assert_close(rotate_batch((1-t)*eps+t*z,k),(1-t)*rotate_batch(eps,k)+t*rotate_batch(z,k))
        torch.testing.assert_close(rotate_batch(z-eps,k),rotate_batch(z,k)-rotate_batch(eps,k))
        for j in range(16):torch.testing.assert_close(rotate_batch(z,k)[j],o.rho(z[j:j+1],int(k[j]))[0])

    def test_identical_raw_weights_different_constraints(self):
        torch.manual_seed(91);plain=o.Velocity(16,False)
        torch.manual_seed(91);equiv=o.Velocity(16,True)
        self.assertEqual(state_digest(plain),state_digest(equiv))
        z=torch.randn(4,4,16);t=torch.rand(4);y=torch.tensor([[0,1],[1,2],[2,3],[3,4]])
        torch.testing.assert_close(equiv(o.rho(z,1),t,y),o.rho(equiv(z,t,y),1),atol=1e-6,rtol=1e-5)

    def test_shared_normalization_commutes_with_slot_action(self):
        z=torch.randn(20,4,16);mean=z.mean((0,1),keepdim=True);std=z.std((0,1),keepdim=True).clamp_min(.02)
        torch.testing.assert_close((o.rho(z,1)-mean)/std,o.rho((z-mean)/std,1))

    def test_fresh_data_excludes_original_orbits(self):
        banned=original_orbits();x,y,meta=render_fresh(24,9212302,'ood',banned)
        hashes=[v['orbit_sha256'] for v in meta]
        self.assertEqual(len(set(hashes)),24)
        self.assertFalse(set(hashes)&banned)
        self.assertTrue(torch.equal(y[:,0],y[:,1]))

    def test_decoder_mean_variance_identity(self):
        target=torch.rand(5,3,64,64);parts=torch.rand(4,5,3,64,64);mean=parts.mean(0)
        total=(parts-target[None]).square().mean()
        torch.testing.assert_close(total,(mean-target).square().mean()+(parts-mean[None]).square().mean())


if __name__=='__main__':unittest.main()
