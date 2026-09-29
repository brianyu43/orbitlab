import unittest
from common import torch, o, state_digest
from decoder_study import ControlledDecoder, oracle_codes

torch.set_num_threads(4)


class DecoderControls(unittest.TestCase):
    def test_oracle_has_the_declared_group_action(self):
        y=torch.tensor([[0,1],[2,4]])
        meta=[{'cx':.4,'cy':.6,'radius':.2},{'cx':.55,'cy':.33,'radius':.15}]
        z=oracle_codes(y,meta,0)
        self.assertEqual(z.shape,(2,4,16))
        for k in range(4):torch.testing.assert_close(oracle_codes(y,meta,k),o.rho(z,k),atol=0,rtol=0)

    def test_same_initial_parameters_and_equivariance(self):
        torch.manual_seed(910);a=ControlledDecoder('pixel_mean')
        torch.manual_seed(910);b=ControlledDecoder('logit_mean')
        self.assertEqual(state_digest(a),state_digest(b))
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        z=torch.randn(2,4,16)
        for model in [a,b]:
            for k in range(4):
                torch.testing.assert_close(model(o.rho(z,k)),o.rotate(model(z),k),atol=2e-6,rtol=1e-5)
            model(z).square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_pixel_mean_matches_original_decoder(self):
        model=ControlledDecoder('pixel_mean');ae=o.Autoencoder(model='equivariant')
        ae.decoder.load_state_dict(model.branch.state_dict());z=torch.randn(3,4,16)
        torch.testing.assert_close(model(z),ae.decode(z),atol=0,rtol=0)


if __name__=='__main__':unittest.main()
