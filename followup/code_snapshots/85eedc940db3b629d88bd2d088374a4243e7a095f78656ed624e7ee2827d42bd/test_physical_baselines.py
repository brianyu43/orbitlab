import unittest
import numpy as np
import dynamics_world as d
from dynamics_physical_baselines import predict


class BaselineChecks(unittest.TestCase):
    def test_matches_single_disk_simulator_with_force_drag_and_walls(self):
        rng=np.random.default_rng(7024)
        for _ in range(100):
            r=float(rng.choice([4,5,6]));state=np.array([[*rng.uniform(-31.5+r,31.5-r,2),*rng.uniform(-2,2,2),r,0,1.]])
            c=np.array([0,.04,*rng.uniform(-.08,.08,2),rng.uniform(0,.01)]);a=rng.uniform(-1,1,(1,2))
            expected,_=d.step(state,c,a);np.testing.assert_allclose(predict(state,c,a),expected[:,:4],atol=1e-12,rtol=0)

    def test_contact_omission_is_detectable_and_rotation_is_joint(self):
        state=np.array([[-4.5,0,1,0,4,0,1],[4.5,0,-1,0,4,1,1]],float);c=np.array([0,.05,0,0,0]);a=np.zeros((2,2))
        true,_=d.step(state,c,a);pred=predict(state,c,a);self.assertGreater(np.max(np.abs(pred-true[:,:4])),1.)
        for k in [1,2,3]:
            rotated=predict(d.rotate_state(state,k),d.rotate_context(c,k),d.rotate_vectors(a,k))
            np.testing.assert_allclose(rotated,np.concatenate([d.rotate_vectors(pred[:,:2],k),d.rotate_vectors(pred[:,2:],k)],-1),atol=1e-12)


if __name__=='__main__':unittest.main()
