import unittest
import numpy as np
import dynamics_world as d


class PhysicsChecks(unittest.TestCase):
    def test_free_flight_and_force(self):
        s=np.array([[0.,0.,1.,-.5,4.,0.,1.]])
        out,e=d.step(s,np.zeros(5));np.testing.assert_allclose(out[0,:4],[1,-.5,1,-.5],atol=1e-12)
        c=np.array([0.,.04,.02,0.,0.]);out,e=d.step(s,c)
        np.testing.assert_allclose(out[0,2:4],s[0,2:4]+d.net_acceleration(c),atol=1e-12)
        np.testing.assert_allclose(out[0,:2],s[0,2:4]+d.net_acceleration(c)*9/16,atol=1e-12)

    def test_pair_energy_momentum_and_wall(self):
        s=np.array([[-4.5,0,1.,.2,4,0,1],[4.5,0,-1.,.2,4,1,1]],float)
        out,e=d.step(s,np.zeros(5));self.assertGreater(e['pair_impulses'],0)
        np.testing.assert_allclose(out[:,2:4].sum(0),s[:,2:4].sum(0),atol=1e-12)
        self.assertAlmostEqual(float((out[:,2:4]**2).sum()),float((s[:,2:4]**2).sum()),places=12)
        np.testing.assert_allclose(out[:,2],[-1.,1.],atol=1e-12)
        one=np.array([[27.,0,1.,.2,4,0,1]],float);nxt,e=d.step(one,np.zeros(5))
        self.assertGreater(e['wall_impulses'],0);np.testing.assert_allclose(nxt[0,:4],[27.,.2,-1.,.2],atol=1e-12)

    def test_joint_rotation_and_fixed_condition_counterexample(self):
        s,c,a,_=d.sample(0,120,'train','variable_force',4)
        st,ev=d.rollout(s,c,a)
        for k in range(4):
            rs,re=d.rollout(d.rotate_state(s,k),d.rotate_context(c,k),d.rotate_vectors(a,k))
            np.testing.assert_allclose(rs,np.stack([d.rotate_state(v,k) for v in st]),atol=2e-10)
            np.testing.assert_array_equal(ev,re)
        c=np.array([0.,.05,0.,0.,0.]);s=np.array([[0.,0.,0.,0.,4.,0.,1.]])
        one,_=d.step(s,c);wrong,_=d.step(d.rotate_state(s,1),c)
        self.assertGreater(np.linalg.norm(wrong[:,:4]-d.rotate_state(one,1)[:,:4]),.05)

    def test_renderer_rotation_and_slot_order(self):
        s,c,a,_=d.sample(3,120,'test','isotropic',4);ids=np.arange(4);perm=[2,0,3,1]
        im=d.render(s)
        for k in range(4):
            out=d.render(d.rotate_state(s,k))
            np.testing.assert_array_equal(out['image'],np.rot90(im['image'],k))
            np.testing.assert_array_equal(out['segmentation'],np.rot90(im['segmentation'],k))
        np.testing.assert_array_equal(d.render(s[perm],ids[perm])['image'],im['image'])
        out,_=d.rollout(s,c,a);re,_=d.rollout(s[perm],c,a[:,perm],ids[perm])
        np.testing.assert_allclose(re,out[:,perm],atol=1e-12)

    def test_unidentifiable_force_decomposition_and_reproducibility(self):
        s,c,a,t=d.sample(4,120,'train','variable_force')
        ss,cc,aa,tt=d.sample(4,120,'train','variable_force')
        np.testing.assert_array_equal(s,ss);np.testing.assert_array_equal(a,aa);self.assertEqual(t,tt)
        other=c.copy();other[:2]+=[.03,-.07];other[2:4]-=[.03,-.07]
        left,_=d.rollout(s,c,a);right,_=d.rollout(s,other,a)
        np.testing.assert_allclose(left,right,atol=1e-11)
        self.assertEqual(d.initial_orbit_hash(s,c,a[3]),d.initial_orbit_hash(s,other,a[3]))
        self.assertNotEqual(d.initial_orbit_hash(s,c,a[3]),d.initial_orbit_hash(ss,cc,aa[3]+.1))


if __name__=='__main__':unittest.main()
