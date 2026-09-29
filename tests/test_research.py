import hashlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'work'))
import torch,numpy as np
import research as r,orbitlab as o

torch.set_num_threads(4)
class ResearchChecks(unittest.TestCase):
 def test_data_provenance_and_orbit_disjointness(self):
  manifest=json.loads((r.DATA/'manifest.json').read_text());hashes=[]
  for split,record in manifest['splits'].items():
   self.assertEqual(r.sha(r.DATA/f'{split}.npz'),record['archive_sha256'])
   self.assertEqual(r.sha(r.DATA/f'{split}_metadata.json'),record['metadata_sha256'])
   x,y,meta=r.load_data(split);self.assertEqual(len(x),record['n'])
   self.assertTrue(all((int(k)==int(c))==(split=='ood') for k,c in y))
   hashes.extend(row['orbit_sha256'] for row in meta)
   for i in [0,len(x)-1]:
    image=(x[i].permute(1,2,0).numpy()*255).round().astype(np.uint8)
    self.assertEqual(r.orbit_hash(image),r.orbit_hash(np.rot90(image)))
    self.assertEqual(r.orbit_hash(image),meta[i]['orbit_sha256'])
  self.assertEqual(len(hashes),len(set(hashes)))
  a,_,_=r.load_data('train',256);b,_,_=r.load_data('train',1024);self.assertTrue(torch.equal(a,b[:256]))
 def test_both_model_widths_and_all_group_elements(self):
  torch.manual_seed(45);x=torch.rand(2,3,64,64)
  for width in [16,31]:
   ae=r.ResearchAE('equivariant',width);z=ae.encode(x)
   for k in range(4):
    torch.testing.assert_close(ae.encode(o.rotate(x,k)),o.rho(z,k),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(ae.decode(o.rho(z,k)),o.rotate(ae.decode(z),k),atol=1e-6,rtol=1e-5)
    for j in range(4):self.assertTrue(torch.equal(o.rho(o.rho(z,k),j),o.rho(z,k+j)))
   ae(x).square().mean().backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in ae.parameters()))
 def test_metrics_identity_and_black_baseline(self):
  x,_,_=r.load_data('val',8);perfect=r.pixel_metrics(x,x);black=r.pixel_metrics(torch.zeros_like(x),x)
  self.assertEqual(float(perfect['mse'].max()),0);self.assertEqual(float(perfect['mask_iou'].min()),1)
  torch.testing.assert_close(black['mse'],black['black_mse'])
  torch.testing.assert_close(black['foreground_mae'],black['black_foreground_mae'])
 def test_parameter_matching(self):
  c=r.parameter_counts();self.assertLess(abs(c['matched_eq_parameters']/c['aug']-1),.05)
 def test_fourier_real_pair_and_invariant_decode(self):
  torch.manual_seed(4);ae=r.ResearchAE();z=torch.randn(2,4,16);f=torch.fft.fft(z,dim=1,norm='ortho');torch.testing.assert_close(f[:,1],f[:,3].conj())
  f[:,[1,3]]=0;self.assertLess(float(torch.fft.ifft(f,dim=1,norm='ortho').imag.abs().max()),1e-6)
  with torch.no_grad():p=ae.decode(z.mean(1,keepdim=True).expand_as(z))
  torch.testing.assert_close(p,o.rotate(p,1),atol=1e-6,rtol=1e-5)

if __name__=='__main__':unittest.main()
