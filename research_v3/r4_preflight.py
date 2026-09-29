"""Meaningful R4 invariants checked before training, no evaluation selection."""
import json, time
import torch
from v3_common import HERE, sha, dump
from r4_data import BASE, DEVELOPMENT
from r4_core import (o, CANDIDATES, ae_model, decoder_model, FactorBridge,
    from_variant, oracle_codes, edge_loss, decoder_loss, state_hash)
from r4_train import data

def main():
    torch.set_num_threads(2);torch.manual_seed(6029)
    x,y,meta=data(DEVELOPMENT,'train');x=x[:4];y=y[:4];meta=meta[:4]
    z=torch.randn(4,4,16);checks=[]
    for rep in ['base','spatial']:
        model=ae_model(rep);encoded=model.encode(x);pred=model(x)
        assert encoded.shape==(4,4,16) and pred.shape==x.shape and torch.isfinite(pred).all()
        enc=max(float((model.encode(o.rotate(x,k))-o.rho(encoded,k)).abs().max().detach()) for k in range(4))
        dec=max(float((model.decode(o.rho(encoded,k))-o.rotate(pred,k)).abs().max().detach()) for k in range(4))
        assert enc<2e-5 and dec<2e-5
        loss=(pred-x).square().mean();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        checks.append({'model':'ae_'+rep,'encoder_c4_max':enc,'decoder_c4_max':dec,'parameters':sum(p.numel() for p in model.parameters()),'backward_finite':True})
    factors=oracle_codes(y,meta,0)
    for k in range(4):torch.testing.assert_close(oracle_codes(y,meta,k),o.rho(factors,k),rtol=0,atol=0)
    center=factors[...,10:12]
    for candidate in CANDIDATES:
        torch.manual_seed(77);model=decoder_model(candidate);pred=model(z)
        assert pred.shape==x.shape and pred.min()>=0 and pred.max()<=1
        eq=max(float((model(o.rho(z,k))-o.rotate(pred,k)).abs().max().detach()) for k in range(4));assert eq<2e-5
        loss,parts=decoder_loss(candidate,model,z,x,center);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        checks.append({'model':candidate,'decoder_c4_max':eq,'parameters':sum(p.numel() for p in model.parameters()),'backward_finite':True})
    torch.manual_seed(920000);a=decoder_model('standard');torch.manual_seed(920000);b=decoder_model('edge');assert state_hash(a)==state_hash(b)
    for model in [FactorBridge(),from_variant('F0')]:
        args=() if isinstance(model,FactorBridge) else (torch.rand(4),y)
        pred=model(z,*args);eq=max(float((model(o.rho(z,k),*args)-o.rho(pred,k)).abs().max().detach()) for k in range(4));assert eq<2e-5
        pred.square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        checks.append({'model':type(model).__name__,'c4_max':eq,'parameters':sum(p.numel() for p in model.parameters()),'backward_finite':True})
    assert edge_loss(x,x).item()==0 and o.reconstruction_loss(x,x).item()==0
    # Analytic pixel-center action equals a90degree image-coordinate rotation.
    base_px=(factors[:,0,10:12]+1)*31.5
    rotated_px=(oracle_codes(y,meta,1)[:,0,10:12]+1)*31.5
    torch.testing.assert_close(rotated_px,torch.stack([base_px[:,1],63-base_px[:,0]],1),atol=4e-6,rtol=0)
    # Coordinate placement must carry center gradients despite grid_sample.
    c=decoder_model('coordinate');c(z).sum().backward();assert sum(float(p.grad.abs().sum()) for p in c.xy.parameters())>0
    dump(BASE/'preflight.json',{'passed':True,'checks':checks,'oracle_c4_exact':True,'standard_edge_initial_states_identical':True,'pixel_center_rotation':True,'coordinate_head_gradient_nonzero':True,'image_identity_losses_zero':True,'source_sha256':sha(__file__),'core_sha256':sha(HERE/'r4_core.py'),'train_sha256':sha(HERE/'r4_train.py'),'time':time.time()})
    print(json.dumps(checks),flush=True)

if __name__=='__main__':main()
