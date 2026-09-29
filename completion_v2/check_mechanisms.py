"""Small mechanism checks before interpreting experimental results."""
from pathlib import Path
import sys,json,ast
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'completion_v2'),str(ROOT/'generation_recovery'),str(ROOT/'work'),str(ROOT/'followup')]
import torch
import numpy as np
import dynamics as d
import perception as p
from dynamics_models import rotate_state,rotate_motion
from object_world import ObjectState,render
from generation_run import edge_loss
from p0 import sha,dump

def main():
    torch.set_num_threads(1);torch.manual_seed(17);checks={}
    for path in (ROOT/'completion_v2').glob('*.py'):ast.parse(path.read_text())
    checks['all_python_sources_parse']=True
    state=torch.randn(12,4,12);state[...,4]=.75;state[...,-1]=1;state[:,3,-1]=0
    motion=torch.randn(12,4,4)*10
    a=d.bound(rotate_motion(motion,1),rotate_state(state,1),2.)
    b=rotate_motion(d.bound(motion,state,2.),1)
    checks['bounded_projection_c4_error']=float((a-b).abs().max());assert checks['bounded_projection_c4_error']<1e-6
    bounded=d.bound(motion,state,2.)
    assert torch.all(bounded[...,2:].norm(dim=-1)<=2.00001)
    assert torch.all(bounded[:,3]==0)
    checks['bounded_velocity_and_padding']=True
    x=torch.rand(3,3,64,64,requires_grad=True);target=torch.zeros_like(x);target[:,:,20:30,20:30]=.8
    loss=edge_loss(x,target);loss.backward();assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    checks['area_edge_loss_has_finite_nonzero_gradients']=True
    scene=(ObjectState(0,0,0,20,20,6,0),ObjectState(1,1,2,44,44,7,1))
    image=render(scene)['image']
    for kind in ['global','crop']:
        tensors,records=p.inputs(image,kind)
        assert tensors.shape==(2,1,33,33) and {v['color'] for v in records}=={0,2}
        assert p.Reader()(tensors).shape==(2,14)
    checks['rgb_only_preprocessing_dimensions_and_color_count']=True
    report={'checks':checks,'sources':{v.name:sha(v) for v in (ROOT/'completion_v2').glob('*.py')}}
    dest=ROOT/'completion_v2/mechanism_checks.json'
    if not dest.exists():dump(dest,report)
    print(json.dumps(checks))

if __name__=='__main__':main()
