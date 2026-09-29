"""Supervised spatial-crop intervention with explicit known-palette prior.

Both arms receive only an RGB-derived color mask. Ground-truth states are
training targets or scoring labels, never inference inputs. This is not
unsupervised object discovery or a claim about arbitrary photographs.
"""
from pathlib import Path
import sys,json,argparse,time
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'generation_recovery'),str(ROOT/'work'),str(ROOT/'followup')]
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import orbitlab as o
from p0 import sha,dump
import object_world as world
from object_oracle_study import image_from_slots
from object_image_edit import apply_commands,select_target,score,load_controller
from object_edit_confirmation import planned_tasks,visible_click
BASE=HERE/'perception_detail'

def freeze():
    p=BASE/'protocol.json'
    cfg={'source_sha256':sha(Path(__file__)),'arms':['binary','alpha'],'training_steps':6000,'batch':64,
         'optimizer':'AdamW lr .001 weight_decay .0001','initializations':[0,1,2],
         'confirmation_seeds':[885201,885202,885203],'splits':['test','ood','count3','count4','occlusion'],
         'scenes_per_split':128,'train_data':'followup/data/object_world_v1/train; same 1024 scenes for both arms',
         'input_prior':'Known six RGB palette directions; distinct colors in source scenes; foreground max>13; each color >=3 pixels. No GT masks, centers or count at inference.',
         'crop':'Both arms use 33x33 mask centered on RGB-mask bounding-box midpoint, zero padded. Predicted center offset is learned.',
         'comparison':'binary crop versus palette-projected intensity crop; same CNN and full four-way pose supervision.',
         'targets':'Supervised shape, pose, radius 5..8 and xy. All shapes use full four-way pose cross entropy. Raster asymmetry can distinguish geometric half-turns.',
         'scoring':'Original strict factor checks retained, including pose; geometric equivalence is supplementary only. No target-dependent inference.',
         'controllers':['analytic','learned_object_seed_matched'],'limits':'Shared original training dataset; three new evaluation seeds do not imply three independent training datasets. Palette-assisted, synthetic domain only.',
         'dependencies':{name:sha(ROOT/'followup'/name) for name in ['object_world.py','object_image_edit.py','object_oracle_study.py','object_edit_confirmation.py']}}
    if p.exists():assert json.loads(p.read_text())==cfg
    else:dump(p,cfg)
    return cfg

def masks_from_rgb(image):
    rgb=np.asarray(image,dtype=np.float32);palette=np.asarray(o.PALETTE,dtype=np.float32)
    unit=rgb/np.maximum(np.linalg.norm(rgb,axis=-1,keepdims=True),1e-6)
    pu=palette/np.linalg.norm(palette,axis=-1,keepdims=True)
    colors=(unit@pu.T).argmax(-1);foreground=rgb.max(-1)>13
    return [(c,((colors==c)&foreground).astype(np.float32)) for c in range(6) if ((colors==c)&foreground).sum()>=3]

def inputs(image,kind):
    tensors=[];records=[]
    for color,mask in masks_from_rgb(image):
        yy,xx=np.nonzero(mask);cx=int(np.rint((xx.min()+xx.max())/2));cy=int(np.rint((yy.min()+yy.max())/2))
        if kind in ['binary','alpha']:
            intensity=np.clip((np.asarray(image,dtype=np.float32)@np.asarray(o.PALETTE[color],dtype=np.float32))/(np.asarray(o.PALETTE[color],dtype=np.float32)**2).sum(),0,1)*mask
            value=intensity if kind=='alpha' else mask
            padded=np.pad(value,16);patch=padded[cy:cy+33,cx:cx+33];offset=[cx,cy];scale=8.
        else:
            patch=F.interpolate(torch.from_numpy(mask)[None,None],size=(33,33),mode='bilinear',align_corners=False)[0,0].numpy()
            offset=[31.5,31.5];scale=31.5
        tensors.append(torch.from_numpy(patch.copy())[None]);records.append({'color':color,'offset':offset,'scale':scale,'area':int(mask.sum())})
    return torch.stack(tensors) if tensors else torch.empty(0,1,33,33),records

class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.SiLU(),nn.Conv2d(16,32,3,2,1),nn.SiLU(),
                               nn.Conv2d(32,32,3,2,1),nn.SiLU(),nn.Flatten(),nn.Linear(800,96),nn.SiLU(),nn.Linear(96,14))
    def forward(self,x):return self.net(x)

def training_data(kind):
    path=ROOT/'followup/data/object_world_v1/train/scenes.npz'
    archive=np.load(path);meta=json.loads(path.with_name('scenes.json').read_text())
    xs=[];ys=[]
    for image,record in zip(archive['images'],meta):
        xx,det=inputs(image,kind);bycolor={v['color']:v for v in record['objects']}
        for x,d in zip(xx,det):
            v=bycolor[d['color']];xs.append(x)
            ys.append([v['shape'],v['pose'],v['radius']-5,(v['x']-d['offset'][0])/d['scale'],(v['y']-d['offset'][1])/d['scale']])
    return torch.stack(xs),torch.tensor(ys,dtype=torch.float32)

def train(kind,seed):
    freeze();folder=BASE/f'runs/{kind}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'run.json').exists():
        result=json.loads((folder/'run.json').read_text());assert result['checkpoint_sha256']==sha(folder/'model.pt');return
    torch.manual_seed(885000+seed);model=Reader();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    x,y=training_data(kind);rng=torch.Generator().manual_seed(885100+seed)
    progress=folder/'progress.pt';start_step=0;prior=0.;logs=[]
    if progress.exists():
        ck=torch.load(progress,weights_only=True);assert ck['source_sha256']==sha(Path(__file__))
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);rng.set_state(ck['rng']);start_step=ck['step'];prior=ck['seconds'];logs=ck['logs']
    start=time.perf_counter()
    for step in range(start_step+1,6001):
        idx=torch.randint(len(x),(64,),generator=rng);target=y[idx];pred=model(x[idx])
        shape=target[:,0].long();pose=target[:,1].long();prob=pred[:,4:8].log_softmax(-1)
        pose_loss=-prob.gather(1,pose[:,None])[:,0]
        loss=F.cross_entropy(pred[:,:4],shape)+pose_loss.mean()+F.cross_entropy(pred[:,8:12],target[:,2].long())+4*F.mse_loss(pred[:,12:14],target[:,3:5])
        assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if step%500==0:
            seconds=prior+time.perf_counter()-start;logs.append({'step':step,'loss':float(loss.detach()),'seconds':seconds})
            ck={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng.get_state(),'step':step,'seconds':seconds,'logs':logs,'source_sha256':sha(Path(__file__))}
            tmp=progress.with_suffix('.tmp');torch.save(ck,tmp);tmp.replace(progress)
            print('perception',kind,seed,logs[-1],flush=True)
    torch.save({'state_dict':model.state_dict(),'kind':kind,'seed':seed},folder/'model.pt')
    dump(folder/'run.json',{'checkpoint_sha256':sha(folder/'model.pt'),'source_sha256':sha(Path(__file__)),
                          'steps':6000,'seconds':prior+time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),'logs':logs,
                          'training_sha256':sha(ROOT/'followup/data/object_world_v1/train/scenes.npz')})

@torch.no_grad()
def infer(images,model,kind):
    states=[]
    dirs=torch.tensor([[1.,0],[0,-1],[-1,0],[0,1]])
    for image in images:
        x,det=inputs(image,kind);slots=torch.zeros(4,16)
        if len(x):
            p=model(x);order=sorted(range(len(det)),key=lambda i:-det[i]['area'])[:4];order.sort()
            for j,i in enumerate(order):
                d=det[i];shape=int(p[i,:4].argmax());pose=int(p[i,4:8].argmax())
                slots[j,shape]=1;slots[j,4+d['color']]=1
                xy=p[i,12:14]*d['scale']+torch.tensor(d['offset'])
                slots[j,10:12]=(xy-31.5)/31.5;slots[j,12]=(p[i,8:12].argmax()+5)/8
                slots[j,13:15]=dirs[pose];slots[j,15]=1
        states.append(slots)
    return torch.stack(states)

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['binary','alpha'],required=True);p.add_argument('--seed',type=int,default=0)
    args=p.parse_args();torch.set_num_threads(2);train(args.kind,args.seed)

if __name__=='__main__':main()
