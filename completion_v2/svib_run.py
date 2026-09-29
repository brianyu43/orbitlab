"""Full official representative split, fixed five-epoch small-model comparison.

Uses upstream OrbitLab preview architecture at native 128px. This is a new
controlled experiment on official data, not reproduction of every published
SVIB baseline or all twelve tasks.
"""
from pathlib import Path
import sys,json,time,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;BASE=HERE/'svib'
sys.path[:0]=[str(HERE),str(ROOT/'followup'),str(ROOT/'generation_recovery'),str(ROOT/'work')]
import numpy as np
import torch
import orbitlab as o
from svib_preview_models import PreviewPredictor
from svib_prepare import sha

def write(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(record,indent=2)+'\n')

def freeze():
    cfg={'source_sha256':sha(Path(__file__)),'model_source_sha256':sha(ROOT/'followup/svib_preview_models.py'),
         'data_manifest_sha256':sha(BASE/'data_manifest.json'),'task':'Official Hard dSprites Single Atomic, shape swap',
         'train_n':64000,'test_n':8000,'resolution':128,'kinds':['plain','c4'],'seeds':[0,1,2],
         'epochs':5,'batch':32,'steps':10000,'optimizer':'Adam .0005, grad_clip 1',
         'sampling':'Deterministic shuffled complete epochs, seed 883100+init; paired between arms',
         'initialization':'883000+init paired between arms','checkpoint_selection':'Fixed last checkpoint, no test tuning',
         'device':'mps','cpu_threads':2,'loss':'Full-image MSE',
         'metrics':['full image MSE','changed pixel MSE','preserved pixel MSE','foreground MSE','identity MSE'],
         'limits':'Full representative official data split, not full 12-task suite or published model reproduction. Fixed five epochs, not a claim of convergence. Same parameters/updates but different C4 compute.'}
    path=BASE/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==cfg
    else:write(path,cfg)
    return cfg

def batch(array,indices,device):
    return torch.from_numpy(np.asarray(array[indices]).copy()).permute(0,3,1,2).float().to(device)/255

def train(kind,seed,device):
    cfg=freeze();out=BASE/f'runs/{kind}_s{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists():
        report=json.loads((out/'run.json').read_text());assert report['checkpoint_sha256']==sha(out/'model.pt');return
    source=np.load(BASE/'train_source.npy',mmap_mode='r');target=np.load(BASE/'train_target.npy',mmap_mode='r')
    torch.manual_seed(883000+seed);model=PreviewPredictor(kind).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.0005)
    rng=np.random.default_rng(883100+seed);orders=np.stack([rng.permutation(64000) for _ in range(5)])
    progress=out/'progress.pt';completed=0;prior=0.;logs=[]
    if progress.exists():
        ck=torch.load(progress,map_location='cpu',weights_only=True);assert ck['source_sha256']==sha(Path(__file__))
        model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer']);completed=ck['step'];prior=ck['seconds'];logs=ck['logs']
    start=time.perf_counter()
    for step in range(completed,10000):
        epoch,within=divmod(step,2000);idx=orders[epoch,within*32:(within+1)*32]
        x=batch(source,idx,device);y=batch(target,idx,device)
        pred=model(x);loss=(pred-y).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('SVIB training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        if not torch.isfinite(norm):raise FloatingPointError('SVIB training gradient')
        optimizer.step()
        if (step+1)%500==0:
            torch.mps.synchronize();seconds=prior+time.perf_counter()-start;logs.append({'step':step+1,'loss':float(loss.detach()),'seconds':seconds})
            ck={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),
                'step':step+1,'seconds':seconds,'logs':logs,'source_sha256':sha(Path(__file__))}
            tmp=progress.with_suffix('.tmp');torch.save(ck,tmp);tmp.replace(progress);print('SVIB',kind,seed,logs[-1],flush=True)
    torch.mps.synchronize()
    torch.save({'kind':kind,'seed':seed,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},out/'model.pt')
    write(out/'run.json',{'checkpoint_sha256':sha(out/'model.pt'),'protocol_sha256':sha(BASE/'protocol.json'),'source_sha256':sha(Path(__file__)),
                         'seconds':prior+time.perf_counter()-start,'steps':10000,'epochs':5,'seed':seed,'kind':kind,'logs':logs,
                         'order_sha256':hashlib.sha256(orders.tobytes()).hexdigest(),'parameters':sum(v.numel() for v in model.parameters())})

def row_metrics(pred,source,target):
    error=(pred-target).square().mean(1);changed=(source!=target).any(1);foreground=(source.amax(1)>0.05)|(target.amax(1)>0.05)
    def masked(mask):return (error*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
    return torch.stack([error.mean((1,2)),masked(changed),masked(~changed),masked(foreground),
                        (source-target).square().mean((1,2,3)),changed.sum((1,2))],1)

@torch.no_grad()
def evaluate(kind,seed,device):
    out=BASE/f'runs/{kind}_s{seed}'
    if (out/'evaluation.json').exists():return
    ck=torch.load(out/'model.pt',weights_only=True,map_location='cpu');model=PreviewPredictor(kind).to(device);model.load_state_dict(ck['state_dict']);model.eval()
    source=np.load(BASE/'test_source.npy',mmap_mode='r');target=np.load(BASE/'test_target.npy',mmap_mode='r');rows=[];samples=[]
    for first in range(0,8000,32):
        idx=np.arange(first,min(first+32,8000));x=batch(source,idx,device);y=batch(target,idx,device);pred=model(x)
        rows.append(row_metrics(pred,x,y).cpu())
        if first<64:samples.append({'source':x.cpu(),'target':y.cpu(),'prediction':pred.cpu()})
    values=torch.cat(rows).numpy();np.save(out/'test_rows.npy',values);torch.save(samples,out/'replay_samples.pt')
    example=torch.stack([v for i in range(8) for v in [samples[0]['source'][i],samples[0]['target'][i],samples[0]['prediction'][i]]])
    o.grid(example,out/'examples.png',6)
    changed=values[:,5]>0
    metrics={'mse':float(values[:,0].mean()),'identity_mse':float(values[:,4].mean()),
             'changed_pixel_mse_on_changed_scenes':float(values[changed,1].mean()),
             'preserved_pixel_mse':float(values[:,2].mean()),'foreground_mse':float(values[:,3].mean()),
             'changed_scenes':int(changed.sum()),'unchanged_scene_mse':float(values[~changed,0].mean()) if (~changed).any() else None}
    write(out/'evaluation.json',{'kind':kind,'seed':seed,'metrics':metrics,'n':8000,'checkpoint_sha256':sha(out/'model.pt'),
                                'source_sha256':sha(Path(__file__)),'rows_sha256':sha(out/'test_rows.npy'),'replay_sha256':sha(out/'replay_samples.pt')})
    print('SVIB evaluation',kind,seed,metrics,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['plain','c4']);parser.add_argument('--seed',type=int)
    args=parser.parse_args();torch.set_num_threads(2)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required; run authorized host execution, no silent device change')
    device=torch.device('mps')
    for seed in ([args.seed] if args.seed is not None else range(3)):
        for kind in ([args.kind] if args.kind else ['plain','c4']):
            train(kind,seed,device);evaluate(kind,seed,device);torch.mps.empty_cache()
