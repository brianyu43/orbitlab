"""Train-only probes, val-only regularization, untouched test/OOD evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.preprocessing import StandardScaler
import research as r
import orbitlab as o


def fit(root,ae_path):
    root=Path(root);data={s:np.load(root/f'{s}_latents.npz') for s in ['train','val','test','ood']}
    raw={s:d['z'].reshape(len(d['z']),-1).astype(np.float64) for s,d in data.items()}
    result={'selection':'fit train, choose validation, report test/ood; base-scene bootstrap','features':{}}
    ae,ck=r.load_model(ae_path,torch.device('cpu'));torch.set_num_threads(4)
    spaces=['full']+(['m0','m2','m13'] if ck['model'] in ['equivariant','invariant'] else [])
    for space in spaces:
        unscaled={}
        for s,d in data.items():
            f=np.fft.fft(d['z'],axis=1,norm='ortho')
            unscaled[s]=raw[s] if space=='full' else (f[:,int(space[1])].real if space!='m13' else np.concatenate([f[:,1].real,f[:,1].imag],1))
        scaler=StandardScaler().fit(unscaled['train']);features={s:scaler.transform(x) for s,x in unscaled.items()}
        res={}
        for target in ['shape','color','rotation']:
            labels={s:d['rotation'] if target=='rotation' else d['labels'][:,0 if target=='shape' else 1] for s,d in data.items()}
            best=None
            for c in [.01,.1,1,10]:
                clf=LogisticRegression(C=c,max_iter=2000).fit(features['train'],labels['train'])
                acc=float(np.mean(clf.predict(features['val'])==labels['val']))
                if best is None or acc>best[0]:best=(acc,c,clf)
            res[target]={'C':best[1],'validation_accuracy':best[0]}
            for s in ['test','ood']:
                pred=best[2].predict(features[s]);correct=pred==labels[s]
                res[target][s]=r.bootstrap(correct.reshape(4,-1).mean(0))
                if target=='rotation':
                    symmetric=(data[s]['labels'][:,0]==3)&((pred-labels[s])%2==0)
                    res[target][s+'_zigzag_mod180']=r.bootstrap((correct|symmetric).reshape(4,-1).mean(0))
        singular=np.linalg.svd(unscaled['train']-unscaled['train'].mean(0),compute_uv=False)**2
        prob=singular/max(singular.sum(),1e-12);res['effective_rank']=float(np.exp(-sum(p*np.log(p) for p in prob if p>0))) if singular.sum()>1e-12 else 0
        result['features'][space]=res
    scaler=StandardScaler().fit(raw['train']);features={s:scaler.transform(x) for s,x in raw.items()}
    pairs=lambda x:(x,np.roll(x.reshape(4,-1,x.shape[1]),-1,axis=0).reshape(x.shape))
    x,y=pairs(features['train']);vx,vy=pairs(features['val']);best=None
    for alpha in [1e-4,1e-2,1,100]:
        a=np.linalg.solve(x.T@x+alpha*np.eye(x.shape[1]),x.T@y);err=float(np.mean((vx@a-vy)**2))
        if best is None or err<best[0]:best=(err,alpha,a)
    a=best[2];action={'alpha':best[1],'validation_mse':best[0],'A4_identity_relative_frobenius':float(np.linalg.norm(np.linalg.matrix_power(a,4)-np.eye(64))/8)}
    for s in ['test','ood']:
        xx,yy=pairs(features[s]);error=((xx@a-yy)**2).mean(1)/max(np.mean(yy**2),1e-12)
        closure=((xx@np.linalg.matrix_power(a,4)-xx)**2).mean(1)/max(np.mean(xx**2),1e-12)
        action[s]={'relative_mse':r.bootstrap(error.reshape(4,-1).mean(0)),'in_data_closure_relative_mse':r.bootstrap(closure.reshape(4,-1).mean(0))}
        latent=torch.from_numpy(scaler.inverse_transform(xx@a).astype(np.float32)).reshape(-1,4,16)
        base,_,_=r.load_data(s,len(xx)//4);truth=torch.cat([o.rotate(base,(k+1)%4) for k in range(4)])
        values=[]
        with torch.no_grad():
            for i in range(0,len(latent),32):values.extend((ae.decode(latent[i:i+32])-truth[i:i+32]).square().mean((1,2,3)).tolist())
        action[s]['decoded_target_mse']=r.bootstrap(np.array(values).reshape(4,-1).mean(0))
        if s=='test':
            with torch.no_grad():o.grid(torch.cat([base[:8],ae.decode(latent[:8]),truth[:8]]),root/'learned_action.png',8)
    result['learned_A90']=action
    result['note']='C4 band labels are representation components, not proven semantic disentanglement. Zigzag mod180 scores identify object pose equivalence; scene positions still rotate.'
    r.dump(root/'probes.json',result)
    np.savez_compressed(root/'learned_A90.npz',A=a,mean=scaler.mean_,scale=scaler.scale_)
    print('probes',root,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--ae',required=True);a=p.parse_args();fit(a.input,a.ae)
