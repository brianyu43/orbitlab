"""Close A05's matched-evaluation gap using frozen 3k/6k checkpoints."""
import json
import time
import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from common import HERE,ROOT,dump,r,o
from diagnostics import probe_data

NAME='probe_step_comparison_v1'


def main():
    torch.set_num_threads(4);root=HERE/'reports'/NAME;root.mkdir(exist_ok=False)
    settings={'steps':[3000,6000],'kinds':['aug','equivariant'],'seeds':[0,1,2],
        'linear_C':[.01,.1,1,10],'rbf_C':[.1,1,10],'splits':['train','val','test','ood'],
        'space':'full','device':'cpu','selection':'StandardScaler and classifiers fit on train only; maximum validation accuracy, first C wins ties. All test/OOD results retained.',
        'source_sha256':r.sha(__file__),'probe_data_manifest_sha256':r.sha(HERE/'data/probe_v1/manifest.json'),
        'scope':'Paired evaluation inputs, not a continuation from the 3k checkpoint. Full-space probes; same fixed hyperparameter grid as the original 6k diagnostic.'}
    dump(root/'protocol.json',settings);results=[]
    for steps in settings['steps']:
        for kind in settings['kinds']:
            for seed in settings['seeds']:
                folder=root/f'{kind}_{steps}_s{seed}';folder.mkdir()
                ck=ROOT/f"runs/{'primary' if steps==3000 else 'repair'}_{kind}_n1024_s{seed}/ae.pt"
                run=json.loads((ck.parent/'run.json').read_text());assert run['steps']==steps
                model,_=r.load_model(ck,torch.device('cpu'));model.eval();features={};labels={}
                with torch.no_grad():
                    for split in settings['splits']:
                        images,y=probe_data(split)
                        features[split]=np.concatenate([model.encode(o.rotate(images[i:i+64],k)).numpy().reshape(len(images[i:i+64]),-1) for k in range(4) for i in range(0,len(images),64)])
                        labels[split]=np.c_[np.tile(y.numpy(),(4,1)),np.repeat(np.arange(4),len(y))]
                scaler=StandardScaler().fit(features['train']);scaled={s:scaler.transform(v).astype(np.float64) for s,v in features.items()}
                np.savez_compressed(folder/'features.npz',**features)
                np.savez_compressed(folder/'labels.npz',**labels)
                fitted={'scaler':scaler,'estimators':{}};saved={};scores={}
                for target,index in [('shape',0),('color',1),('rotation',2)]:
                    for family in ['linear','rbf'] if target!='rotation' else ['linear']:
                        values=[]
                        for C in settings[family+'_C']:
                            estimator=LogisticRegression(C=C,max_iter=2000) if family=='linear' else SVC(C=C,kernel='rbf',gamma='scale')
                            estimator.fit(scaled['train'],labels['train'][:,index]);key=f'{target}_{family}_C{C}'
                            fitted['estimators'][key]=estimator
                            prediction=estimator.predict(scaled['val']);saved[key+'_val']=prediction
                            values.append(float(np.mean(prediction==labels['val'][:,index])))
                        selected=int(np.argmax(values));C=settings[family+'_C'][selected];key=f'{target}_{family}_C{C}'
                        score={'C':C,'validation_grid_accuracies':values,'validation_accuracy':values[selected]}
                        for split in ['test','ood']:
                            pred=fitted['estimators'][key].predict(scaled[split]);saved[key+'_'+split]=pred
                            scene=(pred==labels[split][:,index]).reshape(4,-1).mean(0)
                            score[split]={'mean':float(scene.mean()),'base_scenes':len(scene)}
                        scores[target+'/'+family]=score
                joblib.dump(fitted,folder/'fitted.joblib',compress=1)
                np.savez_compressed(folder/'predictions.npz',**saved)
                record={'steps':steps,'kind':kind,'seed':seed,'checkpoint':str(ck.relative_to(ROOT)),'checkpoint_sha256':r.sha(ck),
                    'scores':scores,'files':{p.name:r.sha(p) for p in folder.iterdir()},'protocol_sha256':r.sha(root/'protocol.json')}
                dump(folder/'result.json',record);results.append(record)
                dump(root/'progress.json',{'models_completed':len(results),'expected_models':12})
                print('matched probe',kind,steps,seed,len(results),'/12',flush=True)
    dump(root/'summary.json',{'results':results,'models':12,'source_sha256':r.sha(__file__),'completed_unix':time.time(),
        'scope':'Full-space probe re-fitting on frozen checkpoints and identical new evaluation inputs; no AE or flow retraining.'})


if __name__=='__main__':main()
