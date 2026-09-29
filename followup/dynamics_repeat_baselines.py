"""Original teacher-forced physical controls on both new datasets, with replay."""
import argparse
import csv
import json
import time
import numpy as np
from common import HERE,dump,r
from dynamics_repeat_data import NAME,freeze as parent_freeze
from dynamics_physical_baselines import predict
from verify_dynamics_observation_eval import physical_reference

CONFIG='dynamics_repeat_baselines_v1'
KINDS=['inertial','force_wall']
CATEGORIES=['all','no_contact','contact','pair','wall_only']
METRICS=['position_mae_pixels','velocity_mae_pixels_per_frame']


def freeze():
    parent=parent_freeze();p=HERE/f'configs/{CONFIG}.json'
    if not p.exists():
        old=json.loads((HERE/'configs/dynamics_state_study_v1.json').read_text())
        dump(p,{'data_seeds':parent['new_data_seeds'],'modes':old['modes'],'splits':old['evaluation_splits'],
            'force_ood_modes':old['force_ood_modes'],'parent_sha256':r.sha(HERE/f'configs/{NAME}.json'),
            'sources':{f:r.sha(HERE/f) for f in ['dynamics_repeat_baselines.py','dynamics_physical_baselines.py','verify_dynamics_observation_eval.py']},
            'frozen_unix':time.time(),'scope':'Repeat the original inertial and known force/wall controls at every heldout transition. No pair interactions, no learning, no future input.'})
    c=json.loads(p.read_text());assert c['parent_sha256']==r.sha(HERE/f'configs/{NAME}.json')
    for f,sha in c['sources'].items():assert r.sha(HERE/f)==sha
    assert json.loads((HERE/f'reports/{NAME}/data_verification.json').read_text())['all_passed']
    return c


def conditions(c):
    return [(ds,mode,split) for ds in c['data_seeds'] for mode in c['modes'] for split in c['splits']
            if split!='force_ood' or mode in c['force_ood_modes']]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args();c=freeze()
    root=HERE/f'reports/{NAME}/state_baselines';root.mkdir(parents=True,exist_ok=True)
    results=[];files=[];predictions=rows_checked=aggregate_checks=0
    if args.verify:
        source=json.loads((root/'summary.json').read_text());assert source['config_sha256']==r.sha(HERE/f'configs/{CONFIG}.json')
        assert len(source['files'])==26 and len(source['results'])==260
        assert {(x['data_seed'],x['mode'],x['split']) for x in source['files']}==set(conditions(c))
    for ds,mode,split in conditions(c):
        path=HERE/f'data/{NAME}/d{ds}/world/{mode}/{split}/trajectories.npz';folder=root/f'd{ds}/{mode}/{split}'
        with np.load(path) as a:s=a['states'].copy();contexts=a['contexts'].copy();actions=a['actions'].copy();events=a['events'].copy()
        n,t=s.shape[0],s.shape[1]-1;assert t==64
        ctx=np.broadcast_to(contexts[:,None,:],(n,t,5));live=s[:,:-1,:,4]>0
        if args.verify:
            receipt=next(x for x in source['files'] if (x['data_seed'],x['mode'],x['split'])==(ds,mode,split))
            assert receipt['data_sha256']==r.sha(path)
            for f,sha in receipt['files'].items():assert r.sha(folder/f)==sha
            with np.load(folder/'predictions.npz') as a:out={k:a[k].copy() for k in a.files}
            assert set(out)==set(KINDS)
            state=s[:,:-1].reshape(-1,4,7);context=np.c_[ctx.reshape(-1,5)[:,:2]+ctx.reshape(-1,5)[:,2:4],ctx.reshape(-1,5)[:,4]]
            for kind in KINDS:
                expected=physical_reference(state,context,actions.reshape(-1,4,2),kind).reshape(n,t,4,4)
                np.testing.assert_allclose(out[kind],expected,atol=1e-12,rtol=0);predictions+=n*t
            records=list(csv.DictReader((folder/'rows.csv').open()))
        else:
            folder.mkdir(parents=True,exist_ok=False)
            out={kind:predict(s[:,:-1],ctx,actions,kind) for kind in KINDS};records=[]
        for kind,pred in out.items():
            delta=np.abs(pred-s[:,1:,:,:4]);position=(delta[...,:2].sum(-1)*live).sum(-1)/(2*live.sum(-1))
            velocity=(delta[...,2:].sum(-1)*live).sum(-1)/(2*live.sum(-1));expected_rows=[]
            for i in range(n):
                pair=events[i,:,0]>0;wall=events[i,:,1]>0
                selections=[np.ones(t,bool),~(pair|wall),pair|wall,pair,wall&~pair]
                for category,selection in zip(CATEGORIES,selections):
                    if selection.any():expected_rows.append({'kind':kind,'episode':i,'category':category,'transitions':int(selection.sum()),
                        METRICS[0]:float(position[i,selection].mean()),METRICS[1]:float(velocity[i,selection].mean())})
            if args.verify:
                actual=[x for x in records if x['kind']==kind];assert len(actual)==len(expected_rows)
                for x,y in zip(actual,expected_rows):
                    for k,v in y.items():
                        if isinstance(v,str):assert x[k]==v
                        else:assert abs(float(x[k])-v)<1e-12
                    rows_checked+=1
            else:records.extend(expected_rows)
            for category in CATEGORIES:
                selected=[v for v in expected_rows if v['category']==category]
                m={key:r.bootstrap([v[key] for v in selected]) for key in METRICS}
                item={'data_seed':ds,'mode':mode,'split':split,'kind':kind,'category':category,'episodes':len(selected),'metrics':m}
                if args.verify:
                    target=[v for v in source['results'] if all(v[k]==item[k] for k in ['data_seed','mode','split','kind','category'])]
                    assert len(target)==1 and target[0]==item;aggregate_checks+=2
                else:results.append(item)
        if not args.verify:
            np.savez_compressed(folder/'predictions.npz',**out);r.write_csv(folder/'rows.csv',records)
            files.append({'data_seed':ds,'mode':mode,'split':split,'data_sha256':r.sha(path),
                          'files':{f:r.sha(folder/f) for f in ['predictions.npz','rows.csv']}})
        print('verified' if args.verify else 'evaluated','repeat baseline',ds,mode,split,flush=True)
    if args.verify:
        dump(root/'verification.json',{'all_passed':True,'prediction_transitions_replayed':predictions,'rows_checked':rows_checked,
            'aggregate_and_interval_checks':aggregate_checks,'summary_sha256':r.sha(root/'summary.json'),
            'verifier_sha256':r.sha(__file__),'independent_reference_sha256':r.sha(HERE/'verify_dynamics_observation_eval.py'),
            'scope':'All heldout transitions replayed with separate physical-reference code; same controls as original C03.'})
    else:dump(root/'summary.json',{'results':results,'files':files,'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json'),'scope':c['scope']})


if __name__=='__main__':main()
