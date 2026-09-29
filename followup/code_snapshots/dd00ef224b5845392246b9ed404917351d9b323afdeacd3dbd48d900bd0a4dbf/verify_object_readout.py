"""Replay the six supervised heads and verify the corrected categorical scoring."""
import csv
import hashlib
import json
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from common import HERE,dump,r,state_digest,digest_tensor
from object_state_readout import StateReadout,discrete_states,attribute_cost,set_loss
from object_readout_metrics_v2 import measure
from train_object_readout import labels,features
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def main():
    torch.set_num_threads(2);cp=HERE/'configs/object_readout_v1.json';c=json.loads(cp.read_text())
    root=HERE/'reports/object_readout_pose_corrected_v1';corrected=json.loads((root/'summary.json').read_text())
    assert c['model_sha256']==r.sha(HERE/'object_state_readout.py') and c['runner_sha256']==r.sha(HERE/'train_object_readout.py')
    assert c['slot_cache_manifest_sha256']==r.sha(HERE/'data/object_slot_cache_v1/manifest.json')
    assert corrected['corrected_evaluator_sha256']==r.sha(HERE/'object_readout_metrics_v2.py')
    assert corrected['runner_sha256']==r.sha(HERE/'reevaluate_object_readout.py')
    assert corrected['original_summary_sha256']==r.sha(HERE/'reports/object_readout_v1/summary.json')
    count=0;rows_checked=0
    for kind in c['kinds']:
        train=features(kind,'train');target=labels('train');mean=train.mean((0,1));std=train.std((0,1),unbiased=False).clamp_min(1e-4)
        for seed in c['seeds']:
            source=HERE/f'runs/object_readout_v1/{kind}_s{seed}';dest=root/f'{kind}_s{seed}'
            run=json.loads((source/'run.json').read_text());metrics=json.loads((dest/'metrics.json').read_text())
            assert run['config_sha256']==r.sha(cp) and run['steps']==c['steps'] and run['parameters']==22800
            assert run['feature_sha256']==digest_tensor(train) and run['training_label_sha256']==digest_tensor(target)
            assert run['checkpoint_sha256']==metrics['checkpoint_sha256']==r.sha(source/'model.pt')
            assert metrics['predictions_sha256']==r.sha(source/'predictions.pt')
            rng=torch.Generator().manual_seed(994100+seed);sample=hashlib.sha256()
            for _ in range(c['steps']):sample.update(torch.randint(len(train),(c['batch'],),generator=rng).numpy().tobytes())
            assert sample.hexdigest()==run['sample_sha256']
            optimizer=torch.load(source/'optimizer.pt',map_location='cpu',weights_only=True)
            assert torch.equal(optimizer['sampling_rng'],rng.get_state()) and all(int(v['step'])==c['steps'] for v in optimizer['optimizer']['state'].values())
            torch.manual_seed(994000+seed);model=StateReadout(mean,std);assert state_digest(model.net)==run['initial_trainable_weights_sha256']
            ck=torch.load(source/'model.pt',map_location='cpu',weights_only=True)
            torch.testing.assert_close(ck['mean'],mean,atol=0,rtol=0);torch.testing.assert_close(ck['std'],std,atol=0,rtol=0)
            model.load_state_dict(ck['state_dict']);model.eval();saved=torch.load(source/'predictions.pt',map_location='cpu',weights_only=True)
            for split in ['train','val']:
                x=features(kind,split);truth=labels(split);raw=torch.cat([model(batch) for batch in x.split(128)])
                torch.testing.assert_close(raw,saved[split],atol=0,rtol=0);pred=discrete_states(raw).numpy();records=list(csv.DictReader((dest/f'{split}_rows.csv').open()))
                for i,(p,t) in enumerate(zip(pred,truth.numpy())):
                    present=np.flatnonzero(p[:,15]>.5);gt=np.flatnonzero(t[:,15]>.5);checks=np.zeros((len(gt),5),bool);position=[]
                    if len(present):
                        cost=np.linalg.norm(t[gt,None,10:12]-p[None,present,10:12],axis=-1)**2;gi,pi=linear_sum_assignment(cost)
                        for a,b in zip(gi,pi):
                            v=p[present[b]];g=t[gt[a]]
                            pose_v=int(np.rint(-np.arctan2(v[14],v[13])/(np.pi/2)))%4
                            pose_g=int(np.rint(-np.arctan2(g[14],g[13])/(np.pi/2)))%4
                            checks[a]=[v[:4].argmax()==g[:4].argmax(),v[4:10].argmax()==g[4:10].argmax(),
                                np.max(np.abs(v[10:12]-g[10:12]))*31.5<=1,abs(v[12]-g[12])*8<=.5,pose_v==pose_g]
                            position.append(float(np.abs(v[10:12]-g[10:12]).mean()*31.5))
                    row=records[i]
                    for j,key in enumerate(['shape_correct','color_correct','position_within_one_pixel','radius_within_half_pixel','pose_correct']):assert abs(checks[:,j].mean()-float(row[key]))<1e-12
                    assert (row['full_scene_success']=='True')==bool(len(present)==len(gt) and checks.all())
                    assert int(row['predicted_objects'])==len(present);rows_checked+=1
                for key,stat in metrics['splits'][split].items():
                    if stat is None:continue
                    selected=[v[key] for v in records if v[key] not in ['',None]]
                    values=[float(v=='True') if v in ['True','False'] else float(v) for v in selected]
                    expected=r.bootstrap(values);assert abs(expected['mean']-stat['mean'])<1e-10
                    np.testing.assert_allclose(expected['base_scene_ci95'],stat['base_scene_ci95'],atol=1e-10,rtol=0)
                if split=='train':
                    # Check the assignment objective against all 20 full losses.
                    small=raw[:8];tt=truth[:8,:2];attrs=attribute_cost(small,tt);costs=[]
                    for slots in itertools.permutations(range(5),2):
                        mask=torch.zeros(8,5);mask[:,list(slots)]=1
                        losses=(attrs[:,0,slots[0]]+attrs[:,1,slots[1]])/2+F.binary_cross_entropy_with_logits(small[...,15],mask,reduction='none').mean(1)
                        costs.append(losses)
                    expected=torch.stack(costs).min(0).values.mean();actual,_=set_loss(small,truth[:8]);torch.testing.assert_close(expected,actual,atol=1e-6,rtol=1e-6)
            count+=1;print('verified readout',kind,seed,flush=True)
    dump(root/'verification.json',{'all_passed':True,'models_verified':count,'prediction_rows_replayed_and_scored':rows_checked,
        'original_model_training_unchanged':True,'train_only_normalization_and_sampling_verified':True,
        'set_matching_checked_against_all_twenty_assignments':True,'pose_scoring_independently_checked_via_atan2':True,
        'verifier_sha256':r.sha(__file__),'claim_boundary':c['claim_boundary']})


if __name__=='__main__':main()
