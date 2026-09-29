"""Artifact replay and base-scene aggregation checks for the true-state experiment."""
import csv
import json
import torch.nn.functional as F
from common import HERE, dump, digest_tensor, np, torch, r, update_status
from object_tasks import load
from object_oracle_study import Predictor, predict, factor_checks


@torch.no_grad()
def main():
    torch.set_num_threads(2);root=HERE/'data/object_edit_tasks_v1';dm=json.loads((root/'manifest.json').read_text())
    for f in dm['files']:assert r.sha(root/f['path'])==f['sha256']
    c=json.loads((HERE/'configs/object_oracle_study_v1.json').read_text());assert c['data_manifest_sha256']==r.sha(root/'manifest.json')
    task_count=0
    for split in dm['splits']:
        data,records=load(split);answer=predict('analytic',data)
        torch.testing.assert_close(answer,data['target_states'],atol=2e-6,rtol=2e-6)
        assert factor_checks(answer,data['target_states']).all();task_count+=len(records)
        if split=='train':
            assert {v['operation'] for v in records}=={'color','rotate','translate'}
            assert not any(v['target_pair_is_unseen'] or v['source_target_pair_is_unseen'] for v in records)
    checked=0;aggregate_count=0;replay_count=0;permutation_errors={};matched={}
    for name in ['identity','analytic']+[f'{k}_s{s}' for s in c['seeds'] for k in c['kinds']]:
        folder=HERE/f'runs/object_oracle_study_v1/{name}';m=json.loads((folder/'metrics.json').read_text())
        assert m['prediction_sha256']==r.sha(folder/'predictions.pt')
        saved=torch.load(folder/'predictions.pt',map_location='cpu',weights_only=True)
        if name in ['identity','analytic']:model=name
        else:
            run=json.loads((folder/'run.json').read_text());ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            assert run['checkpoint_sha256']==m['checkpoint_sha256']==r.sha(folder/'model.pt')
            assert run['steps']==6000 and run['parameters']==34991
            assert run['code_sha256']==r.sha(HERE/'object_oracle_study.py')
            assert min(run['target_slot_exposure'])>100000 and min(run['occupied_slot_exposure'])>200000
            model=Predictor(ck['kind']);model.load_state_dict(ck['state_dict']);model.eval();matched[name]=run;checked+=1
        for split in saved:
            data,records=load(split);pred=saved[split]
            assert pred.shape==data['target_states'].shape and torch.isfinite(pred).all()
            # Replay from checkpoint; all 3,284 evaluation tasks, not just plotted examples.
            torch.testing.assert_close(predict(model,data),pred,atol=2e-6,rtol=2e-6);replay_count+=len(pred)
            rows=list(csv.DictReader((folder/f'{split}_rows.csv').open()));assert len(rows)==len(records)
            checks=factor_checks(pred,data['target_states']);presence=data['target_states'][:,:,15]>.5
            expected=((checks.all(-1)|~presence).all(-1)&((pred[:,:,15]>=.5)==presence).all(-1)).numpy()
            assert all((v['full_scene_success']=='True')==bool(correct) for v,correct in zip(rows,expected))
            groups={'all':rows}
            for op in {v['operation'] for v in rows}:groups[op]=[v for v in rows if v['operation']==op]
            for group,key in [('unseen_source_target','source_target_pair_is_unseen'),('unseen_output_target','target_pair_is_unseen')]:
                rr=[v for v in rows if v[key]=='True']
                if rr:groups[group]=rr
            for group,rr in groups.items():
                ids={int(v['source_index']) for v in rr}
                for key,metric in m['splits'][split][group].items():
                    if not isinstance(metric,dict):continue
                    def number(v):return float(v=='True') if v in ['True','False'] else float(v)
                    values=[np.mean([number(v[key]) for v in rr if int(v['source_index'])==j]) for j in ids]
                    assert abs(float(np.mean(values))-metric['mean'])<1e-10;aggregate_count+=1
            if name=='analytic':assert expected.all()
            if split=='count4' and name.startswith('object'):
                x=data['source_states'][:64];mask=F.one_hot(data['target_ids'][:64],4).float();cmd=data['commands'][:64,0]
                perm=torch.tensor([2,0,3,1]);a=model(x[:,perm],mask[:,perm],cmd);b=model(x,mask,cmd)[:,perm]
                error=float((a-b).abs().max());assert error<1e-6;permutation_errors[name]=error
    for seed in c['seeds']:
        pair=[matched[f'{k}_s{seed}'] for k in c['kinds']]
        for key in ['parameters','steps','sampling_rng_final_sha256','source_state_sha256','target_state_sha256','target_slot_exposure','occupied_slot_exposure']:
            assert pair[0][key]==pair[1][key]
    report={'all_passed':True,'trained_checkpoints':checked,'task_truth_replays':task_count,'prediction_replays':replay_count,
        'base_scene_aggregate_checks':aggregate_count,'trained_shared_model_permutation_errors':permutation_errors,
        'parameter_and_sample_matching':True,'all_training_slots_exposed':True,'verifier_sha256':r.sha(__file__),
        'claim_boundary':'GT-state diagnostic; oracle geometry and identity supplied. No claim of image-based perception or robust generalization outside the tested factors.'}
    dump(HERE/'reports/object_oracle_study_v1/verification.json',report)
    update_status('B03','complete',['reports/object_oracle_study_v1/summary.json','reports/object_oracle_study_v1/verification.json'])
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
