"""Replay every saved trajectory and score, including counterfactuals and failures."""
import json
import numpy as np
import torch
import r3_core as c
import r3_physics as physics
from r3_evaluate import rollout,rows_for,summarize
from dynamics_rgb_baseline import measure_frame,infer
import dynamics_world as w
from v3_common import ROOT,dump,sha,status

def equal(a,b):
    assert a.shape==b.shape
    assert np.allclose(a,b,rtol=0,atol=1e-7,equal_nan=True),float(np.nanmax(np.abs(a-b)))

def main():
    c.freeze();torch.set_num_threads(2);new_keys=set();data=[]
    for manifest in sorted((c.BASE/'data').glob('d*/*/n*/manifest.json')):
        rec=json.loads(manifest.read_text());root=manifest.parent
        for name,digest in rec['files'].items():assert sha(root/name)==digest
        a=np.load(root/'trajectories.npz');states,events=physics.rollout(a['states'][:,0],a['contexts'],a['actions'])
        assert np.array_equal(states,a['states']);assert np.array_equal(events,a['events'])
        keys={w.initial_orbit_hash(s,ctx,act[3]) for s,ctx,act in zip(states[:,0],a['contexts'],a['actions'])}
        assert len(keys)==len(states);assert not keys&new_keys;new_keys.update(keys)
        metadata=json.loads((root/'metadata.json').read_text());assert keys=={v['initial_orbit_sha256'] for v in metadata}
        pair=events[:,3:19,0].sum(1)>0;wall=events[:,3:19,1].sum(1)>0
        cat=np.where(pair,2,np.where(wall,1,0));assert np.array_equal(cat,a['stratum'])
        if rec['split']!='train':
            reverse,rev_events=physics.rollout(states[:,3],a['contexts'],-a['actions'][:,3:])
            force,force_events=physics.rollout(states[:,3],a['force_contexts'],a['actions'][:,3:])
            assert np.array_equal(reverse,a['reversed_states']);assert np.array_equal(rev_events,a['reversed_events'])
            assert np.array_equal(force,a['force_states']);assert np.array_equal(force_events,a['force_events'])
            assert np.allclose(a['force_contexts']-a['contexts'],np.array([0,0,.04,0,0]))
            images=np.load(root/'observations.npz')['images'];estimated=np.load(root/'rgb_context.npz')['contexts']
            for i,sequence in enumerate(images):
                for t,image in enumerate(sequence):assert np.array_equal(w.render(states[i,t])['image'],image)
                f=np.stack([measure_frame(image) for image in sequence]);_,ctx,_=infer(f)
                equal(np.array([*ctx[:2],0,0,ctx[2]]),estimated[i])
        data.append({'path':str(root.relative_to(c.BASE)),'episodes':len(states),'full_reference_rollout_replayed':True,'full_counterfactuals_and_rgb_replayed':rec['split']!='train','manifest_sha256':sha(manifest)})
        print('R3 data verified',data[-1]['path'],flush=True)
    historical_keys=set();historical_files=[]
    roots=[ROOT/'followup/data/dynamics_world_v1',ROOT/'completion_v2/dynamics/data']
    for root in roots:
        for path in root.rglob('trajectories.npz'):
            a=np.load(path)
            for s,ctx,act in zip(a['states'][:,0],a['contexts'],a['actions']):
                alive=s[:,4]>0;historical_keys.add(w.initial_orbit_hash(s[alive],ctx,act[3,alive]))
            historical_files.append(str(path.relative_to(ROOT)))
    assert not historical_keys&new_keys
    units=[];summaries=[];contact_strata=[]
    for path in sorted((c.BASE/'evaluation').glob('d*/*/n*/*/summary.json')):
        rec=json.loads(path.read_text());out=path.parent;arm=rec['arm'];seed=rec['seed'];init=rec['init']
        split='val' if seed==c.DEVELOPMENT else 'test';root=c.BASE/f'data/d{seed}/{split}/n{rec["count"]}'
        assert sha(root/'trajectories.npz')==rec['data_sha256'];assert sha(out/'predictions.npz')==rec['predictions_sha256']
        saved=np.load(out/'predictions.npz');a=np.load(root/'trajectories.npz');model=None;limit=None
        if arm in c.ARMS:
            cp=c.BASE/f'runs/d{seed}_s{init}/{arm}/model.pt';assert sha(cp)==rec['model_sha256'];ck=torch.load(cp,weights_only=True)
            model=c.model_for(arm);model.load_state_dict(ck['state_dict']);model.eval();limit=ck['velocity_limit']
        initial=a['states'][:,3];context=a['contexts'] if rec['input']=='known_force' else np.load(root/'rgb_context.npz')['contexts'];actions=a['actions'][:,3:]
        assert np.array_equal(initial,saved['initial']);assert np.array_equal(context,saved['context']);assert np.array_equal(actions,saved['actions'])
        pred=rollout(initial,context,actions,model,arm,limit);rev=rollout(initial,context,-actions,model,arm,limit)
        ctx=context.copy();ctx[:,2]+=.04;force=rollout(initial,ctx,actions,model,arm,limit)
        equal(pred,saved['prediction']);equal(rev,saved['reversed']);equal(force,saved['force'])
        rows=rows_for(pred,rev,force,a['states'][:,4:,:,:4],a['reversed_states'][:,1:,:,:4],a['force_states'][:,1:,:,:4],initial[:,:,4],a['target_ids'],a['stratum'])
        assert rows==json.loads((out/'rows.json').read_text());assert summarize(rows)==rec['metrics']
        # Actual contact strata can change beyond the initial16-step design label.
        for h in [1,16,61,128]:
            pair=a['events'][:,3:3+h,0].sum(1)>0;wall=a['events'][:,3:3+h,1].sum(1)>0
            for label,mask in [('no_contact',~(pair|wall)),('wall_only',wall&~pair),('pair',pair)]:
                subset=[r for r in rows if r['horizon']==h and mask[r['episode']]]
                contact_strata.append({'seed':seed,'init':init,'arm':arm,'count':rec['count'],'input':rec['input'],'horizon':h,'actual_contact_stratum':label,'episodes':len(subset),
                    'position_mae_all':float(np.mean([r['position_mae_all'] for r in subset])) if subset else None,'failure_rate':float(np.mean([r['failure'] for r in subset])) if subset else None})
        units.append({'path':str(out.relative_to(c.BASE)),'episodes':len(initial),'full_128_step_rollouts_replayed':3*len(initial),'rows_recomputed':len(rows),'summary_sha256':sha(path)})
        summaries.append(rec);print('R3 prediction verified',units[-1]['path'],flush=True)
    choice=json.loads((c.BASE/'selection.json').read_text());expected=64+(192 if choice['development_gate_passed'] else 0)
    assert len(units)==expected,(len(units),expected)
    dump(c.BASE/'actual_contact_strata.json',contact_strata)
    dump(c.BASE/'aggregate.json',{'results':summaries,'selection':choice,'scope':'Development one training seed and initialization; confirmation conditional on the prospectively frozen development gate. Controlled strata with differing action/force magnitudes.'})
    dump(c.BASE/'verification.json',{'data':data,'units':units,'expected_units':expected,'verified_units':len(units),'new_initial_orbits':len(new_keys),'historical_unique_initial_orbits':len(historical_keys),'historical_files_checked':historical_files,'cross_historical_overlap':0,'protocol_sha256':sha(c.BASE/'protocol.json'),
        'scope':'All saved128-step model predictions and both counterfactuals, full reference data, RGB context inference, metrics and hashes replayed; not independent researcher retraining.'})
    status('R3','verified',development_gate_passed=choice['development_gate_passed'],verification='r3/verification.json',report='r3/RESULTS_KO.md')

if __name__=='__main__':main()
