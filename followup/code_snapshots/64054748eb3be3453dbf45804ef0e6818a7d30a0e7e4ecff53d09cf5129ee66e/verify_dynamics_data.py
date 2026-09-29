"""Replay every saved trajectory, with independent invariant checks and C4 audit."""
import json
import dataclasses
import numpy as np
from common import HERE, dump, r
import dynamics_world as d


def main():
    root=HERE/'data/dynamics_world_v1';manifest=json.loads((root/'manifest.json').read_text())
    cp=HERE/'configs/dynamics_world_v1.json';c=json.loads(cp.read_text());physics=d.Physics(**c['physics'])
    assert manifest['config_sha256']==r.sha(cp)
    assert manifest['simulator_sha256']==c['code_sha256']==r.sha(HERE/'dynamics_world.py')
    assert manifest['builder_sha256']==c['builder_sha256']==r.sha(HERE/'build_dynamics_data.py')
    groups=[];orbits=set();image_orbits=set();count=frames=images=counterfactuals=rotations=0
    max_rotation_error=max_energy_error=max_violation=0.;cf_other_changed=0;fixed_errors=[]
    for group in manifest['groups']:
        path=HERE/group['path'];assert r.sha(path)==group['archive_sha256']
        assert r.sha(path.parent/'records.json')==group['records_sha256']
        records=json.loads((path.parent/'records.json').read_text());split=group['split'];mode=group['mode']
        with np.load(path,allow_pickle=False) as archive:a={k:archive[k] for k in archive.files}
        n=group['n_objects'];length=c['observation_frames']+(c['train_future_frames'] if split=='train' else c['eval_future_frames'])
        assert a['states'].shape==(group['episodes'],length,4,7)
        assert a['observations'].shape==(group['episodes'],4,64,64,3)
        assert not np.any(a['states'][:,:,n:]) and not np.any(a['actions'][:,:,n:])
        assert np.all(a['n_objects']==n)
        assert len(records)==group['episodes']
        for i,record in enumerate(records):
            state=a['states'][i,:,:n];context=a['contexts'][i];action=a['actions'][i,:,:n]
            expected,cc,aa,target=d.sample(i,c['seed'],split,mode,n)
            np.testing.assert_array_equal(expected,state[0]);np.testing.assert_array_equal(cc,context)
            np.testing.assert_array_equal(aa[:length-1],action);assert target==record['target_id']==a['target_ids'][i]
            # All constant identity factors and exact held-out binding semantics.
            np.testing.assert_array_equal(state[:,:,4:],np.broadcast_to(state[0,:,4:],state[:,:,4:].shape))
            bound=(state[0,:,4]-4)==state[0,:,5]
            assert np.array_equal(bound,np.array([split=='attribute_ood']+[False]*(n-1)))
            assert len(set(state[0,:,5]))==n
            if mode=='isotropic':assert not np.any(context)
            if mode=='fixed_gravity':np.testing.assert_array_equal(context,[0,.05,0,0,0])
            if mode=='variable_force':assert (d.net_acceleration(context)[1]<0)==(split=='force_ood')
            orbit=d.initial_orbit_hash(state[0],context,action[3]);assert orbit==record['initial_orbit_sha256'] and orbit not in orbits;orbits.add(orbit)
            image_orbit=r.orbit_hash(a['observations'][i,0]);assert image_orbit==record['initial_image_orbit_sha256'] and image_orbit not in image_orbits;image_orbits.add(image_orbit)
            reproduced,event=d.rollout(state[0],context,action,physics=physics)
            np.testing.assert_array_equal(reproduced,state);np.testing.assert_array_equal(event,a['events'][i])
            for t in range(4):
                visual=d.render(state[t]);np.testing.assert_array_equal(visual['image'],a['observations'][i,t])
                np.testing.assert_array_equal(visual['segmentation'],a['observation_segmentation'][i,t]);images+=1
            violation=max(d.violation(s) for s in state);max_violation=max(max_violation,violation)
            assert violation<=physics.overlap_tolerance*1.1
            if mode=='isotropic':
                before=.5*np.sum((state[:-1,:,2:4]+action)**2,axis=(1,2));after=.5*np.sum(state[1:,:,2:4]**2,axis=(1,2))
                error=float(np.max(np.abs(before-after)));max_energy_error=max(max_energy_error,error);assert error<1e-10
            if split in c['counterfactual_splits']:
                cf,ce=d.rollout(state[0],context,-action,physics=physics)
                np.testing.assert_array_equal(cf,a['counterfactual_states'][i,:,:n]);np.testing.assert_array_equal(ce,a['counterfactual_events'][i])
                np.testing.assert_array_equal(cf[:4],state[:4]);counterfactuals+=1
                others=[j for j in range(n) if j!=target]
                cf_other_changed+=int(np.max(np.abs(cf[:,others,:4]-state[:,others,:4]))>1e-6)
            # Stratified first 16 independent episodes per split/mode: full C4,
            # including future images beyond the four saved observations.
            if i<16:
                for k in (1,2,3):
                    rs,re=d.rollout(d.rotate_state(state[0],k),d.rotate_context(context,k),d.rotate_vectors(action,k),physics=physics)
                    target_state=np.stack([d.rotate_state(s,k) for s in state])
                    error=float(np.max(np.abs(rs-target_state)));max_rotation_error=max(max_rotation_error,error)
                    np.testing.assert_allclose(rs,target_state,atol=2e-8,rtol=0);np.testing.assert_array_equal(re,event)
                    for t in [0,3,length-1]:
                        np.testing.assert_array_equal(d.render(rs[t])['image'],np.rot90(d.render(state[t])['image'],k))
                    rotations+=1
                order=np.arange(n)[::-1];ps,pe=d.rollout(state[0,order],context,action[:,order],ids=order,physics=physics)
                np.testing.assert_array_equal(ps,state[:,order]);np.testing.assert_array_equal(pe,event)
                if mode=='fixed_gravity':
                    wrong,_=d.step(d.rotate_state(state[0],1),context,d.rotate_vectors(action[0],1),physics=physics)
                    fixed_errors.append(float(np.max(np.abs(wrong[:,:4]-d.rotate_state(state[1],1)[:,:4]))))
            count+=1;frames+=length
        assert int(a['events'][:,:,0].sum())==group['pair_impulses']
        assert int(a['events'][:,:,1].sum())==group['wall_impulses']
        groups.append({'mode':mode,'split':split,'episodes_verified':len(records)})
        print(f'verified {mode}/{split}: {len(records)}',flush=True)
    assert count==manifest['episodes']==len(orbits)==len(image_orbits)
    assert counterfactuals==manifest['counterfactual_episodes']
    assert min(fixed_errors)>.01 and cf_other_changed>0
    report={'all_passed':True,'episodes_replayed':count,'states_replayed':frames,'observations_rerendered':images,
        'counterfactuals_replayed':counterfactuals,'counterfactuals_with_non_target_physical_response':cf_other_changed,
        'rotated_trajectories_checked':rotations,'max_joint_rotation_state_error':max_rotation_error,
        'max_isotropic_per_step_energy_error':max_energy_error,'max_contact_penetration_pixels':max_violation,
        'fixed_condition_counterexamples':len(fixed_errors),'fixed_condition_state_error_range':[min(fixed_errors),max(fixed_errors)],
        'unique_scene_action_context_orbits':len(orbits),'unique_initial_image_orbits':len(image_orbits),'groups':groups,
        'config_sha256':r.sha(cp),'manifest_sha256':r.sha(root/'manifest.json'),'verifier_sha256':r.sha(__file__),
        'claim_boundary':'Finite discrete simulator checks. Continuous collision timing and real-world physics are not validated. No learned dynamics model has been evaluated.'}
    dump(HERE/'reports/dynamics_world_v1/verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
