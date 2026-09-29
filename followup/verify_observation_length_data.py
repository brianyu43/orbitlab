"""Replay all paired t=7 worlds, images, counterfactuals and partition contracts."""
import json
import time
import numpy as np
from common import HERE,dump,r
import dynamics_world as d
from dynamics_observation_length_data import NAME,DATA,freeze


def main():
    c=freeze();physics=d.Physics(**c['generator']['physics']);path=HERE/f'data/{NAME}/manifest.json';manifest=json.loads(path.read_text())
    assert manifest['config_sha256']==r.sha(HERE/f'configs/{NAME}.json') and manifest['builder_sha256']==r.sha(HERE/'dynamics_observation_length_data.py')
    expected={(g['data_seed'],g['mode'],g['split']) for g in c['groups']}
    assert len(manifest['groups'])==len(expected)==57 and {(g['data_seed'],g['mode'],g['split']) for g in manifest['groups']}==expected
    physical=set();images=set();anchor_images=set();episodes=frames=rgb=counterfactuals=rotations=permutations=0
    maximum_penetration=maximum_rotation=maximum_energy=0.;start=time.time()
    for group_index,g in enumerate(manifest['groups']):
        root=(HERE/g['path']).parent
        for name,sha in g['files'].items():assert r.sha(root/name)==sha
        records=json.loads((root/'records.json').read_text());source_records=json.loads((HERE/g['source_records']).read_text())
        with np.load(HERE/g['source_path']) as a:source={k:a[k].copy() for k in ['states','contexts','actions','target_ids','observations']}
        with np.load(HERE/g['path']) as a:stored={k:a[k].copy() for k in a.files}
        n=g['n_objects'];length=24 if g['split']=='train' else 69;number=g['episodes']
        assert stored['states'].shape==(number,length,4,7) and stored['observations'].shape==(number,8,64,64,3)
        assert np.all(stored['n_objects']==n) and not stored['states'][:,:,n:].any() and not stored['actions'][:,:,n:].any()
        assert not stored['actions'][:,:7].any() and not stored['actions'][:,8:].any()
        np.testing.assert_array_equal(stored['actions'][:,7],source['actions'][:,3]);np.testing.assert_array_equal(stored['contexts'],source['contexts'])
        np.testing.assert_array_equal(stored['target_ids'],source['target_ids'])
        # Original frames 0..3 are also pre-action; later frames must be re-simulated.
        np.testing.assert_array_equal(stored['states'][:,:4],source['states'][:,:4]);np.testing.assert_array_equal(stored['observations'][:,:4],source['observations'])
        for i,record in enumerate(records):
            initial=stored['states'][i,0,:n];context=stored['contexts'][i];actions=stored['actions'][i,:,:n]
            sampled,ctx,original_actions,target=d.sample(i,g['data_seed'],g['split'],g['mode'],n)
            np.testing.assert_array_equal(initial,sampled);np.testing.assert_array_equal(context,ctx)
            np.testing.assert_array_equal(actions[7],original_actions[3]);assert target==stored['target_ids'][i]==record['target_id']
            actual,events=d.rollout(initial,context,actions,physics=physics)
            np.testing.assert_array_equal(actual,stored['states'][i,:,:n]);np.testing.assert_array_equal(events,stored['events'][i])
            np.testing.assert_array_equal(actual[:,:,4:],np.broadcast_to(initial[:,4:],actual[:,:,4:].shape))
            assert len(set(initial[:,5]))==n
            for t in range(8):
                image=d.render(actual[t]);np.testing.assert_array_equal(image['image'],stored['observations'][i,t])
                np.testing.assert_array_equal(image['segmentation'],stored['observation_segmentation'][i,t]);rgb+=1
            ph=d.initial_orbit_hash(initial,context,actions[7]);im=r.orbit_hash(stored['observations'][i,0]);anchor=r.orbit_hash(stored['observations'][i,7])
            assert ph==record['t7_initial_orbit_sha256']==record['source_initial_orbit_sha256']==source_records[i]['initial_orbit_sha256']
            assert im==record['source_initial_image_orbit_sha256']==source_records[i]['initial_image_orbit_sha256']
            assert anchor==record['anchor_image_orbit_sha256']
            assert ph not in physical and im not in images and anchor not in anchor_images
            physical.add(ph);images.add(im);anchor_images.add(anchor)
            penetration=max(d.violation(s) for s in actual);assert penetration<=physics.overlap_tolerance*1.1;maximum_penetration=max(maximum_penetration,penetration)
            if g['mode']=='isotropic':
                before=.5*np.square(actual[:-1,:,2:4]+actions).sum((1,2));after=.5*np.square(actual[1:,:,2:4]).sum((1,2))
                error=float(np.abs(after-before).max());assert error<1e-10;maximum_energy=max(maximum_energy,error)
            if g['split'] in c['generator']['counterfactual_splits']:
                cf,ce=d.rollout(initial,context,-actions,physics=physics)
                np.testing.assert_array_equal(cf,stored['counterfactual_states'][i,:,:n]);np.testing.assert_array_equal(ce,stored['counterfactual_events'][i])
                np.testing.assert_array_equal(cf[:8],actual[:8]);counterfactuals+=1
            if i<16:
                for k in [1,2,3]:
                    rs,re=d.rollout(d.rotate_state(initial,k),d.rotate_context(context,k),d.rotate_vectors(actions,k),physics=physics)
                    expected_states=np.stack([d.rotate_state(s,k) for s in actual]);err=float(np.abs(rs-expected_states).max())
                    np.testing.assert_allclose(rs,expected_states,atol=2e-8,rtol=0);np.testing.assert_array_equal(re,events)
                    for t in [0,7,length-1]:np.testing.assert_array_equal(d.render(rs[t])['image'],np.rot90(d.render(actual[t])['image'],k))
                    maximum_rotation=max(maximum_rotation,err);rotations+=1
                order=np.arange(n)[::-1];ps,pe=d.rollout(initial[order],context,actions[:,order],ids=order,physics=physics)
                np.testing.assert_array_equal(ps,actual[:,order]);np.testing.assert_array_equal(pe,events);permutations+=1
            episodes+=1;frames+=length
        assert g['pair_impulses']==int(stored['events'][:,:,0].sum()) and g['wall_impulses']==int(stored['events'][:,:,1].sum())
        print('verified t7 world',g['data_seed'],g['mode'],g['split'],group_index+1,flush=True)
        dump(HERE/f'reports/{NAME}/data_verification_progress.json',{'groups_verified':group_index+1,'expected_groups':57,'seconds':time.time()-start})
    assert episodes==manifest['episodes']==len(physical)==len(images)==len(anchor_images)==6720
    assert counterfactuals==manifest['counterfactual_episodes']==1536 and frames==360000 and rgb==53760
    dump(HERE/f'reports/{NAME}/data_verification.json',{'all_passed':True,'episodes_replayed':episodes,'states_replayed':frames,
        'observations_rerendered':rgb,'counterfactuals_replayed':counterfactuals,'rotated_trajectories_checked':rotations,
        'permuted_trajectories_checked':permutations,'unique_initial_physical_orbits':len(physical),'unique_initial_image_orbits':len(images),
        'unique_anchor_image_orbits':len(anchor_images),'max_contact_penetration_pixels':maximum_penetration,
        'max_joint_rotation_state_error':maximum_rotation,'max_isotropic_energy_error':maximum_energy,
        'all_initial_draws_paired_with_original_data':True,'original_frames_0_to_3_unchanged':True,'action_at_transition_7_only':True,
        'manifest_sha256':r.sha(path),'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),'verifier_sha256':r.sha(__file__),
        'seconds':time.time()-start,'scope':'Replay of paired synthetic t=7 worlds; same independent data units and original partitions. Length-specific input noninterference and model evaluations remain separate.'})


if __name__=='__main__':main()
