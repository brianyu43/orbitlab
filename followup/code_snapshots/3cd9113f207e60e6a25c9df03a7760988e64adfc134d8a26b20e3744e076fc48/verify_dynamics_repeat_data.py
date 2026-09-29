"""Every new trajectory/RGB replay plus cross-dataset orbit and C4 checks."""
import json
import time
import numpy as np
from common import HERE,dump,r
import dynamics_world as d
from dynamics_repeat_data import NAME,freeze,original_orbits


def main():
    c=freeze();spec=c['generator'];physics=d.Physics(**spec['physics']);started=time.time()
    master=HERE/f'data/{NAME}/manifest.json';manifest=json.loads(master.read_text())
    assert manifest['config_sha256']==r.sha(HERE/f'configs/{NAME}.json')
    assert manifest['builder_sha256']==r.sha(HERE/'dynamics_repeat_data.py')
    orbits,images=original_orbits();total=frames=observations=cf_count=rotations=permutations=other_response=0
    maximum_rotation=maximum_penetration=maximum_energy=0.;by_seed=[];completed_groups=0
    for seed in c['new_data_seeds']:
        seed_groups=[g for g in manifest['groups'] if g['data_seed']==seed];seed_count=seed_cf=0
        expected={(m,s) for m in spec['modes'] for s in spec['sizes'] if s!='force_ood' or m in spec['force_ood_modes']}
        assert {(g['mode'],g['split']) for g in seed_groups}==expected and len(seed_groups)==19
        for g in seed_groups:
            path=HERE/g['path'];assert r.sha(path)==g['archive_sha256']
            assert r.sha(path.parent/'records.json')==g['records_sha256']
            records=json.loads((path.parent/'records.json').read_text())
            with np.load(path) as source:a={k:source[k].copy() for k in source.files}
            mode,split=g['mode'],g['split'];n=g['n_objects'];length=20 if split=='train' else 65
            assert a['states'].shape==(spec['sizes'][split],length,4,7)
            assert a['observations'].shape==(len(records),4,64,64,3)
            assert np.all(a['n_objects']==n) and not a['states'][:,:,n:].any() and not a['actions'][:,:,n:].any()
            assert not a['actions'][:,:3].any() and not a['actions'][:,4:].any()
            assert set(a)==({'states','contexts','actions','events','observations','observation_segmentation','n_objects','target_ids'}
                            | ({'counterfactual_states','counterfactual_events'} if split in spec['counterfactual_splits'] else set()))
            for i,record in enumerate(records):
                s=a['states'][i,:,:n];ctx=a['contexts'][i];act=a['actions'][i,:,:n]
                initial,context,action,target=d.sample(i,seed,split,mode,n)
                np.testing.assert_array_equal(initial,s[0]);np.testing.assert_array_equal(ctx,context)
                np.testing.assert_array_equal(act,action[:length-1]);assert target==record['target_id']==a['target_ids'][i]
                assert record['seed_components']==[seed,split,mode,i]
                np.testing.assert_array_equal(s[:,:,4:],np.broadcast_to(s[0,:,4:],s[:,:,4:].shape))
                assert len(set(s[0,:,5]))==n
                np.testing.assert_array_equal(s[0,:,4]-4==s[0,:,5],[split=='attribute_ood']+[False]*(n-1))
                if mode=='isotropic':assert not ctx.any()
                elif mode=='fixed_gravity':np.testing.assert_array_equal(ctx,[0,.05,0,0,0])
                else:assert (d.net_acceleration(ctx)[1]<0)==(split=='force_ood')
                orbit=d.initial_orbit_hash(initial,ctx,act[3]);image_orbit=r.orbit_hash(a['observations'][i,0])
                assert orbit==record['initial_orbit_sha256'] and orbit not in orbits
                assert image_orbit==record['initial_image_orbit_sha256'] and image_orbit not in images
                orbits.add(orbit);images.add(image_orbit)
                reproduced,event=d.rollout(initial,ctx,act,physics=physics)
                np.testing.assert_array_equal(s,reproduced);np.testing.assert_array_equal(event,a['events'][i])
                for t in range(4):
                    image=d.render(s[t]);np.testing.assert_array_equal(image['image'],a['observations'][i,t])
                    np.testing.assert_array_equal(image['segmentation'],a['observation_segmentation'][i,t]);observations+=1
                penetration=max(d.violation(x) for x in s);maximum_penetration=max(maximum_penetration,penetration)
                assert penetration<=physics.overlap_tolerance*1.1
                if mode=='isotropic':
                    before=.5*((s[:-1,:,2:4]+act)**2).sum((1,2))
                    after=.5*(s[1:,:,2:4]**2).sum((1,2));error=float(np.max(np.abs(after-before)))
                    assert error<1e-10;maximum_energy=max(maximum_energy,error)
                if split in spec['counterfactual_splits']:
                    cf,ce=d.rollout(initial,ctx,-act,physics=physics)
                    np.testing.assert_array_equal(cf,a['counterfactual_states'][i,:,:n]);np.testing.assert_array_equal(ce,a['counterfactual_events'][i])
                    np.testing.assert_array_equal(cf[:4],s[:4]);cf_count+=1;seed_cf+=1
                    others=[j for j in range(n) if j!=target];other_response+=int(np.max(np.abs(cf[:,others,:4]-s[:,others,:4]))>1e-6)
                if i<16:
                    for k in (1,2,3):
                        rs,re=d.rollout(d.rotate_state(initial,k),d.rotate_context(ctx,k),d.rotate_vectors(act,k),physics=physics)
                        target_states=np.stack([d.rotate_state(x,k) for x in s]);error=float(np.max(np.abs(rs-target_states)))
                        np.testing.assert_allclose(rs,target_states,atol=2e-8,rtol=0);np.testing.assert_array_equal(re,event)
                        for t in [0,3,length-1]:np.testing.assert_array_equal(d.render(rs[t])['image'],np.rot90(d.render(s[t])['image'],k))
                        maximum_rotation=max(maximum_rotation,error);rotations+=1
                    reverse=np.arange(n)[::-1]
                    ps,pe=d.rollout(initial[reverse],ctx,act[:,reverse],ids=reverse,physics=physics)
                    np.testing.assert_array_equal(ps,s[:,reverse]);np.testing.assert_array_equal(pe,event);permutations+=1
                total+=1;seed_count+=1;frames+=length
            assert g['pair_impulses']==int(a['events'][:,:,0].sum()) and g['wall_impulses']==int(a['events'][:,:,1].sum())
            completed_groups+=1
            dump(HERE/f'reports/{NAME}/data_verification_progress.json',{'completed_groups':completed_groups,'expected_groups':38,'last':[seed,mode,split],'seconds':time.time()-started})
            print('verified repeat data',seed,mode,split,len(records),flush=True)
        seed_manifest=HERE/f'data/{NAME}/d{seed}/world/manifest.json';sm=json.loads(seed_manifest.read_text())
        assert sm['groups']==seed_groups and sm['episodes']==seed_count==2240 and sm['counterfactual_episodes']==seed_cf==512
        by_seed.append({'data_seed':seed,'episodes_replayed':seed_count,'counterfactuals_replayed':seed_cf,'manifest_sha256':r.sha(seed_manifest)})
    assert len(orbits)==len(images)==6720 and total==manifest['new_episodes']==4480
    assert cf_count==manifest['new_counterfactual_episodes']==1024 and other_response>0
    dump(HERE/f'reports/{NAME}/data_verification.json',{'all_passed':True,'datasets':by_seed,'episodes_replayed':total,
        'states_replayed':frames,'observations_rerendered':observations,'counterfactuals_replayed':cf_count,
        'counterfactuals_with_nontarget_physical_response':other_response,'rotated_trajectories_checked':rotations,
        'permuted_trajectories_checked':permutations,'max_joint_rotation_state_error':maximum_rotation,
        'max_isotropic_per_step_energy_error':maximum_energy,'max_contact_penetration_pixels':maximum_penetration,
        'unique_physical_and_image_orbits_including_original':len(orbits),'original_and_new_splits_disjoint':True,
        'manifest_sha256':r.sha(master),'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),'verifier_sha256':r.sha(__file__),
        'seconds':time.time()-started,'scope':'Finite discrete synthetic-world replay and split checks; no model performance or real-world physics claim.'})


if __name__=='__main__':main()
