"""Independent PNG, official fold, score-mask and exposure checks for the adapter."""
import glob
import hashlib
import json
import numpy as np
from PIL import Image
from common import HERE,dump,r
from svib_preview_data import NAME,SOURCE,ALPHAS,alpha_name,freeze


def signature(obj):return (obj['shape'],tuple(obj['color']),obj['size'])


def canonical_bytes(image):return min(np.ascontiguousarray(np.rot90(image,k)).tobytes() for k in range(4))


def main():
    freeze();path=HERE/f'data/{NAME}/manifest.json';manifest=json.loads(path.read_text())
    assert manifest['config_sha256']==r.sha(HERE/f'configs/{NAME}.json')
    addresses={'heldout'}|{f'{alpha_name(a)}/{s}' for a in ALPHAS for s in ['train','val','test_id']}
    assert len(manifest['groups'])==13 and {x['address'] for x in manifest['groups']}==addresses
    arrays={};bindings={};total=0;source_episodes=set();identity_counts={}
    for g in manifest['groups']:
        address=g['address'];folder=HERE/f'data/{NAME}/{address}'
        for file,sha in g['files'].items():assert r.sha(HERE/file)==sha
        with np.load(folder/'inputs.npz') as a:assert a.files==['source_rgb'];source=a['source_rgb'].copy()
        with np.load(folder/'labels.npz') as a:
            assert set(a.files)=={'target_rgb','changed_mask','foreground_mask'}
            target=a['target_rgb'].copy();changed=a['changed_mask'].copy();foreground=a['foreground_mask'].copy()
        if address=='heldout':origin=SOURCE/'Test';bounds=slice(None)
        else:
            name,split=address.split('/');alpha=name.removeprefix('alpha_').replace('p','.')
            origin=SOURCE/f'Train/alpha-{alpha}'
            # The official loader sorts glob(root/*), then takes 70%, 15%, 15%.
            bounds={'train':slice(None,70),'val':slice(70,85),'test_id':slice(85,None)}[split]
        original=sorted(glob.glob(str(origin/'*')));assert len(original)==100
        expected=original[bounds];records=json.loads((folder/'episodes.json').read_text())
        n=len(expected);assert len(records)==len(source)==len(target)==g['episodes']==n
        assert source.shape==target.shape==(n,128,128,3) and source.dtype==target.dtype==np.uint8
        assert changed.dtype==foreground.dtype==bool
        source_orbits=[];target_orbits=[];sb=set();tb=set()
        for i,(relative,row) in enumerate(zip(expected,records)):
            from pathlib import Path
            raw=Path(relative);assert raw.is_dir()
            assert row['index']==i and row['source_episode']==str(raw.relative_to(SOURCE))
            assert row['source_episode'] not in source_episodes;source_episodes.add(row['source_episode'])
            s=np.asarray(Image.open(raw/'source.png').convert('RGB').resize((128,128)))
            t=np.asarray(Image.open(raw/'target.png').convert('RGB').resize((128,128)))
            np.testing.assert_array_equal(source[i],s);np.testing.assert_array_equal(target[i],t)
            assert row['source_image_sha256']==r.sha(raw/'source.png') and row['target_image_sha256']==r.sha(raw/'target.png')
            sm=json.loads((raw/'source.json').read_text());tm=json.loads((raw/'target.json').read_text())
            assert row['source_objects']==sm['objects'] and row['target_objects']==tm['objects']
            assert len(sm['objects'])==len(tm['objects'])==2
            assert [x['shape'] for x in sm['objects']][::-1]==[x['shape'] for x in tm['objects']]
            for a,b in zip(sm['objects'],tm['objects']):
                for key in ['color','size','rotation','2d_coords']:assert a[key]==b[key]
            sb.update(signature(x) for x in sm['objects']);tb.update(signature(x) for x in tm['objects'])
            source_orbits.append(hashlib.sha256(canonical_bytes(s)).hexdigest())
            target_orbits.append(hashlib.sha256(canonical_bytes(t)).hexdigest())
            assert row['identical_pair']==bool(np.array_equal(s,t))
        np.testing.assert_array_equal(changed,np.max(abs(source.astype(int)-target.astype(int)),axis=-1)>0)
        np.testing.assert_array_equal(foreground,(source.max(axis=-1)>0)|(target.max(axis=-1)>0))
        assert g['identical_pairs']==int((~changed.any(axis=(1,2))).sum())
        identity_counts[address]=g['identical_pairs'];arrays[address]={'source':set(source_orbits),'target':set(target_orbits)}
        bindings[address]={'source':sb,'target':tb};total+=n
    expected_checks=[]
    for alpha in ALPHAS:
        for evaluation in [f'{alpha_name(alpha)}/val',f'{alpha_name(alpha)}/test_id','heldout']:
            for a in ['source','target']:
                for b in ['source','target']:
                    overlap=arrays[f'{alpha_name(alpha)}/train'][a]&arrays[evaluation][b]
                    assert not overlap
                    expected_checks.append({'alpha':alpha,'evaluation':evaluation,'train_side':a,'evaluation_side':b,'rotation_orbit_overlap':len(overlap)})
    assert expected_checks==manifest['image_orbit_overlap_checks']
    expected_exposure={address:{'source_bindings':len(x['source']),'target_bindings':len(x['target']),
        'source_overlap_external_input_bindings':len(x['source']&bindings['heldout']['source']),
        'target_overlap_external_input_bindings':len(x['target']&bindings['heldout']['source'])} for address,x in bindings.items()}
    assert manifest['binding_exposure']==expected_exposure
    assert total==manifest['episodes']==500 and len(source_episodes)==500 and identity_counts['heldout']==26
    for alpha in ALPHAS:assert expected_exposure[f'{alpha_name(alpha)}/train']['source_overlap_external_input_bindings']==0
    dump(HERE/f'reports/{NAME}/data_verification.json',{'all_passed':True,'pairs_verified':500,'groups_verified':13,
        'input_target_pngs_checked':1000,'official_sorted_70_15_15_fold_rule_verified':True,
        'original_saved_color_channels_and_pixels_preserved':True,'input_archives_contain_only_source_rgb':True,
        'target_derived_masks_only_in_labels':True,'independent_rotation_orbit_overlap_checks':len(expected_checks),
        'shared_heldout_identity_pairs':26,'shared_heldout_changed_pairs':74,'binding_exposure':expected_exposure,
        'manifest_sha256':r.sha(path),'verifier_sha256':r.sha(__file__),
        'scope':'Published preview adapter validation only; no model training or full SVIB score claim.'})
    print('SVIB preview adapter audit passed: 500 pairs, 13 partitions, 48 orbit overlap checks.',flush=True)


if __name__=='__main__':main()
