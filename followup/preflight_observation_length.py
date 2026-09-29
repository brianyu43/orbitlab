"""Check hidden-input noninterference and matched capacity before any training."""
import json
import numpy as np
import torch
from common import HERE,dump,r,state_digest,digest_tensor
from dynamics_observation_length_data import NAME,DATA
from dynamics_observation_length_models import LengthEstimator,rgb_inputs,measurement_inputs,load_inputs,load_labels
from dynamics_observation_length_reference import infer_window
from dynamics_rgb_baseline import infer


@torch.no_grad()
def main():
    torch.set_num_threads(2);manifest_path=HERE/f'data/{NAME}/input_manifest.json';manifest=json.loads(manifest_path.read_text())
    assert len(manifest['groups'])==57
    rng=np.random.default_rng(913771);checks=0;initials=[]
    for kind in ['rgb_cnn','measurement_mlp']:
        for seed in [0,1,2]:
            fingerprints=[];counts=[]
            for length in [1,2,4,8]:
                torch.manual_seed(996000+seed);model=LengthEstimator(kind)
                fingerprints.append(state_digest(model));counts.append(sum(p.numel() for p in model.parameters()))
            assert len(set(fingerprints))==len(set(counts))==1
            initials.append({'kind':kind,'seed':seed,'initial_weights_sha256':fingerprints[0],'parameters':counts[0]})
    for g in manifest['groups']:
        ds,mode,split=[g[k] for k in ['data_seed','mode','split']];base=HERE/f'data/{NAME}/d{ds}'
        with np.load(base/f'inputs/{mode}/{split}/observations.npz') as a:images=a['images'][:4].copy()
        with np.load(base/f'measurements/{mode}/{split}/predictions.npz') as a:features=a['features'][:4].copy()
        labels=load_labels(ds,mode,split);label_digest={k:digest_tensor(v) for k,v in labels.items()}
        for length in [1,2,4,8]:
            altered_rgb=images.copy();altered_rgb[:,:8-length]=rng.integers(0,256,altered_rgb[:,:8-length].shape,dtype=np.uint8)
            altered_features=features.copy();altered_features[:,:8-length]=np.nan
            a=rgb_inputs(images,length);b=rgb_inputs(altered_rgb,length)
            assert torch.equal(a['images'],b['images'])
            assert not a['images'][:,:8-length].any() and (a['images'][:,8-length:,3]==1).all()
            ma=measurement_inputs(features,length);mb=measurement_inputs(altered_features,length)
            for key in ma:assert torch.equal(ma[key],mb[key]) and torch.isfinite(ma[key]).all()
            assert (ma['features'][...,:56].reshape(len(features),6,8,7)[:,:,:8-length]==0).all()
            if length==1:assert not ma['features'][...,56:70].any() and not ma['base_state'][...,2:4].any()
            if length<=2:assert not ma['base_context'].any()
            for kind,inputs in [('rgb_cnn',a),('measurement_mlp',ma)]:
                cached=load_inputs(ds,mode,split,kind,length)
                for key in inputs:assert torch.equal(inputs[key],cached[key][:4])
                torch.manual_seed(996000);model=LengthEstimator(kind).eval();other=b if kind=='rgb_cnn' else mb
                first,second=model(inputs),model(other)
                for key in first:assert torch.equal(first[key],second[key])
            if length==4:
                for f in features:
                    old=infer(f[-4:]);new=infer_window(f[-4:])
                    np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1])
            assert {k:digest_tensor(v) for k,v in load_labels(ds,mode,split).items()}==label_digest
            checks+=1
        print('length preflight',ds,mode,split,flush=True)
    dump(HERE/f'reports/{NAME}/model_preflight.json',{'all_passed':True,'groups_checked':57,'length_group_checks':checks,
        'hidden_rgb_and_nan_measurements_do_not_affect_inputs_or_outputs':True,
        'observed_availability_and_short_window_defaults_checked':True,'same_t7_targets_across_lengths':True,
        'per_architecture_capacity_and_initialization_equal_across_lengths':True,'initializations':initials,
        'sampling_plan':'Same generator seed, train size 768, batch 32 and 4000 draws for each length; actual optimizer/sampling audit after training.',
        'manifest_sha256':r.sha(manifest_path),'source_sha256':{f:r.sha(HERE/f) for f in ['preflight_observation_length.py','dynamics_observation_length_models.py','dynamics_observation_length_reference.py']},
        'scope':'Input invariants over 57 groups, four first episodes per group, all four lengths. No trained model performance claim.'})


if __name__=='__main__':main()
