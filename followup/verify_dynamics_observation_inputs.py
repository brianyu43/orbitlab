"""Verify pre-action timing and the physical separation of public inputs/labels."""
import json
import numpy as np
from common import HERE,dump,r
from dynamics_observation_inputs import NAME,freeze
from dynamics_world import render


def main():
    c=freeze();root=HERE/f'data/{NAME}';manifest=json.loads((root/'manifest.json').read_text());source=json.loads((HERE/'data/dynamics_world_v1/manifest.json').read_text())
    assert manifest['config_sha256']==r.sha(HERE/f'configs/{NAME}.json');total=frames=cf=0
    assert len(manifest['groups'])==len(source['groups'])
    for g in manifest['groups']:
        folder=root/g['mode']/g['split'];old=next(x for x in source['groups'] if (x['mode'],x['split'])==(g['mode'],g['split']))
        assert g['source_sha256']==old['archive_sha256']==r.sha(HERE/old['path'])
        for file,sha in g['files'].items():assert sha==r.sha(folder/file)
        with np.load(folder/'observations.npz') as a:assert a.files==['images'];images=a['images'].copy()
        with np.load(folder/'commands.npz') as a:assert set(a.files)=={'target_color','delta_velocity'};colors=a['target_color'].copy();impulses=a['delta_velocity'].copy()
        with np.load(folder/'labels.npz') as a:labels={k:a[k].copy() for k in a.files}
        with np.load(HERE/old['path']) as original:
            np.testing.assert_array_equal(images,original['observations']);states=original['states'];actions=original['actions'];ids=original['target_ids']
            assert not actions[:,:3].any() and not actions[:,4:].any()
            np.testing.assert_array_equal(labels['past_states'],states[:,:4]);np.testing.assert_array_equal(labels['next_states'],states[:,4])
            np.testing.assert_array_equal(labels['net_acceleration'],original['contexts'][:,:2]+original['contexts'][:,2:4]);np.testing.assert_array_equal(labels['drag'],original['contexts'][:,4])
            np.testing.assert_array_equal(labels['past_events'],original['events'][:,:3]);np.testing.assert_array_equal(labels['next_events'],original['events'][:,3]);np.testing.assert_array_equal(labels['target_ids'],ids)
            for i in range(len(images)):
                alive=states[i,3,:,4]>0;assert int(colors[i])==int(states[i,3,ids[i],5])
                assert sum(states[i,3,alive,5]==colors[i])==1
                np.testing.assert_array_equal(impulses[i],actions[i,3,ids[i]])
                reconstructed=np.zeros_like(actions[i,3]);reconstructed[ids[i]]=impulses[i];np.testing.assert_array_equal(reconstructed,actions[i,3])
                for t in range(4):
                    live=states[i,t,:,4]>0;np.testing.assert_array_equal(render(states[i,t,live])['image'],images[i,t]);frames+=1
            if 'counterfactual_states' in original:
                np.testing.assert_array_equal(original['counterfactual_states'][:,:4],states[:,:4]);cf+=len(images)
        assert len(images)==g['episodes'];total+=len(images);print('verified observation contract',g['mode'],g['split'],len(images),flush=True)
    assert total==manifest['episodes']
    dump(HERE/f'reports/{NAME}/verification.json',{'all_passed':True,'episodes':total,'past_rgb_frames_replayed':frames,
        'counterfactuals_with_identical_past':cf,'commands_address_unique_visible_colors':True,
        'future_and_privileged_labels_absent_from_model_input_archives':True,'verifier_sha256':r.sha(__file__)})


if __name__=='__main__':main()
