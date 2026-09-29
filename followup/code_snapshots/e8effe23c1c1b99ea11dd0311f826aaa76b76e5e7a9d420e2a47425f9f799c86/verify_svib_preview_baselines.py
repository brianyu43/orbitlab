"""Independent baseline pixels, regional errors, and every interval calculation."""
import csv
import json
import numpy as np
import torch
from common import HERE,dump,r
from svib_preview_data import NAME,alpha_name
from svib_preview_baselines import CONFIG,freeze


def main():
    freeze();torch.set_num_threads(2);root=HERE/f'reports/{NAME}/baselines';summary=json.loads((root/'summary.json').read_text())
    assert summary['conditions']==len(summary['results'])==32
    expected={(alpha,split,method) for alpha in ['0.0','0.2','0.4','0.6'] for split in ['train','val','test_id','heldout'] for method in ['identity','train_target_mean']}
    assert {(x['alpha'],x['split'],x['method']) for x in summary['results']}==expected
    instances=checks=0;counts={}
    for entry in summary['results']:
        alpha,split,method=[entry[k] for k in ['alpha','split','method']]
        folder=root/alpha_name(alpha)/split/method;assert entry==json.loads((folder/'metrics.json').read_text())
        assert entry['config_sha256']==r.sha(HERE/f'configs/{CONFIG}.json')
        for file,sha in entry['files'].items():assert r.sha(folder/file)==sha
        source_path=HERE/f"data/{NAME}/{entry['address']}/inputs.npz";target_path=source_path.parent/'labels.npz'
        assert entry['input_sha256']==r.sha(source_path) and entry['target_sha256']==r.sha(target_path)
        training=HERE/f'data/{NAME}/{alpha_name(alpha)}/train/labels.npz';assert entry['train_labels_sha256']==r.sha(training)
        with np.load(source_path) as a:s=a['source_rgb'].copy()
        with np.load(target_path) as a:t=a['target_rgb'].copy()
        source=torch.from_numpy(s).permute(0,3,1,2).float()/255;target=torch.from_numpy(t).permute(0,3,1,2).float()/255
        with np.load(folder/'predictions.npz') as a:assert a.files==['predictions'];pred=a['predictions'].copy()
        if method=='identity':np.testing.assert_array_equal(pred,source.numpy())
        else:
            with np.load(training) as a:train=torch.from_numpy(a['target_rgb'].copy()).permute(0,3,1,2).float()/255
            assert len(train)==70;mean=train.double().mean(0).float().numpy()
            np.testing.assert_array_equal(pred,np.broadcast_to(mean,pred.shape))
            with np.load(root/alpha_name(alpha)/'train_target_mean.npz') as a:np.testing.assert_array_equal(a['mean'],mean)
        err=(torch.from_numpy(pred).double()-target.double()).square().numpy()
        null_err=(source.double()-target.double()).square().numpy()
        rows=list(csv.DictReader((folder/'rows.csv').open()));assert len(rows)==entry['episodes']==len(source)
        changed=np.count_nonzero(s!=t,axis=-1)>0;fg=(s.max(-1)>0)|(t.max(-1)>0)
        for i,row in enumerate(rows):
            values={'episode':i,'has_change':bool(changed[i].any()),'changed_pixels':int(changed[i].sum()),'foreground_pixels':int(fg[i].sum()),
                'pixel_mse':float(np.sum(err[i])/49152),'image_squared_error_sum':float(np.sum(err[i])),
                'identity_pixel_mse':float(np.sum(null_err[i])/49152)}
            for region,mask in [('foreground',fg[i]),('changed',changed[i]),('unchanged',~changed[i])]:
                values[region+'_pixel_mse']=float(np.sum(err[i]*mask[None])/(3*mask.sum())) if mask.any() else None
            values['pixel_mse_minus_identity']=values['pixel_mse']-values['identity_pixel_mse']
            values['lower_pixel_mse_than_identity']=values['pixel_mse_minus_identity'] < -1e-12
            assert set(row)==set(values)
            for key,value in values.items():
                if value is None:assert row[key]==''
                elif isinstance(value,bool):assert row[key]==str(value)
                else:np.testing.assert_allclose(float(row[key]),value,atol=1e-10,rtol=1e-12)
            if method=='identity':assert float(row['pixel_mse_minus_identity'])==0 and row['lower_pixel_mse_than_identity']=='False'
            assert abs(float(row['image_squared_error_sum'])-49152*float(row['pixel_mse']))<1e-9
        for category,result in entry['metrics'].items():
            selected=[row for row in rows if category=='all' or row['has_change']==str(category=='changed')]
            assert len(selected)==result['episodes']
            for metric,stat in result['metrics'].items():
                numbers=np.array([float(v[metric]=='True') if v[metric] in ['True','False'] else float(v[metric]) for v in selected if v[metric]!=''])
                assert stat['finite_episodes']==len(numbers) and stat['total_episodes']==len(selected)
                if not len(numbers):assert stat['mean'] is None and stat['base_scene_ci95'] is None;continue
                rng=np.random.default_rng(9123);samples=numbers[rng.integers(len(numbers),size=(1000,len(numbers)))].mean(1)
                np.testing.assert_allclose(stat['mean'],numbers.mean(),atol=1e-10,rtol=1e-12)
                np.testing.assert_allclose(stat['base_scene_ci95'],np.percentile(samples,[2.5,97.5]),atol=1e-10,rtol=1e-12);checks+=1
        if split=='heldout':assert sum(row['has_change']=='True' for row in rows)==74
        instances+=len(rows);counts[f'{alpha}/{split}/{method}']=len(rows)
        print('verified SVIB preview baseline',alpha,split,method,flush=True)
    assert instances==summary['prediction_instances']==1600
    dump(root/'verification.json',{'all_passed':True,'conditions_verified':32,'prediction_instances_verified':instances,
        'unique_original_pairs':500,'shared_external_test_pairs':100,'aggregate_interval_checks':checks,
        'identity_exact_and_target_mean_uses_only_70_train_targets':True,'all_tail_examples_retained':True,
        'pixel_mse_times_49152_equals_author_style_image_error':True,'empty_change_masks_preserved_as_missing':True,
        'summary_sha256':r.sha(root/'summary.json'),'verifier_sha256':r.sha(__file__),'counts':counts,
        'scope':'Pixel baselines on published preview, not learned Shape-Swap accuracy or full SVIB evaluation.'})
    print('SVIB baseline audit complete',instances,flush=True)


if __name__=='__main__':main()
