"""Identity and train-target-mean references for every fixed preview partition."""
import json
import time
import numpy as np
from common import HERE,dump,r
from svib_preview_data import NAME,ALPHAS,alpha_name,freeze as data_freeze
from svib_preview_metrics import load_images,score,summarize

CONFIG='svib_preview_baselines_v1'


def freeze():
    data_freeze();gate=HERE/f'reports/{NAME}/data_verification.json';v=json.loads(gate.read_text());assert v['all_passed']
    assert v['manifest_sha256']==r.sha(HERE/f'data/{NAME}/manifest.json')
    path=HERE/f'configs/{CONFIG}.json'
    if not path.exists():dump(path,{'data_config_sha256':r.sha(HERE/f'configs/{NAME}.json'),
        'data_verification_sha256':r.sha(gate),'sources':{f:r.sha(HERE/f) for f in ['svib_preview_baselines.py','svib_preview_metrics.py']},
        'baselines':['identity','train_target_mean'],'source_normalization':'PIL uint8 RGB -> torch float32 /255, matching official ToTensor scale.',
        'pixel_metric':'Float64 squared differences of the float32 normalized images/predictions. Pixel MSE and per-image squared-error sum, with exact factor 49152.',
        'region_metrics':'Target-derived changed pixels and source/target nonblack union are scoring only. Empty changed region stays missing.',
        'comparison':'Same-scene difference to identity; strict improvement threshold -1e-12 is numerical, not semantic success.',
        'LPIPS':'Not executed in this preview experiment; no official full-metric replication claim.',
        'evaluation':'All examples retained, no drop_last or max_images truncation. Same 100 external test examples reused across alpha.',
        'frozen_unix':time.time()})
    c=json.loads(path.read_text());assert c['data_config_sha256']==r.sha(HERE/f'configs/{NAME}.json') and c['data_verification_sha256']==r.sha(gate)
    for f,sha in c['sources'].items():assert r.sha(HERE/f)==sha
    return c


def main():
    freeze();root=HERE/f'reports/{NAME}/baselines';results=[]
    if root.exists():raise RuntimeError(f'Existing baseline output retained: {root}')
    root.mkdir(parents=True)
    for alpha in ALPHAS:
        training=f'{alpha_name(alpha)}/train';_,train_targets,_=load_images(training)
        mean=train_targets.numpy().astype(np.float64).mean(0).astype(np.float32)
        path=root/alpha_name(alpha);path.mkdir();np.savez_compressed(path/'train_target_mean.npz',mean=mean)
        for split in ['train','val','test_id','heldout']:
            address='heldout' if split=='heldout' else f'{alpha_name(alpha)}/{split}'
            source,target,masks=load_images(address);source=source.numpy();target=target.numpy()
            for method in ['identity','train_target_mean']:
                folder=path/split/method;folder.mkdir(parents=True)
                pred=source.copy() if method=='identity' else np.broadcast_to(mean,source.shape).copy()
                rows=score(pred,source,target,masks);np.savez_compressed(folder/'predictions.npz',predictions=pred);r.write_csv(folder/'rows.csv',rows)
                item={'alpha':alpha,'split':split,'address':address,'method':method,'episodes':len(rows),'metrics':summarize(rows),
                    'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json'),
                    'train_labels_sha256':r.sha(HERE/f'data/{NAME}/{training}/labels.npz'),
                    'input_sha256':r.sha(HERE/f'data/{NAME}/{address}/inputs.npz'),'target_sha256':r.sha(HERE/f'data/{NAME}/{address}/labels.npz'),
                    'files':{f:r.sha(folder/f) for f in ['predictions.npz','rows.csv']}}
                dump(folder/'metrics.json',item);results.append(item)
        print('SVIB preview baselines evaluated',alpha,flush=True)
    dump(root/'summary.json',{'results':results,'conditions':len(results),'prediction_instances':sum(x['episodes'] for x in results),
        'unique_original_pairs':500,'shared_external_test_pairs':100,'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json'),
        'scope':'Non-learned references only; all original PNGs preserved, target mean fitted on 70 training targets per alpha. No learned model or full benchmark result.'})


if __name__=='__main__':main()
