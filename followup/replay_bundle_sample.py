"""Read-only CPU replay at a relocated bundle path; explicit sampled scope."""
import argparse
import json
import platform
import time
import numpy as np
import torch
from common import HERE,ROOT,r
from dynamics_observation_length_models import LengthEstimator,pooled
from svib_preview_models import PreviewPredictor


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    started=time.time();torch.set_num_threads(4);records=[]
    for ds in [640031,997101,997102]:
        for kind in ['rgb_cnn','measurement_mlp']:
            for length in [1,2,4,8]:
                inputs,_=pooled(ds,kind,'val',length);inputs={k:v[:32] for k,v in inputs.items()}
                for seed in [0,1,2]:
                    folder=HERE/f'runs/dynamics_observation_length_v1/d{ds}/L{length}/{kind}_s{seed}'
                    run=json.loads((folder/'run.json').read_text())
                    assert run['checkpoint_sha256']==r.sha(folder/'model.pt')
                    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
                    model=LengthEstimator(kind);model.load_state_dict(ck['state_dict']);model.eval()
                    prediction=model(inputs);saved=torch.load(folder/'development_predictions.pt',map_location='cpu',weights_only=True)['val']
                    gaps={}
                    for key in ['state','context']:
                        expected=saved[key][:32]
                        torch.testing.assert_close(prediction[key],expected,atol=0,rtol=0)
                        gaps[key]=float((prediction[key]-expected).abs().max())
                    records.append({'family':'observation_length','data_seed':ds,'kind':kind,'length':length,'seed':seed,'episodes':32,'maximum_gaps':gaps,'checkpoint_sha256':run['checkpoint_sha256']})
    torch.set_num_threads(2)
    with np.load(HERE/'data/svib_preview_shape_swap_v1/heldout/inputs.npz') as a:
        source=torch.from_numpy(a['source_rgb'][:16].copy()).permute(0,3,1,2).float()/255
    for alpha in ['0p0','0p2','0p4','0p6']:
        for kind in ['plain','c4']:
            for seed in [0,1,2]:
                folder=HERE/f'runs/svib_preview_shape_swap_v1/alpha_{alpha}/{kind}_s{seed}'
                run=json.loads((folder/'run.json').read_text());assert run['checkpoint_sha256']==r.sha(folder/'model.pt')
                ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
                model=PreviewPredictor(kind);model.load_state_dict(ck['state_dict']);model.eval()
                prediction=model(source).numpy()
                with np.load(folder/'heldout/predictions.npz') as a:saved=a['predictions'][:16].copy()
                np.testing.assert_allclose(prediction,saved,atol=2e-7,rtol=1e-6)
                records.append({'family':'svib_preview','alpha':alpha,'kind':kind,'seed':seed,'episodes':16,'maximum_gap':float(abs(prediction-saved).max()),'checkpoint_sha256':run['checkpoint_sha256']})
    assert len(records)==96
    result={'all_passed':True,'bundle_root':str(ROOT),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'length_models_replayed':72,'length_validation_prediction_instances':2304,'svib_models_replayed':24,'svib_heldout_prediction_instances':384,
        'seconds':time.time()-started,'results':records,'script_sha256':r.sha(__file__),
        'scope':'First complete CPU batch from every length estimator and SVIB predictor, using only this relocated tree. 96 models and 2688 prediction instances, not full retraining or all predictions. Previous full replay evidence is enclosed separately.'}
    from pathlib import Path
    with Path(args.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
