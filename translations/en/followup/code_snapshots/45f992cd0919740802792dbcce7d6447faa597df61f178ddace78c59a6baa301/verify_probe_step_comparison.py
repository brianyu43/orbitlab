"""Replay frozen encodings, all validation candidates and chosen test probes."""
import json
import time
import joblib
import numpy as np
import torch
from common import HERE,ROOT,dump,r,o
from diagnostics import probe_data


@torch.no_grad()
def main():
    torch.set_num_threads(4);root=HERE/'reports/probe_step_comparison_v1'
    settings=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
    assert settings['source_sha256']==summary['source_sha256']==r.sha(HERE/'probe_step_comparison.py')
    assert settings['probe_data_manifest_sha256']==r.sha(HERE/'data/probe_v1/manifest.json')
    expected={(steps,kind,seed) for steps in [3000,6000] for kind in ['aug','equivariant'] for seed in [0,1,2]}
    assert {(x['steps'],x['kind'],x['seed']) for x in summary['results']}==expected
    validation_candidates=prediction_instances=0
    for row in summary['results']:
        folder=root/f"{row['kind']}_{row['steps']}_s{row['seed']}"
        assert json.loads((folder/'result.json').read_text())==row
        assert row['protocol_sha256']==r.sha(root/'protocol.json')
        for name,sha in row['files'].items():assert r.sha(folder/name)==sha
        path=ROOT/row['checkpoint'];assert r.sha(path)==row['checkpoint_sha256']
        model,_=r.load_model(path,torch.device('cpu'));model.eval()
        with np.load(folder/'features.npz') as a:features={k:a[k] for k in a.files}
        with np.load(folder/'labels.npz') as a:labels={k:a[k] for k in a.files}
        with np.load(folder/'predictions.npz') as a:saved={k:a[k] for k in a.files}
        for split in settings['splits']:
            images,truth=probe_data(split);parts=[]
            for rotation in range(4):
                for start in range(0,len(images),64):
                    batch=torch.rot90(images[start:start+64],rotation,(-2,-1))
                    parts.append(model.encode(batch).flatten(1).numpy())
            np.testing.assert_array_equal(np.concatenate(parts),features[split])
            np.testing.assert_array_equal(labels[split][:,:2],np.tile(truth.numpy(),(4,1)))
            np.testing.assert_array_equal(labels[split][:,2],np.repeat(np.arange(4),len(truth)))
        fitted=joblib.load(folder/'fitted.joblib');scaler=fitted['scaler']
        np.testing.assert_allclose(scaler.mean_,features['train'].mean(0,dtype=np.float64),atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(scaler.var_,features['train'].var(0,dtype=np.float64),atol=1e-12,rtol=1e-12)
        assert scaler.n_samples_seen_==len(features['train'])
        values={s:scaler.transform(x).astype(np.float64) for s,x in features.items()}
        for spec,result in row['scores'].items():
            target,family=spec.split('/');index={'shape':0,'color':1,'rotation':2}[target];scores=[]
            for C in settings[family+'_C']:
                key=f'{target}_{family}_C{C}';prediction=fitted['estimators'][key].predict(values['val'])
                np.testing.assert_array_equal(prediction,saved[key+'_val'])
                scores.append(float((prediction==labels['val'][:,index]).sum()/len(prediction)))
                prediction_instances+=len(prediction);validation_candidates+=1
            np.testing.assert_array_equal(scores,result['validation_grid_accuracies'])
            assert result['C']==settings[family+'_C'][int(np.argmax(scores))]
            key=f"{target}_{family}_C{result['C']}"
            for split in ['test','ood']:
                pred=fitted['estimators'][key].predict(values[split]);np.testing.assert_array_equal(pred,saved[key+'_'+split])
                expected_score=float((pred==labels[split][:,index]).sum()/len(pred))
                assert expected_score==result[split]['mean'] and len(pred)==4*result[split]['base_scenes']
                prediction_instances+=len(pred)
        print('verified matched probe',row['kind'],row['steps'],row['seed'],flush=True)
    lines=['# Comparison of the same material for 3,000 and 6,000 learning expressions','',
        'We did not change the original weights, but read 12 checkpoints from two training lengths using the CPU in the same train/validation/test/OOD environments. The standardization and reader were trained only on the train set, and C was selected based on validation accuracy. Each sample\'s four rotations are grouped into the same scene. We aligned the model, initialization, full representation space, and reader settings.','',
        'The table shows the three initialization mean accuracy (%) and difference (%p). The independent data generation seed is one, and the continuous training intermediate checkpoint was not saved and compared. The 6,000 values in this table are also the result of re‑tuning the decoder under the same CPU conditions, so they may differ slightly from the rounded value of the previous MPS‑encoding‑based diagnosis. It is a different indicator from the success rate of generation.','',
        '| Model / Reading items | Material | 3,000 times | 6,000 times | Difference (%p) |','| --- | --- | ---: | ---: | ---: |']
    bindings=[]
    for kind in ['aug','equivariant']:
        for metric in ['shape/linear','shape/rbf','color/linear','color/rbf','rotation/linear']:
            for split in ['test','ood']:
                groups={steps:[x['scores'][metric][split]['mean'] for x in summary['results'] if x['kind']==kind and x['steps']==steps] for steps in [3000,6000]}
                means={k:sum(v)/len(v)*100 for k,v in groups.items()};diff=means[6000]-means[3000]
                line=f"| {kind} / {metric} | {split} | {means[3000]:.2f} | {means[6000]:.2f} | {diff:+.2f} |";lines.append(line)
                bindings.append({'kind':kind,'metric':metric,'split':split,'per_seed':groups,'means_percent':means,'difference_pp':diff,'line':line})
    lines+=['','The full shape reading of C4 was 41.67% linear and 75.00% nonlinear in the general test. In the non-learned combination, it was 32.81% nonlinear. Although there were more clues to read shape information in expressions that were learned for a longer period, linear reading or non-learned combinations remain weak. In aug, the shape reading in the general test did not increase.','',
        'The success of a simple reader is not evidence that it changes properties independently or accurately generates images. The previously completed 6,000 m0/m2/m13 diagnoses were preserved as is, and the range of this additional comparison is the full spatial domain. It is a search conducted on the same evaluation data and not a new independent verification experiment.','',
        '[Overall score](summary.json) · [Input/selection rules](protocol.json) · [Replay verification](verification.json) · [Existing 6,000 partial space diagnosis](../probes_6000_v1/summary.json)']
    report=root/'RESULTS_KO.md';report.write_text('\n'.join(lines)+'\n')
    dump(root/'table_bindings.json',bindings)
    dump(root/'verification.json',{'all_passed':True,'checkpoints_verified':12,'all_four_split_encodings_replayed':True,
        'validation_candidates_replayed':validation_candidates,'classifier_prediction_instances_replayed':prediction_instances,
        'table_rows_checked':len(bindings),'summary_sha256':r.sha(root/'summary.json'),'report_sha256':r.sha(report),
        'verifier_sha256':r.sha(__file__),'completed_unix':time.time(),
        'scope':'All saved embeddings, all validation candidates and chosen test/OOD predictions checked, plus train-only scaler moments and fixed hyperparameter selection. Fitted probes replayed; original AE training not repeated.'})
    print('matched checkpoint comparison verified',validation_candidates,prediction_instances,flush=True)


if __name__=='__main__':main()
