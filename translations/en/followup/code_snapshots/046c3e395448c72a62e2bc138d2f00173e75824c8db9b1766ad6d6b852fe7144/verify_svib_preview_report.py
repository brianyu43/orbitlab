"""Check report tables, graph bindings and every fixed example against audited data."""
import csv
import json
import numpy as np
from common import HERE,dump,r
from svib_preview_data import NAME,ALPHAS


def main():
    root=HERE/f'reports/{NAME}';out=root/'report_v1';manifest=json.loads((out/'manifest.json').read_text())
    summary=json.loads((root/'aggregate_v1/summary.json').read_text());audit=json.loads((root/'aggregate_v1/verification.json').read_text())
    assert audit['all_passed'] and manifest['summary_sha256']==audit['summary_sha256']==r.sha(root/'aggregate_v1/summary.json')
    assert manifest['aggregate_verification_sha256']==r.sha(root/'aggregate_v1/verification.json')
    assert manifest['source_sha256']==r.sha(HERE/'report_svib_preview.py')
    assert manifest['report_sha256']==r.sha(root/'RESULTS_KO.md')
    for name,sha in manifest['files'].items():assert r.sha(out/name)==sha
    assert set(manifest['files'])=={'preview_comparison.png','preview_comparison.svg','examples_alpha_0p0.png','examples_alpha_0p6.png'}
    seen=set()
    for binding in manifest['bindings']:
        assert binding['label'] not in seen;seen.add(binding['label'])
        data=summary['paired_conditions'] if binding['kind']=='paired' else summary['conditions']
        assert binding['data'] in data
    text=(root/'RESULTS_KO.md').read_text();tables=0
    def lookup(a,k,metric='pixel_mse',category='all'):
        return next(x for x in summary['conditions'] if (x['alpha'],x['method'],x['split'],x['category'],x['metric'])==(a,k,'heldout',category,metric))
    paired=[]
    for a in ALPHAS:
        cells=[lookup(a,k) for k in ['identity','train_target_mean','plain','c4']]
        assert '| '+a+' | '+' | '.join(f"{x['mean']*1000:.3f}" for x in cells)+' |' in text;tables+=4
        cells=[lookup(a,k,'changed_pixel_mse','changed') for k in ['plain','c4']]+[lookup(a,k,category='unchanged') for k in ['plain','c4']]
        assert '| '+a+' | '+' | '.join(f"{x['mean']*1000:.3f}" for x in cells)+' |' in text;tables+=4
        values=[]
        for key in ['training_seconds','cpu_forward_seconds_100_images']:
            for k in ['plain','c4']:values.append(np.mean([x[key] for x in summary['costs'] if x['alpha']==a and x['method']==k]))
        assert '| '+a+' | '+' | '.join(f'{x:.3f}' for x in values)+' |' in text;tables+=4
        paired.append(next(x for x in summary['paired_conditions'] if (x['alpha'],x['left'],x['right'],x['split'],x['category'],x['metric'])==(a,'c4','plain','heldout','all','pixel_mse')))
    assert manifest['derived_counts']=={'c4_better_alpha_means':sum(x['mean']<0 for x in paired),
        'c4_better_seed_pairs':sum(v<0 for x in paired for v in x['per_seed_mean']),
        'model_better_than_identity':{k:sum(lookup(a,k)['mean']<lookup(a,'identity')['mean'] for a in ALPHAS) for k in ['plain','c4']}}
    assert manifest['cost_records']==summary['costs']
    p=HERE/f'data/{NAME}/heldout/inputs.npz';t=p.parent/'labels.npz'
    assert manifest['all_example_input_sha256']==r.sha(p) and manifest['all_example_target_sha256']==r.sha(t)
    with np.load(p) as a:source=a['source_rgb'].copy()
    with np.load(t) as a:target=a['target_rgb'].copy()
    changed=np.any(source!=target,axis=(1,2,3));selection=manifest['fixed_example_selection']
    indices=np.r_[np.flatnonzero(changed)[:3],np.flatnonzero(~changed)[:2]].tolist()
    assert selection=={'changed':indices[:3],'unchanged':indices[3:],'alphas':[ALPHAS[0],ALPHAS[-1]],'seed':0}
    examples=set()
    for example in manifest['examples']:
        a,k,i=example['alpha'],example['method'],example['episode'];key=(a,k,i);assert key not in examples;examples.add(key)
        assert example['seed']==(None if k=='train_target_mean' else 0)
        predpath=HERE/example['prediction_path'];rowpath=HERE/example['rows_path']
        assert r.sha(predpath)==example['prediction_sha256'] and r.sha(rowpath)==example['rows_sha256']
        rows=list(csv.DictReader(rowpath.open()));assert float(rows[i]['pixel_mse'])==example['pixel_mse']
        with np.load(predpath) as z:pred=z['predictions'][i]
        actual=(target[i].astype(np.float32)/np.float32(255)).transpose(2,0,1)
        np.testing.assert_allclose(np.mean((pred.astype(np.float64)-actual.astype(np.float64))**2),example['pixel_mse'],atol=1e-10,rtol=1e-12)
    assert examples=={(a,k,i) for a in [ALPHAS[0],ALPHAS[-1]] for k in ['train_target_mean','plain','c4'] for i in indices}
    for phrase in ['Reduce external task exploration','It is not the result of replicating the entire SVIB benchmark performance.','LPIPS','Unmeasured','Conditional bootstrap','New dataset','It is not a re-run of the entire MPS learning']:
        assert phrase in text
    dump(root/'report_verification.json',{'all_passed':True,'numeric_table_cells_checked':tables,'plot_and_table_bindings_checked':len(seen),
        'fixed_example_prediction_cells_checked':len(examples),'no_best_case_or_seed_selection':True,
        'report_manifest_sha256':r.sha(out/'manifest.json'),'report_sha256':r.sha(root/'RESULTS_KO.md'),'verifier_sha256':r.sha(__file__),
        'scope':'Every report numeric table, graph binding and fixed-image source checked. Separate visual inspection is still required.'})
    print('SVIB report verified',tables,len(seen),len(examples),flush=True)


if __name__=='__main__':main()
