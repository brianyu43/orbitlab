"""Check every displayed table cell and plotted scalar against audited CSV data."""
import csv
import json
import numpy as np
from common import HERE,dump,r
from dynamics_context_omission import NAME,DATA


def main():
    root=HERE/f'reports/{NAME}';m=json.loads((root/'report_manifest.json').read_text())
    for file,sha in m['files'].items():assert r.sha(root/file)==sha,file
    for file,sha in m['evidence_sha256'].items():assert r.sha(HERE/file)==sha,file
    v=json.loads((root/'across_data/verification.json').read_text());assert v['all_passed']
    assert v['manifest_sha256']==r.sha(root/'across_data/manifest.json')
    assert m['metrics_sha256']==r.sha(root/'across_data/metrics.csv') and m['source_sha256']==r.sha(HERE/'report_context_omission.py')
    fields=['condition','kind','mode','split','category','metric']
    source={tuple(row[k] for k in fields):[float(row[f'data_{ds}_mean']) if row[f'data_{ds}_mean'] else None for ds in DATA]
        for row in csv.DictReader((root/'across_data/metrics.csv').open())}
    data=json.loads((root/'report_data.json').read_text());linked=0
    for x in data['table_values']:
        assert x['data_seeds']==DATA and x['values']==source[tuple(x['key'])];linked+=len(DATA)
    plots=0
    for x in data['plot_values']:
        for condition,value in zip(x['conditions'],x['values']):
            key=(condition,'joint_exact','variable_force',x['split'],'all','velocity_mae_pixels_per_frame')
            assert value==source[key][DATA.index(x['data_seed'])];plots+=1
    assert plots==36
    body=(root/'RESULTS_KO.md').read_text();rows=[line for line in body.splitlines() if line.startswith('| ') and not line.startswith('| ---')]
    tables=[]
    conditions={'All information':'full','Hide the combined years':'no_net','Resistance concealment':'no_drag','Both are hidden':'no_context'}
    splits={'General test':'test','Your object':'count4','Non-aesthetic external force':'force_ood'}
    cells=0
    for line in rows:
        parts=[x.strip() for x in line.strip('|').split('|')]
        if parts[1]=='640031':continue
        left,right=parts[0].split(' / ')
        if left in splits:kind='joint_exact';split=splits[left]
        else:kind={'Only the status rotates':'wrong_exact','Cooperative rotation':'joint_exact'}[left];split='force_ood'
        key=(conditions[right],kind,'variable_force',split,'all','velocity_mae_pixels_per_frame')
        assert parts[1:]==[f'{x:.5f}' for x in source[key]];cells+=3;tables.append(key)
    assert len(tables)==16 and cells==48
    for split,values in data['relative_error_increase_percent'].items():
        full=source[('full','joint_exact','variable_force',split,'all','velocity_mae_pixels_per_frame')]
        hidden=source[('no_net','joint_exact','variable_force',split,'all','velocity_mae_pixels_per_frame')]
        expected=100*(np.asarray(hidden)-np.asarray(full))/np.asarray(full)
        np.testing.assert_allclose(values,expected,atol=1e-12,rtol=1e-12)
        assert all(f'{value:.1f}%' in body for value in expected) and (expected>0).all()
    raw=json.loads((root/'verification.json').read_text());pairs=raw['zero_net_structure_training_comparisons']
    assert len(pairs)==18 and all(x['weights_exactly_equal'] and x['max_weight_difference']==0 for x in pairs)
    dump(root/'report_verification.json',{'all_passed':True,'table_cells_verified':cells,'plot_scalars_verified':plots,
        'linked_table_data_scalars_checked':linked,'relative_changes_checked':6,'same_weight_pairs_checked':18,
        'report_manifest_sha256':r.sha(root/'report_manifest.json'),'verifier_sha256':r.sha(__file__),
        'scope':'Numerical and artifact verification. Figure layout requires separate visual inspection.'})
    print('context report audit passed',cells,'table cells and',plots,'plot scalars',flush=True)


if __name__=='__main__':main()
