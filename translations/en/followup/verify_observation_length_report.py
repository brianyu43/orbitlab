"""Audit every graph series, table cell and complete aggregate index in the length report."""
import csv
import json
import numpy as np
from common import HERE,dump,r
from dynamics_observation_length_data import NAME


def close(a,b):np.testing.assert_allclose(np.asarray(a,dtype=float),b,rtol=5e-12,atol=1e-12,equal_nan=True)


def display(value):
    if value is None or not np.isfinite(value):return 'Missing'
    if abs(value)>=1000 or 0<abs(value)<.0001:return format(value,'.3e')
    return format(value,'.4f')


def main():
    root=HERE/f'reports/{NAME}';folder=root/'report_v1';manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['source_sha256']==r.sha(HERE/'report_observation_length.py')
    assert manifest['protocol_sha256']==r.sha(HERE/'planning_C07_LENGTH_REPORT_KO.md')
    for file,sha in manifest['verification_gates'].items():
        assert r.sha(HERE/file)==sha;assert json.loads((HERE/file).read_text())['all_passed']
    assert manifest['curves_sha256']==r.sha(folder/'curves.json')
    assert manifest['report_sha256']==r.sha(root/'RESULTS_KO.md')
    for name,sha in manifest['files'].items():assert r.sha(folder/name)==sha
    curves=json.loads((folder/'curves.json').read_text());assert curves['lengths']==[1,2,4,8] and curves['data_seeds']==[640031,997101,997102]
    assert curves['pairs']==[[2,1],[4,1],[8,1],[4,2],[8,2],[8,4]]
    by_id={};scalar_values=0;blocks={}
    strata=['all','no_past_contact','past_contact','past_pair_contact','past_wall_contact']
    for c in curves['curves']:
        assert c['id'] not in by_id;by_id[c['id']]=c;p=HERE/c['block_manifest'];selection=c['selector']
        assert r.sha(p)==c['block_manifest_sha256'] and r.sha(p.parent/'verification.json')==c['block_verification_sha256']
        if str(p) not in blocks:
            m=json.loads(p.read_text());v=json.loads((p.parent/'verification.json').read_text())
            assert v['all_passed'] and v['block_manifest_sha256']==r.sha(p)
            assert m['files']['statistics.npz']==r.sha(p.parent/'statistics.npz')
            with np.load(p.parent/'statistics.npz') as a:arrays={k:a[k] for k in a.files}
            blocks[str(p)]=(m,arrays)
        m,arrays=blocks[str(p)]
        assert m['meta']=={k:selection[k if k!='predictor' else 'block_predictor'] for k in ['stage','method','predictor','mode','split']}
        column=m['scalar_keys'].index(selection['scalar_key']);group=strata.index(selection['stratum']);prefix='pair_' if selection['paired'] else ''
        assert c['axis']==(curves['pairs'] if selection['paired'] else curves['lengths'])
        units=arrays[prefix+'unit_mean'][:,:,:,group,column]
        close(c['unit_means'],units)
        close(c['finite_scene_counts'],arrays[prefix+'unit_count'][:,:,:,group,column])
        close(c['total_scene_counts'],arrays[prefix+'unit_total'][:,:,:,group])
        masked=np.ma.masked_invalid(units);data=masked.mean(axis=2)
        overall=data.mean(axis=1).filled(np.nan);n=data.count(axis=1)
        close(c['data_means'],data.filled(np.nan));close(c['mean'],overall)
        close(c['sd'],np.where(n>1,data.std(axis=1,ddof=1).filled(np.nan),np.nan))
        close(c['minimum'],data.min(axis=1).filled(np.nan));close(c['maximum'],data.max(axis=1).filled(np.nan));close(c['data_count'],n)
        if selection['paired']:
            close(c['common_left_unit_means'],arrays['pair_unit_left_mean'][:,:,:,group,column])
            close(c['common_right_unit_means'],arrays['pair_unit_right_mean'][:,:,:,group,column])
        scalar_values+=units.size
    assert len(by_id)==manifest['bound_curve_count']
    text=(root/'RESULTS_KO.md').read_text();cell_count=0
    for row in manifest['table_bindings']:
        assert row['line'] in text
        cells=[x.strip() for x in row['line'].split('|')[1:-1]];formatted=[]
        for binding in row['cells']:
            value=by_id[binding['curve_id']][binding['field']]
            for i in binding['indices']:value=value[i]
            if binding.get('transform')=='one_minus':value=1-value
            formatted.append(display(value));cell_count+=1
        assert cells[2:2+len(formatted)]==formatted
        if 'count_curve_id' in row:
            counts=np.array(by_id[row['count_curve_id']]['finite_scene_counts'][row['count_axis_index']])
            assert cells[-1]==str(counts.min())+'–'+str(counts.max())
    index=list(csv.DictReader((folder/'aggregate_index.csv').open()));assert len(index)==manifest['all_block_count']==520
    expected=[]
    for stage in ['observation','autonomous']:
        expected+=json.loads((root/f'aggregates_v1/{stage}_manifest.json').read_text())['blocks']
    assert {x['manifest_path'] for x in index}=={x['path'] for x in expected}
    source={x['path']:x for x in expected}
    for row in index:
        item=source[row['manifest_path']]
        assert row['manifest_sha256']==item['sha256']==r.sha(HERE/row['manifest_path'])
        for key,value in item['meta'].items():assert row[key]==str(value)
        assert int(row['unit_cells'])==item['unit_scalar_cells'] and int(row['paired_cells'])==item['paired_unit_scalar_cells']
    for phrase in ['Comparison failure rate','Common scene','No new independent data set 3 bundles have been added','Non-response contrast is the whole scene','No correct answer status is provided again']:
        assert phrase in text
    dump(root/'report_verification.json',{'all_passed':True,'curve_series_checked':len(by_id),'underlying_unit_scalar_values_checked':scalar_values,
        'numeric_table_cells_checked':cell_count,'table_rows_checked':len(manifest['table_bindings']),'aggregate_index_blocks_checked':520,
        'report_manifest_sha256':r.sha(folder/'manifest.json'),'report_sha256':r.sha(root/'RESULTS_KO.md'),'verifier_sha256':r.sha(__file__),
        'scope':'All displayed curves/table values tied to independently audited aggregates with explicit denominators; visual layout inspection is separate.'})
    print('length report audit',len(by_id),cell_count,'all 520 blocks indexed',flush=True)


if __name__=='__main__':main()
