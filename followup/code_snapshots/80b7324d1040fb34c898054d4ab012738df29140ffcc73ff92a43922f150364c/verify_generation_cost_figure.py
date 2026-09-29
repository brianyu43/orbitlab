"""Check all plotted points against original JSON values and run summaries."""
import json
import math
from collections import Counter
from common import HERE,dump,r


def main():
    folder=HERE/'reports/generation_cost_figure_v2'
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['source_sha256']==r.sha(HERE/'report_generation_cost.py')
    for name,sha in manifest['files'].items():assert r.sha(folder/name)==sha
    data=json.loads((folder/'chart_data.json').read_text());sources={}
    for name,sha in data['source_sha256'].items():
        assert r.sha(HERE/name)==sha;sources[name]=json.loads((HERE/name).read_text())
    assert len(data['rows'])==manifest['rows']==99
    assert len({(x['panel'],x['category'],x['series'],x['unit']) for x in data['rows']})==99
    for row in data['rows']:
        terms=[]
        for name,path in row['source_terms']:
            item=sources[name]
            for key in path:item=item[key]
            terms.append(item)
        assert math.isclose(math.fsum(terms)*row['scale'],row['value'],rel_tol=1e-13,abs_tol=1e-13)
    counts=Counter(x['panel'] for x in data['rows'])
    assert counts=={'quality_fixed':18,'quality_time':18,'train':18,'sample':18,'confirmation':18,'total':9}
    conf=sources['reports/confirmation_v1/summary.json']
    for mode,stats in conf['summary'].items():
        for split in ['seen','ood']:
            key=split+'_strict_accepted_and_joint'
            for i,ds in enumerate(sorted({x['data_seed'] for x in conf['runs']})):
                values=[x[key] for x in conf['runs'] if x['data_seed']==ds and x['mode']==mode]
                assert len(values)==3
                assert math.isclose(math.fsum(values)/3,stats[key]['per_data_seed'][i],abs_tol=1e-14)
    for tag in ['fixed','time']:
        summary=json.loads((HERE/f'reports/flow_{tag}_summary.json').read_text())
        for row in summary['records']:
            root=f"runs/flow_{tag}_s{row['seed']}_{row['mode']}"
            run=sources[root+'/run.json'];metrics=sources[root+'/samples/metrics.json']
            assert run['train_wall_seconds']==row['train_seconds']
            assert metrics['64']['sample_and_decode_seconds']==row['sample_64_seconds']
            assert metrics['64']['seen']['n']+metrics['64']['ood']['n']==576
            for split in ['seen','ood']:assert metrics['64'][split]['strict_accepted_and_joint']==row[split+'_strict_accepted_and_joint']
    dump(folder/'verification.json',{'all_passed':True,'bound_points_checked':99,'panels_checked':6,'raw_source_files_checked':len(sources),
        'manifest_sha256':r.sha(folder/'manifest.json'),'verifier_sha256':r.sha(__file__),
        'scope':'All plotted points and two-level confirmation averages verified. Original prediction audits retained. Visual inspection separate.'})
    print('generation figure verified: 99 points, 6 panels',flush=True)


if __name__=='__main__':main()
