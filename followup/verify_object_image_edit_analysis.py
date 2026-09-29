"""Check seed/scene aggregation and descriptive error partitions independently."""
import csv
import json
import numpy as np
from common import HERE,dump,r
from object_edit_confirmation import NAME


def main():
    root=HERE/f'reports/{NAME}';a=json.loads((root/'analysis.json').read_text());study=json.loads((root/'summary.json').read_text())
    assert a['study_summary_sha256']==r.sha(root/'summary.json') and a['script_sha256']==r.sha(HERE/'analyze_object_image_edit.py')
    assert json.loads((root/'verification.json').read_text())['all_passed'];cache={};checks=0
    def rows(kind,split,control,seed):
        key=(kind,split,control,seed)
        if key not in cache:
            rr=list(csv.DictReader((HERE/f'runs/{NAME}/{kind}_s{seed}_{control}/{split}/rows.csv').open()))
            cache[key]=[{k:(None if v=='' else (v=='True' if v in ['True','False'] else (v if k=='operation' else float(v)))) for k,v in row.items()} for row in rr]
        return cache[key]
    def validate(by_seed,metrics):
        nonlocal checks
        for key,stat in metrics.items():
            data=[];ids=None
            for seed in range(3):
                rr=[v for v in by_seed[seed] if v[key] is not None];ii=sorted({v['source_index'] for v in rr})
                if ids is None:ids=ii
                assert ii==ids
                data.append([sum(v[key] for v in rr if v['source_index']==i)/sum(v['source_index']==i for v in rr) for i in ii])
            if not ids:assert stat is None;continue
            arr=np.array(data);seed_values=arr.mean(1);scene=arr.mean(0)
            np.testing.assert_allclose(seed_values,stat['values_by_seed'],atol=1e-12,rtol=0)
            assert len(ids)==stat['source_scenes'] and abs(seed_values.mean()-stat['mean'])<1e-12
            assert abs(seed_values.std(ddof=1)-stat['initialization_sample_sd'])<1e-12
            rng=np.random.default_rng(9123);b=scene[rng.integers(len(scene),size=(1000,len(scene)))].mean(1)
            np.testing.assert_allclose(np.quantile(b,[.025,.975]),stat['scene_ci95_conditional_on_three_models'],atol=1e-12,rtol=0);checks+=1
    for group in a['groups']:
        kind,split,control,op=(group[k] for k in ['kind','split','controller','group'])
        selected={s:[v for v in rows(kind,split,control,s) if op=='all' or v['operation']==op] for s in range(3)}
        validate(selected,group['metrics'])
        for key,stat in group['metrics'].items():
            if stat is None:continue
            for seed in range(3):
                original=next(v for v in study['results'] if v['name']==f'{kind}_s{seed}_{control}' and v['split']==split)
                assert abs(original['groups'][op]['metrics'][key]['mean']-stat['values_by_seed'][seed])<1e-12
    for group in a['failure_groups']:
        rr={}
        for seed in range(3):
            rr[seed]=[]
            for row in rows(group['kind'],group['split'],'object',seed):
                label=('no_detected_object' if row['selected_slot']<0 else
                    'wrong_target_selection' if not row['target_selection_correct'] else
                    'target_edited_factors_wrong' if not row['target_success_gt'] else
                    'target_untouched_factors_wrong' if not row['target_other_correct_gt'] else
                    'non_target_factors_wrong' if not row['non_target_correct_gt'] else
                    'count_mismatch' if not row['count_correct_gt'] else 'success')
                assert (label=='success')==row['end_to_end_success']
                rr[seed].append({**row,**{k:label==k for k in group['metrics']}})
        validate(rr,group['metrics'])
    scenes=json.loads((HERE/f'data/{NAME}/occlusion/scenes.json').read_text())
    for group in a['occlusion_target_visibility']:
        low,high=group['visible_fraction_lower'],group['visible_fraction_upper'];rr={}
        for seed in range(3):
            rr[seed]=[v for v in rows(group['kind'],'occlusion','object',seed)
                if (lambda f:low<=f and (f<high or high==1))(scenes[int(v['source_index'])]['visible_fraction'][int(v['target_id'])])]
        assert group['source_scenes']==len({v['source_index'] for v in rr[0]});validate(rr,group['metrics'])
    assert all(v['groups']['all']['metrics']['end_to_end_success']['mean']==0 for v in study['results'] if not v['name'].startswith('oracle_'))
    dump(root/'analysis_verification.json',{'all_passed':True,'groups_checked':len(a['groups']),'failure_groups_checked':len(a['failure_groups']),
        'visibility_groups_checked':len(a['occlusion_target_visibility']),'metric_seed_and_scene_interval_checks':checks,
        'analysis_sha256':r.sha(root/'analysis.json'),'verifier_sha256':r.sha(__file__)})
    print('Image edit analyses independently verified',checks)


if __name__=='__main__':main()
