"""Separate initialization spread from scene uncertainty for the frozen study."""
import csv
import json
import numpy as np
from common import HERE,dump,r


def main():
    root=HERE/'reports/dynamics_state_study_v1';source=root/'summary.json';study=json.loads(source.read_text())
    assert json.loads((root/'verification.json').read_text())['all_passed']
    results=study['results'];groups=[];paired=[]
    for mode in ['isotropic','fixed_gravity','variable_force']:
        selected=[v for v in results if v['mode']==mode]
        for kind in ['flat','object','interaction','augmented','wrong_exact','joint_exact','relaxed']:
            runs=sorted([v for v in selected if v['kind']==kind],key=lambda v:v['seed'])
            assert [v['seed'] for v in runs]==[0,1,2]
            records={v['seed']:list(csv.DictReader((HERE/f'runs/dynamics_state_study_v1/{mode}/{kind}_s{v["seed"]}/rows.csv').open())) for v in runs}
            for split in runs[0]['metrics']['splits']:
                for category in ['all','no_contact','contact','pair','wall_only']:
                    metrics={}
                    for key in ['position_mae_pixels','velocity_mae_pixels_per_frame']:
                        values=[v['metrics']['splits'][split][category][key]['mean'] for v in runs]
                        by_seed={seed:{int(v['episode']):float(v[key]) for v in rr if v['split']==split and v['category']==category} for seed,rr in records.items()}
                        ids=sorted(by_seed[0]);assert all(sorted(by_seed[seed])==ids for seed in [1,2])
                        averages=[np.mean([by_seed[seed][i] for seed in [0,1,2]]) for i in ids]
                        metrics[key]={'mean':float(np.mean(values)),'initialization_sample_sd':float(np.std(values,ddof=1)),
                            'values_by_seed':values,'scene_ci95_conditional_on_three_models':r.bootstrap(averages)['base_scene_ci95']}
                    groups.append({'mode':mode,'kind':kind,'split':split,'category':category,'episodes':len(ids),'metrics':metrics})
        for split in selected[0]['metrics']['splits']:
            for comparator in ['interaction','augmented','wrong_exact','relaxed']:
                diffs=[]
                for seed in [0,1,2]:
                    a=next(v for v in selected if v['kind']=='joint_exact' and v['seed']==seed)
                    b=next(v for v in selected if v['kind']==comparator and v['seed']==seed)
                    diffs.append(a['metrics']['splits'][split]['all']['velocity_mae_pixels_per_frame']['mean']-b['metrics']['splits'][split]['all']['velocity_mae_pixels_per_frame']['mean'])
                paired.append({'mode':mode,'split':split,'comparison':f'joint_exact minus {comparator}',
                    'differences_by_seed':diffs,'mean_difference':float(np.mean(diffs)),'joint_lower_in_all_three':all(v<0 for v in diffs)})
    dump(root/'aggregate.json',{'groups':groups,'paired_comparisons':paired,'source_summary_sha256':r.sha(source),'script_sha256':r.sha(__file__),
        'uncertainty':'Three initialization values and their sample SD are reported separately from scene-bootstrap intervals conditional on those models. One data-generation seed; no population-significance claim.',
        'scope':study['claim_boundary']})
    r.write_csv(root/'comparison_rows.csv',[{'mode':v['mode'],'kind':v['kind'],'split':v['split'],'category':v['category'],
        'velocity_mae':v['metrics']['velocity_mae_pixels_per_frame']['mean'],'initialization_sd':v['metrics']['velocity_mae_pixels_per_frame']['initialization_sample_sd'],
        'position_mae':v['metrics']['position_mae_pixels']['mean']} for v in groups])
    print('Aggregated',len(groups),'groups and',len(paired),'paired comparisons')


if __name__=='__main__':main()
