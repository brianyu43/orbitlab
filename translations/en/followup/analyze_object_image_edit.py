"""Scene-grouped summaries and declared post-hoc failure strata, no model changes."""
import csv
import json
import numpy as np
from common import HERE,dump,r
from object_edit_confirmation import NAME

KEYS=['target_selection_correct','target_success_gt','target_other_correct_gt','non_target_correct_gt','count_correct_gt',
      'end_to_end_success','full_command_success_relative_to_prediction','non_target_preserved_relative_to_prediction',
      'image_foreground_mae','changed_pixels_mae','input_passthrough_foreground_mae']


def read_rows(path):
    result=[]
    for row in csv.DictReader(path.open()):
        out={}
        for k,v in row.items():
            if v in ['True','False']:out[k]=v=='True'
            elif v=='':out[k]=None
            elif k=='operation':out[k]=v
            else:out[k]=float(v)
        result.append(out)
    return result


def seed_scene_stats(by_seed,key):
    arrays=[];ids=None
    for seed in range(3):
        rows=by_seed[seed];valid=[v for v in rows if v[key] is not None];current=sorted({v['source_index'] for v in valid})
        if ids is None:ids=current
        assert current==ids
        arrays.append([np.mean([v[key] for v in valid if v['source_index']==i]) for i in ids])
    if not ids:return None
    a=np.asarray(arrays);seeds=a.mean(1)
    return {'mean':float(seeds.mean()),'values_by_seed':seeds.tolist(),'initialization_sample_sd':float(seeds.std(ddof=1)),
        'scene_ci95_conditional_on_three_models':r.bootstrap(a.mean(0))['base_scene_ci95'],'source_scenes':len(ids)}


def main():
    root=HERE/f'reports/{NAME}';assert json.loads((root/'verification.json').read_text())['all_passed']
    study=json.loads((root/'summary.json').read_text());groups=[];failures=[];visibility=[];cache={}
    splits=['test','ood','count3','count4','occlusion']
    failure_keys=[('no_detected_object',None),('wrong_target_selection','target_selection_correct'),('target_edited_factors_wrong','target_success_gt'),
        ('target_untouched_factors_wrong','target_other_correct_gt'),('non_target_factors_wrong','non_target_correct_gt'),('count_mismatch','count_correct_gt')]
    for kind in ['flat','slot']:
        for split in splits:
            for control in ['identity','analytic','object']:
                by_seed={seed:read_rows(HERE/f'runs/{NAME}/{kind}_s{seed}_{control}/{split}/rows.csv') for seed in range(3)}
                cache[(kind,split,control)]=by_seed
                group_names=['all']+sorted({v['operation'] for v in by_seed[0]})
                for group in group_names:
                    rr={s:[v for v in rows if group=='all' or v['operation']==group] for s,rows in by_seed.items()}
                    groups.append({'kind':kind,'split':split,'controller':control,'group':group,'metrics':{key:seed_scene_stats(rr,key) for key in KEYS}})
                if control=='object':
                    categorized={}
                    for seed,rows in by_seed.items():
                        out=[]
                        for row in rows:
                            label='success'
                            for stage,key in failure_keys:
                                failed=row['selected_slot']<0 if key is None else not row[key]
                                if failed:label=stage;break
                            assert (label=='success')==bool(row['end_to_end_success'])
                            out.append({**row,**{k:label==k for k,_ in failure_keys},'success':label=='success'})
                        categorized[seed]=out
                    metrics={k:seed_scene_stats(categorized,k) for k in [x[0] for x in failure_keys]+['success']}
                    assert abs(sum(v['mean'] for v in metrics.values())-1)<1e-12
                    failures.append({'kind':kind,'split':split,'metrics':metrics})
            if split=='occlusion':
                scenes=json.loads((HERE/f'data/{NAME}/{split}/scenes.json').read_text());by_seed=cache[(kind,split,'object')]
                for low,high in [(0,.5),(.5,.75),(.75,.9),(.9,1.000001)]:
                    selected={}
                    for seed,rows in by_seed.items():
                        selected[seed]=[v for v in rows if low<=scenes[int(v['source_index'])]['visible_fraction'][int(v['target_id'])]<high]
                    visibility.append({'kind':kind,'visible_fraction_lower':low,'visible_fraction_upper':min(high,1),
                        'source_scenes':len({v['source_index'] for v in selected[0]}),'metrics':{key:seed_scene_stats(selected,key) for key in KEYS}})
    aggregate={'groups':groups,'exclusive_failure_priority':failure_keys,'failure_groups':failures,'occlusion_target_visibility':visibility,
        'scope':'Post-hoc descriptive failure strata; no selection or training. Seed spread is separate from conditional source-scene intervals. Failure priority is a reporting partition, not an independent causal decomposition.',
        'study_summary_sha256':r.sha(root/'summary.json'),'script_sha256':r.sha(__file__)}
    dump(root/'analysis.json',aggregate)
    write_report(root,aggregate,study)
    make_figure(root,aggregate)


def write_report(root,a,study):
    def m(kind,split,key,group='all',control='object'):
        return next(v for v in a['groups'] if (v['kind'],v['split'],v['controller'],v['group'])==(kind,split,control,group))['metrics'][key]['mean']
    labels={'test':'Test two objects','ood':'Unlearned shapes and colors','count3':'Three objects','count4':'Your object','occlusion':'Covering'}
    lines=['# Actual editing result after entering the picture and target point markers','',
        '2026-09-23. We completed the connection of the fixed model for B05 and the evaluation of the new verification data. We connected the existing encoder, supervised learning reader, and intervention model without any training. This does not mean that we have actually achieved the ability to stably edit objects.','',
        '## What the experiment asks','',
        'Can you read the object state in the image, change the color, direction, and position of the object pointed by a dot, and then redraw the image? The model only provided RGB values for the (x,y) coordinates of the dot over the visible object and explicit editing commands. It is not a natural language command. The correct answers—state, object count, mask, and ID—are not passed to the image model.','',
        'I created 2,961 tasks by applying color changes, 90-degree rotations, 5-pixel moves, and aesthetic color combinations that are possible after color changes and rotations, to 640 new scenes. Each scene has 128 images for the general/non-learning combination, three objects, four objects, and occlusion. There was no overlap between the existing 8,148 object scenes/editing rotation orbits and the new source family. Multiple commands in the same scene are not counted as independent samples.','',
        'The model leaves four slots with a prediction existence score out of five candidates, resulting in a maximum of four object capacities. The number of objects is not taken from the correct answer. The target is selected as the object closest to the point of the prediction center. The coordinate-based evaluation response before editing is retained even after editing. After coloring, rotation is performed sequentially in the same prediction slot without inserting the intermediate correct answer.','',
        'Each encoder uses one initialization, and the matching intervention model uses three initializations. In each method, we compared the identity/explicit transformation rules and the learned intervention models for each object. Separately, we evaluated the correct state+target and correct state+point selection comparison groups. The original encoder only learned RGB, but the state reader is a supervised learning model that uses the train correct attributes.','',
        '## Selection of the target and overall success','',
        'Below are three initialization means connected to the intervention model learned. The selected answer is the evaluation based on the coordinate transformation before editing. For full success, all target selection, target modification, remaining properties of the target, non‑target properties, and number of objects must be correct. A 1‑pixel tolerance on each coordinate axis and a 0.5‑pixel radius are allowed.','',
        '| Conditions | Select flat target | Select Slot target | Command successful based on flat estimated status | Command successful based on Slot estimated status | Successful for all actual answers |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for split,label in labels.items():
        vals=[m(k,split,key)*100 for key in ['target_selection_correct','full_command_success_relative_to_prediction'] for k in ['flat','slot']]
        full=[m(k,split,'end_to_end_success') for k in ['flat','slot']];assert full==[0,0]
        lines.append('| '+label+' | '+' | '.join(f'{v:.1f}%' for v in vals)+'| 0 observations for both methods |')
    lines+=['',
        'Command success based on the estimated state means that the model has recognized the shape and position it incorrectly as the starting state and has only performed the requested changes. It does not mean that it matches the actual answer. For example, if it already incorrectly read the shape of another object, even if it preserves that error, the preservation based on the estimated state will still succeed.','',
        'In all RGB methods, three initializations, three interventions, and five conditions, there were only 0 strict total successes. This is an observation of this sample and does not mean that the success probability of the population is exactly 0.','',
        '## If you give the correct state to the same command model','',
        'The learning intervention model that provided the correct state and the target ID for the correct answer achieved 100% overall success in all three initializations of the five new conditions. The explicit rule comparison group correctly matched both the state and the output image. In contrast, even with a correct state, the rule that selects the center closest to the target score only achieved 92.2% success in the masking condition. In this case, there are also limitations in the target selection rule beyond recognition.','',
        'Even applying explicit answer transformation rules to the estimated state results in zero total successes. Therefore, the error state that the current system perceives as only making the controller perfect cannot be resolved by this approach. It does not conclude that the encoder lacks information in the first place, nor does it conclude that other readers or learning methods fail.','',
        '## Matching target properties by command type','',
        'This is the test result for two objects that only scored the properties that need to change for the actual target, excluding the overall preservation conditions. Claiming success with editing based on only this value misses the error.','',
        '| Command | flat | Slot Attention |','| --- | ---: | ---: |']
    op_names={'color':'Color change','rotate':'90-degree rotation','translate':'east of ‥','color_then_rotate':'Rotate after color change','novel_color_pair':'Unlearned color combination'}
    for op,label in op_names.items():lines.append(f'| {label} | {100*m("flat","test","target_success_gt",op):.1f}% | {100*m("slot","test","target_success_gt",op):.1f}% |')
    lines+=['','## Error in output picture','',
        'I saved 68,103 outputs as actual prediction states and played them all back. Using evaluation response, I did not match the drawing\'s front-back order to the correct answer. The prediction slot order is the order in which the drawings are drawn, and depth was not learned separately. Coordinate and size rounding and limits were applied only when drawing the drawings, and the state score was calculated as the original real number.','',
        '| Conditions | Flat view error | Slot view error | Reference showing the original picture as it is |','| --- | ---: | ---: | ---: |']
    for split,label in labels.items():lines.append(f'| {label} | {m("flat",split,"image_foreground_mae"):.3f} | {m("slot",split,"image_foreground_mae"):.3f} | {m("slot",split,"input_passthrough_foreground_mae"):.3f} |')
    lines+=['',
        'Setting RGB values to 0 to 1, the average absolute error was calculated from the intersection of the input and correct preview areas. The current editing output is significantly larger in error than that of a reference that returns the input unchanged on average. This serves as a contrast to the fact that the input return does not execute commands, making it an editing tool that cannot be considered useful solely due to high preservation.','',
        '## Covering and error analysis','',
        'The results for object count by category and command type, as well as the visible proportion of the target, were left in four sections in analysis.json. Additional failure classifications are displayed in the order of first failure: No detection → Target selection → Change property → Target remaining property → Non-target property → Number of instances. This order is a reporting classification and does not represent the independent causal effect of each cause.','',
        'Even in occlusion, all targets were required to have at least one visible hard pixel. Therefore, it is not the restoration performance of completely occluded targets. The geometry-only handling of overlapping objects and the lack of depth learning are also limitations of interpretation. It was executed from a single new data generation seed, and the rules of the learning material and the renderer are the same synthetic world.','',
        '## Verification and next step','',
        'I played back 640 data files, 2,961 edits, and 2,560 global rotations. I re-verified 1,280 RGB encoders, 3,840 reading predictions, 68,103 edits/outputs under 115 comparison conditions, and 17,388 scene-specific indicators and segments. The fixed code before evaluation on the reader and controller remained unchanged.','',
        'The connection and verification evaluation for B05 has been completed, and the performance limits are recorded as they are. The full failure analysis for B07 is concluded by linking the existing separation errors with this intervention error. The independent SVIB external evaluation and video-based movement and long-term prediction are also continued.','',
        '[Overall summary](summary.json) · [Initialization/Scene and Failure Analysis](analysis.json) · [Data verification](data_verification.json) · [Prediction/Output verification](verification.json) · [Comparison chart](capability_curves.png)']
    (root/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')


def make_figure(root,a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    splits=['test','ood','count3','count4','occlusion'];x=np.arange(5)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    def stats(kind,key):
        return [next(v for v in a['groups'] if (v['kind'],v['split'],v['controller'],v['group'])==(kind,split,'object','all'))['metrics'][key] for split in splits]
    for kind,color in [('flat','#bf6630'),('slot','#2864a5')]:
        for ax,key in [(axes[0],'target_selection_correct'),(axes[1],'full_command_success_relative_to_prediction')]:
            ss=stats(kind,key);means=np.array([v['mean']*100 for v in ss]);low=np.array([v['scene_ci95_conditional_on_three_models'][0]*100 for v in ss]);high=np.array([v['scene_ci95_conditional_on_three_models'][1]*100 for v in ss])
            ax.errorbar(x,means,yerr=[means-low,high-means],marker='o',capsize=3,label=kind,color=color)
        axes[2].plot(x,[v['mean'] for v in stats(kind,'image_foreground_mae')],marker='o',label=kind,color=color)
    axes[1].axhline(0,color='#555',linestyle='--',label='Actual full-scene success: both 0')
    axes[2].plot(x,[v['mean'] for v in stats('slot','input_passthrough_foreground_mae')],marker='s',linestyle='--',color='#555',label='Input image unchanged')
    for ax in axes:
        ax.set_xticks(x,['2 objects','Pair OOD','3 objects','4 objects','Occlusion'],rotation=20);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    axes[0].set(title='Target selection',ylabel='Correct (%)',ylim=(-5,105));axes[1].set(title='Command follows perceived state',ylabel='Success (%)',ylim=(-5,105));axes[2].set(title='Actual output image error',ylabel='Foreground MAE (0–1)',ylim=(0,.4))
    fig.suptitle('Frozen RGB/click edit pipeline — 640 fresh scenes, three head/controller seeds',fontsize=13)
    fig.savefig(root/'capability_curves.png',dpi=160);fig.savefig(root/'capability_curves.svg');plt.close(fig)
    dump(root/'analysis_manifest.json',{'analysis_sha256':r.sha(root/'analysis.json'),'report_sha256':r.sha(root/'RESULTS_KO.md'),
        'figures':{f:r.sha(root/f) for f in ['capability_curves.png','capability_curves.svg']},'script_sha256':r.sha(__file__),
        'intervals':'Conditional base-source-scene bootstrap, not independent-dataset uncertainty.'})


if __name__=='__main__':main()
