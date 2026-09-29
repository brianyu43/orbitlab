"""C05/C06 source-backed narrative and diagnostic plots; no cherry-picked runs."""
import json
import numpy as np
from common import HERE,dump,r
from dynamics_autonomous import HORIZONS,METHODS,PREDICTORS
from dynamics_autonomous_study import NAME


def main():
    root=HERE/f'reports/{NAME}';v=json.loads((root/'verification.json').read_text());assert v['all_passed']
    a=json.loads((root/'aggregate.json').read_text());raw=json.loads((root/'summary.json').read_text())
    index={(x['method'],x['mode'],x['split'],x['predictor'],x['category'],x['horizon']):x for x in a['results']}
    def record(method,kind,h,category='factual',split='test'):
        return index[method,'variable_force',split,kind,category,h]
    def metric(method,kind,h,key='position_mae_pixels_finite',category='factual',split='test'):
        return record(method,kind,h,category,split)['metrics'][key]['mean_of_seed_means']
    def fmt(x):return 'Cannot be counted' if x is None else (f'{x:.3f}' if abs(x)<1e4 else f'{x:.2e}')
    labels={'oracle':'Correct answer initial state·environment','rgb_cnn':'RGB Direct CNN','measurement_mlp':'Video measurement + learning correction','rgb_analytic':'Video measurement + physical'}
    failures=sum(x['nonfinite_episodes'] for x in raw['results'])
    lines=['# A long future without giving the correct answer and behavior change','',
        '2026-09-23. Completed the fixed model evaluation and reproduction of C05/C06. A momentary prediction accuracy did not guarantee the stability of a long future or the accuracy of the behavior effects on other objects. Achieving a stable world model does not mean that.','',
        '## Execution range','',
        'Provided the state and environment obtained from the initial four frames (t=0..3) once and predicted them autonomously up to 61 steps (t=64). No correct position, speed, count, or environment were supplied in the middle. The actions were applied only to the first transition and the remaining actions were zero. The correct initial state and environment are the explicit diagnostic reference group. CNN, video measurement + learning correction, and video measurement + physical reasoning used the fixed estimates of C04.','',
        'It retained all seven existing learning methods, three initializations, constant speed, and external force/wall reflection references. It used the same variable_force weights for all environments. The learning algorithm learned by a single step loss, and the learning trajectory ends at t=19. The current 32/61 step does not exceed the time range of the learning data, but it is not a verification experiment where the data generation seed is independently changed.','',
        'The original withheld trajectory used 1,280 instances, of which 512 were identical past/opposite action responses. This is the 1,404 condition for the 4 inputs × 3 initialization × 13 groups × 9 predictors. The number of prediction instances repeated on the same data is not counted as an independent sample size. The method for measuring the circle uses known color, shape, and integration rules, and the physical reference excludes collisions between objects.','',
        '## How much wrong will you be if it takes a long time?','',
        'The joint rotation (joint_exact) model\'s position mean absolute error is the general test of 128 trajectories of varying external forces. The unit is pixels, and the smaller it is, the better. After calculating the finite prediction mean for each seed, the three values are averaged. Missing correct objects are filled with 0 coordinates and included in the error, and the error for the existence judgment and the detected objects are also separately provided in the data.','',
        '| Initial Input | 1st Stage | 16th Stage | 61st Stage | 61st Stage Limited Ratio |','| --- | ---: | ---: | ---: | ---: |']
    for m in METHODS:
        lines.append('| '+labels[m]+' | '+' | '.join(fmt(metric(m,'joint_exact',h)) for h in [1,16,61])+f" | {metric(m,'joint_exact',61,'finite_fraction')*100:.1f}% |")
    lines+=['',
        'Even if the correct answer information is provided, errors accumulate, so the problem cannot be solved using only the delayed model. Direct video CNN has large errors from the start. Conversely, the fact that the values are finite does not mean the prediction is accurate or physically plausible. In specific initializations of video measurement + physical equations, very large finite errors were also observed.','',
        'The following are the nine full prediction models for video measurement+learning correction inputs. Large values were not hidden or cut out. A finite proportion was shown together to ensure that the error, excluding failed cases, does not appear overly good.','',
        '| Predictor | 16-step position error | 61-step position error | 61-step finite ratio |','| --- | ---: | ---: | ---: |']
    for k in PREDICTORS:
        lines.append(f"| {k} | {fmt(metric('measurement_mlp',k,16))} | {fmt(metric('measurement_mlp',k,61))} | {metric('measurement_mlp',k,61,'finite_fraction')*100:.1f}% |")
    lines+=['',f'13 hold groups·Basic predictions in all modes Paths that have become limited in number among 138,240 paths{failures:,}It was a dog. This denominator is the number of repeated comparison paths and not the number of independent scenes. After the initial failure, it retained NaN and the failure indicator. Even if the finite values are very large, you cannot judge stability just by the dispersion ratio.','',
        '## Long prediction under new conditions','',
        'This is the result of video measurement + learning correction input, a joint rotation model. Without mixing the three conditions, each aesthetic properties, object number, and external force were evaluated separately.','',
        '| Conditions | 16-stage location error | 61-stage location error | Wall penetration exceeding 1px or failure to reach 61-stage level |','| --- | ---: | ---: | ---: |']
    for split,label in [('test','the public'),('attribute_ood','Non-learning size·color'),('count3','Three objects'),('count4','Your object'),('force_ood','Non-aesthetic external force')]:
        lines.append(f"| {label} | {fmt(metric('measurement_mlp','joint_exact',16,split=split))} | {fmt(metric('measurement_mlp','joint_exact',61,split=split))} | {metric('measurement_mlp','joint_exact',61,'any_wall_violation_over_1px_or_nonfinite',split=split)*100:.1f}% |")
    lines+=['',
        'The color and existence are retained as they were in the initial estimation. Therefore, the success of object tracking is not measured by the retention of the color slot. The position of the predicted same-color object is measured separately to see if it is actually closest to that object, and this correspondence is for scoring purposes. The prediction is not reordered by the correct position sequence.','',
        '## How reliable is the collision point','',
        'The model does not directly output collision events. It calculated contact candidates based on close proximity between two frames and speed changes outside of free movement. The speed residual of 0.1 pixels/frame and distance tolerance of 1.5 pixels were fixed before execution. It does not count as a collision success if there is no speed response simply because an object overlaps or passes through.','',
        'The correct answer applies the same judgment to 78,080 trajectory transfers and compared it to actual simulator events. Below is the performance of this judgment tool itself, not the performance of the learning model.','',
        '| Contact type | TP | FP | FN | Precision | Reproducibility |','| --- | ---: | ---: | ---: | ---: | ---: |']
    cal=json.loads((root/'contact_proxy_calibration/summary.json').read_text())
    for k,label in [('pair','Between objects'),('wall','a wall')]:
        sums={q:sum(x['metrics'][k][q] for x in cal['groups']) for q in ['tp','fp','fn']}
        tp,fp,fn=[sums[q] for q in ['tp','fp','fn']]
        lines.append(f'| {label} | {tp} | {fp} | {fn} | {tp/(tp+fp)*100:.1f}% | {tp/(tp+fn)*100:.1f}% |')
    lines+=['',
        'The candidate assessment is not perfect. Below is the evaluation of the initial contact point, including its limitations, as presented. It is a general test, video measurement + learning correction input, and a 61-step interval. The timing error is conditional in cases where both sides have events and the values are limited, and it must be considered together with the event omission and success within one frame.','',
        '| Predictor | Contact | First-point error (conditional frame) | Within 1 frame of the correct event | Missing correct event or numerical failure |','| --- | --- | ---: | ---: | ---: |']
    for kind in ['interaction','joint_exact','force_wall']:
        for event in ['pair','wall']:
            get=lambda suffix:metric('measurement_mlp',kind,61,event+'_'+suffix)
            lines.append(f"| {kind} | {event} | {fmt(get('first_event_abs_error_both_finite'))} | {get('first_event_within_1_on_true_event')*100:.1f}% | {get('missed_event_on_true_event')*100:.1f}% |")
    lines+=['',
        'Even force_wall, which does not calculate collisions between objects, can generate pair candidates. This is because if another object is present near the instant the speed changes due to wall reflection, it can satisfy the pair candidate rule together. It is not interpreted as evidence of object interactions learned from this. The reason for separately evaluating non-target reactions to behavioral changes is also here.','',
        '## If you change your behavior, does the reaction of other objects also match?','',
        'Only the sign of the change in speed given to the target was reversed in the same initial estimate and environment. It predicted the difference between two futures and compared it with the actual difference between the two futures. Since object collisions transmit the influence of behavior to other objects, unconditionally preserving the non-ideal is not the correct answer. The 512 response data for this evaluation are from two object scenes.','',
        'The following is the location response error in the variable external force test for video measurement + learning correction input. The “no response” comparison assumes that even if behavior changes, the predicted future remains the same. The lower it is, the better.','',
        '| Predictor | Stage | Target response error | Target non-response comparison | Non-target response error | Non-target non-response comparison |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for kind in ['joint_exact','force_wall']:
        for h in [16,61]:
            keys=['target_position_response_mae_finite','target_position_zero_response_mae','nontarget_position_response_mae_finite','nontarget_position_zero_response_mae']
            lines.append(f'| {kind} | {h} | '+' | '.join(fmt(metric('measurement_mlp',kind,h,key,'response')) for key in keys)+' |')
    lines+=['',
        'Currently, the joint rotation model predicts the target\'s own behavioral response better than the passive control in the 16th stage, but the response transmitted to the non-target was worse than this control. In the 61st stage, the superiority in the target\'s response was not maintained either. The external force and wall references omit interactions between objects, so the non-target response is exactly 0. The non-target response error in this reference is identical to the passive control, which aligns with the implementation intention.','',
        '## Verification and remaining range','',
        f"Basic prediction{v['factual_forecast_frames_replayed']:,}Predicting dog transference and counter-behavior{v['counterfactual_forecast_frames_replayed']:,}The dog transition was played back from the initial input storage. The main location, speed, number, spatial response, and overall success and behavior response errors were calculated separately using a separate code, while the remaining indicators were played back using a fixed scoring code. The color-based behavior transmission of the input, the unused number of correct answers, the persistence of numerical failures, and the inspection of data, weights, and code hashes were checked.",
        '',
        'The overall values are in `all_metrics.csv`, and the scene-level bootstrap intervals are in `aggregate.json`. We grouped the three initializations of the same scene together and preserved the denominator of the finite-sample error. This is the uncertainty within a single data generation seed and is not evidence of independent data replication or population significance.','',
        'The evaluation work for C05/C06 has been completed, but it does not claim that long-term prediction ability has been achieved. For C07, there are still gaps in the independent repetition, observation length, condition missingness, SVIB external evaluation, completion of preliminary research, actual human judgment, and final replication bundle.','',
        '[All metrics](all_metrics.csv) · [Aggregation and intervals](aggregate.json) · [Prediction replay verification](verification.json) · [Aggregation audit](aggregate_verification.json) · [Long-horizon curves](long_horizon_curves.png) · [Number failure](stability_overview.png) · [Behavior response](counterfactual_response.png) · [Fixed first case video](autonomous_examples.gif)']
    (root/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    colors={'joint_exact':'#27688b','force_wall':'#5e9660'}
    for ax,m,title in zip(axes,['oracle','measurement_mlp'],['True initial state + context','Measured + learned initial state/context']):
        for k in colors:
            seed_curves=np.array([record(m,k,h)['metrics']['position_mae_pixels_finite']['seed_means'] for h in HORIZONS])
            means=seed_curves.mean(1);ax.plot(HORIZONS,means,'o-',color=colors[k],label=k)
            for s in range(3):ax.plot(HORIZONS,seed_curves[:,s],color=colors[k],alpha=.25,linewidth=.8)
            plots.append({'plot':'long_horizon_curves','method':m,'predictor':k,'horizons':HORIZONS,'seed_values':seed_curves.tolist()})
        ax.axvline(16,color='#777',linestyle=':',label='End of training trajectory range')
        ax.set(title=title,xlabel='Autonomous forecast horizon',ylabel='Position MAE (px)')
        ax.set_xticks(HORIZONS);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Variable-force test: errors accumulate without state refresh',fontsize=13)
    for ext in ['png','svg']:fig.savefig(root/f'long_horizon_curves.{ext}',dpi=160)
    plt.close(fig)
    # All methods/models, pooled across all 1,280 factual episodes per seed.
    failure=np.zeros((len(PREDICTORS),len(METHODS)));finite_error=np.zeros_like(failure)
    for i,k in enumerate(PREDICTORS):
        for j,m in enumerate(METHODS):
            selected=[x for x in raw['results'] if x['method']==m and x['predictor']==k]
            total=sum(x['episodes'] for x in selected);failed=sum(x['nonfinite_episodes'] for x in selected)
            vals=[x['factual'][-1]['metrics']['position_mae_pixels_finite'] for x in selected]
            n=sum(x['finite_episodes'] for x in vals);err=sum(x['mean']*x['finite_episodes'] for x in vals if x['mean'] is not None)/n
            failure[i,j]=100*failed/total;finite_error[i,j]=np.log10(max(err,1e-12))
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    for ax,matrix,title,cmap in zip(axes,[failure,finite_error],['Nonfinite paths by horizon 61 (%)','Finite position MAE: log10(px)'],['Reds','magma']):
        im=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=0)
        ax.set_xticks(range(4),['True','RGB CNN','Measured\n+ learned','Measured\n+ analytic']);ax.set_yticks(range(len(PREDICTORS)),PREDICTORS);ax.set_title(title,fontsize=11)
        for i in range(len(PREDICTORS)):
            for j in range(4):
                rgb=im.cmap(im.norm(matrix[i,j]))[:3]
                luminance=np.dot(rgb,[.2126,.7152,.0722])
                label=f'{matrix[i,j]:.2f}' if cmap=='Reds' and 0<matrix[i,j]<.1 else f'{matrix[i,j]:.1f}'
                ax.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if luminance<.5 else 'black')
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('All held-out groups; repeated seed-episode paths, not independent scenes')
    for ext in ['png','svg']:fig.savefig(root/f'stability_overview.{ext}',dpi=160)
    plt.close(fig);plots.append({'plot':'stability_overview','failure_percent':failure.tolist(),'finite_position_mae_log10':finite_error.tolist(),'methods':METHODS,'predictors':PREDICTORS})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,who in zip(axes,['target','nontarget']):
        for kind,color in [('joint_exact','#27688b'),('force_wall','#5e9660')]:
            values=[metric('measurement_mlp',kind,h,who+'_position_response_mae_finite','response') for h in HORIZONS]
            ax.plot(HORIZONS,values,'o-',color=color,label=kind)
            plots.append({'plot':'counterfactual_response','object':who,'predictor':kind,'values':values})
        null=[metric('measurement_mlp','joint_exact',h,who+'_position_zero_response_mae','response') for h in HORIZONS]
        ax.plot(HORIZONS,null,'--',color='#555',label='No response to changed action')
        ax.set(title=who.capitalize()+' object',xlabel='Autonomous forecast horizon',ylabel='Position response MAE (px)');ax.set_xticks(HORIZONS);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Opposite action, identical past: measured + learned inputs, variable-force test')
    for ext in ['png','svg']:fig.savefig(root/f'counterfactual_response.{ext}',dpi=160)
    plt.close(fig)
    names=[f'{name}.{ext}' for name in ['long_horizon_curves','stability_overview','counterfactual_response'] for ext in ['png','svg']]
    dump(root/'report_manifest.json',{'report_sha256':r.sha(root/'RESULTS_KO.md'),'aggregate_sha256':r.sha(root/'aggregate.json'),
        'figures':{name:r.sha(root/name) for name in names},'plot_data':plots,'script_sha256':r.sha(__file__),
        'selection':'Two specified initial-input examples in long curves; all methods/models retained in stability plot and complete CSV. No best-seed selection.'})
    print('C05/C06 report and three figure families written')


if __name__=='__main__':main()
