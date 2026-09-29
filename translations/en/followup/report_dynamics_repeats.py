"""Verified three-dataset comparison; negative results and finite denominators remain visible."""
import csv
import json
import numpy as np
from common import HERE,dump,r
from dynamics_repeat_data import NAME

DATA=[640031,997101,997102]


def main():
    root=HERE/f'reports/{NAME}';tables=root/'across_data';index={};evidence={}
    keys=['stage','method','mode','split','predictor','information','category','horizon','metric']
    for stage in ['state','observation','autonomous']:
        vp=tables/f'{stage}_verification.json';v=json.loads(vp.read_text());assert v['all_passed']
        assert v['manifest_sha256']==r.sha(tables/f'{stage}_manifest.json');evidence[str(vp.relative_to(HERE))]=r.sha(vp)
        m=json.loads((tables/f'{stage}_manifest.json').read_text())
        assert m['files'][f'{stage}_metrics.csv']==r.sha(tables/f'{stage}_metrics.csv')
        with (tables/f'{stage}_metrics.csv').open() as f:
            for row in csv.DictReader(f):
                if row['mode']!='variable_force':continue
                key=tuple(row[k] for k in keys)
                index[key]=[None if row[f'data_{ds}_mean']=='' else float(row[f'data_{ds}_mean']) for ds in DATA]
    def values(stage,method,split,predictor,metric,information='',category='all',horizon=''):
        return index[(stage,method,'variable_force',split,predictor,information,category,str(horizon),metric)]
    def fmt(v):
        if v is None:return 'Not applicable'
        return f'{v:.3e}' if abs(v)>=10000 else f'{v:.5f}'
    def cells(v):return ' | '.join(fmt(x) for x in v)
    def state(split,kind):return values('state','oracle',split,kind,'velocity_mae_pixels_per_frame')
    def observation(method,split):return values('observation',method,split,'joint_exact','zero_filled_true_slot_velocity_mae',information='estimated_state_estimated_context')
    def future(split,h,metric='position_mae_pixels_finite',kind='joint_exact',method='measurement_mlp'):
        return values('autonomous',method,split,kind,metric,category='factual',horizon=h)
    def response(h,metric):return values('autonomous','measurement_mlp','test','joint_exact',metric,category='response',horizon=h)
    joint_improvement=[100*(a-b)/a for a,b in zip(state('test','interaction'),state('test','joint_exact'))]
    physical_better=sum(b<a for a,b in zip(state('test','joint_exact'),state('test','force_wall')))
    correction_wins={split:sum(a<b for a,b in zip(observation('measurement_mlp',split),observation('rgb_analytic',split))) for split in ['test','force_ood']}
    lines=['# Exercise prediction rechecked from 3 independent data sets','',
        f'The design that rotates both the state and the external force reduced the speed error in a single step of general conditions to both data sets lower than the general interaction model. The reduction range is{min(joint_improvement):.1f}~{max(joint_improvement):.1f}% was. The error in the physical reference that directly calculates known forces and walls was smaller in all three data sets.','',
        f'Data that outperformed the non-learning physical equations in video measurement learning correction are under general conditions.{correction_wins["test"]}/3 pieces, non-learned external force{correction_wins["force_ood"]}/It was 3 pieces. I could not generalize the superiority and inferiority shown in one data to a general conclusion.','',
        f'The long future was not stable. The general conditions of video measurement + learning correction and joint rotation prediction 61-stage position error are different for each data.{", ".join(fmt(v) for v in future("test",61))}It was a pixel. It was a value including the large dispersion that has a finite number of remaining values. Therefore, it does not claim that a gain at one stage led to the completion of a long-term world model.','',
        'We compared the existing data 640031 with the new data 997101·997102. In the new dataset, we trained 126 state prediction models and 12 video estimation models from scratch. We retained the existing initialization order, sample extraction sequence, number of training iterations, model list, and evaluation rules. This report is the independent repeated part of C07, and the experiment with missing observation lengths and environmental information is separate.','',
        '## Comparison units and verification','',
        'Each table\'s one column represents a single independent data generation seed. Each cell is the mean of the fixed three learning initializations, and the initializations were counted as nine independent data samples, not nine. Non-learning references are calculated only once per dataset. The full CSV contains the mean for each dataset, the standard deviation between initializations, and the standard deviation, minimum, and maximum values across the three data sets. Only three datasets are used to claim population significance or generalization of the actual images.','',
        'We played back 4,480 new trajectories and 17,920 video clips, and confirmed that there is no overlap between the existing materials and the rotating orbits. We verified 126 new state models, 312 video connection comparison conditions, and 2,808 conditions for long-term future and counter-action. The basic transfer of the long-term future is 16,865,280, and the response transfer is 6,746,112, which are calculated repeatedly for the same scene and model, not independent scenes. We also confirmed 327,680 one-step transfer of the new physical reference using separate reference codes.','',
        'Upon examining the very small difference sign in the aggregation, a sign reversal was detected. The prediction and aggregation CSV were preserved, and a precise sum of positive integers was used for separate sign verification. The number of such extremely small negative numbers is not interpreted as a practical effect or significance. [Numerical Audit Record](across_data/aggregate_numeric_audit_note.json)','',
        '## One-step prediction that provides the correct answer status every moment','',
        'The unit is the average absolute speed error (px/frame), and the lower it is, the better. It is the C03 condition using all hold transitions.','',
        '| Condition / Model | 640031 | 997101 | 997102 |','| --- | ---: | ---: | ---: |']
    names={'test':'the public','count4':'Your object','force_ood':'Non-aesthetic external force'}
    model_names={'interaction':'General interaction','joint_exact':'State and external force joint rotation','force_wall':'Refer to known external forces and wall physics'}
    for split in names:
        for kind,label in model_names.items():lines.append(f'| {names[split]} / {label} | {cells(state(split,kind))} |')
    lines+=['',f'In general conditions, the error reduction rate of joint rotation varies by data.{", ".join(f"{x:.1f}%" for x in joint_improvement)}It was. Data with smaller error than common rotation that refer to known external forces and walls{physical_better}/3 pieces. This reference ignores collisions between objects by knowing the simulator integration and boundaries. It does not interpret it as learned physical laws or general model superiority.','',
        '## Read from the four videos and predict the next moment','',
        'The same common rotating prediction engine transmitted the estimated state and environment together. Since it is a single speed error (t=3→4), it is not compared directly with the overall transition average across the entire range due to direct performance improvement. Measurements of the circular object and the physical equations use known color, shape, and integration rules.','',
        '| Conditions / How to enter video | 640031 | 997101 | 997102 |','| --- | ---: | ---: | ---: |']
    for split in names:
        for method,label in [('rgb_cnn','Direct CNN'),('measurement_mlp','Video measurement + learning correction'),('rgb_analytic','Video measurement + physical')]:
            lines.append(f'| {names[split]} / {label} | {cells(observation(method,split))} |')
    lines+=['',
        'The results of attaching a measuring device for a circular object are not called performance because they learn concepts and physics on their own using only RGB. Whether the learning correction is better than physical methods must be judged by both conditions and the values per data set. The correct input state/environment change and the results of all comparables and all prediction models were preserved in observation_metrics.csv.','',
        '## A long future predicted by ourselves','',
        'This is the position mean absolute error (px) of the joint rotation predictor for video measurement + learning correction. Since the values are finite, the prediction mean for each initialization was first calculated. The large finite variance was not well suppressed, and the analogous failure was not converted to an 0 error. The finite sample sizes per initialization were left in autonomous_inputs.csv.','',
        '| Condition / Prediction length | 640031 | 997101 | 997102 |','| --- | ---: | ---: | ---: |']
    for split in names:
        for h in [1,16,61]:lines.append(f'| {names[split]} / {h}Step |{cells(future(split,h))} |')
    lines+=['',
        '| Conditions / 61st stage diagnosis | 640031 | 997101 | 997102 |','| --- | ---: | ---: | ---: |']
    for split in names:
        for metric,label in [('finite_fraction','Limited number of path ratios'),('any_wall_violation_over_1px_or_nonfinite','Wall breach 1px over or failure rate of the value'),('scene_success','Overall status success rate')]:
            lines.append(f'| {names[split]} / {label} | {cells(future(split,61,metric))} |')
    lines+=['',
        'Even if large values remain finite, they are not accurate predictions. Since the average is sensitive to some large deviations, we presented both the finite ratio and the overall state success/wall penetration indicators together. The initial answer information is also included in the entire CSV file, as are the values of the remaining eight prediction models. We do not infer long-term stability from a single step improvement.','',
        '## The reaction when you change your behavior','',
        'General conditions, video measurement + learning correction, joint rotation predictor waiting. The table shows the value obtained by subtracting the response location error from the “even if you change your behavior, the future remains unchanged” non‑response comparison error. If the value is negative, it is better than this comparison; if it is positive, it is worse. Accuracy of behavior transfer to other objects due to object collisions is considered separately.','',
        '| Target / Prediction length | 640031 | 997101 | 997102 |','| --- | ---: | ---: | ---: |']
    for h in [16,61]:
        for label,kor in [('target','Target of action'),('nontarget','Other object')]:
            a=response(h,label+'_position_response_mae_finite');b=response(h,label+'_position_zero_response_mae')
            lines.append(f'| {kor} / {h}Step |{cells([x-y if x is not None and y is not None else None for x,y in zip(a,b)])} |')
    lines+=['',
        'The response error is conditional in cases where both predictions are limited, and the no-response comparison is calculated for all response scenes. The simple mean difference under conditional conditions with analogous cases is not a paired comparison of the same sample, so superiority cannot be claimed solely based on this difference. The two original errors and the number of limited samples can be verified in the entire CSV. The accuracy of responses per object and per time is not extended beyond natural language understanding or general causal inference abilities.','',
        '## Remaining research stages','',
        'The effects and failures confirmed in the independent repetition are preserved, and a comparison of observation lengths at the common behavior time points of 1/2/4/8 is conducted, along with a re-learning process to recover and recover missing information on convergence and resistance. The cases where answers cannot be determined from the same observation and the cases where the model failed to learn are distinguished. External SVIB evaluations, comparisons with nearby prior research, real-person judgments, and final drawings, research notes, and replication bundles are also remaining.','',
        '[State prediction overall](across_data/state_metrics.csv) · [Video comparison overall](across_data/observation_metrics.csv) · [Long-term future/behavior change overall](across_data/autonomous_metrics.csv) · [State aggregation verification](across_data/state_verification.json) · [Video aggregation verification](across_data/observation_verification.json) · [Long-term future aggregation verification](across_data/autonomous_verification.json)']
    (root/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#3275a8','#d68a34','#5b986a'];plot_data=[]
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
    for ax,split,title in zip(axes,names,['Standard','4 objects','Unseen force']):
        for j,ds in enumerate(DATA):
            y=[state(split,k)[j] for k in model_names];ax.plot(range(3),y,'o-',color=colors[j],label=f'Data {ds}')
            plot_data.append({'figure':'state','data_seed':ds,'split':split,'values':y})
        ax.set_xticks(range(3),['Interaction','Joint rotation','Force + wall']);ax.tick_params(axis='x',labelsize=8)
        ax.set(title=title,ylabel='One-step velocity MAE (px/frame)',yscale='log');ax.grid(axis='y',alpha=.25)
    axes[0].legend(fontsize=8);fig.suptitle('Three independent data draws; each learned point averages 3 fixed initializations')
    for suffix in ['png','svg']:fig.savefig(root/f'independent_state_comparison.{suffix}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained');hs=[1,4,8,16,32,61]
    for col,(split,title) in enumerate(zip(names,['Standard','4 objects','Unseen force'])):
        for j,ds in enumerate(DATA):
            y=[future(split,h)[j] for h in hs];f=[future(split,h,'finite_fraction')[j] for h in hs]
            axes[0,col].plot(hs,y,'o-',color=colors[j],label=f'Data {ds}')
            axes[1,col].plot(hs,f,'o-',color=colors[j]);plot_data.append({'figure':'rollout','data_seed':ds,'split':split,'horizons':hs,'finite_conditional_mae':y,'finite_fraction':f})
        axes[0,col].set(title=title,yscale='log',ylabel='Position MAE, finite predictions (px)');axes[0,col].grid(alpha=.25)
        axes[1,col].set(xlabel='Forecast steps',ylabel='Finite path fraction',ylim=(-.03,1.03));axes[1,col].grid(alpha=.25)
    axes[0,0].legend(fontsize=8);fig.suptitle('Measured + learned input, joint rotation dynamics — finite does not mean accurate')
    for suffix in ['png','svg']:fig.savefig(root/f'independent_rollout_stability.{suffix}',dpi=160)
    plt.close(fig)
    dump(root/'report_manifest.json',{'report_sha256':r.sha(root/'RESULTS_KO.md'),'evidence_sha256':evidence,
        'figures':{p:r.sha(root/p) for p in ['independent_state_comparison.png','independent_state_comparison.svg','independent_rollout_stability.png','independent_rollout_stability.svg']},
        'plot_data':plot_data,'script_sha256':r.sha(__file__),'C07_complete':False,'full_goal_complete':False})
    print('three-data report and figures written',flush=True)


if __name__=='__main__':main()
