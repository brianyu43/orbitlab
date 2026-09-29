"""Full-study report with fixed selections and inspectable source-bound curve data."""
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,dump,r
from dynamics_observation_length_data import NAME

LENGTHS=[1,2,4,8];DATA=[640031,997101,997102];HORIZONS=[1,4,8,16,32,61]
STRATA=['all','no_past_contact','past_contact','past_pair_contact','past_wall_contact']
PAIRS=[[2,1],[4,1],[8,1],[4,2],[8,2],[8,4]]
METHODS=['rgb_cnn','measurement_mlp','rgb_analytic']
COLORS={'rgb_cnn':'#3971b0','measurement_mlp':'#d44735','rgb_analytic':'#219477','oracle':'#555555','force_wall':'#a77c27'}


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def fmt(value):
    if value is None or not np.isfinite(value):return 'Missing'
    return f'{value:.3e}' if abs(value)>=1000 or (0<abs(value)<.0001) else f'{value:.4f}'


class Curves:
    def __init__(self,root):self.root=root;self.blocks={};self.bound={}
    def block(self,stage,method,predictor,mode,split):
        path=self.root/'aggregates_v1'/stage/method
        if predictor:path/=predictor
        path=path/mode/split
        key=str(path)
        if key not in self.blocks:
            manifest=json.loads((path/'manifest.json').read_text());audit=json.loads((path/'verification.json').read_text())
            assert audit['all_passed'] and audit['block_manifest_sha256']==r.sha(path/'manifest.json')
            assert audit['verifier_sha256']==r.sha(HERE/'verify_length_aggregate.py')
            with np.load(path/'statistics.npz') as a:arrays={k:a[k] for k in a.files}
            assert manifest['files']['statistics.npz']==r.sha(path/'statistics.npz')
            self.blocks[key]=(path,manifest,arrays)
        return self.blocks[key]
    def get(self,stage,method,component,metric,predictor='',horizon=0,mode='variable_force',split='test',stratum='all',paired=False):
        block_predictor=predictor if stage=='autonomous' else ''
        path,m,a=self.block(stage,method,block_predictor,mode,split)
        scalar={'component':component,'predictor':predictor,'horizon':horizon,'metric':metric}
        index=m['scalar_keys'].index(scalar);s=STRATA.index(stratum);p='pair_' if paired else ''
        selector={'stage':stage,'method':method,'block_predictor':block_predictor,'mode':mode,'split':split,'scalar_key':scalar,'stratum':stratum,'paired':paired}
        identifier='|'.join(map(str,[stage,method,predictor,mode,split,component,metric,horizon,stratum,'paired' if paired else 'unit']))
        result={'id':identifier,'selector':selector,'axis':PAIRS if paired else LENGTHS,
            'mean':a[p+'across_data_mean'][:,s,index].tolist(),'sd':a[p+'across_data_sd'][:,s,index].tolist(),
            'minimum':a[p+'across_data_min'][:,s,index].tolist(),'maximum':a[p+'across_data_max'][:,s,index].tolist(),
            'data_count':a[p+'across_data_n'][:,s,index].tolist(),'data_means':a[p+'data_mean'][:,:,s,index].tolist(),
            'unit_means':a[p+'unit_mean'][:,:,:,s,index].tolist(),'finite_scene_counts':a[p+'unit_count'][:,:,:,s,index].tolist(),
            'total_scene_counts':a[p+'unit_total'][:,:,:,s].tolist(),'block_manifest':str((path/'manifest.json').relative_to(HERE)),
            'block_manifest_sha256':r.sha(path/'manifest.json'),'block_verification_sha256':r.sha(path/'verification.json')}
        if paired:
            result['common_left_unit_means']=a['pair_unit_left_mean'][:,:,:,s,index].tolist()
            result['common_right_unit_means']=a['pair_unit_right_mean'][:,:,:,s,index].tolist()
        result=clean(result)
        if identifier in self.bound:assert self.bound[identifier]==result
        self.bound[identifier]=result;return result


def main():
    root=HERE/f'reports/{NAME}';gates={}
    for stage,expected in [('observation',52),('autonomous',468)]:
        p=root/f'aggregates_v1/{stage}_verification.json';v=json.loads(p.read_text())
        assert v['all_passed'] and v['blocks_verified']==expected
        assert v['manifest_sha256']==r.sha(root/f'aggregates_v1/{stage}_manifest.json')
        assert v['verifier_sha256']==r.sha(HERE/'verify_length_aggregate.py')
        gates[str(p.relative_to(HERE))]=r.sha(p)
    out=root/'report_v1'
    if out.exists():raise RuntimeError(f'Preserve existing report: {out}')
    out.mkdir();curves=Curves(root);tables=[]
    def obs(method,metric,split='test',component='inference',predictor='',horizon=0,**kw):
        return curves.get('observation',method,component,metric,predictor,horizon,split=split,**kw)
    def auto(method,metric,horizon,predictor='joint_exact',split='test',component='factual',**kw):
        return curves.get('autonomous',method,component,metric,predictor,horizon,split=split,**kw)
    def plot_length(ax,curve,label,color,style='-',band=True):
        y=np.asarray(curve['mean'],float);ax.plot(LENGTHS,y,'o'+style,color=color,label=label,markersize=4)
        if band:ax.fill_between(LENGTHS,np.asarray(curve['minimum'],float),np.asarray(curve['maximum'],float),color=color,alpha=.10)
        ax.set_xscale('log',base=2);ax.set_xticks(LENGTHS,labels=LENGTHS);ax.grid(alpha=.2)
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    obs_panels=[('zero_filled_true_slot_position_mae','Position estimation (px)'),
        ('zero_filled_true_slot_velocity_mae','Velocity estimation (px/frame)'),
        ('net_force_mae_pixels_per_frame_squared','Net-force estimation (px/frame squared)'),
        ('drag_absolute_error','Drag estimation absolute error'),
        ('state_scene_success','Strict inferred-state success'),
        ('forecast','Next velocity, joint rotation predictor')]
    for ax,(metric,title) in zip(axes.flat,obs_panels):
        for method in METHODS:
            c=obs(method,'zero_filled_true_slot_velocity_mae',component='estimated_state_estimated_context',predictor='joint_exact',horizon=1) if metric=='forecast' else obs(method,metric)
            plot_length(ax,c,method,COLORS[method])
        if metric=='state_scene_success':ax.set_ylim(-.02,1.02)
        ax.set(title=title,xlabel='Observed RGB frames');ax.legend(fontsize=8)
    fig.suptitle('Same anchor, action, and future: variable-force test\nMean and range of 3 data units, after averaging 3 initializations; shaded range is not a confidence interval',fontsize=13)
    fig.savefig(out/'observation_length_effect.png',dpi=150);fig.savefig(out/'observation_length_effect.svg');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(16,12),constrained_layout=True)
    conditions=['test','force_ood','count4']
    for row,split in enumerate(conditions):
        for col,h in enumerate([16,61]):
            ax=axes[row,col]
            for method in ['oracle']+METHODS:
                c=auto(method,'position_mae_pixels_finite',h,split=split);plot_length(ax,c,method,COLORS[method],band=False)
                data=np.asarray(c['data_means'],float)
                for li,l in enumerate(LENGTHS):ax.scatter([l]*3,data[li],color=COLORS[method],s=8,alpha=.3)
            c=auto('measurement_mlp','position_mae_pixels_finite',h,predictor='force_wall',split=split)
            plot_length(ax,c,'measurement + known physics',COLORS['force_wall'],'--',band=False)
            ax.set_yscale('log');ax.set(title=f'{split}: horizon {h}',xlabel='Observed frames',ylabel='Finite position MAE (px, log scale)');ax.legend(fontsize=7)
        ax=axes[row,2]
        for method in ['oracle']+METHODS:
            c=auto(method,'finite_fraction',61,split=split)
            ax.plot(LENGTHS,1-np.asarray(c['mean'],float),'o-',color=COLORS[method],label=method)
        ax.set_xscale('log',base=2);ax.set_xticks(LENGTHS,labels=LENGTHS);ax.set_ylim(-.02,1.02);ax.grid(alpha=.2)
        ax.set(title=f'{split}: nonfinite fraction at 61',xlabel='Observed frames',ylabel='Failure fraction');ax.legend(fontsize=7)
    fig.suptitle('Long predictions: joint-rotation dynamics with four input estimators\nSmall dots are the 3 data means. Huge finite errors remain in the mean; nonfinite failures are shown separately.',fontsize=13)
    fig.savefig(out/'length_rollout_stability.png',dpi=150);fig.savefig(out/'length_rollout_stability.svg');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(16,12),constrained_layout=True)
    length_colors=['#2878b5','#5d9c64','#ca8739','#cc4657']
    for row,split in enumerate(conditions):
        all_h=[auto('measurement_mlp','position_mae_pixels_finite',h,split=split) for h in HORIZONS]
        baseline=[auto('measurement_mlp','position_mae_pixels_finite',h,predictor='force_wall',split=split) for h in HORIZONS]
        for di,ds in enumerate(DATA):
            ax=axes[row,di]
            for li,l in enumerate(LENGTHS):
                y=[c['data_means'][li][di] for c in all_h];ax.plot(HORIZONS,y,'o-',label=f'{l} frames',color=length_colors[li],markersize=3)
            ax.plot(HORIZONS,[c['data_means'][3][di] for c in baseline],'k--',label='8 frames, known physics')
            ax.set_yscale('log');ax.set_xticks(HORIZONS);ax.grid(alpha=.2)
            ax.set(title=f'{split}, data {ds}',xlabel='Future steps',ylabel='Finite position MAE (px)');ax.legend(fontsize=7)
    fig.suptitle('Every horizon and data unit: measurement + MLP, joint-rotation dynamics\nAll length settings use the same final observed time t=7; data units reuse original initial draws.',fontsize=13)
    fig.savefig(out/'length_horizon_by_data.png',dpi=150);fig.savefig(out/'length_horizon_by_data.svg');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    paired_specs=[('inference_velocity','Velocity estimation'),('inference_force','Net-force estimation'),('forecast','Next-step velocity'),('rollout','Horizon-16 position')]
    for ax,(key,title) in zip(axes.flat,paired_specs):
        ax.axhline(0,color='black',lw=1)
        for pi,color,offset in [(2,'#2463a6',-.1),(5,'#c44737',.1)]:
            values=[]
            for stratum in STRATA:
                if key=='inference_velocity':c=obs('measurement_mlp','zero_filled_true_slot_velocity_mae',stratum=stratum,paired=True)
                elif key=='inference_force':c=obs('measurement_mlp','net_force_mae_pixels_per_frame_squared',stratum=stratum,paired=True)
                elif key=='forecast':c=obs('measurement_mlp','zero_filled_true_slot_velocity_mae',component='estimated_state_estimated_context',predictor='joint_exact',horizon=1,stratum=stratum,paired=True)
                else:c=auto('measurement_mlp','position_mae_pixels_finite',16,stratum=stratum,paired=True)
                values.append(c['data_means'][pi])
            values=np.array(values,float);x=np.arange(5)+offset
            for j in range(3):ax.scatter(x,values[:,j],s=15,alpha=.5,color=color)
            valid=np.isfinite(values);n=valid.sum(1);avg=np.divide(np.where(valid,values,0).sum(1),n,out=np.full(5,np.nan),where=n>0)
            ax.plot(x,avg,'o-',color=color,label='8 minus '+str(PAIRS[pi][1]))
        ax.set_xticks(range(5),labels=['all','no contact','any contact','pair contact','wall contact'],rotation=15)
        ax.set(title=title,ylabel='Long minus short error');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Same-scene length differences: measurement + MLP, variable-force test\nFixed 8-frame contact groups; common-finite metric values only. Dots: data units; line: equal-weight mean.',fontsize=13)
    fig.savefig(out/'length_paired_contact.png',dpi=150);fig.savefig(out/'length_paired_contact.svg');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    for row,h in enumerate([16,61]):
        for col,kind in enumerate(['target','nontarget']):
            ax=axes[row,col]
            for method in ['oracle','measurement_mlp','rgb_analytic']:
                c=auto(method,f'{kind}_position_response_mae_finite',h,component='response');plot_length(ax,c,method,COLORS[method],band=False)
            ref=auto('oracle',f'{kind}_position_zero_response_mae',h,component='response');plot_length(ax,ref,'zero response','black','--',band=False)
            ax.set_yscale('log');ax.set(title=f'{kind} response, horizon {h}',xlabel='Observed frames',ylabel='Position response error (px)');ax.legend(fontsize=8)
        ax=axes[row,2]
        for method in ['oracle','measurement_mlp','rgb_analytic']:
            c=auto(method,'both_branches_finite',h,component='response');ax.plot(LENGTHS,1-np.array(c['mean'],float),'o-',label=method,color=COLORS[method])
        ax.set_xscale('log',base=2);ax.set_xticks(LENGTHS,labels=LENGTHS);ax.set_ylim(-.02,1.02)
        ax.set(title=f'Either-branch failure, horizon {h}',xlabel='Observed frames',ylabel='Nonfinite fraction');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Opposite-action response: variable-force test\nFinite response errors and full-cohort failure rates. Zero-response baseline uses all scenes; compare denominators before interpreting gaps.',fontsize=13)
    fig.savefig(out/'length_action_response.png',dpi=150);fig.savefig(out/'length_action_response.svg');plt.close(fig)
    lines=['# How long should we look at the same past?','',
        'Observations 1, 2, 4, and 8 were compared at the same final time point t=7. We completed the 72 estimated models and the replication verification of all one-step, long-term, and opposite behavior evaluations and aggregations. The table below shows the results of averaging the initial three sets of data from the three data sets first, then applying the same weights to each data set. The initial nine sets of data are not counted as nine independent data sets.','',
        'The initial scenes of the three existing datasets were reused, and the action time was shifted to recalculate the future. No new independent dataset bundles were added, and no simple forward-and-backward performance comparison was performed with the existing t=3 results. All paths shared the same state, action, and future, and were checked to ensure that the obscured past video and measurement values did not influence the input.','',
        'RGB CNN directly reads the video. The measurement + MLP and video physics models utilize the known color, rendering, and motion rules of a circular object, so they are not pure video learning comparisons under the same prior knowledge conditions. The estimator\'s capacity, initial weights, and learning sample order were aligned by length, and the final 4,000-step checkpoint was fixed for evaluation.','',
        '## Status and environment estimation and one-step connection','',
        'The error is the missing estimate filled in as 0 based on the correct object position and velocity on the table. It is not the same as the error for the object discovered in reality. All success indicators for detection and strict scene verification have been preserved together with the original dataset. This is a general variable-force test, and the units are position pixels, velocity pixels/frame, and sum-of-squares pixels/frame².','',
        '| Input method | Video count | Location estimation | Speed estimation | Combined estimation | Next speed |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    table_specs=[('position','zero_filled_true_slot_position_mae'),('velocity','zero_filled_true_slot_velocity_mae'),('force','net_force_mae_pixels_per_frame_squared')]
    for method in METHODS:
        values=[obs(method,metric) for _,metric in table_specs]+[obs(method,'zero_filled_true_slot_velocity_mae',component='estimated_state_estimated_context',predictor='joint_exact',horizon=1)]
        for li,l in enumerate(LENGTHS):
            text='| '+method+' | '+str(l)+' | '+' | '.join(fmt(c['mean'][li]) for c in values)+' |';lines.append(text)
            tables.append({'line':text,'cells':[{'curve_id':c['id'],'field':'mean','indices':[li]} for c in values]})
    lines+=['','## Results by data for the long future','',
        'This is the 61st-stage position error of the joint rotation predictor with measurement+MLP input. To avoid hiding even if one data point has a very large deviation, the average of the three data sets was displayed separately. It is a finite prediction error, and the relative failure rate is listed in a separate column. The success status, number of valid scenes, and all initialization values were preserved in the connected data.','',
        '| Conditions | Video count | Data 640031 | Data 997101 | Data 997102 | Hypothetical failure rate |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for split in conditions:
        c=auto('measurement_mlp','position_mae_pixels_finite',61,split=split);failure=auto('measurement_mlp','finite_fraction',61,split=split)
        for li,l in enumerate(LENGTHS):
            text='| '+split+' | '+str(l)+' | '+' | '.join(fmt(v) for v in c['data_means'][li])+' | '+fmt(1-failure['mean'][li])+' |';lines.append(text)
            tables.append({'line':text,'cells':[{'curve_id':c['id'],'field':'data_means','indices':[li,di]} for di in range(3)]+[{'curve_id':failure['id'],'field':'mean','indices':[li],'transform':'one_minus'}]})
    lines+=['','## Difference in length of the same scene','',
        'Below are the average values of the scene-by-scene differences between L8-L1 and L8-L4, averaged across the data. Negative values indicate that the long observation error is smaller. Only scenes where both values are finite are considered for the difference calculation, and the denominator is the minimum/maximum number of scenes for each data set and initialization. We did not simply subtract the average valid scenes from each other.','',
        '| Comparison indicators | Length difference | Data 640031 | Data 997101 | Data 997102 | Range of common scenes |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    pair_curves=[('Speed estimation',obs('measurement_mlp','zero_filled_true_slot_velocity_mae',paired=True)),
                 ('Estimated combined years of experience',obs('measurement_mlp','net_force_mae_pixels_per_frame_squared',paired=True)),
                 ('Next speed',obs('measurement_mlp','zero_filled_true_slot_velocity_mae',component='estimated_state_estimated_context',predictor='joint_exact',horizon=1,paired=True)),
                 ('16-step location',auto('measurement_mlp','position_mae_pixels_finite',16,paired=True)),
                 ('61st stage location',auto('measurement_mlp','position_mae_pixels_finite',61,paired=True))]
    for label,c in pair_curves:
        for pi in [2,5]:
            counts=np.array(c['finite_scene_counts'][pi]);text='| '+label+' | '+str(PAIRS[pi][0])+'−'+str(PAIRS[pi][1])+' | '+' | '.join(fmt(v) for v in c['data_means'][pi])+' | '+f'{counts.min()}–{counts.max()}'+' |';lines.append(text)
            tables.append({'line':text,'cells':[{'curve_id':c['id'],'field':'data_means','indices':[pi,di]} for di in range(3)],'count_curve_id':c['id'],'count_axis_index':pi})
    lines+=['','The scope of contact comparisons was fixed at the past 7 trials shown in L8. The correct contact information was used only for dividing the evaluation group and was not provided for the estimator input. Since the actual past contact scope shown at each length may vary, we do not call it the length effect by directly subtracting the individual average.','',
        '## Criteria for interpreting opposite behavior and long predictions','',
        'The difference between the future with opposite behavior from the same initial past was divided into the target and non-target. The response error only exists on the limited scene of both pathways. The no-response comparison uses the entire scene, so if the effective denominator is different, the average difference is not interpreted as the performance improvement amount directly. All indicators, effective denominators, and failure rates for both pathways were stored.','',
        'Oracle inputs the correct initial state and environment, but does not receive the correct state again afterward. If the prediction continues for too long, it may fail. The known force-wall reference uses simulator rules and external forces, but omits object collisions. The difference with this reference is not proof of a new general physics learning ability.','',
        '## Total output and verification range','',
        'All indicators, horizon, and contact groups for observation 52 blocks and autonomous prediction 468 blocks were preserved in NPZ/JSON format. They include individual values for each length, common scene differences between six length pairs, and mean, effective sample size, total denominator, and range for data and initialization. The accuracy status + accuracy environment comparison and oracle autonomous prediction were confirmed to be identical regardless of observation length. The connected data for tables and figures is `report_v1/curves.json`, and the entire block list is `report_v1/aggregate_index.csv`.','',
        'The current document is responsible for the complete display of results and numbers. The comprehensive conclusion of information deficiency, perception errors, and transfer learning failures is connected to independent repetition and environmental information omission results in the separate C07 comprehensive document.','',
        '![Observation length and initial estimate](report_v1/observation_length_effect.png)','',
        '![Long prediction and failure rate](report_v1/length_rollout_stability.png)','',
        '![All prediction lengths by data](report_v1/length_horizon_by_data.png)','',
        '![Difference in contact for the same scene](report_v1/length_paired_contact.png)','',
        '![Action change response](report_v1/length_action_response.png)','',
        '[Experiment plan](../../planning_C07_OBSERVATION_LENGTH_KO.md) · [Aggregation rules](../../planning_C07_LENGTH_AGGREGATION_KO.md) · [Display rules](../../planning_C07_LENGTH_REPORT_KO.md) · [Observation aggregation verification](aggregates_v1/observation_verification.json) · [Autonomous prediction aggregation verification](aggregates_v1/autonomous_verification.json)']
    report=root/'RESULTS_KO.md';report.write_text('\n'.join(lines)+'\n')
    dump(out/'curves.json',{'data_seeds':DATA,'lengths':LENGTHS,'pairs':PAIRS,'curves':list(curves.bound.values())})
    index=[]
    for stage in ['observation','autonomous']:
        for item in json.loads((root/f'aggregates_v1/{stage}_manifest.json').read_text())['blocks']:
            index.append({**item['meta'],'manifest_path':item['path'],'manifest_sha256':item['sha256'],'unit_cells':item['unit_scalar_cells'],'paired_cells':item['paired_unit_scalar_cells']})
    r.write_csv(out/'aggregate_index.csv',index);assert len(index)==520
    dump(out/'manifest.json',{'source_sha256':r.sha(__file__),'protocol_sha256':r.sha(HERE/'planning_C07_LENGTH_REPORT_KO.md'),
        'verification_gates':gates,'report_sha256':r.sha(report),'curves_sha256':r.sha(out/'curves.json'),
        'table_bindings':tables,'bound_curve_count':len(curves.bound),'all_block_count':len(index),
        'files':{p.name:r.sha(p) for p in out.iterdir() if p.is_file()},
        'scope':'Fixed diagnostic views of all audited length aggregates. Paired values use common-finite scenes and fixed L8 contact cohorts; ranges are not confidence intervals.'})
    print('length report',len(curves.bound),'curves,',len(tables),'table rows,',len(index),'blocks',flush=True)


if __name__=='__main__':main()
