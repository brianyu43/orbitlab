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
    if value is None or not np.isfinite(value):return '결측'
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
    lines=['# 같은 과거를 얼마나 오래 보아야 하는가','',
        '관측 1·2·4·8장을 동일 마지막 시점 t=7에서 비교했다. 72개 추정 모델과 모든 한 단계·긴 미래·반대 행동 평가 및 집계의 재현 검증을 마쳤다. 아래 표는 데이터 세 묶음에서 초기화 세 개를 먼저 평균한 뒤 데이터별로 같은 가중치를 준 결과다. 초기화 아홉 개를 독립 데이터 아홉 개로 세지 않는다.','',
        '기존 세 데이터의 초기 장면을 재사용하고 행동 시점을 옮겨 미래를 다시 계산했다. 새로운 독립 데이터 세 묶음이 추가된 것이 아니며 기존 t=3 결과와 단순한 전후 성능 비교를 하지 않는다. 모든 길이는 동일 상태·행동·미래를 공유하고, 가린 과거 영상·측정값이 입력에 영향을 주지 않는지 검사했다.','',
        'RGB CNN은 영상을 직접 읽는다. 측정+MLP와 영상 물리식은 원형 물체의 알려진 색·렌더링 및 운동 규칙을 활용하므로 같은 사전 지식 조건의 순수 영상 학습 비교가 아니다. 추정기 용량·초기 가중치·학습 표본 순서를 길이별로 맞췄고, 마지막 4,000-step checkpoint를 고정 평가했다.','',
        '## 상태·환경 추정과 한 단계 연결','',
        '표의 위치·속도는 정답 물체 자리를 기준으로 누락된 추정을 0으로 채운 오차다. 실제로 발견한 물체만의 오차와 같지 않다. 모든 검출·엄격 장면 성공 지표는 원본 집계에 함께 보존했다. 일반 variable-force test이며 단위는 위치 픽셀, 속도 픽셀/프레임, 합력 픽셀/프레임²이다.','',
        '| 입력 방식 | 영상 수 | 위치 추정 | 속도 추정 | 합력 추정 | 다음 속도 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    table_specs=[('position','zero_filled_true_slot_position_mae'),('velocity','zero_filled_true_slot_velocity_mae'),('force','net_force_mae_pixels_per_frame_squared')]
    for method in METHODS:
        values=[obs(method,metric) for _,metric in table_specs]+[obs(method,'zero_filled_true_slot_velocity_mae',component='estimated_state_estimated_context',predictor='joint_exact',horizon=1)]
        for li,l in enumerate(LENGTHS):
            text='| '+method+' | '+str(l)+' | '+' | '.join(fmt(c['mean'][li]) for c in values)+' |';lines.append(text)
            tables.append({'line':text,'cells':[{'curve_id':c['id'],'field':'mean','indices':[li]} for c in values]})
    lines+=['','## 긴 미래의 데이터별 결과','',
        '측정+MLP 입력과 공동 회전 예측기의 61단계 위치 오차다. 한 데이터가 매우 크게 발산해도 숨기지 않기 위해 세 데이터 평균을 따로 표시했다. 유한한 예측의 오차이며 비유한 실패율은 별도 열에 있다. 성공 여부·유효 장면 수·모든 초기화 값은 연결 자료에 보존했다.','',
        '| 조건 | 영상 수 | 데이터 640031 | 데이터 997101 | 데이터 997102 | 비유한 실패율 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for split in conditions:
        c=auto('measurement_mlp','position_mae_pixels_finite',61,split=split);failure=auto('measurement_mlp','finite_fraction',61,split=split)
        for li,l in enumerate(LENGTHS):
            text='| '+split+' | '+str(l)+' | '+' | '.join(fmt(v) for v in c['data_means'][li])+' | '+fmt(1-failure['mean'][li])+' |';lines.append(text)
            tables.append({'line':text,'cells':[{'curve_id':c['id'],'field':'data_means','indices':[li,di]} for di in range(3)]+[{'curve_id':failure['id'],'field':'mean','indices':[li],'transform':'one_minus'}]})
    lines+=['','## 같은 장면의 길이 차이','',
        '아래는 L8-L1과 L8-L4의 장면별 차이를 데이터별로 평균한 값이다. 음수는 긴 관측의 오차가 더 작다는 뜻이다. 양쪽 값이 모두 유한한 장면만 차이를 계산하며, 분모는 데이터·초기화별 최소/최대 장면 수다. 서로 다른 유효 장면 평균을 그냥 빼지 않았다.','',
        '| 비교 지표 | 길이 차이 | 데이터 640031 | 데이터 997101 | 데이터 997102 | 공통 장면 수 범위 |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    pair_curves=[('속도 추정',obs('measurement_mlp','zero_filled_true_slot_velocity_mae',paired=True)),
                 ('합력 추정',obs('measurement_mlp','net_force_mae_pixels_per_frame_squared',paired=True)),
                 ('다음 속도',obs('measurement_mlp','zero_filled_true_slot_velocity_mae',component='estimated_state_estimated_context',predictor='joint_exact',horizon=1,paired=True)),
                 ('16단계 위치',auto('measurement_mlp','position_mae_pixels_finite',16,paired=True)),
                 ('61단계 위치',auto('measurement_mlp','position_mae_pixels_finite',61,paired=True))]
    for label,c in pair_curves:
        for pi in [2,5]:
            counts=np.array(c['finite_scene_counts'][pi]);text='| '+label+' | '+str(PAIRS[pi][0])+'−'+str(PAIRS[pi][1])+' | '+' | '.join(fmt(v) for v in c['data_means'][pi])+' | '+f'{counts.min()}–{counts.max()}'+' |';lines.append(text)
            tables.append({'line':text,'cells':[{'curve_id':c['id'],'field':'data_means','indices':[pi,di]} for di in range(3)],'count_curve_id':c['id'],'count_axis_index':pi})
    lines+=['','접촉별 비교의 범주는 L8에서 보이는 과거 7전이로 고정했다. 정답 접촉 정보는 평가 집단을 나누는 데만 사용하며 추정기 입력에 제공하지 않았다. 각 길이에서 실제 보이는 과거 접촉 범주는 달라질 수 있으므로 그 개별 평균을 직접 빼서 길이 효과라고 부르지 않는다.','',
        '## 반대 행동과 긴 예측을 해석하는 기준','',
        '같은 초기 과거에서 반대 행동을 준 미래와의 차이를 대상·비대상으로 나눴다. 반응 오차는 두 경로가 모두 유한한 장면에만 존재한다. 무반응 대조는 전체 장면을 사용하므로 유효 분모가 다른 경우 평균 차이를 그대로 성능 개선량으로 해석하지 않는다. 모든 지표·유효 분모와 양쪽 경로 실패율을 저장했다.','',
        'oracle는 정답 초기 상태와 환경을 입력하지만 이후 정답 상태를 다시 공급받지 않는다. 그 예측도 길게 이어가면 실패할 수 있다. 알려진 force-wall 참조는 simulator 규칙과 외력을 이용하지만 물체 간 충돌은 생략한다. 이 참조와의 차이가 새로운 일반 물리 학습 능력의 증명은 아니다.','',
        '## 전체 산출물과 검증 범위','',
        '관측 52블록과 자율 예측 468블록의 모든 지표, horizon, 접촉 집단을 NPZ/JSON으로 보존했다. 각 길이의 개별 값과 여섯 길이 쌍의 공통 장면 차이, 데이터·초기화별 평균·유효 수·전체 분모·범위를 포함한다. 정답 상태+정답 환경 대조와 oracle 자율 예측이 관측 길이와 무관하게 동일한지도 확인했다. 표·그림의 연결 자료는 `report_v1/curves.json`이며 전체 블록 목록은 `report_v1/aggregate_index.csv`다.','',
        '현재 문서는 결과와 수치의 완전한 표시를 담당한다. 정보 부족·인식 오류·전이 학습 실패의 종합 결론은 별도 C07 종합 문서에서 독립 반복과 환경 정보 누락 결과에 연결한다.','',
        '![관측 길이와 초기 추정](report_v1/observation_length_effect.png)','',
        '![긴 예측과 실패율](report_v1/length_rollout_stability.png)','',
        '![데이터별 모든 예측 길이](report_v1/length_horizon_by_data.png)','',
        '![같은 장면 접촉별 차이](report_v1/length_paired_contact.png)','',
        '![행동 변경 반응](report_v1/length_action_response.png)','',
        '[실험 계획](../../planning_C07_OBSERVATION_LENGTH_KO.md) · [집계 규칙](../../planning_C07_LENGTH_AGGREGATION_KO.md) · [표시 규칙](../../planning_C07_LENGTH_REPORT_KO.md) · [관측 집계 검증](aggregates_v1/observation_verification.json) · [자율 예측 집계 검증](aggregates_v1/autonomous_verification.json)']
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
