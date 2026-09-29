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
        if v is None:return '해당 없음'
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
    lines=['# 독립 데이터 세 묶음에서 다시 확인한 운동 예측','',
        f'상태와 외력을 함께 회전시키는 설계는 일반 조건의 한 단계 속도 오차를 일반 상호작용 모델보다 세 데이터 모두 낮췄다. 감소 폭은 {min(joint_improvement):.1f}~{max(joint_improvement):.1f}%였다. 알려진 힘과 벽을 직접 계산하는 물리 참조의 오차는 세 데이터 모두 더 작았다.','',
        f'영상 측정의 학습 보정이 비학습 물리식보다 나았던 데이터는 일반 조건 {correction_wins["test"]}/3개, 미학습 외력 {correction_wins["force_ood"]}/3개였다. 한 데이터에서 보인 우열을 일반적인 결론으로 확대할 수 없었다.','',
        f'긴 미래는 안정적이지 않았다. 영상 측정+학습 보정과 공동 회전 예측기의 일반 조건 61단계 위치 오차는 데이터별 {", ".join(fmt(v) for v in future("test",61))}픽셀이었다. 수치가 유한하게 남아 있는 큰 발산까지 포함한 값이다. 따라서 한 단계의 이득이 장기 세계 모델의 완성도로 이어졌다고 주장하지 않는다.','',
        '기존 데이터 640031과 새 데이터 997101·997102를 비교했다. 새 자료에서 126개 상태 예측기와 12개 영상 추정기를 처음부터 학습했다. 기존 초기화와 표본 추출 순서, 학습 횟수, 모델 목록, 평가 규칙을 유지했다. 이 보고서는 C07의 독립 반복 부분이며 관측 길이·환경 정보 누락 실험은 별도다.','',
        '## 비교 단위와 검증','',
        '각 표의 열 하나가 독립 데이터 생성 seed 하나다. 각 셀은 고정한 세 학습 초기화의 평균이며, 초기화 9회를 독립 데이터 9개라고 세지 않았다. 비학습 참조는 데이터마다 한 번만 계산한다. 전체 CSV에는 데이터별 평균, 초기화 간 표준편차, 데이터 세 묶음 사이의 표준편차·최솟값·최댓값을 담았다. 세 데이터만으로 모집단 유의성이나 실제 영상의 일반화를 주장하지 않는다.','',
        '새 궤적 4,480개와 영상 17,920장을 재생했고 기존 자료와 회전 궤도 중복이 없음을 확인했다. 새 상태 모델 126개, 영상 연결 대조 312개 조건, 긴 미래·반대 행동 2,808개 조건을 검증했다. 긴 미래의 기본 전이 16,865,280개와 대응 전이 6,746,112개는 같은 장면·모델을 반복 계산한 수이며 독립 장면 수가 아니다. 새 물리 참조의 한 단계 전이 327,680개도 별도 참조 코드로 확인했다.','',
        '집계의 매우 작은 차이 부호를 검사하다 부동소수점 상쇄가 발견돼, 예측과 집계 CSV는 보존하고 부호의 별도 검증에 정확한 유리수 합을 사용했다. 이런 극소 음수의 개수를 실질적 효과나 유의성으로 해석하지 않는다. [수치 감사 기록](across_data/aggregate_numeric_audit_note.json)','',
        '## 정답 상태를 매 순간 제공한 한 단계 예측','',
        '단위는 속도 평균 절대 오차(px/frame)이며 낮을수록 좋다. 모든 보류 전이를 사용한 C03 조건이다.','',
        '| 조건 / 모델 | 640031 | 997101 | 997102 |','| --- | ---: | ---: | ---: |']
    names={'test':'일반','count4':'네 물체','force_ood':'미학습 외력'}
    model_names={'interaction':'일반 상호작용','joint_exact':'상태·외력 공동 회전','force_wall':'알려진 외력·벽 물리 참조'}
    for split in names:
        for kind,label in model_names.items():lines.append(f'| {names[split]} / {label} | {cells(state(split,kind))} |')
    lines+=['',f'일반 조건에서 공동 회전의 오차 감소율은 데이터별로 {", ".join(f"{x:.1f}%" for x in joint_improvement)}였다. 알려진 외력·벽 참조가 공동 회전보다 오차가 작은 데이터는 {physical_better}/3개였다. 이 참조는 simulator 적분과 경계를 알고 물체끼리의 충돌은 생략한다. 학습한 물리 법칙이나 일반적인 모델 우월성으로 해석하지 않는다.','',
        '## 네 장의 영상에서 읽고 다음 순간 예측','',
        '같은 공동 회전 예측기에 추정 상태·환경을 함께 전달했다. t=3→4 한 번의 속도 오차이므로 위 전체 전이 평균과 직접 성능 향상으로 비교하지 않는다. 원형 물체의 측정과 물리식은 알려진 색·모양·적분 규칙을 이용한다.','',
        '| 조건 / 영상 입력 방법 | 640031 | 997101 | 997102 |','| --- | ---: | ---: | ---: |']
    for split in names:
        for method,label in [('rgb_cnn','직접 CNN'),('measurement_mlp','영상 측정+학습 보정'),('rgb_analytic','영상 측정+물리식')]:
            lines.append(f'| {names[split]} / {label} | {cells(observation(method,split))} |')
    lines+=['',
        '원형 물체용 측정기를 붙인 결과를 RGB만으로 개념·물리를 스스로 학습한 성능으로 부르지 않는다. 학습 보정이 물리식보다 나은지도 조건·데이터별 수치를 함께 보고 판단해야 한다. 입력의 정답 상태/환경 교체 네 대조와 모든 예측기의 결과는 observation_metrics.csv에 보존했다.','',
        '## 스스로 이어 예측하는 긴 미래','',
        '영상 측정+학습 보정과 공동 회전 예측기의 위치 평균 절대 오차(px)다. 수치가 유한한 각 초기화의 예측 평균을 먼저 구했다. 큰 유한 발산을 잘라내지 않았고, 비유한 실패를 0 오차로 바꾸지 않았다. 초기화별 유한 표본 수는 autonomous_inputs.csv에 남겼다.','',
        '| 조건 / 예측 길이 | 640031 | 997101 | 997102 |','| --- | ---: | ---: | ---: |']
    for split in names:
        for h in [1,16,61]:lines.append(f'| {names[split]} / {h}단계 | {cells(future(split,h))} |')
    lines+=['',
        '| 조건 / 61단계 진단 | 640031 | 997101 | 997102 |','| --- | ---: | ---: | ---: |']
    for split in names:
        for metric,label in [('finite_fraction','수치가 유한한 경로 비율'),('any_wall_violation_over_1px_or_nonfinite','벽 침범 1px 초과 또는 수치 실패 비율'),('scene_success','전체 상태 성공 비율')]:
            lines.append(f'| {names[split]} / {label} | {cells(future(split,61,metric))} |')
    lines+=['',
        '큰 수치가 유한하게 남아 있어도 정확한 예측은 아니다. 평균이 일부 큰 발산에 민감하므로 유한 비율과 전체 상태 성공·벽 침범 지표를 함께 제시했다. 정답 초기 정보를 준 경우와 나머지 여덟 예측기의 수치도 전체 CSV에 있다. 한 단계 개선에서 긴 미래의 안정성을 추론하지 않는다.','',
        '## 행동을 반대로 바꿨을 때의 반응','',
        '일반 조건, 영상 측정+학습 보정, 공동 회전 예측기다. 표는 반응 위치 오차에서 “행동을 바꿔도 미래가 그대로”라는 무반응 대조의 오차를 뺀 값이다. 음수면 이 대조보다 낫고 양수면 나쁘다. 물체 충돌로 행동이 다른 물체에 전달되는 정확성을 별도로 본다.','',
        '| 대상 / 예측 길이 | 640031 | 997101 | 997102 |','| --- | ---: | ---: | ---: |']
    for h in [16,61]:
        for label,kor in [('target','행동 대상'),('nontarget','다른 물체')]:
            a=response(h,label+'_position_response_mae_finite');b=response(h,label+'_position_zero_response_mae')
            lines.append(f'| {kor} / {h}단계 | {cells([x-y if x is not None and y is not None else None for x,y in zip(a,b)])} |')
    lines+=['',
        '반응 오차는 두 예측이 모두 유한한 사례에 조건부이고 무반응 대조는 모든 대응 장면에서 계산된다. 비유한 사례가 있는 조건의 단순 평균 차이는 동일 표본의 쌍 비교가 아니므로 그 차이만으로 우월성을 주장하지 않는다. 두 가지 원래 오차와 유한 표본 수는 전체 CSV에서 확인할 수 있다. 물체별·시간별 반응의 정확성을 자연어 의도 이해나 일반적인 인과 추론 능력으로 확대하지 않는다.','',
        '## 남은 연구 단계','',
        '독립 반복에서 확인한 효과와 실패를 보존하고, 공통 행동 시점의 관측 길이 1/2/4/8 비교와 합력·저항 정보 누락 재학습을 진행한다. 같은 관측으로 답을 결정할 수 없는 경우와 모델이 학습하지 못한 경우를 구분한다. SVIB 외부 평가, 가까운 선행연구 비교, 실제 사람 판정, 최종 그림·연구 메모·재현 묶음도 남아 있다.','',
        '[상태 예측 전체](across_data/state_metrics.csv) · [영상 대조 전체](across_data/observation_metrics.csv) · [긴 미래·행동 변경 전체](across_data/autonomous_metrics.csv) · [상태 집계 검증](across_data/state_verification.json) · [영상 집계 검증](across_data/observation_verification.json) · [긴 미래 집계 검증](across_data/autonomous_verification.json)']
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
