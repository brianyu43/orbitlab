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
    def fmt(x):return '집계 불가' if x is None else (f'{x:.3f}' if abs(x)<1e4 else f'{x:.2e}')
    labels={'oracle':'정답 초기 상태·환경','rgb_cnn':'RGB 직접 CNN','measurement_mlp':'영상 측정+학습 보정','rgb_analytic':'영상 측정+물리식'}
    failures=sum(x['nonfinite_episodes'] for x in raw['results'])
    lines=['# 정답을 다시 주지 않는 긴 미래와 행동 변경','',
        '2026-09-23. C05/C06의 고정 모델 평가와 재생을 완료했다. 한 순간의 예측 정확도가 긴 미래의 안정성이나 다른 물체에 미치는 행동 효과의 정확성을 보장하지 않았다. 안정적인 세계 모델을 달성했다는 뜻은 아니다.','',
        '## 실행한 범위','',
        '초기 네 장(t=0..3)에서 얻은 상태·환경을 한 번만 제공하고 61단계(t=64)까지 스스로 예측했다. 중간에 정답 위치·속도·개수·환경을 다시 공급하지 않았다. 행동은 첫 전이에만 적용하고 나머지 행동은 0이다. 정답 초기 상태·환경은 명시적인 진단 대조군이다. CNN, 영상 측정+학습 보정, 영상 측정+물리식은 C04의 고정 추정값을 사용했다.','',
        '기존 일곱 학습 방식과 세 초기화, 등속도 및 외력·벽 반사 참조를 모두 유지했다. 모든 환경에 같은 variable_force 가중치를 사용했다. 학습기는 한 단계 손실로 학습했고 학습 궤적은 t=19에서 끝난다. 이번 32/61단계는 학습 자료의 시간 범위를 넘지만, 데이터 생성 seed를 독립적으로 바꾼 확인 실험은 아니다.','',
        '원본 보류 궤적 1,280개, 그중 동일 과거·반대 행동 대응 512개를 사용했다. 4가지 입력×3초기화×13그룹×9예측기의 1,404개 조건이다. 같은 자료를 반복한 예측 건수를 독립 표본 수로 세지 않는다. 원을 측정하는 방법은 알려진 색·모양·적분 규칙을 이용하고, 물리 참조는 물체 사이 충돌을 생략한다.','',
        '## 시간이 길어지면 얼마나 틀리는가','',
        '변하는 외력의 일반 test 128개 궤적에서 공동 회전(joint_exact) 모델의 위치 평균 절대 오차다. 단위는 픽셀이고 낮을수록 좋다. 각 seed의 유한한 예측 평균을 구한 뒤 세 값을 평균했다. 누락된 정답 물체는 0 좌표로 채워 오차에 포함하며, 존재 판정과 검출된 물체만의 오차도 별도 원자료에 있다.','',
        '| 초기 입력 | 1단계 | 16단계 | 61단계 | 61단계 유한한 비율 |','| --- | ---: | ---: | ---: | ---: |']
    for m in METHODS:
        lines.append('| '+labels[m]+' | '+' | '.join(fmt(metric(m,'joint_exact',h)) for h in [1,16,61])+f" | {metric(m,'joint_exact',61,'finite_fraction')*100:.1f}% |")
    lines+=['',
        '정답 정보를 제공해도 오차가 누적되므로 문제를 지각 모델만으로 돌릴 수 없다. 영상 직접 CNN은 시작점부터 오차가 크다. 반대로 수치가 유한하다는 것은 예측이 정확하거나 물리적으로 타당하다는 의미가 아니다. 영상 측정+물리식의 특정 초기화에서는 매우 큰 유한 오차도 관측됐다.','',
        '다음은 영상 측정+학습 보정 입력의 아홉 예측기 전체다. 큰 값은 숨기거나 잘라내지 않았다. 실패한 사례를 제외한 오차가 과도하게 좋아 보이지 않도록 유한 비율을 함께 표시했다.','',
        '| 예측기 | 16단계 위치 오차 | 61단계 위치 오차 | 61단계 유한 비율 |','| --- | ---: | ---: | ---: |']
    for k in PREDICTORS:
        lines.append(f"| {k} | {fmt(metric('measurement_mlp',k,16))} | {fmt(metric('measurement_mlp',k,61))} | {metric('measurement_mlp',k,61,'finite_fraction')*100:.1f}% |")
    lines+=['',f'13개 보류 그룹·모든 방식의 기본 예측 138,240개 경로 중 수치가 유한하지 않게 된 경로는 {failures:,}개였다. 이 분모는 반복 비교 경로의 수이며 독립 장면 수가 아니다. 최초 실패 이후는 NaN과 실패 표시를 유지했다. 유한한 값도 매우 클 수 있어 발산 비율 하나만으로 안정성을 판단할 수 없다.','',
        '## 새로운 조건에서의 긴 예측','',
        '영상 측정+학습 보정 입력, 공동 회전 모델의 결과다. 세 조건을 섞지 않고 미학습 속성·물체 수·외력을 각각 평가했다.','',
        '| 조건 | 16단계 위치 오차 | 61단계 위치 오차 | 61단계까지 벽 침범 1px 초과 또는 수치 실패 |','| --- | ---: | ---: | ---: |']
    for split,label in [('test','일반'),('attribute_ood','미학습 크기·색'),('count3','세 물체'),('count4','네 물체'),('force_ood','미학습 외력')]:
        lines.append(f"| {label} | {fmt(metric('measurement_mlp','joint_exact',16,split=split))} | {fmt(metric('measurement_mlp','joint_exact',61,split=split))} | {metric('measurement_mlp','joint_exact',61,'any_wall_violation_over_1px_or_nonfinite',split=split)*100:.1f}% |")
    lines+=['',
        '색과 존재 여부는 초기 추정에서 그대로 유지한다. 따라서 색 slot이 유지되는 것을 학습한 물체 추적 성공이라고 세지 않았다. 예측한 같은 색 물체의 위치가 실제로도 그 물체에 가장 가까운지 별도로 측정하며, 이 대응은 채점용이다. 예측을 정답 위치 순서로 다시 정렬하지 않는다.','',
        '## 충돌 시점은 얼마나 믿을 수 있는가','',
        '모델은 충돌 사건을 직접 출력하지 않는다. 두 프레임 사이 가까운 접근과 자유 운동에서 벗어난 속도 변화로 접촉 후보를 계산했다. 속도 잔차 0.1픽셀/프레임, 거리 여유 1.5픽셀을 실행 전에 고정했다. 단순히 물체가 겹치거나 통과해도 속도 반응이 없으면 충돌 성공으로 세지 않는다.','',
        '정답 궤적 78,080개 전이에도 같은 판정을 적용해 실제 simulator 사건과 비교했다. 아래는 이 판정기 자체의 성능이며 학습 모델의 성능이 아니다.','',
        '| 접촉 종류 | TP | FP | FN | 정밀도 | 재현율 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    cal=json.loads((root/'contact_proxy_calibration/summary.json').read_text())
    for k,label in [('pair','물체 사이'),('wall','벽')]:
        sums={q:sum(x['metrics'][k][q] for x in cal['groups']) for q in ['tp','fp','fn']}
        tp,fp,fn=[sums[q] for q in ['tp','fp','fn']]
        lines.append(f'| {label} | {tp} | {fp} | {fn} | {tp/(tp+fp)*100:.1f}% | {tp/(tp+fn)*100:.1f}% |')
    lines+=['',
        '후보 판정은 완벽하지 않다. 그 한계를 포함한 첫 접촉 시점 평가를 아래에 제시한다. 일반 test, 영상 측정+학습 보정 입력, 61단계 구간이다. 시점 오차는 양쪽 모두 사건이 있고 수치가 유한한 사례에 조건부이며, 사건 누락과 1프레임 이내 성공을 함께 봐야 한다.','',
        '| 예측기 | 접촉 | 첫 시점 오차 (조건부 frame) | 정답 사건 중 1frame 이내 | 정답 사건 누락 또는 수치 실패 |','| --- | --- | ---: | ---: | ---: |']
    for kind in ['interaction','joint_exact','force_wall']:
        for event in ['pair','wall']:
            get=lambda suffix:metric('measurement_mlp',kind,61,event+'_'+suffix)
            lines.append(f"| {kind} | {event} | {fmt(get('first_event_abs_error_both_finite'))} | {get('first_event_within_1_on_true_event')*100:.1f}% | {get('missed_event_on_true_event')*100:.1f}% |")
    lines+=['',
        '물체 사이 충돌을 계산하지 않는 force_wall에도 pair 후보가 생길 수 있다. 벽 반사로 속도가 바뀐 순간 근처에 다른 물체가 있으면 이 후보 규칙을 함께 만족할 수 있기 때문이다. 이를 학습한 물체 상호작용의 증거로 해석하지 않는다. 행동 변경에 대한 비대상 반응을 별도로 평가하는 이유도 여기에 있다.','',
        '## 행동을 바꾸면 다른 물체의 반응도 맞는가','',
        '같은 초기 추정·환경에서 대상에 주는 속도 변화량의 부호만 반대로 바꿨다. 두 미래의 차이를 예측하고 실제 두 미래의 차이와 비교했다. 물체 충돌은 행동의 영향을 다른 물체로 전달하므로 비대상을 무조건 보존하는 것은 정답이 아니다. 이 평가의 대응 자료 512개는 두 물체 장면이다.','',
        '다음은 변하는 외력 test에서 영상 측정+학습 보정 입력의 위치 반응 오차다. “무반응” 대조는 행동이 바뀌어도 예측 미래가 똑같다고 가정한 것이다. 낮을수록 좋다.','',
        '| 예측기 | 단계 | 대상 반응 오차 | 대상 무반응 대조 | 비대상 반응 오차 | 비대상 무반응 대조 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for kind in ['joint_exact','force_wall']:
        for h in [16,61]:
            keys=['target_position_response_mae_finite','target_position_zero_response_mae','nontarget_position_response_mae_finite','nontarget_position_zero_response_mae']
            lines.append(f'| {kind} | {h} | '+' | '.join(fmt(metric('measurement_mlp',kind,h,key,'response')) for key in keys)+' |')
    lines+=['',
        '현재 공동 회전 모델은 16단계에서 대상 자체의 행동 반응은 무반응 대조보다 잘 예측했지만, 비대상으로 전달되는 반응은 이 대조보다 나빴다. 61단계에서는 대상 반응에서도 우위가 유지되지 않았다. 외력·벽 참조는 물체 사이 상호작용을 생략하므로 비대상 반응이 정확히 0이다. 이 참조의 비대상 반응 오차가 무반응 대조와 같다는 것은 구현 의도와 일치한다.','',
        '## 검증과 남은 범위','',
        f"기본 예측 {v['factual_forecast_frames_replayed']:,}개 전이와 반대 행동 예측 {v['counterfactual_forecast_frames_replayed']:,}개 전이를 저장 초기 입력에서 재생했다. 주요 위치·속도·개수·공간 대응·전체 성공과 행동 반응 오차는 별도 코드로 계산했고, 나머지 지표도 고정 채점 코드로 재생했다. 입력의 색 기반 행동 전달, 개수 정답 미사용, 수치 실패의 지속, 데이터·가중치·코드 해시를 검사했다.",
        '',
        '전체 수치는 `all_metrics.csv`, 장면 단위 bootstrap 구간은 `aggregate.json`에 있다. 같은 장면의 세 초기화를 묶어 재표집했으며, 유한 사례 오차의 분모를 보존했다. 이것은 한 데이터 생성 seed 내부의 불확실성이고 독립 데이터 반복이나 모집단 유의성의 증거가 아니다.','',
        'C05/C06의 평가 작업은 완료했지만 장기 예측 능력을 달성했다고 주장하지 않는다. C07의 독립 반복·관측 길이·조건 누락, SVIB 외부 평가, 선행연구 마무리, 실제 사람 판정과 최종 재현 묶음은 남아 있다.','',
        '[전체 지표](all_metrics.csv) · [집계와 구간](aggregate.json) · [예측 재생 검증](verification.json) · [집계 감사](aggregate_verification.json) · [장기 곡선](long_horizon_curves.png) · [수치 실패](stability_overview.png) · [행동 반응](counterfactual_response.png) · [고정 첫 사례 영상](autonomous_examples.gif)']
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
