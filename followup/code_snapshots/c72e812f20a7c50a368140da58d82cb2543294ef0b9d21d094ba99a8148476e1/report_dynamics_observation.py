"""Source-backed C04 report and component-substitution comparison figure."""
import json
import numpy as np
from common import HERE,dump,r
from dynamics_observation_eval import NAME


def main():
    root=HERE/f'reports/{NAME}';study=json.loads((root/'summary.json').read_text())['results'];assert json.loads((root/'verification.json').read_text())['all_passed']
    labels={'rgb_cnn':'RGB 직접 CNN','measurement_mlp':'영상 측정 + 학습 보정','rgb_analytic':'영상 측정 + 물리식','true_positions_analytic':'정답 위치 + 물리식 (특권 대조)'}
    def infer(method,mode,split,key):
        values=[v['inference']['all']['metrics'][key]['mean'] for v in study if (v['method'],v['mode'],v['split'])==(method,mode,split)]
        assert len(values)==3;return np.mean(values)
    def forecast(method,mode,split,info,kind='joint_exact',key='zero_filled_true_slot_velocity_mae'):
        values=[v['forecasts'][info+'__'+kind]['all']['metrics'][key]['mean'] for v in study if (v['method'],v['mode'],v['split'])==(method,mode,split)]
        assert len(values)==3;return np.asarray(values)
    lines=['# 네 장의 과거 영상에서 상태·환경을 추정한 결과','',
        '2026-09-23. C04의 영상 추정과 한 단계 연결 대조를 완료했다. 미래 전체를 스스로 이어 예측한 결과는 아직 아니다.','',
        '## 입력과 학습','',
        't=0,1,2,3의 RGB 네 장을 입력하고 마지막 물체 상태 및 일정한 합력·저항을 추정했다. 행동은 t=3 이후 첫 이동에 적용하므로 네 입력 장면에는 아직 행동 효과가 없다. 명령은 대상 색과 속도 변화량으로 제공한다. 세계의 물체는 서로 다른 색이라 이 정보는 영상에서 확인할 수 있다. 모델에는 숨은 ID·정답 개수·상태·환경 종류를 제공하지 않는다.','',
        'RGB 직접 CNN과 영상 측정값을 보정하는 공유 MLP를 각각 세 초기화로 학습했다. 세 환경의 train 768개 궤적에서 초기 네 장만 사용했고, batch 32·Adam 0.0005·고정 4,000 steps를 맞췄다. 감독 신호는 train의 t=3 상태와 합력·저항이다. CNN은 1,112,103개 파라미터로 MPS에서, 보정 MLP는 31,369개로 CPU에서 학습했다. 파라미터·시간·사전 지식까지 같게 맞춘 비교는 아니다.','',
        '영상 측정은 알려진 여섯 색, 검은 배경, 원 모양과 radial alpha 규칙으로 중심·크기를 맞춘다. 네 장의 변위로 합력과 저항을 추정할 때에도 simulator의 여덟 substep 적분 규칙을 이용한다. 이 강한 비학습 참조를 물체 개념이나 물리 법칙을 스스로 발견한 모델이라고 부르지 않는다. MLP는 이 참조를 입력과 초기 추정으로 이용한다.','',
        'RGB 직접 CNN은 train에서 좌표 오차 약 0.64픽셀이었지만 validation은 약 3.80픽셀이었다. 현재 데이터·모델·학습 예산에서 큰 일반화 격차가 관측됐다. 이미지 모델 전반의 불가능성으로 확대하지 않는다.','',
        '## 보류 자료의 상태·환경 추정','',
        '변하는 외력 환경의 일반 test 128개 궤적이다. CNN/MLP는 세 초기화 평균이며, 물리식 추정은 학습 초기화가 없는 동일 결과다. 위치·속도는 검출한 정답 색 slot에 조건부인 평균 절대 오차다. 물체 색의 존재 여부도 따로 표시했다.','',
        '| 방법 | 모든 물체 색 존재 판정 일치 | 위치 오차 (px) | 속도 오차 (px/frame) | 합력 오차 (px/frame²) | 저항 오차 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for method,label in labels.items():
        keys=['all_presence_correct','matched_position_mae_pixels','matched_velocity_mae_pixels_per_frame','net_force_mae_pixels_per_frame_squared','drag_absolute_error']
        v=[infer(method,'variable_force','test',k) for k in keys];lines.append(f'| {label} | {v[0]*100:.1f}% | {v[1]:.5f} | {v[2]:.4f} | {v[3]:.5f} | {v[4]:.5f} |')
    lines+=['',
        '원 그리는 규칙을 아는 측정기의 위치 오차는 매우 작다. 그러나 관측 중 충돌이 있으면 자유 운동 식으로 외력을 맞추는 가정이 깨진다. 이 환경의 test에서 비접촉 119개 궤적의 물리식 속도 오차는 약 0.00035, 과거 접촉이 있는 9개는 약 0.2893픽셀/프레임이었다. 정확한 위치를 제공해도 접촉 구간의 오차는 거의 같아서, 이 경우는 픽셀 위치 측정만의 문제가 아니다.','',
        '학습 보정은 일반 test의 합력 오차를 낮췄지만 위치·속도 모두를 비학습 참조보다 개선하지는 못했다. 학습에 없던 외력에서의 성능도 별도로 봐야 한다.','',
        '## 한 단계 예측에 무엇이 영향을 주는가','',
        '이미 학습한 variable_force의 일반 상호작용·상태만 회전·상태와 외력 공동 회전 모델을 고정했다. 어떤 환경을 평가하든 같은 가중치를 사용하므로 실제 환경 이름으로 모델을 고르지 않는다. 등속도와 알려진 외력·벽 반사 참조도 비교했다. 후자는 물체 간 충돌을 생략한다.','',
        '아래는 공동 회전 모델의 속도 오차다. 정답 상태와 정답 환경을 교체하는 네 대조에서 시작 정보만 바꾸었다. 행동은 예측으로 찾은 대상 색에 전달하며, 대상을 놓치면 그 실패도 기록한다.','',
        '| 상태/환경 제공 방식 | RGB CNN | 측정 + 학습 보정 | 측정 + 물리식 |',
        '| --- | ---: | ---: | ---: |']
    info_labels={'true_state_true_context':'정답 상태 + 정답 환경','estimated_state_true_context':'추정 상태 + 정답 환경','true_state_estimated_context':'정답 상태 + 추정 환경','estimated_state_estimated_context':'추정 상태 + 추정 환경'}
    for info,label in info_labels.items():lines.append('| '+label+' | '+' | '.join(f'{forecast(m,"variable_force","test",info).mean():.4f}' for m in list(labels)[:3])+' |')
    lines+=['',
        '현재 RGB CNN에서는 환경만 틀리는 것보다 위치·속도를 틀리게 읽는 영향이 훨씬 컸다. 다만 지표 차이를 독립적·가산적인 인과 효과로 해석하지 않는다. 구성 요소의 오류는 상호작용할 수 있다.','',
        '이 수치는 t=3→4만 평가했다. C03의 전체 보류 trajectory transition 평균 0.0411과 평가 시점의 분포가 달라, 이번 정답 상태 대조 0.0246을 새 모델의 성능 향상으로 해석하지 않는다. 모델 가중치는 같다.','',
        '## 물체 수와 미학습 외력','',
        '추정 상태와 추정 환경을 모두 사용한 공동 회전 예측기의 속도 오차다. 빠뜨린 정답 색 slot은 0 상태로 채워 계산했고, 검출·개수 지표는 원자료에 함께 있다.','',
        '| 조건 | RGB CNN | 측정 + 학습 보정 | 측정 + 물리식 | 정답 상태 + 정답 환경 |','| --- | ---: | ---: | ---: | ---: |']
    for split,label in [('test','일반 test'),('attribute_ood','미학습 크기·색'),('count3','세 물체'),('count4','네 물체'),('force_ood','미학습 외력')]:
        values=[forecast(m,'variable_force',split,'estimated_state_estimated_context').mean() for m in list(labels)[:3]]
        values.append(forecast('rgb_cnn','variable_force',split,'true_state_true_context').mean())
        lines.append('| '+label+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines+=['',
        '학습 보정은 미학습 외력에서 물리식 참조보다 나빴다. 일반 조건의 보정 이득을 미학습 환경까지 그대로 옮겨 말할 수 없다. 네 물체에서 RGB CNN의 모든 색 존재 판정 일치는 7.3%로 떨어졌다. 원의 영상 측정은 이 조건에서도 검출이 잘 됐지만, 정답 상태 대조에도 미래 예측 오차가 남아 운동 모델의 개수 일반화 문제와 구분해야 한다.','',
        '## 완벽한 영상이어도 남는 정보의 한계','',
        '합력이 a=gamma*v인 일정 속도 궤적에서는 서로 다른 저항과 합력이 같은 과거를 만들 수 있다. 같은 행동 이후에는 미래가 달라진다. 실제 renderer에서 과거 네 장이 완전히 같은 두 환경을 구성해 검사했다. 이는 특정 경계 사례이며 일반 평가 자료 전체의 식별 불가능성을 주장하지 않는다. [설명과 수치](../observation_identifiability_v1/EXPLANATION_KO.md)','',
        '## 검증과 다음 단계','',
        '- 입력 계약: 2,240개 궤적·RGB 8,960장과 행동 시점, 512개 반사실의 동일 과거를 확인했다.',
        '- 비학습 참조: 모든 RGB를 재실행하고 2,240개 추정 및 6,720개 지표 행, 3,384개 집계·구간을 확인했다.',
        '- 학습 모델: 6개 checkpoint의 train/validation 예측 5,760개를 CPU에서 재생하고 train 전용 입력·표본·optimizer step을 확인했다.',
        '- 보류 연결 대조: 156개 조건의 추정 행 15,360개와 한 단계 예측 307,200개, 집계·구간 93,456개를 재검증했다. 비학습 추정과 정답 대조는 반복 사용됐으므로 이 수만큼 독립적인 학습이나 장면이 있다는 뜻은 아니다.',
        '- 평가 대상은 13개 보류 그룹의 원본 궤적 총 1,280개, 데이터 생성 seed 하나다. 세 학습 초기화와 궤적 bootstrap 구간을 모집단 유의성으로 해석하지 않는다.',
        '- 다음 C05/C06은 t=3 추정 상태·환경을 한 번만 넣고 최대 61단계 예측, 같은 과거에서 반대 행동에 따른 비대상 물체 반응을 평가한다. 이후 C07에서 독립 반복과 관측 길이·조건 누락의 경계를 평가한다.','',
        '[전체 결과](summary.json) · [재생 검증](verification.json) · [구성 요소 비교 그림](state_context_substitution.png) · [학습 개발 결과](../dynamics_observation_models_v1/development_summary.json)']
    (root/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');colors=['#934341','#38729a','#579369']
    names=['RGB CNN','Measured + learned','Measured + analytic'];information=list(info_labels);table=[]
    for ax,split,title in zip(axes,['test','force_ood'],['Variable force: test','Variable force: held-out force']):
        for j,(method,label,color) in enumerate(zip(list(labels)[:3],names,colors)):
            values=np.array([forecast(method,'variable_force',split,info) for info in information]);means=values.mean(1);x=np.arange(4)+(j-1)*.24
            ax.bar(x,means,width=.22,label=label,color=color,alpha=.85)
            for k in range(3):ax.scatter(x,values[:,k],s=10,color='black',alpha=.5,zorder=4)
            table.extend({'method':method,'split':split,'information':info,'values_by_seed':v.tolist(),'mean':float(v.mean())} for info,v in zip(information,values))
        ax.set_xticks(range(4),['True S\nTrue C','Pred S\nTrue C','True S\nPred C','Pred S\nPred C']);ax.set_yscale('log');ax.set(title=title,ylabel='One-step velocity MAE (px/frame; log scale)');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('State (S) and context (C) substitution — fixed joint-symmetry transition',fontsize=13)
    fig.savefig(root/'state_context_substitution.png',dpi=160);fig.savefig(root/'state_context_substitution.svg');plt.close(fig)
    # Report/plot means are directly checked against the independently verified seed metrics.
    for row in table:assert abs(np.mean(row['values_by_seed'])-row['mean'])<1e-12
    dump(root/'report_manifest.json',{'report_sha256':r.sha(root/'RESULTS_KO.md'),'summary_sha256':r.sha(root/'summary.json'),
        'figures':{f:r.sha(root/f) for f in ['state_context_substitution.png','state_context_substitution.svg']},
        'plot_data':table,'script_sha256':r.sha(__file__),'seed_dots':'Three paired learned estimator/transition seeds; the analytic estimator itself is fixed.'})
    print('C04 report and substitution figure written')


if __name__=='__main__':main()
