"""Write the complete state-only comparison from verified machine-readable results."""
import json
from common import HERE,dump,r


def main():
    root=HERE/'reports/dynamics_state_study_v1';agg=json.loads((root/'aggregate.json').read_text())
    for path in [root/'verification.json',root/'aggregate_verification.json',HERE/'reports/dynamics_state_baselines_v1/verification.json']:
        assert json.loads(path.read_text())['all_passed']
    base=json.loads((HERE/'reports/dynamics_state_baselines_v1/summary.json').read_text())['results']
    labels={'flat':'장면 전체 MLP','object':'독립 물체 MLP','interaction':'물체 상호작용','augmented':'상태·외력 회전 증강',
        'wrong_exact':'상태만 회전하는 제약','joint_exact':'상태·외력 함께 회전하는 제약','relaxed':'상태 회전 제약과 일반 모델 혼합',
        'inertial':'등속도 참조','force_wall':'알려진 외력·벽 반사 참조'}
    def value(mode,kind,split,category='all'):
        if kind in ['inertial','force_wall']:
            return next(x for x in base if x['mode']==mode and x['kind']==kind and x['split']==split and x['category']==category)['metrics']['velocity_mae_pixels_per_frame']['mean']
        return next(x for x in agg['groups'] if x['mode']==mode and x['kind']==kind and x['split']==split and x['category']==category)['metrics']['velocity_mae_pixels_per_frame']['mean']
    kinds=list(labels);lines=['# 정답 상태에서 한 단계 운동 예측: 본 비교 완료','',
        '2026-09-23. 3환경 × 7방식 × 3초기화의 63개 학습 모델을 평가했다. 기존에 검증한 9개 checkpoint를 재사용하고 54개를 새로 학습했다. 이 비교는 C03이며, 영상에서 상태를 알아내는 C04와 연속 미래 예측 C05는 별도 단계다.','',
        '## 공통 입력과 비교 조건','',
        '매 순간 실제 물체 상태·행동·알려진 합력과 저항 계수를 제공했다. 두 물체의 train 궤적에서 균일하게 transition을 뽑아 batch 64, Adam 0.001, 공통 12,000 steps를 사용했다. 여러 checkpoint 중 test가 좋은 것을 고르지 않았다. 대칭 대조군의 raw 초기 가중치와 표본 흐름을 맞췄다. 파라미터 수와 실제 계산 시간은 다르며, 시간까지 맞춘 비교는 아니다. 데이터 생성 seed는 하나, 초기화는 세 개다.','',
        '물체별 모델은 다른 물체를 보지 않는다. 상호작용 모델은 다른 물체의 메시지를 합한다. 상태만 회전시키는 제약은 방향성 외력을 고정한 채 상태를 회전한다. 상태·외력 공동 제약은 좌표계를 바꿀 때 외력도 함께 바꾼다. 방향성 외력이 없는 환경에서는 두 제약이 같아진다.','',
        '## 일반 test','',
        '속도 평균 절대 오차이며 단위는 픽셀/프레임이다. 낮을수록 좋다. 먼저 궤적 내 평균을 구하고, 학습 모델은 세 초기화 평균을 표시했다.','',
        '| 방법 | 방향성 없는 환경 | 고정 중력 | 변하는 외력 |','| --- | ---: | ---: | ---: |']
    for kind in kinds:lines.append('| '+labels[kind]+' | '+' | '.join(f'{value(m,kind,"test"):.4f}' for m in ['isotropic','fixed_gravity','variable_force'])+' |')
    lines+=['','## 변하는 외력: 보류 조건과 물체 수','',
        '| 방법 | 일반 test | 미학습 속성 조합 | 세 물체 | 네 물체 | 미학습 외력 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for kind in kinds:lines.append('| '+labels[kind]+' | '+' | '.join(f'{value("variable_force",kind,s):.4f}' for s in ['test','attribute_ood','count3','count4','force_ood'])+' |')
    lines+=['','## 무엇이 개선되고 무엇이 남았나','',
        '변하는 외력의 일반 test와 미학습 외력에서 공동 회전 제약은 일반 상호작용·회전 증강·상태만 회전·혼합 모델보다 세 초기화 모두 오차가 작았다. 하지만 네 물체에서는 상태만 회전하는 모델보다 세 초기화 모두 좋지는 않았다. 특정 조건에서의 효과를 모든 일반화 능력으로 확대하지 않는다.','',
        '물체 수 증가로 여러 상호작용 모델의 오차가 크게 늘었다. 메시지를 개수로 정규화하지 않고 합하는 구조가 후보 원인이지만, 집계 방식만 바꾼 대조 실험을 하지 않아 원인으로 확정하지 않는다. 독립 물체 모델은 상호작용을 배제한 구조라서 개수 변화에는 덜 민감할 수 있지만 충돌 모델로 충분하다는 뜻은 아니다.','',
        '평균에는 충돌이 없는 많은 순간이 포함된다. 충돌 순간을 분리한 변하는 외력 test의 속도 오차는 아래와 같다.','',
        '| 방법 | 비접촉 | 물체끼리 접촉 | 벽에만 접촉 |','| --- | ---: | ---: | ---: |']
    for kind in ['interaction','joint_exact','inertial','force_wall']:
        lines.append('| '+labels[kind]+' | '+' | '.join(f'{value("variable_force",kind,"test",cat):.4f}' for cat in ['no_contact','pair','wall_only'])+' |')
    lines+=['',
        '외력·벽 반사 참조는 simulator의 알려진 적분 규칙과 경계를 그대로 이용하고 물체 간 접촉을 생략한다. 학습으로 물리 법칙을 알아낸 모델이 아니다. 전체 평균에서 이 강한 비학습 참조가 더 정확하므로, 학습 모델이 이 세계를 전반적으로 더 잘 예측한다고 주장하지 않는다.','',
        '## 검증과 불확실성','',
        '- 63개 모델의 보류 transition 예측 1,720,320건을 재생했다. optimizer·표본·초기화·재사용 해시와 2,730개 지표 집계 검사를 통과했다.',
        '- 물리 참조는 원래 simulator로 개별 원형 물체 200,704건을 다시 계산하고, 12,022개 지표 행 및 260개 집계·구간을 검사했다.',
        '- 전체 63개 원자료 CSV에서 455개 조건의 평균·초기화 표준편차·장면 bootstrap 구간 910개와 52개 짝 비교를 별도로 검사했다.',
        '- 초기화 세 값과 표본 표준편차를 장면 재표집 구간과 분리했다. 구간은 이 세 모델에 조건부이며, 독립 데이터 seed 반복이나 모집단의 유의성 검정은 아니다.',
        '- 매 순간 정답 상태가 다시 들어가는 한 단계 예측이다. 긴 rollout, 시각 인식, 환경 추론, 반사실 예측의 성능은 여기서 확인하지 않았다.','',
        '## 다음 단계','',
        '같은 과거 관측 길이로 영상에서 위치·속도·환경을 추정하고, 정답 상태를 다시 주지 않는 연속 예측으로 오차 누적을 측정한다. 물체 수 실패와 외력의 식별 가능성, 행동 변화에 따른 비대상 물체 반응도 별도로 평가한다.','',
        '[모든 조건 집계](aggregate.json) · [원자료 비교 표](comparison_rows.csv) · [모델 재생 검증](verification.json) · [집계 감사](aggregate_verification.json) · [물리 참조 검증](../dynamics_state_baselines_v1/verification.json)']
    path=root/'RESULTS_KO.md';path.write_text('\n'.join(lines)+'\n')
    dump(root/'report_manifest.json',{'report_sha256':r.sha(path),'aggregate_sha256':r.sha(root/'aggregate.json'),
        'baseline_summary_sha256':r.sha(HERE/'reports/dynamics_state_baselines_v1/summary.json'),'script_sha256':r.sha(__file__)})
    print('Wrote verified C03 report')


if __name__=='__main__':main()
