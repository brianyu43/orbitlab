"""Point-in-time progress only, never evidence that a process is currently live."""
import json
import time
from common import HERE,dump,r
from dynamics_context_omission import NAME


def main():
    root=HERE/f'reports/{NAME}';p=json.loads((root/'progress.json').read_text());v=json.loads((root/'verification_progress.json').read_text())
    theory=json.loads((root/'identifiability/verification.json').read_text());assert theory['all_passed']
    completed=v['all_passed'] and (root/'report_verification.json').exists() and json.loads((root/'report_verification.json').read_text())['all_passed']
    n=p['completed_conditions'];fresh=p['completed_new_trainings'];checked=v['conditions_verified'];pairs=v['zero_net_structure_training_comparisons']
    lines=['# 환경 정보 누락 대조 진행 기록','',
        '현재 상태·행동은 정확하게 주고 합력 또는 저항만 가린 채 처음부터 다시 학습한다. 학습 시점과 평가 시점의 정보가 같으며, 전체 정보 모델의 입력만 사후에 지운 대조가 아니다. 이 문서는 저장 산출물의 스냅샷이며 실제 실행 중 여부는 작업 세션에서 별도로 확인한다.','',
        '| 작업 | 문서 생성 시점 |','| --- | --- |',
        f'| 신규 누락 조건 학습 | {fresh}/81 완료 |',
        f'| 전체 정보 대조 평가 | {n-fresh}/27 완료 |',
        f'| 전체 조건 학습·평가 | {n}/108 완료 |',
        f'| 별도 재생 검증 | {checked}/108 완료 |',
        f"| 재생한 전이 예측 | {v['predictions_replayed']:,}개; 독립 표본 수가 아님 |",
        f"| 합력 숨김 회전 모델 학습 비교 | {len(pairs)}/18쌍 확인, 그중 가중치가 완전히 같은 쌍 {sum(x['weights_exactly_equal'] for x in pairs)}개 |",
        '| 같은 입력·다른 정답 구성 | 세 사례와 자유 운동식·제곱 손실 분해 검사 완료 |','',
        '독립 데이터 세 묶음과 세 초기화, 세 가지 상호작용/회전 모델을 비교한다. 학습 자료는 variable_force만 사용하고 평가 환경별로 다른 모델을 선택하지 않는다. 따라서 환경별로 각각 학습했던 C03의 표와 다른 비교다. 전체 정보 대조 27개는 기존 가중치를 재사용하지만, 누락 조건 81개는 새로 학습한다.','',
        '합력을 가리면 상태만 회전과 공동 회전 구조에서 회전시킬 관측 외력 벡터가 0이다. 같은 가중치일 때 함수와 gradient가 같다는 사전 검사를 통과했다. 이것이 숨긴 실제 외력이 없어졌거나 평가 분포가 회전 대칭이라는 뜻은 아니다.','',
        '세 구성 사례에서는 현재 상태·행동과 관측 환경 입력이 같은데 다음 정답이 달랐다. 해당 두 사례를 같은 비중으로 둔 제곱 손실의 최솟값을 검사했다. 이 값을 전체 무작위 평가 자료의 오차 하한이라고 주장하지 않는다.','',
        ('환경 정보 누락 대조의 전체 학습·재생·데이터 seed별 집계와 보고서 수치 검증이 완료됐다. [결과 보고서](RESULTS_KO.md)에서 조건별 효과와 한계를 설명한다.' if completed else '전체 학습·재생·데이터 seed별 집계가 끝나기 전까지 누락 정보의 최종 성능 효과는 판단하지 않는다.')+' 공통 시점의 관측 길이 1/2/4/8 실험, 외부 평가, 사람 판정과 최종 연구 산출물도 남아 있다.','',
        '[세부 실행안](../../planning_C07_CONTEXT_OMISSION_KO.md) · [입력·계산 검사](preflight.json) · [정보 부족 예제](identifiability/EXPLANATION_KO.md) · [학습 진행](progress.json) · [재생 진행](verification_progress.json)']
    (root/'PROGRESS_KO.md').write_text('\n'.join(lines)+'\n')
    dump(root/'progress_manifest.json',{'snapshot_unix':time.time(),'completed_conditions':n,'completed_new_trainings':fresh,'verified_conditions':checked,
        'context_omission_complete':completed,'C07_complete':False,'full_goal_complete':False,'report_sha256':r.sha(root/'PROGRESS_KO.md'),
        'source_sha256':r.sha(__file__),'evidence_sha256':{f:r.sha(root/f) for f in ['progress.json','verification_progress.json','preflight.json','identifiability/verification.json']}})
    print('omission progress',fresh,'fresh trained,',checked,'conditions verified',flush=True)


if __name__=='__main__':main()
