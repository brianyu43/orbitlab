"""Evidence-bounded progress snapshot, never substitutes for a live process poll."""
import json
import time
from common import HERE,dump,r
from dynamics_repeat_data import NAME


def main():
    root=HERE/f'reports/{NAME}'
    data=json.loads((root/'data_verification.json').read_text());rgb=json.loads((root/'observation_data_verification.json').read_text())
    obs=json.loads((root/'observation_development_verification.json').read_text())
    state=json.loads((root/'state_progress.json').read_text());checked=json.loads((root/'state_verification_progress.json').read_text())
    assert data['all_passed'] and rgb['all_passed'] and obs['all_passed']
    n,m=state['completed_models'],checked['models_verified']
    across_complete=all((root/f'across_data/{stage}_verification.json').exists() and
        json.loads((root/f'across_data/{stage}_verification.json').read_text()).get('all_passed',False)
        for stage in ['state','observation','autonomous'])
    evaluations={}
    for ds in [997101,997102]:
        stages={}
        for stage in ['observation_eval','autonomous']:
            p=root/f'd{ds}/{stage}_progress.json'
            if p.exists():
                stages[stage]=json.loads(p.read_text())
        evaluations[str(ds)]=stages
    evaluation_text=[]
    for ds,stages in evaluations.items():
        if not stages:
            evaluation_text.append(f'데이터 {ds}: 연결 평가 저장 결과 없음.')
        for stage,v in stages.items():
            label='영상 입력 대조' if stage=='observation_eval' else '긴 미래·행동 변경'
            verdict=root/f'd{ds}/{stage}_verification.json'
            audited=verdict.exists() and json.loads(verdict.read_text()).get('all_passed',False)
            evaluation_text.append(f"데이터 {ds}의 {label}: {v['completed_conditions']}/{v['expected_conditions']}개 조건 저장, 별도 재생 검증 {'완료' if audited else '미완료'}.")
    lines=['# 독립 데이터 반복 진행 기록','',
        '2026-09-23. 아래 개수는 문서 생성 시점의 저장 산출물 기준이며, 실행 중 여부는 실제 작업 세션으로 별도 확인한다. 전체 C07 및 후속 목표는 진행 중이다.','',
        ('독립 데이터 반복의 학습·C03–C06 평가·집계 검증과 비교 보고를 완료했다. [결과 보고서](RESULTS_KO.md). 관측 길이 실험과 환경 정보 누락 대조는 별도다.' if across_complete else '독립 데이터 반복의 최종 비교는 아직 진행 중이다.'),'',
        '## 비교를 유지한 조건','',
        '원래 데이터 seed 640031에 새 seed 997101·997102를 더한다. 모델 초기화 세 개와 학습 표본 추출 순서는 기존과 같다. 데이터 seed와 초기화를 교차한 설계이므로 이를 독립 데이터 아홉 묶음이라고 세지 않는다. 원래 학습 가중치를 옮겨 쓰지 않고 새 train에서 처음부터 학습한다.','',
        '상태 예측은 세 환경×일곱 모델×세 초기화×새 데이터 두 묶음으로 총 126개다. 기존과 같은 12,000 steps·batch 64·Adam 0.001·hidden 48·무작위 물체 순서를 유지했다. 영상 추정은 직접 CNN과 측정+보정 MLP 각각 세 초기화, 두 데이터 묶음으로 총 12개다. 4,000 steps·batch 32·Adam 0.0005 및 알려진 원·색·물리 규칙의 사용 범위를 유지했다.','',
        '## 실제 완료와 남은 작업','',
        '| 항목 | 완료한 증거 | 아직 남은 범위 |','| --- | --- | --- |',
        f"| 새 세계 자료 | 궤적 {data['episodes_replayed']:,}개, 상태 {data['states_replayed']:,}개, RGB {data['observations_rerendered']:,}장 재생 | 아래 모델 평가 상태와 구분 |",
        f"| 행동 변경·대칭 | 반대 행동 {data['counterfactuals_replayed']:,}개, 회전 {data['rotated_trajectories_checked']:,}개, 물체 순서 변경 {data['permuted_trajectories_checked']:,}개 검사 | 아래 장기 행동 평가 상태와 구분 |",
        f"| 자료 독립성 | 기존과 신규를 합친 {data['unique_physical_and_image_orbits_including_original']:,}개 초기 물리/영상 회전 궤도가 중복 없이 분리됨 | 더 넓은 실제 환경으로 일반화한다는 의미는 아님 |",
        f"| 영상 입력과 비학습 참조 | RGB 측정 {rgb['rgb_measurements_replayed']:,}장, 정답·추론 파일 분리, 기준선 지표 검증 | 새 학습 모델의 보류 조건 연결 대조 |",
        f"| 새 영상 추정기 | {obs['models_verified']}/12 학습 및 검증, train/validation 예측 {obs['train_validation_predictions_replayed']:,}개 CPU 재생 | test/OOD·물체 수·미학습 외력 및 장기 연결 평가 |",
        f'| 새 상태 예측기 | {n}/126 학습·보류 평가 완료, 이 중 {m}개 재생 검증 완료 | 나머지 학습·검증 및 데이터 seed별 통합 비교 |',
        '| 관측 길이·정보 누락 | 별도 세부 계획과 식별 한계 구성 예제 | 1/2/4/8프레임은 미실행; 정보 누락은 별도 진행 기록 참고 |','',
        '영상 추정기의 train/validation 개발 검증과 보류 조건의 연결 평가는 별도로 기록한다. 데이터가 바뀌었을 때 validation 오차가 달라진 것을 test 성능 향상이나 독립 확인의 최종 결론으로 발표하지 않는다. 상태 모델의 완료된 보류 예측도 모든 데이터 seed의 비교가 끝나기 전에는 중간 결과다. 좋은 초기화만 골라 중단하거나 원래 기준을 낮추지 않는다.','',
        '현재 `dynamics_repeat_eval.py`는 새 가중치·새 입력의 C04 성분 대조와 C05/C06 자율 예측을 연결한다. 신규 validation 추론 2,304개를 저장된 개발 예측과 비교해 입력 경로·복호화가 일치함을 확인했다. 전체 연결 평가는 해당 데이터 묶음의 상태 모델 63개 검증 영수증과 영상 모델 검증을 확인한 후에만 실행한다. 독립 평가의 별도 재생 검증과 데이터 묶음 간 집계도 수행해야 한다.','',
        ' '.join(evaluation_text),'',
        '[환경 정보 누락 대조 진행](../dynamics_context_omission_v1/PROGRESS_KO.md) · [관측 길이 세부 실행안](../../planning_C07_OBSERVATION_LENGTH_KO.md)','',
        '[자료 검증](data_verification.json) · [영상 입력 검증](observation_data_verification.json) · [영상 모델 검증](observation_development_verification.json) · [상태 학습 개수](state_progress.json) · [상태 재생 검사](state_verification_progress.json) · [전체 C07 실행안](../../planning_C07_EXECUTION_KO.md)','',
        'SVIB 외부 평가, 선행연구 비교, 실제 사람 판정, 최종 연구 메모·그림·재현 묶음의 전체 범위도 유지한다.']
    (root/'PROGRESS_KO.md').write_text('\n'.join(lines)+'\n')
    dump(root/'progress_manifest.json',{'snapshot_unix':time.time(),'state_models_completed':n,'state_models_verified':m,
        'expected_state_models':126,'observation_models_verified':12,'C07_complete':False,'full_goal_complete':False,
        'independent_repeat_aggregate_verified':across_complete,
        'evaluation_progress':evaluations,
        'report_sha256':r.sha(root/'PROGRESS_KO.md'),'source_sha256':r.sha(__file__),
        'evidence_sha256':{p:r.sha(root/p) for p in ['data_verification.json','observation_data_verification.json','observation_development_verification.json','state_progress.json','state_verification_progress.json']},
        'warning':'Progress-file hashes are a point-in-time snapshot, not a current process-liveness claim.'})
    print('repeat progress report',n,'state models completed,',m,'verified; 12/12 estimators verified',flush=True)


if __name__=='__main__':main()
