"""Point-in-time evidence for the complete 3-data, 4-length experiment."""
import json
import time
from common import HERE,dump,r
from dynamics_observation_length_data import NAME,DATA


def optional(path):return json.loads(path.read_text()) if path.exists() else {}


def main():
    root=HERE/f'reports/{NAME}';training=optional(root/'training_progress.json');verified=optional(root/'model_verification_progress.json')
    world=optional(root/'data_verification.json');inputs=optional(root/'input_verification.json');preflight=optional(root/'model_preflight.json')
    assert world['all_passed'] and inputs['all_passed'] and preflight['all_passed']
    lines=['# 관측 길이 1·2·4·8 비교 진행 기록','',
        '기존 독립 데이터 세 묶음의 초기 장면을 재사용하고 행동 시점을 t=7로 옮겼다. 모든 관측 길이가 같은 마지막 상태·행동·미래를 예측한다. 새로운 독립 데이터가 세 묶음 더 생긴 것으로 세지 않는다. 이 문서는 산출물 스냅샷이며 실행 중 여부는 세션에서 별도로 확인한다.','',
        f'궤적 {world["episodes_replayed"]:,}개, RGB {world["observations_rerendered"]:,}장을 재생했다. 모든 RGB 측정, 길이별 참조 추정 {inputs["visible_window_estimates_independently_recomputed"]:,}건과 집계를 검사했다. 가린 입력을 바꿔도 계산이 같고, 길이별 모델 크기·초기 가중치가 같은지 사전 검사했다.','',
        f'신규 영상 추정기 {training.get("completed_models",0)}/72개 학습 완료, 별도 학습·개발 예측 재생 검증 {verified.get("models_verified",0)}/72개 완료. 학습은 최종 4,000 steps 고정이며 중간 점수로 모델을 선택하지 않는다.','',
        '| 데이터 / 관측 장수 | 한 단계 대조 완료 / 검증 | 긴 미래·반대 행동 완료 / 검증 |','| --- | --- | --- |']
    records=[]
    for ds in DATA:
        for length in [1,2,4,8]:
            folder=root/f'd{ds}/L{length}'
            obs=optional(folder/'observation_eval_progress.json').get('completed_conditions',0)
            obsv=optional(folder/'observation_eval_verification.json').get('conditions_verified',0)
            auto=optional(folder/'autonomous_progress.json').get('completed_conditions',0)
            av=optional(folder/'autonomous_verification.json').get('conditions_verified',0)
            partial=optional(folder/'autonomous_verification_progress.json').get('verified_conditions',0)
            av=max(av,partial)
            lines.append(f'| {ds} / {length}장 | {obs}/156 · {obsv}/156 | {auto}/1404 · {av}/1404 |')
            records.append({'data_seed':ds,'length':length,'observation_completed':obs,'observation_verified':obsv,'autonomous_completed':auto,'autonomous_verified':av})
    lines+=['',
        '한 단계 대조는 지각·환경 입력을 정답으로 바꾸어 원인을 나누고, 긴 미래는 1/4/8/16/32/61단계와 반대 행동의 영향을 비교한다. 관측 중 접촉 범주는 보이는 L−1개 전이만 사용하므로 길이마다 구성원이 달라질 수 있다. 길이 효과의 주 비교는 같은 전체 장면을 짝맞춘다.','',
        '단일 영상에서 추정한 속도·환경의 성능, 긴 관측의 이득과 그 한계는 전체 평가·재생·길이별 집계가 끝나기 전 확정하지 않는다. 현재 입력·모델 검증 통과가 세계 모델의 정확도 달성을 뜻하지 않는다.','',
        '남은 일: 모델 학습/재생 마무리, 모든 데이터와 길이의 연결·장기·반대 행동 평가 및 재생, 길이별 짝 비교 집계와 보고서. 이후 환경 정보 누락 결과와 종합한다. 외부 SVIB·선행연구·실제 사람 판정·최종 연구 묶음도 별도로 남아 있다.','',
        '[입력 검증](input_verification.json) · [학습 입력 사전 검사](model_preflight.json) · [실행 계획](../../planning_C07_OBSERVATION_LENGTH_KO.md) · [실행 세션](execution_sessions.json)']
    (root/'PROGRESS_KO.md').write_text('\n'.join(lines)+'\n')
    dump(root/'progress_snapshot.json',{'snapshot_unix':time.time(),'trained_models':training.get('completed_models',0),
        'verified_models':verified.get('models_verified',0),'evaluations':records,'observation_length_complete':False,
        'C07_complete':False,'full_goal_complete':False,'source_sha256':r.sha(__file__),'report_sha256':r.sha(root/'PROGRESS_KO.md')})
    print('length progress snapshot',training.get('completed_models',0),'trained,',verified.get('models_verified',0),'verified',flush=True)


if __name__=='__main__':main()
