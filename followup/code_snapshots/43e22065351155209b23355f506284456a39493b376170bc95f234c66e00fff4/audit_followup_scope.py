"""Requirement-by-requirement closeout; missing human data can never pass."""
import argparse
import csv
import json
import re
import time
from pathlib import Path
from common import HERE,dump,r

CHECKS={
 'A01':[('reports/original_integrity_before_followup_release_v1.json','all_match',True),('reports/original_integrity_before_followup_release_v1.json','checked_files',1324)],
 'A02':[('data/probe_v1/manifest.json','new_splits_disjoint_from_all_original_orbits',True)],
 'A03':[('reports/evaluator_audit_v1/summary.json','base_scenes',256)],
 'A03b':[('reports/evaluator_calibration_v1/summary.json','human_validated',False)],
 'A05':[('reports/probe_step_comparison_v1/verification.json','all_passed',True),('reports/probe_step_comparison_v1/verification.json','checkpoints_verified',12)],
 'A06':[('reports/initial_verification.json','all_passed',True)],
 'A07':[('reports/flow_fixed_summary.json','matching_verified',True),('reports/initial_verification.json','trained_flows_verified',18)],
 'A08':[('reports/flow_time_summary.json','matching_verified',True),('reports/generation_cost_figure_v2/verification.json','all_passed',True)],
 'A09':[('reports/confirmation_v1/verification.json','all_passed',True),('reports/confirmation_v1/verification.json','ae_checkpoints',9),('reports/confirmation_v1/verification.json','flow_checkpoints',27)],
 'A10':[('reports/decoder_study_v1/verification.json','all_passed',True),('reports/decoder_study_v1/verification.json','decoder_trainings',12),('reports/decoder_transfer_v1/verification.json','all_passed',True)],
 'A11':[('reports/invariant_bound_v1/summary.json','all_512_scenes_above_bound',True)],
 'A12':[('reports/final_memo_verification.json','all_passed',True)],
 'B01':[('reports/object_world_v1/verification.json','all_passed',True),('reports/object_world_v1/verification.json','source_scenes',1792)],
 'B02':[('reports/object_world_v1/verification.json','edit_pairs',4820),('reports/object_world_v1/verification.json','cross_split_source_and_target_orbit_overlap',0)],
 'B03':[('reports/object_oracle_study_v1/verification.json','all_passed',True),('reports/object_oracle_study_v1/verification.json','parameter_and_sample_matching',True)],
 'B04':[('reports/object_layers_heldout_v1/verification.json','all_passed',True),('reports/object_layers_heldout_v1/verification.json','five_learned_slots_per_model_verified',True)],
 'B05':[('reports/object_image_edit_v1/verification.json','all_passed',True),('reports/object_image_edit_v1/verification.json','conditions_verified',115),('reports/object_image_edit_v1/verification.json','rendered_outputs_replayed',68103)],
 'B06':[('reports/svib_preview_shape_swap_v1/report_verification.json','all_passed',True),('reports/svib_preview_shape_swap_v1/data_verification.json','pairs_verified',500)],
 'B07':[('reports/object_failure_audit_v1/verification.json','all_passed',True),('reports/object_image_edit_v1/analysis_verification.json','all_passed',True)],
 'C01':[('reports/dynamics_world_v1/verification.json','all_passed',True),('reports/dynamics_world_v1/verification.json','episodes_replayed',2240)],
 'C02':[('reports/dynamics_world_v1/verification.json','max_joint_rotation_state_error',0),('reports/dynamics_world_v1/verification.json','fixed_condition_counterexamples',96)],
 'C03':[('reports/dynamics_state_study_v1/verification.json','all_passed',True),('reports/dynamics_state_study_v1/verification.json','models_verified',63),('reports/dynamics_state_baselines_v1/verification.json','all_passed',True)],
 'C04':[('reports/dynamics_observation_eval_v1/verification.json','all_passed',True),('reports/dynamics_observation_eval_v1/verification.json','one_step_forecasts_replayed_and_scored',307200)],
 'C05':[('reports/dynamics_autonomous_v1/verification.json','all_passed',True),('reports/dynamics_autonomous_v1/verification.json','all_recurrences_replayed_without_future_labels',True)],
 'C06':[('reports/dynamics_autonomous_v1/verification.json','counterfactual_forecast_frames_replayed',3373056),('reports/dynamics_autonomous_v1/aggregate_verification.json','all_passed',True)],
 'C07':[('reports/C07_synthesis_verification.json','all_passed',True),('reports/C07_milestones.json','C07_complete',True)],
 'D01':[('reports/nearest_methods_review_v2_verification.json','all_passed',True),('reports/nearest_methods_review_v2_verification.json','methods_compared',9)],
 'D02':[('reports/figure_catalog_verification.json','all_passed',True),('reports/figure_catalog_verification.json','group_count',10)],
 'D03':[('reports/final_memo_verification.json','all_passed',True)],
}

ASSESSMENT={
'A01':'원본 전체 1,324파일 해시·크기 확인. 후속 출력 분리. 기존 실행 환경·MPS 학습 기록은 원본과 후속 run에 보존.',
'A02':'원래 train만 fitting에 사용하고 새 validation/test/OOD의 회전 궤도 중복을 제거. 잠긴 설정·자료 해시 확인.',
'A03':'8종 변형의 모양/색/통과율과 4종 음성 대조의 임계값 민감도. 원래 판정기가 변형과 잡음을 받아들이는 한계를 공개.',
'A03b':'calibration에서만 임계값 선택, 독립 test와 기존 생성물 재평가. 사람 검증 아님.',
'A04':'240장과 출처 비공개 HTML·연결표 준비. 실제 사람 응답 0. 브라우저 시각/상호작용 검사도 로컬 URL 정책으로 미실시. 준비를 실제 수집으로 세지 않음.',
'A05':'train fitting/validation 선택, 새 test/OOD, 기존 6k 부분공간 및 3k/6k 같은 입력의 full 공간 비교. 누락됐던 직접 비교는 최종 감사 중 보완.',
'A06':'같은 평가기의 복원/생성 8조건 및 방향별 decoder 분산 분해. 불일치가 곧 속성의 인과 분리라는 주장 없음.',
'A07':'같은 frozen C4 AE별 일반/증강/C4 flow, 9회×4,000 steps; 초기 가중치·표본·평가 noise 짝맞춤.',
'A08':'같은 시간 flow 9회, 실제 수행 step·AE+flow 비용·576장 추론 시간 기록. 같은 파라미터와 같은 연산량은 구분.',
'A09':'새 데이터 3개×초기화 3개, AE 9/flow 27개와 새 장면 중복·결과 재생. 데이터 3개와 noise 조건부 구간을 구분.',
'A10':'같은 예산의 두 새 decoder와 oracle/학습 latent 12회 비교, 생성 전이 9개. 기존 pool의 탐색이며 독립 확인 아님.',
'A11':'표준 MSE 항등식 유도, 512장 검사, invariant-only 복원과 full C4 구분. 새로운 무한 정리 또는 모든 표현의 하한 아님.',
'A12':'유지되는 효과·평가기/decoder 한계·다음 단계 선정 이유 기록. 당시 남은 작업 문구는 최신 연구 메모와 종합 결과로 갱신해 해석.',
'B01':'정답 물체 ID·마스크·상태·RGB를 재현 생성, 분할과 전역 회전 검사.',
'B02':'색/회전/이동, 합성·순서와 변경 외 속성 보존을 정답 쌍에서 확인.',
'B03':'정답 상태·정답 대상 표시, 평평한/물체별 공유 모델 동일 표본·파라미터 대조. 실제 RGB 인식 점수와 구분.',
'B04':'평가 입력 RGB만 사용, slot 배정·복원·가림·물체 수 포함. 축소 고정 모델의 실패 진단이며 원논문 성능 재현 아님.',
'B05':'충분한 slot 용량, 새 640장·2,961명령, 115조건·68,103출력, 목표 변경/비목표 보존/전체 정답을 따로 검증.',
'B06':'공식 preview 500쌍·24모델·공유 test100. 공식 fold·원본 픽셀·중복·노출 감사. 전체 공식 SVIB 및 LPIPS 미실행을 공개.',
'B07':'가림 정도·물체 수·합성 명령·오류 유형별 표/곡선, 고정 첫 예제 영상. 기술 통계를 인과 효과로 일반화하지 않음.',
'C01':'이산 충돌 simulator, 상태/영상과 궤적 split·에너지·겹침 수치 검사. 연속/실제 물리 검증 아님.',
'C02':'상태·행동·외력을 함께 회전한 정답 동치와 외력을 고정할 때의 반례. 정확한 적용 범위 공개.',
'C03':'63개 모델과 등속도·알려진 물리 참조; 정답 상태·환경의 한 단계만 비교. 초기화·표본·재사용 감사.',
'C04':'동일 과거 4장, RGB/측정 모델과 정답 상태·환경 교체, 307,200개 다음 예측. 인식과 정보 추정 분리.',
'C05':'61단계 자율 예측, 미학습 힘/속성/물체 수, 위치·속도·접촉 후보/실패 평가. 물체 색/존재는 복사하므로 학습 추적 성공으로 주장하지 않음.',
'C06':'같은 과거의 반대 행동 정답과 자율 반응을 대상/비대상별 검사. 없는 네 물체 반사실을 만들어내지 않음. 비유한 분모 공개.',
'C07':'세 독립 데이터 반복·108개 정보 누락 조건·72개 길이 모델, 520개 전체 집계, 고정 접촉 집단·정보 모호성 사례와 종합 결론.',
'D01':'9개 가까운 방법 본문·가능한 공식 코드 비교. 공개 코드 없는 방법은 미검토 표시. 타 논문의 전 학습 재현이나 독창성 증명 아님.',
'D02':'원인 분리·계산 비용·조건·개입/조합·긴 미래: 10묶음 25파일. 원자료·수치 검사·이미지 디코딩·명시된 시각 검토 범위 연결.',
'D03':'가설·방법·결과·실패·한계를 담은 짧은 연구 메모와 쉬운 설명. 수치/근거/미실행 제안/사람 미수집을 분리.',
'D04':'원본 불변·전체 항목 감사·별도 ZIP·모든 ZIP 바이트 재검사·다른 경로에서의 CPU 표본 재생. 마지막 두 검사는 봉인 뒤 별도 영수증으로 확인.',
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--release',type=Path);a=p.parse_args()
    status=json.loads((HERE/'status.json').read_text());plan=(HERE/'PLAN_KO.md').read_text()
    ids=re.findall(r'^\| ([ABCD]\d{2}b?) \|',plan,re.M);assert len(ids)==31 and set(ids)==set(status['tasks'])==set(ASSESSMENT)
    items=[];issues=[]
    for key in ids:
        task=status['tasks'][key];checks=[];evidence={}
        for rel in task.get('evidence',[]):
            path=HERE/rel
            if not path.is_file():issues.append({'task':key,'missing':rel})
            else:evidence[rel]=r.sha(path)
        for rel,field,expected in CHECKS.get(key,[]):
            path=HERE/rel;data=json.loads(path.read_text());actual=data[field];passed=actual==expected
            checks.append({'file':rel,'field':field,'expected':expected,'actual':actual,'passed':passed});evidence[rel]=r.sha(path)
            if not passed:issues.append({'task':key,'failed_check':checks[-1]})
        if key=='A04':
            h=json.loads((HERE/'reports/human_review_v1/manifest.json').read_text());assert h['n']==240 and h['human_responses_received']==0
            assessment='incomplete_requires_actual_human_responses'
        elif key=='D04':assessment='incomplete_archive_and_relocated_replay' if a.release is None else 'pending_release_checks'
        else:
            assert checks and all(x['passed'] for x in checks)
            assert task['status']=='complete',key;assessment='complete_with_stated_scope'
        items.append({'id':key,'requirement':task['completion_evidence_required'],'assessment':assessment,
            'coverage_and_limits':ASSESSMENT[key],'specific_checks':checks,'evidence_sha256':evidence})
    audit=json.loads((HERE/'reports/evaluator_audit_v1/summary.json').read_text())
    assert len(audit['transforms'])==8 and set(audit['negative_false_acceptance'])=={'disk','empty','random_noise','rectangle'}
    calibration=json.loads((HERE/'reports/evaluator_calibration_v1/summary.json').read_text())
    for split in ['calibration','test']:
        assert all(calibration[split][k]['accepted_n']==0 for k in ['disk','empty','random_noise','rectangle'])
    diagnostic=json.loads((HERE/'reports/decode_diagnostic_v1/summary.json').read_text());assert len(diagnostic['conditions'])==8
    branches=diagnostic['branches'];assert abs(branches['branch_mse_mean']-branches['ensemble_mse']-branches['branch_disagreement'])<1e-9
    assert len(json.loads((HERE/'reports/probes_6000_v1/summary.json').read_text())['models'])==6
    for tag in ['fixed','time']:
        flows=json.loads((HERE/f'reports/flow_{tag}_summary.json').read_text());assert len(flows['records'])==9
        if tag=='fixed':assert all(x['steps']==4000 for x in flows['records'])
    release_checks={}
    if a.release:
        for name in ['archive_verification.json','extracted_integrity.json','relocated_replay.json']:
            path=a.release/name;data=json.loads(path.read_text());assert data['all_passed'];release_checks[name]={'sha256':r.sha(path),'scope':data['scope']}
        replay=json.loads((a.release/'relocated_replay.json').read_text())
        assert replay['length_models_replayed']==72 and replay['svib_models_replayed']==24
        assert Path(replay['bundle_root']).resolve()!=HERE.parent.resolve()
        next(x for x in items if x['id']=='D04')['assessment']='complete_with_stated_scope'
    if issues:raise AssertionError(issues)
    output=HERE/a.out
    if output.exists():raise RuntimeError('Preserve prior audit; choose a new path')
    report={'audit_checks_passed':True,'full_goal_complete':False,'requirement_count':31,
        'complete_requirements':sum(x['assessment']=='complete_with_stated_scope' for x in items),
        'remaining':[x['id'] for x in items if x['assessment']!='complete_with_stated_scope'],
        'items':items,'release_checks':release_checks,'plan_sha256':r.sha(HERE/'PLAN_KO.md'),'status_at_audit_sha256':r.sha(HERE/'status.json'),
        'source_sha256':r.sha(__file__),'checked_unix':time.time(),
        'scope':'Explicit requirement coverage and authoritative evidence inventory with targeted content checks. Does not repeat every prior training/replay. Prior verifier scope was reviewed and limitations retained. Actual human data is required for A04.'}
    dump(output,report)
    lines=['# 후속 계획 전체 항목 감사','',f"31개 항목 중 {report['complete_requirements']}개는 명시한 범위에서 완료했다. 남은 항목: {', '.join(report['remaining'])}. 전체 목표는 미완료다.",'',
        '| ID | 요구한 증거 | 현재 판정과 실제 범위 |','| --- | --- | --- |']
    for item in items:lines.append(f"| {item['id']} | {item['requirement']} | **{item['assessment']}**. {item['coverage_and_limits']} |")
    lines+=['','검사 범위: 파일이 있다는 사실만으로 완료를 판정하지 않았다. 원래 보고서·verifier 범위를 검토하고 항목별 수치·재생·집계 근거를 연결했다. 이 감사 도구가 전체 모델을 다시 학습한 것은 아니다. 각 자료의 해시와 실제 검사 필드는 연결 JSON에 있다.','',f'[{output.name}]({output.name})']
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print('scope audit',report['complete_requirements'],'/31; remaining',report['remaining'],flush=True)


if __name__=='__main__':main()
