"""Requirement/evidence audit for the whole next-research plan, without shrinking it.

Hash-checks completed studies' saved replay evidence and inspects full external
accounting. Does not run another training cycle or silently waive pending work.
"""
import argparse
import json
import time
from v3_common import HERE, ROOT, dump, sha
from r5_external_aggregate import collect as external_collect


def read(path):
    return json.loads(path.read_text())


def check(path, expected, checked):
    actual = sha(path)
    assert actual == expected, f'Changed evidence: {path}'
    checked[str(path.relative_to(ROOT))] = actual


def note(path, checked):
    checked[str(path.relative_to(ROOT))] = sha(path)
    return read(path)


def report(root, checked):
    receipt = note(root / 'report_receipt.json', checked)
    if 'report_sha256' in receipt:
        check(root / 'RESULTS_KO.md', receipt['report_sha256'], checked)
    if 'figure_sha256' in receipt:
        check(root / 'results.png', receipt['figure_sha256'], checked)
    for key in ('figures', 'files'):
        for name, digest in receipt.get(key, {}).items():
            check(root / name, digest, checked)
    if 'visual_verification_sha256' in receipt:
        check(root / 'visual_verification.json', receipt['visual_verification_sha256'], checked)
    assert receipt.get('visual_review') == 'verified' or receipt.get('visually_reviewed') is True or receipt.get('visual_inspection', '').startswith('passed')


def completed_development(checked):
    rows = []
    for stage in ('r1', 'r3', 'r2', 'r4'):
        root = HERE / stage
        value = note(root / 'verification.json', checked)
        selection = note(root / 'selection.json', checked)
        gate = selection['gate_passed'] if stage == 'r4' else selection['development_gate_passed']
        assert gate is False, 'A positive gate requires its own confirmation audit'
        for name, digest in value.get('receipts', {}).items():
            check(root / name, digest, checked)
        if 'report_receipt_sha256' in value:
            check(root / 'report_receipt.json', value['report_receipt_sha256'], checked)
        if 'protocol_sha256' in value:
            check(root / 'protocol.json', value['protocol_sha256'], checked)
        for key in ('sources', 'source_hashes'):
            for name, digest in selection.get(key, {}).items():
                check(root / name, digest, checked)
        report(root, checked)
        if stage == 'r1':
            assert value['verified_units'] == value['expected_units'] == len(value['conditions']) == 42
            for unit in value['conditions']:
                check(root / unit['condition'] / 'summary.json', unit['summary_sha256'], checked)
            coverage = {'conditions': 42, 'model_and_template_inferences_replayed': sum(v['all_inferences_replayed'] for v in value['conditions']),
                        'edit_rows_recomputed': sum(v['all_edit_rows_recomputed'] for v in value['conditions']),
                        'renderers': 3, 'symmetry_pairs': value['symmetry_pairs']}
        elif stage == 'r3':
            assert value['verified_units'] == value['expected_units'] == len(value['units']) == 64
            for unit in value['units']:
                check(root / unit['path'] / 'summary.json', unit['summary_sha256'], checked)
            coverage = {'conditions': 64, 'full_128step_rollouts_replayed': sum(v['full_128_step_rollouts_replayed'] for v in value['units']),
                        'metric_rows_recomputed': sum(v['rows_recomputed'] for v in value['units']),
                        'cross_historical_orbit_overlap': value['cross_historical_overlap']}
        elif stage == 'r2':
            reader = read(root / 'reader_verification.json')
            decoder = read(root / 'decoder/verification.json')
            assert reader['units'] == reader['expected_units'] == 32 and reader['gate_recomputed']
            assert decoder['verified_units'] == decoder['expected_units'] == 36
            for unit in decoder['conditions']:
                target = root / 'decoder' / unit['condition']
                check(target / 'summary.json', unit['summary_sha256'], checked)
                check(target / 'verification.json', unit['verification_sha256'], checked)
            coverage = {'reader_conditions': 32, 'reader_inferences_replayed': value['full_reader_replays'],
                        'decoder_conditions': 36, 'decoder_RGB_replayed': value['full_decoder_output_replays'],
                        'additional_diagnosis': 'diagnosis/verification.json', 'legacy_reference': 'legacy_palette/verification.json'}
        else:
            assert value['full_updates'] == 100000 and value['model_reconstruction_images_replayed'] == 24576 and value['generated_images_replayed'] == 12288
            check(root / 'RESULTS_KO.md', value['report_sha256'], checked)
            coverage = {key: value[key] for key in ('full_updates', 'model_reconstruction_images_replayed', 'generated_images_replayed')}
        rows.append({'requirement': stage.upper(), 'status': 'development_complete_gate_failed',
                     'confirmation_not_triggered': True, 'ability_goal_achieved': False,
                     'coverage': coverage, 'evidence': str((root / 'verification.json').relative_to(HERE)),
                     'interpretation': 'Complete planned development and failure analysis; failed gate prevents automatic 3x3 expansion. No broader capability or independent replication claim.'})
    root = HERE / 'r5/disks'
    closure = note(root / 'closure.json', checked)
    for name in ('verification', 'report_receipt', 'visual_verification'):
        check(root / f'{name}.json', closure[f'{name}_sha256'], checked)
    value = read(root / 'verification.json')
    for name in ('data_verification', 'training_verification', 'evaluation_protocol'):
        check(root / f'{name}.json', value[f'{name}_sha256'], checked)
    check(root / 'development_gate.json', value['gate_sha256'], checked)
    for name, digest in value['unit_verifications'].items():
        check(root / name, digest, checked)
    assert len(value['unit_verifications']) == 57 and value['all_development_units_complete']
    assert closure['development_complete'] and not closure['gate_passed']
    assert value['full_trajectory_replays'] == 13824 and value['full_RGB_replays'] == 124416
    report(root, checked)
    rows.append({'requirement': 'R5_matched_disk_integration', 'status': 'development_complete_gate_failed',
                 'confirmation_not_triggered': True, 'ability_goal_achieved': False,
                 'coverage': {'data_families': 1440, 'training_components': 5, 'verified_units': 57,
                              'full_trajectories_replayed': 13824, 'full_RGB_replayed': 124416},
                 'evidence': 'r5/disks/closure.json', 'interpretation': 'Matched data/component retraining and full swaps completed; restricted black-background circle task, not original textured polygon perception.'})
    return rows


def protocol_sources(checked):
    paths = ['r1/protocol.json', 'r3/protocol.json', 'r2/training_protocol.json', 'r2/decoder/protocol.json',
             'r4/training_protocol.json', 'r5/disks/training_protocol.json', 'r5/dsprites_training_protocol.json',
             'r5/clevr_training_protocol.json', 'r5/clevrtex_training_protocol.json']
    sources = {}
    for name in paths:
        protocol = note(HERE / name, checked)
        for relative, digest in protocol['sources'].items():
            path = HERE / relative if '/' not in relative else ROOT / relative
            check(path, digest, checked)
            sources[str(path.relative_to(ROOT))] = digest
    return sources


def main(require_complete=False):
    checked = {}
    stopped = (HERE / 'r5/user_stop.json').exists()
    if stopped:
        note(HERE / 'r5/user_stop.json', checked)
        note(HERE / 'review/stopped_checkpoint_verification.json', checked)
        preservation = note(HERE / 'prior_integrity_user_stop.json', checked)
        assert all(row['no_new_drift'] for row in preservation['manifests'])
    plan = note(HERE / 'source_plan.json', checked)
    check(ROOT / plan['path'], plan['sha256'], checked)
    rows = completed_development(checked)
    source_locks = protocol_sources(checked)
    external = external_collect(False)
    snapshot = HERE / 'r5/external_scope_snapshot.json'
    note(snapshot, checked)
    rows.append({'requirement': 'R5_external_development_and_conditional_confirmation',
                 'status': 'complete' if external['external_scope_complete'] else 'incomplete',
                 'coverage': {key: external[key] for key in ('primary_development_runs_verified', 'primary_development_runs_required',
                                                            'primary_development_pixel_units_verified', 'primary_development_pixel_units_required',
                                                            'primary_development_attribute_units_verified', 'confirmation_runs_required_from_available_selections',
                                                            'confirmation_runs_verified')},
                 'pending_count': len(external['pending']), 'evidence': str(snapshot.relative_to(HERE)),
                 'interpretation': 'Every task/arm/mode and all validation-gated confirmation cells retained. Conditional confirmation workload remains unknown until all six development selections exist.'})
    rows.extend([
        {'requirement': 'independent_human_applicability_and_actual_evaluation', 'status': 'unresolved',
         'interpretation': 'Determine applicability after improved results. No new independent human responses exist; AI or prior participant answers do not fulfill this requirement.'},
        {'requirement': 'final_next_study_plan', 'status': 'draft_pending_final_R5_results', 'evidence': 'NEXT_STUDY_DRAFT_KO.md'},
        {'requirement': 'final_information_and_cost_ledger', 'status': 'user_stop_snapshot' if stopped else 'snapshot_only_while_training',
         'evidence': ['REVIEW_INDEX_KO.md', 'review/resource_ledger.json']},
        {'requirement': 'final_historical_preservation_audit', 'status': 'verified_at_user_stop' if stopped else 'pending_final_repeat',
         'evidence': 'prior_integrity_user_stop.json' if stopped else 'verify_preservation.py --output prior_integrity_final.json'},
        {'requirement': 'final_review_package_and_reproduction_instructions', 'status': 'reproduction_guide_saved_final_bundle_not_built' if stopped else 'in_preparation',
         'interpretation': 'Final package must bind all required results, code, weights, data provenance, costs, failures, and replay commands; not the older public Git snapshot.'},
    ])
    result = {'source_sha256': sha(__file__), 'snapshot_unix': time.time(), 'requirements': rows,
              'checked_file_hashes': checked, 'frozen_training_source_hashes_checked': source_locks,
              'completion_proven': False, 'independent_retraining': False, 'stopped_by_user': stopped,
              'scope': 'Current-plan evidence audit: checks linked prior full-replay receipts/results and current full external accounting. Does not repeat all training/inference or substitute hash matches for an independent scientific replication. Missing final requirements remain explicit.'}
    root = HERE / 'review'
    dump(root / 'scope_audit_snapshot.json', result)
    lines = ['# 원계획 항목별 검토 현황', '', '**전체 완료는 아직 입증되지 않았다.** 아래는 원계획의 범위를 유지한 점검표다. 실패한 개발 단계의 종료와 목표 능력 달성을 구별한다.', '',
             '| 원계획 요구 | 현재 증거의 상태 | 근거 |', '|---|---|---|']
    for row in rows:
        evidence = row.get('evidence', '실제 참여·적용 여부 확인 필요')
        if isinstance(evidence, list):
            evidence = ', '.join(evidence)
        lines.append(f'| {row["requirement"]} | {row["status"]} | {evidence} |')
    lines += ['', f'이번 점검은 {len(checked)}개 연결 파일의 해시와 {len(source_locks)}개 학습 소스 잠금을 확인했다. R1/R3/R2/R4 및 원형 물체 통합의 전수 재생 영수증과 현재 결과 연결을 검사했다. 전부를 독립적으로 다시 학습했다는 뜻은 아니다.', '',
              f'외부 R5 필수 작업 {len(external["pending"])}개가 아직 대기 중이다. 각 셀은 `../r5/external_scope_snapshot.json`에 남아 있다. 상한 또는 실패 때문에 남는 셀도 전체 평균에서 제외해 완료로 바꾸지 않는다.', '',
              ('사용자 요청으로 실험을 중단했다. 원본 보존 감사와 재현 안내는 중단 시점에 저장했다. 최종 사람 평가·새 연구 계획 확정·전체 검토 패키지는 미완료다. 자동 재개하지 않는다.' if stopped else '최종 사람 평가, 새 연구 계획 확정, 마지막 원본 보존 감사와 검토 패키지는 별도 완료 증거가 필요하다. 현재 파일의 부재를 “필요 없음”으로 해석하지 않는다.'), '']
    (root / 'SCOPE_AUDIT_KO.md').write_text('\n'.join(lines))
    print(json.dumps({'files_hash_checked': len(checked), 'source_locks_checked': len(source_locks),
                      'external_pending_items': len(external['pending']), 'completion_proven': False}, indent=2))
    if require_complete:
        raise RuntimeError('Whole-plan closure is unproven; final requirements remain explicit in scope_audit_snapshot.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    main(args.require_complete)
