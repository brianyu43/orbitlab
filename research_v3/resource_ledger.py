"""Measured component costs with pilots, partial runs, and overlap kept separate.

This is an accounting snapshot, not a proof of model performance or completion.
No training, selection, source protocols, or historical artifacts are changed.
"""
import csv
import json
import platform
import time
from collections import defaultdict
from importlib.metadata import version
import numpy as np
from v3_common import HERE, ROOT, dump, sha
from r5_intake import TASKS


def read(path):
    return json.loads(path.read_text())


def ordinary_run(path, stage, category):
    value = read(path)
    checkpoints = {}
    if 'checkpoint_sha256' in value:
        names = ['model.pt', 'checkpoint.pt', 'reader.pt']
        matches = [path.parent / name for name in names if (path.parent / name).is_file()]
        assert len(matches) == 1, (path, matches)
        assert sha(matches[0]) == value['checkpoint_sha256']
        checkpoints[str(matches[0].relative_to(HERE))] = value['checkpoint_sha256']
    for epoch, digest in value.get('checkpoints', {}).items():
        target = path.parent / f'epoch{epoch}.pt'
        assert sha(target) == digest
        checkpoints[str(target.relative_to(HERE))] = digest
    return {'stage': stage, 'category': category, 'path': str(path.relative_to(HERE)),
            'receipt_sha256': sha(path), 'steps': value['steps'], 'parameters': value.get('parameters'),
            'training_seconds': value['seconds'], 'validation_seconds': value.get('validation_seconds'),
            'reported_peak_process_RSS_bytes': value.get('max_rss_bytes', value.get('peak_process_rss_bytes', value.get('process_peak_RSS_bytes'))),
            'checkpoint_hashes': checkpoints, 'completed_training_receipt': True,
            'scientific_verification_required_separately': True}


def collect():
    records = []
    # Enumerate each experiment root once. R2 has an explicitly archived failed
    # preflight; retain its cost instead of counting it as scientific training.
    for stage in ('r1', 'r3', 'r2', 'r4', 'r5/disks'):
        for path in sorted((HERE / stage).rglob('run.json')):
            relative = path.relative_to(HERE / stage)
            is_pilot = 'benchmark' in relative.parts or read(path).get('status') == 'benchmark_complete'
            category = 'discarded_pilot' if is_pilot else 'completed_main_training'
            if 'training_preflight_v1' in relative.parts:
                category = 'archived_discarded_pilot'
            records.append(ordinary_run(path, stage, category))
    r5 = HERE / 'r5'
    for path in sorted((r5 / 'external').rglob('run.json')):
        records.append(ordinary_run(path, 'r5/external', 'completed_confirmation_training' if 'confirmation' in path.parts else 'completed_main_training'))
    for path in sorted((r5 / 'attributes').rglob('run.json')):
        records.append(ordinary_run(path, 'r5/measurement', 'completed_reader_training'))
    for path in sorted((r5 / 'attributes').rglob('preflight.json')):
        value = read(path)
        assert value['weights_discarded']
        records.append({'stage': 'r5/measurement', 'category': 'discarded_pilot', 'path': str(path.relative_to(HERE)),
                        'receipt_sha256': sha(path), 'steps': 100, 'parameters': value['parameters'],
                        'training_seconds': value['100step_seconds'], 'completed_training_receipt': False,
                        'scientific_verification_required_separately': True, 'checkpoint_hashes': {}})
    for family, folder in [('dsprites', 'model_preflight_v1'), ('clevr', 'clevr_model_preflight_v1'), ('clevrtex', 'clevrtex_model_preflight_v1')]:
        for kind in ('plain', 'c4', 'object'):
            path = r5 / folder / f'{kind}.json'
            if path.exists():
                value = read(path)
                assert value['steps'] == 100 and value['benchmark_weights_discarded']
                records.append({'stage': 'r5/external', 'category': 'discarded_pilot', 'path': str(path.relative_to(HERE)),
                                'receipt_sha256': sha(path), 'steps': 100, 'parameters': value.get('parameters'),
                                'training_seconds': value['seconds'], 'completed_training_receipt': False,
                                'scientific_verification_required_separately': True, 'checkpoint_hashes': {}})
    # Progress records are atomic snapshots, not frozen run receipts. Do not
    # hash a changing progress checkpoint or mistake a completed file for replay.
    partial = []
    for base, stage in [(r5 / 'external', 'r5/external'), (r5 / 'attributes', 'r5/measurement')]:
        for path in sorted(base.rglob('progress.json')):
            if not (path.parent / 'run.json').exists():
                value = read(path)
                partial.append({'stage': stage, 'path': str(path.relative_to(HERE)), 'steps': value['step'],
                                'measured_seconds': value.get('train_seconds', value.get('seconds')),
                                'complete': False, 'runtime_liveness_not_inferred_from_file': True})
    # A completed receipt must be unique even if modules share trained weights.
    assert len({r['path'] for r in records}) == len(records)
    totals = defaultdict(lambda: {'components': 0, 'updates': 0, 'sum_component_training_seconds': 0.0})
    for value in records:
        item = totals[value['stage'] + '/' + value['category']]
        item['components'] += 1
        item['updates'] += value['steps']
        item['sum_component_training_seconds'] += value['training_seconds']
    measured = {stage: sum(v['steps'] for v in records if v['stage'] == stage and v['category'] == 'completed_main_training')
                for stage in ('r1', 'r3', 'r2', 'r4', 'r5/disks')}
    assert measured == {'r1': 36000, 'r3': 12000, 'r2': 30000, 'r4': 100000, 'r5/disks': 14000}, measured
    selections = []
    for family, task in [('dsprites', task) for task in TASKS] + [('clevr', 'Single_Atomic'), ('clevrtex', 'Single_Atomic')]:
        path = r5 / 'external' / f'{family}_hard' / 'evaluation' / task / 'development_selection.json'
        selections.append({'family': family, 'task': task, 'selection_recorded': path.exists(),
                           'candidate': read(path)['selected_candidate'] if path.exists() else None})
    start = read(r5 / 'training_started.json')['unix']
    stop_path = HERE / 'r5/user_stop.json'
    stop = read(stop_path) if stop_path.exists() else None
    observed_end = stop['stopped_unix'] if stop else time.time()
    size = sum(p.stat().st_size for p in r5.rglob('*') if p.is_file() and p.suffix != '.tmp')
    return {'source_sha256': sha(__file__), 'snapshot_unix': time.time(), 'records': records, 'partial': partial,
            'totals_by_stage_and_category': dict(totals), 'required_main_development_updates_including_readers': 876000,
            'main_development_components_required': 63,
            'main_development_components_completed': sum(v['category'] in ('completed_main_training', 'completed_reader_training') for v in records),
            'main_development_updates_completed': sum(v['steps'] for v in records if v['category'] in ('completed_main_training', 'completed_reader_training')),
            'conditional_confirmation_selections': selections,
            'conditional_confirmation_updates_known_required': 648000 * sum(v['candidate'] is not None for v in selections),
            'confirmation_requirement_fully_known': all(v['selection_recorded'] for v in selections),
            'R5_cap': {'started_unix': start, 'elapsed_wall_seconds': observed_end - start, 'limit_wall_seconds': 43200,
                       'user_requested_stop': bool(stop), 'stopped_unix': observed_end if stop else None,
                       'current_logical_file_bytes_excluding_tmp': size, 'limit_logical_bytes': 30000000000},
            'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'machine': platform.machine(),
                            'package_versions': {name: version(name) for name in ('torch', 'numpy', 'scipy', 'pillow', 'matplotlib')},
                            'root_requirements_sha256': sha(ROOT / 'requirements.lock.txt')},
            'not_in_summed_training_costs': ['data preparation and downloads', 'full inference/evaluation/replay and reports',
                                            'waiting, packaging, and archival hash checks', 'extra five-update CPU/MPS parity checks',
                                            'GPU scheduling benchmarks, separately recorded under r5/gpu_schedule_benchmark'],
            'interpretation': 'Per-process training wall-clock seconds can overlap and include contention/pauses. Their sum is neither elapsed project time nor CPU/GPU energy or utilization. Completed training receipt does not itself prove replay/selection/experiment completion. Partial steps and discarded pilots never count as completed scientific runs.'}


def main():
    value = collect()
    dest = HERE / 'review'
    dump(dest / 'resource_ledger.json', value)
    fields = ['stage', 'category', 'path', 'steps', 'parameters', 'training_seconds', 'validation_seconds', 'reported_peak_process_RSS_bytes', 'receipt_sha256']
    with (dest / 'resource_ledger.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in value['records']:
            writer.writerow({key: row.get(key, '') for key in fields})
    stopped = value['R5_cap']['user_requested_stop']
    lines = ['# 실제 학습 비용 장부', '',
             '**사용자 요청으로 실험을 중단한 시점의 장부다. 미완료 학습은 그대로 남긴다.**' if stopped else '**진행 중인 시각별 장부다. 연구 전체 완료나 모델 성능을 증명하는 파일이 아니다.**', '',
             f'본 개발 학습은 측정기까지 포함해 63개 구성 요소·876,000업데이트가 계획되어 있다. 현재 학습 종료 파일이 있는 구성 요소는 {value["main_development_components_completed"]}개, 업데이트는 {value["main_development_updates_completed"]:,}회다. 모든 종료 파일의 체크포인트 해시를 확인했지만 추론 재생·연구 판정은 별도 검증 파일로 확인해야 한다.', '',
             '| 단계 / 구분 | 구성 요소 | 업데이트 | 각 학습 구간의 벽시계 초 합 |', '|---|---:|---:|---:|']
    for label, total in value['totals_by_stage_and_category'].items():
        lines.append(f'| {label} | {total["components"]} | {total["updates"]:,} | {total["sum_component_training_seconds"]:.1f} |')
    cap = value['R5_cap']
    lines += ['', f'R5 기존 시작 시각 이후 {cap["elapsed_wall_seconds"]/3600:.2f}시간이 지났고 현재 논리 파일 크기는 {cap["current_logical_file_bytes_excluding_tmp"]/1e9:.2f}GB다. 초기 상한은 12시간/30GB이며 진행 파일을 읽은 시각에 따라 값이 달라진다.', '',
              f'현재 잠긴 양성 후보가 요구하는 확인 업데이트는 {value["conditional_confirmation_updates_known_required"]:,}회다. 모든 과제의 후보 선택이 끝나지 않았다면 이 값은 최종 확인 예산이 아니다.', '',
              '숫자를 읽을 때의 구분:', '',
              '- 동시 프로세스의 초를 더해 전체 경과 시간으로 사용하지 않는다. GPU 스케줄링 시험 중 일시 정지와 자원 경쟁도 각 실행의 시간에 반영될 수 있다.',
              '- discarded_pilot은 초기 가중치를 버린 시간 측정이며 본 학습과 합쳐 성공 업데이트 수로 세지 않는다. archived_discarded_pilot도 비용에서 숨기지 않는다.',
              '- 학습이 끝나지 않은 progress는 아래에 별도로 남긴다. 진행 파일만 보고 프로세스가 살아 있다고 판단하지 않는다.',
              '- 데이터 준비·검증·추론·출력 재생·패키징·작업 대기는 위 학습 시간 합에 포함되지 않는다. GPU 병렬 벤치마크 및 CPU/MPS 일치 확인용 업데이트는 별도 기록에 있다.',
              '- 학습량·파라미터·입력 감독·물리 사전 지식이 다른 비교는 같은 계산/정보 예산이라고 부르지 않는다.', '',
              '| 미완료 학습 파일 | 저장된 업데이트 | 저장된 학습 초 |', '|---|---:|---:|']
    for row in value['partial']:
        seconds = '—' if row['measured_seconds'] is None else f'{row["measured_seconds"]:.1f}'
        lines.append(f'| {row["path"]} | {row["steps"]:,} | {seconds} |')
    lines += ['', '전체 구성 요소별 파라미터/시간/체크포인트 출처는 `resource_ledger.csv`와 `resource_ledger.json`에 있다. 현재 환경 버전도 JSON에 보존한다. 새 환경에서 의존성을 설치하고 전부 재학습한 증거로 사용하지 않는다.', '']
    (dest / 'RESOURCE_LEDGER_KO.md').write_text('\n'.join(lines))
    print(json.dumps({key: value[key] for key in ('main_development_components_completed', 'main_development_updates_completed', 'R5_cap')}, indent=2))


if __name__ == '__main__':
    main()
