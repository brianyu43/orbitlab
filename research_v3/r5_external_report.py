"""Render all required external cells, including incomplete and failed ones.

Reporting is descriptive. It never changes frozen selection or training.
Only verified rows enter plots. Resplit means require all planned paired cells.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v3_common import HERE, dump, sha
from r5_intake import BASE
from r5_attribute_evaluate import backend
from r5_external_aggregate import collect

COLORS = {'copy': '#64748b', 'plain': '#d97706', 'c4': '#2563eb', 'object': '#059669'}
PIXELS = ('image_mse', 'changed_scene_changed_pixel_mse', 'preserved_pixel_mse')
MODES = ('validation_selected', 'fixed20epoch')


def title(task):
    return task['family'] + ' / ' + task['task'].replace('_', ' ')


def number(value, digits=6):
    return '—' if value is None else f'{value:.{digits}f}'


def percent(value):
    return '—' if value is None else f'{100 * value:.2f}%'


def scene_readout_counts(value):
    """Exact scene counts from already verified two-object accuracy totals."""
    if not value['attributes_complete']:
        return None
    metrics = value['attribute_metrics']
    assert metrics['object_count'] == 16000
    correct_objects = int(round(16000 * metrics['all_attributes_accuracy']))
    both = int(round(8000 * metrics['both_objects_all_attributes_accuracy']))
    one = correct_objects - 2 * both
    none = 8000 - one - both
    assert min(none, one, both) >= 0 and none + one + both == 8000
    return none, one, both


def unit(task, kind, mode='validation_selected'):
    return next(v for v in task['test_units'] if v['kind'] == kind and v['mode'] == mode)


def save_figure(fig, path, files):
    fig.savefig(path, dpi=165, bbox_inches='tight')
    plt.close(fig)
    files.append(path)


def figures(data, destination):
    files = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, task in zip(axes.flat, data['tasks']):
        shown = 0
        for run in task['runs']:
            if run['complete']:
                values = [run['validation'][str(ep)]['image_mse'] for ep in (5, 10, 20)]
                ax.plot([5, 10, 20], values, marker='o', label=run['kind'], color=COLORS[run['kind']])
                shown += 1
        if shown:
            ax.set_yscale('log')
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, 'Awaiting completed training\nand full validation replay',
                    transform=ax.transAxes, ha='center', va='center', color='#64748b')
        ax.set_title(title(task))
        ax.set_xticks([5, 10, 20])
        ax.set_xlabel('Epoch (57,600 training scenes / epoch)')
        ax.set_ylabel('Validation full-image MSE')
        ax.grid(alpha=.2)
    fig.suptitle('SVIB learning curves: only verified runs; missing curves are pending', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .95])
    save_figure(fig, destination / 'learning_curves.png', files)

    fig, axes = plt.subplots(6, 2, figsize=(12, 15))
    for row, task in enumerate(data['tasks']):
        for col, (metric, label) in enumerate(zip(PIXELS[1:], ['Changed-region error', 'Preserved-region damage'])):
            ax = axes[row, col]
            known = [(k, unit(task, k)) for k in COLORS if unit(task, k)['pixels_complete']]
            if known:
                ax.bar([k for k, _ in known], [v['pixel_metrics'][metric] for _, v in known],
                       color=[COLORS[k] for k, _ in known])
            else:
                ax.text(.5, .5, 'Official test evaluation pending', ha='center', va='center', transform=ax.transAxes, color='#64748b')
                ax.set_xticks([])
            ax.set_title(title(task) + ' | ' + label)
            ax.set_ylabel('MSE (lower is better)')
            ax.grid(axis='y', alpha=.2)
    fig.suptitle('Validation-selected checkpoints | all 8,000 official test scenes retained', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .974])
    save_figure(fig, destination / 'pixel_comparison.png', files)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for ax, task in zip(axes.flat, data['tasks']):
        known = [(k, unit(task, k)) for k in COLORS if unit(task, k)['attributes_complete']]
        if known:
            ax.bar([k for k, _ in known], [100 * v['attribute_metrics']['both_objects_all_attributes_accuracy'] for _, v in known],
                   color=[COLORS[k] for k, _ in known])
        else:
            ax.text(.5, .22, 'Generated-image readout pending', transform=ax.transAxes, ha='center', color='#64748b', fontsize=8)
            ax.set_xticks([])
        if task['reader'] is not None:
            clean = task['reader']['official_test']['target_both_objects_all_attributes_accuracy'] * 100
            ax.axhline(clean, color='#7c3aed', linestyle='--', label=f'Clean target reader: {clean:.1f}%')
            ax.legend(fontsize=8, loc='upper center')
        ax.set_ylim(0, 105)
        ax.set_title(title(task))
        ax.set_ylabel('Both objects / all attributes correct (%)')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Imperfect measurement instrument: dashed line is a reference, not a ceiling', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .95])
    save_figure(fig, destination / 'attribute_readout.png', files)

    for task in data['tasks']:
        available = [k for k in ('plain', 'c4', 'object') if unit(task, k)['pixels_complete']]
        if not available:
            continue
        fig, axes = plt.subplots(2 * len(available), 3, figsize=(8, 3.7 * len(available)), squeeze=False)
        mod = backend(task['family'])
        mod.training.SPLIT_SEED = 890101
        for index, kind in enumerate(available):
            root = mod.unit(task['task'], kind, 0, 'validation_selected')
            rec = json.loads((root / 'summary.json').read_text())
            assert sha(root / 'examples.pt') == rec['examples_sha256']
            samples = torch.load(root / 'examples.pt', map_location='cpu', weights_only=True)
            rows = np.load(root / 'rows.npy')
            changed = np.flatnonzero(rows[:, 3] > 0)
            worst = int(changed[np.argmax(rows[changed, 1])]) if len(changed) else 0
            stored_ids = samples['test_indices'].tolist()
            for offset, scene in enumerate((0, worst)):
                match = stored_ids.index(scene)
                for col, field in enumerate(('source', 'target', 'prediction')):
                    ax = axes[2 * index + offset, col]
                    ax.imshow(samples[field][match].permute(1, 2, 0).numpy().clip(0, 1))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if index == 0 and offset == 0:
                        ax.set_title(field.capitalize())
                    if col == 0:
                        ax.set_ylabel(f'{kind}: scene {scene}\n' + ('Fixed first scene' if offset == 0 else 'Largest changed-pixel error'), fontsize=8)
        fig.suptitle(title(task) + '\nFirst scene and each model\'s own worst; no score exclusions', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, .95])
        save_figure(fig, destination / f'examples_{task["family"]}_{task["task"]}.png', files)
    return files


def confirmation_summary(data):
    result = []
    for task in data['tasks']:
        selection = task['selection']
        if selection is None or selection['selected_candidate'] is None:
            continue
        candidate = selection['selected_candidate']
        cells = [c for c in data['conditional_confirmation'] if c['family'] == task['family'] and c['task'] == task['task']]
        for mode in MODES:
            pairs = []
            for seed in (890201, 890202, 890203):
                for init in (0, 1, 2):
                    pair = {k: next(c for c in cells if c['run']['partition_seed'] == seed and c['run']['initialization'] == init and c['run']['kind'] == k) for k in ('plain', candidate)}
                    selected = {k: next(v for v in c['test_units'] if v['mode'] == mode) for k, c in pair.items()}
                    complete = all(c['run']['complete'] for c in pair.values()) and all(v['complete'] for v in selected.values())
                    value = {'partition_seed': seed, 'initialization': init, 'complete': complete}
                    if complete:
                        value['pixels'] = {k: v['pixel_metrics'] for k, v in selected.items()}
                        value['candidate_minus_plain'] = {metric: selected[candidate]['pixel_metrics'][metric] - selected['plain']['pixel_metrics'][metric] for metric in PIXELS}
                        value['attribute_both_objects'] = {k: v['attribute_metrics']['both_objects_all_attributes_accuracy'] for k, v in selected.items()}
                    pairs.append(value)
            summary = {'family': task['family'], 'task': task['task'], 'candidate': candidate, 'mode': mode, 'pairs': pairs,
                       'all_nine_pairs_complete': all(v['complete'] for v in pairs), 'partition_means': None, 'range_of_partition_means': None,
                       'population_confidence_interval': None, 'new_success_threshold_introduced': False}
            if summary['all_nine_pairs_complete']:
                means = []
                for seed in (890201, 890202, 890203):
                    q = [v for v in pairs if v['partition_seed'] == seed]
                    means.append({'partition_seed': seed, 'initializations': 3,
                                  'candidate_minus_plain': {m: float(np.mean([v['candidate_minus_plain'][m] for v in q])) for m in PIXELS},
                                  'pixels': {k: {m: float(np.mean([v['pixels'][k][m] for v in q])) for m in PIXELS} for k in ('plain', candidate)}})
                summary['partition_means'] = means
                summary['range_of_partition_means'] = {m: {'mean': float(np.mean([v['candidate_minus_plain'][m] for v in means])),
                                                           'min': min(v['candidate_minus_plain'][m] for v in means),
                                                           'max': max(v['candidate_minus_plain'][m] for v in means)} for m in PIXELS}
            result.append(summary)
    return result


def export_csv(data, destination):
    paths = []
    units = [v for task in data['tasks'] for v in task['test_units']]
    units += [v for cell in data['conditional_confirmation'] for v in cell['test_units']]
    path = destination / 'all_test_cells.csv'
    fields = ['family', 'task', 'partition_seed', 'initialization', 'kind', 'mode', 'pixels_complete', 'attributes_complete', 'selected_epoch', *PIXELS,
              'unchanged_scene_mse', 'both_objects_all_attributes_accuracy', 'clean_target_both_objects_accuracy']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for value in units:
            metrics = {**value, **value.get('pixel_metrics', {}), **value.get('attribute_metrics', {})}
            writer.writerow({f: metrics.get(f, '') for f in fields})
    paths.append(path)
    path = destination / 'attribute_metrics.csv'
    with path.open('w', newline='') as stream:
        fields = ['family', 'task', 'partition_seed', 'initialization', 'kind', 'mode', 'attribute', 'all_accuracy', 'changed_accuracy', 'preserved_accuracy', 'changed_count']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for value in units:
            if not value['attributes_complete']:
                continue
            m = value['attribute_metrics']
            for name in m['per_attribute_accuracy']:
                writer.writerow({**{f: value[f] for f in fields[:6]}, 'attribute': name, 'all_accuracy': m['per_attribute_accuracy'][name],
                                 'changed_accuracy': m['changed_attribute_accuracy'][name], 'preserved_accuracy': m['preserved_attribute_accuracy'][name],
                                 'changed_count': m['changed_attribute_counts'][name]})
    paths.append(path)
    return paths


def render(data, destination):
    destination.mkdir(parents=True, exist_ok=True)
    dump(destination / 'source_snapshot.json', data)
    confirmations = confirmation_summary(data)
    dump(destination / 'confirmation_comparison.json', confirmations)
    files = [destination / 'source_snapshot.json', destination / 'confirmation_comparison.json']
    files.extend(export_csv(data, destination))
    files.extend(figures(data, destination))
    complete = data['external_scope_complete']
    stamp = datetime.fromtimestamp(data['snapshot_unix'], timezone.utc).isoformat()
    stopped = (HERE / 'r5/user_stop.json').exists()
    scope_text = '사용자 요청으로 실험을 중단했다. 필수 외부 과제와 확인 실험은 미완료로 남아 있다.' if stopped else ('필수 외부 실험과 재생 검증이 완료됐다.' if complete else '진행 중인 결과다. 전체 실험 완료 보고서가 아니다.')
    lines = ['# R5 외부 과제: 학습량·규칙·시각 복잡도 비교', '',
             f'기준 시각: {stamp}. **{scope_text}**', '',
             f'개발 학습·전체 validation 재생 {data["primary_development_runs_verified"]}/18, 공식 test 픽셀 재생 {data["primary_development_pixel_units_verified"]}/42, 속성 판독 재생 {data["primary_development_attribute_units_verified"]}/42이다. 선택 규칙으로 현재 요구되는 확인 학습은 {data["confirmation_runs_verified"]}/{data["confirmation_runs_required_from_available_selections"]}개 검증됐다. 아직 후보가 선택되지 않은 과제의 확인 필요 여부는 미정이다.', '',
             '미완료 항목을 제외한 평균을 전체 성능으로 제시하지 않는다. 외부 실험 종료는 능력 향상·독립 사람 평가·프로젝트 전체 종료와 다르다.', '',
             '## 실험과 정보 예산', '',
             '- 공식 Hard dSprites의 네 과제, CLEVR Single Atomic, CLEVRTex Single Atomic을 각각 평가한다. 과제별 공식 학습 64,000개 중 57,600개로 학습하고 6,400개로 선택한다. 공식 test 8,000개는 고정한다.',
             '- primary plain/C4/object 모델은 원본 RGB 한 장과 target RGB 학습 손실만 받는다. 정답 속성·위치·마스크는 받지 않는다. plain/C4는 1,266,467개 파라미터와 초기 가중치가 같고, object는 1,081,892개로 파라미터/계산량이 일치하지 않는다.',
             '- 각 모델은 batch 32로 36,000회, 즉 20 epoch를 모두 학습한다. 5/10/20 epoch 중 validation 전체 MSE가 가장 작은 것을 선택한다. 별도로 고정 20 epoch 결과도 모두 남긴다.',
             '- 후보는 validation 전체·변경 영역 MSE가 plain과 입력 복사보다 낮고, 보존 영역 손상이 plain 대비 5% 넘게 악화되지 않아야 한다. 통과 후보 중 전체 MSE가 가장 낮은 하나만 확인 실험으로 보낸다. 공식 test로 후보나 epoch를 선택하지 않는다.',
             '- 확인은 같은 공식 학습 자료의 세 train/validation 재분할 × 세 초기화다. 서로 독립인 세 데이터셋 또는 새로운 공식 test가 아니다. 기존 dSprites Single Atomic test는 앞선 프로젝트에서도 사용했다.', '',
             '## 학습 상태와 선택', '', '| 과제 | 모델 | 전체 학습·검증 | 선택 epoch | 파라미터 | 학습 초 |', '|---|---|---|---:|---:|---:|']
    for task in data['tasks']:
        for run in task['runs']:
            if run['complete']:
                row = ['완료', str(run['selected_epoch']), f'{run["parameters"]:,}', f'{run["training_seconds"]:.1f}']
            else:
                progress = run.get('progress')
                row = [f'{progress["step"]:,}/36,000 업데이트; 검증 대기' if progress else '대기', '—', '—', '—']
            lines.append('| ' + ' | '.join([title(task), run['kind'], *row]) + ' |')
    lines += ['', '학습 초는 해당 실행이 측정한 값이며 검증 시간이 별도로 저장된다. 동시 실행·일시 정지의 영향이 들어가므로 서로 더한 값을 전체 경과 시간이나 독립 하드웨어 속도로 해석하지 않는다.', '', '![검증된 학습 곡선](learning_curves.png)', '',
              '| 과제 | validation으로 선택한 확인 후보 |', '|---|---|']
    for task in data['tasks']:
        choice = '선택 대기' if task['selection'] is None else (task['selection']['selected_candidate'] or '없음: 개발 기준 미달')
        lines.append(f'| {title(task)} | {choice} |')
    lines += ['', '## 공식 test: 모든 과제와 두 checkpoint 방식', '',
              '픽셀은 0–1 RGB이다. 변경 영역은 source/target 픽셀이 실제로 다른 위치이며, 의미가 바뀐 물체와 동일한 정의가 아니다. 작은 영역의 개선이 전체 배경 평균에 가려지지 않도록 전체·변경·보존 MSE를 분리한다.', '',
              '| 과제 | 방식 | 모델 | epoch | 전체 MSE | 변경 영역 MSE | 보존 영역 MSE | 두 물체 모든 속성 판독 |', '|---|---|---|---:|---:|---:|---:|---:|']
    for task in data['tasks']:
        for value in task['test_units']:
            values = [number(value['pixel_metrics'][m]) for m in PIXELS] if value['pixels_complete'] else ['대기'] * 3
            score = percent(value['attribute_metrics']['both_objects_all_attributes_accuracy']) if value['attributes_complete'] else '대기'
            epoch = value.get('selected_epoch')
            lines.append('| ' + ' | '.join([title(task), value['mode'], value['kind'], '—' if epoch is None else str(epoch), *values, score]) + ' |')
    lines += ['', '![변경·보존 영역 비교](pixel_comparison.png)', '',
              '## 자동 채점기를 함께 검증하기', '',
              '속성 판독기는 primary 모델과 별도로 정답 속성 감독을 받아 학습했다. 평가 때 정답 source 중심의 112px crop을 사용하므로 위치를 스스로 찾는 모델도, 완벽한 의미 판정기도 아니다. 같은 판독기를 모든 생성 모델에 사용한다. 실제 정답 그림을 읽었을 때의 정확도를 아래에 공개한다.', '',
              '| 과제 | validation 정답 target 물체별 모든 속성 | test 정답 target 물체별 모든 속성 | test 정답 target 두 물체 동시 |', '|---|---:|---:|---:|']
    for task in data['tasks']:
        reader = task['reader']
        values = ['대기'] * 3 if reader is None else [percent(reader['validation']['target']['all_attributes_accuracy']), percent(reader['official_test']['target']['all_attributes_accuracy']), percent(reader['official_test']['target_both_objects_all_attributes_accuracy'])]
        lines.append('| ' + ' | '.join([title(task), *values]) + ' |')
    lines += ['', '![채점기의 정답 그림 판독 정확도](attribute_readout.png)', '',
              '보라색 점선은 정답 그림에서의 판독 정확도이며 생성 점수의 수학적 상한은 아니다. 이 수치로 생성 점수를 나누어 보정하지 않는다. 정답을 제대로 읽은 물체/장면의 고정 부분집합과 그 여집합을 추가 진단으로 남기며, 쉬운 부분집합의 결과를 전체 성능으로 바꾸지 않는다. 깨끗한 정답을 읽을 수 있어도 생성물의 왜곡은 잘못 읽을 수 있다.', '',
              '속성별 전체·변경·보존 정확도와 변경 개수는 `attribute_metrics.csv`, 모든 평가 조건은 `all_test_cells.csv`, 부분집합 및 의미 변화별 픽셀 결과는 `source_snapshot.json`에 있다. 학습 지원 조합과 새 조합의 판독 차이는 [사후 분포 진단](../READER_DISTRIBUTION_AUDIT_KO.md)에 따로 기록한다.', '',
              '### 물체 하나의 평균과 장면 전체 성공을 구별하기', '',
              '픽셀 오차나 물체별 평균이 개선되어도 두 물체를 동시에 맞히는 비율은 낮아질 수 있다. 아래는 validation-selected 모델의 8,000장 전체에서 모든 속성을 맞게 읽은 물체가 0/1/2개인 장면 수다. 자동 판독 결과이며 독립 사람의 정답 판정으로 해석하지 않는다.', '',
              '| 과제 | 모델 | 물체별 모든 속성 정확도 | 0개 맞은 장면 | 1개 맞은 장면 | 2개 맞은 장면 | clean-correct 고정 장면 수 | 그 부분집합의 두 물체 정확도 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for task in data['tasks']:
        for kind in COLORS:
            value = unit(task, kind)
            counts = scene_readout_counts(value)
            if counts is None:
                values = ['대기'] * 6
            else:
                subset = value['clean_calibration_diagnostic']['both_clean_correct_scenes']
                values = [percent(value['attribute_metrics']['all_attributes_accuracy']), *(f'{v:,}' for v in counts),
                          f'{subset["scenes"]:,}', percent(subset['both_objects_all_attributes_accuracy'])]
            lines.append('| ' + ' | '.join([title(task), kind, *values]) + ' |')
    lines += ['', '마지막 열은 깨끗한 정답의 두 물체를 모두 제대로 읽은 동일한 장면 부분집합에 한정한다. 모든 모델에 같은 membership을 적용하고 전체 결과·여집합도 보존한다. 이 부분집합에서도 생성물 판독 오류가 가능하므로 인과적 원인이나 보정된 전체 정확도를 주장하지 않는다. 이 분해로 잠긴 후보·기준을 바꾸지 않는다.', '',
              '## 확인 반복: 동일 공식 자료의 재분할', '']
    if not confirmations:
        lines.append('현재 잠긴 양성 후보에 대한 확인 결과가 없다. 선택 대기와 개발 기준 미달은 앞 표에서 구분한다.')
    for result in confirmations:
        lines += ['', f'### {title(result)} / {result["candidate"]} / {result["mode"]}', '',
                  '| 재분할 seed | 초기화 | 상태 | 전체 MSE 차이 | 변경 MSE 차이 | 보존 MSE 차이 |', '|---:|---:|---|---:|---:|---:|']
        for pair in result['pairs']:
            values = [number(pair['candidate_minus_plain'][m]) for m in PIXELS] if pair['complete'] else ['대기'] * 3
            lines.append('| ' + ' | '.join([str(pair['partition_seed']), str(pair['initialization']), '완료' if pair['complete'] else '미완료', *values]) + ' |')
        if result['all_nine_pairs_complete']:
            lines += ['', '차이는 후보−plain이며 음수일수록 오차가 감소한다. 세 초기화 평균을 재분할별로 먼저 구한 뒤 세 재분할 평균과 범위를 보고한다. 세 값으로 정밀한 모집단 신뢰구간을 주장하지 않는다.', '',
                      '| MSE 종류 | 세 재분할 평균 차이 | 재분할 평균의 최소 | 최대 |', '|---|---:|---:|---:|']
            for metric, value in result['range_of_partition_means'].items():
                lines.append('| ' + ' | '.join([metric, number(value['mean']), number(value['min']), number(value['max'])]) + ' |')
        else:
            lines += ['', '예정된 9개 짝이 모두 완료되지 않아 확인 평균과 범위를 계산하지 않았다.']
    lines += ['', '## 원자료 감사와 해석 범위', '',
              '- dSprites Multiple Non-Atomic은 세계 전체를 90도 회전하면 규칙 라벨이 변한다. 이 과제에서 정확한 C4 구조의 실패가 학습량 부족만을 뜻하지 않는다. 규칙의 대칭성은 데이터 감사에 별도로 기록했다.',
              '- dSprites 일부 원본 target 마스크는 그림의 겹침 순서와 어긋난다. 원본을 고치지 않았고 primary 학습은 마스크를 사용하지 않는다.',
              '- CLEVR/CLEVRTex는 그림자·조명과 작은 렌더링 차이가 있어 의미 변화와 RGB 변화가 다르다. CLEVRTex에서 속성이 같은 test 1,424개 중 1,402개는 작은 RGB 차이를 갖는다. 원래 픽셀 지표와 의미 변화별 집계를 모두 보존한다.',
              '- 모든 개발 test 셀은 8,000개 원본 순서를 유지한다. 성공 사례만 골라 평균하지 않는다. 체크포인트·출력·지표·속성 판독 재생은 독립 재학습 또는 독립 사람 평가를 대신하지 않는다.',
              '- 데이터 출처·라이선스·분할·마스크 문제는 [원자료 감사](../DATA_AUDIT_KO.md), 같은 움직이는 원형 물체에서의 인식·전이·디코더 교체는 [통합 실험](../disks/RESULTS_KO.md)에 있다.', '',
              '## 고정 첫 사례와 실패 사례', '',
              '그림은 공식 test 첫 장면과 각 모델의 변경 영역 MSE가 가장 큰 장면이다. 모델마다 최악 장면이 다를 수 있다. 사례 선택은 설명용이며 수치에서 어떤 장면도 제외하지 않는다.']
    for path in files:
        if path.name.startswith('examples_'):
            lines += ['', f'![{path.stem}]({path.name})']
    if not any(p.name.startswith('examples_') for p in files):
        lines += ['', '전체 픽셀 재생을 통과한 생성 사례를 기다리고 있다.']
    lines += ['', '## 재생 명령과 종료 조건', '',
              '프로젝트 루트의 기존 `.venv`에서 다음을 실행한다. `--require-complete`는 남은 셀이 있으면 실패하여 완료 보고서 생성을 막는다.', '',
              '```bash', '.venv/bin/python research_v3/r5_external_report.py', '.venv/bin/python research_v3/r5_external_report.py --require-complete', '```', '',
              '초기 R5 상한은 기존 시작 시각부터 12시간/30GB이다. 상한 때문에 미완료가 남으면 그 셀과 저장된 진행 상태를 공개한다. GPU 병렬 처리는 epoch·batch·표본 순서·정밀도·평가 기준을 바꾸지 않는다. 이 문서에 독립 사람 응답을 추가하지 않았으며 AI 판독을 사람 평가로 세지 않는다.', '']
    path = destination / 'RESULTS_KO.md'
    path.write_text('\n'.join(lines))
    files.append(path)
    dump(destination / 'report_receipt.json', {'source_sha256': sha(__file__), 'aggregate_source_sha256': data['aggregate_source_sha256'],
                                             'source_snapshot_sha256': sha(destination / 'source_snapshot.json'), 'external_scope_complete': complete,
                                             'files': {p.name: sha(p) for p in files}, 'visual_reviewed': False,
                                             'scope': 'Descriptive report of verified artifacts; no checkpoint/model/gate changes and no independent human evaluation.'})
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    data = collect(args.require_complete)
    print(render(data, BASE / 'external_report'), flush=True)
