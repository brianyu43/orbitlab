"""Post-hoc clean-reader diagnosis; no fitting or changes to official scores.

Compare true and predicted attribute combinations against the reader's actual
training pool. This is descriptive association, not a causal shortcut test.
"""
import json
import time
import numpy as np
from v3_common import dump, sha
from r5_intake import BASE, TASKS
from r5_attributes import folder, audit_folder, load_factors, schema


def encode(rows, counts):
    return np.ravel_multi_index(np.asarray(rows).T, counts)


def analyze(family, task):
    root = folder(family, task)
    needed = [root / f'calibration_{split}_verification.json' for split in ('val', 'test')]
    if not all(p.exists() for p in needed):
        return {'family': family, 'task': task, 'complete': False}
    names, counts = schema(family)
    factors, _ = load_factors(family, task)
    partition = np.load(audit_folder(family, task) / 'partition.npz')
    training = factors[partition['train']].reshape(-1, len(names))
    support = np.bincount(encode(training, counts), minlength=int(np.prod(counts)))
    records = []
    for split in ('val', 'test'):
        summary_path = root / f'calibration_{split}.json'
        receipt = json.loads((root / f'calibration_{split}_verification.json').read_text())
        cal = json.loads(summary_path.read_text())
        assert receipt['summary_sha256'] == sha(summary_path) and receipt['all_rows_bitexact']
        assert cal['predictions_sha256'] == sha(root / f'calibration_{split}.npz')
        arrays = np.load(root / f'calibration_{split}.npz')
        for role, name in enumerate(('source', 'target')):
            selected = arrays['items'][:, 1] == role
            truth = arrays['truth'][selected]
            prediction = arrays['prediction'][selected]
            truth_codes, predicted_codes = encode(truth, counts), encode(prediction, counts)
            eq = truth == prediction
            supported = support[truth_codes] > 0
            groups = {}
            for group, mask in [('all', np.ones(len(truth), bool)), ('training_supported_combinations', supported), ('unseen_combinations', ~supported)]:
                groups[group] = {'objects': int(mask.sum()), 'all_attributes_accuracy': float(eq.all(1)[mask].mean()) if mask.any() else None,
                                 'per_attribute_accuracy': {attribute: float(eq[mask, i].mean()) if mask.any() else None for i, attribute in enumerate(names)},
                                 'prediction_in_training_support_fraction': float((support[predicted_codes[mask]] > 0).mean()) if mask.any() else None}
            matrices = {attribute: np.bincount(truth[:, i] * count + prediction[:, i], minlength=count * count).reshape(count, count).tolist()
                        for i, (attribute, count) in enumerate(zip(names, counts))}
            combinations = [{'class_ids': np.array(np.unravel_index(code, counts)).tolist(), 'training_pool_objects': int(support[code]),
                             'evaluation_objects': int((truth_codes == code).sum()), 'all_attributes_accuracy': float(eq.all(1)[truth_codes == code].mean())}
                            for code in np.unique(truth_codes)]
            records.append({'split': split, 'role': name, 'groups': groups, 'truth_row_prediction_column_confusions': matrices,
                            'per_combination': combinations, 'calibration_sha256': sha(summary_path),
                            'verification_sha256': sha(root / f'calibration_{split}_verification.json')})
    return {'family': family, 'task': task, 'complete': True, 'attribute_names': names, 'class_counts': counts,
            'training_families': len(partition['train']), 'training_pool_objects_both_roles': len(training),
            'supported_combinations': int((support > 0).sum()), 'possible_combinations': len(support), 'records': records,
            'factor_sha256': sha(audit_folder(family, task) / 'factors.npz'), 'partition_sha256': sha(audit_folder(family, task) / 'partition.npz')}


def main():
    tasks = [('dsprites', task) for task in TASKS] + [('clevr', 'Single_Atomic'), ('clevrtex', 'Single_Atomic')]
    records = [analyze(family, task) for family, task in tasks]
    result = {'source_sha256': sha(__file__), 'snapshot_unix': time.time(), 'records': records,
              'post_hoc': True, 'no_model_or_reader_retraining': True, 'primary_scores_or_selection_modified': False,
              'scope': 'Diagnose clean-reader distribution shift with all rows. Training support includes source and target objects from only 57600 development families. All pairwise confusions and observed combinations retained. Associations do not establish a causal mechanism; no evaluation correction or exclusion.'}
    output = BASE / 'reader_distribution_audit.json'
    dump(output, result)
    lines = ['# 자동 채점기의 조합 일반화 진단', '',
             '**사후 진단이다. 학습·선택·공식 점수·시험 표본은 바꾸지 않았다.** 정답 그림을 채점하는 모델 자체가 새 속성 조합에서 실패하는지 조사했다.', '',
             '학습 지원 조합은 실제 판독기 학습 풀 57,600장에 나타난 source/target 두 물체의 속성 튜플이다. 개별 속성이 학습에 있었다는 것과 속성 조합이 있었다는 것은 다르다. 아래 수치는 실제 정답 target 그림을 읽은 결과이며 생성기의 정확도가 아니다.', '',
             '| 과제 | 학습 지원 조합/가능 조합 | validation 물체 전체 정확도 | test 지원 조합 물체 수·정확도 | test 새 조합 물체 수·정확도 | 새 조합의 예측이 학습 조합인 비율 |',
             '|---|---:|---:|---:|---:|---:|']
    def pct(value):
        return '—' if value is None else f'{100 * value:.2f}%'
    for task in records:
        label = task['family'] + '/' + task['task']
        if not task['complete']:
            lines.append(f'| {label} | 대기 | 대기 | 대기 | 대기 | 대기 |')
            continue
        val = next(v for v in task['records'] if v['split'] == 'val' and v['role'] == 'target')['groups']
        test = next(v for v in task['records'] if v['split'] == 'test' and v['role'] == 'target')['groups']
        seen, unseen = test['training_supported_combinations'], test['unseen_combinations']
        values = [label, f'{task["supported_combinations"]}/{task["possible_combinations"]}', pct(val['all']['all_attributes_accuracy']),
                  f'{seen["objects"]:,} · {pct(seen["all_attributes_accuracy"])}', f'{unseen["objects"]:,} · {pct(unseen["all_attributes_accuracy"])}', pct(unseen['prediction_in_training_support_fraction'])]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '학습에 없던 정답 조합을 학습 조합으로 많이 읽는 현상은 조합 일반화 실패와 일치한다. 다만 이 표만으로 색·모양 중 어느 시각적 단서를 지름길로 썼는지, 분포 변화의 유일한 원인이 무엇인지 입증하지 않는다. 가림·크기·위치·공동 출현 등이 함께 달라질 수 있다.', '',
              '판독기가 정답 그림도 잘 읽지 못하는 과제에서는 낮은 생성 판독 점수를 생성 실패율로 단정할 수 없다. 원래 픽셀 지표를 유지하고 판독기 성능을 함께 공개한다. 신뢰 가능한 다음 평가기는 새로운 독립 검증 자료에서 조합 지원·가림·위치 이동을 시험하고 고정해야 한다. 이번 test를 본 뒤 판독기를 고쳐 같은 test 성능을 독립 검증으로 재포장하지 않는다.', '',
              '모든 속성별 혼동 행렬과 조합별 개수·정확도는 `reader_distribution_audit.json`에 있다. 희소한 조합의 불확실성 때문에 이 진단을 새로운 성공 기준이나 모집단 추정으로 사용하지 않는다.', '']
    report = BASE / 'READER_DISTRIBUTION_AUDIT_KO.md'
    report.write_text('\n'.join(lines))
    dump(BASE / 'reader_distribution_audit_receipt.json', {'source_sha256': sha(__file__), 'data_sha256': sha(output), 'report_sha256': sha(report),
                                                        'completed_readers': sum(r['complete'] for r in records), 'required_readers': 6})
    print(report, flush=True)


if __name__ == '__main__':
    main()
