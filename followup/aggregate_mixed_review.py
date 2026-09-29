"""Aggregate locked visual ratings; preserve human and AI provenance separately."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def score(rows):
    n = len(rows)
    counts = {
        'shape_match': sum(int(x['shape']) == int(x['target_shape']) for x in rows),
        'color_match': sum(int(x['color']) == int(x['target_color']) for x in rows),
        'joint_match': sum(int(x['shape']) == int(x['target_shape']) and int(x['color']) == int(x['target_color']) for x in rows),
        'clear_form': sum(int(x['valid']) == 1 for x in rows),
        'strict_match': sum(int(x['shape']) == int(x['target_shape']) and int(x['color']) == int(x['target_color']) and int(x['valid']) == 1 for x in rows),
        'shape_unknown_or_other': sum(int(x['shape']) == -1 for x in rows),
        'color_unknown_or_mixed': sum(int(x['color']) == -1 for x in rows),
    }
    return {'n': n, 'counts': counts, 'rates': {k: v/n for k, v in counts.items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--key', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    lock = json.loads((a.review/'annotation_lock.json').read_text())
    for name, digest in lock['files_sha256'].items():
        assert sha(a.review/name) == digest, name
    human = read(a.review/'human_original.csv')
    ai = read(a.review/'ai_visual_annotations.csv')
    assert len(human) == 9 and len(ai) == 231
    assert all(x['observer_type'] == 'human_self_reported' for x in human)
    assert all(x['observer_type'] == 'ai_visual_assistant' for x in ai)
    key_rows = read(a.key)
    key = {x['id']: x for x in key_rows}
    assert len(key) == len(key_rows) == 240
    assert {x['id'] for x in human+ai} == set(key)
    assert len({x['id'] for x in human+ai}) == len(human+ai)
    combined = sorted(({**key[x['id']], **x, 'note': x.get('note', '')} for x in human+ai), key=lambda x:x['id'])
    groups = []
    for observer in ['human_self_reported', 'ai_visual_assistant']:
        rows = [x for x in combined if x['observer_type'] == observer]
        groups.append({'observer_type': observer, 'model': 'all', 'split': 'all', **score(rows)})
        for model in sorted({x['model'] for x in rows}):
            for split in ['all', 'seen', 'ood']:
                sub = [x for x in rows if x['model'] == model and (split == 'all' or x['split'] == split)]
                if sub:
                    groups.append({'observer_type': observer, 'model': model, 'split': split, **score(sub)})
    result = {'coverage': {'human':9, 'ai':231, 'total_unique':240}, 'groups':groups,
        'source_sha256': {'human':sha(a.review/'human_original.csv'), 'ai':sha(a.review/'ai_visual_annotations.csv'),
            'annotation_lock':sha(a.review/'annotation_lock.json'), 'private_key':sha(a.key), 'aggregator':sha(__file__)},
        'claim_scope':'Descriptive mixed review of the original 240 development images. AI judgments are not human validation. No independent rater overlap or agreement estimate. Not the later 3-data-seed confirmation pool.'}
    a.out.mkdir(exist_ok=False)
    with (a.out/'ratings_with_targets.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(combined[0]));writer.writeheader();writer.writerows(combined)
    (a.out/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    labels={'human_self_reported':'사람 응답', 'ai_visual_assistant':'AI 시각 판정'}
    lines=['# 사람·AI 시각 판정 결과', '',
        '사용자는 10개까지 판정했다고 알려 주었으며, 실제 내려받은 CSV에는 Q001–Q009의 9개 응답이 저장되어 있었다. 이 9개를 그대로 보존하고 Q010–Q240의 231개는 요청에 따라 AI가 직접 보고 판정했다.', '',
        '모델 이름과 요청 정답을 가린 12개 이미지 묶음을 보고 AI 판정을 먼저 저장·해시 고정한 뒤 정답표와 결합했다. 자동 분류기 출력을 AI 시각 판정으로 바꿔 적지 않았다.', '',
        '| 판정 주체 | 표본 | 모양 일치 | 색 일치 | 모양·색 일치 | 명확한 도형 | 모양·색·명확성 모두 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    metrics=['shape_match','color_match','joint_match','clear_form','strict_match']
    for g in groups:
        if g['model']=='all':
            lines.append('| '+labels[g['observer_type']]+' | '+str(g['n'])+' | '+' | '.join(f"{g['counts'][k]}/{g['n']} ({100*g['rates'][k]:.1f}%)" for k in metrics)+' |')
    lines+=['','사람 9개는 작은 편의 표본이고 AI와 평가한 그림도 다르다. 두 행의 차이를 판정자 정확도나 일치도로 해석하지 않는다. 두 주체를 섞은 수치를 사람 평가 성공률로 제시하지 않는다.','',
        '## AI 판정의 모델·조건별 기술 통계','',
        '| 원래 모델 | 조합 | AI 판정 수 | 모양·색·명확성 모두 |',
        '| --- | --- | ---: | ---: |']
    for g in groups:
        if g['observer_type']=='ai_visual_assistant' and g['model']!='all' and g['split']!='all':
            lines.append(f"| {g['model']} | {g['split']} | {g['n']} | {g['counts']['strict_match']}/{g['n']} ({100*g['rates']['strict_match']:.1f}%) |")
    lines+=['','`aug`와 `equivariant`는 이 240장을 만들었던 기존 생성 조건이다. 이후 실시한 독립 데이터 3개 확인 실험의 C4 15.6%·OOD 11.4%와는 표본·모델·평가자가 다르다. 이번 값을 그 수치의 사람 검증이나 대체값으로 사용하지 않는다.', '',
        '원래 A04의 전체 사람 판정은 사용자의 “나머지는 네가 해라” 요청에 따라 사람·AI 구분 평가로 변경했다. 원래 계획과 기존 ZIP은 그대로 보존한다. **변경된 요청은 이행했으나, 240장 전체의 독립 사람 검증은 수행하지 않았다.**', '',
        '[사용자 응답 원본](../human_original.csv) · [AI 판정과 개별 이유](../ai_visual_annotations.csv) · [정답 공개 전 해시 고정](../annotation_lock.json) · [행별 대조](ratings_with_targets.csv) · [집계 JSON](summary.json)']
    (a.out/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([g for g in groups if g['model']=='all'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
