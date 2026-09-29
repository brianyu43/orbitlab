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
    labels={'human_self_reported':'People\'s response', 'ai_visual_assistant':'AI visual judgment'}
    lines=['# Human·AI visual judgment results', '',
        'The user reported that they had determined up to 10 items, and the actual CSV file downloaded contained 9 responses for Q001–Q009. These 9 items were preserved as-is, while the 231 responses for Q010–Q240 were directly reported and determined by AI based on the user\'s request.', '',
        'After viewing the 12 image sets containing the model name and the requested answer, we first saved and hashed the AI judgment, then combined it with the answer sheet. The output from the automatic classifier was converted into an AI visual judgment and not used for scoring.', '',
        '| Judging subject | Sample | Shape match | Color match | Shape·color match | Clear figure | Shape·color·clarity all |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    metrics=['shape_match','color_match','joint_match','clear_form','strict_match']
    for g in groups:
        if g['model']=='all':
            lines.append('| '+labels[g['observer_type']]+' | '+str(g['n'])+' | '+' | '.join(f"{g['counts'][k]}/{g['n']} ({100*g['rates'][k]:.1f}%)" for k in metrics)+' |')
    lines+=['','Nine people are small convenience samples, and the drawings evaluated by AI are also different. They do not interpret the difference between the two rows as the judge\'s accuracy or consistency. They do not present the values mixed between the two subjects as the success rate of human evaluation.','',
        '## Technical statistics by model and conditions of AI judgment','',
        '| Original model | Combination | AI judgment score | Shape, color, and clarity all |',
        '| --- | --- | ---: | ---: |']
    for g in groups:
        if g['observer_type']=='ai_visual_assistant' and g['model']!='all' and g['split']!='all':
            lines.append(f"| {g['model']} | {g['split']} | {g['n']} | {g['counts']['strict_match']}/{g['n']} ({100*g['rates']['strict_match']:.1f}%) |")
    lines+=['','`aug` and `equivariant` are the existing generation conditions used to create these 240 samples. They differ from the C4 15.6% and OOD 11.4% results from the subsequent independent data verification experiment with three samples. These values are not used as human verification or as an alternative value for those figures.', '',
        'Originally, the entire person detection for A04 was changed to a person-AI distinction evaluation based on the user\'s "You handle the rest" request. The original plan and existing ZIP files are preserved as they are. **The modified request was fulfilled, but the independent person verification for all 240 pages was not performed.**', '',
        '[User response original](../human_original.csv) · [AI decision and individual reasons](../ai_visual_annotations.csv) · [Hash fixed before answer disclosure](../annotation_lock.json) · [Comparison by row](ratings_with_targets.csv) · [Summary JSON](summary.json)']
    (a.out/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([g for g in groups if g['model']=='all'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
