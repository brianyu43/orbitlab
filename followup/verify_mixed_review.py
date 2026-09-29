"""Check review coverage, frozen provenance, joins and every published tally."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'reports/mixed_review_v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def main():
    lock=json.loads((BASE/'annotation_lock.json').read_text())
    for name,digest in lock['files_sha256'].items():
        assert sha(BASE/name)==digest,name
    human=rows(BASE/'human_original.csv');ai=rows(BASE/'ai_visual_annotations.csv')
    assert len(human)==9 and len(ai)==231
    assert len({x['id'] for x in human+ai})==240
    assert {x['id'] for x in human+ai}=={f'Q{i:03}' for i in range(1,241)}
    key={x['id']:x for x in rows(ROOT/'reports/human_review_v1/private_key.csv')}
    rated={x['id']:x for x in human+ai}
    joined=rows(BASE/'aggregate_v1/ratings_with_targets.csv')
    assert len(joined)==240
    for x in joined:
        for k,v in rated[x['id']].items():assert x[k]==v,(x['id'],k)
        for k,v in key[x['id']].items():assert x[k]==v,(x['id'],k)
    summary=json.loads((BASE/'aggregate_v1/summary.json').read_text())
    assert summary['coverage']=={'human':9,'ai':231,'total_unique':240}
    for group in summary['groups']:
        subset=[x for x in joined if x['observer_type']==group['observer_type'] and
            (group['model']=='all' or x['model']==group['model']) and
            (group['split']=='all' or x['split']==group['split'])]
        assert len(subset)==group['n']>0
        counts=dict.fromkeys(group['counts'],0)
        for x in subset:
            sm=x['shape']==x['target_shape'];cm=x['color']==x['target_color'];clear=x['valid']=='1'
            for metric,truth in [('shape_match',sm),('color_match',cm),('joint_match',sm and cm),
                ('clear_form',clear),('strict_match',sm and cm and clear),
                ('shape_unknown_or_other',x['shape']=='-1'),('color_unknown_or_mixed',x['color']=='-1')]:
                counts[metric]+=bool(truth)
        assert counts==group['counts']
        assert all(group['rates'][k]==v/len(subset) for k,v in counts.items())
    # Revalidate every recorded evidence hash for the 30 completed original tasks.
    old=json.loads((ROOT/'reports/scope_audit_final_v1.json').read_text())
    assert old['complete_requirements']==30 and old['remaining']==['A04']
    checked={}
    for item in old['items']:
        if item['id']=='A04':continue
        assert item['assessment']=='complete_with_stated_scope'
        for name,digest in item['evidence_sha256'].items():
            assert sha(ROOT/name)==digest,name
            checked[name]=digest
        for check in item['specific_checks']:
            assert json.loads((ROOT/check['file']).read_text())[check['field']]==check['expected']
    receipt={'all_passed':True,'human_rows':9,'ai_rows':231,'unique_ids':240,
        'joined_rows_verified':240,'aggregate_groups_verified':len(summary['groups']),
        'prior_completed_requirements_revalidated':30,'prior_unique_evidence_files_checked':len(checked),
        'human_source_preserved':True,'human_ai_types_separated':True,
        'source_sha256':{'verifier':sha(Path(__file__)),'annotation_lock':sha(BASE/'annotation_lock.json'),
            'summary':sha(BASE/'aggregate_v1/summary.json'),'joined_rows':sha(BASE/'aggregate_v1/ratings_with_targets.csv')},
        'scope':'Checks data integrity, coverage, provenance, joins, arithmetic and prior task evidence. Does not establish subjective visual labels as ground truth or AI-human agreement.'}
    with (BASE/'verification.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
