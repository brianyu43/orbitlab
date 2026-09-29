"""Trace key final-memo claims to audited result sources."""
import json
import re
from common import HERE,dump,r


def main():
    reports=['RESEARCH_MEMO_KO.md','EASY_EXPLANATION_KO.md'];contents={f:(HERE/'reports'/f).read_text() for f in reports}
    evidence={}
    for name in ['confirmation_v1/verification.json','object_image_edit_v1/verification.json',
        'C07_synthesis_verification.json','svib_preview_shape_swap_v1/report_verification.json',
        'nearest_methods_review_v2_verification.json','probe_step_comparison_v1/verification.json',
        'generation_cost_figure_v2/verification.json','figure_catalog_verification.json']:
        p=HERE/'reports'/name;d=json.loads(p.read_text());assert d['all_passed'];evidence[str(p.relative_to(HERE))]=r.sha(p)
    generation=json.loads((HERE/'reports/confirmation_v1/summary.json').read_text())['summary']
    numbers={'c4_seen':generation['equivariant_fixed']['seen_strict_accepted_and_joint']['mean']*100,
             'c4_ood':generation['equivariant_fixed']['ood_strict_accepted_and_joint']['mean']*100,
             'plain_time_seen':generation['plain_time']['seen_strict_accepted_and_joint']['mean']*100}
    for value in numbers.values():assert format(value,'.1f')+'%' in contents['RESEARCH_MEMO_KO.md']
    edits=json.loads((HERE/'reports/object_image_edit_v1/summary.json').read_text())['results']
    rgb=[x for x in edits if x['name'].endswith('_object') and x['name'].startswith(('flat_','slot_'))]
    oracle=[x for x in edits if x['name'].startswith('oracle_id_object_')]
    assert len(rgb)==30 and len(oracle)==15
    assert all(x['groups']['all']['metrics']['end_to_end_success']['mean']==0 for x in rgb)
    assert all(x['groups']['all']['metrics']['end_to_end_success']['mean']==1 for x in oracle)
    svib=json.loads((HERE/'reports/svib_preview_shape_swap_v1/aggregate_v1/summary.json').read_text())['conditions']
    comparisons=[]
    for alpha in ['0.0','0.2','0.4','0.6']:
        values={x['method']:x['mean'] for x in svib if x['alpha']==alpha and x['split']=='heldout' and x['category']=='all' and x['metric']=='pixel_mse'}
        assert values['identity']<values['c4']<values['plain'];comparisons.append({'alpha':alpha,**values})
    human=json.loads((HERE/'reports/human_review_v1/manifest.json').read_text());assert human['n']==240 and human['human_responses_received']==0
    assert json.loads((HERE/'reports/original_integrity_before_followup_release_v1.json').read_text())['all_match']
    links=[]
    for name,text in contents.items():
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if not target.startswith(('https:','http:')):assert (HERE/'reports'/target).is_file(),target
            links.append(target)
        assert '사람' in text
    memo=contents['RESEARCH_MEMO_KO.md']
    for phrase in ['통합한 단일 모델은 아니다','실제 사람 평가는 미수집','세 추가 실험은 아직 실행하지 않았다','독립 데이터 3개']:
        assert phrase in memo
    dump(HERE/'reports/final_memo_verification.json',{'all_passed':True,'report_sha256':{f:r.sha(HERE/'reports'/f) for f in reports},
        'prerequisite_evidence_sha256':evidence,'generation_percentages':numbers,'rgb_edit_conditions_with_zero_observed_success':30,
        'oracle_edit_conditions_with_full_success':15,'external_preview_comparisons':comparisons,'local_links_checked':len(links),
        'human_responses_received':0,'verifier_sha256':r.sha(__file__),
        'scope':'Major quantitative conclusions recomputed from already-audited summaries; local evidence links and explicit limits checked. Narrative interpretation manually reviewed. No new claim of overall model accuracy, novelty, human validation or training reproducibility.'})
    print('research memo and easy explanation verified',flush=True)


if __name__=='__main__':main()
