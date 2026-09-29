"""Check synthesis's new length table, paired directions and prerequisite evidence."""
import json
import re
from common import HERE,dump,r


def main():
    report=HERE/'reports/C07_SYNTHESIS_KO.md';text=report.read_text();gates={}
    for rel in ['dynamics_repeats_v1/report_verification.json','dynamics_context_omission_v1/report_verification.json',
                'dynamics_observation_length_v1/report_verification.json','dynamics_observation_length_v1/figure_visual_review.json',
                'observation_identifiability_v1/verification.json','dynamics_context_omission_v1/identifiability/verification.json']:
        p=HERE/'reports'/rel;d=json.loads(p.read_text());assert d['all_passed'];gates[str(p.relative_to(HERE))]=r.sha(p)
    curves=json.loads((HERE/'reports/dynamics_observation_length_v1/report_v1/curves.json').read_text())['curves']
    def select(metric,component='inference',stage='observation',horizon=0,paired=False,method='measurement_mlp'):
        selected=[c for c in curves if c['selector']['method']==method and c['selector']['stage']==stage
            and c['selector']['mode']=='variable_force' and c['selector']['split']=='test'
            and c['selector']['scalar_key']['metric']==metric and c['selector']['scalar_key']['component']==component
            and c['selector']['scalar_key']['horizon']==horizon and c['selector']['paired']==paired
            and c['selector']['stratum']=='all']
        assert len(selected)==1,(metric,component,horizon,paired,len(selected));return selected[0]
    velocity=select('zero_filled_true_slot_velocity_mae');force=select('net_force_mae_pixels_per_frame_squared')
    nxt=select('zero_filled_true_slot_velocity_mae','estimated_state_estimated_context',horizon=1)
    for i,length in enumerate([1,2,4,8]):
        line=f"| {length}장 | {velocity['mean'][i]:.4f} | {force['mean'][i]:.4f} | {nxt['mean'][i]:.4f} |";assert line in text
    comparisons=[]
    for metric,component,stage,h in [('zero_filled_true_slot_velocity_mae','inference','observation',0),
        ('net_force_mae_pixels_per_frame_squared','inference','observation',0),
        ('zero_filled_true_slot_velocity_mae','estimated_state_estimated_context','observation',1),
        ('position_mae_pixels_finite','factual','autonomous',16)]:
        c=select(metric,component,stage,h,True);assert all(v<0 for v in c['data_means'][2])
        if metric in ['net_force_mae_pixels_per_frame_squared','position_mae_pixels_finite']:assert all(v>0 for v in c['data_means'][5])
        comparisons.append({'id':c['id'],'L8_minus_L1':c['data_means'][2],'L8_minus_L4':c['data_means'][5]})
    cnn=select('zero_filled_true_slot_velocity_mae',method='rgb_cnn');assert format(cnn['mean'][3],'.4f')=='0.8155'
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if not target.startswith(('https:','http:')):assert (report.parent/target).is_file(),target
    for phrase in ['독립 데이터 단위는 세 개','더 많은 정보가 본질적으로 해롭다는 뜻은 아니다','전체의 Bayes 오차 하한','전체 목표']:
        if phrase=='전체 목표':continue
        assert phrase in text
    dump(HERE/'reports/C07_synthesis_verification.json',{'all_passed':True,'report_sha256':r.sha(report),
        'required_evidence_sha256':gates,'length_table_cells_checked':12,'paired_directions':comparisons,
        'verifier_sha256':r.sha(__file__),'scope':'All new synthesis length-table values and comparative directions checked; prior repeat/omission claims linked to their previously audited reports. Interpretation and stated limitations manually reviewed, not a statistical proof of causality.'})
    print('C07 synthesis verified',flush=True)


if __name__=='__main__':main()
