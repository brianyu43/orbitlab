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
'A01':'Verify hash and size of all 1,324 files in the original. Separate output for subsequent runs. Existing execution environment and MPS learning records are preserved in the original and subsequent runs.',
'A02':'Originally only train was used in fitting, and remove the overlap of rotation orbits between new validation/test/OOD. Check the locked settings and data hash.',
'A03':'Shape/color/passability of 8 types of deformation and critical sensitivity of 4 types of voice contrast. The limits of the original detector in accepting deformation and noise are disclosed.',
'A03b':'Only in calibration, the threshold is selected, independent test and existing product re-evaluation. Not human verification.',
'A04':'Prepared 240 pages and source HTML/link tables. Actual human responses: 0. Browser visual/interaction checks are also not performed due to the local URL policy. Preparation is not counted as actual collection.',
'A05':'Selection of train fitting/validation, new test/OOD, comparison of the full space of existing 6k partial spaces and inputs such as 3k/6k. Missing direct comparisons will be completed during the final audit.',
'A06':'Recovery/generation 8 conditions of the same evaluator and decoder dispersion decomposition by direction. No claim that inconsistency is the cause separation of properties.',
'A07':'Same frozen C4 AE by general/enhanced/C4 flow, 9 times×4,000 steps; initial weighting, sample, evaluation noise matching.',
'A08':'Record the flow 9 times at the same time, the actual execution step·AE+flow cost·576 sheets inference time. Distinguish the same parameters and the same calculation amount.',
'A09':'Duplicate and play results of 3 new data × 3 initializations, AE 9/flow 27, and new scenes. Separate 3 data and noise conditional segments.',
'A10':'Comparison of two new decoders with the same budget and oracle/learning latent 12 times, 9 generation transfer. This is a search in the existing pool and not an independent verification.',
'A11':'Standard MSE identity derivation, 512-page inspection, invariant-only recovery and full C4 distinction. Not a new infinite theorem or a bound on all expressions.',
'A12':'Record the retained effect, evaluation device/decoder limitations, and reasons for selecting the next stage. At that time, the remaining task phrases were updated and interpreted by integrating the latest research notes and comprehensive results.',
'B01':'Reproduce, split, and global rotation test the correct object ID, mask, status, and RGB.',
'B02':'Check the answer pairs to preserve properties such as color/rotation/movement, synthesis/sequence and changes.',
'B03':'Display correct answer status and correct answer target, flat/object-specific sharing model identical sample/parameter comparison. Distinguish from actual RGB recognition scores.',
'B04':'Only uses RGB input for evaluation, including slot assignment, restoration, masking, and object count. This is a failure diagnosis for the reduced fixed model and not a replication of the original research\'s performance.',
'B05':'Enough slot capacity, 640 new pages·2,961 commands, 115 conditions·68,103 outputs, change of target/preserve non-target/verify the entire answer separately.',
'B06':'Official preview 500 pairs·24 model·share test100. Official fold·original pixel·duplicate·exposure thanks. All official SVIB and LPIPS not executed are disclosed.',
'B07':'Degree of shading·number of objects·synthetic command·table/curve by error type, fixed first example video. Do not generalize technical statistics as causal effects.',
'C01':'Isan collision simulator, state/video and trajectory split·energy·overlap number check. Not continuous/real physics verification.',
'C02':'The correct answer that rotates state, action, and external forces together, and the counterexample when fixing the equilibrium and external forces. Accurate application range disclosed.',
'C03':'63 models with equal speed and known physical references; only compare the correct state and one step of the environment. Initialization, sample, reuse audit.',
'C04':'Same past 4 images, RGB/measurement models and correct state/environment replacement, 307,200 next prediction. Recognition and information estimation separation.',
'C05':'Step 61: Autonomous prediction, aesthetic power/properties/object count, position/speed/contact candidate/failure assessment. Object color/existence is copied, so it is not claimed as a success in learning tracking.',
'C06':'Same past\'s opposite behavior answers and autonomous reactions are tested by subject/non-subject. Do not create the falsity of the four objects that are not present. Reveal the metaphorical denominator.',
'C07':'Three independent repeated data·108 missing information conditions·72 length models, 520 total aggregation, fixed contact group·information ambiguity cases and overall conclusion.',
'D01':'Compare about 9 methods: text and possible official codes. Methods without public codes are marked as unreviewed. Not a replication of prior research or proof of originality from other papers.',
'D02':'Cause separation·calculation cost·conditions·intervention/association·long future: 10 sets 25 files. Material·numerical inspection·image decoding·connection of specified visual review range.',
'D03':'A short research memo containing hypotheses, methods, results, failures, and limitations, along with easy explanations. Separate numerical/evidence/unimplemented suggestions/person under-collection.',
'D04':'Original immutable/complete item audit/separate ZIP/re-examine all ZIP bytes/playback of CPU samples in other paths. The last two audits are verified with separate receipts after sealing.',
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
    lines=['# Full review of all subsequent plans','',f"Among 31 items{report['complete_requirements']}The dog completed within the specified range. Remaining items:{', '.join(report['remaining'])}. The overall goal is not completed.",'',
        '| ID | Required evidence | Current judgment and actual range |','| --- | --- | --- |']
    for item in items:lines.append(f"| {item['id']} | {item['requirement']} | **{item['assessment']}**. {item['coverage_and_limits']} |")
    lines+=['','Scope of inspection: Completion was not determined solely by the presence of a file. The original report and verifier scope were reviewed, and the numerical values, replication, and aggregation reasons for each item were linked. This audit tool did not re-learn the entire model. The hashes of each data item and the actual inspection fields are linked in the linked JSON.','',f'[{output.name}]({output.name})']
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print('scope audit',report['complete_requirements'],'/31; remaining',report['remaining'],flush=True)


if __name__=='__main__':main()
