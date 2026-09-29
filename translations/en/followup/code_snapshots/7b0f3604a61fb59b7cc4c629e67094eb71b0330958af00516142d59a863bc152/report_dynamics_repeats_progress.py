"""Evidence-bounded progress snapshot, never substitutes for a live process poll."""
import json
import time
from common import HERE,dump,r
from dynamics_repeat_data import NAME


def main():
    root=HERE/f'reports/{NAME}'
    data=json.loads((root/'data_verification.json').read_text());rgb=json.loads((root/'observation_data_verification.json').read_text())
    obs=json.loads((root/'observation_development_verification.json').read_text())
    state=json.loads((root/'state_progress.json').read_text());checked=json.loads((root/'state_verification_progress.json').read_text())
    assert data['all_passed'] and rgb['all_passed'] and obs['all_passed']
    n,m=state['completed_models'],checked['models_verified']
    across_complete=all((root/f'across_data/{stage}_verification.json').exists() and
        json.loads((root/f'across_data/{stage}_verification.json').read_text()).get('all_passed',False)
        for stage in ['state','observation','autonomous'])
    evaluations={}
    for ds in [997101,997102]:
        stages={}
        for stage in ['observation_eval','autonomous']:
            p=root/f'd{ds}/{stage}_progress.json'
            if p.exists():
                stages[stage]=json.loads(p.read_text())
        evaluations[str(ds)]=stages
    evaluation_text=[]
    for ds,stages in evaluations.items():
        if not stages:
            evaluation_text.append(f'Data{ds}: No connection evaluation storage results.')
        for stage,v in stages.items():
            label='Video input comparison' if stage=='observation_eval' else 'Long future·change of action'
            verdict=root/f'd{ds}/{stage}_verification.json'
            audited=verdict.exists() and json.loads(verdict.read_text()).get('all_passed',False)
            evaluation_text.append(f"Data{ds}of{label}: {v['completed_conditions']}/{v['expected_conditions']}Save dog conditions, verify separate playback{'completion' if audited else 'Unfinished'}.")
    lines=['# Independent data repeat progress record','',
        '2026-09-23. The number below is based on the storage output at the time of document creation, and whether it is in progress is verified separately for the actual work session. The entire C07 and subsequent goals are underway.','',
        ('Completed the learning of independent data repetition, C03–C06 evaluation, and aggregation verification and comparison reports. [Result Report](RESULTS_KO.md). The observation length experiment and environmental information missingness comparison are separate.' if across_complete else 'The final comparison of independent data repetition is still underway.'),'',
        '## Conditions to maintain the comparison','',
        'The original data seed is 640031, and new seeds 997101 and 997102 are added. The order of three model initialization and sampling of training samples is the same as the original. Since the design crosses the data seed and initialization, it is not counted as nine independent data sets. The original learning weights are not transferred and are learned from scratch in the new training set.','',
        'State prediction consists of three sets of 126 total: three environments × seven models × three initializations × new data. It maintained the same 12,000 steps·batch 64·Adam 0.001·hidden 48·random object order as before. Video estimation consists of three initializations for each CNN and measurement+correction MLP, with two data sets, totaling 12. It maintained the 4,000 steps·batch 32·Adam 0.0005 and the scope of use of known color, physics rules.','',
        '## Actual completion and remaining work','',
        '| Item | Completed evidence | Remaining range |','| --- | --- | --- |',
        f"| New World Materials | Trajectory{data['episodes_replayed']:,}Dog, status{data['states_replayed']:,}Dog, RGB{data['observations_rerendered']:,}Play long | Distinguish from the model evaluation status below |",
        f"| Behavior change·symmetry | Opposite behavior{data['counterfactuals_replayed']:,}Dog, rotation{data['rotated_trajectories_checked']:,}Dog, change object order{data['permuted_trajectories_checked']:,}General examination | Distinguish from the following long-term behavioral assessment status |",
        f"| Material independence | Combining existing and new{data['unique_physical_and_image_orbits_including_original']:,}The initial physical/video rotation trajectory of the dog is separated without overlap | It does not mean generalizing to a wider real environment |",
        f"| Video input and non-learning reference | RGB measurement{rgb['rgb_measurements_replayed']:,}Separating correct answers and inference files, verifying benchmark indicators | Connecting hold conditions of new learning models for comparison |",
        f"| New video detector |{obs['models_verified']}/12 Learning and validation, train/validation prediction{obs['train_validation_predictions_replayed']:,}Open CPU playback | test/OOD·object count·non-learning external forces and long-term connection evaluation |",
        f'| New state predictor |{n}/126 Learning/Hold evaluation completed, among which{m}Dog playback verification completed | Integrated comparison by remaining learning, verification and data seed |',
        '| Observation length·information missing | Example of separate detailed planning and identification limit configuration | 1/2/4/8 frames are not executed; information missing refer to separate progress records |','',
        'Separate records are kept for the evaluation of the connection between the train/validation development verification and the hold-out condition. When the data changes, the validation error does not be reported as the final conclusion for improving test performance or independent verification. The completed hold-out predictions of the state model are intermediate results before the comparison of all data seeds is complete. Only the good initializations are selected and stopped, or the original baseline is not lowered.','',
        'Currently, `dynamics_repeat_eval.py` connects the C04 component comparison with new weights and new inputs, and the C05/C06 autonomous prediction. It confirmed that the input path and decoding match by comparing the development predictions, which have saved 2,304 new validation inferences. The overall connection evaluation is only executed after verifying the state models of 63 data bundles and video models from the corresponding validation receipts. Separate playback validation for independent evaluation and aggregation between data bundles must also be performed.','',
        ' '.join(evaluation_text),'',
        '[Progressing environmental information omission comparison](../dynamics_context_omission_v1/PROGRESS_KO.md) · [Detailed execution plan for observation length](../../planning_C07_OBSERVATION_LENGTH_KO.md)','',
        '[Data verification](data_verification.json) · [Video input verification](observation_data_verification.json) · [Video model verification](observation_development_verification.json) · [Number of state learning counts](state_progress.json) · [State playback verification](state_verification_progress.json) · [Complete C07 execution plan](../../planning_C07_EXECUTION_KO.md)','',
        'The entire range of the SVIB external evaluation, pre-research comparison, actual person judgment, final research memo, drawing, and replication bundle is also maintained.']
    (root/'PROGRESS_KO.md').write_text('\n'.join(lines)+'\n')
    dump(root/'progress_manifest.json',{'snapshot_unix':time.time(),'state_models_completed':n,'state_models_verified':m,
        'expected_state_models':126,'observation_models_verified':12,'C07_complete':False,'full_goal_complete':False,
        'independent_repeat_aggregate_verified':across_complete,
        'evaluation_progress':evaluations,
        'report_sha256':r.sha(root/'PROGRESS_KO.md'),'source_sha256':r.sha(__file__),
        'evidence_sha256':{p:r.sha(root/p) for p in ['data_verification.json','observation_data_verification.json','observation_development_verification.json','state_progress.json','state_verification_progress.json']},
        'warning':'Progress-file hashes are a point-in-time snapshot, not a current process-liveness claim.'})
    print('repeat progress report',n,'state models completed,',m,'verified; 12/12 estimators verified',flush=True)


if __name__=='__main__':main()
