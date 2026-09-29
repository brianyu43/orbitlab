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
    lines=['# Independent data repeat progress record','',
        '2026-09-23. The repeated learning and development verification of the data and video estimator has been completed. The final performance comparison for the entire C07 or independent repetitions has not been completed. The number below is based on the storage output at the time of document creation, and whether it was executed is separately verified for the actual work session.','',
        '## Conditions to maintain the comparison','',
        'The original data seed is 640031, and new seeds 997101 and 997102 are added. The order of three model initialization and sampling of training samples is the same as the original. Since the design crosses the data seed and initialization, it is not counted as nine independent data sets. The original learning weights are not transferred and are learned from scratch in the new training set.','',
        'State prediction consists of three sets of 126 total: three environments × seven models × three initializations × new data. It maintained the same 12,000 steps·batch 64·Adam 0.001·hidden 48·random object order as before. Video estimation consists of three initializations for each CNN and measurement+correction MLP, with two data sets, totaling 12. It maintained the 4,000 steps·batch 32·Adam 0.0005 and the scope of use of known color, physics rules.','',
        '## Actual completion and remaining work','',
        '| Item | Completed evidence | Remaining range |','| --- | --- | --- |',
        f"| New World Materials | Trajectory{data['episodes_replayed']:,}Dog, status{data['states_replayed']:,}Dog, RGB{data['observations_rerendered']:,}Play the song again | Compare model results of new data |",
        f"| Behavior change·symmetry | Opposite behavior{data['counterfactuals_replayed']:,}Dog, rotation{data['rotated_trajectories_checked']:,}Dog, change object order{data['permuted_trajectories_checked']:,}Dog test | Reconfirm long-term behavioral response of learned model |",
        f"| Material independence | Combining existing and new{data['unique_physical_and_image_orbits_including_original']:,}The initial physical/video rotation trajectory of the dog is separated without overlap | It does not mean generalizing to a wider real environment |",
        f"| Video input and non-learning reference | RGB measurement{rgb['rgb_measurements_replayed']:,}Separating correct answers and inference files, verifying benchmark indicators | Connecting hold conditions of new learning models for comparison |",
        f"| New video detector |{obs['models_verified']}/12 Learning and validation, train/validation prediction{obs['train_validation_predictions_replayed']:,}Open CPU playback | test/OOD·object count·non-learning external forces and long-term connection evaluation |",
        f'| New state predictor |{n}/126 Learning/Hold evaluation completed, among which{m}Dog playback verification completed | Integrated comparison by remaining learning, verification and data seed |',
        '| Observation length·information missing | Example of detailed planning and existing identification limit composition | Comparison of 1/2/4/8 frames and conditions at the common time point with missing conditions is not implemented |','',
        'The video estimator only scored train/validation. When the data changes, the validation error does not be reported as a final conclusion for improved test performance or independent verification. The completed hold‑back predictions of the state model are intermediate results before the comparison of all data seeds is finished. It does not select only good initializations, stop them, or lower the original baseline.','',
        'Currently, `dynamics_repeat_eval.py` has implemented a connection between the C04 component comparison of new weights and new inputs and the C05/C06 autonomous prediction. It confirmed that the input path and decoding match when compared to the development predictions saved with 2,304 new validation inferences. The overall connection evaluation is only executed after verifying the status models of 63 data bundles and video models from the corresponding data set verification receipts. At the time of this document, the connection evaluation has not yet been executed. Separate playback verification and aggregation for the independent evaluation must also be performed.','',
        '[Data verification](data_verification.json) · [Video input verification](observation_data_verification.json) · [Video model verification](observation_development_verification.json) · [Number of state learning counts](state_progress.json) · [State playback verification](state_verification_progress.json) · [Complete C07 execution plan](../../planning_C07_EXECUTION_KO.md)','',
        'The entire range of the SVIB external evaluation, pre-research comparison, actual person judgment, final research memo, drawing, and replication bundle is also maintained.']
    (root/'PROGRESS_KO.md').write_text('\n'.join(lines)+'\n')
    dump(root/'progress_manifest.json',{'snapshot_unix':time.time(),'state_models_completed':n,'state_models_verified':m,
        'expected_state_models':126,'observation_models_verified':12,'C07_complete':False,'full_goal_complete':False,
        'report_sha256':r.sha(root/'PROGRESS_KO.md'),'source_sha256':r.sha(__file__),
        'evidence_sha256':{p:r.sha(root/p) for p in ['data_verification.json','observation_data_verification.json','observation_development_verification.json','state_progress.json','state_verification_progress.json']},
        'warning':'Progress-file hashes are a point-in-time snapshot, not a current process-liveness claim.'})
    print('repeat progress report',n,'state models completed,',m,'verified; 12/12 estimators verified',flush=True)


if __name__=='__main__':main()
