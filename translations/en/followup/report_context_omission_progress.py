"""Point-in-time progress only, never evidence that a process is currently live."""
import json
import time
from common import HERE,dump,r
from dynamics_context_omission import NAME


def main():
    root=HERE/f'reports/{NAME}';p=json.loads((root/'progress.json').read_text());v=json.loads((root/'verification_progress.json').read_text())
    theory=json.loads((root/'identifiability/verification.json').read_text());assert theory['all_passed']
    completed=v['all_passed'] and (root/'report_verification.json').exists() and json.loads((root/'report_verification.json').read_text())['all_passed']
    n=p['completed_conditions'];fresh=p['completed_new_trainings'];checked=v['conditions_verified'];pairs=v['zero_net_structure_training_comparisons']
    lines=['# Environmental information missing comparison progress record','',
        'Current state and actions are accurately provided, and only the combined or resisted inputs are used to re-learn from scratch. The information at the learning and evaluation stages is identical, and this is not a comparison where only the input of the full information model is deleted post-hoc. This document is a snapshot of the saved output, and whether it is actually running is verified separately during the work session.','',
        '| Work | Document creation time |','| --- | --- |',
        f'| Learning new missing condition |{fresh}/81 completed |',
        f'| Overall information comparison evaluation |{n-fresh}/27 completed |',
        f'| Full conditions learning·assessment |{n}/108 completed |',
        f'| Separate playback verification |{checked}/108 completed |',
        f"| Recalled transition prediction |{v['predictions_replayed']:,}Dog; not an independent sample |",
        f"| Combined hidden rotation model learning comparison |{len(pairs)}/18 pairs checked, among them pairs with completely the same weight{sum(x['weights_exactly_equal'] for x in pairs)}Dog |",
        '| Same input, different answer configuration | Completed three cases and free movement equation·square loss decomposition test |','',
        'We compare three independent data sets with three initializations and three interaction/rotation models. The training data only use variable_force and do not select different models depending on the evaluation environment. Therefore, it is a comparison with the table of C03 that was learned separately for each environment. The 27 total information comparisons reuse the existing weights, but the 81 missing conditions are learned anew.','',
        'If the combined force is removed, the observation force vector that causes rotation in the state alone, in the rotating and co-rotating structures, is zero. When the same weight is applied, the pre-test that the function and gradient are identical was passed. This does not mean that there is no hidden actual force or that the evaluation distribution is rotationally symmetric.','',
        'In the three configuration cases, the inputs for the current state, behavior, and observation environment were the same, but the next answer differed. The minimum squared loss was examined by weighting these two cases equally. This value is not claimed to be the lower bound of the error for the entire random evaluation data set.','',
        ('The complete aggregation and report number verification of learning, replication, and data seed by environmental information missing comparison has been completed. [Results Report](RESULTS_KO.md) explains the effects and limitations by condition.' if completed else 'The final performance effect of missing information is not judged until the total learning, playback, and data seed aggregation is complete.')+'There are also remaining experiments with observation lengths of 1/2, 4, and 8 at the common time point, external evaluations, human judgments, and final research outputs.','',
        '[Detailed execution plan](../../planning_C07_CONTEXT_OMISSION_KO.md) · [Input/calculation inspection](preflight.json) · [Information deficiency example](identifiability/EXPLANATION_KO.md) · [Learning progress](progress.json) · [Replication progress](verification_progress.json)']
    (root/'PROGRESS_KO.md').write_text('\n'.join(lines)+'\n')
    dump(root/'progress_manifest.json',{'snapshot_unix':time.time(),'completed_conditions':n,'completed_new_trainings':fresh,'verified_conditions':checked,
        'context_omission_complete':completed,'C07_complete':False,'full_goal_complete':False,'report_sha256':r.sha(root/'PROGRESS_KO.md'),
        'source_sha256':r.sha(__file__),'evidence_sha256':{f:r.sha(root/f) for f in ['progress.json','verification_progress.json','preflight.json','identifiability/verification.json']}})
    print('omission progress',fresh,'fresh trained,',checked,'conditions verified',flush=True)


if __name__=='__main__':main()
