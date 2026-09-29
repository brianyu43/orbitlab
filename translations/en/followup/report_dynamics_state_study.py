"""Write the complete state-only comparison from verified machine-readable results."""
import json
from common import HERE,dump,r


def main():
    root=HERE/'reports/dynamics_state_study_v1';agg=json.loads((root/'aggregate.json').read_text())
    for path in [root/'verification.json',root/'aggregate_verification.json',HERE/'reports/dynamics_state_baselines_v1/verification.json']:
        assert json.loads(path.read_text())['all_passed']
    base=json.loads((HERE/'reports/dynamics_state_baselines_v1/summary.json').read_text())['results']
    labels={'flat':'The whole scene MLP','object':'Independent object MLP','interaction':'Object interaction','augmented':'State and external force rotation enhancement',
        'wrong_exact':'Restriction that only rotates the state','joint_exact':'Constraints that rotate together with state and external forces','relaxed':'State rotation restriction and general model combination',
        'inertial':'Reference to acceleration','force_wall':'Reference to known external forces and wall reflection'}
    def value(mode,kind,split,category='all'):
        if kind in ['inertial','force_wall']:
            return next(x for x in base if x['mode']==mode and x['kind']==kind and x['split']==split and x['category']==category)['metrics']['velocity_mae_pixels_per_frame']['mean']
        return next(x for x in agg['groups'] if x['mode']==mode and x['kind']==kind and x['split']==split and x['category']==category)['metrics']['velocity_mae_pixels_per_frame']['mean']
    kinds=list(labels);lines=['# One-step exercise prediction in correct state: Complete this comparison','',
        '2026-09-23. Evaluated 63 learning models for 3 environments × 7 methods × 3 initialization. Reused the 9 previously verified checkpoints and trained 54 new ones. This comparison is C03, and C04, which detects the state from video, and C05, which predicts the continuous future, are separate stages.','',
        '## Common input and comparison conditions','',
        'It provided the actual object state, behavior, known total power and resistance coefficients at every moment. Two objects were transitioned uniformly along the train trajectory, using batch 64, Adam 0.001, and a common 12,000 steps. We did not select the best test among several checkpoints. We matched the raw initial weights of the symmetric contrast group with the sample flow. The number of parameters and actual calculation time are different, and this is not a comparison that aligns the time. There is one data generation seed, and three initializations.','',
        'The model for each object does not see other objects. The interaction model combines the messages of other objects. The constraint that only rotates the state rotates the state while fixing the directional external force. The joint constraint of state and external force also changes the external force when the coordinate system is changed. In an environment without directional external force, the two constraints become the same.','',
        '## General test','',
        'It is the average absolute error in speed, and the unit is pixels/frames. The lower it is, the better. First, calculate the average within the trajectory, and the learning model displayed three initialization averages.','',
        '| Method | Directionless environment | Fixed gravity | Variable external force |','| --- | ---: | ---: | ---: |']
    for kind in kinds:lines.append('| '+labels[kind]+' | '+' | '.join(f'{value(m,kind,"test"):.4f}' for m in ['isotropic','fixed_gravity','variable_force'])+' |')
    lines+=['','## Changing external force: Restraining conditions and object number','',
        '| Method | General test | Combination of non-learned properties | Three objects | Four objects | Non-learned external force |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for kind in kinds:lines.append('| '+labels[kind]+' | '+' | '.join(f'{value("variable_force",kind,s):.4f}' for s in ['test','attribute_ood','count3','count4','force_ood'])+' |')
    lines+=['','## What was improved and what was left','',
        'The common test for variable external forces and the joint rotation constraint in non-learned external forces showed that both initializations were less error-prone than the general interaction, rotation enhancement, state-only rotation, and hybrid models. However, in the four objects, neither initialization was better than the model that only rotates the state. It does not extend the effect under specific conditions to all generalized capabilities.','',
        'The increase in the number of objects has significantly increased the error in several interaction models. The structure of summing without normalizing the message by the number of messages is a candidate cause, but because a comparative experiment was not conducted only by changing the aggregation method, it cannot be confirmed as the cause. The independent object model is a structure that excludes interaction, so it may be less sensitive to changes in the number of objects, but it does not mean that a collision model is sufficient.','',
        'The average includes many moments without collisions. The speed error of the variable external force test, which separates the collision moments, is as follows.','',
        '| Method | Non-contact | Contact between objects | Contact only with the wall |','| --- | ---: | ---: | ---: |']
    for kind in ['interaction','joint_exact','inertial','force_wall']:
        lines.append('| '+labels[kind]+' | '+' | '.join(f'{value("variable_force",kind,"test",cat):.4f}' for cat in ['no_contact','pair','wall_only'])+' |')
    lines+=['',
        'The external force and wall reflection reference uses the known integration rules and boundaries of the simulator directly and ignores the contact between objects. It is not a model that learns the physical laws through learning. Since this strong non‑learned reference is more accurate on the overall average, the learning model does not claim that it predicts this world overall better.','',
        '## Verification and uncertainty','',
        '- 1,720,320 transition predictions for 63 models were played back. optimizer·sample·initialization·reuse hash and 2,730 metric aggregation tests passed.',
        '- The physical reference was used to recalculate 200,704 individual spherical objects using the original simulator, and 12,022 indicator rows and 260 aggregates/regions were examined.',
        '- In the total 63 data sources in CSV, 910 and 52 paired comparisons were separately examined for the average, initialization standard deviation, and bootstrap intervals of 455 conditions.',
        '- Separated the initial three values and sample standard deviation from the scene resampling interval. The interval is conditional on these three models, and it is not an independent data seed repetition or a population significance test.',
        '- It is a step-by-step prediction as long as the correct state returns every moment. We did not verify the performance of long rollout, visual recognition, environmental inference, and counterfactual prediction here.','',
        '## Next step','',
        'Using the same past observation length, position, velocity, and environment are estimated from the video, and error accumulation is measured through continuous prediction that does not provide the correct state again. The possibility of object failure and external forces are separately evaluated, as well as the response of non-target objects to behavioral changes.','',
        '[Aggregate all conditions](aggregate.json) · [Material comparison table](comparison_rows.csv) · [Model replay verification](verification.json) · [Aggregate audit](aggregate_verification.json) · [Physical reference verification](../dynamics_state_baselines_v1/verification.json)']
    path=root/'RESULTS_KO.md';path.write_text('\n'.join(lines)+'\n')
    dump(root/'report_manifest.json',{'report_sha256':r.sha(path),'aggregate_sha256':r.sha(root/'aggregate.json'),
        'baseline_summary_sha256':r.sha(HERE/'reports/dynamics_state_baselines_v1/summary.json'),'script_sha256':r.sha(__file__)})
    print('Wrote verified C03 report')


if __name__=='__main__':main()
