"""Point-in-time evidence for the complete 3-data, 4-length experiment."""
import json
import time
from common import HERE,dump,r
from dynamics_observation_length_data import NAME,DATA


def optional(path):return json.loads(path.read_text()) if path.exists() else {}


def main():
    root=HERE/f'reports/{NAME}';training=optional(root/'training_progress.json');verified=optional(root/'model_verification_progress.json')
    world=optional(root/'data_verification.json');inputs=optional(root/'input_verification.json');preflight=optional(root/'model_preflight.json')
    assert world['all_passed'] and inputs['all_passed'] and preflight['all_passed']
    lines=['# Observation length 1·2·4·8 comparison progress record','',
        'The initial scene of the three existing independent data sets was reused, and the action time was shifted to t=7. The final state, action, and future are predicted with equal observation lengths. The new independent data sets are not counted as three additional sets. This document is a output snapshot, and execution status is checked separately in the session.','',
        f'trajectory{world["episodes_replayed"]:,}Dog, RGB{world["observations_rerendered"]:,}Played the page. All RGB measurements, length-based reference estimates{inputs["visible_window_estimates_independently_recomputed"]:,}The number of items and the count were checked. Even if the masked input is changed, the calculation is the same, and the model size and initial weight per length were checked in advance to ensure they are the same.','',
        f'New video detection device{training.get("completed_models",0)}/72 completed learning, separate learning/development prediction playback verification{verified.get("models_verified",0)}/72 completed. Learning is fixed at the final 4,000 steps and the model is not selected by the intermediate score.','',
        '| Data / Observation duration | One-step comparison completed / Verification | Long future·opposite action completed / Verification |','| --- | --- | --- |']
    records=[]
    for ds in DATA:
        for length in [1,2,4,8]:
            folder=root/f'd{ds}/L{length}'
            obs=optional(folder/'observation_eval_progress.json').get('completed_conditions',0)
            obsv=optional(folder/'observation_eval_verification.json').get('conditions_verified',0)
            auto=optional(folder/'autonomous_progress.json').get('completed_conditions',0)
            av=optional(folder/'autonomous_verification.json').get('conditions_verified',0)
            partial=optional(folder/'autonomous_verification_progress.json').get('verified_conditions',0)
            av=max(av,partial)
            lines.append(f'| {ds} / {length}Jiang |{obs}/156 · {obsv}/156 | {auto}/1404 · {av}/1404 |')
            records.append({'data_seed':ds,'length':length,'observation_completed':obs,'observation_verified':obsv,'autonomous_completed':auto,'autonomous_verified':av})
    lines+=['',
        'Step-by-step contrast converts perception and environmental inputs into the correct answers, dividing the causes, and compares the effects of opposite behaviors between the 1/4/8/16/32/61-step and long-term future. During observation, the contact category only uses the L−1 observed transitions, so the composition of participants may vary depending on the length. The main comparison of the length effect is between matching the same overall scene.','',
        'The performance of speed and environment estimated in a single video, as well as the benefits and limitations of long observation, are not confirmed before the overall evaluation, reproduction, and aggregation by length are complete. Currently, passing input and model verification does not mean achieving the accuracy of a world model.','',
        'Remaining tasks: model learning/reproduction completion, connection of all data and lengths, long-term and opposite behavior evaluation and reproduction, length-based pair comparison aggregation and report. Subsequently, integrate with the results of environmental information missing. External SVIB, prior research, real human judgment, and final research bundles also remain separately.','',
        '[Input Verification](input_verification.json) · [Learning Input Pre-flight](model_preflight.json) · [Execution Plan](../../planning_C07_OBSERVATION_LENGTH_KO.md) · [Execution Sessions](execution_sessions.json)']
    (root/'PROGRESS_KO.md').write_text('\n'.join(lines)+'\n')
    dump(root/'progress_snapshot.json',{'snapshot_unix':time.time(),'trained_models':training.get('completed_models',0),
        'verified_models':verified.get('models_verified',0),'evaluations':records,'observation_length_complete':False,
        'C07_complete':False,'full_goal_complete':False,'source_sha256':r.sha(__file__),'report_sha256':r.sha(root/'PROGRESS_KO.md')})
    print('length progress snapshot',training.get('completed_models',0),'trained,',verified.get('models_verified',0),'verified',flush=True)


if __name__=='__main__':main()
