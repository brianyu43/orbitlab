"""Correct C4 pose scoring by category, preserving raw readout-v1 results."""
import numpy as np
from object_state_readout import discrete_states,geometric_assignment


def measure(raw,truth):
    predictions=discrete_states(raw).detach().cpu().numpy();truth=truth.detach().cpu().numpy();rows=[];assignments=[]
    directions=np.array([[1,0],[0,-1],[-1,0],[0,1]])
    for i,(pred,target) in enumerate(zip(predictions,truth)):
        matched,p,g=geometric_assignment(pred,target);assignments.append(matched);correct=[];errors=[]
        for gid in g:
            if gid not in matched:correct.append(np.zeros(6,bool));continue
            v=pred[matched[gid]];t=target[gid]
            checks=np.array([v[:4].argmax()==t[:4].argmax(),v[4:10].argmax()==t[4:10].argmax(),
                np.all(np.abs(v[10:12]-t[10:12])*31.5<=1),abs(v[12]-t[12])*8<=.5,
                (directions@v[13:15]).argmax()==(directions@t[13:15]).argmax(),v[15]>.5])
            correct.append(checks);errors.append(float(np.abs(v[10:12]-t[10:12]).mean()*31.5))
        correct=np.stack(correct)
        rows.append({'base_id':i,'true_objects':len(g),'predicted_objects':len(p),'count_correct':len(g)==len(p),
            'matched_fraction':len(matched)/len(g),'shape_correct':float(correct[:,0].mean()),'color_correct':float(correct[:,1].mean()),
            'position_within_one_pixel':float(correct[:,2].mean()),'radius_within_half_pixel':float(correct[:,3].mean()),
            'pose_correct':float(correct[:,4].mean()),'object_all_factors_correct':float(correct.all(-1).mean()),
            'full_scene_success':bool(len(g)==len(p) and correct.all()),
            'matched_position_mae_pixels':float(np.mean(errors)) if errors else None})
    return rows,assignments
