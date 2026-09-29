"""Hand-check common-cohort arithmetic, missingness, and real artifact parsing."""
import numpy as np
from common import HERE,dump,r
from dynamics_observation_length_data import NAME
from observation_length_aggregation import stats,reduce_scenes,pair_scenes,masks,load_unit


def main():
    left=np.array([[1,np.nan,5],[3,7,np.inf],[9,11,2.]])
    right=np.array([[0,9,8],[1,3,4],[np.nan,10,5.]])
    groups=np.array([[1,1,1],[0,0,0],[1,0,1]],bool)
    mean,count,total,lm,rm=pair_scenes(left,right,groups)
    np.testing.assert_array_equal(count,[[2,2,2],[0,0,0],[1,1,2]])
    np.testing.assert_array_equal(total,[3,0,2])
    np.testing.assert_allclose(mean,[[1.5,2.5,-3],[np.nan]*3,[1,1,-3]],equal_nan=True)
    np.testing.assert_allclose(lm[0],[2,9,3.5]);np.testing.assert_allclose(rm[0],[.5,6.5,6.5])
    a,_,_=reduce_scenes(left,groups);b,_,_=reduce_scenes(right,groups)
    assert abs(a[0,0]-b[0,0]-mean[0,0])>1
    s=stats(np.array([[1,3,np.nan],[np.nan,np.nan,np.nan],[2,2,2.]]),axis=1)
    np.testing.assert_allclose(s['mean'],[2,np.nan,2],equal_nan=True)
    np.testing.assert_allclose(s['sd'],[np.sqrt(2),np.nan,0],equal_nan=True)
    np.testing.assert_array_equal(s['n'],[2,0,3])
    events=np.zeros((3,7,2),int);events[0,0,0]=1;events[1,6,1]=1
    np.testing.assert_array_equal(masks(events),[[1,1,1],[0,0,1],[1,1,0],[1,0,0],[0,1,0]])
    np.testing.assert_array_equal(masks(events[:,7:]),[[1,1,1],[1,1,1],[0,0,0],[0,0,0],[0,0,0]])
    sources={};real=[]
    for stage in ['observation','autonomous']:
        meta={'stage':stage,'method':'measurement_mlp','mode':'variable_force','split':'test','predictor':'joint_exact'}
        matrix,keys,event=load_unit(640031,1,0,meta,sources)
        assert matrix.shape[0]==128 and event.shape==(128,7,2)
        assert matrix.shape[1]==len(keys)==(192 if stage=='observation' else 480)
        real.append({'stage':stage,'episodes':len(matrix),'scalar_columns':len(keys)})
    dump(HERE/f'reports/{NAME}/aggregation_preflight.json',{'all_passed':True,'common_cohort_hand_case_checked':True,
        'unmatched_mean_difference_is_distinct':True,'empty_strata_stay_missing':True,'short_window_and_fixed_contact_masks_checked':True,
        'real_source_parsing_and_existing_summary_checks':real,'sources':sources,
        'source_sha256':{f:r.sha(HERE/f) for f in ['preflight_length_aggregation.py','observation_length_aggregation.py','planning_C07_LENGTH_AGGREGATION_KO.md']},
        'scope':'Arithmetic and artifact parsing preflight only. Complete length comparisons require all 12 audited evaluations and independent aggregate verification.'})
    print('length aggregation preflight passed',flush=True)


if __name__=='__main__':main()
