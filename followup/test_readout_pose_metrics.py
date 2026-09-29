import unittest
import numpy as np
import torch
from object_world import ObjectState,state_slots
from object_state_readout import measure as old_measure
from object_readout_metrics_v2 import measure


class PoseChecks(unittest.TestCase):
    def test_all_renderer_poses_are_perfect_in_their_own_category(self):
        labels=[];pred=[]
        for pose in range(4):
            target=torch.tensor(state_slots((ObjectState(0,0,1,20,20,6,pose),ObjectState(1,1,2,40,40,7,pose))))
            raw=torch.zeros(5,16);raw[...,15]=-10
            raw[:2]=target[:2];raw[:2,:10]*=10;raw[:2,15]=10;labels.append(target);pred.append(raw)
        truth=torch.stack(labels);raw=torch.stack(pred)
        rows,_=measure(raw,truth);old,_=old_measure(raw,truth)
        self.assertTrue(all(v['full_scene_success'] and v['pose_correct']==1 for v in rows))
        self.assertTrue(any(v['pose_correct']<1 for v in old))
        raw[:,0,13:15]*=-1;wrong,_=measure(raw,truth);self.assertTrue(all(v['pose_correct']==.5 for v in wrong))


if __name__=='__main__':unittest.main()
