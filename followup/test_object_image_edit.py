import dataclasses
import numpy as np
import torch
from object_world import ObjectState, state_slots, render, edit
from object_tasks import command
from object_image_edit import pack_slots, select_target, apply_commands, score
from object_oracle_study import image_from_slots


def fixture():
    scene=(ObjectState(0,0,1,20,20,6,0),ObjectState(1,1,2,44,42,6,1))
    x=torch.from_numpy(state_slots(scene))[None]
    cmds=torch.zeros(1,2,13);cmds[0,0]=torch.from_numpy(command('color',4));cmds[0,1]=torch.from_numpy(command('rotate',1))
    target=edit(edit(scene,0,'color',4),0,'rotate',1);y=torch.from_numpy(state_slots(target))[None]
    record={'source_index':0,'target_object_id':0,'operation':'color_then_rotate','source_target_pair_is_unseen':False,'target_pair_is_unseen':False}
    return scene,target,x,y,cmds,[record]


def evaluate(x,p,sel,cmds,truth,y,records,scene,target):
    return score(x,p,sel,cmds,truth,y,records,np.stack([image_from_slots(p[0])]),
        np.stack([render(target)['image']]),np.stack([render(scene)['image']]),torch.zeros(1,dtype=torch.long))[0][0]


def test_source_relative_success_is_separate_from_ground_truth_and_selection():
    scene,target,truth,y,cmds,records=fixture();x=truth.clone();x[0,1,:4]=torch.tensor([0.,0.,0.,1.])
    selected=select_target(x,torch.tensor([[20.,20.]]));p=apply_commands(x,selected,cmds,'analytic')
    row=evaluate(x,p,selected,cmds,truth,y,records,scene,target)
    assert row['target_selection_correct'] and row['full_command_success_relative_to_prediction']
    assert not row['non_target_correct_gt'] and not row['end_to_end_success']
    wrong=torch.tensor([1]);p=apply_commands(truth,wrong,cmds,'analytic');row=evaluate(truth,p,wrong,cmds,truth,y,records,scene,target)
    assert row['full_command_success_relative_to_prediction'] and not row['target_selection_correct'] and not row['end_to_end_success']


def test_matching_is_not_redone_after_edit_and_missing_target_fails():
    scene,target,x,y,cmds,records=fixture();sel=torch.tensor([0]);p=apply_commands(x,sel,cmds,'analytic')
    row=evaluate(x,p,sel,cmds,x,y,records,scene,target);assert row['end_to_end_success']
    swapped=p.clone();swapped[:,[0,1]]=swapped[:,[1,0]]
    row=evaluate(x,swapped,sel,cmds,x,y,records,scene,target);assert not row['end_to_end_success']
    empty=torch.zeros_like(x);none=select_target(empty,torch.tensor([[20.,20.]]));assert none.item()==-1
    out=apply_commands(empty,none,cmds,'analytic');assert torch.equal(empty,out)
    row=evaluate(empty,out,none,cmds,x,y,records,scene,target)
    assert not row['target_success_gt'] and not row['full_command_success_relative_to_prediction']


def test_capacity_uses_presence_and_preserves_retained_slot_order():
    raw=torch.zeros(1,5,16);raw[...,15]=torch.tensor([[3.,1.,5.,-2.,4.]])
    raw[0,:,10]=torch.arange(5)/10
    state,order,overflow=pack_slots(raw)
    assert order.tolist()==[[0,1,2,4]] and overflow.tolist()==[0]
    assert torch.equal(state[0,:,10],raw[0,order[0],10])
    raw[0,3,15]=2;state,order,overflow=pack_slots(raw)
    assert order.tolist()==[[0,2,3,4]] and overflow.tolist()==[1]
