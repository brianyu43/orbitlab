"""Image-only edit inference with explicit truth-state control and fixed metrics."""
import argparse,json
import numpy as np
import torch
import r2_decoder_core as c
import r2_world as w
import r2_core as readers
import r2_evaluate as state_eval
from v3_common import sha,dump,lock
ARMS=['truth_state']+readers.ARMS

def freeze():
    return lock(c.BASE/'evaluation_protocol_v2.json',{'version':2,'supersedes_protocol_sha256':sha(c.BASE/'evaluation_protocol.json'),'amendment':'Read compressed target arrays once per condition, not once per row; unchanged model, target and metric arithmetic. Preserve old predictions and summaries in evaluation/.','source_sha256':sha(__file__),'decoder_protocol_sha256':sha(c.BASE/'protocol.json'),'arms':ARMS,'counts':[2,3,4,6],'commands':c.OPS[1:],'oracle_identity':True,'copy_input_baseline':True,
        'prediction_storage':'Save uint8 edited RGB before loading target pixels for scoring. Foreground raster diagnostics are recomputed from model state, with first8scenes saved as examples.','foreground_score':'Compare learned premultiplied RGBA to true canonical foreground, including alpha IoU. This is separate from full background/panel editing.','non_target_metric':'Pixel error against the correct target in the originally visible non-target region. Legitimate target-induced overlap is not labeled preservation failure.','pixel_metrics':'MAE in0..1; changed/preserved use any source-target uint8 difference. Fractional panel/non-target weights retained from stored masks. Empty regions have0error and recorded0area.','success':'Thresholded pixel gate and joint pixel-and-analytic-state gate are both reported. No semantic equivalence inferred from pixel gate alone.'})

def load_models():
    patch=c.PatchDecoder();editor=c.ImageEditor()
    for kind,model in [('patch',patch),('editor',editor)]:
        cp=c.BASE/f'runs/{kind}/model.pt';r=json.loads((cp.parent/'run.json').read_text());assert r['checkpoint_sha256']==sha(cp)
        model.load_state_dict(torch.load(cp,weights_only=True)['state_dict']);model.eval()
    return patch,editor

def inputs(count,arm,identity=False):
    folder=w.BASE/f'data/d{w.DEVELOPMENT}/val/n{count}';source=np.load(folder/'rgb.npz')['images'][:,0]
    target_data=c.BASE/f'data/d{w.DEVELOPMENT}/val/n{count}';commands=json.loads((target_data/'commands.json').read_text())
    indices=[i*4+j for i in range(len(source)) for j in ([0] if identity else [1,2,3])]
    commands=[commands[j] for j in indices];source_ids=np.array(indices)//4
    if arm=='truth_state':scenes=[r['objects'] for r in json.loads((folder/'labels.json').read_text())]
    else:
        raw=np.load(w.BASE/f'evaluation/d{w.DEVELOPMENT}_s0/{arm}/n{count}/inference.npz')['raw'];scenes=[state_eval.discrete(v) for v in raw]
    original=[scenes[i] for i in source_ids];edited=[c.edit(s,cmd)[0] for s,cmd in zip(original,commands)]
    return source[source_ids],original,edited,commands,indices

@torch.no_grad()
def predict(patch,editor,source,original,edited,commands):
    images=[];current=[];after=[]
    for i in range(0,len(source),16):
        x=torch.from_numpy(source[i:i+16]).permute(0,3,1,2).float()/255
        a=c.render(patch,c.states(original[i:i+16]));b=c.render(patch,c.states(edited[i:i+16]))
        y,_=editor(x,a.half().float(),b.half().float(),c.command_maps(commands[i:i+16]))
        images.append((y.permute(0,2,3,1)*255).round().clamp(0,255).byte().numpy())
        current.append(a.permute(0,2,3,1).numpy());after.append(b.permute(0,2,3,1).numpy())
    return np.concatenate(images),np.concatenate(current),np.concatenate(after)

def region(error,mask):
    area=float(mask.sum());return float((error*mask[...,None]).sum()/(3*area)) if area else 0.,area

def analytic_success(pred,row,cmd):
    truth=row['objects'];assignment=state_eval.match(pred,truth);tid=row['target_id']
    edited,selected=c.edit(pred,cmd);target,_=c.edit(truth,cmd,tid)
    selected_ok=tid in assignment and assignment[tid]==selected and np.hypot(pred[selected]['x']-truth[tid]['x'],pred[selected]['y']-truth[tid]['y'])<=4
    checks=[all(state_eval.checks(edited[assignment[j]] if j in assignment else None,g)) for j,g in enumerate(target)]
    return bool(selected_ok and len(pred)==len(truth) and all(checks)),bool(all(v for j,v in enumerate(checks) if j!=tid)),bool(selected_ok)

def rows_for(count,arm,source,pred,current,after,objects,commands,indices):
    folder=w.BASE/f'data/d{w.DEVELOPMENT}/val/n{count}';records=json.loads((folder/'labels.json').read_text())
    target_data=c.BASE/f'data/d{w.DEVELOPMENT}/val/n{count}';target_archive=np.load(target_data/'targets.npz');target_npz={key:target_archive[key] for key in target_archive.files};target_archive.close();metadata=json.loads((target_data/'metadata.json').read_text())
    rows=[]
    for j,(image,out,idx,cmd) in enumerate(zip(source,pred,indices,commands)):
        i=idx//4;row=records[i];target=target_npz['targets'][idx];err=np.abs(out.astype(np.float64)-target)/255;change=(image!=target).any(-1).astype(float)
        panel=target_npz['panels'][idx].astype(float)/255;non_target=target_npz['non_targets'][idx].astype(float)/255
        _,true_current,_=c.truth_render(row['objects'],np.zeros((64,64,3),np.float32),np.zeros((64,64),np.float32))
        _,true_after,_=c.truth_render(metadata[idx]['objects'],np.zeros((64,64,3),np.float32),np.zeros((64,64),np.float32))
        background=(true_current[...,3]<.01)&(true_after[...,3]<.01)&(panel==0)
        changed,ca=region(err,change);preserved,pa=region(err,1-change);other,oa=region(err,non_target);occluder,ba=region(err,panel);bg,bga=region(err,background)
        px=changed<=.05 and preserved<=.01 and other<=.02 and occluder<=.01
        r={'source_index':i,'op':cmd['op'],'count':count,'requested_occlusion':row['requested_occlusion'],'actual_target_occlusion':row['actual_target_occlusion'],'same_color':row['same_color'],'continuous_color':row['continuous_color'],'textured_background':row['textured_background'],
           'whole_mae':float(err.mean()),'changed_mae':changed,'preserved_mae':preserved,'non_target_mae':other,'panel_mae':occluder,'background_mae':bg,
           'changed_area':ca,'preserved_area':pa,'non_target_area':oa,'panel_area':ba,'background_area':bga,'pixel_gate':float(px)}
        if arm!='copy_input':
            state,non,select=analytic_success(objects[j],row,cmd)
            r.update({'analytic_state_gate':float(state),'analytic_non_target_correct':float(non),'click_selection':float(select),'joint_state_pixel_gate':float(px and state)})
            for tag,a,b in [('source',current[j],true_current),('edited',after[j],true_after)]:
                support=b[...,3]>.01;rgb_err=np.abs(a[...,:3]-b[...,:3]);fg_mae,_=region(rgb_err,support)
                pa=a[...,3]>.5;gt=b[...,3]>.5;union=(pa|gt).sum()
                r[f'{tag}_foreground_mae']=fg_mae;r[f'{tag}_foreground_alpha_iou']=float((pa&gt).sum()/union) if union else 1.
        rows.append(r)
    return rows

def summarize(rows):
    omit={'source_index','op','count','requested_occlusion','same_color','continuous_color','textured_background'}
    def means(rr):return {'edits':len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in rr[0] if k not in omit}}
    result={'all':means(rows)}
    for field,vals in [('op',c.OPS),('requested_occlusion',[0.,.25,.5,.75]),('same_color',[0,1]),('continuous_color',[0,1]),('textured_background',[0,1])]:
        for v in vals:
            rr=[r for r in rows if r[field]==v]
            if rr:result[f'{field}_{v}']=means(rr)
    return result

def evaluate(arm,count,verify=False,identity=False):
    freeze();torch.set_num_threads(2);source,objects,edited,commands,indices=inputs(count,'truth_state' if arm=='copy_input' else arm,identity)
    dest=c.BASE/f'evaluation_v2/{arm}{"_identity" if identity else ""}/n{count}';dest.mkdir(parents=True,exist_ok=True)
    if not verify and (dest/'summary.json').exists():
        r=json.loads((dest/'summary.json').read_text());assert r['evaluator_sha256']==sha(__file__) and r['prediction_sha256']==sha(dest/'predictions.npz');return r
    if arm=='copy_input':pred=source.copy();current=after=None
    else:pred,current,after=predict(*load_models(),source,objects,edited,commands)
    if verify:np.testing.assert_array_equal(pred,np.load(dest/'predictions.npz')['images'])
    else:
        np.savez_compressed(dest/'predictions.npz',images=pred)
        if current is not None:np.savez_compressed(dest/'foreground_examples.npz',current=current[:24],edited=after[:24])
    # Target pixels and masks are accessed by rows_for only after prediction save.
    rows=rows_for(count,arm,source,pred,current,after,objects,commands,indices);metrics=summarize(rows)
    if arm in readers.ARMS and not identity:
        old=json.loads((w.BASE/f'evaluation/d{w.DEVELOPMENT}_s0/{arm}/n{count}/rows.json').read_text())
        for i,row in enumerate(old):
            assert np.mean([r['analytic_state_gate'] for r in rows[i*3:i*3+3]])==row['joint_edit_success']
            assert np.mean([r['analytic_non_target_correct'] for r in rows[i*3:i*3+3]])==row['non_target_correct']
    dependencies={'source':sha(w.BASE/f'data/d{w.DEVELOPMENT}/val/n{count}/rgb.npz'),'target_manifest':sha(c.BASE/f'data/d{w.DEVELOPMENT}/val/n{count}/manifest.json')}
    if arm!='copy_input':
        dependencies.update({kind:sha(c.BASE/f'runs/{kind}/model.pt') for kind in ['patch','editor']})
        if arm!='truth_state':dependencies['reader_inference']=sha(w.BASE/f'evaluation/d{w.DEVELOPMENT}_s0/{arm}/n{count}/inference.npz')
    result={'arm':arm,'count':count,'identity':identity,'metrics':metrics,'evaluator_sha256':sha(__file__),'protocol_sha256':sha(c.BASE/'evaluation_protocol_v2.json'),'prediction_sha256':sha(dest/'predictions.npz'),'dependencies':dependencies}
    if verify:
        assert result==json.loads((dest/'summary.json').read_text());assert rows==json.loads((dest/'rows.json').read_text())
        if current is not None:
            ex=np.load(dest/'foreground_examples.npz');np.testing.assert_array_equal(current[:24],ex['current']);np.testing.assert_array_equal(after[:24],ex['edited'])
        dump(dest/'verification.json',{'all_rgb_predictions_replayed':len(pred),'all_foreground_diagnostics_and_metric_rows_recomputed':len(rows),'summary_sha256':sha(dest/'summary.json'),'evaluator_sha256':sha(__file__)})
    else:dump(dest/'rows.json',rows);dump(dest/'summary.json',result)
    print('R2 decoder','verify' if verify else 'evaluate',arm,count,'identity' if identity else 'edits',{k:metrics['all'][k] for k in ['changed_mae','preserved_mae','pixel_gate']},flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS+['copy_input'],required=True);p.add_argument('--count',type=int,choices=[2,3,4,6],required=True);p.add_argument('--verify',action='store_true');p.add_argument('--identity',action='store_true');a=p.parse_args();evaluate(a.arm,a.count,a.verify,a.identity)
