"""Exhaustive R2 data, inference, calibration and metric replay.

Writes only new verification records; frozen training/evaluation inputs remain
unchanged. A saved summary alone is not treated as successful verification.
"""
import argparse,json
import numpy as np
import torch
import r2_core as c
import r2_world as w
import r2_data as data
import r2_evaluate as e
from v3_common import HERE,sha,dump

def verify_data():
    data.freeze();seen=set();checks=[];frames=0
    for path in sorted((w.BASE/'data').glob('d*/*/n*/manifest.json')):
        m=json.loads(path.read_text());folder=path.parent
        assert m['protocol_sha256']==sha(w.BASE/'data_protocol.json')
        for name,digest in m['files'].items():assert sha(folder/name)==digest
        records=json.loads((folder/'labels.json').read_text())
        rgb=np.load(folder/'rgb.npz')['images'];masks=np.load(folder/'masks.npz')
        assert rgb.shape==(len(records),4,64,64,3)
        for i,row in enumerate(records):
            cond=(row['same_color'],row['continuous_color'],row['textured_background'],row['requested_occlusion'])
            assert cond==w.CONDITIONS[i%len(w.CONDITIONS)]
            images,amodal,visible,replay=w.sample(m['seed'],m['split'],row['candidate_index'],m['count'],cond)
            replay['candidate_index']=row['candidate_index'];replay['source_index']=i
            assert replay==row,(path,i,'labels')
            np.testing.assert_array_equal(images,rgb[i])
            np.testing.assert_array_equal(np.rint(amodal*255).astype(np.uint8),masks['amodal'][i])
            np.testing.assert_array_equal(np.rint(visible*255).astype(np.uint8),masks['visible'][i])
            key=w.vector_key(row['objects']);assert key==row['family_sha256'] and key not in seen;seen.add(key)
            assert row['window_order']=='current,t-1,t-2,t-3' and row['current_reference_index']==0
            bg=w.background(row['background_seed'],row['textured_background'])
            # The oldest observation has no occluder, an intentional favorable
            # visibility assumption, not evidence for permanently hidden objects.
            assert not w.render(row['objects'],bg,row['requested_occlusion'],row['target_id'],3)[3].any()
            for v,a in zip(row['objects'],w.layers(row['objects'],3)):
                assert 0<=v['x']-3*v['vx']<64 and 0<=v['y']-3*v['vy']<64
            frames+=4
        checks.append({'group':str(folder.relative_to(w.BASE)),'clips':len(records),'all_rgb_frames_masks_and_labels_replayed':True,'manifest_sha256':sha(path)})
        print('R2 verified data',checks[-1]['group'],len(records),flush=True)
    assert len(checks)>=6 and len(seen)>=3200
    result={'groups':checks,'clips':len(seen),'rgb_frames':frames,'cross_new_family_overlap':0,'source_sha256':sha(__file__),'data_protocol_sha256':sha(w.BASE/'data_protocol.json')}
    dump(w.BASE/'data_verification.json',result);return result

def verify_models():
    c.freeze();torch.set_num_threads(2)
    frozen=json.loads((w.BASE/'evaluation_protocol.json').read_text())
    assert frozen['evaluator_sha256']==sha(HERE/'r2_evaluate.py')
    assert frozen['training_protocol_sha256']==sha(w.BASE/'training_protocol.json')
    choice=json.loads((w.BASE/'selection.json').read_text());assert not choice['development_gate_passed'], 'Confirmation runs require extending expected-unit audit before closure'
    verified=[];calibrations=[];summaries=[]
    for arm in c.ARMS:
        run=w.BASE/f'runs/d{w.DEVELOPMENT}_s0/{arm}';rr=json.loads((run/'run.json').read_text());cp=run/'model.pt'
        assert rr['checkpoint_sha256']==sha(cp) and rr['steps']==4000
        assert rr['protocol_sha256']==sha(w.BASE/'training_protocol.json')
        assert rr['data_sha256']==sha(w.BASE/f'data/d{w.DEVELOPMENT}/train/n2/rgb.npz')
        model=c.model_for(arm);model.load_state_dict(torch.load(cp,weights_only=True)['state_dict']);model.eval()
        for repeat in ([False,True] if arm.startswith('recurrent') else [False]):
            name=arm+('_repeat_current' if repeat else '')
            output=w.BASE/f'evaluation/d{w.DEVELOPMENT}_s0/{name}'
            folder=w.BASE/f'data/d{w.DEVELOPMENT}/calibration/n2';images=np.load(folder/'rgb.npz')['images']
            raw,_=e.predict(model,images,repeat);np.testing.assert_array_equal(raw,np.load(output/'calibration_inference.npz')['raw'])
            cal=e.calibration(raw,json.loads((folder/'labels.json').read_text()));assert cal==json.loads((output/'calibration.json').read_text())
            calibrations.append({'arm':name,'all_inferences_replayed':len(images),'calibration_sha256':sha(output/'calibration.json')})
            for count in [2,3,4,6]:
                folder=w.BASE/f'data/d{w.DEVELOPMENT}/val/n{count}';dest=output/f'n{count}'
                rec=json.loads((dest/'summary.json').read_text())
                assert rec['model_sha256']==sha(cp) and rec['input_sha256']==sha(folder/'rgb.npz')
                assert rec['inference_sha256']==sha(dest/'inference.npz') and rec['evaluator_sha256']==sha(HERE/'r2_evaluate.py')
                assert rec['calibration_sha256']==sha(output/'calibration.json')
                images=np.load(folder/'rgb.npz')['images'];raw,masks=e.predict(model,images,repeat)
                saved=np.load(dest/'inference.npz');np.testing.assert_array_equal(raw,saved['raw']);np.testing.assert_array_equal(masks,saved['masks'])
                rows=e.score(raw,masks,json.loads((folder/'labels.json').read_text()),np.load(folder/'masks.npz')['visible'].astype(np.float32)/255,cal)
                assert rows==json.loads((dest/'rows.json').read_text());assert e.summarize(rows)==rec['metrics']
                verified.append({'arm':name,'count':count,'all_inference_and_metric_rows_replayed':len(rows),'summary_sha256':sha(dest/'summary.json')});summaries.append(rec)
                print('R2 verified inference',name,count,flush=True)
    assert len(verified)==32 and len(calibrations)==8
    # Recompute the frozen gate, including tie rules, without changing its file.
    from r2_evaluation_driver import select
    select()
    for rel,digest in choice['sources'].items():assert sha(w.BASE/rel)==digest
    dump(w.BASE/'reader_aggregate.json',{'development':summaries,'confirmation':[],'selection':choice,'scope':'State and fixed analytic-command diagnostics; excludes learned pixel editing.'})
    dump(w.BASE/'reader_verification.json',{'conditions':verified,'calibrations':calibrations,'units':32,'expected_units':32,'all_eval_scenes':sum(x['all_inference_and_metric_rows_replayed'] for x in verified),'source_sha256':sha(__file__),'gate_recomputed':True,'independent_retraining':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-only',action='store_true');p.add_argument('--models-only',action='store_true');a=p.parse_args()
    if not a.models_only:verify_data()
    if not a.data_only:verify_models()
