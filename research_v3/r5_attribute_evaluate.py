"""Read actual generated pixels with a frozen independently calibrated reader."""
import argparse,hashlib,json,time
import numpy as np
import torch
from v3_common import HERE,sha,dump,lock
from r5_intake import BASE,TASKS
from r5_attributes import Reader,crops,folder as reader_folder,schema,load_factors,store_for,summary

def backend(family):
    if family=='dsprites':
        import r5_evaluate as module
    else:
        import r5_3d_evaluate as module
        module.training.FAMILY=family
    return module

def measurements(pred,true,source,clean,names):
    eq=pred==true;changed=source!=true;result=summary(pred.reshape(-1,len(names)),true.reshape(-1,len(names)),names)
    result['both_objects_all_attributes_accuracy']=float(eq.all((1,2)).mean())
    result['clean_target_both_objects_accuracy']=float((clean==true).all((1,2)).mean())
    result['agreement_with_clean_target_readout']=float((pred==clean).all((1,2)).mean())
    result['changed_attribute_accuracy']={n:float(eq[:,:,i][changed[:,:,i]].mean()) if changed[:,:,i].any() else None for i,n in enumerate(names)}
    result['preserved_attribute_accuracy']={n:float(eq[:,:,i][~changed[:,:,i]].mean()) if (~changed[:,:,i]).any() else None for i,n in enumerate(names)}
    result['changed_attribute_counts']={n:int(changed[:,:,i].sum()) for i,n in enumerate(names)}
    return result

@torch.no_grad()
def evaluate(family,task,kind,mode,init=0,verify=False,split_seed=890101):
    assert torch.backends.mps.is_available();torch.set_num_threads(2);device=torch.device('mps');mod=backend(family);mod.training.SPLIT_SEED=split_seed;mod.freeze()
    target=mod.unit(task,kind,init,mode);pixel=json.loads((target/'summary.json').read_text());pixel_verify=json.loads((target/'verification.json').read_text())
    assert pixel['provenance']==mod.provenance(task,kind,init,mode)
    assert pixel_verify['summary_sha256']==sha(target/'summary.json') and pixel_verify['all_RGBoutputs_replayed']==8000
    reader_dir=reader_folder(family,task);cal=json.loads((reader_dir/'calibration_test.json').read_text());cal_verify=json.loads((reader_dir/'calibration_test_verification.json').read_text())
    assert cal_verify['summary_sha256']==sha(reader_dir/'calibration_test.json') and cal_verify['all_rows_bitexact']
    assert cal['checkpoint_sha256']==sha(reader_dir/'reader.pt') and cal['predictions_sha256']==sha(reader_dir/'calibration_test.npz')
    names,counts=schema(family);model=mod.load(task,kind,init,mode,device);reader=Reader(counts)
    reader.load_state_dict(torch.load(reader_dir/'reader.pt',weights_only=True)['state_dict']);reader.eval()
    out=target/'attributes';out.mkdir(exist_ok=True)
    protocol={'source_sha256':sha(__file__),'reader_source_sha256':sha(HERE/'r5_attributes.py'),'pixel_summary_sha256':sha(target/'summary.json'),
        'pixel_verification_sha256':sha(target/'verification.json'),'reader_checkpoint_sha256':sha(reader_dir/'reader.pt'),'reader_protocol_sha256':sha(reader_dir/'protocol.json'),
        'clean_test_calibration_sha256':sha(reader_dir/'calibration_test.json'),'provenance':mod.provenance(task,kind,init,mode),
        'scope':'Every8000officialtestprediction,2objects,allattributes; knownSOURCEcenters andfixed112crop. Same frozen reader acrossallmodels. Not a detector, independently recruited person, or perfect semantic oracle.',
        'verification':'Regenerate everyprediction usingoriginalMPSbatch32; require all250float32batchSHA256values match original pixel evaluation before readingattributes. IndependentCPUreader; replay allattribute labels andconfidences.',
        'selection':'Diagnostic only after validation-only pixel candidate locking. Attribute testaccuracy never changes checkpoints,primarymodel,candidateorcriteria.'}
    lock(out/'protocol.json',protocol)
    if (out/('verification.json' if verify else 'summary.json')).exists():
        saved=json.loads((out/'summary.json').read_text());assert saved['protocol_sha256']==sha(out/'protocol.json') and saved['rows_sha256']==sha(out/'rows.npz')
        if verify:assert json.loads((out/'verification.json').read_text())['summary_sha256']==sha(out/'summary.json')
        return
    store=store_for(family,task);labels,centers=load_factors(family,task);allpred=[];confidence=[];expected=json.loads((target/'batch_hashes.json').read_text());started=time.perf_counter();matched=0
    for first in range(0,8000,32):
        ids=np.arange(first,first+32);x,y=store.batch('test',ids,device);pred=x if kind=='copy' else model(x);values=pred.cpu()
        assert hashlib.sha256(values.numpy().tobytes()).hexdigest()==expected[first//32],'Prediction differs from fully verifiedpixelunit; inspect before attribute scoring'
        matched+=1;xy=torch.from_numpy(centers[64000+ids].astype(np.float32)).reshape(-1,2)
        region=crops(values[:,None].expand(-1,2,-1,-1,-1).reshape(-1,3,128,128),xy);logits=reader(region)
        allpred.append(torch.stack([v.argmax(1) for v in logits],1).reshape(32,2,len(names)).numpy())
        confidence.append(torch.stack([v.softmax(1).max(1).values for v in logits],1).reshape(32,2,len(names)).numpy())
    store.close();p=np.concatenate(allpred);conf=np.concatenate(confidence);truth=labels[64000:,1];source=labels[64000:,0]
    clean=np.load(reader_dir/'calibration_test.npz')['prediction'].reshape(8000,2,2,len(names))[:,1]
    metrics=measurements(p,truth,source,clean,names)
    if verify:
        saved=json.loads((out/'summary.json').read_text());assert saved['protocol_sha256']==sha(out/'protocol.json') and saved['rows_sha256']==sha(out/'rows.npz')
        z=np.load(out/'rows.npz')
        for name,array in [('prediction',p),('truth',truth),('source',source),('clean_readout',clean)]:assert np.array_equal(z[name],array)
        np.testing.assert_allclose(z['confidence'],conf,rtol=0,atol=1e-6);assert saved['metrics']==metrics
        dump(out/'verification.json',{'summary_sha256':sha(out/'summary.json'),'all_generated_images_replayed':8000,'all_object_readouts_replayed':16000,'input_batches_bitexact':matched,'confidence_max_abs_difference':float(np.max(np.abs(z['confidence']-conf))),'seconds':time.perf_counter()-started})
    else:
        np.savez_compressed(out/'rows.npz',prediction=p,truth=truth,source=source,clean_readout=clean,confidence=conf)
        dump(out/'summary.json',{'protocol_sha256':sha(out/'protocol.json'),'rows_sha256':sha(out/'rows.npz'),'metrics':metrics,'input_batches_bitexact':matched,'seconds':time.perf_counter()-started,'calibration_target_accuracy':cal['target']})
    print('R5 generated attributes',family,task,kind,mode,'verify',verify,metrics,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['dsprites','clevr','clevrtex'],default='dsprites');p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--kind',choices=['plain','c4','object','copy'],required=True);p.add_argument('--mode',choices=['validation_selected','fixed20epoch'],default='validation_selected');p.add_argument('--init',type=int,default=0);p.add_argument('--split-seed',type=int,choices=[890101,890201,890202,890203],default=890101);p.add_argument('--verify',action='store_true');a=p.parse_args();evaluate(a.family,a.task,a.kind,a.mode,a.init,a.verify,a.split_seed)
