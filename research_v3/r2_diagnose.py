"""Post-selection failure diagnosis, without retraining or threshold changes."""
import argparse,json
import numpy as np
import torch
import r2_core as c
import r2_world as w
import r2_evaluate as e
from v3_common import HERE,sha,dump,lock
BASE=w.BASE/'diagnosis'

def freeze():
    choice=json.loads((w.BASE/'selection.json').read_text());assert not choice['development_gate_passed']
    return lock(BASE/'protocol.json',{'source_sha256':sha(__file__),'selection_sha256':sha(w.BASE/'selection.json'),'post_selection_diagnostic':True,
        'questions':['Training versus new n2scenes: underfitting versus generalization gap','Cardinality shortfall versus >4px localized-center failure','Shape/pose/radius/RGB errors conditional on<=4px matching','Temporal reader versus repeated-current-input ablation'],
        'scope':'All6final learned arms on1024training scenes; existing32validation units including temporal input controls. No tuning, model selection or statistical independence claim for training scenes. Raw training predictions saved and fully replayed.'})

def decompose(raw,records):
    rows=[]
    for i,(values,row) in enumerate(zip(raw,records)):
        pred=e.discrete(values);truth=row['objects'];mapping=e.match(pred,truth);n=len(truth);close=[];checks=[]
        for j,g in enumerate(truth):
            p=pred[mapping[j]] if j in mapping else None
            close.append(p is not None and np.hypot(p['x']-g['x'],p['y']-g['y'])<=4)
            checks.append(e.checks(p,g)[:5])
        checks=np.array(checks);close=np.array(close);localized=int(close.sum());short=n-len(mapping);far=len(mapping)-localized
        target=row['target_id'];click=row['click'];selected=int(np.argmin([(p['x']-click[0])**2+(p['y']-click[1])**2 for p in pred])) if pred else -1
        rows.append({'source_index':i,'count':n,'requested_occlusion':row['requested_occlusion'],'same_color':row['same_color'],'continuous_color':row['continuous_color'],'textured_background':row['textured_background'],
            'predicted_count':len(pred),'count_correct':float(len(pred)==n),'full_state_correct':float(len(pred)==n and checks.all()),'target_selection':float(target in mapping and mapping[target]==selected and close[target]),
            'true_objects':n,'unmatched_due_to_cardinality':short,'matched_farther_than4px':far,'localized_objects':localized,'missing_fraction':(short+far)/n,
            **{f'localized_correct_{key}':int(checks[close,j].sum()) for j,key in enumerate(['shape','pose','radius','position1px','rgb'])},
            'localized_all_attributes_correct':int(checks[close].all(-1).sum()),'target_localized':int(close[target]),'target_attributes_correct':int(checks[target].all())})
    return rows

def summarize(rows):
    def stats(rr):
        total=sum(r['true_objects'] for r in rr);localized=sum(r['localized_objects'] for r in rr)
        hist={str(k):sum(r['predicted_count']==k for r in rr) for k in range(7)}
        result={'scenes':len(rr),'objects':total,'count_correct':float(np.mean([r['count_correct'] for r in rr])),
                'full_state_correct':float(np.mean([r['full_state_correct'] for r in rr])),'target_selection':float(np.mean([r['target_selection'] for r in rr])),
                'predicted_count_histogram':hist,'mean_predicted_count':float(np.mean([r['predicted_count'] for r in rr])),
                'unmatched_fraction':sum(r['unmatched_due_to_cardinality'] for r in rr)/total,'mislocalized_fraction':sum(r['matched_farther_than4px'] for r in rr)/total,'localized_fraction':localized/total}
        for key in ['shape','pose','radius','position1px','rgb','all_attributes']:
            col='localized_all_attributes_correct' if key=='all_attributes' else 'localized_correct_'+key
            result['accuracy_given_localized_'+key]=sum(r[col] for r in rr)/localized if localized else None
        return result
    out={'all':stats(rows)}
    for field,vals in [('requested_occlusion',[0.,.25,.5,.75]),('same_color',[0,1]),('continuous_color',[0,1]),('textured_background',[0,1])]:
        for val in vals:out[f'{field}_{val}']=stats([r for r in rows if r[field]==val])
    return out

def run(verify=False):
    freeze();torch.set_num_threads(2);results=[]
    for arm in c.ARMS:
        train=w.BASE/f'data/d{w.DEVELOPMENT}/train/n2';cp=w.BASE/f'runs/d{w.DEVELOPMENT}_s0/{arm}/model.pt'
        model=c.model_for(arm);model.load_state_dict(torch.load(cp,weights_only=True)['state_dict']);model.eval()
        raw,_=e.predict(model,np.load(train/'rgb.npz')['images']);dest=BASE/arm;dest.mkdir(parents=True,exist_ok=True)
        if verify:np.testing.assert_array_equal(raw,np.load(dest/'training_raw.npz')['raw'])
        else:np.savez_compressed(dest/'training_raw.npz',raw=raw)
        groups=[('train',2,arm,raw,train,sha(dest/'training_raw.npz'))]
        for name in [arm]+([arm+'_repeat_current'] if arm.startswith('recurrent') else []):
            for count in [2,3,4,6]:
                pred=w.BASE/f'evaluation/d{w.DEVELOPMENT}_s0/{name}/n{count}/inference.npz';folder=w.BASE/f'data/d{w.DEVELOPMENT}/val/n{count}'
                groups.append(('val',count,name,np.load(pred)['raw'],folder,sha(pred)))
        for split,count,name,raw,folder,digest in groups:
            rows=decompose(raw,json.loads((folder/'labels.json').read_text()));result={'arm':name,'split':split,'count':count,'metrics':summarize(rows),'prediction_sha256':digest,'labels_sha256':sha(folder/'labels.json'),'checkpoint_sha256':sha(cp),'protocol_sha256':sha(BASE/'protocol.json')}
            path=BASE/name/f'{split}_n{count}.json';rowpath=path.with_name(path.stem+'_rows.json')
            if verify:assert result==json.loads(path.read_text());assert rows==json.loads(rowpath.read_text())
            else:dump(path,result);dump(rowpath,rows)
            results.append(result)
        print('R2 diagnosis',arm,'verified' if verify else 'computed',flush=True)
    assert len(results)==38
    if verify:dump(BASE/'verification.json',{'units':38,'training_clips_fully_reinferred':6*1024,'all_decomposition_rows_recomputed':6*1024+32*512,'source_sha256':sha(__file__)})
    else:dump(BASE/'aggregate.json',{'results':results,'post_selection_failure_diagnostic':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args();run(a.verify)
