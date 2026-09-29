"""Calibrated rejection diagnostics; never called a substitute for human review."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from common import HERE, dump, fresh_dir, original_orbits, render_fresh, np, torch, r
import generation as g
from diagnostics import png, pil_tensor, generated_images


class DetailedEvaluator(g.StateEvaluator):
    def predict(self,images):
        rows=[]
        for mask,color,cx,cy,occupancy,scale in g.state_features(images):
            intersection=(self.templates*mask).sum((1,2))
            union=np.maximum(self.templates,mask).sum((1,2))
            scores=intersection/np.maximum(union,1)
            by_kind=np.array([scores[self.kinds==k].max() for k in range(4)])
            order=np.argsort(by_kind)
            rows.append({'predicted_shape':int(order[-1]),'predicted_color':int(((self.palette-color)**2).sum(1).argmin()),
                'template_iou':float(by_kind[order[-1]]),'class_margin':float(by_kind[order[-1]]-by_kind[order[-2]]),
                'cx':cx,'cy':cy,'occupancy':occupancy,'scale':scale,'nonempty':occupancy>0})
        return rows


def accepted(row,threshold):
    return bool(row['nonempty'] and row['occupancy']<=threshold['max_occupancy'] and
                row['template_iou']>=threshold['min_iou'] and row['class_margin']>=threshold['min_margin'])


def negative_images(seed,n=64):
    rng=np.random.default_rng(seed);images=[];kinds=[]
    for i in range(n):
        for kind in ['empty','disk','rectangle','random_noise']:
            p=Image.new('RGB',(64,64));d=ImageDraw.Draw(p);cx,cy=rng.integers(18,46,size=2)
            radius=int(rng.integers(5,16));color=tuple(int(c) for c in rng.integers(65,226,size=3))
            box=(int(cx-radius),int(cy-radius),int(cx+radius),int(cy+radius))
            if kind=='disk':d.ellipse(box,fill=color)
            if kind=='rectangle':d.rectangle(box,fill=color)
            if kind=='random_noise':p=Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8))
            images.append(pil_tensor(p));kinds.append(kind)
    return torch.stack(images),kinds


def population(seed,ev,forbidden):
    xs=[];ys=[];meta=[]
    for split in ['test','ood']:
        x,y,m=render_fresh(128,seed,split,forbidden)
        forbidden.update(v['orbit_sha256'] for v in m)
        xs.append(x);ys.append(y);meta.extend(m)
    base=torch.cat(xs);y=torch.cat(ys).tolist();rows=[]
    for name,x in [('clean',base),('mild_blur',torch.stack([pil_tensor(png(v).filter(ImageFilter.GaussianBlur(1.2))) for v in base]))]:
        for i,(p,label) in enumerate(zip(ev.predict(x),y)):
            rows.append({'population':name,'base_id':i,'known_shape':label[0],'known_color':label[1],**p})
    neg,kinds=negative_images(seed+100)
    for i,(p,kind) in enumerate(zip(ev.predict(neg),kinds)):
        rows.append({'population':kind,'base_id':i,'known_shape':-1,'known_color':-1,**p})
    return rows,meta


def measure(rows,threshold):
    result={}
    for kind in sorted({v['population'] for v in rows}):
        subset=[v for v in rows if v['population']==kind];take=[v for v in subset if accepted(v,threshold)]
        result[kind]={'n':len(subset),'accepted_n':len(take),'accepted_fraction':len(take)/len(subset),
            'shape_accuracy_among_accepted':float(np.mean([v['known_shape']==v['predicted_shape'] for v in take])) if take and subset[0]['known_shape']>=0 else None}
    return result


def run(out,device):
    settings={'calibration_seed':9212304,'test_seed':9212305,'iou_grid':[.65,.70,.75,.80,.85,.90],
        'margin_grid':[0,.02,.04,.06,.08],'max_occupancy':.35,
        'selection':'On calibration only: require >=95% clean retention and <=1% synthetic negative acceptance; among feasible choose highest mild-blur retention, then clean retention, then smaller threshold complexity. No tuning on generated images.',
        'limit':'Shape acceptance screen; not a validated semantic quality metric for arbitrary generated images.'}
    dump(out/'settings.json',settings)
    ev=DetailedEvaluator();forbidden=original_orbits()
    for p in (HERE/'data/probe_v1').glob('*_metadata.json'):
        forbidden.update(v['orbit_sha256'] for v in json.loads(p.read_text()))
    audit_sources=HERE/'reports/evaluator_audit_v1/sources.json'
    if audit_sources.exists():forbidden.update(v['orbit_sha256'] for v in json.loads(audit_sources.read_text()))
    cal,meta=population(settings['calibration_seed'],ev,forbidden)
    dump(out/'calibration_sources.json',meta)
    candidates=[]
    for iou in settings['iou_grid']:
        for margin in settings['margin_grid']:
            t={'min_iou':iou,'min_margin':margin,'max_occupancy':settings['max_occupancy']}
            m=measure(cal,t);neg=sum(m[k]['accepted_n'] for k in ['empty','disk','rectangle','random_noise'])/256
            candidates.append({'threshold':t,'metrics':m,'negative_acceptance':neg,
                               'feasible':m['clean']['accepted_fraction']>=.95 and neg<=.01})
    feasible=[c for c in candidates if c['feasible']]
    dump(out/'calibration_candidates.json',candidates)
    if not feasible:
        dump(out/'summary.json',{'status':'no_threshold_satisfies_prespecified_gate','generated_evaluation':'not performed','settings':settings})
        raise RuntimeError('No feasible calibration threshold: do not silently relax gate')
    best=max(feasible,key=lambda c:(c['metrics']['mild_blur']['accepted_fraction'],c['metrics']['clean']['accepted_fraction'],-c['threshold']['min_iou'],-c['threshold']['min_margin']))
    threshold=best['threshold'];dump(out/'locked_threshold.json',threshold)
    test,test_meta=population(settings['test_seed'],ev,forbidden)
    dump(out/'test_sources.json',test_meta)
    r.write_csv(out/'calibration_rows.csv',cal);r.write_csv(out/'test_rows.csv',test)
    model_results={};generated_rows=[]
    for model in ['aug','equivariant']:
        for seed in [0,1,2]:
            images,y=generated_images(model,seed,device);pred=ev.predict(images)
            for i,(p,label) in enumerate(zip(pred,y.tolist())):
                joint=p['predicted_shape']==label[0] and p['predicted_color']==label[1]
                generated_rows.append({'model':model,'seed':seed,'sample':i,'ood':label[0]==label[1],
                    'shape':label[0],'color':label[1],**p,'joint_correct':joint,
                    'legacy_valid':p['nonempty'] and p['template_iou']>=.65,
                    'strict_accepted':accepted(p,threshold),'strict_accepted_and_joint':bool(joint and accepted(p,threshold))})
            print('new screen',model,seed,flush=True)
    for model in ['aug','equivariant']:
        for split in ['seen','ood']:
            subset=[v for v in generated_rows if v['model']==model and v['ood']==(split=='ood')]
            keys=['joint_correct','legacy_valid','strict_accepted','strict_accepted_and_joint']
            model_results[f'{model}/{split}']={k:{'mean':float(np.mean([v[k] for v in subset])),
                'per_seed':[float(np.mean([v[k] for v in subset if v['seed']==seed])) for seed in [0,1,2]]} for k in keys}
    r.write_csv(out/'generated_rows.csv',generated_rows)
    summary={'status':'calibrated_and_evaluated','threshold':threshold,'calibration':best['metrics'],'test':measure(test,threshold),
        'generated':model_results,'human_validated':False,'settings':settings,
        'completed_unix':time.time(),'code_sha256':r.sha(__file__)}
    dump(out/'summary.json',summary)
    from common import update_status
    update_status('A03b','complete',[str((out/'summary.json').relative_to(HERE))],
                  'Threshold calibrated on fresh clean/mild/negative controls; generated-image human validation remains pending.')
    print(json.dumps({'threshold':threshold,'test':summary['test'],'generated':model_results},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--device',default='mps');a=p.parse_args()
    from common import o
    torch.set_num_threads(4);run(fresh_dir(a.out),o.choose_device(a.device))
