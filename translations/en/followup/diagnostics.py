"""Independent evaluator stress tests, blind review, and frozen-AE diagnostics."""
from __future__ import annotations
import argparse
import base64
import csv
import io
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from common import HERE, ROOT, dump, fresh_dir, original_orbits, render_fresh, update_status, np, torch, o, r
import generation as g

CFG = json.loads((HERE / 'configs/stage_a_v1.json').read_text())


def provenance():
    return {'code_sha256': r.sha(__file__), 'config_sha256': r.sha(HERE / 'configs/stage_a_v1.json'),
            'completed_unix': time.time()}


def png(image):
    a = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(a)


def pil_tensor(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255


def prepared_probe_data():
    folder = HERE / 'data/probe_v1'
    if (folder / 'manifest.json').exists():
        m = json.loads((folder / 'manifest.json').read_text())
        for split, h in m['archive_hashes'].items():
            assert r.sha(folder / f'{split}.npz') == h
        return folder
    fresh_dir(folder)
    forbidden = original_orbits()
    x, y, meta = r.load_data('train', CFG['old_train_probe_n'])
    sets = {'train': (x, y, meta)}
    for split in ['val', 'test', 'ood']:
        sets[split] = render_fresh(CFG['fresh_probe_n_per_split'], CFG['fresh_probe_seed'], split, forbidden)
        forbidden.update(v['orbit_sha256'] for v in sets[split][2])
    hashes = {}
    for split, (x, y, meta) in sets.items():
        np.savez_compressed(folder / f'{split}.npz', images=(x.permute(0, 2, 3, 1).numpy()*255).round().astype(np.uint8), labels=y.numpy())
        dump(folder / f'{split}_metadata.json', meta)
        hashes[split] = r.sha(folder / f'{split}.npz')
    dump(folder / 'manifest.json', {'seed': CFG['fresh_probe_seed'], 'train': 'original training prefix only',
         'archive_hashes': hashes, 'new_splits_disjoint_from_all_original_orbits': True,
         'new_base_scenes': sum(len(sets[s][0]) for s in ['val','test','ood']),
         'note': 'Diagnostic set, not a permanently untouched confirmatory test after this analysis.'})
    update_status('A02', 'complete', ['configs/stage_a_v1.json', 'data/probe_v1/manifest.json'])
    return folder


def probe_data(split):
    p = prepared_probe_data()
    d = np.load(p / f'{split}.npz')
    return torch.from_numpy(d['images'].copy()).permute(0,3,1,2).float()/255, torch.from_numpy(d['labels'].copy())


def audit(out):
    ev = g.StateEvaluator()
    xs, ys, meta = [], [], []
    seen = original_orbits()
    for split in ['test', 'ood']:
        x, y, m = render_fresh(CFG['audit_n_per_split'], CFG['audit_seed'], split, seen)
        seen.update(v['orbit_sha256'] for v in m)
        xs.append(x); ys.append(y); meta.extend(m)
    base, labels = torch.cat(xs), torch.cat(ys)
    images = torch.cat([o.rotate(base, k) for k in range(4)])
    labels = labels.repeat(4,1)
    dump(out / 'sources.json', meta)
    transforms = [('clean', 0), ('blur_0.6', .6), ('blur_1.2', 1.2), ('blur_2.5', 2.5), ('blur_4.0', 4.0),
                  ('erode_1', None), ('fragment', None), ('color_mix', None)]
    rows, summary = [], {}
    for name, blur in transforms:
        batch = []
        for image in images:
            p = png(image)
            if blur:
                p = p.filter(ImageFilter.GaussianBlur(blur))
            elif name == 'erode_1':
                p = p.filter(ImageFilter.MinFilter(3))
            a = pil_tensor(p)
            if name == 'fragment':
                _, _, cx, cy, _, _ = g.state_features(image[None])[0]
                c = round(cx*64)
                a[:, :, max(0,c-2):min(64,c+3)] = 0
            elif name == 'color_mix':
                a = (a + a.roll(1, dims=0)) / 2
            batch.append(a)
        changed = torch.stack(batch)
        pred = ev.predict(changed)
        sr = []
        for i, (p, y) in enumerate(zip(pred, labels.tolist())):
            row = {'condition': name, 'base_id': i % len(base), 'rotation': i // len(base),
                   'source_shape': y[0], 'source_color': y[1], **p,
                   'source_shape_agreement': int(p['predicted_shape']==y[0]),
                   'source_color_agreement': int(p['predicted_color']==y[1])}
            row['valid_065'] = bool(p['nonempty'] and p['template_iou'] >= .65)
            rows.append(row); sr.append(row)
        summary[name] = {'n': len(sr), 'source_shape_agreement': float(np.mean([v['source_shape_agreement'] for v in sr])),
                         'source_color_agreement': float(np.mean([v['source_color_agreement'] for v in sr])),
                         'valid_fraction': float(np.mean([v['valid_065'] for v in sr])),
                         'source_shape_agreement_among_accepted': float(np.mean([v['source_shape_agreement'] for v in sr if v['valid_065']])) if any(v['valid_065'] for v in sr) else None}
        o.grid(changed[:32], out / f'{name}.png', 8)
        print('audit', name, summary[name], flush=True)
    # Known out-of-vocabulary silhouettes. The evaluator must be allowed to reject them.
    negatives, negative_types = [], []
    rng = np.random.default_rng(9212351)
    for i in range(32):
        for kind in ['empty', 'disk', 'rectangle', 'random_noise']:
            p = Image.new('RGB', (64,64)); draw = ImageDraw.Draw(p)
            color = o.PALETTE[i % 6]; cx, cy = rng.integers(22,42,size=2); radius = int(rng.integers(7,14))
            box = (int(cx-radius),int(cy-radius),int(cx+radius),int(cy+radius))
            if kind=='disk': draw.ellipse(box, fill=color)
            if kind=='rectangle': draw.rectangle(box, fill=color)
            if kind=='random_noise': p = Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8))
            negatives.append(pil_tensor(p)); negative_types.append(kind)
    negative_preds = ev.predict(torch.stack(negatives))
    negative_rows = [{'kind': k, **p} for k,p in zip(negative_types,negative_preds)]
    neg_summary = {kind: {str(t): float(np.mean([p['nonempty'] and p['template_iou'] >= t for p in negative_rows if p['kind']==kind]))
                          for t in CFG['template_validity_thresholds']} for kind in sorted(set(negative_types))}
    r.write_csv(out / 'stress_rows.csv', rows); r.write_csv(out / 'negative_rows.csv', negative_rows)
    o.grid(torch.stack(negatives[:32]), out / 'negatives.png', 8)
    dump(out / 'summary.json', {'transforms': summary, 'negative_false_acceptance': neg_summary,
         'interpretation': 'Source-label agreement, not semantic accuracy for severe corruption or color mixing. Clean and mild perturbations are diagnostics; generated-image human validation remains separate.',
         'base_scenes': len(base), 'rotations_per_scene': 4, 'original_orbit_overlap': 0, **provenance()})
    update_status('A03', 'complete', [str((out/'summary.json').relative_to(HERE))],
                  'Stress and negative controls complete; does not validate human agreement on generated images.')


@torch.no_grad()
def generated_images(model, seed, device):
    folder = ROOT / f'runs/flow_{model}_s{seed}'
    saved = torch.load(folder / 'samples/samples.pt', map_location='cpu', weights_only=True)
    ck = torch.load(folder / 'flow.pt', map_location='cpu', weights_only=True)
    assert r.sha(ck['ae_path']) == ck['ae_sha256']
    ae, _ = r.load_model(ck['ae_path'], device)
    z = saved['latents'][64]
    images = torch.cat([ae.decode(z[i:i+64].to(device)).cpu() for i in range(0,len(z),64)])
    return images, saved['labels']


def review(out, device):
    rng = np.random.default_rng(CFG['review_shuffle_seed'])
    entries = []
    for model in ['aug', 'equivariant']:
        for seed in CFG['ae_seeds']:
            images, labels = generated_images(model, seed, device)
            ood = (labels[:,0] == labels[:,1]).numpy()
            for split in ['seen', 'ood']:
                indices = rng.choice(np.where(ood == (split=='ood'))[0], CFG['human_review_n_per_model_seed_split'], replace=False)
                for i in indices:
                    entries.append({'model': model, 'seed': seed, 'split': split, 'source_index': int(i),
                                    'target_shape': int(labels[i,0]), 'target_color': int(labels[i,1]), 'image': images[i]})
    rng.shuffle(entries)
    public, private = [], []
    for n,e in enumerate(entries):
        id_ = f'Q{n+1:03d}'
        b = io.BytesIO(); png(e['image']).save(b, format='PNG')
        public.append({'id': id_, 'png': base64.b64encode(b.getvalue()).decode()})
        private.append({'id': id_, **{k:v for k,v in e.items() if k!='image'}})
    r.write_csv(out / 'private_key.csv', private)
    body = r'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Shape review</title>
<style>body{font:17px system-ui;max-width:760px;margin:30px auto;line-height:1.6;background:#fafafa;color:#222}img{width:256px;height:256px;image-rendering:pixelated;background:#000}button,select,input{font:inherit;padding:7px;margin:5px}button{cursor:pointer}small{display:block;color:#555}</style>
<h1>Shape review</h1><p>Judge the visible image without seeing the model or requested condition. If the shape is unclear, select “Unclear.” Choose both shape and color before moving to the next image.</p>
<label>Reviewer name or ID <input id="reviewer"></label><p id="progress"></p><img id="picture"><br>
<label>Shape <select id="shape"><option value="">Select</option><option value="0">L</option><option value="1">T</option><option value="2">Arrow</option><option value="3">Zigzag</option><option value="-1">Unclear / other shape</option></select></label><br>
<label>Color <select id="color"><option value="">Select</option><option value="0">Red</option><option value="1">Green</option><option value="2">Blue</option><option value="3">Yellow</option><option value="4">Purple</option><option value="5">Cyan</option><option value="-1">Unclear / mixed color</option></select></label><br>
<label>Validity <select id="valid"><option value="">Select</option><option value="1">One clear shape</option><option value="0">Broken / blotchy / multiple pieces</option><option value="-1">Unclear</option></select></label><br>
<button id="prev">Previous</button><button id="next">Save and next</button><button id="export">Download responses CSV</button><small>Responses are saved in this browser. Download the file to provide them as research data. Requested conditions and model information are hidden from this page.</small>
<script>const items=__ITEMS__;const key='orbitlab-review-9212399-v1';let answers=JSON.parse(localStorage.getItem(key)||'{}'),index=0;
const el=id=>document.getElementById(id);function show(){const q=items[index],a=answers[q.id]||{};el('picture').src='data:image/png;base64,'+q.png;el('progress').textContent=`${index+1} / ${items.length} · ${Object.keys(answers).length} responses`;['shape','color','valid'].forEach(k=>el(k).value=a[k]??'');}
function save(){if(['shape','color','valid'].some(k=>el(k).value==='')){alert('Please select all three fields.');return false;}answers[items[index].id]={shape:el('shape').value,color:el('color').value,valid:el('valid').value};localStorage.setItem(key,JSON.stringify(answers));return true;}
el('next').onclick=()=>{if(save()){index=Math.min(index+1,items.length-1);show();}};el('prev').onclick=()=>{index=Math.max(0,index-1);show();};el('export').onclick=()=>{const reviewer=el('reviewer').value.trim();if(!reviewer){alert('Please enter a reviewer ID.');return;}const quote=v=>'"'+String(v).replaceAll('"','""')+'"';const rows=[['id','reviewer','shape','color','valid','observer_type'],...Object.entries(answers).map(([id,a])=>[id,reviewer,a.shape,a.color,a.valid,'human_self_reported'])];const blob=new Blob(['\ufeff'+rows.map(r=>r.map(quote).join(',')).join('\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='orbitlab-human-review.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};show();</script></html>'''
    (out / 'review.html').write_text(body.replace('__ITEMS__',json.dumps(public)))
    dump(out / 'manifest.json', {'n':len(public), 'strata':'2 models x 3 seeds x 2 splits x 20 samples',
         'human_responses_received':0, 'status':'prepared_not_human_validated',
         'public_document_contains_model_or_target_labels':False, **provenance()})
    update_status('A04', 'awaiting_human_review', [str((out/'review.html').relative_to(HERE)), str((out/'manifest.json').relative_to(HERE))],
                  '240 blinded items prepared; no human judgments have been received. Other stages may proceed.')
    print('review prepared', len(public), flush=True)


@torch.no_grad()
def diagnose_decode(out, device):
    ev = g.StateEvaluator(); rows = []; branch_rows = []
    prepared_probe_data()
    for model in ['aug','equivariant']:
        for seed in CFG['ae_seeds']:
            ae_path = ROOT / f'runs/repair_{model}_n1024_s{seed}/ae.pt'
            ae,_ = r.load_model(ae_path, device)
            for split in ['test','ood']:
                x,y = probe_data(split)
                for rotation in range(4):
                    for i in range(0,len(x),64):
                        target = o.rotate(x[i:i+64].to(device),rotation)
                        z = ae.encode(target); pred = ae.decode(z)
                        pm = r.pixel_metrics(pred,target); evaluations = ev.predict(pred)
                        for j, (p,label) in enumerate(zip(evaluations,y[i:i+64].tolist())):
                            rows.append({'model':model,'seed':seed,'source':'reconstruction','split':split,'base_id':i+j,'rotation':rotation,
                                 'shape_correct':int(p['predicted_shape']==label[0]),'color_correct':int(p['predicted_color']==label[1]),
                                 'valid_joint':int(p['predicted_shape']==label[0] and p['predicted_color']==label[1] and p['template_iou']>=.65),
                                 'template_iou':p['template_iou'], **{k:float(v[j]) for k,v in pm.items()}})
                        if model == 'equivariant':
                            b = ae.decoder(z.reshape(-1,16)).reshape(len(z),4,3,64,64)
                            branches = torch.stack([o.rotate(b[:,k],k) for k in range(4)])
                            mean = branches.mean(0)
                            branch_mse = (branches-target[None]).square().mean((2,3,4)).mean(0)
                            mean_mse = (mean-target).square().mean((1,2,3))
                            disagreement = (branches-mean[None]).square().mean((2,3,4)).mean(0)
                            assert torch.allclose(branch_mse,mean_mse+disagreement,atol=1e-7,rtol=1e-4)
                            for j in range(len(z)):
                                branch_rows.append({'seed':seed,'split':split,'base_id':i+j,'rotation':rotation,
                                    'branch_mse_mean':float(branch_mse[j]),'ensemble_mse':float(mean_mse[j]),'branch_disagreement':float(disagreement[j])})
                            if i==0 and rotation==0:
                                o.grid(torch.cat([target[:8],mean[:8]]+[br[:8] for br in branches]),out/f'branches_s{seed}_{split}.png',8)
                if seed == 0:
                    o.grid(torch.cat([x[:16],ae(x[:16].to(device)).cpu()]),out/f'{model}_{split}_reconstruction.png',8)
            images, labels = generated_images(model, seed, device)
            for j,(p,label) in enumerate(zip(ev.predict(images),labels.tolist())):
                rows.append({'model':model,'seed':seed,'source':'generation','split':'ood' if label[0]==label[1] else 'test',
                    'base_id':j,'rotation':-1,'shape_correct':int(p['predicted_shape']==label[0]),'color_correct':int(p['predicted_color']==label[1]),
                    'valid_joint':int(p['predicted_shape']==label[0] and p['predicted_color']==label[1] and p['template_iou']>=.65),
                    'template_iou':p['template_iou'], **{k:'' for k in ['mse','psnr','foreground_mae','mask_iou','centroid_error','black_mse','black_foreground_mae']}})
            print('decode diagnostic',model,seed,flush=True)
    summary = {}
    for model in ['aug','equivariant']:
        for source in ['reconstruction','generation']:
            for split in ['test','ood']:
                selected=[v for v in rows if v['model']==model and v['source']==source and v['split']==split]
                summary[f'{model}/{source}/{split}']={'n_rows':len(selected),**{k:float(np.mean([v[k] for v in selected])) for k in ['shape_correct','color_correct','valid_joint','template_iou']}}
    r.write_csv(out/'rows.csv',rows); r.write_csv(out/'branch_rows.csv',branch_rows)
    dump(out/'summary.json',{'conditions':summary,'branches':{k:float(np.mean([v[k] for v in branch_rows])) for k in ['branch_mse_mean','ensemble_mse','branch_disagreement']},
         'interpretation':'Means across 3 seeds and 4 rotations; row counts are not independent sample counts. Branch diagnostics do not prove a different decoder architecture would be better.',**provenance()})
    update_status('A06','complete',[str((out/'summary.json').relative_to(HERE)),str((out/'branch_rows.csv').relative_to(HERE))])


def probe(out, device):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    prepared_probe_data()
    settings={'linear_C':[.01,.1,1,10],'rbf_C':[.1,1,10],'rbf_gamma':'scale',
              'selection':'Train fitting and standardization, validation C selection; no test selection.',
              'rbf_role':'Nonlinear readout diagnostic, not proof of independent editable concepts.'}
    dump(out/'settings.json',settings)
    all_results={}
    for model in ['aug','equivariant']:
        for seed in CFG['ae_seeds']:
            path=ROOT/f'runs/repair_{model}_n1024_s{seed}/ae.pt'
            ae,_=r.load_model(path,device); encoded={}; labels={}; rotations={}
            with torch.no_grad():
                for split in ['train','val','test','ood']:
                    x,y=probe_data(split)
                    encoded[split]=np.concatenate([ae.encode(o.rotate(x[i:i+64].to(device),k)).cpu().numpy() for k in range(4) for i in range(0,len(x),64)])
                    labels[split]=np.tile(y.numpy(),(4,1));rotations[split]=np.repeat(np.arange(4),len(x))
            result={'ae_sha256':r.sha(path),'steps':6000,'spaces':{}}
            for space in (['full','m0','m2','m13'] if model=='equivariant' else ['full']):
                features={}
                for split,z in encoded.items():
                    f=np.fft.fft(z,axis=1,norm='ortho')
                    features[split]=z.reshape(len(z),-1) if space=='full' else (np.concatenate([f[:,1].real,f[:,1].imag],1) if space=='m13' else f[:,int(space[1])].real)
                scaler=StandardScaler().fit(features['train'])
                features={s:scaler.transform(x).astype(np.float64) for s,x in features.items()}
                sr={}
                for target in ['shape','color','rotation']:
                    ys={s:rotations[s] if target=='rotation' else labels[s][:,0 if target=='shape' else 1] for s in labels}
                    for family in (['linear','rbf'] if space=='full' and target!='rotation' else ['linear']):
                        best=None
                        for c in settings[f'{family}_C']:
                            clf=(LogisticRegression(C=c,max_iter=2000) if family=='linear' else SVC(C=c,kernel='rbf',gamma='scale'))
                            clf.fit(features['train'],ys['train']); val=float(np.mean(clf.predict(features['val'])==ys['val']))
                            if best is None or val>best[0]:best=(val,c,clf)
                        scores={'C':best[1],'validation_accuracy':best[0]}
                        for split in ['test','ood']:
                            correct=best[2].predict(features[split])==ys[split]
                            scores[split]=r.bootstrap(correct.reshape(4,-1).mean(0))
                        sr[f'{target}/{family}']=scores
                result['spaces'][space]=sr
            dump(out/f'{model}_s{seed}.json',result);all_results[f'{model}_s{seed}']=result
            print('probe',model,seed,result['spaces']['full']['shape/linear']['test']['mean'],result['spaces']['full']['shape/rbf']['test']['mean'],flush=True)
    dump(out/'summary.json',{'models':all_results,'settings':settings,**provenance()})
    update_status('A05','complete',[str((out/'summary.json').relative_to(HERE))], 'New 6,000-step checkpoint probes on fresh diagnostic splits; original 3,000-step report preserved.')


def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['audit','review','decode','probe']);p.add_argument('--out',required=True);p.add_argument('--device',default='mps')
    a=p.parse_args();torch.set_num_threads(4);out=fresh_dir(a.out)
    dump(out/'started.json',{'args':vars(a),'started_unix':time.time(),'code_sha256':r.sha(__file__)})
    device=o.choose_device(a.device)
    if a.task=='audit':audit(out)
    elif a.task=='review':review(out,device)
    elif a.task=='decode':diagnose_decode(out,device)
    else:probe(out,device)


if __name__=='__main__':main()
