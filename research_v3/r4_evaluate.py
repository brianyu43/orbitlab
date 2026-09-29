"""Frozen R4 scoring, full output replay and development-only selection."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from v3_common import ROOT, HERE, sha, dump, lock, event
from r4_data import BASE, DEVELOPMENT, THRESHOLD, READER
from r4_core import o, r, CANDIDATES, representation, oracle_codes, tensor_hash
from r4_train import freeze as training_freeze, root, data, load_model, save_torch
from evaluator_v2 import DetailedEvaluator
from p1_evaluate import extract_features, categorical, integrate
from p4_evaluate import accepted_diversity

EVAL=BASE/'evaluation';SPLITS=('val','ood');ROLES=('oracle','learned','bridge')
SCORES=('shape_correct','color_correct','joint_correct','strict_accepted_and_joint','template_iou')
ATTRS=('cx','cy','scale','orientation')

def freeze():
    training_freeze()
    return lock(BASE/'evaluation_protocol.json',{
        'version':1,'source_sha256':sha(__file__),'training_protocol_sha256':sha(BASE/'training_protocol.json'),
        'seed':DEVELOPMENT,'init':0,'splits':list(SPLITS),'test_used':False,
        'roles':list(ROLES),'reconstruction_units':24,'generation_units':4,
        'reconstruction':'All256base scenes per split, all4rotations. True renderer size thirds. Primary small arrow=kind2andtrue radius<.13+.11/3. Equal scene average across rotations; rotations are not independent scenes.',
        'success':'Original frozen strict shape screen AND correct shape AND correct color, not independent human evaluation. Identity/clean input ceiling and pixel metrics separately reported.',
        'generation':'Stored paired3072Gaussian samples(128per24condition), same noise for all candidates;64Eulersteps, guidance1. Base flow shared by standard/edge/coordinate. Generated size/cx/cy/orientation are template-reader estimates.',
        'diversity':'Historical bins and thresholds unchanged; use all384reference samples/condition versus128generated/condition. Each histogram normalized by its own count. Coverage>=.9,TV<=.2,outside<=.1 means per attribute; accepted subset requires>=32strict samples EACHcondition plus same historical thresholds.',
        'reference_validation':'Report all reference strict scores; if any clean reference fails strict joint, comparable-reference gate is false. Retain every reference, no outcome filtering or threshold adjustment.',
        'primary_gate':{'small_arrow_strict_reconstruction_min_each_val_and_ood':.8,'generation_strict_gain_balanced_mean_seen_ood':.1,'all_image_diversity':True,'strict_subset_diversity':True,'clean_reference_all_strict':True},
        'selection':'Candidate must meet every gate using learned role only; maximize balanced seen/OOD strict, tie by CANDIDATES order. If no candidate passes, select none and diagnose. No automatic3x3confirmation. Oracle/bridge outcomes never satisfy primary gate.',
        'oracle_caveat':'Oracle and learned decoders have separately trained weights/input spaces. Bridge is additional factor-supervised readout into same oracle decoder, a fixed-decoder diagnostic but not primary unsupervised performance.',
        'verification':'Replay every reconstruction and generated image from saved checkpoints at same64batch size, all64flowsteps, every metric row and aggregation. Source/weights/data/noise hash checks. Not independent retraining or a new implementation.',
        'threshold_sha256':sha(THRESHOLD),'reader_sha256':sha(READER),
        'dependencies':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'followup/evaluator_v2.py',ROOT/'work/generation.py',ROOT/'generation_recovery/p1_evaluate.py',ROOT/'generation_recovery/p4_evaluate.py',ROOT/'generation_recovery/p0_extended.py']}})

def context():
    return {'evaluation_protocol_sha256':sha(BASE/'evaluation_protocol.json'),
        'data_manifest_sha256':sha(BASE/f'data/d{DEVELOPMENT}/manifest.json')}

def metrics(rows, keys=SCORES):
    if not rows:return {'n':0,**{k:None for k in keys}}
    return {'n':len(rows),**{k:float(np.mean([row[k] for row in rows])) for k in keys}}

def full_diversity(rows, real):
    """Same historical formula, separate128vs384sample histogram denominators."""
    reader=json.loads(READER.read_text());conditions=[]
    for kind in range(4):
        for color in range(6):
            g=[v for v in rows if (v['kind'],v['color'])==(kind,color)]
            ref=[v for v in real if (v['kind'],v['color'])==(kind,color)]
            assert len(g)==128 and len(ref)==384
            for key in ATTRS:
                if key=='orientation':
                    categories=reader['orientation'][str(kind)]['modulo'];gv=np.asarray([v[key] for v in g]);rv=np.asarray([v[key] for v in ref]);outside=0.
                else:
                    edges=reader['center_edges'][key] if key!='scale' else reader['scale_edges_by_kind'][str(kind)]
                    gv,outside=categorical([v[key] for v in g],edges);rv,_=categorical([v[key] for v in ref],edges);categories=3
                gh=np.bincount(gv,minlength=categories)[:categories];rh=np.bincount(rv,minlength=categories)[:categories]
                possible=int(np.count_nonzero(rh));covered=int(np.count_nonzero((gh>0)&(rh>0)))
                conditions.append({'kind':kind,'color':color,'attribute':key,'generated_n':len(g),'reference_n':len(ref),
                    'reference_occupied_bins':possible,'coverage':covered/possible if possible else 1.,
                    'tv':.5*float(np.abs(gh/gh.sum()-rh/rh.sum()).sum()),'outside':outside})
    agg={key:{name:float(np.mean([v[name] for v in conditions if v['attribute']==key])) for name in ['coverage','tv','outside']} for key in ATTRS}
    return {'passed':all(v['coverage']>=.9 and v['tv']<=.2 and v['outside']<=.1 for v in agg.values()),'aggregate':agg,'conditions':conditions}

def extract_batched(images, labels, evaluator):
    rows=[]
    for i in range(0,len(images),64):rows.extend(extract_features(images[i:i+64],labels[i:i+64],evaluator))
    return rows

def write_unit(folder, images, rows, summary, provenance, verify):
    if verify:
        saved=json.loads((folder/'summary.json').read_text());assert saved['context']==context() and saved['provenance']==provenance
        assert saved['metrics']==summary and json.loads((folder/'rows.json').read_text())==rows
        assert sha(folder/'rows.json')==saved['rows_sha256']
        if images is not None:
            assert sha(folder/'images.pt')==saved['images_sha256'];old=torch.load(folder/'images.pt',map_location='cpu',weights_only=True)
            torch.testing.assert_close(images,old,rtol=0,atol=1e-6)
        dump(folder/'verification.json',{'all_images_replayed':len(rows),'all_rows_recomputed':len(rows),'summary_sha256':sha(folder/'summary.json'),'context':context(),'max_image_tolerance':1e-6,'independent_retraining':False})
    else:
        if (folder/'summary.json').exists():
            saved=json.loads((folder/'summary.json').read_text());assert saved['context']==context() and saved['provenance']==provenance and saved['metrics']==summary
            assert json.loads((folder/'rows.json').read_text())==rows
            if images is not None:torch.testing.assert_close(images,torch.load(folder/'images.pt',weights_only=True),rtol=0,atol=1e-6)
            return
        dump(folder/'rows.json',rows)
        if images is not None:save_torch(folder/'images.pt',images)
        dump(folder/'summary.json',{'context':context(),'provenance':provenance,'metrics':summary,'rows_sha256':sha(folder/'rows.json'),'images_sha256':sha(folder/'images.pt') if images is not None else None})

@torch.no_grad()
def reference(verify=False):
    freeze();torch.set_num_threads(2);ev=DetailedEvaluator()
    x,y,_=data(DEVELOPMENT,'reference');rows=extract_batched(x,y,ev)
    summary={**metrics(rows),'all_strict':all(v['strict_accepted_and_joint'] for v in rows),'by_condition':{f'{k},{c}':metrics([v for v in rows if (v['kind'],v['color'])==(k,c)]) for k in range(4) for c in range(6)}}
    write_unit(EVAL/'reference',None,rows,summary,{'data_sha256':sha(BASE/f'data/d{DEVELOPMENT}/reference.npz')},verify)
    print('R4 reference','verified' if verify else 'evaluated',summary['strict_accepted_and_joint'],flush=True)

def reconstruction_summary(rows):
    keys=(*SCORES,'mse','foreground_mae','mask_iou','centroid_error_px')
    grouped={}
    for row in rows:grouped.setdefault(row['base_id'],[]).append(row)
    assert all(len(group)==4 for group in grouped.values())
    small=[v for v in rows if v['kind']==2 and v['true_size_bin']==0]
    scenes={key:float(np.mean([v['strict_accepted_and_joint'] for v in group])) for key,group in grouped.items()}
    return {'overall':metrics(rows,keys),'base_scenes':len(grouped),'scene_mean_strict':float(np.mean(list(scenes.values()))),
        'all_four_rotations_strict':float(np.mean([all(v['strict_accepted_and_joint'] for v in group) for group in grouped.values()])),
        'small_arrow':{**metrics(small,keys),'base_scenes':len({v['base_id'] for v in small})},
        'by_shape_size':{f'{k},{size}':metrics([v for v in rows if v['kind']==k and v['true_size_bin']==size],keys) for k in range(4) for size in range(3)}}

@torch.no_grad()
def reconstruct(candidate, role, split, verify=False):
    freeze();torch.set_num_threads(2);assert split in SPLITS;assert candidate in (*CANDIDATES,'clean') and role in (*ROLES,'clean')
    x,y,meta=data(DEVELOPMENT,split);ev=DetailedEvaluator();run=root(DEVELOPMENT,0)
    provenance={'data_sha256':sha(BASE/f'data/d{DEVELOPMENT}/{split}.npz'),'metadata_sha256':sha(BASE/f'data/d{DEVELOPMENT}/{split}_metadata.json')}
    rep=representation(candidate);model=encoder=bridge=norm=None
    if candidate!='clean':
        component=f'decoder_{"oracle" if role=="bridge" else role}_{candidate}'
        model,_=load_model(DEVELOPMENT,0,component);provenance['decoder_sha256']=sha(run/component/'checkpoint.pt')
        norm=torch.load(run/'pools/normalization.pt',map_location='cpu',weights_only=True);provenance['normalization_sha256']=sha(run/'pools/normalization.pt')
        if role!='oracle':encoder,_=load_model(DEVELOPMENT,0,f'ae_{rep}');provenance['ae_sha256']=sha(run/f'ae_{rep}/checkpoint.pt')
        if role=='bridge':bridge,_=load_model(DEVELOPMENT,0,f'bridge_{rep}');provenance['bridge_sha256']=sha(run/f'bridge_{rep}/checkpoint.pt')
    images=[];rows=[];max_eq=0.;extra=[]
    for rotation in range(4):
        for i in range(0,len(x),64):
            target=o.rotate(x[i:i+64],rotation);labels=y[i:i+64];info=meta[i:i+64]
            if candidate=='clean':pred=target
            else:
                raw=oracle_codes(labels,info,rotation) if role=='oracle' else encoder.encode(target)
                source='oracle' if role=='oracle' else rep;z=(raw-norm[source]['mean'])/norm[source]['std']
                if role=='bridge':z=bridge(z)
                pred=model(z)
                if i==0:
                    max_eq=max(max_eq,float((model(o.rho(z,1))-o.rotate(pred,1)).abs().max()))
                if candidate=='coordinate':
                    true=oracle_codes(labels,info,rotation)[...,10:12]
                    xyerr=((model.centers(z)-true)*31.5).square().sum(-1).sqrt().mean(1)
                if role=='bridge':
                    truth=oracle_codes(labels,info,rotation);normaltruth=(truth-norm['oracle']['mean'])/norm['oracle']['std']
                    factor_error=(z-normaltruth).square().mean((1,2))
            rr=extract_features(pred,labels,ev);pm=r.pixel_metrics(pred,target)
            for j,row in enumerate(rr):
                radius=info[j]['radius'];row.update(base_id=info[j]['base_id'],rotation=rotation,true_radius=radius,true_size_bin=int(np.searchsorted([.13+.11/3,.13+2*.11/3],radius)),
                    mse=float(pm['mse'][j]),foreground_mae=float(pm['foreground_mae'][j]),mask_iou=float(pm['mask_iou'][j]),centroid_error_px=float(pm['centroid_error'][j]*64))
                if candidate=='coordinate':row['coordinate_head_error_px']=float(xyerr[j])
                if role=='bridge':row['normalized_factor_mse']=float(factor_error[j])
            rows.extend(rr);images.append(pred)
    summary=reconstruction_summary(rows);summary['decoder_c4_max']=max_eq
    if candidate=='coordinate':summary['coordinate_head_error_px']=float(np.mean([v['coordinate_head_error_px'] for v in rows]))
    if role=='bridge':summary['normalized_factor_mse']=float(np.mean([v['normalized_factor_mse'] for v in rows]))
    dest=EVAL/'reconstruction'/f'{role}_{candidate}'/split
    write_unit(dest,None if candidate=='clean' else torch.cat(images),rows,summary,provenance,verify)
    print('R4 reconstruction',role,candidate,split,'verified' if verify else 'evaluated',summary['small_arrow']['strict_accepted_and_joint'],flush=True)

@torch.no_grad()
def flow_latent(rep, verify=False):
    freeze();torch.set_num_threads(2);assert rep in ['base','spatial']
    model,_=load_model(DEVELOPMENT,0,f'flow_{rep}');a=np.load(BASE/f'data/d{DEVELOPMENT}/generation_noise.npz');noise=torch.from_numpy(a['noise'].copy());labels=torch.from_numpy(a['labels'].copy())
    provenance={**context(),'noise_sha256':sha(BASE/f'data/d{DEVELOPMENT}/generation_noise.npz'),'flow_sha256':sha(root(DEVELOPMENT,0)/f'flow_{rep}/checkpoint.pt')}
    dest=EVAL/'flow'/rep;cp=dest/'latent.pt'
    if cp.exists() and not verify:
        ck=torch.load(cp,map_location='cpu',weights_only=True);assert ck['provenance']==provenance
        torch.testing.assert_close(ck['noise'],noise,rtol=0,atol=0);return ck['latent'],labels
    start=time.perf_counter();latent=torch.cat([integrate(model,noise[i:i+128],labels[i:i+128],1.) for i in range(0,len(noise),128)])
    assert torch.isfinite(latent).all()
    if verify:
        ck=torch.load(cp,map_location='cpu',weights_only=True);assert ck['provenance']==provenance
        torch.testing.assert_close(noise,ck['noise'],rtol=0,atol=0);torch.testing.assert_close(labels,ck['labels'],rtol=0,atol=0);torch.testing.assert_close(latent,ck['latent'],rtol=0,atol=1e-6)
        dump(dest/'verification.json',{'all_latents_replayed':len(latent),'steps_each':64,'latent_sha256':sha(cp),'provenance':provenance})
    else:save_torch(cp,{'noise':noise,'labels':labels,'latent':latent,'provenance':provenance,'seconds':time.perf_counter()-start})
    print('R4 flow',rep,'verified' if verify else 'sampled',flush=True)
    return latent,labels

@torch.no_grad()
def generate(candidate, verify=False):
    freeze();torch.set_num_threads(2);rep=representation(candidate);latent,labels=flow_latent(rep)
    model,_=load_model(DEVELOPMENT,0,f'decoder_learned_{candidate}');images=torch.cat([model(latent[i:i+64]) for i in range(0,len(latent),64)])
    ev=DetailedEvaluator();rows=extract_batched(images,labels,ev);real=json.loads((EVAL/'reference/rows.json').read_text())
    summary={'splits':{s:metrics([v for v in rows if v['ood']==(s=='ood')]) for s in ['seen','ood']},
        'by_condition':{f'{k},{c}':metrics([v for v in rows if (v['kind'],v['color'])==(k,c)]) for k in range(4) for c in range(6)},
        'full_diversity':full_diversity(rows,real),'strict_diversity':accepted_diversity(rows,real)}
    summary['balanced_strict']=float(np.mean([v['strict_accepted_and_joint'] for v in summary['splits'].values()]))
    provenance={'latent_sha256':sha(EVAL/'flow'/rep/'latent.pt'),'decoder_sha256':sha(root(DEVELOPMENT,0)/f'decoder_learned_{candidate}/checkpoint.pt'),'reference_rows_sha256':sha(EVAL/'reference/rows.json')}
    write_unit(EVAL/'generation'/candidate,images,rows,summary,provenance,verify)
    print('R4 generation',candidate,'verified' if verify else 'evaluated',summary['balanced_strict'],flush=True)

def aggregate():
    freeze();ref=json.loads((EVAL/'reference/summary.json').read_text())['metrics'];recon={};gen={};units=[]
    for role in ROLES:
        for candidate in CANDIDATES:
            recon[f'{role}_{candidate}']={}
            for split in SPLITS:
                folder=EVAL/'reconstruction'/f'{role}_{candidate}'/split
                record=json.loads((folder/'summary.json').read_text());v=json.loads((folder/'verification.json').read_text());assert v['summary_sha256']==sha(folder/'summary.json')
                recon[f'{role}_{candidate}'][split]=record['metrics'];units.append(str(folder.relative_to(BASE)))
    for candidate in CANDIDATES:
        folder=EVAL/'generation'/candidate;record=json.loads((folder/'summary.json').read_text());v=json.loads((folder/'verification.json').read_text());assert v['summary_sha256']==sha(folder/'summary.json')
        gen[candidate]=record['metrics'];units.append(str(folder.relative_to(BASE)))
    baseline=gen['standard']['balanced_strict'];candidates=[]
    for candidate in CANDIDATES[1:]:
        a=recon['learned_'+candidate];g=gen[candidate];small=min(a[s]['small_arrow']['strict_accepted_and_joint'] for s in SPLITS);gain=g['balanced_strict']-baseline
        gates={'small_arrow_reconstruction':small>=.8,'generation_gain':gain>=.1,'full_diversity':g['full_diversity']['passed'],'strict_diversity':g['strict_diversity']['passed'],'clean_reference':ref['all_strict']}
        candidates.append({'candidate':candidate,'small_arrow_strict_min':small,'balanced_strict_gain':gain,'gates':gates,'passed':all(gates.values())})
    passing=[v for v in candidates if v['passed']];selected=max(passing,key=lambda v:gen[v['candidate']]['balanced_strict'])['candidate'] if passing else None
    result={'context':context(),'reference':ref,'reconstruction':recon,'generation':gen,'candidate_gates':candidates,'selected':selected,'development_gate_passed':selected is not None,'confirmation_started':False,'units':units}
    lock(BASE/'aggregate.json',result)
    lock(BASE/'selection.json',{'aggregate_sha256':sha(BASE/'aggregate.json'),'selected':selected,'gate_passed':selected is not None,'candidate_gates':candidates,'confirmation_started':False})
    allfolders=[EVAL/'reference',*[EVAL/'reconstruction'/'clean_clean'/s for s in SPLITS],*[BASE/u for u in units]]
    for folder in allfolders:assert json.loads((folder/'verification.json').read_text())['summary_sha256']==sha(folder/'summary.json')
    for rep in ['base','spatial']:assert json.loads((EVAL/'flow'/rep/'verification.json').read_text())['latent_sha256']==sha(EVAL/'flow'/rep/'latent.pt')
    dump(BASE/'evaluation_verification.json',{'units':{str(p.relative_to(BASE)):sha(p/'verification.json') for p in allfolders},'all_reconstructed_model_images':24*1024,'all_generated_images':4*3072,'clean_reference_images':9216,'clean_reconstruction_images':2048,'all_flow_latents':6144,'aggregate_sha256':sha(BASE/'aggregate.json'),'selection_sha256':sha(BASE/'selection.json'),'source_sha256':sha(__file__)})
    print(json.dumps(result['candidate_gates']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['reference','reconstruction','flow','generation','aggregate']);p.add_argument('--candidate',choices=[*CANDIDATES,'clean']);p.add_argument('--role',choices=[*ROLES,'clean']);p.add_argument('--split',choices=SPLITS);p.add_argument('--representation',choices=['base','spatial']);p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.stage=='reference':reference(a.verify)
    elif a.stage=='reconstruction':reconstruct(a.candidate,a.role,a.split,a.verify)
    elif a.stage=='flow':flow_latent(a.representation,a.verify)
    elif a.stage=='generation':generate(a.candidate,a.verify)
    else:aggregate()
