"""Frozen scoring and selection for completion-v2 generation experiments."""
from pathlib import Path
import sys,json,argparse,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'completion_v2'),str(ROOT/'generation_recovery'),str(ROOT/'work'),str(ROOT/'followup')]
import numpy as np
import torch
import orbitlab as o
import generation_run as g
from p0 import dump,sha
from models import from_variant
from decoder_study import ControlledDecoder
from evaluator_v2 import DetailedEvaluator
from p1_evaluate import labels_all,extract_features,integrate,categorical,READER

VARIANTS=['uniform_flow/standard_decoder','stratified_flow/standard_decoder',
          'uniform_flow/area_edge_decoder','stratified_flow/area_edge_decoder']

def diversity(rows,reference,strict=False):
    reader=json.loads(READER.read_text());details=[]
    for k in range(4):
        for c in range(6):
            group=[v for v in rows if (v['kind'],v['color'])==(k,c) and (not strict or v['strict_accepted_and_joint'])]
            real=[v for v in reference if (v['kind'],v['color'])==(k,c)]
            if len(group)<32:
                details.append({'kind':k,'color':c,'n':len(group),'insufficient':True});continue
            features={}
            for key in ['cx','cy','scale','orientation']:
                if key=='orientation':
                    count=reader['orientation'][str(k)]['modulo'];gv=[v[key] for v in group];rv=[v[key] for v in real];outside=0.
                else:
                    edges=reader['scale_edges_by_kind'][str(k)] if key=='scale' else reader['center_edges'][key]
                    gv,outside=categorical([v[key] for v in group],edges);rv,_=categorical([v[key] for v in real],edges);count=3
                gh=np.bincount(gv,minlength=count)[:count];rh=np.bincount(rv,minlength=count)[:count]
                features[key]={'coverage':float(((gh>0)&(rh>0)).sum()/(rh>0).sum()),
                               'tv':float(abs(gh/gh.sum()-rh/rh.sum()).sum()/2),'outside':outside}
            details.append({'kind':k,'color':c,'n':len(group),'features':features,'insufficient':False})
    usable=[v for v in details if not v['insufficient']]
    aggregate={key:{m:float(np.mean([v['features'][key][m] for v in usable])) for m in ['coverage','tv','outside']}
               for key in ['cx','cy','scale','orientation']} if usable else {}
    passed=len(usable)==24 and all(v['coverage']>=.9 and v['tv']<=.2 and v['outside']<=.1 for v in aggregate.values())
    return {'passed':passed,'aggregate':aggregate,'conditions':details}

def metrics(rows):
    return {split:{key:float(np.mean([v[key] for v in rows if v['ood']==(split=='ood')]))
                   for key in ['strict_accepted_and_joint','shape_correct','color_correct']}
            for split in ['seen','ood']}

def evaluate(seed,init):
    torch.set_num_threads(4);g.freeze()
    eval_lock=g.BASE/'evaluation_lock.json'
    locked={'source_sha256':sha(Path(__file__)),'training_lock_sha256':sha(g.LOCK),
            'reader_sha256':sha(READER),'samples_per_pair':128,'reference_per_pair':384,
            'noise_seed_rule':'seed+20000 shared across initializations and models',
            'selection_rule':'Among nonbaseline variants passing both diversity guards, maximize mean strict seen/OOD. If none pass, select best mean diagnostic candidate and mark failed gate.'}
    if eval_lock.exists():assert json.loads(eval_lock.read_text())==locked
    else:dump(eval_lock,locked)
    ev=DetailedEvaluator();run=g.RUNS/f'd{seed}_s{init}';out=g.BASE/f'evaluation/d{seed}_s{init}'
    out.mkdir(parents=True,exist_ok=True)
    ref_path=g.BASE/f'evaluation/reference_{seed}.json'
    if ref_path.exists():reference=json.loads(ref_path.read_text())['rows']
    else:
        a=np.load(g.DATA/f'd{seed}/reference.npz')
        x=torch.from_numpy(a['images'].copy()).permute(0,3,1,2).float()/255;y=torch.from_numpy(a['labels'].copy())
        reference=[]
        for i in range(0,len(x),128):reference.extend(extract_features(x[i:i+128],y[i:i+128],ev))
        assert all(v['strict_accepted_and_joint'] for v in reference)
        dump(ref_path,{'rows':reference,'data_sha256':sha(g.DATA/f'd{seed}/reference.npz'),'source_sha256':sha(Path(__file__))})
    variants=VARIANTS
    if seed!=880101:
        selected=json.loads((g.BASE/'selection.json').read_text())
        variants=[VARIANTS[0],selected['candidate']]
    labels=labels_all(128);noise=torch.randn(len(labels),4,16,generator=torch.Generator().manual_seed(seed+20000))
    results={}
    for variant in variants:
        fk,dk=variant.split('/');dest=out/variant.replace('/','__');dest.mkdir(exist_ok=True)
        fp=run/fk/'checkpoint.pt';dp=run/('decoder_logit_mean/decoder.pt' if dk=='standard_decoder' else 'area_edge_decoder/checkpoint.pt')
        if (dest/'result.json').exists():
            result=json.loads((dest/'result.json').read_text());assert result['flow_sha256']==sha(fp) and result['decoder_sha256']==sha(dp)
            results[variant]=result;continue
        fc=torch.load(fp,map_location='cpu',weights_only=True);dc=torch.load(dp,map_location='cpu',weights_only=True)
        assert fc['ae_sha256']==dc['ae_sha256']
        flow=from_variant('F0');flow.load_state_dict(fc['state_dict']);flow.eval()
        decoder=ControlledDecoder('logit_mean');decoder.load_state_dict(dc['state_dict']);decoder.eval()
        start=time.perf_counter()
        with torch.no_grad():
            latent=torch.cat([integrate(flow,noise[i:i+128],labels[i:i+128],1.) for i in range(0,len(noise),128)])
            raw=latent*fc['std']+fc['mean']
            images=torch.cat([decoder((raw[i:i+64]-dc['mean'])/dc['std']) for i in range(0,len(raw),64)])
        rows=extract_features(images,labels,ev)
        torch.save({'noise':noise,'labels':labels,'latent':latent,'images':images},dest/'samples.pt')
        dump(dest/'rows.json',rows);o.grid(images[:48],dest/'samples.png',8)
        result={'variant':variant,'seed':seed,'init':init,'metrics':metrics(rows),
                'diversity_all':diversity(rows,reference),'diversity_strict':diversity(rows,reference,True),
                'flow_sha256':sha(fp),'decoder_sha256':sha(dp),'rows_sha256':sha(dest/'rows.json'),
                'samples_sha256':sha(dest/'samples.pt'),'source_sha256':sha(Path(__file__)),
                'seconds':time.perf_counter()-start}
        dump(dest/'result.json',result);results[variant]=result
        print('generation score',seed,init,variant,result['metrics'],result['diversity_strict']['passed'],flush=True)
    if seed==880101 and not (g.BASE/'selection.json').exists():
        candidates=[v for v in VARIANTS[1:] if results[v]['diversity_all']['passed'] and results[v]['diversity_strict']['passed']]
        passed=bool(candidates)
        if not candidates:candidates=VARIANTS[1:]
        winner=max(candidates,key=lambda v:np.mean([results[v]['metrics'][s]['strict_accepted_and_joint'] for s in ['seen','ood']]))
        dump(g.BASE/'selection.json',{'candidate':winner,'development_diversity_gate_passed':passed,
                                    'confirmation_claim':'Success confirmation' if passed else 'Diagnostic replication; development diversity milestone failed',
                                    'development_seed':seed,'development_init':init,'evaluation_lock_sha256':sha(eval_lock),
                                    'result_hashes':{v:sha(out/v.replace('/','__')/'result.json') for v in VARIANTS}})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=880101);p.add_argument('--init',type=int,default=0)
    a=p.parse_args();evaluate(a.seed,a.init)
