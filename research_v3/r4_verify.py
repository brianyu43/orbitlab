"""R4 training provenance, full cached-code replay and independent metric checks."""
import argparse,json
import numpy as np
import torch
from v3_common import HERE,sha,dump
from r4_data import BASE,DEVELOPMENT
from r4_train import COMPONENTS,root,data,freeze,settings,load_model
from r4_core import o,oracle_codes,state_hash

@torch.no_grad()
def verify_pools():
    freeze();torch.set_num_threads(2);run=root(DEVELOPMENT,0);p=run/'pools';manifest=json.loads((p/'manifest.json').read_text())
    for n,h in manifest['files'].items():assert sha(p/n)==h
    norm=torch.load(p/'normalization.pt',map_location='cpu',weights_only=True);models={rep:load_model(DEVELOPMENT,0,f'ae_{rep}')[0] for rep in ['base','spatial']};checks=[]
    for split in ['train','val','ood']:
        x,y,meta=data(DEVELOPMENT,split);saved=torch.load(p/f'{split}.pt',map_location='cpu',weights_only=True)
        torch.testing.assert_close(y.repeat(4,1),saved['labels'],rtol=0,atol=0)
        expected={rep:[] for rep in ['base','spatial','oracle']}
        for k in range(4):
            expected['oracle'].append(oracle_codes(y,meta,k))
            for rep,model in models.items():expected[rep].append(torch.cat([model.encode(o.rotate(x[i:i+64],k)) for i in range(0,len(x),64)]))
        for rep,values in expected.items():
            raw=torch.cat(values);torch.testing.assert_close(raw,saved['raw'][rep],rtol=0,atol=1e-6)
            if split=='train':
                torch.testing.assert_close(norm[rep]['mean'],raw.mean((0,1),keepdim=True),rtol=0,atol=0)
                torch.testing.assert_close(norm[rep]['std'],raw.std((0,1),unbiased=False,keepdim=True).clamp_min(.02),rtol=0,atol=0)
            torch.testing.assert_close((raw-norm[rep]['mean'])/norm[rep]['std'],saved['z'][rep],rtol=0,atol=1e-6)
            for k in range(4):torch.testing.assert_close(values[k],o.rho(values[0],k),rtol=0,atol=1e-5)
        torch.testing.assert_close(saved['center'],saved['raw']['oracle'][...,10:12],rtol=0,atol=0)
        checks.append({'split':split,'images_per_encoder':len(x)*4,'all_latents_and_labels_replayed':True})
    dump(p/'verification.json',{'manifest_sha256':sha(p/'manifest.json'),'checks':checks,'train_only_normalization_recomputed':True,'source_sha256':sha(__file__)})
    print('R4 pools fully replayed',flush=True)

def verify_training():
    freeze();run=root(DEVELOPMENT,0);rows={}
    for component in COMPONENTS:
        folder=run/component;info=json.loads((folder/'run.json').read_text());model,ck=load_model(DEVELOPMENT,0,component)
        progress=torch.load(folder/'progress.pt',map_location='cpu',weights_only=True)
        assert ck['steps']==info['steps']==progress['step']==settings(component)[0]
        assert state_hash(model)==info['final_state_sha256'] and info['parameters']==sum(p.numel() for p in model.parameters())
        for key,value in model.state_dict().items():torch.testing.assert_close(value,progress['state_dict'][key],rtol=0,atol=0)
        assert info['logs']==progress['logs'];assert [v['step'] for v in info['logs']]==list(range(500,info['steps']+1,500))
        pilot=json.loads((run/'benchmarks'/component/'run.json').read_text());assert pilot['steps']==100 and pilot['context']==info['context'] and pilot['initial_state_sha256']==info['initial_state_sha256']
        assert info['seconds']>=info['logs'][-1]['seconds']>0
        rows[component]={'run_sha256':sha(folder/'run.json'),'steps':info['steps'],'parameters':info['parameters'],'seconds':info['seconds']}
    for role in ['oracle','learned']:
        folders=[run/f'decoder_{role}_{c}' for c in ['standard','edge']]
        reports=[json.loads((f/'run.json').read_text()) for f in folders]
        assert reports[0]['initial_state_sha256']==reports[1]['initial_state_sha256']
        rngs=[torch.load(f/'progress.pt',map_location='cpu',weights_only=True)['rng'] for f in folders];torch.testing.assert_close(*rngs,rtol=0,atol=0)
    assert sum(v['steps'] for v in rows.values())==100000
    dump(BASE/'training_verification.json',{'components':rows,'total_updates':100000,'pilot_updates_separate':1400,'standard_edge_initial_weights_and_final_sampling_rng_equal':True,'source_sha256':sha(__file__),'scope':'Checkpoint/progress/log consistency, not independent full retraining.'})
    print('R4 training receipts verified',flush=True)

def verify_metric_formulas():
    # Synthetic checks happen before real model scoring and do not tune gates.
    from r4_evaluate import full_diversity
    import p1_evaluate as old
    from p4_evaluate import accepted_diversity
    reader=json.loads(old.READER.read_text());generated=[]
    for i in range(128):
        for kind in range(4):
            for color in range(6):
                row={'kind':kind,'color':color,'strict_accepted_and_joint':True,'orientation':i%reader['orientation'][str(kind)]['modulo']}
                for key in ['cx','cy','scale']:
                    edges=reader['center_edges'][key] if key!='scale' else reader['scale_edges_by_kind'][str(kind)]
                    j=i%3;row[key]=(edges[j]+edges[j+1])/2
                generated.append(row)
    real=generated*3;current=full_diversity(generated,real);previous_n=old.PER_PAIR
    try:old.PER_PAIR=128;previous=old.diversity(generated,generated)
    finally:old.PER_PAIR=previous_n
    assert current['passed'] and previous['passed'] and accepted_diversity(generated,real)['passed']
    for key in ['cx','cy','scale','orientation']:
        a=current['aggregate'][key];b=previous['aggregate'][key]
        assert a['coverage']==b['coverage_mean']==1 and abs(a['tv'])<1e-12 and a['outside']==b['outside_mean']==0
    rejected=[{**v,'strict_accepted_and_joint':False} for v in generated]
    assert not accepted_diversity(rejected,real)['passed']
    # All samples collapsed into the first bin must fail coverage/TV gates.
    collapsed=[{**v,'cx':reader['center_edges']['cx'][0],'cy':reader['center_edges']['cy'][0]} for v in generated]
    assert not full_diversity(collapsed,real)['passed']
    dump(BASE/'metric_preflight.json',{'historical_formula_identity_and_unequal_count_normalization':True,'strict_empty_rejected':True,'collapsed_distribution_rejected':True,'source_sha256':sha(__file__),'evaluator_sha256':sha(HERE/'r4_evaluate.py')})
    print('R4 metric formula checks passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pools','training','metrics']);a=p.parse_args()
    {'pools':verify_pools,'training':verify_training,'metrics':verify_metric_formulas}[a.stage]()
