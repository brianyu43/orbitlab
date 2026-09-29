"""Descriptive seed-level aggregates; no pseudo-replicated confidence intervals."""
from pathlib import Path
from collections import defaultdict
import json,sys
import numpy as np
B=Path(__file__).resolve().parent

def load(p):return json.loads(p.read_text())
def stat(v):
    v=[float(x) for x in v if x is not None]
    return {'mean':float(np.mean(v)),'min':min(v),'max':max(v),'n':len(v)} if v else {'mean':None,'min':None,'max':None,'n':0}
def paired(records,groupkeys,seedkey,extract):
    groups=defaultdict(list)
    for r in records:groups['/'.join(str(r[k]) for k in groupkeys)].append(r)
    out={}
    for key,rr in sorted(groups.items()):
        seedmeans={str(seed):{m:stat([extract(r).get(m) for r in rr if r[seedkey]==seed])['mean'] for m in extract(rr[0])}
                   for seed in sorted({r[seedkey] for r in rr})}
        out[key]={'units':len(rr),'per_data_seed':seedmeans,'metrics':{m:stat([v[m] for v in seedmeans.values()]) for m in extract(rr[0])}}
    return out

def main():
    report={'scope':'Descriptive summaries. Ranges are data-seed means; no formal population confidence claim.','counts':{}}
    for branch in ['perception','perception_detail']:
        paths=list((B/branch/'evaluation').glob('d*/*/*/summary.json'));rr=[load(p) for p in paths]
        report[branch]=paired(rr,['kind','split'],'data_seed',lambda r:r['metrics']['learned'])
        report['counts'][branch]={'units':len(rr),'expected':90,'verified':sum((p.parent/'verification.json').exists() for p in paths)}
    paths=[p for p in (B/'dynamics/evaluation').glob('d*/*/*/*/summary.json') if '/d640031_' not in str(p)];rr=[]
    for p in paths:
        r=load(p);r['geometry']=load(p.parent/'geometry.json') if (p.parent/'geometry.json').exists() else None;rr.append(r)
    def dm(r):
        metrics={f'h{h}_{k}':v['mean'] for h,values in r['summary'].items() for k,v in values.items()}
        if r['geometry']:
            metrics.update({f'h{h}_geometry_{k}':v for h,values in r['geometry']['summary'].items() for k,v in values.items() if k.endswith('rate_finite')})
        return metrics
    report['dynamics']=paired(rr,['mode','input','variant','split'],'data_seed',dm)
    report['counts']['dynamics']={'units':len(rr),'expected':1404,'verified':sum((p.parent/'verification.json').exists() for p in paths),'geometry':sum((p.parent/'geometry.json').exists() for p in paths)}
    paths=list((B/'generation/evaluation').glob('d*/*/result.json'));rr=[load(p) for p in paths]
    report['counts']['generation']={'units':len(rr),'expected':22,'verified':sum((p.parent/'verification.json').exists() for p in paths)}
    report['generation']=paired([r for r in rr if r['seed']!=880101],['variant'],'seed',lambda r:{f'{s}_{k}':v for s,a in r['metrics'].items() for k,v in a.items()})
    baseline='uniform_flow/standard_decoder'
    candidates=[k for k in report['generation'] if k!=baseline]
    if candidates and baseline in report['generation']:
        candidate=candidates[0];a=report['generation'][candidate]['per_data_seed'];b=report['generation'][baseline]['per_data_seed']
        changes={seed:{m:a[seed][m]-b[seed][m] for m in a[seed]} for seed in sorted(a.keys()&b.keys())}
        report['generation_paired_changes']={'candidate':candidate,'baseline':baseline,'direction':'candidate minus baseline; positive is improvement',
            'per_data_seed':changes,'metrics':{m:stat([v[m] for v in changes.values()]) for m in next(iter(changes.values()))}}
    report['generation_development']={r['variant']:{'metrics':r['metrics'],'all_diversity':r['diversity_all']['passed'],'strict_diversity':r['diversity_strict']['passed']} for r in rr if r['seed']==880101}
    report['generation_individual_diversity']=[{'seed':a['seed'],'initialization':a['init'],'variant':a['variant'],
        'all_passed':a['diversity_all']['passed'],'strict_passed':a['diversity_strict']['passed']} for a in rr if a['seed']!=880101]
    report['generation_pooled_diversity']={}
    from generation_evaluate import diversity
    for seed in [880201,880202,880203]:
        for variant in sorted({r['variant'] for r in rr if r['seed']==seed}):
            pp=[p for p in paths if (r:=load(p))['seed']==seed and r['variant']==variant]
            if len(pp)!=3:continue
            rows=sum([load(p.parent/'rows.json') for p in pp],[]);reference=load(B/f'generation/evaluation/reference_{seed}.json')['rows']
            report['generation_pooled_diversity'][f'{seed}/{variant}']={'all':diversity(rows,reference),'strict':diversity(rows,reference,True)}
    paths=list((B/'svib/runs').glob('*/evaluation.json'));rr=[load(p) for p in paths]
    report['svib']=paired(rr,['kind'],'seed',lambda r:r['metrics'])
    report['svib']['independence']='One official train/test split, three model initializations. seed field denotes initialization here, not independently sampled datasets.'
    report['counts']['svib']={'units':len(rr),'expected':6,'verified':sum((p.parent/'verification.json').exists() for p in paths)}
    report['all_experiments_and_primary_verification_complete']=all(v['units']==v['expected']==v['verified'] for v in report['counts'].values())
    path=B/'aggregate.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(path)
    print(json.dumps(report['counts'],indent=2))
if __name__=='__main__':main()
