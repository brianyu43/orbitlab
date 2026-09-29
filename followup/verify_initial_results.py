"""Verify the initial follow-up evidence; not a whole-project completion audit."""
import csv
import json
import time
from common import HERE, ROOT, dump, digest_tensor, np, torch, r


def read_csv(p):
    with p.open() as f:return list(csv.DictReader(f))


def main():
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    for record in manifest['files']:
        check('original '+record['path'],r.sha(ROOT/record['path'])==record['sha256'])
    pm=json.loads((HERE/'data/probe_v1/manifest.json').read_text());hashes=[]
    original={v['orbit_sha256'] for s in ['train','val','test','ood'] for v in json.loads((ROOT/f'data/{s}_metadata.json').read_text())}
    for split in ['val','test','ood']:
        p=HERE/f'data/probe_v1/{split}.npz';check('probe archive '+split,r.sha(p)==pm['archive_hashes'][split])
        meta=json.loads((p.parent/f'{split}_metadata.json').read_text());a=np.load(p)
        for i,v in enumerate(meta):
            check(f'probe pixels {split}/{i}',r.orbit_hash(a['images'][i])==v['orbit_sha256']);hashes.append(v['orbit_sha256'])
    check('fresh probe disjoint',len(set(hashes))==len(hashes) and not set(hashes)&original)
    for tag in ['fixed','time']:
        summary=json.loads((HERE/f'reports/flow_{tag}_summary.json').read_text())
        check(tag+' nine runs',len(summary['records'])==9)
        noise_by_seed={}
        for record in summary['records']:
            folder=HERE/f'runs/flow_{tag}_s{record["seed"]}_{record["mode"]}'
            run=json.loads((folder/'run.json').read_text());metrics=json.loads((folder/'samples/metrics.json').read_text())
            saved=torch.load(folder/'samples/samples.pt',map_location='cpu',weights_only=True)
            check(str(folder)+' checkpoint',r.sha(folder/'flow.pt')==run['checkpoint_sha256']==metrics['flow_sha256'])
            check(str(folder)+' noise',digest_tensor(saved['noise'])==record['noise_sha256']==metrics['noise_sha256'])
            noise_by_seed[record['seed']]=record['noise_sha256']
            rows=read_csv(folder/'samples/rows.csv');check(str(folder)+' row count',len(rows)==576*3)
            for steps in [16,32,64]:
                sr=[v for v in rows if int(v['steps'])==steps]
                check(str(folder)+f' paired sample ids {steps}',{int(v['sample']) for v in sr}==set(range(576)))
                for split in ['seen','ood']:
                    values=[v for v in sr if (v['ood']=='True')==(split=='ood')]
                    check(str(folder)+f' count {steps}/{split}',len(values)==(480 if split=='seen' else 96))
                    for key in ['joint_correct','strict_accepted_and_joint']:
                        mean=float(np.mean([int(v[key]) for v in values]))
                        check(str(folder)+f' {steps}/{split}/{key}',abs(mean-metrics[str(steps)][split][key])<1e-12)
            threshold=metrics['strict_threshold']
            for v in rows:
                joint=int(v['predicted_shape'])==int(v['shape']) and int(v['predicted_color'])==int(v['color'])
                screen=v['nonempty']=='True' and float(v['occupancy'])<=threshold['max_occupancy'] and float(v['template_iou'])>=threshold['min_iou'] and float(v['class_margin'])>=threshold['min_margin']
                check(str(folder)+f' sample predicates {v["steps"]}/{v["sample"]}',int(v['joint_correct'])==int(joint) and int(v['strict_accepted_and_joint'])==int(joint and screen))
            if tag=='time':check(str(folder)+' time budget',run['train_wall_seconds']>=run['budget_seconds'] and run['train_wall_seconds']-run['budget_seconds']<.1)
        check(tag+' independent noise seeds',len(set(noise_by_seed.values()))==3)
    bound=json.loads((HERE/'reports/invariant_bound_v1/summary.json').read_text())
    check('invariant bound residual',bound['max_identity_residual']<1e-6 and bound['all_512_scenes_above_bound'])
    status=json.loads((HERE/'status.json').read_text())
    check('human review not falsely completed',status['tasks']['A04']['status']=='awaiting_human_review')
    check('goal still active',status['goal_status']=='active')
    result={'checked_unix':time.time(),'checks_passed':len(checks),'original_files_verified':len(manifest['files']),
            'trained_flows_verified':18,'scope':'Initial follow-up diagnostics and original-data fixed/time flow matrices only; not confirmation, multiple objects, dynamics or full goal completion.',
            'all_passed':True}
    dump(HERE/'reports/initial_verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
