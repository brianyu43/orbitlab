"""Check paired saved latents, source checkpoints, and decoder transfer aggregates."""
import csv
import json
from statistics import mean
from common import HERE, ROOT, dump, digest_tensor, torch, r


def main():
    p=HERE/'reports/decoder_transfer_v1';d=json.loads((p/'summary.json').read_text());checks=0
    for run in d['runs']:
        folder=HERE/f'runs/flow_fixed_s{run["seed"]}_equivariant'
        samples=torch.load(folder/'samples/samples.pt',map_location='cpu',weights_only=True)
        assert digest_tensor(samples['latents'][64])==run['raw_latent_sha256']
        assert digest_tensor(samples['noise'])==run['noise_sha256']
        assert r.sha(folder/'flow.pt')==run['flow_sha256']
        ae=ROOT/f'runs/repair_equivariant_n1024_s{run["seed"]}/ae.pt';assert r.sha(ae)==run['ae_sha256']
        decoder=ae if run['kind']=='original' else HERE/f'runs/decoder_study_v1/learned_{run["kind"]}_s{run["seed"]}/decoder.pt'
        assert r.sha(decoder)==run['decoder_sha256']
        rows=list(csv.DictReader((p/f'{run["kind"]}_s{run["seed"]}'/'rows.csv').open()));assert len(rows)==576
        for split in ['seen','ood']:
            rr=[v for v in rows if (v['ood']=='True')==(split=='ood')];assert len(rr)==run['metrics'][split]['n']
            for key in ['joint_correct','strict_accepted_and_joint','template_iou']:
                actual=mean(float(v[key]=='True') if key!='template_iou' else float(v[key]) for v in rr)
                assert abs(actual-run['metrics'][split][key])<1e-12;checks+=1
        if run['kind']=='original':
            source=json.loads((folder/'samples/metrics.json').read_text())['64']
            for split in ['seen','ood']:
                for key in ['joint_correct','strict_accepted_and_joint']:assert run['metrics'][split][key]==source[split][key]
    report={'all_passed':True,'metrics_reaggregated':checks,'original_decode_matches_previous_evaluation':True,
        'decoder_runs':len(d['runs']),'rows':len(d['runs'])*576,'source_checkpoint_and_latent_hashes_verified':True,'verifier_sha256':r.sha(__file__)}
    dump(p/'verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
