"""Independent artifact and row aggregation checks for the decoder mechanism study."""
import csv
import json
from common import HERE, ROOT, dump, np, r, update_status


def main():
    c=json.loads((HERE/'configs/decoder_study_v1.json').read_text());total=0;checks=0;equiv=[]
    for seed in c['ae_seeds']:
        for role in c['roles']:
            matched=[]
            for kind in c['decoders']:
                folder=HERE/f'runs/decoder_study_v1/{role}_{kind}_s{seed}'
                run=json.loads((folder/'run.json').read_text());m=json.loads((folder/'metrics.json').read_text())
                assert run['checkpoint_sha256']==r.sha(folder/'decoder.pt')==m['checkpoint_sha256']
                assert run['ae_sha256']==r.sha(ROOT/f'runs/repair_equivariant_n1024_s{seed}/ae.pt')
                assert run['steps']==6000 and run['config_sha256']==r.sha(HERE/'configs/decoder_study_v1.json')
                assert run['code_sha256']==r.sha(HERE/'decoder_study.py')
                rows=list(csv.DictReader((folder/'rows.csv').open()));assert len(rows)==3072
                for split in ['val','test','ood']:
                    sr=[v for v in rows if v['split']==split]
                    assert len(sr)==1024 and len({(int(v['base_id']),int(v['rotation'])) for v in sr})==1024
                    assert {int(v['base_id']) for v in sr}==set(range(256)) and {int(v['rotation']) for v in sr}==set(range(4))
                    for key,v in m['splits'][split].items():
                        actual=float(np.mean([float(row[key]) for row in sr]));assert abs(actual-v['mean'])<1e-10
                        assert v['base_scene_ci95'][0]<=v['mean']<=v['base_scene_ci95'][1];checks+=1
                assert m['max_c4_equivariance_error']<1e-5;equiv.append(m['max_c4_equivariance_error'])
                matched.append(run);total+=1
            for key in ['parameters','initial_weights_sha256','train_image_sha256','train_latent_sha256']:
                assert len({v[key] for v in matched})==1
    result={'all_passed':True,'decoder_trainings':total,'aggregate_metrics_checked':checks,
        'max_c4_equivariance_error':max(equiv),'original_frozen_ae_hashes_verified':True,
        'controlled_pair_matching_verified':True,'verifier_sha256':r.sha(__file__),
        'claim_boundary':'Reconstruction mechanism experiment; data seed is original 42, evaluation reuses the diagnostic pool. Not independent confirmation of a new generation method.'}
    dump(HERE/'reports/decoder_study_v1/verification.json',result)
    update_status('A10','complete',['reports/decoder_study_v1/summary.json','reports/decoder_study_v1/verification.json'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
