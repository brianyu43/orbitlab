"""Correct pose metrics without changing or retraining any readout checkpoint."""
import json
import torch
from common import HERE,dump,fresh_dir,r
from train_object_readout import labels
from object_readout_metrics_v2 import measure


def main():
    c=json.loads((HERE/'configs/object_readout_v1.json').read_text());root=fresh_dir(HERE/'reports/object_readout_pose_corrected_v1');results=[]
    for kind in c['kinds']:
        for seed in c['seeds']:
            source=HERE/f'runs/object_readout_v1/{kind}_s{seed}';dest=fresh_dir(root/f'{kind}_s{seed}')
            raw=torch.load(source/'predictions.pt',map_location='cpu',weights_only=True);summary={};all_matches={}
            for split in ['train','val']:
                rows,matches=measure(raw[split],labels(split));all_matches[split]=matches;r.write_csv(dest/f'{split}_rows.csv',rows)
                keys=['count_correct','matched_fraction','shape_correct','color_correct','position_within_one_pixel','radius_within_half_pixel','pose_correct','object_all_factors_correct','full_scene_success']
                summary[split]={key:r.bootstrap([v[key] for v in rows]) for key in keys}
                pos=[v['matched_position_mae_pixels'] for v in rows if v['matched_position_mae_pixels'] is not None]
                summary[split]['matched_position_mae_pixels']=r.bootstrap(pos) if pos else None
            dump(dest/'assignments.json',all_matches);dump(dest/'metrics.json',{'splits':summary,'predictions_sha256':r.sha(source/'predictions.pt'),
                'checkpoint_sha256':r.sha(source/'model.pt'),'corrected_evaluator_sha256':r.sha(HERE/'object_readout_metrics_v2.py')})
            results.append({'kind':kind,'seed':seed,'splits':summary})
    dump(root/'summary.json',{'results':results,'original_summary_sha256':r.sha(HERE/'reports/object_readout_v1/summary.json'),
        'runner_sha256':r.sha(__file__),'corrected_evaluator_sha256':r.sha(HERE/'object_readout_metrics_v2.py'),
        'correction':'Original evaluation used exact vector equality for C4 poses; target sin/cos contain roundoff near zero. Compare nearest C4 categories instead. Model training and stored predictions are unchanged.',
        'claim_boundary':c['claim_boundary']})


if __name__=='__main__':main()
