import csv
import json
import numpy as np
import torch
from common import HERE,dump,r


def main():
    root=HERE/'reports/object_failure_audit_v1';cp=HERE/'configs/object_failure_audit_v1.json';c=json.loads(cp.read_text());summary=json.loads((root/'summary.json').read_text())
    assert summary['config_sha256']==r.sha(cp) and c['source_sha256']==r.sha(HERE/'object_failure_audit.py')
    assert c['input_manifest_sha256']==r.sha(HERE/'reports/object_layers_heldout_v1/summary.json')
    partitions=list(csv.DictReader((root/'partition_rows.csv').open()));recon=list(csv.DictReader((root/'reconstruction_rows.csv').open()))
    hidden={};recon_checked=partition_checked=0
    for split in c['splits']:
        saved=torch.load(HERE/f'reports/object_layers_heldout_v1/{split}/outputs.pt',map_location='cpu',weights_only=True)['outputs']
        with np.load(HERE/f'data/object_world_v1/{split}/scenes.npz') as a:
            gt=a['segmentation'].copy();rgb=a['images'].astype(np.float32).transpose(0,3,1,2)/255
            amodal=a['amodal_masks'].sum((2,3));visible=a['visible_masks'].sum((2,3))
        ratio=np.min(np.divide(visible,amodal,out=np.ones_like(visible,dtype=float),where=amodal>0),axis=1)
        hidden[split]=sum(int(np.count_nonzero(amodal[i]))-len(np.unique(gt[i][gt[i]>0])) for i in range(len(gt)))
        for row in [v for v in recon if v['split']==split]:
            i=int(row['base_id']);pred=saved[row['kind']]['image'][i].numpy();target=rgb[i]
            intensity=target.max(0);mask=intensity>.05;pm=pred.max(0)>.2;tm=intensity>.2
            mse=np.mean((pred-target)**2);fg=np.sum(np.abs(pred-target)*mask[None])/(3*max(1,mask.sum()))
            iou=np.count_nonzero(pm&tm)/max(1,np.count_nonzero(pm|tm))
            for key,value in [('mse',mse),('foreground_mae',fg),('mask_iou',iou)]:assert abs(float(row[key])-value)<1e-6
            recon_checked+=1
        for row in [v for v in partitions if v['split']==split]:
            i=int(row['base_id']);kind=row['kind'];variant=row['variant'];pred=saved[kind]['segmentations'][variant][i].numpy()
            objects=np.unique(gt[i][gt[i]>0]);dominant=[];fragmented=0
            for obj in objects:
                labels,counts=np.unique(pred[gt[i]==obj],return_counts=True);dominant.append(int(labels[np.argmax(counts)]))
                fragmented+=int(counts.max()/counts.sum()<.8)
            native=variant.endswith('_native');merged=any(dominant.count(v)>1 for v in set(dominant) if native or v!=0)
            assert float(row['merged_scene'])==float(merged)
            assert abs(float(row['fragmented_object_fraction'])-fragmented/len(objects))<1e-12
            assert abs(float(row['minimum_visibility'])-ratio[i])<1e-12;partition_checked+=1
    for result in summary['partitions']:
        selected=[v for v in partitions if all(v[key]==result[key] for key in ['split','kind','variant'])]
        for key,stat in result['metrics'].items():assert abs(np.mean([float(v[key]) for v in selected])-stat['mean'])<1e-12
    for result in summary['reconstruction']:
        selected=[v for v in recon if all(v[key]==result[key] for key in ['split','kind'])]
        for key,stat in result['metrics'].items():assert abs(np.mean([float(v[key]) for v in selected])-stat['mean'])<1e-12
    for result in summary['occlusion_bins']:
        selected=[v for v in partitions if v['split']=='occlusion' and v['kind']==result['kind'] and v['variant']==result['variant']
                  and result['low']<=float(v['minimum_visibility'])<(result['high'] if result['high']<1 else 1.000001)]
        assert len(selected)==result['scenes']
        assert abs(np.mean([float(v['matched_visible_iou']) for v in selected])-result['matched_visible_iou']['mean'])<1e-12
    dump(root/'verification.json',{'all_passed':True,'partition_rows_verified':partition_checked,'reconstruction_rows_verified':recon_checked,
        'objects_without_any_hard_visible_pixel_by_split':hidden,'verifier_sha256':r.sha(__file__),'claim_boundary':c['boundary']})


if __name__=='__main__':main()
