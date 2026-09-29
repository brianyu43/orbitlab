"""Reconstruction, merge/fragment errors and occlusion severity of frozen outputs."""
import json
import time
import numpy as np
import torch
from common import HERE,dump,fresh_dir,r
from object_perception import images_only
from object_layer_diagnostic import score


def partition_errors(gt,pred,native):
    dominant=[];fragmented=0;objects=np.unique(gt[gt>0])
    for object_id in objects:
        labels,counts=np.unique(pred[gt==object_id],return_counts=True);largest=int(counts.argmax())
        dominant.append(int(labels[largest]));fragmented+=int(counts[largest]/counts.sum()<.8)
    merged=any(dominant.count(v)>1 for v in set(dominant) if native or v!=0)
    return {'merged_scene':float(merged),'fragmented_object_fraction':fragmented/len(objects)}


def main():
    torch.set_num_threads(2);cp=HERE/'configs/object_failure_audit_v1.json'
    if not cp.exists():dump(cp,{'splits':['test','ood','count3','count4','occlusion'],
        'occlusion_bins':[0,.5,.75,.9,1.000001],'fragmentation':'Largest predicted region covers less than 80% of a true visible object.',
        'merging':'At least two true visible objects have the same dominant predicted foreground label.',
        'severity':'Minimum per-object visible-alpha sum / amodal-alpha sum, over present objects.',
        'source_sha256':r.sha(__file__),'input_manifest_sha256':r.sha(HERE/'reports/object_layers_heldout_v1/summary.json'),
        'boundary':'Post-hoc descriptive failure analysis of fixed held-out outputs; no parameter or threshold tuning.', 'frozen_unix':time.time()})
    c=json.loads(cp.read_text());root=fresh_dir(HERE/'reports/object_failure_audit_v1');rows=[];recon=[];bins=[]
    for split in c['splits']:
        folder=HERE/f'reports/object_layers_heldout_v1/{split}'
        saved=torch.load(folder/'outputs.pt',map_location='cpu',weights_only=True)['outputs'];x=images_only(split)
        with np.load(HERE/f'data/object_world_v1/{split}/scenes.npz') as a:
            gt=a['segmentation'].copy();amodal=a['amodal_masks'].astype(float).sum((2,3));visible=a['visible_masks'].astype(float).sum((2,3))
        severity=np.min(np.divide(visible,amodal,out=np.ones_like(visible),where=amodal>0),axis=1)
        for kind,values in saved.items():
            if 'image' in values:
                metrics=r.pixel_metrics(values['image'],x)
                for i in range(len(x)):recon.append({'split':split,'kind':kind,'base_id':i,**{k:float(v[i]) for k,v in metrics.items()}})
            for variant,segments in values['segmentations'].items():
                native=variant.endswith('_native')
                for i,pred in enumerate(segments.numpy()):
                    rows.append({'split':split,'kind':kind,'variant':variant,'base_id':i,'minimum_visibility':float(severity[i]),
                        **score(gt[i],pred,native,4 if kind=='component' else 5),**partition_errors(gt[i],pred,native)})
        if split=='occlusion':
            for low,high in zip(c['occlusion_bins'][:-1],c['occlusion_bins'][1:]):
                for kind,variant in [('component','native'),('flat','decoder_rgb_foreground'),('slot','decoder_rgb_foreground')]:
                    selected=[v for v in rows if v['split']==split and v['kind']==kind and v['variant']==variant and low<=v['minimum_visibility']<high]
                    if selected:bins.append({'low':low,'high':min(high,1.),'kind':kind,'variant':variant,'scenes':len(selected),
                        'matched_visible_iou':r.bootstrap([v['matched_visible_iou'] for v in selected]),
                        'merged_scene_rate':float(np.mean([v['merged_scene'] for v in selected]))})
    summary=[]
    for split,kind,variant in sorted({(v['split'],v['kind'],v['variant']) for v in rows}):
        selected=[v for v in rows if (v['split'],v['kind'],v['variant'])==(split,kind,variant)]
        summary.append({'split':split,'kind':kind,'variant':variant,'scenes':len(selected),
            'metrics':{key:r.bootstrap([v[key] for v in selected]) for key in ['matched_visible_iou','merged_scene','fragmented_object_fraction']}})
    reconstructed=[]
    for split,kind in sorted({(v['split'],v['kind']) for v in recon}):
        selected=[v for v in recon if v['split']==split and v['kind']==kind]
        reconstructed.append({'split':split,'kind':kind,'scenes':len(selected),
            'metrics':{key:r.bootstrap([v[key] for v in selected]) for key in ['mse','foreground_mae','mask_iou']}})
    r.write_csv(root/'partition_rows.csv',rows);r.write_csv(root/'reconstruction_rows.csv',recon)
    dump(root/'summary.json',{'partitions':summary,'reconstruction':reconstructed,'occlusion_bins':bins,
        'config_sha256':r.sha(cp),'claim_boundary':c['boundary']})
    print(json.dumps({'partition_rows':len(rows),'reconstruction_rows':len(recon),'occlusion_bins':bins},indent=2))


if __name__=='__main__':main()
