"""Replay, source-family, symmetry and raster-order audits for the R1 experiment."""
import dataclasses,json
import numpy as np
import torch
import r1_core as c
from r1_evaluate import score,template_infer
from v3_common import dump,sha,status

def verify():
    c.freeze();torch.set_num_threads(2);seen=set();data_checks=[]
    for data in sorted((c.BASE/'data').glob('d*')):
        manifest=json.loads((data/'manifest.json').read_text())
        for rel,digest in manifest['files'].items():assert sha(data/rel)==digest
        for split in ['train','val','test','ood']:
            folder=data/split;records=json.loads((folder/'labels.json').read_text())
            for row in records:
                keys={c.canonical(row['objects']),*(c.canonical(t['target']) for t in c.tasks(row['objects'],row['source_index']))}
                assert keys==set(row['family_keys']);assert not keys&seen
                seen.update(keys)
            for renderer in c.RENDERERS:
                images=np.load(folder/f'{renderer}.npz')['images']
                for i,row in enumerate(records):
                    assert np.array_equal(c.render(row['objects'],renderer)[0],images[i])
                    if renderer=='pil4_lanczos':
                        objects=tuple(c.w.ObjectState(**v) for v in row['objects'])
                        assert np.array_equal(c.w.render(objects)['image'],images[i])
            data_checks.append({'seed':manifest['seed'],'split':split,'scenes':len(records),'all_three_renderer_images_replayed':True,'original_renderer_independently_matches_historical_implementation':True})
    # The original source files are read-only; verify against the lock again.
    verified=[];summaries=[];pose_rows=[];raster_order=[]
    for path in sorted((c.BASE/'evaluation').glob('d*/*/*/*/summary.json')):
        rec=json.loads(path.read_text());folder=path.parent
        for name,digest in rec['files'].items():assert sha(folder/name)==digest
        data=c.BASE/f'data/d{rec["seed"]}/{rec["split"]}'
        assert sha(data/f'{rec["renderer"]}.npz')==rec['input_sha256'];assert sha(data/'labels.json')==rec['labels_sha256']
        images=np.load(data/f'{rec["renderer"]}.npz')['images']
        if rec['arm']=='template':pred=template_infer(images,rec['renderer'])
        else:
            cp=c.BASE/f'runs/d{rec["seed"]}_s{rec["init"]}/{rec["arm"]}/model.pt';assert sha(cp)==rec['model_sha256']
            model=c.Reader();model.load_state_dict(torch.load(cp,weights_only=True)['state_dict']);model.eval()
            pred=c.infer(images,model,rec['arm'].split('_')[0])
        assert pred==json.loads((folder/'inference.json').read_text())
        labels=json.loads((data/'labels.json').read_text())
        rows,metrics,_=score(pred,labels,images,rec['renderer'])
        assert rows==json.loads((folder/'rows.json').read_text());assert metrics==rec['metrics']
        # Separate latent raster fingerprints from true half-turn identifiability.
        by_shape={shape:[] for shape in range(4)}
        order_only=0;nonidentical=0
        for p,gt in zip(pred,labels):
            mapping={v['color']:v for v in p}
            for v in gt['objects']:
                guess=mapping.get(v['color'])
                by_shape[v['shape']].append(guess is not None and guess['pose']==v['pose'])
            if set(mapping)=={v['color'] for v in gt['objects']}:
                rendered=c.render(p,rec['renderer'])[0]
                source=c.render(gt['objects'],rec['renderer'])[0]
                ordered=[mapping[v['color']] for v in gt['objects']]
                if not np.array_equal(rendered,source):
                    nonidentical+=1
                    if np.array_equal(c.render(ordered,rec['renderer'])[0],source):order_only+=1
        pose_rows.append({'seed':rec['seed'],'init':rec['init'],'arm':rec['arm'],'split':rec['split'],'renderer':rec['renderer'],
                         'by_shape_pose_accuracy':{str(k):float(np.mean(v)) for k,v in by_shape.items()},'by_shape_count':{str(k):len(v) for k,v in by_shape.items()}})
        raster_order.append({'condition':str(folder.relative_to(c.BASE)),'source_nonidentical_rasters':nonidentical,'source_rasters_repaired_only_by_gt_painter_order':order_only,
                             'warning':'GT order is used only for this audit, never to repair or rescore reported predictions. Tiny antialias fringes can overlap despite >.05 support separation.'})
        verified.append({'condition':str(folder.relative_to(c.BASE)),'all_inferences_replayed':len(images),'all_edit_rows_recomputed':len(rows),'summary_sha256':sha(path)})
        summaries.append(rec)
        print('R1 verified',str(folder.relative_to(c.BASE)),flush=True)
    choice=json.loads((c.BASE/'selection.json').read_text())
    expected=42+(126 if choice['development_gate_passed'] else 0)
    assert len(verified)==expected,(len(verified),expected)
    # Exact pixel equality for balanced counterfactual half turns is a stronger
    # identifiability statement than a low accuracy on a randomly sampled test.
    equal=0;distinct_original=0
    for radius in range(5,9):
        for color in range(6):
            for pose in [0,1]:
                a=[{'id':0,'shape':3,'color':color,'radius':radius,'x':32,'y':32,'pose':pose}]
                b=[{**a[0],'pose':pose+2}]
                assert np.array_equal(c.render(a,'symmetric4_lanczos')[0],c.render(b,'symmetric4_lanczos')[0]);equal+=1
                distinct_original+=int(not np.array_equal(c.render(a,'pil4_lanczos')[0],c.render(b,'pil4_lanczos')[0]))
    dump(c.BASE/'pose_audit.json',{'rows':pose_rows,'exact_equal_balanced_half_turn_pairs':equal,'original_renderer_distinct_pairs':distinct_original,
        'conclusion':'For these paired symmetric-renderer inputs, a balanced four-label target cannot be distinguished beyond50% within a half-turn pair from the image. Original renderer differences expose a raster cue, not geometric orientation.'})
    dump(c.BASE/'raster_order_audit.json',raster_order)
    dump(c.BASE/'aggregate.json',{'development':[s for s in summaries if s['seed']==c.DEVELOPMENT],'confirmation':[s for s in summaries if s['seed']!=c.DEVELOPMENT],
        'selection':choice,'statistical_scope':'Development: one generated training dataset and one initialization. Conditional confirmation: independent generated data and initialization crossed; do not claim confidence from development scenes alone.'})
    dump(c.BASE/'verification.json',{'data':data_checks,'conditions':verified,'verified_units':len(verified),'expected_units':expected,'source_family_keys':len(seen),
        'symmetry_pairs':equal,'protocol_sha256':sha(c.BASE/'protocol.json'),'independent_human_evaluation':False,'scope':'Artifact hashes and complete CPU inference/metric/raster replay; not independent researcher reimplementation or clean-install retraining.'})
    status('R1','verified',development_gate_passed=choice['development_gate_passed'],verification='r1/verification.json',report='r1/RESULTS_KO.md')

if __name__=='__main__':verify()
