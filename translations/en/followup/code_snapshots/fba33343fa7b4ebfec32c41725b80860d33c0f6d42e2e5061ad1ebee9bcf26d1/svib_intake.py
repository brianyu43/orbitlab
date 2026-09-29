"""Inspect the authors' published preview without treating it as a full benchmark."""
import collections
import json
import subprocess
from pathlib import Path
from PIL import Image
from common import HERE,dump,fresh_dir,update_status,np,torch,o,r


def binding(v):return (v['shape'],tuple(v['color']),v['size'])


def main():
    root=HERE/'external/svib-samples/samples/dSprites/Single_Atomic'
    out=fresh_dir(HERE/'reports/svib_intake_v1')
    bins=collections.defaultdict(lambda:{'source':set(),'target':set()});counts=collections.Counter()
    identical=collections.Counter();bad=[];files=[];examples=[];row_data=[]
    for p in sorted(root.rglob('source.json')):
        split=str(p.parent.parent.relative_to(root));source=json.loads(p.read_text());target=json.loads((p.parent/'target.json').read_text())
        a=np.asarray(Image.open(p.parent/'source.png').convert('RGB'));b=np.asarray(Image.open(p.parent/'target.png').convert('RGB'))
        assert a.shape==b.shape==(128,128,3)
        counts[split]+=1;identical[split]+=int(np.array_equal(a,b))
        for side,d in [('source',source),('target',target)]:bins[split][side].update(binding(v) for v in d['objects'])
        valid=len(source['objects'])==len(target['objects'])==2
        valid=valid and [v['shape'] for v in source['objects']][::-1]==[v['shape'] for v in target['objects']]
        valid=valid and all(all(x[k]==y[k] for k in ['color','size','rotation','2d_coords']) for x,y in zip(source['objects'],target['objects']))
        if not valid:bad.append(str(p.relative_to(root)))
        row_data.append({'split':split,'episode':p.parent.name,'source_target_identical':bool(np.array_equal(a,b)),
            'identity_pixel_mse':float(np.mean((a.astype(float)/255-b.astype(float)/255)**2)),'shape_swap_rule_matches':valid})
        for name in ['source.png','target.png','source.json','target.json']:
            f=p.parent/name;files.append({'path':str(f.relative_to(root)),'sha256':r.sha(f)})
        if counts[split]<=4:examples.extend([torch.from_numpy(a.copy()).permute(2,0,1).float()/255,torch.from_numpy(b.copy()).permute(2,0,1).float()/255])
    assert len(row_data)==500 and not bad
    exposure={}
    for split in sorted(counts):
        exposure[split]={'episodes':counts[split],'observed_source_bindings':len(bins[split]['source']),
            'observed_target_bindings':len(bins[split]['target']),
            'source_overlap_test_source':len(bins[split]['source']&bins['Test']['source']),
            'target_overlap_test_source':len(bins[split]['target']&bins['Test']['source']),
            'identical_source_target_episodes':identical[split]}
    commits={name:subprocess.check_output(['git','-C',str(HERE/'external'/name),'rev-parse','HEAD'],text=True).strip() for name in ['svib','svib-samples']}
    dump(out/'manifest.json',{'repositories':{'svib':'https://github.com/systematic-visual-imagination/svib','svib-samples':'https://github.com/systematic-visual-imagination/svib-samples'},
        'commits':commits,'download_scope':'Sparse checkout of dSprites Single Atomic preview only','files':files,'exposure':exposure,
        'rule_failures':bad,'resolution':[128,128],'masks_present':False,
        'status':'intake_complete_external_training_not_yet_performed','code_sha256':r.sha(__file__)})
    r.write_csv(out/'episodes.csv',row_data);o.grid(torch.stack(examples),out/'source_target_pairs.png',8)
    text=['# SVIB Representative Task Collection and Protocol Inspection','',
        'I only selected dSprites / Single Atomic (Shape-Swap) from the public example repository linked in the official project. The original image is 128×128 and this preview does not have a object mask. There is the correct metadata and image pair.','',
        f'Official code commit: `{commits["svib"]}`. Public example commit: `{commits["svib-samples"]}`.','',
        '| Division | Example count | Input combination count | Target combination count | Input overlaps with test input combination | Target overlaps with test input combination | Input=Target |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split,d in exposure.items():text.append('| '+split+' | '+' | '.join(str(d[k]) for k in ['episodes','observed_source_bindings','observed_target_bindings','source_overlap_test_source','target_overlap_test_source','identical_source_target_episodes'])+' |')
    text += ['', 'All 500 metadata pairs followed the rule of only changing the shapes of two objects while maintaining their color, position, and size. Some examples have identical shapes of the two objects, meaning the input and target images are the same. Therefore, a baseline that does not change anything must be included, and the performance of examples that actually involve changes must also be reported separately.','',
        'The learning/test separation of input combinations was confirmed in this preview. However, some combinations similar to test inputs appear in the learning target images. This is a recorded exposure condition that includes the scope of evaluating generalization of input combinations for this task. Cases where image-to-image learning and source-only representation learning are distinguished, and supervised pretraining is performed by mixing all input and target images, should be marked separately. This observation alone does not definitively indicate a leakage across the entire official benchmark.','',
        'The official generation code saves the RGB renderer array to cv2.imwrite and the official reading code uses PIL RGB. When creating an evaluator that compares metadata color and saved PNG color, the channel order must be checked. The original protocol that learns and evaluates the input and target PNGs unchanged is not arbitrarily changed.','',
        'The next task is to connect this form of data adapter with the identity baseline and a reduced prediction model. Each of the 100 examples in this preview is not reported as part of the full 64,000 training/8,000 test benchmark results.','',
        '[Official benchmark description](https://systematic-visual-imagination.github.io/) · [Official code](https://github.com/systematic-visual-imagination/svib) · [Official examples](https://github.com/systematic-visual-imagination/svib-samples)','']
    (out/'INTAKE_KO.md').write_text('\n'.join(text))
    update_status('B06','in_progress',['reports/svib_intake_v1/manifest.json','reports/svib_intake_v1/INTAKE_KO.md'],
        'Official preview recovered and inspected; external model training/evaluation remains pending.')
    print(json.dumps({'commits':commits,'exposure':exposure,'episodes_checked':len(row_data)},indent=2))


if __name__=='__main__':torch.set_num_threads(2);main()
