"""Official split integrity, semantic transformation and exact RGB overlap audit."""
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np
from svib_prepare import sha
import svib_run as s
manifest=json.loads((s.BASE/'data_manifest.json').read_text())
for name,digest in manifest['files'].items():assert sha(s.BASE/name)==digest
assert sha(s.BASE/'metadata.json')==manifest['metadata_sha256']
meta=json.loads((s.BASE/'metadata.json').read_text());summary={};hashes={};combos={}
for split,n in [('train',64000),('test',8000)]:
    source=np.load(s.BASE/f'{split}_source.npy',mmap_mode='r');target=np.load(s.BASE/f'{split}_target.npy',mmap_mode='r')
    assert source.shape==target.shape==(n,128,128,3)
    errors=[];pairs=Counter();states=set();hashes[split]=set()
    for i in range(n):
        prefix=f'{split}/{i:08d}'
        a=meta[prefix+'/source.json']['objects'];b=meta[prefix+'/target.json']['objects'];assert len(a)==len(b)==2
        for j in range(2):
            if b[j]['shape']!=a[1-j]['shape']:errors.append([i,j,'shape'])
            for k in ['size','rotation','2d_coords','color']:
                if a[j][k]!=b[j][k]:errors.append([i,j,k])
        pairs[' / '.join(sorted(v['shape'] for v in a))]+=1
        for v in a:states.add((v['shape'],v['size'],tuple(v['color'])))
        hashes[split].add(hashlib.sha256(source[i].tobytes()).hexdigest())
    assert not errors
    combos[split]=states
    summary[split]={'n':n,'shape_swap_errors':errors,'shape_pairs':dict(pairs),'unique_source_RGB':len(hashes[split]),'unique_shape_size_color_attributes':len(states)}
summary['exact_source_RGB_train_test_overlap']=len(hashes['train']&hashes['test'])
summary['test_object_attribute_combinations_unseen_in_train']=len(combos['test']-combos['train'])
summary['scope']='Full official representative split; exact RGB overlap audit is not a proof of semantic/orbit disjointness.'
summary['source_sha256']=sha(Path(__file__))
s.write(s.BASE/'data_audit.json',summary)
print(json.dumps(summary,indent=2))
