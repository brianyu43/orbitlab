"""Pack official full-size dSprites Hard/Single-Atomic images losslessly."""
from pathlib import Path
import tarfile,io,json,hashlib,time
import numpy as np
from PIL import Image
BASE=Path(__file__).resolve().parent/'svib'
ARCHIVE=BASE.parent/'svib_dsprites_hard.archive'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    BASE.mkdir(exist_ok=True)
    if (BASE/'data_manifest.json').exists():return
    arrays={};counts={};seen=set();metadata={};tasks=set();started=time.time()
    for split,n in [('train',64000),('test',8000)]:
        for role in ['source','target']:
            arrays[split,role]=np.lib.format.open_memmap(BASE/f'{split}_{role}.npy',mode='w+',dtype=np.uint8,shape=(n,128,128,3))
            counts[f'{split}_{role}']=0
    with tarfile.open(ARCHIVE,'r|gz') as tf:
        for member in tf:
            parts=Path(member.name).parts
            if len(parts)>=2:tasks.add(parts[1])
            if not member.isfile() or len(parts)!=5 or parts[1]!='Single_Atomic':continue
            _,task,split,index,name=parts
            if split not in ['train','test']:continue
            i=int(index)
            if name in ['source.png','target.png']:
                role=name.split('.')[0];key=(split,role,i);assert key not in seen;seen.add(key)
                image=np.asarray(Image.open(io.BytesIO(tf.extractfile(member).read())).convert('RGB'))
                assert image.shape==(128,128,3);arrays[split,role][i]=image;counts[f'{split}_{role}']+=1
            elif name in ['source.json','target.json']:
                metadata[f'{split}/{index}/{name}']=json.load(tf.extractfile(member))
            if sum(counts.values()) and sum(counts.values())%20000==0 and name=='target.png':print('SVIB packing',counts,flush=True)
    assert counts=={'train_source':64000,'train_target':64000,'test_source':8000,'test_target':8000},(counts,tasks)
    for array in arrays.values():array.flush()
    (BASE/'metadata.json').write_text(json.dumps(metadata,separators=(',',':')))
    manifest={'archive_sha256':sha(ARCHIVE),'counts':counts,'tasks_in_archive':sorted(tasks),
              'files':{path.name:sha(path) for path in BASE.glob('*.npy')},'metadata_sha256':sha(BASE/'metadata.json'),
              'source_sha256':sha(Path(__file__)),'seconds':time.time()-started,
              'url':'https://systematic-visual-imagination.github.io/',
              'download_id':'1cANAcSP3_fhsy04lv2K17907HaFQGG6P',
              'scope':'Official full dSprites Hard Single Atomic split, original 128x128 PNG values; other three tasks retained only in downloaded archive. Not all 12 SVIB tasks.'}
    (BASE/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest,flush=True)

if __name__=='__main__':main()
