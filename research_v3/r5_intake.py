"""Lossless, indexed preparation of the existing official dSprites archive.

PNG/JSON/mask payloads remain byte-identical. This only prepares data, never
trains, selects checkpoints, mutates historical files or uploads datasets.
"""
import argparse,hashlib,io,json,os,sqlite3,tarfile,time
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from v3_common import ROOT,HERE,sha,dump,lock,event

BASE=HERE/'r5'
ARCHIVE=ROOT/'completion_v2/svib_dsprites_hard.archive'
TASKS=('Single_Atomic','Multiple_Atomic','Single_Non-Atomic','Multiple_Non-Atomic')
NAMES=('source.png','target.png','source.json','target.json','source_mask.png','target_mask.png')
COUNTS={'train':64000,'test':8000}

def freeze():
    historical=json.loads((ROOT/'completion_v2/svib/data_manifest.json').read_text())
    return lock(BASE/'intake_protocol.json',{'source_sha256':sha(__file__),'task_scope':list(TASKS),
        'origin':'https://systematic-visual-imagination.github.io/','official_archive_id':'1cANAcSP3_fhsy04lv2K17907HaFQGG6P',
        'archive_sha256':historical['archive_sha256'],'split_counts':COUNTS,'files_each':list(NAMES),
        'format':'Concatenated original PNG/JSON payload bytes plus read-only SQLite offset/size/SHA256 index. No resampling, color conversion or label editing.',
        'license_source':'Official paper arXiv2311.09064v1 AppendixA.6 Question4 states datasetCC0; official code LICENSE isCC0-1.0. Local research only; no new public dataset redistribution.',
        'scope':'Prepare existing archive only while R4 trains. New model/split/epoch protocols are still pending; none of these bytes are new research results.',
        'storage_cap_bytes':30000000000})

def member_key(name):
    parts=Path(name).parts
    if len(parts)!=5:return None
    difficulty,task,split,episode,filename=parts
    if difficulty!='Hard' or task not in TASKS or split not in COUNTS or filename not in NAMES:return None
    number=int(episode);assert 0<=number<COUNTS[split]
    index=(number+(64000 if split=='test' else 0))*6+NAMES.index(filename)
    return task,split,number,filename,index

class Store:
    def __init__(self,task):
        self.path=BASE/'data/dsprites_hard'/task
        self.db=sqlite3.connect(f'file:{self.path / "index.sqlite"}?mode=ro',uri=True)
        self.fd=os.open(self.path/'payload.bin',os.O_RDONLY)
    def read(self,split,index,filename):
        key=(int(index)+(64000 if split=='test' else 0))*6+NAMES.index(filename)
        row=self.db.execute('SELECT offset,size,sha256 FROM records WHERE id=?',(key,)).fetchone();assert row is not None
        value=os.pread(self.fd,row[1],row[0]);assert len(value)==row[1]
        return value,row[2]
    def image(self,split,index,role='source'):
        raw,_=self.read(split,index,role+'.png');return np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))
    def metadata(self,split,index,role='source'):
        return json.loads(self.read(split,index,role+'.json')[0])
    def close(self):self.db.close();os.close(self.fd)

def prepare():
    protocol=freeze();assert sha(ARCHIVE)==protocol['archive_sha256'];dest=BASE/'data/dsprites_hard'
    if (dest/'manifest.json').exists():
        m=json.loads((dest/'manifest.json').read_text());assert m['protocol_sha256']==sha(BASE/'intake_protocol.json')
        for path,h in m['files'].items():assert sha(dest/path)==h
        return
    assert not dest.exists(),'Retain and inspect incomplete intake; never overwrite original or partial evidence'
    dest.mkdir(parents=True);handles={};counts=Counter();started=time.perf_counter()
    for task in TASKS:
        p=dest/task;p.mkdir();db=sqlite3.connect(p/'index.sqlite');db.execute('CREATE TABLE records (id INTEGER PRIMARY KEY, offset INTEGER NOT NULL, size INTEGER NOT NULL, sha256 TEXT NOT NULL)')
        handles[task]=(db,(p/'payload.bin').open('wb'))
    try:
        with tarfile.open(ARCHIVE,'r|gz') as tar:
            for member in tar:
                key=member_key(member.name)
                if not member.isfile() or key is None:continue
                task,split,index,name,pk=key;raw=tar.extractfile(member).read();assert len(raw)==member.size
                db,handle=handles[task];db.execute('INSERT INTO records VALUES (?,?,?,?)',(pk,handle.tell(),len(raw),hashlib.sha256(raw).hexdigest()));handle.write(raw);counts[f'{task}/{split}/{name}']+=1
                if sum(counts.values())%50000==0:
                    for d,h in handles.values():d.commit();h.flush()
                    size=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file());assert size<30000000000
                    dump(BASE/'intake_progress.json',{'files':sum(counts.values()),'bytes':size,'seconds':time.perf_counter()-started,'phase':'copying original payloads'});print('R5 intake',sum(counts.values()),size,flush=True)
        for task in TASKS:
            for split,n in COUNTS.items():
                for name in NAMES:assert counts[f'{task}/{split}/{name}']==n
            db,handle=handles[task];db.commit();handle.close()
            n,lo,hi=db.execute('SELECT count(*),min(id),max(id) FROM records').fetchone();assert (n,lo,hi)==(432000,0,431999)
            assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok';db.close()
    except BaseException:
        for db,handle in handles.values():
            try:db.commit();db.close();handle.close()
            except Exception:pass
        raise
    files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}
    dump(dest/'manifest.json',{'protocol_sha256':sha(BASE/'intake_protocol.json'),'archive_sha256':protocol['archive_sha256'],'files':files,'counts':dict(counts),'total_episodes':288000,'total_payloads':1728000,'seconds':time.perf_counter()-started,'new_model_training':False})
    event('R5_existing_archive_prepared',episodes=288000);print('R5 data prepared',flush=True)

def verify():
    freeze();dest=BASE/'data/dsprites_hard';m=json.loads((dest/'manifest.json').read_text());assert sha(ARCHIVE)==m['archive_sha256']
    for path,h in m['files'].items():assert sha(dest/path)==h
    stores={task:Store(task) for task in TASKS};counts=Counter();started=time.perf_counter();examples={}
    with tarfile.open(ARCHIVE,'r|gz') as tar:
        for member in tar:
            key=member_key(member.name)
            if not member.isfile() or key is None:continue
            task,split,index,name,_=key;expected=tar.extractfile(member).read();actual,h=stores[task].read(split,index,name)
            assert expected==actual and hashlib.sha256(actual).hexdigest()==h;counts[f'{task}/{split}/{name}']+=1
            if index<8:
                if name.endswith('.png'):
                    image=np.asarray(Image.open(io.BytesIO(actual)));assert image.shape[:2]==(128,128);examples[f'{task}/{split}/{name}']={'shape':list(image.shape),'dtype':str(image.dtype)}
                else:assert len(json.loads(actual)['objects'])==2
    for s in stores.values():s.close()
    assert dict(counts)==m['counts']
    dump(dest/'verification.json',{'manifest_sha256':sha(dest/'manifest.json'),'every_original_archive_payload_replayed':sum(counts.values()),'complete_contiguous_episodes':288000,'native_format_spot_checks':examples,'seconds':time.perf_counter()-started,'source_sha256':sha(__file__),'scope':'Lossless byte/source/index verification; exhaustive decoded semantic/split audit is a later step.'})
    print('R5 archive bytes fully replayed',sum(counts.values()),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args();verify() if a.verify else prepare()
