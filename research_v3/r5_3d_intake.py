"""Stream an official3Darchive, retain only the preregistered SingleAtomic task.

No authentication, upload, image alteration or archive extraction onto arbitrary
paths. Full HTTP bytes are SHA256-hashed, selected original members indexed.
"""
import argparse,hashlib,io,json,os,sqlite3,tarfile,time,urllib.request
from pathlib import Path
from collections import Counter
from PIL import Image
from v3_common import HERE,sha,dump,lock,event
from r5_intake import BASE,NAMES

SOURCES={'clevr':{'id':'1NkbROrTu38_Wydp_3z7v_1AtSkgXEF0s','advertised_bytes':13921263003},
         'clevrtex':{'id':'1YgJwN6Efl7kXlXRRKiX8g9BxIKBye1zx','advertised_bytes':None}}
TASK='Single_Atomic'

def freeze(family):
    return lock(BASE/f'{family}_intake_protocol.json',{'source_sha256':sha(__file__),'family':family,'task':TASK,'difficulty':'Hard','source':SOURCES[family],
        'origin':'https://systematic-visual-imagination.github.io/','license':'DatasetCC0, officialpaperAppendixA.6Q4','selection':'SingleAtomic chosen before3Dmodel scoring; otherthree tasks transmitted but not stored or trained.',
        'expected_episodes':{'train':64000,'test':8000},'files_each':list(NAMES),'format':'OriginalPNG/JSONbytes, concatenatedpayload andSQLiteoffset,size,SHA256index; no resize/recolor. RGB conversion only when a later model reads pixels.',
        'verification':'Hash all HTTParchivebytes; requireContentLength ifprovided; rehash everyretainedpayload fromdisk andcheck contiguousindex. Official publisher provides no checksum here, so hashis local provenance, not an independentpublisherchecksum.',
        'recovery':'Onconnectionfailure keep partialpayload/index and restart stream frombeginning. Existingmembers mustmatch exactbytes before continuing; never overwrite a completed manifest.',
        'storage_cap_bytes':30000000000})

class HashedStream:
    def __init__(self,response,path):self.response=response;self.digest=hashlib.sha256();self.count=0;self.last=0;self.start=time.perf_counter();self.path=path
    def read(self,n=-1):
        value=self.response.read(n);self.digest.update(value);self.count+=len(value)
        if self.count-self.last>=67108864:
            self.last=self.count;dump(self.path,{'received_archive_bytes':self.count,'seconds':time.perf_counter()-self.start,'phase':'streaming_official_archive'})
            print('R5 download',self.count,round(time.perf_counter()-self.start,1),flush=True)
        return value

def key(name):
    parts=Path(name).parts
    if len(parts)!=5:return None
    difficulty,task,split,index,filename=parts
    if difficulty!='Hard' or task!=TASK or split not in ['train','test'] or filename not in NAMES:return None
    number=int(index);assert 0<=number<(64000 if split=='train' else 8000)
    return split,number,filename,(number+(64000 if split=='test' else 0))*6+NAMES.index(filename)

def prepare(family):
    freeze(family);dest=BASE/'data'/f'{family}_hard'/TASK;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists():return verify(family)
    url=f'https://drive.usercontent.google.com/download?id={SOURCES[family]["id"]}&export=download&confirm=t'
    db=sqlite3.connect(dest/'index.sqlite');db.execute('CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY, offset INTEGER NOT NULL, size INTEGER NOT NULL, sha256 TEXT NOT NULL)');db.commit()
    payload=open(dest/'payload.bin','a+b');counts=Counter();started=time.perf_counter();all_names=set()
    try:
        with urllib.request.urlopen(url,timeout=120) as response:
            assert response.status==200 and 'text/html' not in response.headers.get('Content-Type',''),'Expected official binaryarchive'
            length=int(response.headers.get('Content-Length',0));reader=HashedStream(response,BASE/f'{family}_download_progress.json')
            with tarfile.open(fileobj=reader,mode='r|gz') as tar:
                for member in tar:
                    parts=Path(member.name).parts
                    if len(parts)>=2:all_names.add(parts[1])
                    value=key(member.name)
                    if not member.isfile() or value is None:continue
                    split,index,name,pk=value;raw=tar.extractfile(member).read();assert len(raw)==member.size;digest=hashlib.sha256(raw).hexdigest()
                    old=db.execute('SELECT offset,size,sha256 FROM records WHERE id=?',(pk,)).fetchone()
                    if old:
                        assert (old[1],old[2])==(len(raw),digest);payload.flush();assert os.pread(payload.fileno(),old[1],old[0])==raw
                    else:
                        payload.seek(0,os.SEEK_END);offset=payload.tell();payload.write(raw);db.execute('INSERT INTO records VALUES (?,?,?,?)',(pk,offset,len(raw),digest))
                    counts[f'{split}/{name}']+=1
                    if sum(counts.values())%25000==0:
                        payload.flush();db.commit();size=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file());assert size<30000000000
                        print('R5 retained',family,sum(counts.values()),'payloadbytes',payload.tell(),flush=True)
            while reader.read(1048576):pass
            if length:assert reader.count==length
            archive_hash=reader.digest.hexdigest();received=reader.count
        for split,n in [('train',64000),('test',8000)]:
            for name in NAMES:assert counts[f'{split}/{name}']==n
        payload.flush();db.commit();assert db.execute('SELECT count(*),min(id),max(id) FROM records').fetchone()==(432000,0,431999)
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    finally:payload.close();db.commit();db.close()
    dump(dest/'manifest.json',{'intake_protocol_sha256':sha(BASE/f'{family}_intake_protocol.json'),'archive_sha256':archive_hash,'archive_received_bytes':received,'response_content_length':length,
        'tasks_in_archive':sorted(all_names),'counts':dict(counts),'selected_task':TASK,'episodes':72000,'seconds':time.perf_counter()-started,'files':{n:sha(dest/n) for n in ['payload.bin','index.sqlite']},'source':SOURCES[family]})
    event('R5_3d_intake_complete',family=family,episodes=72000,archive_bytes=received);verify(family)

def verify(family):
    freeze(family);dest=BASE/'data'/f'{family}_hard'/TASK;m=json.loads((dest/'manifest.json').read_text());assert m['intake_protocol_sha256']==sha(BASE/f'{family}_intake_protocol.json')
    for n,h in m['files'].items():assert sha(dest/n)==h
    db=sqlite3.connect(f'file:{dest/"index.sqlite"}?mode=ro',uri=True);fd=os.open(dest/'payload.bin',os.O_RDONLY);count=0;formats={};started=time.perf_counter()
    for pk,offset,length,h in db.execute('SELECT id,offset,size,sha256 FROM records ORDER BY id'):
        assert pk==count;raw=os.pread(fd,length,offset);assert hashlib.sha256(raw).hexdigest()==h;count+=1
        if pk<6*16 or 64000*6<=pk<(64000+16)*6:
            name=NAMES[pk%6]
            if name.endswith('.png'):
                image=Image.open(io.BytesIO(raw));formats[name]={'mode':image.mode,'size':list(image.size)};image.load()
            else:assert len(json.loads(raw)['objects'])==2
    assert count==432000;os.close(fd);db.close()
    dump(dest/'verification.json',{'manifest_sha256':sha(dest/'manifest.json'),'all_original_payload_hashes_rechecked':count,'native_format_samples':formats,'seconds':time.perf_counter()-started,'source_sha256':sha(__file__),'independent_second_download':False})
    print('R5 3D intake verified',family,count,formats,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('family',choices=SOURCES);p.add_argument('--verify',action='store_true');a=p.parse_args();verify(a.family) if a.verify else prepare(a.family)
