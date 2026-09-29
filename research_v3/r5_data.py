"""Read-only random access to official PNG/JSON payloads, native pixels."""
import io,json,os,sqlite3
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from r5_intake import BASE,NAMES

class Pairs:
    def __init__(self,task):
        self.path=BASE/'data/dsprites_hard'/task
        db=sqlite3.connect(f'file:{self.path/"index.sqlite"}?mode=ro',uri=True)
        self.offsets=np.array(db.execute('SELECT id,offset,size FROM records ORDER BY id').fetchall(),np.int64)
        assert np.array_equal(self.offsets[:,0],np.arange(432000));db.close()
        self.fd=os.open(self.path/'payload.bin',os.O_RDONLY)
    def raw(self,split,index,name):
        key=(int(index)+(64000 if split=='test' else 0))*6+NAMES.index(name);_,offset,length=self.offsets[key]
        value=os.pread(self.fd,int(length),int(offset));assert len(value)==length;return value
    def image(self,split,index,role):
        return np.asarray(Image.open(io.BytesIO(self.raw(split,index,role+'.png'))).convert('RGB'))
    def batch(self,split,indices,device):
        x=np.stack([self.image(split,int(i),'source') for i in indices]);y=np.stack([self.image(split,int(i),'target') for i in indices])
        return [torch.from_numpy(v.copy()).permute(0,3,1,2).float().to(device)/255 for v in [x,y]]
    def metadata(self,split,index,role):return json.loads(self.raw(split,index,role+'.json'))
    def close(self):os.close(self.fd)
