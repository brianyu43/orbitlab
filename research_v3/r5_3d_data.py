"""Official native 3D RGB pairs with the same read-only byte loader as dSprites."""
import os,sqlite3
import numpy as np
from r5_data import Pairs
from r5_intake import BASE

class Pairs3D(Pairs):
    def __init__(self,family,task='Single_Atomic'):
        assert family in ('clevr','clevrtex') and task=='Single_Atomic'
        self.path=BASE/'data'/f'{family}_hard'/task
        db=sqlite3.connect(f'file:{self.path/"index.sqlite"}?mode=ro',uri=True)
        self.offsets=np.array(db.execute('SELECT id,offset,size FROM records ORDER BY id').fetchall(),np.int64)
        assert np.array_equal(self.offsets[:,0],np.arange(432000));db.close()
        self.fd=os.open(self.path/'payload.bin',os.O_RDONLY)
