"""Same-family3Dconfirmation resplits; keep officialtest and orbit groups fixed."""
import json
import numpy as np
from v3_common import sha,dump
from r5_intake import BASE
from r5_partitions import DEVELOPMENT,CONFIRMATION
import r5_audit

def prepare(family,task,seed):
    assert family in ('clevr','clevrtex') and task=='Single_Atomic' and seed in (DEVELOPMENT,*CONFIRMATION)
    audit=BASE/'audits'/f'{family}_hard'/task
    if seed==DEVELOPMENT:return audit/'partition.npz'
    folder=BASE/'partitions'/f'{family}_hard'/task/f's{seed}';dest=folder/'partition.npz'
    if (folder/'receipt.json').exists():
        d=json.loads((folder/'receipt.json').read_text());assert d['partition_sha256']==sha(dest) and d['factors_sha256']==sha(audit/'factors.npz');return dest
    hashes=np.load(audit/'factors.npz')['orbit_hashes'];old=r5_audit.SEED
    try:r5_audit.SEED=seed;train,val,stats=r5_audit.make_split(hashes)
    finally:r5_audit.SEED=old
    folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,train=train,val=val,test=np.arange(8000,dtype=np.int64))
    dump(folder/'receipt.json',{'family':family,'seed':seed,'source_sha256':sha(__file__),'factors_sha256':sha(audit/'factors.npz'),'partition_sha256':sha(dest),'audit':stats,
        'scope':'Train/validation resplits of the SAME official dataset; overlapping training examples and same officialtest. Not fresh independent generated data.'})
    return dest
