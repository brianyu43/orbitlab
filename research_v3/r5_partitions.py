"""Train-only confirmation partitions; never manufacture new official datasets."""
import json
import numpy as np
from v3_common import sha,dump
from r5_intake import BASE
from r5_audit import make_split
import r5_audit

DEVELOPMENT=890101
CONFIRMATION=(890201,890202,890203)

def path(task,seed=DEVELOPMENT):
    assert seed in (DEVELOPMENT,*CONFIRMATION)
    audit=BASE/'audits/dsprites_hard'/task
    if seed==DEVELOPMENT:return audit/'partition.npz'
    folder=BASE/'partitions/dsprites_hard'/task/f's{seed}'
    return folder/'partition.npz'

def prepare(task,seed):
    dest=path(task,seed);audit=BASE/'audits/dsprites_hard'/task
    if seed==DEVELOPMENT:return dest
    if (dest.parent/'receipt.json').exists():
        d=json.loads((dest.parent/'receipt.json').read_text())
        assert d['partition_sha256']==sha(dest) and d['factors_sha256']==sha(audit/'factors.npz')
        return dest
    hashes=np.load(audit/'factors.npz')['orbit_hashes']
    old=r5_audit.SEED
    try:
        r5_audit.SEED=seed;train,val,stats=make_split(hashes)
    finally:r5_audit.SEED=old
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,train=train,val=val,test=np.arange(8000,dtype=np.int64))
    dump(dest.parent/'receipt.json',{'seed':seed,'source_sha256':sha(__file__),'factors_sha256':sha(audit/'factors.npz'),
        'partition_sha256':sha(dest),'audit':stats,'scope':'Different train/validation partitions of the SAME official dataset; overlapping training sets and the same official test. Not independently generated datasets.'})
    return dest
