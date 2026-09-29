"""Supplementary wall/overlap diagnostics on saved autonomous predictions."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'generation_recovery')]
import numpy as np
from p0 import dump,sha

def main():
    count=0
    for path in sorted((BASE/'dynamics/evaluation').glob('d*/*/*/*/summary.json')):
        folder=path.parent;dest=folder/'geometry.json'
        if dest.exists():count+=1;continue
        a=np.load(folder/'predictions.npz');p=a['predictions'];initial=a['initial'];present=initial[:,:,4]>0
        limit=31.5-initial[:,:,4];xy=p[:,:,:,:2]
        wall=np.maximum(np.abs(xy)-limit[:,None,:,None],0)
        wall=np.where(present[:,None,:,None],wall,0).max((2,3))
        overlap=np.zeros(p.shape[:2])
        for i in range(4):
            for j in range(i+1,4):
                distance=np.linalg.norm(xy[:,:,i]-xy[:,:,j],axis=-1)
                pen=np.maximum(initial[:,None,i,4]+initial[:,None,j,4]-distance,0)
                overlap=np.maximum(overlap,np.where((present[:,i]&present[:,j])[:,None],pen,0))
        summary={}
        for h in [1,16,61]:
            finite=np.isfinite(p[:,:h]).all((1,2,3));w=wall[:,:h].max(1);v=overlap[:,:h].max(1)
            summary[str(h)]={'finite_episodes':int(finite.sum()),'nonfinite_episodes':int((~finite).sum()),
                             'wall_penetration_over_1px_rate_finite':float((w[finite]>1).mean()) if finite.any() else None,
                             'overlap_over_1px_rate_finite':float((v[finite]>1).mean()) if finite.any() else None,
                             'max_wall_penetration_finite':float(w[finite].max()) if finite.any() else None,
                             'max_overlap_finite':float(v[finite].max()) if finite.any() else None}
        dump(dest,{'summary':summary,'predictions_sha256':sha(folder/'predictions.npz'),'source_sha256':sha(Path(__file__)),
                   'scope':'Max violation up to horizon using input-estimated radii; nonfinite episodes explicitly separated. Projection is not evidence of learned physical laws.'})
        count+=1
    print('geometry units',count,flush=True)

if __name__=='__main__':main()
