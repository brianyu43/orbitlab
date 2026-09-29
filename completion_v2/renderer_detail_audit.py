"""Distinguish ideal shape symmetry, raster asymmetry and preprocessing loss."""
from pathlib import Path
import json
import numpy as np
import perception_detail as p
import object_world as w
from p0 import dump,sha
rows=[]
for radius in range(5,9):
    for color in range(6):
        images=[w.render([w.ObjectState(0,3,color,30,30,radius,pose)])['image'] for pose in [0,2]]
        a,b=[w.local_alpha(3,radius,pose) for pose in [0,2]]
        row={'radius':radius,'color':color,'raw_alpha_max_difference':float(np.abs(a-b).max()),'raw_rgb_different_values':int((images[0]!=images[1]).sum())}
        for kind in ['binary','alpha']:
            x,y=[p.inputs(im,kind)[0] for im in images]
            row[kind+'_crop_different_values']=int((x!=y).sum());row[kind+'_crop_max_difference']=float((x-y).abs().max())
        rows.append(row)
assert all(r['raw_rgb_different_values']>0 for r in rows)
dump(p.HERE/'renderer_detail_audit.json',{'rows':rows,'conclusion':'Geometric half-turn symmetry is not exact raster symmetry. Binary preprocessing can erase pose-dependent raster detail; full RGB has differing values in these examples.','source_sha256':sha(Path(__file__))})
