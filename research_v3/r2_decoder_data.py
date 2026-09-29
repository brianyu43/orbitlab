"""Fixed-panel targets: keep background and source occluder unchanged after edits."""
import json
import numpy as np
import torch
import r2_decoder_core as c
import r2_world as w
from v3_common import dump,sha

def build(seed,split,count,write=False):
    folder=w.BASE/f'data/d{seed}/{split}/n{count}';records=json.loads((folder/'labels.json').read_text());images=np.load(folder/'rgb.npz')['images'][:,0]
    targets=[];panels=[];non_targets=[];foreground=[];commands=[];metadata=[]
    for i,row in enumerate(records):
        bg=w.background(row['background_seed'],row['textured_background']);amodal=w.layers(row['objects']);panel=w.blocker(amodal[row['target_id']],row['requested_occlusion'])
        source,fg,layers=c.truth_render(row['objects'],bg,panel)
        np.testing.assert_array_equal(source,images[i])
        union=1-np.prod(1-layers,axis=0)
        trans=np.ones((64,64),np.float32);visible=layers.copy()
        for j in reversed(range(count)):visible[j]*=trans;trans*=1-layers[j]
        visible*=1-panel
        non=visible[[j for j in range(count) if j!=row['target_id']]].sum(0).clip(0,1)
        for op in c.OPS:
            cmd=c.command(i,op,row['click']);objects,_=c.edit(row['objects'],cmd,row['target_id']);target,target_fg,_=c.truth_render(objects,bg,panel)
            if op=='identity':np.testing.assert_array_equal(source,target)
            # Pixels inside opaque panels are exactly the source panel for every command.
            np.testing.assert_array_equal(target[panel==1],source[panel==1])
            targets.append(target);panels.append(panel);non_targets.append(non);foreground.append(union);commands.append(cmd)
            metadata.append({'source_index':i,'op':op,'target_id':row['target_id'],'objects':objects})
    result={'targets':np.stack(targets),'panels':np.stack(panels),'non_targets':np.stack(non_targets),'foreground':np.stack(foreground),'commands':commands,'metadata':metadata,'records':records,'source':images}
    if write:
        dest=c.BASE/f'data/d{seed}/{split}/n{count}';dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'targets.npz',targets=result['targets'],panels=np.rint(result['panels']*255).astype(np.uint8),non_targets=np.rint(result['non_targets']*255).astype(np.uint8),foreground=np.rint(result['foreground']*255).astype(np.uint8))
        dump(dest/'commands.json',commands);dump(dest/'metadata.json',metadata)
        dump(dest/'manifest.json',{'seed':seed,'split':split,'count':count,'clips':len(records),'commands':len(commands),'source_manifest_sha256':sha(folder/'manifest.json'),'protocol_sha256':sha(c.BASE/'protocol.json'),'files':{n:sha(dest/n) for n in ['targets.npz','commands.json','metadata.json']},'fixed_original_panel_verified':True})
    return result

if __name__=='__main__':
    c.freeze();build(w.DEVELOPMENT,'train',2,True)
    for count in [2,3,4,6]:build(w.DEVELOPMENT,'val',count,True)
    print('R2 fixed-panel targets prepared',flush=True)
