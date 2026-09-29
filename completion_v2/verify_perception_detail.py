"""Recompute saved metrics and replay inference for completed experiment units."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE),str(ROOT/'generation_recovery'),str(ROOT/'work'),str(ROOT/'followup')]
import numpy as np
import torch
from p0 import dump
from svib_prepare import sha

def perception_detail():
    import perception_detail as p
    from perception_detail_evaluate import summarize
    import object_image_edit as edit
    from object_oracle_study import image_from_slots
    count=0
    for path in sorted((BASE/'perception_detail/evaluation').glob('d*/*/*/summary.json')):
        out=path.parent;marker=out/'verification.json'
        if marker.exists():count+=1;continue
        record=json.loads(path.read_text());run=BASE/f"perception_detail/runs/{record['kind']}_s{record['initialization']}"
        assert sha(run/'model.pt')==record['model_sha256']
        for name,digest in record['files'].items():assert sha(out/name)==digest
        ck=torch.load(run/'model.pt',weights_only=True);model=p.Reader();model.load_state_dict(ck['state_dict']);model.eval()
        data=BASE/f"perception_detail/data/d{record['data_seed']}/{record['split']}"
        images=np.load(data/'rgb.npz')['images'];saved=torch.load(out/'inference.pt',weights_only=True)
        replay=p.infer(images[:8],model,record['kind']);torch.testing.assert_close(replay,saved['states'][:8],rtol=0,atol=1e-6)
        queries=np.load(data/'queries.npz');idx=queries['source_indices'];source=saved['states'][idx]
        commands=torch.from_numpy(queries['commands'].copy());clicks=torch.from_numpy(queries['click_xy'].copy())
        selected=edit.select_target(source,clicks);assert torch.equal(selected,saved['selected'])
        for name,controller in [('analytic','analytic'),('learned',edit.load_controller(record['initialization']))]:
            prediction=edit.apply_commands(source,selected,commands,controller)
            expected=torch.load(out/f'{name}_states.pt',weights_only=True);torch.testing.assert_close(prediction,expected,rtol=0,atol=1e-6)
            rows=json.loads((out/f'{name}_rows.json').read_text());assert summarize(rows)==record['metrics'][name]
            raster=np.load(out/f'{name}_images.npz')['images']
            for j in range(8):assert np.array_equal(image_from_slots(prediction[j]),raster[j])
        dump(marker,{'all_passed':True,'rgb_replay_scenes':8,'command_replay_tasks':len(idx),'metric_recomputed':True,
                     'verifier_sha256':sha(Path(__file__)),'summary_sha256':sha(path)})
        count+=1
    print('verified perception_detail units',count,flush=True)

if __name__=="__main__":
    torch.set_num_threads(2);perception_detail()
