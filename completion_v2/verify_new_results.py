"""Recompute saved metrics and replay inference for completed experiment units."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE),str(ROOT/'generation_recovery'),str(ROOT/'work'),str(ROOT/'followup')]
import numpy as np
import torch
from p0 import dump
from svib_prepare import sha

def perception():
    import perception as p
    from perception_evaluate import summarize
    import object_image_edit as edit
    from object_oracle_study import image_from_slots
    count=0
    for path in sorted((BASE/'perception/evaluation').glob('d*/*/*/summary.json')):
        out=path.parent;marker=out/'verification.json'
        if marker.exists():count+=1;continue
        record=json.loads(path.read_text());run=BASE/f"perception/runs/{record['kind']}_s{record['initialization']}"
        assert sha(run/'model.pt')==record['model_sha256']
        for name,digest in record['files'].items():assert sha(out/name)==digest
        ck=torch.load(run/'model.pt',weights_only=True);model=p.Reader();model.load_state_dict(ck['state_dict']);model.eval()
        data=BASE/f"perception/data/d{record['data_seed']}/{record['split']}"
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
    print('verified perception units',count,flush=True)

def dynamics():
    import dynamics_evaluate as e
    from dynamics_models import Transition
    count=0
    for path in sorted((BASE/'dynamics/evaluation').glob('d*/*/*/*/summary.json')):
        out=path.parent;marker=out/'verification.json'
        if marker.exists():count+=1;continue
        record=json.loads(path.read_text());data=e.study.data_root(record['data_seed'])/record['mode']/record['split']/'trajectories.npz'
        assert sha(data)==record['data_sha256'] and sha(out/'predictions.npz')==record['predictions_sha256']
        arrays=dict(np.load(out/'predictions.npz'));truth=dict(np.load(data));variant=record['variant']
        kind={'one_bounded':'one','multi_projected':'multi'}.get(variant,variant);model=None;limit=None
        if kind!='force_wall':
            cp=BASE/f"dynamics/runs/d{record['data_seed']}_s{record['init']}/{kind}/model.pt"
            assert sha(cp)==record['checkpoint_sha256'];ck=torch.load(cp,weights_only=True)
            model=Transition('joint_exact');model.load_state_dict(ck['state_dict']);model.eval()
            if variant in ['one_bounded','multi_projected','multi_bounded']:limit=ck['velocity_limit']
        pred=e.rollout(arrays['initial'],arrays['context'],arrays['actions'],model,limit,kind=='force_wall')
        np.testing.assert_allclose(pred,arrays['predictions'],rtol=1e-6,atol=1e-5,equal_nan=True)
        cf=arrays.get('counterfactual')
        rows,summary=e.score(arrays['predictions'],truth['states'][:,3:],arrays['colors'],truth['target_ids'],cf,
                             truth['counterfactual_states'][:,3:] if cf is not None else None)
        assert summary==record['summary']
        dump(marker,{'all_passed':True,'replayed_episodes':len(pred),'steps':61,'metrics_recomputed':True,
                     'summary_sha256':sha(path),'verifier_sha256':sha(Path(__file__))})
        count+=1
    print('verified dynamics units',count,flush=True)

def generation():
    from models import from_variant
    from decoder_study import ControlledDecoder
    from evaluator_v2 import DetailedEvaluator
    from p1_evaluate import extract_features,integrate
    from generation_evaluate import metrics
    ev=DetailedEvaluator();count=0
    for path in sorted((BASE/'generation/evaluation').glob('d*/*/result.json')):
        out=path.parent;marker=out/'verification.json'
        if marker.exists():count+=1;continue
        record=json.loads(path.read_text());sample=torch.load(out/'samples.pt',weights_only=True)
        assert sha(out/'samples.pt')==record['samples_sha256'] and sha(out/'rows.json')==record['rows_sha256']
        rows=extract_features(sample['images'],sample['labels'],ev);assert metrics(rows)==record['metrics']
        run=BASE/f"generation/runs/d{record['seed']}_s{record['init']}";fk,dk=record['variant'].split('/')
        fp=run/fk/'checkpoint.pt';dp=run/('decoder_logit_mean/decoder.pt' if dk=='standard_decoder' else 'area_edge_decoder/checkpoint.pt')
        assert sha(fp)==record['flow_sha256'] and sha(dp)==record['decoder_sha256']
        fc=torch.load(fp,weights_only=True);dc=torch.load(dp,weights_only=True)
        flow=from_variant('F0');flow.load_state_dict(fc['state_dict']);flow.eval()
        decoder=ControlledDecoder('logit_mean');decoder.load_state_dict(dc['state_dict']);decoder.eval()
        with torch.no_grad():
            latent=integrate(flow,sample['noise'][:128],sample['labels'][:128],1.)
            torch.testing.assert_close(latent,sample['latent'][:128],rtol=1e-5,atol=1e-5)
            images=decoder(((latent[:64]*fc['std']+fc['mean'])-dc['mean'])/dc['std'])
            torch.testing.assert_close(images,sample['images'][:64],rtol=1e-5,atol=1e-5)
        dump(marker,{'all_passed':True,'scored_images':len(rows),'replayed_latents':128,'replayed_images':64,
                     'summary_sha256':sha(path),'verifier_sha256':sha(Path(__file__))})
        count+=1
    print('verified generation units',count,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('branch',choices=['perception','dynamics','generation']);a=p.parse_args()
    torch.set_num_threads(2);globals()[a.branch]()
