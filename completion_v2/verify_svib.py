"""Independent saved-row aggregation and CPU checkpoint replay on fixed examples."""
from pathlib import Path
import json
import numpy as np
import torch
import svib_run as s
from svib_prepare import sha

torch.set_num_threads(2)
for path in sorted((s.BASE/'runs').glob('*/evaluation.json')):
    out=path.parent
    if (out/'verification.json').exists():continue
    r=json.loads(path.read_text());v=np.load(out/'test_rows.npy');assert v.shape==(8000,6) and np.isfinite(v).all()
    assert sha(out/'test_rows.npy')==r['rows_sha256'] and sha(out/'model.pt')==r['checkpoint_sha256']
    assert sha(out/'replay_samples.pt')==r['replay_sha256']
    changed=v[:,5]>0
    calc={'mse':float(v[:,0].mean()),'identity_mse':float(v[:,4].mean()),'changed_pixel_mse_on_changed_scenes':float(v[changed,1].mean()),'preserved_pixel_mse':float(v[:,2].mean()),'foreground_mse':float(v[:,3].mean()),'changed_scenes':int(changed.sum()),'unchanged_scene_mse':float(v[~changed,0].mean())}
    assert calc==r['metrics']
    ck=torch.load(out/'model.pt',map_location='cpu',weights_only=True);m=s.PreviewPredictor(r['kind']);m.load_state_dict(ck['state_dict']);m.eval()
    samples=torch.load(out/'replay_samples.pt',weights_only=True);err=0.;start=0;eq=0.
    with torch.no_grad():
        for q in samples:
            pred=m(q['source']);err=max(err,float((pred-q['prediction']).abs().max()))
            torch.testing.assert_close(pred,q['prediction'],rtol=2e-4,atol=2e-5)
            metrics=s.row_metrics(q['prediction'],q['source'],q['target']).numpy()
            np.testing.assert_allclose(metrics,v[start:start+len(pred)],rtol=2e-5,atol=1e-6)
            for k in [1,2,3]:eq=max(eq,float((m(torch.rot90(q['source'][:2],k,(-2,-1)))-torch.rot90(pred[:2],k,(-2,-1))).abs().max()))
            start+=len(pred)
    if r['kind']=='c4':assert eq<2e-5
    s.write(out/'verification.json',{'all_passed':True,'metric_rows':8000,'CPU_replay_images':start,'max_cpu_mps_prediction_difference':err,'C4_max_difference':eq,'summary_sha256':sha(path),'verifier_sha256':sha(Path(__file__))})
    print('verified SVIB',out.name,err,eq,flush=True)
