"""Replay every stored counterfactual using the saved input and opposite action."""
from pathlib import Path
import json
import numpy as np
import torch
import dynamics_evaluate as e
from dynamics_models import Transition
from p0 import dump,sha
B=Path(__file__).resolve().parent
torch.set_num_threads(2);n=0
for path in sorted((B/'dynamics/evaluation').glob('d*/*/*/*/summary.json')):
    r=json.loads(path.read_text());out=path.parent;dest=out/'counterfactual_verification.json'
    if dest.exists():n+=1;continue
    a=dict(np.load(out/'predictions.npz'))
    if 'counterfactual' not in a:continue
    variant=r['variant'];kind={'one_bounded':'one','multi_projected':'multi'}.get(variant,variant);model=None;limit=None
    if kind!='force_wall':
        cp=B/f"dynamics/runs/d{r['data_seed']}_s{r['init']}/{kind}/model.pt";assert sha(cp)==r['checkpoint_sha256']
        ck=torch.load(cp,weights_only=True);model=Transition('joint_exact');model.load_state_dict(ck['state_dict']);model.eval()
        if variant in ['one_bounded','multi_projected','multi_bounded']:limit=ck['velocity_limit']
    pred=e.rollout(a['initial'],a['context'],-a['actions'],model,limit,kind=='force_wall')
    np.testing.assert_allclose(pred,a['counterfactual'],rtol=1e-6,atol=1e-5,equal_nan=True)
    dump(dest,{'all_passed':True,'episodes':len(pred),'steps':61,'source_sha256':sha(Path(__file__)),'summary_sha256':sha(path)})
    n+=1
print('verified counterfactual units',n,flush=True)
