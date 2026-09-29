"""Fixed-budget independent component training on one shared disk split."""
import argparse
import hashlib
import json
import os
import resource
import time
import numpy as np
import torch
from v3_common import HERE, ROOT, dump, lock, sha
from r4_core import state_hash
import r5_disk_data as data
import r5_disk_core as core


def freeze():
    data.freeze()
    return lock(data.BASE / 'training_protocol.json', {
        'version': 1, 'sources': {p: sha(HERE / p) for p in ['r5_disk_core.py', 'r5_disk_train.py', 'r3_core.py', 'r3_physics.py', 'r4_core.py', 'v3_common.py']},
        'dependencies': {'followup/dynamics_models.py': sha(ROOT / 'followup/dynamics_models.py')},
        'data_protocol_sha256': sha(data.BASE / 'data_protocol.json'),
        'data_verification_sha256': sha(data.BASE / 'data_verification.json'),
        'arms': list(core.ARMS), 'updates': core.STEPS, 'initialization': 0,
        'optimizer': 'AdamW .001, weight_decay .0001, global grad norm clip1, CPU float32,2threads',
        'batch': {'reader': 16, 'force': 32, 'decoder': 32, 'bounded': 32, 'contact': 32},
        'sampling': 'Component-specific reproducible torch generators; uniform960train family and time window. Reader t3..20, force t3 only (before action), decoder t0..20. Both transitions share exactepisode/t0..12 sequences and8step rollout loss. No calibration/test labels in training.',
        'reader': 'R2-derived6-peak class-agnostic heatmap with6pxNMS, four-frame33pxRGB crops. Supervised coordinate/velocity/radius/RGB andpresence loss, exhaustive2truth-to6slot assignment for training only. Fixedpresence>=.5 at inference; no GTcount/IDs/positions/colors. Radius3..7px and velocity+-6px/frame fixed support. Every output object is a disk by prior.',
        'force': 'Separate four-frame RGB CNN trained on normalized netforce+drag. Gravity and wind decomposition is not identified or scored separately. Only pre-action frames0..3; finite observation precision can make netforce/drag ambiguous.',
        'decoder': 'Train radius/RGB->17pxRGBA patch and bilinearXY placement jointly against sharedtrain sceneRGB, weights1+8*targetforeground; black background and disk supports are explicit priors. No alpha supervision or sourceRGB skip. Targetimage supplies loss only.',
        'dynamics': 'Nine features xy/31.5,velocity/3,r/8,continuousRGB,presence; allreal variableN slots. JointC4 interaction network. Bounded uses sum messages and true-train-derived1.5max velocity limit. Contact uses local mean messages plus separate coarse physical solver andbounded.15px/frame velocity residual. Samelearnedparametercount butdifferentphysics/messagepriors; no claimlearningalone causesdifference.',
        'checkpoint': 'Final fixed updates only, no test or calibration model selection.100step timing pilots discarded; full run resets same initialization/generator. Atomicprogress every250steps includesoptimizer/RNG.',
        'evaluation_plan': 'Compare both learnedtransitions and coarsephysics across true/learnedstate xknown/inferredforce xknown/neuralrenderer at1/16/61/128. Count2/3/4/6,same/differentcolor,contactstrata,reverseimpulse andforcecounterfactual. Truefuture->neuraldecoder andinputcopycontrols. No automatic integrated3x3 based solely on component gates; fixed development diagnosis first.',
        'resource_caps': 'Same globalR5 initial12h/30GB budget. Do not reduce declaredupdates to fit. Stop and disclose remaining work if exceeded.',
    })


def resource_check():
    root = data.BASE.parent
    started = json.loads((root / 'training_started.json').read_text())['unix']
    assert time.time() - started < 43200, 'R5 time cap; retain partial checkpoint'
    assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) < 30000000000, 'R5 disk cap'


def save(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def inputs():
    root = data.path('train', 2)
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(root / name) == digest
    a = np.load(root / 'trajectories.npz')
    s = core.encode(a['states'], a['colors'])
    return {'states': s, 'actions': torch.tensor(a['actions'], dtype=torch.float32) / 3,
            'contexts': core.context(a['contexts']), 'images': torch.from_numpy(np.load(root / 'images.npy')),
            'velocity_limit': float(s[..., 2:4].norm(dim=-1).max()) * 1.5}


def loss_for(model, arm, a, rng):
    b = 16 if arm == 'reader' else 32
    ep = torch.randint(len(a['states']), (b,), generator=rng)
    if arm == 'reader':
        t = torch.randint(3, 21, (b,), generator=rng)
        return core.reader_loss(model(core.rgb_window(a['images'], ep, t)), a['states'][ep, t]), (ep, t)
    if arm == 'force':
        t = torch.full((b,), 3)
        loss = (model(core.rgb_window(a['images'], ep, t)) - a['contexts'][ep]).square().mean()
        return loss, (ep, t)
    if arm == 'decoder':
        t = torch.randint(21, (b,), generator=rng)
        y = a['images'][ep, t].permute(0, 3, 1, 2).float() / 255
        weights = 1 + 8 * (y.amax(1, keepdim=True) > 0)
        loss = ((model(a['states'][ep, t]) - y).square() * weights).sum() / (3 * weights.sum())
        return loss, (ep, t)
    t = torch.randint(13, (b,), generator=rng)
    current = a['states'][ep, t]
    loss = 0
    for k in range(8):
        current = model(current, a['actions'][ep, t + k], a['contexts'][ep], a['velocity_limit'])
        loss = loss + (current[..., :4] - a['states'][ep, t + k + 1, :, :4]).square().mean() / 8
    return loss, (ep, t)


def train(arm, benchmark=False):
    freeze()
    torch.set_num_threads(2)
    folder = data.BASE / ('benchmark' if benchmark else 'runs') / arm
    folder.mkdir(parents=True, exist_ok=True)
    protocol_hash = sha(data.BASE / 'training_protocol.json')
    if (folder / 'run.json').exists():
        r = json.loads((folder / 'run.json').read_text())
        assert r['protocol_sha256'] == protocol_hash
        if not benchmark:
            assert sha(folder / 'model.pt') == r['checkpoint_sha256']
        return r
    resource_check()
    torch.manual_seed(961000)
    model = core.make(arm)
    initial = state_hash(model)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng_seed = 961100 if arm in ('bounded', 'contact') else 961101 + core.ARMS.index(arm)
    rng = torch.Generator().manual_seed(rng_seed)
    a = inputs()
    steps = 100 if benchmark else core.STEPS[arm]
    start, prior, logs, order_bytes = 0, 0., [], b''
    progress = folder / 'progress.pt'
    if progress.exists():
        ck = torch.load(progress, weights_only=True)
        assert ck['protocol_sha256'] == protocol_hash
        model.load_state_dict(ck['state_dict'])
        opt.load_state_dict(ck['optimizer'])
        rng.set_state(ck['rng'])
        start, prior, logs, order_bytes = ck['step'], ck['seconds'], ck['logs'], ck['sample_order_bytes']
    begin = time.perf_counter()
    for step in range(start + 1, steps + 1):
        loss, order = loss_for(model, arm, a, rng)
        order_bytes += torch.stack(order, 1).numpy().astype('<i8').tobytes()
        if not torch.isfinite(loss):
            dump(folder / 'failure.json', {'step': step, 'reason': 'nonfinite loss'})
            raise FloatingPointError('nonfinite loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        assert torch.isfinite(norm)
        opt.step()
        if step % 250 == 0 or step == steps:
            seconds = prior + time.perf_counter() - begin
            logs.append({'step': step, 'loss': float(loss.detach()), 'seconds': seconds})
            save(progress, {'state_dict': model.state_dict(), 'optimizer': opt.state_dict(), 'rng': rng.get_state(), 'step': step, 'seconds': seconds,
                            'logs': logs, 'sample_order_bytes': order_bytes, 'protocol_sha256': protocol_hash})
            dump(folder / 'progress.json', {'pid': os.getpid(), 'arm': arm, 'step': step, 'total': steps, 'seconds': seconds})
            print('disk train', arm, logs[-1], flush=True)
            resource_check()
    result = {'protocol_sha256': protocol_hash, 'train_manifest_sha256': sha(data.path('train', 2) / 'manifest.json'),
              'arm': arm, 'benchmark': benchmark, 'steps': steps, 'seconds': prior + time.perf_counter() - begin, 'logs': logs,
              'initial_state_sha256': initial, 'sample_order_sha256': hashlib.sha256(order_bytes).hexdigest(), 'rng_seed': rng_seed,
              'parameters': sum(p.numel() for p in model.parameters()), 'velocity_limit': a['velocity_limit'],
              'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if benchmark:
        result['estimated_full_seconds'] = result['seconds'] * core.STEPS[arm] / 100
    else:
        assert (data.BASE / 'benchmark' / arm / 'run.json').exists()
        save(folder / 'model.pt', {'state_dict': model.state_dict(), 'arm': arm, 'velocity_limit': a['velocity_limit'], 'protocol_sha256': protocol_hash})
        result['checkpoint_sha256'] = sha(folder / 'model.pt')
    dump(folder / 'run.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=core.ARMS, required=True)
    p.add_argument('--benchmark', action='store_true')
    args = p.parse_args()
    train(args.arm, args.benchmark)
