"""Matched, resumable CPU training with a fixed 100-update timing probe."""
import argparse, json, os, resource, time
import torch
import r1_core as c
from r1_data import training_data
from v3_common import sha, dump, event

def train(seed, init, arm, benchmark=False):
    cfg = c.freeze(); kind, pose_loss = arm.split('_'); torch.set_num_threads(2)
    folder = c.BASE / ('benchmark' if benchmark else 'runs') / f'd{seed}_s{init}' / arm
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'run.json').exists():
        report = json.loads((folder/'run.json').read_text())
        if not benchmark: assert report['checkpoint_sha256'] == sha(folder/'model.pt')
        return report
    steps = 100 if benchmark else cfg['training_steps']
    torch.manual_seed(886000+init); model = c.Reader()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = torch.Generator().manual_seed(886100+init)
    x, y = training_data(seed, kind)
    start_step, prior, logs = 0, 0., []
    progress = folder / 'progress.pt'
    if progress.exists():
        ck = torch.load(progress, weights_only=True)
        assert ck['protocol_sha256'] == sha(c.BASE/'protocol.json')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); rng.set_state(ck['rng'])
        start_step, prior, logs = ck['step'], ck['seconds'], ck['logs']
    started = time.perf_counter()
    for step in range(start_step+1, steps+1):
        idx = torch.randint(len(x), (cfg['batch'],), generator=rng)
        loss = c.loss(model(x[idx]), y[idx], pose_loss)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        assert torch.isfinite(norm); opt.step()
        if step % 500 == 0 or step == steps:
            seconds = prior + time.perf_counter()-started
            logs.append({'step': step, 'loss': float(loss.detach()), 'seconds': seconds})
            tmp = progress.with_suffix('.tmp')
            torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'rng': rng.get_state(), 'step': step, 'seconds': seconds, 'logs': logs, 'protocol_sha256': sha(c.BASE/'protocol.json')}, tmp)
            tmp.replace(progress)
            dump(folder/'progress.json', {'pid': os.getpid(), 'step': step, 'total': steps, 'seconds': seconds})
            print('R1 train', seed, init, arm, logs[-1], flush=True)
    result = {'steps': steps, 'seconds': prior+time.perf_counter()-started, 'parameters': sum(p.numel() for p in model.parameters()), 'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'device': 'cpu', 'threads': 2, 'logs': logs, 'protocol_sha256': sha(c.BASE/'protocol.json'), 'train_sha256': sha(c.BASE/f'data/d{seed}/train/pil4_lanczos.npz')}
    if not benchmark:
        torch.save({'state_dict': model.state_dict(), 'seed': seed, 'initialization': init, 'arm': arm}, folder/'model.pt')
        result['checkpoint_sha256'] = sha(folder/'model.pt')
    else:
        result['estimated_seconds_per_6000_step_run'] = result['seconds']*60
        result['note'] = 'Timing probe only; discarded weights. Excludes data construction and evaluation.'
    dump(folder/'run.json', result); event('R1_train_complete', seed=seed, init=init, arm=arm, benchmark=benchmark)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, default=c.DEVELOPMENT); parser.add_argument('--init', type=int, default=0)
    parser.add_argument('--arm', choices=c.ARMS, required=True); parser.add_argument('--benchmark', action='store_true')
    a = parser.parse_args(); train(a.seed, a.init, a.arm, a.benchmark)
