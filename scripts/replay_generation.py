"""Relocate a saved flow checkpoint without changing the source artifact."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--flow', required=True)
    p.add_argument('--ae', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    p.add_argument('--n', type=int, default=576)
    p.add_argument('--seed', type=int, default=53000)
    a = p.parse_args()
    ck = torch.load(a.flow, map_location='cpu', weights_only=True)
    if digest(a.ae) != ck['ae_sha256']:
        raise ValueError('AE content differs from the frozen checkpoint used by this flow')
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_hash = digest(a.flow)
    ck['ae_path'] = str(Path(a.ae).resolve())
    torch.save(ck, out / 'relocated_flow.pt')
    (out / 'relocation.json').write_text(json.dumps({
        'source_flow_sha256': source_hash, 'source_flow': str(Path(a.flow).resolve()),
        'ae_sha256': ck['ae_sha256'], 'relocated_flow_sha256': digest(out / 'relocated_flow.pt'),
        'change': 'Only ae_path metadata is relocated; state_dict and normalization unchanged.'
    }, indent=2))
    subprocess.run([sys.executable, str(ROOT / 'work/generation.py'), 'sample',
                    '--flow', str(out / 'relocated_flow.pt'), '--device', a.device,
                    '--n', str(a.n), '--seed', str(a.seed), '--out', str(out / 'samples')],
                   cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
