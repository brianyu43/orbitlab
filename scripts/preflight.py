"""Independent device/math preflight; does not validate the missing starter."""
import copy
import datetime
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import PIL
import torch
from torch import nn


def main():
    torch.manual_seed(20260923)
    torch.set_num_threads(4)
    report = {
        'scope': 'Independent CPU/MPS and C4 algebra diagnostics; NOT original OrbitLab model validation',
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'platform': platform.platform(), 'machine': platform.machine(),
        'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
        'pillow': PIL.__version__,
        'mps_built': torch.backends.mps.is_built(),
        'mps_available': torch.backends.mps.is_available(),
        'mps_fallback_env': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
        'tests': {},
    }
    tests = report['tests']
    x = torch.randn(4, 3, 32, 32)
    z = torch.randn(4, 4, 16)
    rot = lambda a, k: torch.rot90(a, k, (-2, -1))
    rho = lambda a, k: torch.roll(a, k, dims=1)
    tests['c4_image_composition'] = all(torch.equal(rot(rot(x, a), b), rot(x, (a+b)%4)) for a in range(4) for b in range(4))
    tests['c4_slot_composition'] = all(torch.equal(rho(rho(z, a), b), rho(z, (a+b)%4)) for a in range(4) for b in range(4))
    f = torch.fft.fft(z, dim=1, norm='ortho')
    tests['fft_roundtrip_max_abs'] = (torch.fft.ifft(f, dim=1, norm='ortho').real-z).abs().max().item()
    tests['fft_conjugacy_max_abs'] = (f[:, 1]-f[:, 3].conj()).abs().max().item()
    tests['fft_shift_theorem_max_abs'] = max((torch.fft.fft(rho(z,k),dim=1,norm='ortho')-f*torch.exp(-2j*torch.pi*torch.arange(4)[None,:,None]*k/4)).abs().max().item() for k in range(4))
    cpu = nn.Sequential(nn.Conv2d(3,8,3,padding=1), nn.GELU(), nn.Conv2d(8,3,3,padding=1))
    target = torch.randn_like(x)
    y = cpu(x)
    loss = (y-target).square().mean()
    loss.backward()
    tests['cpu_loss_finite'] = bool(torch.isfinite(loss))
    tests['cpu_gradients_finite'] = all(bool(torch.isfinite(p.grad).all()) for p in cpu.parameters())
    if report['mps_available']:
        try:
            mps = copy.deepcopy(cpu).to('mps')
            mps.zero_grad(set_to_none=True)
            xm, tm = x.to('mps'), target.to('mps')
            ym = mps(xm)
            lm = (ym-tm).square().mean()
            lm.backward()
            torch.mps.synchronize()
            tests['mps_forward_max_abs'] = (ym.cpu()-y).abs().max().item()
            tests['mps_loss_abs_diff'] = abs(lm.item()-loss.item())
            tests['mps_gradient_max_abs'] = max((a.grad-b.grad.cpu()).abs().max().item() for a,b in zip(cpu.parameters(),mps.parameters()))
            tests['mps_gradients_finite'] = all(bool(torch.isfinite(p.grad).all()) for p in mps.parameters())
            tests['mps_rot90_exact'] = all(torch.equal(rot(xm,k).cpu(),rot(x,k)) for k in range(4))
            tests['mps_roll_exact'] = all(torch.equal(rho(z.to('mps'),k).cpu(),rho(z,k)) for k in range(4))
            # Warm up before a small timing probe; this is not an AE benchmark.
            optim = torch.optim.Adam(mps.parameters(),lr=1e-3)
            for _ in range(3):
                optim.zero_grad(set_to_none=True)
                (mps(xm)-tm).square().mean().backward()
                optim.step()
            torch.mps.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                optim.zero_grad(set_to_none=True)
                (mps(xm)-tm).square().mean().backward()
                optim.step()
            torch.mps.synchronize()
            report['diagnostic_cnn_10_steps_seconds'] = time.perf_counter()-start
            report['mps_current_allocated_bytes'] = torch.mps.current_allocated_memory()
            try:
                fm = torch.fft.fft(z.to('mps'),dim=1,norm='ortho')
                torch.mps.synchronize()
                tests['mps_fft_supported'] = True
                tests['mps_fft_max_abs'] = (fm.cpu()-f).abs().max().item()
            except (RuntimeError, NotImplementedError, TypeError) as e:
                tests['mps_fft_supported'] = False
                report['mps_fft_note'] = str(e)
        except Exception as e:
            report['mps_error'] = repr(e)
    report['independent_cpu_checks_pass'] = bool(tests['c4_image_composition'] and tests['c4_slot_composition'] and tests['fft_roundtrip_max_abs']<1e-5 and tests['fft_shift_theorem_max_abs']<1e-5 and tests['cpu_loss_finite'] and tests['cpu_gradients_finite'])
    report['independent_mps_checks_pass'] = bool(report['mps_available'] and 'mps_error' not in report and tests.get('mps_forward_max_abs',1)<1e-4 and tests.get('mps_gradient_max_abs',1)<1e-4 and tests.get('mps_rot90_exact') and tests.get('mps_roll_exact') and tests.get('mps_gradients_finite'))
    report['original_starter_validated'] = False
    out = Path(sys.argv[1] if len(sys.argv)>1 else 'reports/preflight.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0 if report['independent_cpu_checks_pass'] and report['independent_mps_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
