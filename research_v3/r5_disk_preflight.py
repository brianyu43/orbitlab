"""Nontrivial equivariance/gradient and shared-data checks before full training."""
import copy
import json
import numpy as np
import torch
from v3_common import dump, lock, sha
import r5_disk_core as c
import r5_disk_data as d
import r5_disk_train as tr
from dynamics_models import rotate_state, rotate_vec, rotate_context


def main():
    torch.set_num_threads(2)
    tr.freeze()
    a = tr.inputs()
    records = {}
    for arm in c.ARMS:
        torch.manual_seed(961000)
        m = c.make(arm)
        rng = torch.Generator().manual_seed(120)
        loss, _ = tr.loss_for(m, arm, a, rng)
        loss.backward()
        assert torch.isfinite(loss)
        gradients = [v.grad for v in m.parameters() if v.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
        records[arm] = {'finite_nonzero_backward': True, 'initial_loss': float(loss.detach()), 'parameters': sum(v.numel() for v in m.parameters())}
        if arm in ('bounded', 'contact'):
            # Nonzero corrections avoid a vacuous zero-initialization C4 test.
            with torch.no_grad():
                torch.nn.init.normal_(m.net[-1].weight, std=.01)
                torch.nn.init.normal_(m.net[-1].bias, std=.01)
            s = a['states'][:8, 3]
            act = a['actions'][:8, 3]
            ctx = a['contexts'][:8]
            out = m(s, act, ctx, a['velocity_limit'])
            errors = []
            for k in range(4):
                y = m(rotate_state(s, k), rotate_vec(act, k), rotate_context(ctx, k), a['velocity_limit'])
                err = float((y - rotate_state(out, k)).abs().max().detach())
                assert err <= 1e-5, (arm, err)
                errors.append(err)
            records[arm]['nonzero_correction_C4_max_abs_errors'] = errors
            p = torch.tensor([1, 0])
            err = float((m(s[:, p], act[:, p], ctx, a['velocity_limit']) - out[:, p]).abs().max().detach())
            assert err <= 1e-5
            records[arm]['two_slot_permutation_max_abs_error'] = err
    # No incorrect memory alias when encoding per-episode RGB across time.
    original = np.load(d.path('train', 2) / 'trajectories.npz')
    assert torch.equal(a['states'][:, 0, :, 5:8], a['states'][:, 20, :, 5:8])
    np.testing.assert_allclose(a['states'][:, 0, :, 5:8], original['colors'], rtol=0, atol=3e-8)
    p = d.BASE / 'preflight_protocol.json'
    lock(p, {'source_sha256': sha(__file__), 'training_protocol_sha256': sha(d.BASE / 'training_protocol.json'), 'checks': 'All5forward/backward, nonzeroC4andtwo-slotpermutation ofbothtransitions, sharedcoloralignment; CPUonly.100trainingsteps percomponent benchmarked separately and reset.'})
    dump(d.BASE / 'preflight.json', {'protocol_sha256': sha(p), 'checks': records, 'shared_color_alignment': True})
    for arm in c.ARMS:
        tr.train(arm, benchmark=True)
    print('disk component preflight and five100step pilots passed', flush=True)


if __name__ == '__main__':
    main()
