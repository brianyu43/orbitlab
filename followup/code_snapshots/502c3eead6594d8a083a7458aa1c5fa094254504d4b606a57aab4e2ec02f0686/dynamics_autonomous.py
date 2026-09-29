"""Autonomous prediction accepts initial estimates and commands, never labels."""
import numpy as np
import torch
from dynamics_models import encode_states
from dynamics_physical_baselines import predict as physical_predict

HORIZONS = [1, 4, 8, 16, 32, 61]
LEARNED = ['flat', 'object', 'interaction', 'augmented', 'wrong_exact', 'joint_exact', 'relaxed']
PREDICTORS = LEARNED + ['inertial', 'force_wall']
METHODS = ['oracle', 'rgb_cnn', 'measurement_mlp', 'rgb_analytic']


@torch.no_grad()
def autonomous(initial, context, action, kind, model=None, steps=61):
    """Keep radius/color/presence and context fixed; absorb nonfinite failures.

    Motion is stored in physical units. Learned recurrence keeps float32
    normalized outputs in state, avoiding a decode/encode round trip.
    No wall/pair projection is applied to learned trajectories.
    """
    initial = np.asarray(initial, dtype=np.float64)
    context = np.asarray(context, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    n = len(initial)
    if initial.shape != (n, 4, 7) or context.shape != (n, 3) or action.shape != (n, 4, 2):
        raise ValueError('Bad initial input shape')
    if not all(np.isfinite(v).all() for v in [initial, context, action]):
        raise ValueError('Initial inputs must be finite')
    motion = np.full((n, steps + 1, 4, 4), np.nan, dtype=np.float64)
    motion[:, 0] = initial[..., :4]
    finite = np.ones((n, steps + 1), dtype=bool)
    first_nonfinite = np.full(n, -1, dtype=np.int16)
    alive = np.ones(n, dtype=bool)
    if kind in LEARNED:
        state = encode_states(initial)
        ctx = torch.as_tensor(context, dtype=torch.float32) / torch.tensor([.1, .1, .01])
        impulse = torch.as_tensor(action, dtype=torch.float32) / 3
        zero = torch.zeros_like(impulse)
    else:
        state = initial.copy()
        ctx = np.column_stack([context[:, :2], np.zeros((n, 2)), context[:, 2]])
    for t in range(steps):
        if kind in LEARNED:
            prediction = model(state, impulse if t == 0 else zero, ctx)
            current = (prediction * prediction.new_tensor([31.5, 31.5, 3, 3])).numpy().astype(np.float64)
            valid = np.isfinite(current).all(axis=(1, 2)) & alive
            state = torch.cat([prediction, state[..., 4:]], dim=-1)
            # This affects only failed episodes. Networks do not mix batch rows.
            state[~torch.from_numpy(valid)] = 0
        else:
            with np.errstate(over='ignore', invalid='ignore'):
                current = physical_predict(state, ctx, action if t == 0 else np.zeros_like(action), kind)
            valid = np.isfinite(current).all(axis=(1, 2)) & alive
            state[..., :4] = current
            state[~valid] = 0
        first_nonfinite[alive & ~valid] = t + 1
        alive = valid
        finite[:, t + 1] = valid
        motion[valid, t + 1] = current[valid]
    return {'motion': motion, 'finite': finite, 'first_nonfinite': first_nonfinite}


def unpack_motion(motion, initial, order):
    """Canonical color addressing; no truth, rematching, or hidden count."""
    n, length = motion.shape[:2]
    out = np.zeros((n, length, 6, 4), dtype=np.float64)
    presence = np.zeros((n, 6), dtype=bool)
    radii = np.zeros((n, 6), dtype=np.float64)
    for i in range(n):
        for j, color in enumerate(order[i]):
            if initial[i, j, 6] > .5:
                out[i, :, color] = motion[i, :, j]
                presence[i, color] = True
                radii[i, color] = initial[i, j, 4]
    return out, presence, radii


def canonical_sequence(raw):
    n, length = raw.shape[:2]
    out = np.zeros((n, length, 6, 7), dtype=np.float64)
    for i in range(n):
        for j in range(4):
            if raw[i, 0, j, 4] > 0:
                color = int(raw[i, 0, j, 5])
                assert np.all(raw[i, :, j, 5] == color)
                out[i, :, color] = raw[i, :, j]
    return out


def contact_proxy(motion, radii, presence, context, action, finite):
    """Frame-level geometric + velocity-residual proxy, NOT simulator events.

    Arrays have color slots (six) or packed slots (four). NaN rows are marked
    invalid separately. A crossing with no velocity response is not a contact.
    """
    n, length, slots = motion.shape[:3]
    start, end = motion[:, :-1], motion[:, 1:]
    valid = finite[:, :-1] & finite[:, 1:]
    a = np.zeros((n, length - 1, slots, 2))
    a[:, 0] = action
    drag = context[:, 2]
    decay = np.exp(-drag)
    gain = np.ones_like(drag)
    np.divide(-np.expm1(-drag), drag, out=gain, where=drag != 0)
    free_v = ((start[..., 2:4] + a) * decay[:, None, None, None]
              + context[:, None, None, :2] * gain[:, None, None, None])
    residual = np.linalg.norm(end[..., 2:4] - free_v, axis=-1) > .1
    limit = 31.5 - radii
    near_wall = np.any(np.maximum(np.abs(start[..., :2]), np.abs(end[..., :2]))
                       >= limit[:, None, :, None] - 1.5, axis=-1)
    wall = np.any(residual & near_wall & presence[:, None], axis=-1)
    pair = np.zeros((n, length - 1), dtype=bool)
    for i in range(slots):
        for j in range(i + 1, slots):
            d0 = start[:, :, i, :2] - start[:, :, j, :2]
            d1 = end[:, :, i, :2] - end[:, :, j, :2]
            change = d1 - d0
            norm2 = (change * change).sum(-1)
            u = np.zeros_like(norm2)
            np.divide(-(d0 * change).sum(-1), norm2, out=u, where=norm2 > 0)
            u = u.clip(0, 1)
            distance = np.linalg.norm(d0 + u[..., None] * change, axis=-1)
            pair |= ((distance <= (radii[:, i] + radii[:, j])[:, None] + 1.5)
                     & (residual[:, :, i] | residual[:, :, j])
                     & (presence[:, i] & presence[:, j])[:, None])
    return np.stack([pair & valid, wall & valid], axis=-1), valid
