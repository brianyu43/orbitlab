"""Truth-factor swaps on matched moving disks, with complete deterministic replay.

Predicted slots retain their inference order throughout dynamics and rendering.
Truth labels are used only for scoring and for explicitly named oracle controls.
"""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from v3_common import HERE, dump, lock, sha
import r5_disk_data as d
import r5_disk_core as c
import r5_disk_train as tr
from r3_core import coarse
from r4_core import state_hash

SCENARIOS = ['baseline', 'reverse_impulse', 'force_delta']
STATE_METRICS = ['position_mae_px', 'velocity_mae_px', 'failed_path', 'wall_path_gt1px', 'overlap_path_gt0_5px', 'missing_fraction', 'extra_fraction', 'count_correct', 'state_strict', 'click_correct', 'target_response_error_px', 'nontarget_response_error_px', 'zero_target_response_error_px', 'zero_nontarget_response_error_px']
PIXEL_METRICS = ['image_mse', 'changed_pixel_mse', 'preserved_pixel_mse', 'changed_fraction', 'changed_mae', 'preserved_mae', 'pixel_gate', 'copy_mse']


def freeze():
    tr.freeze()
    return lock(d.BASE / 'evaluation_protocol.json', {
        'version': 1, 'source_sha256': sha(__file__), 'training_protocol_sha256': sha(d.BASE / 'training_protocol.json'),
        'evaluation_groups': 'Reader/force calibration on96valcount2. Integrated development test96/count at2/3/4/6; this is one fresh development dataset, not3independent confirmations.',
        'factorial': 'Three transitions bounded/contact/coarsephysical xtrue/learnedinitialstate xknown/inferrednetforce+drag xknown/neuralrenderer. Sameobservations,requestedimpulse andreferencefuture. Noend-to-endfine-tuning. Both learnedtransitions retrained onshareddata; exactsimulator is anoracle usedforlabels.',
        'horizons': d.HORIZONS, 'scenarios': SCENARIOS,
        'inference': 'FourRGBframes0..3 only. Predictedpresentprobability>=.5, allup-to6peaks retained, noGTcount. Clickattrue current targetcenter is suppliedexternalcommand; choose nearestpresentprediction. Negate sameclickedimpulse forreversecondition. Forcecondition adds known.04x toestimatedbaselineforce aswell.',
        'matching': 'Initialone-to-one minimumsquaredXY assignment forSCORINGONLY. Matchedinitialdistance>8px countsasmissing. Sameassignment retained throughallfutureframes andcounterfactuals. Neverreorderpredictedslots bytruth. Extraobjects remaininphysics/rendering and reported; zeroobjects getsblankimage andallmissing penalties.',
        'state_metrics': STATE_METRICS,
        'state_thresholds': 'AnytrajectorynonfiniteorabsXY>64px isfailure. Failure/missingobjectposition128px andvelocity16px/frame; cannotdropfrommeans. Initialfalsepositivescountasextra; countmismatchseparatelyreported. Strictallobjectsrequirescorrectcount, eachXYaxis<=1px,eachvelocityaxis<=.5px/frame,radius<=.5px,eachRGBchannel<=20/255,correctclick,no failedpath. Fullpathwall>1px oroverlap>.5px measuredatframeendpoints includinginitialstate; reportpriorprojectioneffects separately throughinitialviolations.',
        'pixel_metrics': PIXEL_METRICS, 'pixel_gate': 'ChangedMAE<=.05 andpreservedMAE<=.01, sameR2tolerances; semantic/statecriterion remainsseparate. Emptychanged/preservedregion storeszeroerror andzeroarea; aggregateregiondenominators explicitly.',
        'counterfactual_metric': 'Displacementresponse(predictedintervention-predictedbaseline) versusmatchedtrueintervention-truebaseline; targetandnontargetseparate. Missing/failedgets128px; zeroresponsebaseline retained. No identicalimagehumanjudgement claim.',
        'controls': 'Inputcopy for everyfuture; truefuturestate throughknownrenderer (exactpixel uppercontrol) and samelearnedneuraldecoder. Coarsephysicalwithoutlearnedresidual mandatory. Blackbackground/diskshape/sizepriors disclosed; this isnot texturedpolygonintegration.',
        'gate': 'Contactcandidate diagnostic advances only ifcount4step61true-stateknown-force error<=.8bounded, allcount61failed<.01/wall<.01/overlap<.05, count4nontargetreverse-response<=1.05bounded, count2step16everycontactstratum<=1px, andlearnedreader count6missing<=.05. Passing triggersrequirednew3datax3init confirmations, never silentlyclaimed. Failingstopsautomaticexpansionandretainsdiagnosis. No testthreshold/hyperparameter tuning.',
        'replay': 'Storeallnumerictrajectoriesandper-episode rows; everyinference repeated CPU2threads withsamebatchsize. Everyrenderedfloat32image SHA256 recorded andmatched onrepeat; avoidslargeRGBcache. First4exampleimagesretainedperunit. Numericpredictionsbitexact required; metricsatol1e-10. Thisisforwardreplay,notindependentretraining.',
        'checkpoint_audit': 'Verifyeveryfullstepcount, savedoptimizerparameterstepcounts, initialstate, exactsampleorderandRNG, finalprogress/checkpointidentity, shared960trainmanifest.100steppilots separatelyreported.',
    })


def load_model(arm):
    root = d.BASE / 'runs' / arm
    run = json.loads((root / 'run.json').read_text())
    assert run['steps'] == c.STEPS[arm] and run['protocol_sha256'] == sha(d.BASE / 'training_protocol.json')
    assert sha(root / 'model.pt') == run['checkpoint_sha256']
    ck = torch.load(root / 'model.pt', weights_only=True)
    m = c.make(arm)
    m.load_state_dict(ck['state_dict'])
    return m.eval(), run


def data(split, count):
    root = d.path(split, count)
    rec = json.loads((root / 'manifest.json').read_text())
    for name, value in rec['files'].items():
        assert sha(root / name) == value
    return np.load(root / 'trajectories.npz'), np.load(root / 'images.npy')


def context_record(split, count):
    return {'evaluation_protocol_sha256': sha(d.BASE / 'evaluation_protocol.json'), 'data_manifest_sha256': sha(d.path(split, count) / 'manifest.json'),
            'run_hashes': {arm: sha(d.BASE / 'runs' / arm / 'run.json') for arm in c.ARMS}}


def assignment(initial, truth):
    """Predicted rows already compacted, truthN known only to this metric."""
    result = np.full(len(truth), -1, np.int64)
    if len(initial):
        distance = np.linalg.norm(truth[:, None, :2] - initial[None, :, :2], axis=-1) * 31.5
        ri, pi = linear_sum_assignment(distance ** 2)
        keep = distance[ri, pi] <= 8
        result[ri[keep]] = pi[keep]
    return result


@torch.no_grad()
def perceive(split, count, verify=False):
    freeze()
    torch.set_num_threads(2)
    a, images = data(split, count)
    out = d.BASE / 'evaluation' / split / f'n{count}' / 'perception'
    out.mkdir(parents=True, exist_ok=True)
    ctx = context_record(split, count)
    if not verify and (out / 'summary.json').exists():
        assert json.loads((out / 'summary.json').read_text())['context'] == ctx
        return
    reader, _ = load_model('reader')
    force, _ = load_model('force')
    states, forces = [], []
    for first in range(0, len(images), 16):
        window = torch.from_numpy(images[first:first + 16]).permute(0, 1, 4, 2, 3).float() / 255
        states.append(reader(window)['state'].numpy())
        forces.append(force(window).numpy())
    states, forces = np.concatenate(states), np.concatenate(forces)
    assert np.isfinite(states).all() and np.isfinite(forces).all()
    truth = c.encode(a['states'][:, 3], a['colors']).numpy()
    rows = []
    for i, raw in enumerate(states):
        present = raw[raw[:, 8] >= .5]
        match = assignment(present, truth[i])
        valid = match >= 0
        pos = np.full(count, 128., np.float64)
        vel = np.full(count, 16., np.float64)
        rad = np.full(count, 8., np.float64)
        rgb = np.ones(count, np.float64)
        if valid.any():
            diff = np.abs(present[match[valid]] - truth[i, valid])
            pos[valid] = (diff[:, :2] * 31.5).mean(-1)
            vel[valid] = (diff[:, 2:4] * 3).mean(-1)
            rad[valid] = diff[:, 4] * 8
            rgb[valid] = diff[:, 5:8].mean(-1)
        target = a['target_ids'][i]
        clicked = -1 if len(present) == 0 else int(np.argmin(np.linalg.norm(present[:, :2] - truth[i, target, :2], axis=-1)))
        rows.append([len(present), float(len(present) == count), 1 - valid.mean(), (len(present) - valid.sum()) / count,
                     pos.mean(), vel.mean(), rad.mean(), rgb.mean(), float(match[target] >= 0 and clicked == match[target])])
    rows = np.asarray(rows)
    force_error = np.abs(forces - c.context(a['contexts']).numpy()) * np.asarray([.1, .1, .01])
    payload = {'states': states, 'forces': forces, 'rows': rows, 'force_errors': force_error}
    names = ['predicted_count', 'count_correct', 'missing_fraction', 'extra_per_trueobject', 'position_mae_px', 'velocity_mae_px', 'radius_mae_px', 'rgb_mae', 'click_correct']
    if verify:
        saved = np.load(out / 'predictions.npz')
        rec = json.loads((out / 'summary.json').read_text())
        assert rec['context'] == ctx and sha(out / 'predictions.npz') == rec['predictions_sha256']
        for key, value in payload.items():
            np.testing.assert_array_equal(value, saved[key])
        dump(out / 'verification.json', {'summary_sha256': sha(out / 'summary.json'), 'all_episodes_replayed': len(images), 'all_arrays_bitexact': True})
    else:
        assert not (out / 'predictions.npz').exists(), 'Preserve partial evidence; explicit recovery required'
        np.savez_compressed(out / 'predictions.npz', **payload)
        groups = {}
        for name, mask in [('all', np.ones(len(images), bool)), ('same_color', a['same_color']), ('different_colors', ~a['same_color'])] + [(f'stratum{k}', a['stratum'] == k) for k in range(3)]:
            groups[name] = {'episodes': int(mask.sum()), 'metrics': dict(zip(names, rows[mask].mean(0).tolist())), 'force_mae_xy_drag': force_error[mask].mean(0).tolist()}
        dump(out / 'summary.json', {'context': ctx, 'metric_names': names, 'groups': groups, 'predictions_sha256': sha(out / 'predictions.npz')})


def prepared_inputs(a, perception, state_kind, force_kind):
    truth = c.encode(a['states'][:, 3], a['colors']).numpy()
    initial = []
    matches, clicks = [], []
    for i in range(len(truth)):
        if state_kind == 'true':
            s = truth[i].copy()
        else:
            raw = perception['states'][i]
            s = raw[raw[:, 8] >= .5].copy()
            s[:, 8] = 1
        initial.append(s)
        matches.append(assignment(s, truth[i]))
        clicks.append(-1 if len(s) == 0 else int(np.argmin(np.linalg.norm(s[:, :2] - truth[i, a['target_ids'][i], :2], axis=-1))))
    contexts = c.context(a['contexts']).numpy() if force_kind == 'known' else perception['forces'].copy()
    return initial, np.asarray(matches), np.asarray(clicks), contexts


@torch.no_grad()
def rollout(a, perception, arm, state_kind, force_kind):
    initial, matches, clicks, contexts = prepared_inputs(a, perception, state_kind, force_kind)
    model, run = (None, None) if arm == 'coarse' else load_model(arm)
    counts = np.asarray([len(v) for v in initial])
    all_states = np.zeros((len(initial), 3, 129, 6, 9), np.float32)
    for n in range(1, 7):
        ids = np.flatnonzero(counts == n)
        if not len(ids):
            continue
        # Group by predictedN; never pad absentobjects into collision handling.
        base = np.stack([initial[i] for i in ids])
        current = torch.tensor(np.repeat(base[:, None], 3, axis=1).reshape(-1, n, 9))
        ctx = np.repeat(contexts[ids, None], 3, axis=1)
        ctx[:, 2, 0] += .4
        ctx = torch.tensor(ctx.reshape(-1, 3), dtype=torch.float32)
        impulse = a['actions'][ids, 3, a['target_ids'][ids]] / 3
        action = np.zeros((len(ids), 3, n, 2), np.float32)
        for k, i in enumerate(ids):
            action[k, 0, clicks[i]] = impulse[k]
            action[k, 1, clicks[i]] = -impulse[k]
            action[k, 2, clicks[i]] = impulse[k]
        action = torch.from_numpy(action.reshape(-1, n, 2))
        sequence = [current.numpy().copy()]
        for step in range(128):
            act = action if step == 0 else torch.zeros_like(action)
            if arm == 'coarse':
                current = torch.cat([coarse(current, act, ctx), current[..., 4:]], -1)
            else:
                current = model(current, act, ctx, run['velocity_limit'])
            sequence.append(current.numpy().copy())
        seq = np.stack(sequence, 1).reshape(len(ids), 3, 129, n, 9)
        all_states[ids, :, :, :n] = seq
    return {'states': all_states, 'counts': counts, 'matches': matches, 'clicks': clicks}


def physical_truth(a):
    return np.stack([a['states'][:, 3:], a['reversed_states'], a['force_states']], 1)


def state_rows(pred, a):
    truth = physical_truth(a)
    ntrue = truth.shape[3]
    result = np.zeros((len(truth), 3, len(d.HORIZONS), len(STATE_METRICS)), np.float64)
    for i, n in enumerate(pred['counts']):
        matches = pred['matches'][i]
        valid = matches >= 0
        target = int(a['target_ids'][i])
        clicked_correct = valid[target] and pred['clicks'][i] == matches[target]
        for scenario in range(3):
            for hi, h in enumerate(d.HORIZONS):
                path = pred['states'][i, scenario, :h + 1, :n]
                failed = not np.isfinite(path).all() or bool(np.any(np.abs(path[..., :2]) * 31.5 > 64))
                wall, overlap = False, False
                if n and not failed:
                    wall = bool((np.abs(path[..., :2]) * 31.5 - (31.5 - path[..., 4, None] * 8) > 1).any())
                    for j in range(n):
                        for k in range(j + 1, n):
                            penetration = (path[:, j, 4] + path[:, k, 4]) * 8 - np.linalg.norm(path[:, j, :2] - path[:, k, :2], axis=-1) * 31.5
                            overlap |= bool((penetration > .5).any())
                pos, vel = np.full(ntrue, 128.), np.full(ntrue, 16.)
                strict = np.zeros(ntrue, bool)
                response = np.full(ntrue, 128.)
                true_response = truth[i, scenario, h, :, :2] - truth[i, 0, h, :, :2]
                zero_response = np.abs(true_response).mean(-1)
                base_path = pred['states'][i, 0, :h + 1, :n]
                base_failed = not np.isfinite(base_path).all() or bool(np.any(np.abs(base_path[..., :2]) * 31.5 > 64))
                if valid.any() and not failed:
                    v = path[-1, matches[valid]]
                    gt = truth[i, scenario, h, valid]
                    dp = np.abs(v[:, :2] * 31.5 - gt[:, :2])
                    dv = np.abs(v[:, 2:4] * 3 - gt[:, 2:4])
                    pos[valid], vel[valid] = dp.mean(-1), dv.mean(-1)
                    strict[valid] = (dp <= 1).all(-1) & (dv <= .5).all(-1) & (np.abs(v[:, 4] * 8 - gt[:, 4]) <= .5) & (np.abs(v[:, 5:8] - a['colors'][i, valid]) <= 20 / 255).all(-1)
                    if not base_failed:
                        delta = (v[:, :2] - pred['states'][i, 0, h, matches[valid], :2]) * 31.5
                        response[valid] = np.abs(delta - true_response[valid]).mean(-1)
                others = np.arange(ntrue) != target
                result[i, scenario, hi] = [pos.mean(), vel.mean(), failed, wall, overlap, 1 - valid.mean(), (n - valid.sum()) / ntrue,
                    n == ntrue, bool(strict.all() and n == ntrue and clicked_correct and not failed), clicked_correct,
                    response[target], response[others].mean(), zero_response[target], zero_response[others].mean()]
    return result


def pixel_rows(pred, source, target):
    changed = (source != target).any(-1)
    err = pred.astype(np.float64) - target / 255.
    square, absolute = (err ** 2).mean(-1), np.abs(err).mean(-1)
    n_changed = changed.sum((1, 2))
    n_preserved = 4096 - n_changed
    cm = (square * changed).sum((1, 2)) / np.maximum(1, n_changed)
    pm = (square * ~changed).sum((1, 2)) / np.maximum(1, n_preserved)
    ca = (absolute * changed).sum((1, 2)) / np.maximum(1, n_changed)
    pa = (absolute * ~changed).sum((1, 2)) / np.maximum(1, n_preserved)
    copy = ((source / 255. - target / 255.) ** 2).mean((1, 2, 3))
    return np.stack([square.mean((1, 2)), cm, pm, n_changed / 4096, ca, pa, (ca <= .05) & (pa <= .01), copy], -1)


@torch.no_grad()
def render_states(states, counts, kind, decoder):
    out = np.zeros((len(states), 64, 64, 3), np.float32)
    for n in range(1, 7):
        ids = np.flatnonzero(counts == n)
        for first in range(0, len(ids), 16):
            group = ids[first:first + 16]
            s = states[group, :n]
            # Failed paths rendered blank and still scored, never discarded.
            good = np.isfinite(s).all((1, 2)) & (np.abs(s[..., :2]) * 31.5 <= 64).all((1, 2))
            group, s = group[good], s[good]
            if not len(group):
                continue
            if kind == 'neural':
                out[group] = decoder(torch.from_numpy(s)).permute(0, 2, 3, 1).numpy()
            else:
                raw = np.zeros((*s.shape[:2], 7), np.float64)
                raw[..., :5] = s[..., :5].astype(np.float64) * np.asarray([31.5, 31.5, 3, 3, 8])
                raw[..., 6] = 1
                out[group] = d.render(raw, s[..., 5:8]) / np.float32(255)
    return out


def hashes(images):
    return [hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() for image in images]


def aggregates(a, state, pixels):
    result = []
    for scenario, name in enumerate(SCENARIOS):
        for hi, h in enumerate(d.HORIZONS):
            for group, mask in [('all', np.ones(len(state), bool)), ('same_color', a['same_color']), ('different_colors', ~a['same_color'])] + [(f'stratum{k}', a['stratum'] == k) for k in range(3)]:
                item = {'scenario': name, 'horizon': h, 'group': group, 'episodes': int(mask.sum()),
                        'state': dict(zip(STATE_METRICS, state[mask, scenario, hi].mean(0).tolist())), 'renderers': {}}
                for renderer, rows in pixels.items():
                    r = rows[mask, scenario, hi]
                    values = dict(zip(PIXEL_METRICS, r.mean(0).tolist()))
                    changed = r[:, 3] > 0
                    values['changed_scenes'] = int(changed.sum())
                    values['changed_scene_changed_pixel_mse'] = float(r[changed, 1].mean()) if changed.any() else None
                    values['state_and_pixel_strict'] = float((r[:, 6] * state[mask, scenario, hi, STATE_METRICS.index('state_strict')]).mean())
                    item['renderers'][renderer] = values
                result.append(item)
    return result


@torch.no_grad()
def evaluate(count, arm, state_kind, force_kind, verify=False):
    freeze()
    torch.set_num_threads(2)
    a, images = data('test', count)
    ctx = context_record('test', count)
    base = d.BASE / 'evaluation' / 'test' / f'n{count}'
    per_root = base / 'perception'
    assert json.loads((per_root / 'verification.json').read_text())['summary_sha256'] == sha(per_root / 'summary.json')
    perception = np.load(per_root / 'predictions.npz')
    unit = base / f'{arm}_{state_kind}_{force_kind}'
    unit.mkdir(exist_ok=True)
    ctx.update(arm=arm, state_kind=state_kind, force_kind=force_kind, perception_sha256=sha(per_root / 'predictions.npz'))
    if not verify and (unit / 'summary.json').exists():
        assert json.loads((unit / 'summary.json').read_text())['context'] == ctx
        return
    started = time.perf_counter()
    pred = rollout(a, perception, arm, state_kind, force_kind)
    state = state_rows(pred, a)
    decoder, _ = load_model('decoder')
    truth = physical_truth(a)
    pixels = {kind: np.zeros((len(images), 3, 4, len(PIXEL_METRICS)), np.float64) for kind in ('known', 'neural')}
    rgb_hashes, examples = {}, {}
    for si, scenario in enumerate(SCENARIOS):
        for hi, horizon in enumerate(d.HORIZONS):
            target = d.render(truth[:, si, horizon], a['colors'])
            for renderer in pixels:
                rgb = render_states(pred['states'][:, si, horizon], pred['counts'], renderer, decoder)
                pixels[renderer][:, si, hi] = pixel_rows(rgb, images[:, 3], target)
                key = f'{scenario}_h{horizon}_{renderer}'
                rgb_hashes[key] = hashes(rgb)
                if si == 0:
                    examples[key] = rgb[:4]
            if si == 0:
                examples[f'truth_h{horizon}'] = target[:4]
    arrays = {**pred, 'state_rows': state, **{f'{k}_pixel_rows': v for k, v in pixels.items()}}
    if verify:
        receipt = json.loads((unit / 'summary.json').read_text())
        assert receipt['context'] == ctx and sha(unit / 'arrays.npz') == receipt['arrays_sha256']
        old = np.load(unit / 'arrays.npz')
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, old[key])
        assert rgb_hashes == json.loads((unit / 'image_hashes.json').read_text())
        assert receipt['aggregates'] == aggregates(a, state, pixels)
        dump(unit / 'verification.json', {'summary_sha256': sha(unit / 'summary.json'), 'all_numeric_arrays_bitexact': True,
                                          'full_trajectory_replays': len(images) * 3, 'full_RGB_replays_bitexact': len(images) * 3 * 4 * 2,
                                          'all_metric_rows_recomputed': True, 'seconds': time.perf_counter() - started})
    else:
        assert not (unit / 'arrays.npz').exists(), 'Preserve partial evidence; explicit recovery required'
        np.savez_compressed(unit / 'arrays.npz', **arrays)
        np.savez_compressed(unit / 'examples.npz', **examples, source=images[:4, 3])
        dump(unit / 'image_hashes.json', rgb_hashes)
        dump(unit / 'summary.json', {'context': ctx, 'arrays_sha256': sha(unit / 'arrays.npz'), 'image_hashes_sha256': sha(unit / 'image_hashes.json'),
                                    'examples_sha256': sha(unit / 'examples.npz'), 'aggregates': aggregates(a, state, pixels), 'seconds': time.perf_counter() - started})
    print('disk', 'verify' if verify else 'evaluate', count, arm, state_kind, force_kind, 'seconds', time.perf_counter() - started, flush=True)


@torch.no_grad()
def controls(count, verify=False):
    freeze()
    torch.set_num_threads(2)
    a, images = data('test', count)
    root = d.BASE / 'evaluation' / 'test' / f'n{count}' / 'controls'
    root.mkdir(parents=True, exist_ok=True)
    ctx = context_record('test', count)
    if not verify and (root / 'summary.json').exists():
        assert json.loads((root / 'summary.json').read_text())['context'] == ctx
        return
    decoder, _ = load_model('decoder')
    truth = physical_truth(a)
    rows, image_hashes, example = {}, {}, {}
    for si, scenario in enumerate(SCENARIOS):
        for hi, h in enumerate(d.HORIZONS):
            target = d.render(truth[:, si, h], a['colors'])
            encoded = c.encode(truth[:, si, h], a['colors']).numpy()
            for kind in ('copy', 'true_future_neural', 'exact_known_renderer'):
                if kind == 'copy':
                    rgb = images[:, 3].astype(np.float32) / 255
                elif kind == 'exact_known_renderer':
                    rgb = target.astype(np.float32) / 255
                else:
                    rgb = render_states(encoded, np.full(len(images), count), 'neural', decoder)
                key = f'{scenario}_h{h}_{kind}'
                rows[key] = pixel_rows(rgb, images[:, 3], target)
                image_hashes[key] = hashes(rgb)
                if si == 0:
                    example[key] = rgb[:4]
    if verify:
        rec = json.loads((root / 'summary.json').read_text())
        assert rec['context'] == ctx and rec['rows_sha256'] == sha(root / 'rows.npz')
        old = np.load(root / 'rows.npz')
        for k, value in rows.items():
            np.testing.assert_array_equal(value, old[k])
        assert image_hashes == json.loads((root / 'image_hashes.json').read_text())
        dump(root / 'verification.json', {'summary_sha256': sha(root / 'summary.json'), 'full_RGB_replays_bitexact': len(images) * 3 * 4 * 3, 'all_metric_rows_bitexact': True})
    else:
        assert not (root / 'rows.npz').exists()
        np.savez_compressed(root / 'rows.npz', **rows)
        np.savez_compressed(root / 'examples.npz', **example)
        dump(root / 'image_hashes.json', image_hashes)
        dump(root / 'summary.json', {'context': ctx, 'rows_sha256': sha(root / 'rows.npz'), 'image_hashes_sha256': sha(root / 'image_hashes.json'),
                                    'examples_sha256': sha(root / 'examples.npz'), 'all_episode_pixel_metrics': {k: dict(zip(PIXEL_METRICS, v.mean(0).tolist())) for k, v in rows.items()}})


def verify_training():
    freeze()
    torch.set_num_threads(2)
    records = []
    for arm in c.ARMS:
        model, run = load_model(arm)
        root = d.BASE / 'runs' / arm
        torch.manual_seed(961000)
        initial = c.make(arm)
        assert state_hash(initial) == run['initial_state_sha256']
        assert sum(p.numel() for p in model.parameters()) == run['parameters']
        ck = torch.load(root / 'progress.pt', weights_only=True)
        assert ck['step'] == c.STEPS[arm]
        assert ck['protocol_sha256'] == sha(d.BASE / 'training_protocol.json')
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, ck['state_dict'][name], rtol=0, atol=0)
        steps = [int(v['step']) for v in ck['optimizer']['state'].values()]
        assert len(steps) == len(list(model.parameters())) and all(v == c.STEPS[arm] for v in steps)
        rng = torch.Generator().manual_seed(run['rng_seed'])
        parts = []
        for _ in range(c.STEPS[arm]):
            b = 16 if arm == 'reader' else 32
            ep = torch.randint(960, (b,), generator=rng)
            if arm == 'reader':
                t = torch.randint(3, 21, (b,), generator=rng)
            elif arm == 'force':
                t = torch.full((b,), 3)
            else:
                t = torch.randint(21 if arm == 'decoder' else 13, (b,), generator=rng)
            parts.append(torch.stack([ep, t], 1).numpy().astype('<i8').tobytes())
        order = b''.join(parts)
        assert order == ck['sample_order_bytes'] and hashlib.sha256(order).hexdigest() == run['sample_order_sha256']
        assert torch.equal(rng.get_state(), ck['rng'])
        assert run['train_manifest_sha256'] == sha(d.path('train', 2) / 'manifest.json')
        pilot = json.loads((d.BASE / 'benchmark' / arm / 'run.json').read_text())
        assert pilot['steps'] == 100 and pilot['initial_state_sha256'] == run['initial_state_sha256']
        records.append({'arm': arm, 'steps': c.STEPS[arm], 'run_sha256': sha(root / 'run.json'), 'optimizer_all_parameters_step_count_verified': True,
                        'final_progress_weights_bitexact': True, 'initial_weights_sampleorder_RNG_verified': True, 'pilot_reset_initialization_verified': True})
    bounded = json.loads((d.BASE / 'runs' / 'bounded' / 'run.json').read_text())
    contact = json.loads((d.BASE / 'runs' / 'contact' / 'run.json').read_text())
    assert bounded['initial_state_sha256'] == contact['initial_state_sha256']
    assert bounded['sample_order_sha256'] == contact['sample_order_sha256']
    dump(d.BASE / 'training_verification.json', {'evaluation_protocol_sha256': sha(d.BASE / 'evaluation_protocol.json'), 'components': records,
                                               'shared_transition_initialization_and_samples': True, 'independent_retraining': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['freeze', 'perceive', 'evaluate', 'controls', 'verify_training'])
    p.add_argument('--count', type=int, choices=d.COUNTS, default=2)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--arm', choices=['bounded', 'contact', 'coarse'], default='bounded')
    p.add_argument('--state', choices=['true', 'learned'], default='true')
    p.add_argument('--force', choices=['known', 'inferred'], default='known')
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'perceive':
        perceive(args.split, args.count, args.verify)
    elif args.stage == 'controls':
        controls(args.count, args.verify)
    elif args.stage == 'verify_training':
        verify_training()
    else:
        evaluate(args.count, args.arm, args.state, args.force, args.verify)
