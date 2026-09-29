"""Episode-level metrics. Failed rollouts are never silently dropped."""
import numpy as np
from dynamics_autonomous import HORIZONS, unpack_motion, contact_proxy


def first_event(flags):
    return np.where(flags.any(axis=1), flags.argmax(axis=1) + 1, -1)


def mean_masked(values, mask):
    count = mask.sum(axis=-1)
    out = np.full(len(values), np.nan)
    np.divide(np.where(mask, values, 0).sum(axis=-1), count, out=out, where=count > 0)
    return out


def score_rollout(result, initial, order, context, action, truth, events, found, overflow):
    pred, presence, radii = unpack_motion(result['motion'], initial, order)
    true_presence = truth[:, 0, :, 6] > .5
    finite = result['finite']
    n, length = finite.shape
    all_presence = np.all(presence == true_presence, axis=1)
    matched = presence & true_presence
    pred_events, event_valid = contact_proxy(result['motion'], initial[..., 4], initial[..., 6] > .5,
                                             context, action, finite)
    wall_penetration = np.maximum(np.abs(pred[..., :2]) - (31.5-radii)[:, None, :, None], 0)
    wall_penetration = np.where(presence[:, None, :, None], wall_penetration, 0).max(axis=(2, 3))
    overlap = np.zeros((n, length))
    for i in range(6):
        for j in range(i+1, 6):
            d = np.linalg.norm(pred[:, :, i, :2]-pred[:, :, j, :2], axis=-1)
            penetration = np.maximum((radii[:, i]+radii[:, j])[:, None]-d, 0)
            overlap = np.maximum(overlap, np.where((presence[:, i]&presence[:, j])[:, None], penetration, 0))
    rows = []
    for h in HORIZONS:
        good = finite[:, h]
        diff = np.abs(pred[:, h]-truth[:, h, :, :4])
        position = mean_masked(diff[..., :2].mean(-1), true_presence)
        velocity = mean_masked(diff[..., 2:].mean(-1), true_presence)
        position[~good] = np.nan; velocity[~good] = np.nan
        mp = mean_masked(diff[..., :2].mean(-1), matched); mp[~good] = np.nan
        mv = mean_masked(diff[..., 2:].mean(-1), matched); mv[~good] = np.nan
        precise = np.all(diff[..., :2] <= 1, axis=-1) & np.all(diff[..., 2:] <= .1, axis=-1)
        precise &= np.abs(radii-truth[:, h, :, 4]) <= .5
        success = good & all_presence & np.all(precise | ~true_presence, axis=-1)
        distance = np.sum((pred[:, h, :, None, :2]-truth[:, h, None, :, :2])**2, axis=-1)
        distance = np.where(true_presence[:, None], distance, np.inf)
        closest = distance.argmin(axis=-1)
        association = mean_masked((closest == np.arange(6)[None]) & matched, true_presence)
        association[~good] = 0
        row = {
            'finite_fraction': good.astype(float),
            'all_presence_correct': all_presence.astype(float),
            'recall': mean_masked(presence.astype(float), true_presence),
            'command_target_present': found.astype(float),
            'discarded_present_colors': overflow.astype(float),
            'position_mae_pixels_finite': position,
            'velocity_mae_pixels_per_frame_finite': velocity,
            'matched_position_mae_pixels_finite': mp,
            'matched_velocity_mae_pixels_per_frame_finite': mv,
            'scene_success': success.astype(float),
            'own_color_nearest_position_fraction': association,
            'wall_penetration_pixels_finite': np.where(good, wall_penetration[:, h], np.nan),
            'pair_overlap_pixels_finite': np.where(good, overlap[:, h], np.nan),
            'any_wall_violation_over_1px_or_nonfinite': ((np.nanmax(wall_penetration[:, :h+1], axis=1)>1) | ~good).astype(float),
            'any_pair_overlap_over_1px_or_nonfinite': ((np.nanmax(overlap[:, :h+1], axis=1)>1) | ~good).astype(float),
        }
        for j, kind in enumerate(['pair', 'wall']):
            actual = events[:, :h, j] > 0
            predicted = pred_events[:, :h, j]
            pf, tf = first_event(predicted), first_event(actual)
            both = (pf > 0) & (tf > 0) & good
            valid = event_valid[:, :h]
            row.update({
                kind+'_event_tp': (actual & predicted & valid).sum(axis=1).astype(float),
                kind+'_event_fp': (~actual & predicted & valid).sum(axis=1).astype(float),
                # Invalid frames cannot be successful detections, so actual events remain false negatives.
                kind+'_event_fn': (actual & ~(predicted & valid)).sum(axis=1).astype(float),
                kind+'_event_invalid_frames': (~valid).sum(axis=1).astype(float),
                kind+'_true_event_present': (tf > 0).astype(float),
                kind+'_first_event_abs_error_both_finite': np.where(both, np.abs(pf-tf), np.nan),
                kind+'_first_event_within_1_on_true_event': np.where(tf > 0, both & (np.abs(pf-tf)<=1), np.nan),
                kind+'_missed_event_on_true_event': np.where(tf > 0, (pf < 0) | ~good, np.nan),
                kind+'_spurious_event_on_no_true_event': np.where(tf < 0, pf > 0, np.nan),
            })
        rows.append(row)
    keys=list(rows[0])
    values=np.stack([np.column_stack([row[k] for k in keys]) for row in rows], axis=1)
    return keys, values, pred_events, event_valid


def score_counterfactual(base, cf, initial, order, truth, cf_truth, target_colors):
    p, _, _ = unpack_motion(base['motion'], initial, order)
    q, _, _ = unpack_motion(cf['motion'], initial, order)
    true_presence = truth[:, 0, :, 6] > .5
    targets = np.arange(6)[None] == target_colors[:, None]
    nontarget = true_presence & ~targets
    effect = q-p
    expected = cf_truth[..., :4]-truth[..., :4]
    rows=[]
    for h in HORIZONS:
        good = base['finite'][:, h] & cf['finite'][:, h]
        error = np.abs(effect[:, h]-expected[:, h])
        row={'both_branches_finite': good.astype(float)}
        for label, mask in [('target', targets), ('nontarget', nontarget)]:
            for feature, sl in [('position', slice(0, 2)), ('velocity', slice(2, 4))]:
                value = mean_masked(error[..., sl].mean(-1), mask); value[~good] = np.nan
                row[label+'_'+feature+'_response_mae_finite'] = value
                row[label+'_'+feature+'_zero_response_mae'] = mean_masked(np.abs(expected[:, h, :, sl]).mean(-1), mask)
        actual = np.linalg.norm(expected[:, h, :, :2], axis=-1) > .5
        predicted = np.linalg.norm(effect[:, h, :, :2], axis=-1) > .5
        valid = good[:, None]
        row.update({
            'nontarget_affected_count': (actual & nontarget).sum(axis=1).astype(float),
            'nontarget_response_tp': (actual & predicted & nontarget & valid).sum(axis=1).astype(float),
            'nontarget_response_fp': (~actual & predicted & nontarget & valid).sum(axis=1).astype(float),
            'nontarget_response_fn': (actual & ~(predicted & valid) & nontarget).sum(axis=1).astype(float),
        })
        # Conditioning on a true response is declared and its denominator retained.
        affected = nontarget & actual
        value=mean_masked(error[..., :2].mean(-1), affected);value[~good]=np.nan
        row['affected_nontarget_position_response_mae_finite']=value
        rows.append(row)
    keys=list(rows[0]);values=np.stack([np.column_stack([v[k] for k in keys]) for v in rows], axis=1)
    return keys,values


def summarize(keys, values):
    output=[]
    for h_index,h in enumerate(HORIZONS):
        metrics={}
        for k_index,key in enumerate(keys):
            column=values[:,h_index,k_index]; ok=np.isfinite(column)
            metrics[key]={'mean':float(column[ok].mean()) if ok.any() else None,
                          'finite_episodes':int(ok.sum()),'total_episodes':len(column)}
        output.append({'horizon':h,'metrics':metrics})
    return output
