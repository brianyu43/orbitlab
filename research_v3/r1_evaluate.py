"""RGB-only inference cached before labels; geometric and raster scores separated."""
import argparse, json
from functools import lru_cache
import numpy as np
import torch
import r1_core as c
from v3_common import sha, dump, event

@lru_cache(maxsize=32)
def template_bank(renderer, color):
    tensors, records = [], []
    for shape in range(4):
        for radius in range(5, 9):
            for pose in range(4):
                scene = [{'id': 0, 'shape': shape, 'color': color, 'radius': radius, 'pose': pose, 'x': 32, 'y': 32}]
                image = c.render(scene, renderer)[0]; x, d = c.inputs(image, 'alpha')
                assert len(x) == 1
                tensors.append(x[0]); records.append({'shape': shape, 'radius': radius, 'pose': pose, 'dx': 32-d[0]['cx'], 'dy': 32-d[0]['cy']})
    return torch.stack(tensors).flatten(1), records

@torch.no_grad()
def template_infer(images, renderer):
    output = []
    for image in images:
        x, detections = c.inputs(image, 'alpha'); scene = []
        for i, (patch, det) in enumerate(zip(x, detections)):
            bank, records = template_bank(renderer, det['color'])
            error = (bank-patch.flatten()).square().mean(-1); idx = int(error.argmin()); rec = records[idx]
            scene.append({'id': i, 'shape': rec['shape'], 'radius': rec['radius'], 'pose': rec['pose'], 'x': det['cx']+rec['dx'], 'y': det['cy']+rec['dy'], 'color': det['color']})
        output.append(scene)
    return output

def check(pred, truth, quotient=False):
    if pred is None: return False
    return bool(pred['shape'] == truth['shape'] and pred['color'] == truth['color'] and abs(pred['x']-truth['x']) <= 1 and abs(pred['y']-truth['y']) <= 1 and pred['radius'] == truth['radius'] and ((pred['pose']-truth['pose']) % (2 if quotient and truth['shape'] == 3 else 4) == 0))

def score(predictions, labels, images, renderer):
    rows, examples = [], []
    for i, (pred, rec, image) in enumerate(zip(predictions, labels, images)):
        truth = rec['objects']; color_to_index = {v['color']: j for j, v in enumerate(pred)}
        strict = [check(pred[color_to_index[v['color']]] if v['color'] in color_to_index else None, v) for v in truth]
        eq = [check(pred[color_to_index[v['color']]] if v['color'] in color_to_index else None, v, True) for v in truth]
        gt_to_pred = [pred[color_to_index[v['color']]] if v['color'] in color_to_index else None for v in truth]
        identifiable = [j for j, v in enumerate(truth) if not (renderer == 'symmetric4_lanczos' and v['shape'] == 3)]
        obj_metrics = {'source_strict': float(all(strict) and len(pred) == len(truth)), 'source_quotient': float(all(eq) and len(pred) == len(truth)),
            'shape_accuracy': float(np.mean([p is not None and p['shape'] == v['shape'] for p, v in zip(gt_to_pred, truth)])),
            'color_accuracy': float(np.mean([p is not None and p['color'] == v['color'] for p, v in zip(gt_to_pred, truth)])),
            'size_accuracy': float(np.mean([p is not None and p['radius'] == v['radius'] for p, v in zip(gt_to_pred, truth)])),
            'position_mae_pixels': float(np.mean([abs(p['x']-v['x'])/2+abs(p['y']-v['y'])/2 if p else 64 for p, v in zip(gt_to_pred, truth)])),
            'raw_pose_accuracy': float(np.mean([p is not None and p['pose'] == v['pose'] for p, v in zip(gt_to_pred, truth)])),
            'identifiable_pose_accuracy': float(np.mean([gt_to_pred[j] is not None and gt_to_pred[j]['pose'] == truth[j]['pose'] for j in identifiable])) if identifiable else None}
        for task in c.tasks(truth, i):
            tid = task['target_id']; target_object = truth[tid]
            # Click is an input annotation: nearest visible foreground pixel of the target color to its mask centroid.
            rgb = image.astype(float); palette = np.asarray(c.o.PALETTE, dtype=float)
            unit = rgb / np.linalg.norm(rgb, axis=-1, keepdims=True).clip(1e-6)
            label = (unit @ (palette / np.linalg.norm(palette, axis=-1, keepdims=True)).T).argmax(-1)
            yy, xx = np.nonzero((label == target_object['color']) & (rgb.max(-1) > 13))
            nearest = int(np.argmin((xx-xx.mean())**2+(yy-yy.mean())**2))
            click = [int(xx[nearest]), int(yy[nearest])]
            chosen = int(np.argmin([(p['x']-click[0])**2+(p['y']-click[1])**2 for p in pred])) if pred else -1
            selected_ok = chosen >= 0 and pred[chosen]['color'] == target_object['color']
            output = [dict(p) for p in pred]
            if chosen >= 0:
                if 'color' in task['kind']: output[chosen]['color'] = task['color']
                if 'rotate' in task['kind']: output[chosen]['pose'] = (output[chosen]['pose']+1) % 4
                if task['kind'] == 'translate': output[chosen]['x'] += task['dx']
            target = task['target']; full, quotient, preserved = [], [], []
            for j, v in enumerate(target):
                p = output[color_to_index[truth[j]['color']]] if truth[j]['color'] in color_to_index else None
                full.append(check(p, v)); quotient.append(check(p, v, True))
                if j != tid: preserved.append(check(p, v, True))
            rendered = c.render(output, renderer)[0] if output else np.zeros_like(image)
            expected = c.render(target, renderer)[0]
            fg = (expected.max(-1)>13) | (image.max(-1)>13)
            # Pixel identity is stricter than <=1-pixel state tolerance and is never substituted for it.
            rows.append({'source_index': i, 'operation': task['kind'], **obj_metrics,
                'strict_edit': float(selected_ok and len(pred) == len(truth) and all(full)),
                'quotient_edit': float(selected_ok and len(pred) == len(truth) and all(quotient)),
                'target_selection': float(selected_ok), 'non_target_preservation': float(all(preserved)),
                'raster_equal': float(np.array_equal(rendered, expected)),
                'foreground_mae': float(np.abs(rendered.astype(float)-expected)[fg].mean()/255)})
            if i < 4: examples.extend([image, expected, rendered])
    numeric = [k for k in rows[0] if k not in ['source_index', 'operation']]
    metrics = {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in numeric}
    return rows, metrics, examples

def evaluate(seed, init, arm, splits=None):
    c.freeze(); torch.set_num_threads(2)
    splits = splits or (['val', 'ood'] if seed == c.DEVELOPMENT else ['test', 'ood'])
    if arm != 'template':
        path = c.BASE/f'runs/d{seed}_s{init}/{arm}/model.pt'
        ck = torch.load(path, weights_only=True); model = c.Reader(); model.load_state_dict(ck['state_dict']); model.eval(); model_hash = sha(path)
    else: model_hash = 'nonlearned_known_renderer_template'
    for split in splits:
        data = c.BASE/f'data/d{seed}/{split}'
        for renderer in c.RENDERERS:
            out = c.BASE/f'evaluation/d{seed}_s{init}/{arm}/{split}/{renderer}'; out.mkdir(parents=True, exist_ok=True)
            if (out/'summary.json').exists():
                old = json.loads((out/'summary.json').read_text()); assert old['model_sha256'] == model_hash
                continue
            images = np.load(data/f'{renderer}.npz')['images']
            predictions = template_infer(images, renderer) if arm == 'template' else c.infer(images, model, arm.split('_')[0])
            dump(out/'inference.json', predictions)
            # Labels enter only after the RGB-only model predictions are saved.
            labels = json.loads((data/'labels.json').read_text())
            rows, metrics, examples = score(predictions, labels, images, renderer)
            dump(out/'rows.json', rows)
            c.o.grid(torch.from_numpy(np.stack(examples)).permute(0, 3, 1, 2).float()/255, out/'examples.png', 6)
            dump(out/'summary.json', {'seed': seed, 'init': init, 'arm': arm, 'split': split, 'renderer': renderer, 'metrics': metrics, 'model_sha256': model_hash,
                'input_sha256': sha(data/f'{renderer}.npz'), 'labels_sha256': sha(data/'labels.json'), 'protocol_sha256': sha(c.BASE/'protocol.json'),
                'files': {p.name: sha(p) for p in out.iterdir() if p.is_file()}})
            print('R1 evaluation', seed, init, arm, split, renderer, {k: round(metrics[k], 4) for k in ['strict_edit', 'quotient_edit', 'raster_equal', 'position_mae_pixels']}, flush=True)
    event('R1_evaluation_complete', seed=seed, init=init, arm=arm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, default=c.DEVELOPMENT); parser.add_argument('--init', type=int, default=0)
    parser.add_argument('--arm', choices=c.ARMS+['template'], required=True); a = parser.parse_args(); evaluate(a.seed, a.init, a.arm)
