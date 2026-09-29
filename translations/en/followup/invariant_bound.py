"""Numerically verify the standard orbit-mean MSE decomposition."""
import json
import time
from common import HERE, ROOT, dump, fresh_dir, update_status, torch, np, o, r
from diagnostics import probe_data


@torch.no_grad()
def main():
    torch.set_num_threads(4);dev=o.choose_device('mps')
    out=fresh_dir(HERE/'reports/invariant_bound_v1')
    path=ROOT/'runs/diagnostic_invariant_n1024_s0/ae.pt';ae,_=r.load_model(path,dev)
    rows=[];max_identity=0.;max_invariance=0.
    for split in ['test','ood']:
        x,y=probe_data(split)
        for i in range(0,len(x),32):
            base=x[i:i+32].to(dev);orbit=torch.stack([o.rotate(base,k) for k in range(4)])
            center=orbit.mean(0);pred=torch.stack([ae(orbit[k]) for k in range(4)])
            max_invariance=max(max_invariance,float((pred-pred[0:1]).abs().max()))
            p=pred.mean(0)
            lower=(orbit-center[None]).square().mean((0,2,3,4))
            error=(orbit-p[None]).square().mean((0,2,3,4))
            excess=(p-center).square().mean((1,2,3))
            max_identity=max(max_identity,float((error-lower-excess).abs().max()))
            actual=(orbit-pred).square().mean((0,2,3,4))
            for j in range(len(base)):
                rows.append({'split':split,'base_id':i+j,'shape':int(y[i+j,0]),'color':int(y[i+j,1]),
                             'lower_bound_mse':float(lower[j]),'b3_mse':float(actual[j]),'excess_mse':float(excess[j])})
    assert max_identity<1e-6 and max_invariance<1e-5
    assert min(v['b3_mse']-v['lower_bound_mse'] for v in rows)>-1e-6
    r.write_csv(out/'rows.csv',rows)
    summary={split:{k:float(np.mean([v[k] for v in rows if v['split']==split])) for k in ['lower_bound_mse','b3_mse','excess_mse']} for split in ['test','ood']}
    dump(out/'summary.json',{'per_split':summary,'max_identity_residual':max_identity,'max_b3_rotation_output_difference':max_invariance,
         'all_512_scenes_above_bound':True,'ae_sha256':r.sha(path),'code_sha256':r.sha(__file__),
         'note':'Standard least-squares identity, not a new theorem. Unweighted pixel MSE; not foreground MAE. Applies to invariant deterministic codes, not the full C4 representation.',
         'completed_unix':time.time()})
    (out/'EXPLANATION_KO.md').write_text('''# Limitations of Reconstruction of Expressions that Have Lost Directional Information\n\nLet's say the four rotated inputs are x_g, the mean is m, and the output is shared by all inputs, p.\n\n`Mean_g ||x_g-p||² = Mean_g ||x_g-m||² + ||m-p||²`\n\nSquaring the difference between x_g and p (divided by (x_g-m)+(m-p)) shows that the cross term disappears because the first term's rotation average is zero. Since the second squared term is not negative, the first term is the possible lower bound of loss.\n\nIn simple terms, if you compress different directional images into the same code, the reconstruction cannot distinguish which direction to choose. Even if you output the mean image, the remaining difference is an unavoidable error.\n\nThis is a known least squares identity. It has been verified numerically with the stored B3 model and the new 512 scenes, and the results apply only to the general pixel MSE.\n''')
    update_status('A11','complete',['reports/invariant_bound_v1/summary.json','reports/invariant_bound_v1/EXPLANATION_KO.md'])
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
