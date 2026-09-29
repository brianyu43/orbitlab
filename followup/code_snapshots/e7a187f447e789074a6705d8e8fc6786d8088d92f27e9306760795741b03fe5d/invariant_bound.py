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
    (out/'EXPLANATION_KO.md').write_text('''# 방향 정보를 버린 표현의 복원 한계\n\n회전한 네 입력을 x_g, 평균을 m, 모든 입력에 공통으로 내는 출력을 p라고 하자.\n\n`평균_g ||x_g-p||² = 평균_g ||x_g-m||² + ||m-p||²`\n\nx_g-p를 (x_g-m)+(m-p)로 나누어 제곱하면, 첫 항의 회전 평균이 0이므로 교차항이 사라진다. 두 번째 제곱 항은 음수가 아니므로 첫 항이 가능한 손실의 하한이다.\n\n쉽게 말하면 서로 다른 방향의 그림을 같은 암호로 압축했다면, 복원기는 어떤 방향을 골라야 할지 구별할 수 없다. 평균 그림을 출력해도 남는 차이가 피할 수 없는 오차다.\n\n이것은 알려진 최소제곱 항등식이다. 저장된 B3 모델과 새로운 512개 장면에서 수치로 확인했으며, 결과는 일반 픽셀 MSE에만 적용한다.\n''')
    update_status('A11','complete',['reports/invariant_bound_v1/summary.json','reports/invariant_bound_v1/EXPLANATION_KO.md'])
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
