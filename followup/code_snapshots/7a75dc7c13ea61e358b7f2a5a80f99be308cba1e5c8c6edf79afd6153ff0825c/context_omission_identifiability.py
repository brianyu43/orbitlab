"""Constructed identical observed state/action, different next-state labels."""
import json
import numpy as np
from common import HERE,dump,r
import dynamics_world as d
from dynamics_context_omission import NAME,MASKS,freeze


def closed_form(state,context,action):
    net=context[:2]+context[2:4];gamma=context[4];times=np.arange(1,9)/8
    gain=-np.expm1(-gamma*times)/gamma if gamma else times
    velocity=np.exp(-gamma*times[:,None,None])*(state[None,:,2:4]+action[None])+gain[:,None,None]*net
    return np.concatenate([state[:,:2]+velocity.sum(0)/8,velocity[-1]],axis=-1)


def main():
    freeze();root=HERE/f'reports/{NAME}/identifiability';root.mkdir(parents=True,exist_ok=False)
    state=np.array([[-12.,0.,1.1,.2,4.,0.,1.],[12.,1.,-.4,.3,5.,1.,1.]])
    action=np.array([[.6,-.2],[0.,0.]])
    cases={'no_net':(np.array([0.,.03,-.02,0.,.004]),np.array([0.,.05,.03,0.,.004])),
           'no_drag':(np.array([0.,.04,.02,.01,.002]),np.array([0.,.04,.02,.01,.009])),
           'no_context':(np.array([0.,.03,-.02,0.,.002]),np.array([0.,.05,.03,0.,.009]))}
    rows=[];arrays={'current_state':state,'action':action};scale=np.array([31.5,31.5,3,3]);checks=0
    rng=np.random.default_rng(997501)
    for condition,(ca,cb) in cases.items():
        observed=[np.r_[c[:2]+c[2:4],c[4]]/np.array([.1,.1,.01])*MASKS[condition] for c in [ca,cb]]
        np.testing.assert_array_equal(observed[0],observed[1])
        aa,ea=d.step(state,ca,action);bb,eb=d.step(state,cb,action)
        assert ea['pair_impulses']==eb['pair_impulses']==ea['wall_impulses']==eb['wall_impulses']==0
        for c,out in [(ca,aa),(cb,bb)]:np.testing.assert_allclose(out[:,:4],closed_form(state,c,action),atol=1e-14,rtol=0)
        ya,yb=aa[:,:4]/scale,bb[:,:4]/scale;middle=(ya+yb)/2
        floor=float(np.mean((ya-yb)**2)/4);assert floor>0
        for prediction in [middle,*[middle+rng.normal(0,.03,middle.shape) for _ in range(32)]]:
            actual=(np.mean((prediction-ya)**2)+np.mean((prediction-yb)**2))/2
            expected=np.mean((prediction-middle)**2)+floor
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-18);checks+=1
        rows.append({'condition':condition,'identical_observed_input':True,'different_labels':True,
                     'max_next_position_difference_pixels':float(np.abs(aa[:,:2]-bb[:,:2]).max()),
                     'max_next_velocity_difference_pixels_per_frame':float(np.abs(aa[:,2:4]-bb[:,2:4]).max()),
                     'equal_pair_normalized_MSE_minimum':floor})
        for key,value in [('context_a',ca),('context_b',cb),('observed_context',observed[0]),('next_a',aa),('next_b',bb)]:arrays[condition+'_'+key]=value
    np.savez_compressed(root/'pairs.npz',**arrays)
    lines=['# 환경 정보가 없으면 정답이 하나로 정해지지 않는 예','',
        '현재 두 물체의 정확한 상태와 행동을 완전히 같게 두고, 모델에 숨긴 환경 값만 바꿨다. 힘 또는 저항을 가린 입력은 같지만 다음 정답은 달랐다. 세 예제의 물체 크기·속도·환경 값은 variable_force의 학습 범위 안에서 골랐다. 단, 같은 현재 상태에 서로 다른 환경을 지정한 구성 예제로서 무작위 자료의 빈도를 대표하지 않는다.','',
        '| 숨긴 정보 | 다음 위치의 최대 차이(px) | 다음 속도의 최대 차이(px/frame) | 두 사례의 정규화 MSE 최솟값 |','| --- | ---: | ---: | ---: |']
    labels={'no_net':'합력','no_drag':'저항','no_context':'합력과 저항'}
    for x in rows:lines.append(f"| {labels[x['condition']]} | {x['max_next_position_difference_pixels']:.8f} | {x['max_next_velocity_difference_pixels_per_frame']:.8f} | {x['equal_pair_normalized_MSE_minimum']:.10g} |")
    lines+=['',
        '같은 입력을 받는 결정적 예측기는 같은 값을 출력한다. 두 정답을 a, b라고 하면, 두 사례를 같은 비중으로 둔 제곱 오차는 “예측과 중간값 (a+b)/2의 제곱 거리”에 “두 정답 거리 제곱의 1/4”을 더한 값이다. 따라서 중간값에서 최솟값을 얻어도 오차가 남는다. 표는 실제 학습과 같은 위치/31.5, 속도/3 정규화 후 물체·좌표 평균으로 계산했다.','',
        'simulator의 한 단계 결과를 8개 시점의 닫힌 형태 자유 운동식으로 별도 검사했고, 벽·물체 충돌이 없음을 확인했다. 중간값과 다른 예측 32개씩에서 제곱 오차 분해도 검사했다. 이는 세 구성 사례의 증명 보조 검사이며 임의의 실제 평가 자료 전체의 Bayes 오차, 장기 예측 오차 하한, 학습 성능 달성을 뜻하지 않는다.','',
        '전체 환경 정보를 제공하면 두 입력은 달라져 이 모호성이 사라진다. 그래도 학습한 모델이 틀릴 수 있으므로, 정보 부족과 모델의 학습 실패를 별도로 평가한다.']
    (root/'EXPLANATION_KO.md').write_text('\n'.join(lines)+'\n')
    dump(root/'verification.json',{'all_passed':True,'constructed_pairs':3,'independent_free_flight_reference_checks':6,
        'squared_loss_identity_checks':checks,'results':rows,'pairs_sha256':r.sha(root/'pairs.npz'),
        'explanation_sha256':r.sha(root/'EXPLANATION_KO.md'),'script_sha256':r.sha(__file__),
        'simulator_sha256':r.sha(HERE/'dynamics_world.py'),'scope':'Three constructed equal-weight pairs only; not an empirical benchmark-wide Bayes bound.'})
    print('context omission identifiability checked',rows,flush=True)


if __name__=='__main__':main()
