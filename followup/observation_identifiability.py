"""An exact-past ambiguity example under the declared simulator, not a dataset rate."""
import json
import numpy as np
from PIL import Image,ImageDraw
from common import HERE,dump,fresh_dir,r
from dynamics_world import rollout,render


def main():
    root=fresh_dir(HERE/'reports/observation_identifiability_v1')
    velocity=np.array([.7,.4]);initial=np.array([[-15.,-10.,*velocity,4.,1.,1.],[10.,10.,*velocity,5.,3.,1.]])
    actions=np.zeros((19,2,2));actions[3,0]=[.8,-.3];contexts=[];states=[];past=[]
    for drag in [.004,.009]:
        net=drag*velocity;context=np.array([0.,.0205,net[0],net[1]-.0205,drag])
        assert .02<=context[1]<=.06 and -.05<=context[2]<=.05 and -.02<=context[3]<=.02 and 0<=drag<.01
        trajectory,_=rollout(initial,context,actions);contexts.append(context);states.append(trajectory)
        past.append(np.stack([render(s)['image'] for s in trajectory[:4]]))
    np.testing.assert_allclose(states[0][:4],states[1][:4],atol=1e-13,rtol=0);np.testing.assert_array_equal(past[0],past[1])
    target0=states[0][4,:,:4];target1=states[1][4,:,:4]
    expected_difference=actions[3,0]*(np.exp(-contexts[0][4])-np.exp(-contexts[1][4]))
    np.testing.assert_allclose(states[0][4,0,2:4]-states[1][4,0,2:4],expected_difference,atol=1e-13,rtol=0)
    lower_bound=float(np.mean((target0-target1)**2)/4)
    np.savez_compressed(root/'counterexample.npz',initial=initial,contexts=np.stack(contexts),actions=actions,
        trajectories=np.stack(states),observed_images=np.stack(past))
    canvas=Image.new('RGB',(4*192,222),'white');draw=ImageDraw.Draw(canvas)
    for i,frame in enumerate(past[0]):
        canvas.paste(Image.fromarray(frame).resize((192,192),Image.Resampling.NEAREST),(i*192,30));draw.text((i*192+8,10),f't={i}: identical RGB in both worlds',fill='black')
    canvas.save(root/'identical_past.png')
    text='''# 같은 과거 영상으로 다른 미래가 가능한 경우

2026-09-23. 이것은 simulator 안에서 구성한 경계 사례이며, 실제 평가 자료 중 이런 경우의 빈도를 측정한 결과가 아니다.

두 물체가 같은 일정한 속도 v로 움직이고 외부 합력 a가 저항 계수 gamma와 a=gamma*v 관계를 이루면 힘과 저항이 상쇄된다. 매 substep의 속도 갱신은 v*exp(-gamma*h)+a*(1-exp(-gamma*h))/gamma=v이므로, gamma가 달라도 과거 위치와 속도가 같을 수 있다.

gamma=0.004와 0.009인 두 환경을 만들었다. 두 환경의 중력·바람·저항 값은 원래 variable_force의 허용 범위 안에 있다. 초기 상태와 네 장의 RGB, 입력 행동은 같지만 첫 물체에 같은 속도 변화 k를 가하면 그다음 속도는 v+k*exp(-gamma)가 되어 서로 다르다.

따라서 이 두 상황 중 어느 것인지 추가 정보 없이 반드시 맞히는 결정적 영상 예측기는 없다. 입력이 같으면 출력도 같기 때문이다. 두 경우를 같은 확률로 놓고 물체의 다음 x/y/vx/vy 좌표 평균 제곱 오차를 평가한다면, 공통 출력의 최저 평균 오차는 두 정답 차이의 제곱 평균을 4로 나눈 값이다. 이 조건부 두 사례 하한을 전체 데이터셋의 오차 하한으로 확대하지 않는다.

여러 물체의 속도가 충분히 다르거나 관측 중 속도가 변하면 저항과 합력을 구분할 수 있는 정보가 생길 수 있다. 일반적인 비접촉 궤적 모두가 식별 불가능하다는 주장이 아니다. 과거 영상이 완벽하더라도 입력에 정보가 없는 경계와, 모델이 정보를 제대로 사용하지 못하는 경우를 구분하기 위한 예제다.

실제 두 궤적과 입력은 counterexample.npz, 수치와 검증은 verification.json, 같은 과거 네 장은 identical_past.png에 있다. C07의 관측 길이·독립 반복 전체 평가를 대체하지 않는다.
'''
    (root/'EXPLANATION_KO.md').write_text(text)
    dump(root/'verification.json',{'all_passed':True,'past_rgb_identical':True,'past_state_max_difference':float(abs(states[0][:4]-states[1][:4]).max()),
        'next_motion_max_difference':float(abs(target0-target1).max()),'equal_prior_next_motion_mse_lower_bound':lower_bound,
        'closed_form_kick_velocity_difference_verified':True,'contexts_within_declared_variable_force_ranges':True,
        'data_sha256':r.sha(root/'counterexample.npz'),'figure_sha256':r.sha(root/'identical_past.png'),'script_sha256':r.sha(__file__),
        'scope':'Constructed boundary example, not incidence in the sampled benchmark or a general dataset lower bound.'})
    print('Identical past, different next velocity verified; conditional two-case MSE floor',lower_bound)


if __name__=='__main__':main()
