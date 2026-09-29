"""Report all locked R2 comparisons, separating state and full-image editing."""
import os,json
import numpy as np
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import r2_core as c
import r2_world as w
import r2_decoder_core as d
from v3_common import dump,sha,status

def main():
    reader=json.loads((w.BASE/'reader_aggregate.json').read_text());rv=json.loads((w.BASE/'reader_verification.json').read_text());dv=json.loads((d.BASE/'verification.json').read_text())
    assert rv['units']==32 and dv['verified_units']==36 and dv['all_rgb_predictions_replayed']==51200
    for p in [w.BASE/'data_verification.json',w.BASE/'legacy_palette/verification.json',w.BASE/'diagnosis/verification.json',d.BASE/'data_verification.json']:assert p.exists()
    diag=json.loads((w.BASE/'diagnosis/aggregate.json').read_text())['results'];pixel=json.loads((d.BASE/'aggregate.json').read_text())['conditions'];choice=reader['selection'];assert not choice['development_gate_passed']
    candidate=choice['chosen']['candidate'];baseline=choice['chosen']['baseline'];counts=[2,3,4,6];occlusions=[0.,.25,.5,.75]
    def state(arm,n,group='all'):return next(r for r in reader['development'] if r['arm']==arm.removesuffix('_repeat_current') and r['repeat_current']==arm.endswith('_repeat_current') and r['count']==n)['metrics'][group]
    def diagnosis(arm,split,n):return next(r for r in diag if r['arm']==arm and r['split']==split and r['count']==n)['metrics']['all']
    def pix(arm,n,identity=False):return next(r for r in pixel if r['arm']==arm and r['count']==n and r['identity']==identity)['metrics']['all']
    short=lambda a:a.replace('detector','det').replace('recurrent','rec').replace('_mask',' m')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for arm in [baseline,candidate,candidate+'_repeat_current']:
        vals=[100*np.mean([state(arm,n,f'occlusion_{v}')['joint_edit_success'] for n in counts]) for v in occlusions]
        axes[0,0].plot(np.array(occlusions)*100,vals,'o-',label=short(arm).replace('_repeat_current',' repeat'))
    axes[0,0].set(xlabel='Requested current-target occlusion (%)',ylabel='State + analytic edit success (%)',title='A. Strict state errors dominate');axes[0,0].legend(fontsize=8)
    x=np.arange(6);unmatched=[100*diagnosis(a,'val',6)['unmatched_fraction'] for a in c.ARMS];far=[100*diagnosis(a,'val',6)['mislocalized_fraction'] for a in c.ARMS]
    axes[0,1].bar(x,unmatched,label='Too few present slots');axes[0,1].bar(x,far,bottom=unmatched,label='Matched center >4px');axes[0,1].set(xticks=x,xticklabels=[short(a) for a in c.ARMS],ylabel='Missed true objects (%)',title='B. Six-object failure decomposition',ylim=(0,100));axes[0,1].tick_params(axis='x',rotation=25);axes[0,1].legend(fontsize=8)
    for off,split,color in [(-.18,'train','#297a99'),(.18,'val','#d7813d')]:
        axes[1,0].bar(x+off,[100*diagnosis(a,split,2)['accuracy_given_localized_rgb'] for a in c.ARMS],width=.36,label=split,color=color)
    axes[1,0].set(xticks=x,xticklabels=[short(a) for a in c.ARMS],ylabel='RGB within20/255 per channel (%)',title='C. Color accuracy given center <=4px',ylim=(0,100));axes[1,0].tick_params(axis='x',rotation=25);axes[1,0].legend(fontsize=8)
    for arm in [baseline,'slot_mask0',candidate]:axes[1,1].plot(counts,[diagnosis(arm,'val',n)['mean_predicted_count'] for n in counts],'o-',label=short(arm))
    axes[1,1].plot(counts,counts,'k--',label='True count');axes[1,1].set(xlabel='True object count',ylabel='Mean predicted count',title='D. Training count is two',xticks=counts);axes[1,1].legend(fontsize=8)
    fig.suptitle('R2 development: six supervised RGB readers | one data seed, one initialization',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(w.BASE/f'reader_results.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');plotted=['copy_input','truth_state',baseline,candidate]
    for arm in plotted:
        axes[0,0].plot(counts,[100*pix(arm,n)['pixel_gate'] for n in counts],'o-',label=short(arm))
        axes[0,1].plot(counts,[pix(arm,n)['changed_mae'] for n in counts],'o-',label=short(arm))
        axes[1,0].plot(counts,[pix(arm,n)['preserved_mae'] for n in counts],'o-',label=short(arm))
    for arm in ['truth_state',baseline,candidate]:axes[1,1].plot(counts,[100*pix(arm,n)['joint_state_pixel_gate'] for n in counts],'o-',label=short(arm))
    for ax,title,y in zip(axes.flat,['A. Pixel-only tolerance gate','B. Changed-pixel error','C. Unchanged-pixel damage','D. Both state and pixel gates'],['Pixel success (%)','Changed MAE (0..1)','Preserved MAE (0..1)','Joint success (%)']):
        ax.set(title=title,xlabel='True object count',ylabel=y,xticks=counts);ax.legend(fontsize=8)
    fig.suptitle('R2 neural editing: fixed source background and occluder | truth-state control vs inferred state',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(w.BASE/f'decoder_results.{ext}',dpi=170)
    plt.close(fig)
    # Display selected difficult cases, not an estimate of population quality.
    selections=[];fig,axes=plt.subplots(6,5,figsize=(11,12),layout='constrained')
    for rownum,(n,occ) in enumerate((n,occ) for n in [2,6] for occ in [0.,.5,.75]):
        folder=d.BASE/f'evaluation_v2/{candidate}/n{n}';rows=json.loads((folder/'rows.json').read_text());pool=[(i,r) for i,r in enumerate(rows) if r['op']=='translate' and r['requested_occlusion']==occ];idx,r=max(pool,key=lambda z:(z[1]['changed_mae'],-z[0]))
        sid=r['source_index'];source=np.load(w.BASE/f'data/d{w.DEVELOPMENT}/val/n{n}/rgb.npz')['images'][sid,0];target=np.load(d.BASE/f'data/d{w.DEVELOPMENT}/val/n{n}/targets.npz')['targets'][sid*4+3]
        images=[source,target,*[np.load(d.BASE/f'evaluation_v2/{a}/n{n}/predictions.npz')['images'][idx] for a in ['truth_state',baseline,candidate]]]
        for ax,image in zip(axes[rownum],images):ax.imshow(image);ax.set_xticks([]);ax.set_yticks([])
        axes[rownum,0].set_ylabel(f'n={n}, occ={occ:.0%}\nscene{sid}',fontsize=9)
        selections.append({'count':n,'occlusion':occ,'source_index':sid,'op':'translate','selection':'Largest recurrent-model changed-region error within this count/occlusion/operation; lowest index breaks ties','changed_mae':r['changed_mae']})
    for ax,title in zip(axes[0],['Source','True edited image','NN + true state','NN + detector state','NN + recurrent state']):ax.set_title(title,fontsize=10)
    fig.suptitle('Selected translation failures; display selection only, no model tuning',fontsize=12);fig.savefig(w.BASE/'failure_examples.png',dpi=160);plt.close(fig);dump(w.BASE/'failure_examples.json',selections)
    rtable=[]
    for arm in c.ARMS:
        rtable.append(f"| {arm} | {100*state(arm,2)['joint_edit_success']:.2f}% | {100*state(arm,6)['joint_edit_success']:.2f}% | {diagnosis(arm,'val',6)['mean_predicted_count']:.2f} | {100*state(arm,6)['missing_fraction']:.2f}% |")
    ptable=[]
    for arm in plotted:
        m=pix(arm,2);n=pix(arm,6)
        joint='해당 없음' if arm=='copy_input' else f"{100*m['joint_state_pixel_gate']:.2f}% / {100*n['joint_state_pixel_gate']:.2f}%"
        ptable.append(f"| {arm} | {100*m['pixel_gate']:.2f}% / {100*n['pixel_gate']:.2f}% | {m['changed_mae']:.4f} / {n['changed_mae']:.4f} | {m['preserved_mae']:.4f} / {n['preserved_mae']:.4f} | {joint} |")
    legacy=[]
    for n in counts:
        rows=json.loads((w.BASE/f'legacy_palette/n{n}/rows.json').read_text());compatible=[r for r in rows if not r['same_color'] and not r['continuous_color'] and not r['textured_background'] and not r['requested_occlusion']]
        legacy.append({'count':n,'all_edit_success':float(np.mean([r['joint_edit_success'] for r in rows])),'compatible_visual_scenes':len(compatible),'compatible_visual_edit_success':float(np.mean([r['joint_edit_success'] for r in compatible])),'capacity_exceeded':n>4})
    temporal=[]
    for arm in ['recurrent_mask0','recurrent_mask1']:
        temporal.append({'arm':arm,**{name:float(np.mean([state(arm+suffix,n,'occlusion_0.5')['joint_edit_success'] for n in counts])) for name,suffix in [('past_and_current',''),('repeated_current','_repeat_current')]}})
    calibration=[]
    for arm in c.ARMS:
        cal=json.loads((w.BASE/f'evaluation/d{w.DEVELOPMENT}_s0/{arm}/calibration.json').read_text())
        calibration.append({'arm':arm,'position_half_width_pixels':cal['position_half_width'],'rgb_half_width_255':cal['rgb_half_width'],'count2_joint_coverage':state(arm,2)['predictive_set_coverage'],'count6_joint_coverage':state(arm,6)['predictive_set_coverage'],'count2_mean_categorical_set_size':state(arm,2)['mean_categorical_set_size']})
    runs=[json.loads(p.read_text()) for p in (w.BASE/'runs').glob('d*/*/run.json')]+[json.loads(p.read_text()) for p in (d.BASE/'runs').glob('*/run.json')]
    assert len(runs)==8 and sum(r['steps'] for r in runs)==30000
    costs={'trained_components':8,'optimizer_updates':30000,'summed_training_seconds':sum(r['seconds'] for r in runs),'pilot_updates_current_protocols':800,'old_preflight_probe_updates_separate':100,'cpu_threads_per_job':2,'stage_disk_bytes':sum(p.stat().st_size for p in w.BASE.rglob('*') if p.is_file()),'reused_legacy_checkpoint_not_counted_as_new_training':True}
    dump(w.BASE/'supplement.json',{'legacy_palette':legacy,'temporal_ablation':temporal,'calibrated_predictive_sets':calibration,'costs':costs})
    gain=choice['chosen']['edit_gain']*100;miss=choice['chosen']['count6_missing']*100
    body=f'''# R2 결과 — 물체를 찾는 것과 정확히 다시 그리는 것은 다른 문제다

**이번 개발 실험은 다중 프레임 후보의 확대 기준을 통과하지 못했다.** 알려진 색상표를 없애자 검출 모델은 색 복원에서, 슬롯 모델은 물체 수 증가에서 큰 오류를 보였다. 학습형 영상 편집기는 정답 상태와 추정 상태를 분리해 평가했으며, 아래 실패를 성공으로 바꾸어 해석하지 않는다.

## 자료와 정보 예산

- 새 seed887101, 학습1,024/보정128/개발평가2,048클립. 각 클립은 현재,t−1,t−2,t−3의4프레임이다. 미래 영상은 입력하지 않았다.
- 서로 같은/다른 색, 알려진 팔레트/연속RGB, 검정/텍스처 배경, 현재 대상0/25/50/75%가림을 모두 교차했다. 평가는 물체2/3/4/6개에서 각512장면이다.
- 각 시각 조건에는 서로 다른 장면 가족이 배정됐다. 가림 수준 사이의 차이는 동일 장면의 가림만 바꾼 완전한 짝 비교가 아니다.
- 가장 오래된 영상에서는 가림막이 없고 과거일수록 잘 보이도록 만들었다. 다중 프레임에 유리한 관측 조건이며 영구적으로 가려진 물체를 복구했다고 주장하지 않는다.
- 검출+crop, 정적6슬롯, 순환6슬롯 × 마스크 지도 유무의6개 모델, 각각4,000업데이트다. 모든 모델은 정답 물체 상태로 지도학습했다. 마스크를 주지 않은 모델도 비지도학습은 아니다.
- 검출기는 좌표 정답에서 만든 중심 heatmap도 학습했다. 검출기와 슬롯 구조의 정보·연산 차이가 있으므로 구조 하나만의 인과 효과로 해석하지 않는다.

## 상태를 읽고 해석적 명령을 적용한 결과

아래 편집 성공은 클릭 대상 선택, 물체 수, 모양/방향/크기/좌표/색 및 비대상 상태가 모두 맞아야 한다. 이미지를 잘 그렸다는 점수와 구별한다.

| 모델 | 2개 편집 성공 | 6개 편집 성공 | 6개에서 평균 출력 수 | 6개 누락률 |
|---|---:|---:|---:|---:|
{chr(10).join(rtable)}

사전 규칙으로 선택된 `{candidate}`는 동일한 마스크 감독을 받은 `{baseline}`보다50%가림 조건에서 **{gain:.2f}%p** 개선했다. 필요한15%p에 못 미쳤고,6개 물체 누락률도 **{miss:.2f}%**로5%목표를 넘었다. 따라서3개 새 데이터×3개 초기화 확인은 실행하지 않았다. 유리한 seed만 남겨 평균을 만들지 않았다.

누락은 단순히 출력 개수가 적은 경우와, 배정된 중심이 실제 물체에서4px를 넘게 벗어난 경우로 나눴다. 슬롯 모델은2개 학습에서 배운 출력 수를6개로 확장하지 못했다. 검출기는 더 많은 중심을 찾지만 색 판독이 약했다. 학습 장면에서도 RGB허용오차 통과율이 낮아, 모든 실패를 새로운 장면 일반화나 가림의 정보 부족으로 설명할 수 없다. 이는 추가 진단이며 손실 가중치나 평가 기준을 결과에 맞춰 바꾸지는 않았다.

같은 현재 프레임을4번 반복한 순환 모델도 별도 평가했다. 실제 과거 입력과의 비교는 `supplement.json`의 `temporal_ablation`에 있다. 추가 관측과 순환 구조의 효과를 섞어 성공이라고 주장하지 않는다.

![Reader diagnosis](reader_results.png)

## 학습한 신경망으로 전체 영상 편집

먼저 상태에서33×33 RGBA패치를 만드는 신경 decoder를2,000업데이트 학습했다. 좌표를 이용한 패치 배치·합성은 명시적인 기하학적 사전 지식이다. 그 다음 현재RGB, 학습된 원래/편집 전경, 클릭과 명령으로 전체 영상을 편집하는 작은 신경망을4,000업데이트 학습했다.

학습은 정답 상태로 했고, 평가에서는 동일한 편집기에 정답 상태 또는 각 인식기의 추정 상태를 넣었다. 추정 상태 조건에서는 정답 배경·가림막·물체ID·물체 수를 주지 않았다. 정답 상태에서 추정 상태로 바뀌는 분포 차이도 이 비교의 일부이며, 인식기까지 함께 미세조정한 결과가 아니다.

정답 영상을 만들 때 **원래 배경과 가림막을 고정**하고 대상 물체만 바꿨다. 가림막의 부분 투명 경계 픽셀에서는 물체의 보이는 부분이 바뀌므로 혼합 RGB가 달라질 수 있지만, 완전히 불투명한 패널 픽셀은 정확히 같다.

픽셀 성공 기준은 바뀐 영역MAE≤0.05, 보존 영역≤0.01, 비대상 영역≤0.02, 패널 영역≤0.01이다. 값은0–1색상 범위다. 이 허용오차를 만족하는 것만으로 모양·물체 상태가 옳다고 가정하지 않고 **상태 기준까지 동시에 통과한 점수**를 따로 제시한다. 비대상 영역 오차는 올바른 목표 영상과 비교하므로, 정당한 물체 겹침 변화 자체를 실패로 세지 않는다.

| 입력 상태 | 픽셀 기준 성공:2개/6개 | 바뀐 영역MAE:2개/6개 | 보존 영역MAE:2개/6개 | 상태+픽셀 성공:2개/6개 |
|---|---:|---:|---:|---:|
{chr(10).join(ptable)}

신경 편집기는 원본RGB도 보기 때문에 부정확한 상태를 영상 단서로 보완할 수 있다. 상태+픽셀 동시 점수는 내부 상태와 출력을 함께 요구하는 기준이며, 최종 영상의 의미 정확도를 독립적으로 판독한 점수는 아니다.

입력 복사는 보존 영역 오차가0이어도 명령을 수행하지 않는다. 이 기준을 유지해 배경만 잘 보존한 모델이 좋은 편집기로 오해되지 않도록 했다. 원래 전경 복원과 편집 전경 복원, 패널·배경·비대상 오차, 정답 상태의 identity명령도 모두 원자료에 있다.

![Pixel editing](decoder_results.png)

![Selected failure cases](failure_examples.png)

위 사례는 각 물체 수·가림 조건의 평행이동에서 순환 모델의 바뀐 영역 오차가 큰 예를 골랐다. 대표 표본이나 사람 품질 평가가 아니며 학습/선택에 사용하지 않았다.

## 기존 팔레트 모델과 불확실성

예전4슬롯 강도crop모델을 그대로 재사용했다. 전체 새 조건과, 다른 팔레트 색·검정 배경·무가림의 익숙한 시각 조건을 구분했다. 조건별 표본은16개에 불과하고6개 물체는 원래4슬롯 용량을 넘는다. 연속색/같은색/텍스처까지 포함한 상한 모델로 취급하지 않는다. 상세 값은 `supplement.json`에 있다.

형태·방향·크기는 누적확률0.9집합, 위치·색은 별도128장면 보정 오차의95%분위수 구간을 사용했다. 누락 물체는 포함 실패로 센다. 집합 크기·구간 폭과 포함률을 함께 보존했으며 OOD보장 또는 숨은 상태의 정확한 식별을 주장하지 않는다.

## 검증과 비용

- 원 데이터3,200클립/12,800RGB프레임, 현재 마스크와 라벨을 전부 재생했다.
- 인식 평가32조건/16,384장면과8개 보정 추론을 재실행하고 모든 상태 점수·선택 기준을 재계산했다.
- 추가 진단38조건, 학습 장면6,144추론 및22,528분해 행을 다시 확인했다. 예전 팔레트 모델도2,048장면 전체 추론을 재생했다.
- 전체 편집 정답12,288영상을 원래 고정 패널로 재생했다. decoder36조건/51,200RGB출력과 모든 전경·픽셀·상태 행을 다시 계산했다.
- 새 학습 모델8개, 합계30,000업데이트, 학습 시간 합계{costs['summed_training_seconds']:.1f}초. 현재 프로토콜의100-step측정800업데이트와 이전 reshape사전검사의100업데이트는 성능 학습과 별개다. 작업이 겹친 시간이 있어 하드웨어 벤치마크로 해석하지 않는다.
- 평가 중 반복 압축 해제 병목을 고쳤다. 이전 소스·출력은 보존했고, 기존1,536개 복사 기준 예측·평가 행과 수치가 정확히 같음을 확인했다. 모델·정답·기준은 바꾸지 않았다.

이번 단계는 실패와 원인 분리를 포함한 개발 연구다. 독립 재학습/독립 사람 평가나 실제 사진 일반화는 아니다. 다음은R4작은 모양 생성과R5외부·통합 검증이며 연구 전체는 아직 미완료다.

자료: [원자료 검사](data_verification.json), [인식 재생](reader_verification.json), [선택 판정](selection.json), [진단 검증](diagnosis/verification.json), [decoder프로토콜](decoder/protocol.json), [decoder평가](decoder/evaluation_protocol_v2.json), [decoder재생](decoder/verification.json), [추가 수치·비용](supplement.json).
'''
    (w.BASE/'RESULTS_KO.md').write_text(body)
    dump(w.BASE/'report_receipt.json',{'source_sha256':sha(__file__),'report_sha256':sha(w.BASE/'RESULTS_KO.md'),'figures':{name:sha(w.BASE/name) for name in ['reader_results.png','reader_results.pdf','decoder_results.png','decoder_results.pdf','failure_examples.png']},'visual_review':'pending','development_gate_passed':False})
    status('R2','all_replays_passed_report_visual_review_pending',development_gate_passed=False,report='r2/RESULTS_KO.md')
    print('R2 report and figures generated; visual review required',flush=True)

if __name__=='__main__':main()
