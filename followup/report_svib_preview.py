"""Complete public-preview report, plots, and fixed unselected image examples."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE, dump, r
from svib_preview_data import NAME, ALPHAS, alpha_name


def main():
    root=HERE/f'reports/{NAME}';aggregate=root/'aggregate_v1';summary=json.loads((aggregate/'summary.json').read_text())
    audit=json.loads((aggregate/'verification.json').read_text());assert audit['all_passed'] and audit['summary_sha256']==r.sha(aggregate/'summary.json')
    assert audit['verifier_sha256']==r.sha(HERE/'verify_svib_preview_aggregate.py')
    out=root/'report_v1'
    if out.exists():raise RuntimeError(f'Preserve existing report: {out}')
    out.mkdir();bindings=[];example_records=[]
    def lookup(alpha,method,metric='pixel_mse',category='all',split='heldout'):
        matches=[x for x in summary['conditions'] if (x['alpha'],x['method'],x['metric'],x['category'],x['split'])==(alpha,method,metric,category,split)]
        assert len(matches)==1;return matches[0]
    def bind(x,label):
        item={'label':label,'kind':'paired' if 'left' in x else 'condition','data':x};bindings.append(item);return x
    def mean(alpha,method,**kw):return lookup(alpha,method,**kw)['mean']
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    colors={'identity':'#777777','train_target_mean':'#a66b32','plain':'#2463a6','c4':'#d14836'}
    panels=[('pixel_mse','all','heldout','External test: all 100 scenes'),
            ('changed_pixel_mse','changed','heldout','Changed pixels: 74 changed scenes'),
            ('pixel_mse','unchanged','heldout','No-change test: 26 scenes'),
            ('pixel_mse','all','train','Training: 70 scenes per alpha')]
    for ax,(metric,category,split,title) in zip(axes.flat,panels):
        methods=['plain','c4'] if category=='unchanged' else ['identity','train_target_mean','plain','c4']
        for method in methods:
            stats=[bind(lookup(a,method,metric,category,split),f'plot/{title}/{method}/{a}') for a in ALPHAS]
            y=np.array([x['mean'] for x in stats])*1000;interval=np.array([x['scene_ci95'] for x in stats])*1000
            ax.plot([float(a) for a in ALPHAS],y,'o-',label=method,color=colors[method]);ax.fill_between([float(a) for a in ALPHAS],interval[:,0],interval[:,1],alpha=.10,color=colors[method])
        ax.set(title=title,xlabel='Alpha',ylabel='Pixel MSE x 1000 (lower is better)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,1];paired=[]
    for a in ALPHAS:
        x=next(x for x in summary['paired_conditions'] if (x['alpha'],x['left'],x['right'],x['split'],x['category'],x['metric'])==(a,'c4','plain','heldout','all','pixel_mse'))
        paired.append(bind(x,f'plot/paired/{a}'))
    ax.axhline(0,color='black',lw=1)
    for i,x in enumerate(paired):
        a=float(ALPHAS[i]);ax.plot(a,x['mean']*1000,'o',color=colors['c4']);ax.vlines(a,*np.array(x['scene_ci95'])*1000,color=colors['c4'])
    ax.set(title='Same-scene difference: C4 minus plain',xlabel='Alpha',ylabel='MSE difference x 1000');ax.grid(alpha=.2)
    ax=axes[1,2];costs=summary['costs']
    for kind in ['plain','c4']:
        rows=[[x['training_seconds'] for x in costs if x['alpha']==a and x['method']==kind] for a in ALPHAS]
        ax.plot([float(a) for a in ALPHAS],np.mean(rows,axis=1),'o-',label=kind,color=colors[kind])
        for a,values in zip(ALPHAS,rows):ax.vlines(float(a),min(values),max(values),color=colors[kind],alpha=.5)
    ax.set(title='Actual local training time (shared host)',xlabel='Alpha',ylabel='Seconds for 2,000 steps');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('SVIB public preview: 70 training pairs per alpha, one shared 100-pair external test\nBands: scene bootstrap after averaging fixed initializations; not independent-dataset intervals',fontsize=13)
    fig.savefig(out/'preview_comparison.png',dpi=150);fig.savefig(out/'preview_comparison.svg');plt.close(fig)
    with np.load(HERE/f'data/{NAME}/heldout/inputs.npz') as a:source=a['source_rgb'].copy()
    with np.load(HERE/f'data/{NAME}/heldout/labels.npz') as a:target=a['target_rgb'].copy();changed=a['changed_mask'].any(axis=(1,2))
    indices=np.r_[np.flatnonzero(changed)[:3],np.flatnonzero(~changed)[:2]].tolist()
    for alpha in [ALPHAS[0],ALPHAS[-1]]:
        methods=['train_target_mean','plain','c4'];arrays=[];rows=[];paths=[]
        for method in methods:
            folder=HERE/(f'reports/{NAME}/baselines/{alpha_name(alpha)}/heldout/{method}' if method=='train_target_mean' else f'runs/{NAME}/{alpha_name(alpha)}/{method}_s0/heldout')
            with np.load(folder/'predictions.npz') as a:arrays.append(a['predictions'].copy().transpose(0,2,3,1))
            rows.append(list(csv.DictReader((folder/'rows.csv').open())));paths.append(folder)
        fig,axes=plt.subplots(5,5,figsize=(11,11.7),constrained_layout=True)
        for row,index in enumerate(indices):
            images=[source[index],target[index]]+[x[index] for x in arrays]
            for col,image in enumerate(images):
                ax=axes[row,col];ax.imshow(image);ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
                if row==0:ax.set_title(['Source','Target','Train-target mean','Plain (seed 0)','C4 (seed 0)'][col],fontsize=10)
                if col==0:ax.set_ylabel(f'Scene {index:03d}\n'+('changed' if changed[index] else 'no change'),fontsize=9)
                if col>=2:
                    value=float(rows[col-2][index]['pixel_mse']);ax.set_xlabel(f'MSE {value:.5f}',fontsize=8)
                    example_records.append({'alpha':alpha,'method':methods[col-2],'seed':None if col==2 else 0,'episode':index,'pixel_mse':value,
                         'prediction_path':str((paths[col-2]/'predictions.npz').relative_to(HERE)),
                         'prediction_sha256':r.sha(paths[col-2]/'predictions.npz'),'rows_path':str((paths[col-2]/'rows.csv').relative_to(HERE)),
                         'rows_sha256':r.sha(paths[col-2]/'rows.csv')})
        fig.suptitle(f'Published preview, alpha={alpha}; fixed first 3 changed + first 2 no-change scenes\nFinal checkpoint, initialization 0; no selection by prediction quality',fontsize=12)
        fig.savefig(out/f'examples_{alpha_name(alpha)}.png',dpi=140);plt.close(fig)
    lines=['# SVIB 공개 예제에서의 외부 과제 연결','',
        '공식 dSprites / Single Atomic의 Shape-Swap 공개 예제 500쌍으로 작은 RGB 예측기를 처음부터 학습했다. 24개 모델과 모든 평가 예측·점수의 검증을 마쳤다. 이 실험은 축소 외부 과제 탐색이며, 전체 SVIB 벤치마크 성능을 재현한 결과는 아니다.','',
        '입력은 이전 이미지 한 장이고 출력은 변환 후 이미지다. 정답 물체 속성·마스크·변환 규칙을 입력하지 않았다. 일반 모델과 같은 예측기를 네 회전 방향에 적용하는 C4 모델을 비교했다. 기존 OrbitLab checkpoint의 전이 성능은 측정하지 않았다.','',
        '## 자료와 비교 조건','',
        'alpha별 100쌍을 공식 정렬 분할 방식으로 학습 70·검증 15·ID 평가 15쌍으로 나눴다. 모든 모델은 별도 Test 100쌍을 공유한다. 74쌍은 픽셀이 실제로 바뀌고 26쌍은 그대로 두는 것이 정답이다. 서로 다른 alpha와 초기화를 독립 데이터 반복으로 세지 않는다.','',
        '동일 파라미터 1,266,467개, 같은 초기 가중치·미니배치 순서, 2,000회 업데이트로 고정했다. 방식 2 × alpha 4 × 초기화 3의 마지막 checkpoint를 모두 평가했으며 점수에 따른 선택은 없다. C4는 네 번의 기본 예측을 평균하므로 동일 계산량 비교가 아니다.','',
        '## 외부 100쌍의 픽셀 오차','',
        '표의 단위는 픽셀 MSE × 1,000이며 낮을수록 좋다. 일반/C4는 고정 초기화 세 개의 평균이다. 정답 모양 교환의 의미상 성공률이 아니며, 낮은 픽셀 오차만으로 물체 구조를 이해했다고 주장할 수 없다.','',
        '| alpha | 그대로 두기 | 학습 정답 평균 | 일반 | C4 |','| --- | ---: | ---: | ---: | ---: |']
    for alpha in ALPHAS:
        cells=[bind(lookup(alpha,k),f'table/main/{alpha}/{k}') for k in ['identity','train_target_mean','plain','c4']]
        lines.append('| '+alpha+' | '+' | '.join(f"{x['mean']*1000:.3f}" for x in cells)+' |')
    wins=sum(v<0 for x in paired for v in x['per_seed_mean'])
    means_better=sum(x['mean']<0 for x in paired)
    beats_identity={k:sum(mean(a,k)<mean(a,'identity') for a in ALPHAS) for k in ['plain','c4']}
    lines+=['',f'C4의 전체 픽셀 오차가 일반 모델보다 낮은 것은 alpha 평균 {means_better}/4개, 같은 초기화의 짝 비교 {wins}/12개였다. 그대로 두기 대조보다 낮은 alpha 평균은 일반 {beats_identity["plain"]}/4개, C4 {beats_identity["c4"]}/4개였다. 이 개수는 같은 작은 외부 평가 집합을 재사용한 기술 통계다.','',
        '## 실제 바꿀 영역과 보존할 장면','',
        '| alpha | 바뀐 픽셀 오차: 일반 | 바뀐 픽셀 오차: C4 | 무변화 장면 오차: 일반 | 무변화 장면 오차: C4 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for a in ALPHAS:
        cells=[bind(lookup(a,k,metric='changed_pixel_mse',category='changed'),f'table/changed/{a}/{k}') for k in ['plain','c4']]
        cells += [bind(lookup(a,k,category='unchanged'),f'table/nochange/{a}/{k}') for k in ['plain','c4']]
        lines.append('| '+a+' | '+' | '.join(f"{x['mean']*1000:.3f}" for x in cells)+' |')
    lines+=['','변경 영역은 source와 target의 픽셀이 다른 곳이며 채점에만 사용한다. 변경이 없는 26쌍의 변경 영역 점수는 결측으로 남겼다. 무변화 장면에서 그대로 두기 대조의 오차는 0이다. 원래 유지해야 할 부분을 얼마나 훼손했는지도 모든 장면의 unchanged-pixel 지표로 별도 저장했다.','',
        '## 학습 비용과 회전 일관성','',
        '| alpha | 일반 학습 초 | C4 학습 초 | 일반 100장 CPU 추론 초 | C4 100장 CPU 추론 초 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for a in ALPHAS:
        row=[]
        for key in ['training_seconds','cpu_forward_seconds_100_images']:
            for kind in ['plain','c4']:row.append(float(np.mean([x[key] for x in costs if x['alpha']==a and x['method']==kind])))
        lines.append('| '+a+' | '+' | '.join(f'{x:.3f}' for x in row)+' |')
    rotation=max(x['rotation_maximum_absolute_gap'] for x in costs if x['method']=='c4')
    lines+=['',f'C4 모델의 모든 외부 입력과 세 회전에서 최대 픽셀 일관성 오차는 {rotation:.3e}였다. 이는 출력 회전의 일관성 검사이며 목표 이미지 정확도의 보장이 아니다. 학습은 로컬 MPS, 저장 checkpoint 추론은 CPU에서 기록했다. 다른 연구 작업과 컴퓨터를 공유했으므로 표의 시간은 이 실행의 측정값이며 일반적인 속도 순위가 아니다.','',
        '## 공식 프로토콜과의 차이','',
        '| 항목 | 공식 자료·코드 | 이번 실행 |','| --- | --- | --- |',
        '| 범위 | 여러 도메인·규칙의 12개 과제 | dSprites / Single Atomic 공개 preview 한 과제 |',
        '| 자료 규모 | 과제별 64,000 학습 pool·8,000 test | alpha별 100쌍 pool·공통 test 100쌍 |',
        '| 분할 | 정렬 후 70%/15%/15% | 같은 규칙의 70/15/15쌍 |',
        '| 평가 표본 | 코드 기본 batch/drop_last/max_images 설정 | 마지막 묶음까지 모든 100쌍 |',
        '| MSE | 픽셀·채널 오차 합을 이미지 수로 나눔 | 원소별 MSE와 이미지별 합을 모두 저장; 49,152배 관계 검증 |',
        '| LPIPS | 공식 평가 지표에 포함 | 미측정 |',
        '| 모델 | 논문에서 비교한 모델과 학습 설정 | 작은 잔차 CNN 일반/C4, 고정 2,000 updates |','',
        '학습 source 조합과 test source 조합의 분리는 확인했지만 학습 target의 일부 조합은 test source와 겹친다. source와 target 노출을 따로 기록했으며 이 사실을 픽셀 중복이나 전체 공식 벤치마크의 누수로 확대 해석하지 않는다. 원본 저장 RGB 채널과 픽셀을 보존했다.','',
        '## 검증과 해석 범위','',
        '24개 모델의 4,800개 예측과 7,200개 회전 입력 예측을 재생했다. 모든 픽셀·영역 점수, 1,536개 집계와 1,920개 짝 차이, 학습 초기값·표본 순서·최종 optimizer 상태를 검사했다. 같은 MPS 학습 전체를 다시 실행한 검사는 아니다. 구간은 초기화 평균 후 고유 장면을 재표집한 조건부 bootstrap이며 새로운 데이터셋에 대한 불확실성을 대표하지 않는다.','',
        '첫 기준선 검증에서 평균 연산 순서의 약 10⁻¹⁷ 잔차를 발견했고, 기존 허용 범위로 검사한 v2가 통과했다. 별도 회전 수식 검사에서는 float32 배치 연산의 약 5.6×10⁻⁶ 차이를 float64 수식 검사로 분리했다. 저장된 예측·지표 및 실제 출력 재생·회전 검사의 허용 기준은 바꾸지 않았으며 실패 코드와 로그를 보존했다.','',
        '공개 예제의 작은 학습 자료에서 과적합이 가능하다. 이 결과만으로 자연어 편집, 안정적인 물체 조작, 기존 checkpoint의 외부 전이, 또는 최신 방법 대비 우위를 주장할 수 없다. 전체 공식 자료와 원저자의 비교 모델을 같은 프로토콜로 실행하는 것은 이번 축소 실험에 포함되지 않았다.','',
        '![모든 alpha 비교](report_v1/preview_comparison.png)','',
        '![고정 예제 alpha 0](report_v1/examples_alpha_0p0.png)','',
        '![고정 예제 alpha 0.6](report_v1/examples_alpha_0p6.png)','',
        '[공식 프로젝트](https://systematic-visual-imagination.github.io/) · [실행 계획](../../planning_B06_SVIB_PREVIEW_KO.md) · [자료 검증](data_verification.json) · [모델 검증](model_verification.json) · [전체 집계](aggregate_v1/summary.json) · [집계 검증](aggregate_v1/verification.json)']
    report=root/'RESULTS_KO.md';report.write_text('\n'.join(lines)+'\n')
    dump(out/'manifest.json',{'summary_sha256':r.sha(aggregate/'summary.json'),'aggregate_verification_sha256':r.sha(aggregate/'verification.json'),
        'source_sha256':r.sha(__file__),'report_sha256':r.sha(report),'bindings':bindings,'cost_records':costs,
        'fixed_example_selection':{'changed':indices[:3],'unchanged':indices[3:],'alphas':[ALPHAS[0],ALPHAS[-1]],'seed':0},
        'examples':example_records,'all_example_input_sha256':r.sha(HERE/f'data/{NAME}/heldout/inputs.npz'),
        'all_example_target_sha256':r.sha(HERE/f'data/{NAME}/heldout/labels.npz'),
        'derived_counts':{'c4_better_alpha_means':means_better,'c4_better_seed_pairs':wins,'model_better_than_identity':beats_identity},
        'files':{p.name:r.sha(p) for p in out.iterdir() if p.is_file()}})
    print('SVIB report created',len(bindings),'bound statistics,',len(example_records),'example prediction cells',flush=True)


if __name__=='__main__':main()
