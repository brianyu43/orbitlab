"""Report only; never train, alter gates or mark a figure visually verified."""
import json,os,time
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(__file__).parent/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from v3_common import HERE,sha,dump
from r4_data import BASE,DEVELOPMENT
from r4_train import root,data
from r4_evaluate import EVAL,CANDIDATES,SPLITS

COLORS=['#64748b','#2563eb','#14b8a6','#e76f51']
LABELS=['Standard','Area / edge','Spatial map','Coordinates']

def save(fig,name):
    fig.savefig(BASE/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(BASE/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def main():
    agg=json.loads((BASE/'aggregate.json').read_text());verify=json.loads((BASE/'evaluation_verification.json').read_text());assert verify['aggregate_sha256']==sha(BASE/'aggregate.json')
    training=json.loads((BASE/'training_verification.json').read_text());assert training['total_updates']==100000
    recon=agg['reconstruction'];gen=agg['generation'];plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,2,figsize=(10,10),sharey=True)
    for row,role in enumerate(['oracle','learned','bridge']):
        for col,split in enumerate(SPLITS):
            ax=axes[row,col];values=[100*recon[f'{role}_{c}'][split]['small_arrow']['strict_accepted_and_joint'] for c in CANDIDATES]
            bars=ax.bar(LABELS,values,color=COLORS);ax.axhline(80,color='black',ls='--',lw=1)
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9);ax.set_ylim(0,113);ax.set_ylabel('Small-arrow strict success (%)');ax.tick_params(axis='x',rotation=15)
            counts=recon[f'{role}_standard'][split]['small_arrow']['base_scenes'];ax.set_title(f'{role.title()} input | {split} | {counts} base scenes')
    fig.suptitle('R4 reconstruction: true factors, actual image code, and supervised bridge\nOnly the learned-input row counts toward the primary gate',y=1.01);fig.tight_layout();save(fig,'reconstruction_results')
    fig,axes=plt.subplots(2,2,figsize=(11,8));ax=axes[0,0];x=np.arange(4)
    for off,split,color in [(-.18,'seen','#2563eb'),(.18,'ood','#e76f51')]:
        bars=ax.bar(x+off,[100*gen[c]['splits'][split]['strict_accepted_and_joint'] for c in CANDIDATES],width=.36,color=color,label=split);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
    ax.set_xticks(x,LABELS,rotation=15);ax.set_ylim(0,110);ax.set_ylabel('Strict success (%)');ax.legend();ax.set_title('Generated samples: 128 per condition')
    ax=axes[0,1];values=np.array([[gen[c]['full_diversity']['aggregate'][a]['tv'] for a in ['cx','cy','scale','orientation']] for c in CANDIDATES]);im=ax.imshow(values,vmin=0,vmax=1,cmap='magma_r');ax.set_xticks(range(4),['Center x','Center y','Scale','Orientation']);ax.set_yticks(range(4),LABELS)
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]>.55 else 'black')
    ax.set_title('All-image diversity: TV (gate <= 0.20)');fig.colorbar(im,ax=ax,fraction=.045)
    ax=axes[1,0];mins=[min(v['n'] for v in gen[c]['strict_diversity']['conditions']) for c in CANDIDATES];bars=ax.bar(LABELS,mins,color=COLORS);ax.bar_label(bars,padding=3);ax.axhline(32,color='black',ls='--',label='Required in every condition');ax.set_ylim(0,140);ax.tick_params(axis='x',rotation=15);ax.set_ylabel('Minimum accepted count across 24 conditions');ax.legend(fontsize=8);ax.set_title('Strict-subset diversity needs enough samples')
    ax=axes[1,1]
    for i,c in enumerate(CANDIDATES):ax.plot(range(6),[100*gen[c]['by_condition'][f'2,{color}']['strict_accepted_and_joint'] for color in range(6)],marker='o',label=LABELS[i],color=COLORS[i])
    ax.set_xticks(range(6));ax.set_ylim(0,105);ax.set_ylabel('Strict success (%)');ax.set_xlabel('Requested arrow color index (2 is held-out pair)');ax.set_title('Arrow generation by requested color');ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'generation_results')
    # Four clearly labelled baseline failures, paired across every candidate.
    selected=[];fig,axes=plt.subplots(4,5,figsize=(11,9))
    for sidx,split in enumerate(SPLITS):
        rows=json.loads((EVAL/'reconstruction/learned_standard'/split/'rows.json').read_text());eligible=[(i,v) for i,v in enumerate(rows) if v['rotation']==0 and v['kind']==2 and v['true_size_bin']==0]
        chosen=sorted(eligible,key=lambda iv:iv[1]['foreground_mae'],reverse=True)[:2];truth,_,_=data(DEVELOPMENT,split)
        predictions={c:torch.load(EVAL/'reconstruction'/f'learned_{c}'/split/'images.pt',map_location='cpu',weights_only=True) for c in CANDIDATES}
        for j,(idx,row) in enumerate(chosen):
            selected.append({'split':split,'row':idx,'base_id':row['base_id'],'criterion':'Two largest standard-decoder foreground errors among rotation0true-small-arrow scenes in each split; display only'})
            rr=2*sidx+j
            for col,image in enumerate([truth[idx],*[predictions[c][idx] for c in CANDIDATES]]):
                axes[rr,col].imshow(image.permute(1,2,0).clamp(0,1));axes[rr,col].set_xticks([]);axes[rr,col].set_yticks([])
                if rr==0:axes[rr,col].set_title(['True input',*LABELS][col])
                if col==0:axes[rr,col].set_ylabel(f'{split}\nradius {row["true_radius"]:.3f}')
    fig.suptitle('Selected small-arrow reconstruction failures | actual image encoder input');fig.tight_layout();save(fig,'failure_examples');dump(BASE/'failure_examples.json',selected)
    rows=[]
    for c,label in zip(CANDIDATES,LABELS):
        values=[]
        for role in ['oracle','learned','bridge']:values.append('/'.join(f'{100*recon[f"{role}_{c}"][s]["small_arrow"]["strict_accepted_and_joint"]:.1f}%' for s in SPLITS))
        rows.append(f'| {label} | '+' | '.join(values)+' |')
    grows=[]
    for c,label in zip(CANDIDATES,LABELS):
        g=gen[c];grows.append(f'| {label} | {100*g["splits"]["seen"]["strict_accepted_and_joint"]:.1f}% | {100*g["splits"]["ood"]["strict_accepted_and_joint"]:.1f}% | {g["full_diversity"]["passed"]} | {g["strict_diversity"]["passed"]} |')
    gate_rows=[]
    for v in agg['candidate_gates']:gate_rows.append(f'| {v["candidate"]} | {100*v["small_arrow_strict_min"]:.1f}% | {100*v["balanced_strict_gain"]:+.1f}%p | {v["passed"]} |')
    components=training['components'];seconds=sum(v['seconds'] for v in components.values());cost='\n'.join(f'| {c} | {v["parameters"]:,} | {v["steps"]:,} | {v["seconds"]:.1f} |' for c,v in components.items())
    outcome='개발 기준을 통과한 후보: '+str(agg['selected']) if agg['selected'] else '네 방식의 개발 비교를 마쳤지만 모든 진행 기준을 통과한 개선 후보는 없다.'
    text=f'''# R4: 작은 화살표 복원과 생성 다양성

{outcome} 이 보고서는 개발 seed889101·초기화0의 결과다. 새로운 세 데이터×세 초기화 확인 결과와 구분한다. 테스트 split은 평가하지 않았다.

## 비교와 추가 정보

현재 방식과 area/edge 방식은 동일한 고정 인코더·디코더 구조·초기 가중치·학습 표본 순서·6,000업데이트를 사용하며 이미지 손실만 다르다. Spatial은 인코더와 디코더를 함께 바꾸고1×4×4배치를 유지하며 area/edge 손실을 쓴다. Coordinate는 기준 인코더를 공유하고 학습한 RGBA패치와 중심을 명시적으로 배치한다. 이 방식에는 정답 중심 감독이 추가된다. 따라서 네 방식 전체를 파라미터 수와 감독 정보가 같은 순수 구조 비교라고 해석하지 않는다.

정답 속성 역할은 실제 renderer의 모양·색·중심·반경·방향을 제공한다. 실제 이미지 역할과 **별도 가중치의 디코더**를 학습하므로 두 점수 차이를 고정된 디코더의 단순 인코더 교체로 해석하지 않는다. Bridge는 고정 인코더 코드에서 정답 속성 코드를 예측하도록 추가 감독한 작은 판독기로, 같은 oracle 디코더에 넣는 진단이다. 이 진단은 주 모델의 무감독 성능이나 후보 선정에 포함하지 않는다.

## 실제 작은 화살표의 복원

작음은 실제 renderer 반경0.13–0.24의 아래1/3이다. 각 원본 장면의 네 회전은 같은 표본이며 독립 장면 네 개로 세지 않는다. 값은 val/OOD이고, 엄격한 모양 기준과 요청된 모양·색을 모두 통과한 비율이다.

| 방식 | 정답 속성 | 실제 이미지 인코더 | 추가 감독 bridge |
|---|---:|---:|---:|
{chr(10).join(rows)}

![Reconstruction](reconstruction_results.png)

## 생성과 다양성

조건별128개, 총3,072개 표본을 같은 Gaussian noise로 짝맞췄다. 기존·area/edge·coordinate는 같은 flow와 인코더 코드를 공유하며, spatial만 별도 flow를 학습했다.64단계 Euler와 guidance1을 사용했다. 생성물의 중심·크기·방향은 고정 판독기의 추정치다.

| 방식 | Seen strict | OOD strict | 전체 다양성 통과 | Strict 부분집합 다양성 통과 |
|---|---:|---:|---|---|
{chr(10).join(grows)}

참조는 조건별384개이며 모든9,216개 깨끗한 이미지의 strict 성공은{100*agg['reference']['strict_accepted_and_joint']:.1f}%다. 생성128개와 참조384개 히스토그램은 각각의 분모로 정규화한다. 기존 구간·coverage/TV/outside 기준과 조건별 최소32개 strict 표본 기준은 유지했다. 성공 표본만 골라 다양성이 좋아졌다고 보고하지 않는다.

![Generation](generation_results.png)

## 사전 고정한 진행 기준

실제 이미지 인코더의 작은 화살표 복원이 val/OOD각80% 이상, seen/OOD동일 가중 평균 생성 strict가 기준보다10%p 이상 향상, 전체·strict부분집합 다양성 모두 통과해야 한다. 정답 속성이나 bridge 성능으로 이 조건을 대신하지 않는다.

| 후보 | 작은 화살표 낮은 쪽 | 생성 strict 변화 | 전체 gate |
|---|---:|---:|---|
{chr(10).join(gate_rows)}

통과가 없으면3×3확인을 자동 확대하지 않는다. 실패를 포함한 결과이며 생성 문제 해결 또는 독립 사람 검증 완료를 뜻하지 않는다.

![Selected failures](failure_examples.png)

그림은 split별 기준 모델의 foreground오차가 큰 작은 화살표 두 장을 선택한 실패 예시다. 전체 성공률은 예시 선택과 무관하게 모든 장면에서 계산했다.

## 비용·검증·재현

14개 학습 구성 요소, 본 학습100,000업데이트,100회씩의 별도 예비 학습1,400업데이트를 실행했다. 본 학습 시간 합은{seconds:.1f}초다. 이는 각 프로세스의 학습 구간 벽시계 시간 합이며 데이터 준비·평가·대기 시간이나 순수 CPU시간과 동일하지 않다. 검증과 학습이 일부 겹쳤다. 기준/공간 인코더 학습에 쓰인 decoder는 이후 분리 비교에서 교체되며, 그 선행 학습 비용도 아래 표에 포함했다.

| 구성 요소 | 파라미터 | 업데이트 | 학습 초 |
|---|---:|---:|---:|
{cost}

24,576개 모델 복원 이미지와12,288개 생성 이미지, 모든 지표 행, 두 flow의6,144개 latent와각64적분 단계를 재생했다. 또한 clean입력2,048개, 참조9,216개와 모든 캐시 인코더 코드를 검증했다. 동일 구현을 재실행한 검증이며, 독립 구현·전체 재학습 또는 사람의 의미 품질 평가가 아니다.

원본 파일은 읽기 전용으로 재사용하며 새로운 파일은research_v3아래에 있다. 실행 순서: r4_data.py및--verify → r4_preflight.py → r4_driver.py → r4_verify.py metrics → r4_evaluation_driver.py → r4_report.py. 모델/자료/평가 코드는 잠긴SHA256과 다르면 중단한다. 실제 논문 수준의 일반화는 이 단일 개발 실행만으로 주장하지 않는다.
'''
    (BASE/'RESULTS_KO.md').write_text(text)
    figures={p.name:sha(p) for name in ['reconstruction_results','generation_results','failure_examples'] for ext in ['png','pdf'] if (p:=BASE/f'{name}.{ext}').exists()}
    dump(BASE/'report_receipt.json',{'source_sha256':sha(__file__),'aggregate_sha256':sha(BASE/'aggregate.json'),'report_sha256':sha(BASE/'RESULTS_KO.md'),'figures':figures,'visual_review':'pending','generated_unix':time.time()})
    print('R4 report generated; actual visual review still required',flush=True)

if __name__=='__main__':main()
