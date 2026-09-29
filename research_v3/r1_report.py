"""Source-backed R1 report and scientific comparison figure."""
import json,os
import numpy as np
import r1_core as c
from v3_common import dump,sha
os.environ.setdefault('MPLCONFIGDIR',str(c.HERE/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    a=json.loads((c.BASE/'aggregate.json').read_text());v=json.loads((c.BASE/'verification.json').read_text())
    assert v['verified_units']==v['expected_units']
    rows=a['development'];grid={(r['arm'],r['split'],r['renderer']):r['metrics'] for r in rows}
    labels=['Original PIL4','Symmetric PIL4','PIL8 BOX']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x=np.arange(3)
    for off,key,color,label in [(-.18,'strict_edit','#e88462','Full orientation labels'),(.18,'quotient_edit','#187c80','Geometric equivalence')]:
        values=[np.mean([grid['alpha_full',s,r][key] for s in ['val','ood']])*100 for r in c.RENDERERS]
        bars=axes[0].bar(x+off,values,.36,color=color,label=label)
        axes[0].bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    axes[0].set(xticks=x,xticklabels=labels,ylim=(0,112),ylabel='Scene editing success (%)',title='Intensity reader: label versus visible shape')
    axes[0].legend(loc='lower left',fontsize=8)
    values=[np.mean([grid[arm,s,r]['quotient_edit'] for s in ['val','ood'] for r in c.RENDERERS[1:]])*100 for arm in c.ARMS]
    bars=axes[1].barh(np.arange(6),values,color=['#6a8faf','#187c80','#e88462','#6a8faf','#6a8faf','#6a8faf'])
    axes[1].set(yticks=np.arange(6),yticklabels=[a.replace('_',' / ') for a in c.ARMS],xlim=(0,112),xlabel='Geometric editing success (%)',title='Two unseen renderers: development only')
    axes[1].bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    fig.suptitle('R1: one training-data seed, one initialization; no confirmation claim',fontsize=12)
    fig.savefig(c.BASE/'results.png',dpi=180);fig.savefig(c.BASE/'results.pdf');plt.close(fig)
    table='\n'.join(f'| {arm} | '+ ' | '.join(f'{np.mean([grid[arm,s,r]["quotient_edit"] for s in ["val","ood"]])*100:.2f}%' for r in c.RENDERERS)+' |' for arm in c.ARMS+['template'])
    seconds=sum(json.loads(p.read_text())['seconds'] for p in (c.BASE/'runs').glob('d*/*/run.json'))
    report=f'''# R1 결과 — 방향 이름과 실제 보이는 모양을 분리해야 한다

**이번 개발 실험에서는 기존 강도 모델의 새 렌더러 편집 성능이 무너지지 않았다.** 대신, 원래 픽셀이 같아 구분할 수 없는 방향 이름을 정답으로 요구하면 정확도가 낮아진다는 점을 확인했다.

지그재그를 180도 돌리면 벡터 모양은 같다. 기존 PIL 그림에는 경계의 미세한 차이가 남아 모델이 방향 라벨까지 구별했다. 이 차이를 없앤 렌더러에서는 동일한 그림에 다른 정답 이름이 붙는다. 이때 틀린 이름을 고른 것을 모양 이해 실패로 해석하면 안 된다.

## 실제 실행

- 새 학습 장면 1,024개, 개발용 일반 조합 128개와 미학습 조합 128개. 시험 split 128개는 생성만 했고 모델 선택·평가에 쓰지 않았다.
- 이진/강도/RGB 입력 × 전체 방향/대칭 동등류 손실: 6개 모델. 각각 92,590개 파라미터, 6,000회 업데이트, 같은 미니배치·초기 가중치.
- 원래 PIL4/Lanczos, 대칭 보존 PIL4/Lanczos, PIL8/BOX 세 렌더러. 학습은 첫 렌더러만 사용했다.
- 렌더러를 알고 모든 모양을 대조하는 비학습 템플릿 참조도 평가했다. 알려진 색상표, 다른 물체 색, 정답 상태 감독, 해석적 편집기와 알려진 렌더러를 사용한 제한된 합성 환경이다.

## 눈에 보이는 모양 기준 편집 성공률

같은 모양으로 보이는 지그재그의 180도 차이를 동등하게 인정한다. 위치는 축마다 1px 이내, 모양·색·크기는 일치해야 하고 클릭 대상 선택도 맞아야 한다. 아래 값은 두 개발 split의 동일 가중 평균이다.

| 입력·손실 | 원래 렌더러 | 대칭 보존 | 다른 샘플링 |
|---|---:|---:|---:|
{table}

강도/전체 방향 모델의 엄격한 장면 편집 점수는 원래 렌더러에서 100%, 대칭 보존에서는 일반 조합 75.78%, 미학습 조합 74.22%였다. 모양 동등성을 반영하면 모두 100%다. 대칭 보존 영상의 지그재그 방향 라벨 정확도는 각각 51.47%, 36.84%로, 일반적인 의미의 시각적 실패율이 아니다.

별도 검사에서 색·크기·축을 바꾼 **48개 방향 쌍의 대칭 보존 RGB가 정확히 같음**을 확인했다. 각 쌍의 두 라벨을 균등하게 제시하면 영상만으로 두 이름을 구분할 수 없다. 이는 소규모 시험 정확도와 별개인 픽셀 동일성 검사다.

## 판정과 한계

선택 후보는 `binary_quotient`였지만 기준 `alpha_full`도 두 새 렌더러에서 100%였다. 사전 등록한 ‘두 렌더러에서 각각 5%p 개선’ 조건을 통과하지 못했으므로 **3개 데이터×3개 초기화 확인 실험은 실행하지 않았다.** 기준 모델의 천장 효과가 있어 이번 결과로 후보 우월성이나 전체 렌더러에 대한 견고성을 주장하지 않는다. 렌더러 변화가 반드시 모양 편집을 망친다는 가설도 이 두 변화에서는 지지되지 않았다.

엄격한 픽셀 동일성과 상태 허용 오차 점수는 다르다. 일부 이미지는 미세한 안티앨리어싱 가장자리가 겹쳐 물체를 그리는 순서만 바뀌어도 달라진다. `raster_order_audit.json`에 이를 분리했으며 정답 순서로 보고 점수를 고쳐 높이지 않았다.

## 검증·비용

42/42 조건의 전체 이미지 추론과 21,504개 편집 행을 다시 계산했다. 4,224개 데이터 이미지를 세 렌더러에서 재생하고, 원래 렌더러는 보존된 이전 구현과도 일치함을 확인했다. 모든 source/target 벡터 장면 가족을 회전 동등류 단위로 분리했다.

6개 모델 학습 시간 합계는 {seconds:.1f}초다. 별도 100-update 측정은 학습 성능 결과에 포함하지 않았다. 일부 다른 연구 준비 작업과 겹쳐 실행되었으므로 이 시간은 장치 성능 벤치마크가 아니다. 독립 연구자 재현이나 사람 평가를 대신하는 검증도 아니다.

다음 작업은 R3의 정답 상태 충돌 예측이다. 연구 전체 R2/R4/R5와 독립 사람 평가는 아직 완료되지 않았다.

![R1 comparison](results.png)

자료: [사전 등록](protocol.json), [선택 판정](selection.json), [전체 수치](aggregate.json), [검증](verification.json), [방향 감사](pose_audit.json), [픽셀 순서 감사](raster_order_audit.json).
'''
    (c.BASE/'RESULTS_KO.md').write_text(report)
    dump(c.BASE/'report_receipt.json',{'report_sha256':sha(c.BASE/'RESULTS_KO.md'),'figure_sha256':sha(c.BASE/'results.png'),'visual_inspection':'pending'})

if __name__=='__main__':main()
