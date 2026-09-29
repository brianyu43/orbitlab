"""Report the verified information-omission effects and their limits."""
import csv
import json
from common import HERE,dump,r
from dynamics_context_omission import NAME,DATA


def main():
    root=HERE/f'reports/{NAME}';out=root/'across_data';evidence={}
    for path in [root/'verification.json',out/'verification.json',root/'identifiability/verification.json']:
        check=json.loads(path.read_text());assert check['all_passed'];evidence[str(path.relative_to(HERE))]=r.sha(path)
    aggregate=json.loads((out/'verification.json').read_text());assert aggregate['manifest_sha256']==r.sha(out/'manifest.json')
    manifest=json.loads((out/'manifest.json').read_text());assert manifest['files']['metrics.csv']==r.sha(out/'metrics.csv')
    index={tuple(row[k] for k in ['condition','kind','mode','split','category','metric']):row for row in csv.DictReader((out/'metrics.csv').open())}
    used=[]
    def get(condition,kind,split,metric='velocity_mae_pixels_per_frame'):
        key=(condition,kind,'variable_force',split,'all',metric);row=index[key]
        values=[float(row[f'data_{ds}_mean']) for ds in DATA]
        used.append({'key':list(key),'data_seeds':DATA,'values':values})
        return values
    def cells(v):return ' | '.join(f'{x:.5f}' for x in v)
    conditions={'full':'전체 정보','no_net':'합력 숨김','no_drag':'저항 숨김','no_context':'둘 다 숨김'}
    split_names={'test':'일반 test','count4':'네 물체','force_ood':'미학습 외력'}
    values={(condition,split):get(condition,'joint_exact',split) for split in split_names for condition in conditions}
    ratios={split:[100*(b-a)/a for a,b in zip(values['full',split],values['no_net',split])] for split in ['test','force_ood']}
    assert all(x>0 for vv in ratios.values() for x in vv)
    wrong_full=get('full','wrong_exact','force_ood');wrong_hidden=get('no_net','wrong_exact','force_ood')
    assert all(a>b for a,b in zip(wrong_full,wrong_hidden))
    v=json.loads((root/'verification.json').read_text());same=v['zero_net_structure_training_comparisons']
    assert len(same)==18 and all(x['weights_exactly_equal'] for x in same)
    lines=['# 환경 정보를 가렸을 때의 운동 예측','',
        f'상태와 외력을 함께 회전시키는 모델에서 합력(중력+바람)을 가리고 다시 학습하면, 일반 test의 한 단계 속도 오차가 데이터별 {", ".join(f"{x:.1f}%" for x in ratios["test"])} 커졌다. 미학습 외력에서도 세 데이터 모두 나빠졌다. 이 모델의 한 단계 이득에 환경 방향 정보가 기여한다는 관측이다.','',
        '저항만 가린 효과는 데이터와 평가 조건에 따라 달랐다. 또한 외력 방향을 함께 회전시키지 않는 모델은 미학습 외력에서 합력을 숨긴 쪽이 오히려 나았다. 정보가 많다는 이유만으로 유한한 학습 모델의 성능이 항상 좋아지지는 않았다.','',
        '## 무엇을 같게 두었는가','',
        '정확한 현재 물체 상태와 행동을 입력한다. 영상 인식이나 긴 미래 예측 점수가 아니다. 합력·저항·둘 모두를 가린 세 조건 각각에서 처음부터 다시 학습했으며, 학습과 평가에 같은 가림을 적용했다. 전체 정보 조건 27개는 이미 검증한 variable_force 가중치를 재사용했다. 신규 학습 81개와 전체 108조건의 평가를 완료했다.','',
        '모든 모델은 variable_force 자료로만 학습하고 다른 환경에도 그대로 적용했다. 모델별 12,000 steps, 초기 가중치, 표본 선택과 물체 순서 변경을 맞췄다. 각 표의 열은 독립 데이터 생성 seed 하나이고, 숫자는 고정 초기화 세 개의 평균이다. 독립 자료는 세 묶음이며 아홉 묶음이 아니다.','',
        f'108조건의 전이 예측 {v["predictions_replayed"]:,}개, 장면·접촉별 행 {v["episode_rows_verified"]:,}개와 집계를 재생했다. 동일 장면을 여러 조건에서 반복 평가한 수이므로 독립 표본 수와 다르다. 모든 환경·접촉 범주·train/validation 진단을 CSV에 보존했고, 아래 표는 variable_force의 보류 조건이다.','',
        '## 공동 회전 모델: 환경 정보의 효과','',
        '속도 평균 절대 오차(px/frame), 낮을수록 좋다. 정답 상태를 매 순간 다시 제공한 한 단계 평가이며, 시간에 따라 오차를 누적시키는 자율 예측과 다르다.','',
        '| 평가 조건 / 제공 정보 | 640031 | 997101 | 997102 |','| --- | ---: | ---: | ---: |']
    for split,label in split_names.items():
        for condition,info in conditions.items():lines.append(f'| {label} / {info} | {cells(values[condition,split])} |')
    lines+=['',f'합력을 숨기면 미학습 외력의 오차는 데이터별 {", ".join(f"{x:.1f}%" for x in ratios["force_ood"])} 커졌다. 저항을 숨겼을 때의 변화는 일관되지 않았다. 학습 저항 범위, 고정 예산과 해당 구조에 한정된 결과이며 저항 정보가 일반적으로 불필요하다는 뜻은 아니다.','',
        '## 잘못 적용한 회전 제약에서는 다른 현상','',
        '아래 두 모델은 같은 회전 평균 구조를 사용한다. 상태만 회전하는 모델은 외력 벡터를 고정하고, 공동 회전 모델은 외력도 상태와 함께 회전한다. 미학습 외력 조건에서 비교했다.','',
        '| 모델 / 제공 정보 | 640031 | 997101 | 997102 |','| --- | ---: | ---: | ---: |',
        f'| 상태만 회전 / 전체 정보 | {cells(wrong_full)} |',
        f'| 상태만 회전 / 합력 숨김 | {cells(wrong_hidden)} |',
        f'| 공동 회전 / 전체 정보 | {cells(values["full","force_ood"])} |',
        f'| 공동 회전 / 합력 숨김 | {cells(values["no_net","force_ood"])} |','',
        '합력을 가리면 입력 외력 벡터가 0이므로 두 회전 구조의 함수가 같아진다. 합력 숨김과 전체 환경 숨김의 18개 대응 학습 쌍에서 가중치까지 완전히 같았고, 예측 비교에서도 이 동등성이 확인됐다. 이때 두 구조의 우열을 별개의 실험 효과로 해석하지 않는다. 숨긴 실제 외력이 0이거나 평가 분포 자체가 회전 대칭이라는 뜻도 아니다.','',
        '상태만 회전하는 구조에서 정보를 지워 점수가 좋아진 현상은, 해당 구조가 제공된 방향 정보를 잘못 사용하거나 적절히 일반화하지 못할 수 있음을 보여준다. 최적 예측기에서 정보의 가치가 음수가 됐다는 결론이나 모든 모델에 대한 원인 증명으로 확대하지 않는다.','',
        '## 정보가 없어서 못 푸는 경우와 학습에 실패한 경우','',
        '별도 구성한 세 쌍에서는 현재 상태·행동·보이는 환경 입력이 같아도 숨긴 환경에 따라 다음 정답이 달랐다. 같은 입력에 같은 답을 내는 모델은 두 정답을 모두 맞출 수 없다. 각 쌍의 제곱 오차 최솟값을 닫힌 형태 물리식과 수치로 검사했다.','',
        '이 세 쌍은 정보 부족으로 생기는 모호성의 구체적 예다. 무작위 평가 자료에서 이런 사례가 얼마나 자주 생기는지나 전체 데이터의 최적 오차 하한을 계산한 것은 아니다. 현재 상태에도 과거 환경의 흔적이 남을 수 있다. 따라서 위 평균 오차 차이 전부를 식별 불가능성으로 설명하지 않는다.','',
        '## 다음 연결','',
        '환경 정보 누락 대조의 학습·평가·집계는 완료됐다. 현재 진행 중인 관측 길이 1/2/4/8 실험과 연결해, 과거를 더 보면 실제로 어느 정보를 회복하는지 확인한다. C07 전체, SVIB 외부 평가, 선행연구 비교, 실제 사람 판정과 최종 연구 묶음은 아직 완료되지 않았다.','',
        '[전체 수치](across_data/metrics.csv) · [같은 초기화의 차이](across_data/paired.csv) · [집계 검증](across_data/verification.json) · [동일 입력·다른 정답 예제](identifiability/EXPLANATION_KO.md)']
    (root/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    colors=['#3275a8','#d68a34','#5b986a'];plot_data=[]
    for ax,split,title in zip(axes,split_names,['Standard','4 objects','Unseen force']):
        for j,ds in enumerate(DATA):
            y=[values[c,split][j] for c in conditions];ax.plot(range(4),y,'o-',color=colors[j],label=f'Data {ds}')
            plot_data.append({'split':split,'data_seed':ds,'conditions':list(conditions),'values':y})
        ax.set_xticks(range(4),['Full','No force','No drag','Neither']);ax.tick_params(axis='x',labelsize=9)
        ax.set(title=title,ylabel='One-step velocity MAE (px/frame)');ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);fig.suptitle('Joint rotation dynamics: masked inputs in both training and evaluation\nEach point averages 3 fixed initializations; true current state and action provided',fontsize=12)
    for ext in ['png','svg']:fig.savefig(root/f'context_information_effect.{ext}',dpi=160)
    plt.close(fig)
    dump(root/'report_data.json',{'table_values':used,'plot_values':plot_data,'relative_error_increase_percent':ratios})
    dump(root/'report_manifest.json',{'evidence_sha256':evidence,'metrics_sha256':r.sha(out/'metrics.csv'),'source_sha256':r.sha(__file__),
        'files':{f:r.sha(root/f) for f in ['RESULTS_KO.md','context_information_effect.png','context_information_effect.svg','report_data.json']},
        'scope':'Verified context omission study. C07 observation-length study and full research objective still pending.'})
    print('context report and figure written',flush=True)


if __name__=='__main__':main()
