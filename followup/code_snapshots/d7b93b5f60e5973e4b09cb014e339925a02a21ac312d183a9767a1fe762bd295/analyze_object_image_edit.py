"""Scene-grouped summaries and declared post-hoc failure strata, no model changes."""
import csv
import json
import numpy as np
from common import HERE,dump,r
from object_edit_confirmation import NAME

KEYS=['target_selection_correct','target_success_gt','target_other_correct_gt','non_target_correct_gt','count_correct_gt',
      'end_to_end_success','full_command_success_relative_to_prediction','non_target_preserved_relative_to_prediction',
      'image_foreground_mae','changed_pixels_mae','input_passthrough_foreground_mae']


def read_rows(path):
    result=[]
    for row in csv.DictReader(path.open()):
        out={}
        for k,v in row.items():
            if v in ['True','False']:out[k]=v=='True'
            elif v=='':out[k]=None
            elif k=='operation':out[k]=v
            else:out[k]=float(v)
        result.append(out)
    return result


def seed_scene_stats(by_seed,key):
    arrays=[];ids=None
    for seed in range(3):
        rows=by_seed[seed];valid=[v for v in rows if v[key] is not None];current=sorted({v['source_index'] for v in valid})
        if ids is None:ids=current
        assert current==ids
        arrays.append([np.mean([v[key] for v in valid if v['source_index']==i]) for i in ids])
    if not ids:return None
    a=np.asarray(arrays);seeds=a.mean(1)
    return {'mean':float(seeds.mean()),'values_by_seed':seeds.tolist(),'initialization_sample_sd':float(seeds.std(ddof=1)),
        'scene_ci95_conditional_on_three_models':r.bootstrap(a.mean(0))['base_scene_ci95'],'source_scenes':len(ids)}


def main():
    root=HERE/f'reports/{NAME}';assert json.loads((root/'verification.json').read_text())['all_passed']
    study=json.loads((root/'summary.json').read_text());groups=[];failures=[];visibility=[];cache={}
    splits=['test','ood','count3','count4','occlusion']
    failure_keys=[('no_detected_object',None),('wrong_target_selection','target_selection_correct'),('target_edited_factors_wrong','target_success_gt'),
        ('target_untouched_factors_wrong','target_other_correct_gt'),('non_target_factors_wrong','non_target_correct_gt'),('count_mismatch','count_correct_gt')]
    for kind in ['flat','slot']:
        for split in splits:
            for control in ['identity','analytic','object']:
                by_seed={seed:read_rows(HERE/f'runs/{NAME}/{kind}_s{seed}_{control}/{split}/rows.csv') for seed in range(3)}
                cache[(kind,split,control)]=by_seed
                group_names=['all']+sorted({v['operation'] for v in by_seed[0]})
                for group in group_names:
                    rr={s:[v for v in rows if group=='all' or v['operation']==group] for s,rows in by_seed.items()}
                    groups.append({'kind':kind,'split':split,'controller':control,'group':group,'metrics':{key:seed_scene_stats(rr,key) for key in KEYS}})
                if control=='object':
                    categorized={}
                    for seed,rows in by_seed.items():
                        out=[]
                        for row in rows:
                            label='success'
                            for stage,key in failure_keys:
                                failed=row['selected_slot']<0 if key is None else not row[key]
                                if failed:label=stage;break
                            assert (label=='success')==bool(row['end_to_end_success'])
                            out.append({**row,**{k:label==k for k,_ in failure_keys},'success':label=='success'})
                        categorized[seed]=out
                    metrics={k:seed_scene_stats(categorized,k) for k in [x[0] for x in failure_keys]+['success']}
                    assert abs(sum(v['mean'] for v in metrics.values())-1)<1e-12
                    failures.append({'kind':kind,'split':split,'metrics':metrics})
            if split=='occlusion':
                scenes=json.loads((HERE/f'data/{NAME}/{split}/scenes.json').read_text());by_seed=cache[(kind,split,'object')]
                for low,high in [(0,.5),(.5,.75),(.75,.9),(.9,1.000001)]:
                    selected={}
                    for seed,rows in by_seed.items():
                        selected[seed]=[v for v in rows if low<=scenes[int(v['source_index'])]['visible_fraction'][int(v['target_id'])]<high]
                    visibility.append({'kind':kind,'visible_fraction_lower':low,'visible_fraction_upper':min(high,1),
                        'source_scenes':len({v['source_index'] for v in selected[0]}),'metrics':{key:seed_scene_stats(selected,key) for key in KEYS}})
    aggregate={'groups':groups,'exclusive_failure_priority':failure_keys,'failure_groups':failures,'occlusion_target_visibility':visibility,
        'scope':'Post-hoc descriptive failure strata; no selection or training. Seed spread is separate from conditional source-scene intervals. Failure priority is a reporting partition, not an independent causal decomposition.',
        'study_summary_sha256':r.sha(root/'summary.json'),'script_sha256':r.sha(__file__)}
    dump(root/'analysis.json',aggregate)
    write_report(root,aggregate,study)
    make_figure(root,aggregate)


def write_report(root,a,study):
    def m(kind,split,key,group='all',control='object'):
        return next(v for v in a['groups'] if (v['kind'],v['split'],v['controller'],v['group'])==(kind,split,control,group))['metrics'][key]['mean']
    labels={'test':'두 물체 test','ood':'미학습 모양·색','count3':'세 물체','count4':'네 물체','occlusion':'가림'}
    lines=['# 그림과 대상 점 표시를 입력한 실제 편집 결과','',
        '2026-09-23. B05의 고정 모델 연결과 새 확인 자료 평가를 완료했다. 학습 없이 기존 encoder·감독 학습 판독기·개입 모델을 연결했다. 실제로 물체를 안정적으로 편집하는 능력을 달성했다는 뜻은 아니다.','',
        '## 실험이 묻는 것','',
        '그림에서 물체 상태를 읽고, 점으로 지목한 대상의 색·방향·위치를 바꾼 뒤 그림을 다시 그릴 수 있는가? 모델에는 RGB, 보이는 대상 위 점의 (x,y), 명시적인 편집 명령만 전달했다. 자연어 명령은 아니다. 정답 상태·물체 수·마스크·ID는 이미지 모델에 전달하지 않는다.','',
        '640개 새 장면에 색 변경·90도 회전·5픽셀 이동·색 변경 후 회전과 가능한 미학습 색 조합 변경을 적용해 2,961개 과제를 만들었다. 일반·미학습 조합·세 물체·네 물체·가림에 각 128장이다. 기존 8,148개 물체 장면/편집 회전 궤도와 새 source family 사이 중복은 없었다. 같은 장면의 여러 명령은 독립 표본으로 세지 않는다.','',
        '모델은 다섯 후보 slot의 예측 존재 점수로 네 자리를 남겨 최대 네 물체 용량을 갖는다. 물체 수를 정답에서 가져오지 않는다. 대상은 예측 중심 중 점에 가장 가까운 물체로 선택한다. 편집 전 좌표 기반 평가 대응을 편집 후에도 유지한다. 색 후 회전은 중간 정답을 넣지 않고 같은 예측 slot에 순서대로 수행한다.','',
        '각 encoder는 한 초기화, 판독기와 대응 개입 모델은 세 초기화다. 각 방식에서 identity·명시적 변환 규칙·학습한 물체별 개입 모델을 비교했다. 별도로 정답 상태+정답 대상과 정답 상태+점 선택 대조군을 평가했다. 원래 encoder는 RGB만 학습했지만 상태 판독기는 train 정답 속성을 이용한 감독 학습이다.','',
        '## 대상 선택과 전체 성공','',
        '아래는 학습한 개입 모델을 연결한 세 초기화 평균이다. 선택 정답은 편집 전 좌표 기반 대응에 따른 평가다. 전체 성공에는 대상 선택, 목표 변경, 목표의 나머지 속성, 비대상 속성, 물체 수가 모두 맞아야 한다. 좌표 각 축 1픽셀, 반지름 0.5픽셀 허용이다.','',
        '| 조건 | flat 대상 선택 | Slot 대상 선택 | flat 추정 상태 기준 명령 성공 | Slot 추정 상태 기준 명령 성공 | 실제 정답 전체 성공 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for split,label in labels.items():
        vals=[m(k,split,key)*100 for key in ['target_selection_correct','full_command_success_relative_to_prediction'] for k in ['flat','slot']]
        full=[m(k,split,'end_to_end_success') for k in ['flat','slot']];assert full==[0,0]
        lines.append('| '+label+' | '+' | '.join(f'{v:.1f}%' for v in vals)+' | 두 방식 모두 관측 0건 |')
    lines+=['',
        '추정 상태 기준 명령 성공은 모델이 잘못 인식한 모양·위치도 출발 상태로 인정하고 요청한 변경만 수행했는지 본 값이다. 실제 정답과 일치했다는 뜻이 아니다. 예를 들어 다른 물체의 모양을 이미 틀리게 읽었다면, 그 오류를 그대로 보존해도 추정 상태 기준 보존은 성공한다.','',
        '모든 RGB 방식·세 초기화·세 개입 방식·다섯 조건에서 엄격한 전체 성공은 0건이었다. 이는 이번 표본의 관측이며 모집단의 성공 확률이 정확히 0이라는 주장은 아니다.','',
        '## 같은 명령 모델에 정답 상태를 주면','',
        '정답 상태와 정답 대상 ID를 제공한 학습 개입 모델은 새 다섯 조건의 세 초기화 모두 전체 성공 100%였다. 명시적 규칙 대조군은 상태·출력 이미지까지 정확히 맞았다. 반면 정답 상태가 있어도 대상 점과 가장 가까운 중심을 고르는 규칙은 가림 조건에서 92.2%만 성공했다. 이 경우 인식과 별개로 대상 선택 규칙의 한계도 있다.','',
        '추정 상태에 명시적 정답 변환 규칙을 적용해도 전체 성공은 0건이다. 따라서 현재 시스템에서 controller만 완벽하게 만드는 것으로는 인식한 상태의 오류가 해결되지 않는다. encoder의 정보가 원천적으로 없다는 결론이나 다른 판독기·학습법도 실패한다는 결론은 아니다.','',
        '## 명령 종류별 목표 속성 일치','',
        '전체 보존 조건을 제외하고, 실제 대상의 바뀌어야 할 속성만 채점한 두 물체 test 결과다. 이 값만으로 편집 성공을 주장하면 오류를 놓친다.','',
        '| 명령 | flat | Slot Attention |','| --- | ---: | ---: |']
    op_names={'color':'색 변경','rotate':'90도 회전','translate':'이동','color_then_rotate':'색 변경 후 회전','novel_color_pair':'미학습 색 조합'}
    for op,label in op_names.items():lines.append(f'| {label} | {100*m("flat","test","target_success_gt",op):.1f}% | {100*m("slot","test","target_success_gt",op):.1f}% |')
    lines+=['','## 출력 그림의 오차','',
        '실제 예측 상태로 68,103개 출력을 그려 보관하고 전부 재생했다. 평가 대응을 이용해 그림의 앞뒤 순서를 정답에 맞추지 않았다. 예측 slot 순서가 그리는 순서이며 깊이를 따로 학습하지 않았다. 좌표·크기 반올림 및 제한은 그림을 그릴 때만 적용했고, 상태 점수는 원래 실수값으로 계산했다.','',
        '| 조건 | flat 전경 오차 | Slot 전경 오차 | 원본 그림을 그대로 내보낸 참조 |','| --- | ---: | ---: | ---: |']
    for split,label in labels.items():lines.append(f'| {label} | {m("flat",split,"image_foreground_mae"):.3f} | {m("slot",split,"image_foreground_mae"):.3f} | {m("slot",split,"input_passthrough_foreground_mae"):.3f} |')
    lines+=['',
        'RGB를 0~1로 놓고 입력·정답 전경의 합집합에서 평균 절대 오차를 계산했다. 현재 편집 출력은 평균적으로 입력을 그대로 반환한 참조보다도 오차가 크다. 입력 반환은 명령을 수행하지 않으므로 높은 보존만으로 유용한 편집이라고 부를 수 없다는 대조군이다.','',
        '## 가림과 오류 분석','',
        '물체 수별·명령 종류별 결과와 대상의 보이는 비율 네 구간을 analysis.json에 남겼다. 추가 실패 분류는 무검출→대상 선택→변경 속성→목표 나머지 속성→비대상 속성→개수 순서로 첫 실패를 표시한다. 이 순서는 보고용 분류이며 각 원인의 독립적인 인과 효과를 추정한 것이 아니다.','',
        '가림에서도 모든 대상에 최소 한 개의 보이는 hard pixel을 요구했다. 따라서 완전히 가려진 대상의 복원 성능은 아니다. 겹치는 물체의 geometry-only 대응과 깊이 미학습도 해석의 한계다. 새로운 데이터 생성 seed 하나에서 실행했으며 학습 자료와 renderer의 규칙은 같은 합성 세계다.','',
        '## 검증과 다음 단계','',
        '데이터 640장·편집 2,961개·전역 회전 2,560건을 재생했다. RGB encoder 1,280개, 판독 예측 3,840개, 115개 비교 조건의 편집/출력 68,103건, 장면별 지표·구간 17,388개를 재검증했다. 판독기와 controller, 평가 전 고정한 코드는 바뀌지 않았다.','',
        'B05의 연결·확인 평가를 마쳤으며 성능 한계를 그대로 기록한다. B07의 전체 실패 분석은 기존 분리 오류와 이번 개입 오류를 연결해 마무리한다. 독립 작업인 SVIB 외부 평가와 영상 기반 운동·긴 미래 예측도 계속한다.','',
        '[전체 집계](summary.json) · [초기화/장면 및 실패 분석](analysis.json) · [데이터 검증](data_verification.json) · [예측·출력 검증](verification.json) · [비교 그림](capability_curves.png)']
    (root/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')


def make_figure(root,a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    splits=['test','ood','count3','count4','occlusion'];x=np.arange(5)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    def stats(kind,key):
        return [next(v for v in a['groups'] if (v['kind'],v['split'],v['controller'],v['group'])==(kind,split,'object','all'))['metrics'][key] for split in splits]
    for kind,color in [('flat','#bf6630'),('slot','#2864a5')]:
        for ax,key in [(axes[0],'target_selection_correct'),(axes[1],'full_command_success_relative_to_prediction')]:
            ss=stats(kind,key);means=np.array([v['mean']*100 for v in ss]);low=np.array([v['scene_ci95_conditional_on_three_models'][0]*100 for v in ss]);high=np.array([v['scene_ci95_conditional_on_three_models'][1]*100 for v in ss])
            ax.errorbar(x,means,yerr=[means-low,high-means],marker='o',capsize=3,label=kind,color=color)
        axes[2].plot(x,[v['mean'] for v in stats(kind,'image_foreground_mae')],marker='o',label=kind,color=color)
    axes[1].axhline(0,color='#555',linestyle='--',label='Actual full-scene success: both 0')
    axes[2].plot(x,[v['mean'] for v in stats('slot','input_passthrough_foreground_mae')],marker='s',linestyle='--',color='#555',label='Input image unchanged')
    for ax in axes:
        ax.set_xticks(x,['2 objects','Pair OOD','3 objects','4 objects','Occlusion'],rotation=20);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    axes[0].set(title='Target selection',ylabel='Correct (%)',ylim=(-5,105));axes[1].set(title='Command follows perceived state',ylabel='Success (%)',ylim=(-5,105));axes[2].set(title='Actual output image error',ylabel='Foreground MAE (0–1)',ylim=(0,.4))
    fig.suptitle('Frozen RGB/click edit pipeline — 640 fresh scenes, three head/controller seeds',fontsize=13)
    fig.savefig(root/'capability_curves.png',dpi=160);fig.savefig(root/'capability_curves.svg');plt.close(fig)
    dump(root/'analysis_manifest.json',{'analysis_sha256':r.sha(root/'analysis.json'),'report_sha256':r.sha(root/'RESULTS_KO.md'),
        'figures':{f:r.sha(root/f) for f in ['capability_curves.png','capability_curves.svg']},'script_sha256':r.sha(__file__),
        'intervals':'Conditional base-source-scene bootstrap, not independent-dataset uncertainty.'})


if __name__=='__main__':main()
