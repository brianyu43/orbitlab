"""Inspect the authors' published preview without treating it as a full benchmark."""
import collections
import json
import subprocess
from pathlib import Path
from PIL import Image
from common import HERE,dump,fresh_dir,update_status,np,torch,o,r


def binding(v):return (v['shape'],tuple(v['color']),v['size'])


def main():
    root=HERE/'external/svib-samples/samples/dSprites/Single_Atomic'
    out=fresh_dir(HERE/'reports/svib_intake_v1')
    bins=collections.defaultdict(lambda:{'source':set(),'target':set()});counts=collections.Counter()
    identical=collections.Counter();bad=[];files=[];examples=[];row_data=[]
    for p in sorted(root.rglob('source.json')):
        split=str(p.parent.parent.relative_to(root));source=json.loads(p.read_text());target=json.loads((p.parent/'target.json').read_text())
        a=np.asarray(Image.open(p.parent/'source.png').convert('RGB'));b=np.asarray(Image.open(p.parent/'target.png').convert('RGB'))
        assert a.shape==b.shape==(128,128,3)
        counts[split]+=1;identical[split]+=int(np.array_equal(a,b))
        for side,d in [('source',source),('target',target)]:bins[split][side].update(binding(v) for v in d['objects'])
        valid=len(source['objects'])==len(target['objects'])==2
        valid=valid and [v['shape'] for v in source['objects']][::-1]==[v['shape'] for v in target['objects']]
        valid=valid and all(all(x[k]==y[k] for k in ['color','size','rotation','2d_coords']) for x,y in zip(source['objects'],target['objects']))
        if not valid:bad.append(str(p.relative_to(root)))
        row_data.append({'split':split,'episode':p.parent.name,'source_target_identical':bool(np.array_equal(a,b)),
            'identity_pixel_mse':float(np.mean((a.astype(float)/255-b.astype(float)/255)**2)),'shape_swap_rule_matches':valid})
        for name in ['source.png','target.png','source.json','target.json']:
            f=p.parent/name;files.append({'path':str(f.relative_to(root)),'sha256':r.sha(f)})
        if counts[split]<=4:examples.extend([torch.from_numpy(a.copy()).permute(2,0,1).float()/255,torch.from_numpy(b.copy()).permute(2,0,1).float()/255])
    assert len(row_data)==500 and not bad
    exposure={}
    for split in sorted(counts):
        exposure[split]={'episodes':counts[split],'observed_source_bindings':len(bins[split]['source']),
            'observed_target_bindings':len(bins[split]['target']),
            'source_overlap_test_source':len(bins[split]['source']&bins['Test']['source']),
            'target_overlap_test_source':len(bins[split]['target']&bins['Test']['source']),
            'identical_source_target_episodes':identical[split]}
    commits={name:subprocess.check_output(['git','-C',str(HERE/'external'/name),'rev-parse','HEAD'],text=True).strip() for name in ['svib','svib-samples']}
    dump(out/'manifest.json',{'repositories':{'svib':'https://github.com/systematic-visual-imagination/svib','svib-samples':'https://github.com/systematic-visual-imagination/svib-samples'},
        'commits':commits,'download_scope':'Sparse checkout of dSprites Single Atomic preview only','files':files,'exposure':exposure,
        'rule_failures':bad,'resolution':[128,128],'masks_present':False,
        'status':'intake_complete_external_training_not_yet_performed','code_sha256':r.sha(__file__)})
    r.write_csv(out/'episodes.csv',row_data);o.grid(torch.stack(examples),out/'source_target_pairs.png',8)
    text=['# SVIB 대표 과제 회수 및 프로토콜 검사','',
        '공식 프로젝트가 링크한 공개 예제 저장소에서 dSprites / Single Atomic(Shape-Swap)만 선택했다. 원본 이미지는 128×128이며 이 preview에는 물체 마스크가 없다. 정답 메타데이터와 이미지 쌍은 있다.','',
        f'공식 코드 commit: `{commits["svib"]}`. 공개 예제 commit: `{commits["svib-samples"]}`.','',
        '| 분할 | 예제 수 | 입력 조합 수 | 목표 조합 수 | 입력이 test 입력 조합과 겹침 | 목표가 test 입력 조합과 겹침 | 입력=목표 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split,d in exposure.items():text.append('| '+split+' | '+' | '.join(str(d[k]) for k in ['episodes','observed_source_bindings','observed_target_bindings','source_overlap_test_source','target_overlap_test_source','identical_source_target_episodes'])+' |')
    text += ['', '500개 메타데이터 쌍 모두 두 물체의 모양만 서로 바꾸고 색·위치·크기를 유지하는 규칙과 일치했다. 일부 예제는 두 물체의 모양이 같아 입력과 목표 이미지가 동일하다. 따라서 아무것도 바꾸지 않는 기준선을 포함하고, 실제 변경이 있는 예제의 성능도 따로 보고해야 한다.','',
        '입력 조합의 학습/test 분리는 이 preview에서 확인했다. 다만 학습 목표 이미지에는 test 입력과 같은 조합이 일부 나타난다. 이는 이 과제가 입력 조합 일반화를 평가한다는 범위와 함께 기록할 노출 조건이다. 이미지-to-image 학습과 source-only 표현 학습을 구분하고, 모든 입력·목표 이미지를 섞어 비지도 pretraining을 수행하는 경우는 별도 설정으로 표시해야 한다. 이 관찰만으로 공식 벤치마크 전체에 누수가 있다고 단정하지 않는다.','',
        '공식 생성 코드는 RGB renderer 배열을 cv2.imwrite로 저장하고 공식 읽기 코드는 PIL RGB를 사용한다. 메타데이터 색과 저장 PNG 색을 비교하는 평가기를 만들 때 채널 순서를 확인해야 한다. 입력·목표 PNG를 그대로 학습·평가하는 원래 프로토콜을 임의로 바꾸지 않는다.','',
        '다음 작업은 이 형식의 데이터 어댑터와 identity 기준선, 축소한 예측 모델을 연결하는 것이다. 이 preview의 각 100개 예제를 전체 64,000개 학습 / 8,000개 test 벤치마크 결과로 보고하지 않는다.','',
        '[공식 벤치마크 설명](https://systematic-visual-imagination.github.io/) · [공식 코드](https://github.com/systematic-visual-imagination/svib) · [공식 예제](https://github.com/systematic-visual-imagination/svib-samples)','']
    (out/'INTAKE_KO.md').write_text('\n'.join(text))
    update_status('B06','in_progress',['reports/svib_intake_v1/manifest.json','reports/svib_intake_v1/INTAKE_KO.md'],
        'Official preview recovered and inspected; external model training/evaluation remains pending.')
    print(json.dumps({'commits':commits,'exposure':exposure,'episodes_checked':len(row_data)},indent=2))


if __name__=='__main__':torch.set_num_threads(2);main()
