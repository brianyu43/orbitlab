"""Replay frozen encodings, all validation candidates and chosen test probes."""
import json
import time
import joblib
import numpy as np
import torch
from common import HERE,ROOT,dump,r,o
from diagnostics import probe_data


@torch.no_grad()
def main():
    torch.set_num_threads(4);root=HERE/'reports/probe_step_comparison_v1'
    settings=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
    assert settings['source_sha256']==summary['source_sha256']==r.sha(HERE/'probe_step_comparison.py')
    assert settings['probe_data_manifest_sha256']==r.sha(HERE/'data/probe_v1/manifest.json')
    expected={(steps,kind,seed) for steps in [3000,6000] for kind in ['aug','equivariant'] for seed in [0,1,2]}
    assert {(x['steps'],x['kind'],x['seed']) for x in summary['results']}==expected
    validation_candidates=prediction_instances=0
    for row in summary['results']:
        folder=root/f"{row['kind']}_{row['steps']}_s{row['seed']}"
        assert json.loads((folder/'result.json').read_text())==row
        assert row['protocol_sha256']==r.sha(root/'protocol.json')
        for name,sha in row['files'].items():assert r.sha(folder/name)==sha
        path=ROOT/row['checkpoint'];assert r.sha(path)==row['checkpoint_sha256']
        model,_=r.load_model(path,torch.device('cpu'));model.eval()
        with np.load(folder/'features.npz') as a:features={k:a[k] for k in a.files}
        with np.load(folder/'labels.npz') as a:labels={k:a[k] for k in a.files}
        with np.load(folder/'predictions.npz') as a:saved={k:a[k] for k in a.files}
        for split in settings['splits']:
            images,truth=probe_data(split);parts=[]
            for rotation in range(4):
                for start in range(0,len(images),64):
                    batch=torch.rot90(images[start:start+64],rotation,(-2,-1))
                    parts.append(model.encode(batch).flatten(1).numpy())
            np.testing.assert_array_equal(np.concatenate(parts),features[split])
            np.testing.assert_array_equal(labels[split][:,:2],np.tile(truth.numpy(),(4,1)))
            np.testing.assert_array_equal(labels[split][:,2],np.repeat(np.arange(4),len(truth)))
        fitted=joblib.load(folder/'fitted.joblib');scaler=fitted['scaler']
        np.testing.assert_allclose(scaler.mean_,features['train'].mean(0,dtype=np.float64),atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(scaler.var_,features['train'].var(0,dtype=np.float64),atol=1e-12,rtol=1e-12)
        assert scaler.n_samples_seen_==len(features['train'])
        values={s:scaler.transform(x).astype(np.float64) for s,x in features.items()}
        for spec,result in row['scores'].items():
            target,family=spec.split('/');index={'shape':0,'color':1,'rotation':2}[target];scores=[]
            for C in settings[family+'_C']:
                key=f'{target}_{family}_C{C}';prediction=fitted['estimators'][key].predict(values['val'])
                np.testing.assert_array_equal(prediction,saved[key+'_val'])
                scores.append(float((prediction==labels['val'][:,index]).sum()/len(prediction)))
                prediction_instances+=len(prediction);validation_candidates+=1
            np.testing.assert_array_equal(scores,result['validation_grid_accuracies'])
            assert result['C']==settings[family+'_C'][int(np.argmax(scores))]
            key=f"{target}_{family}_C{result['C']}"
            for split in ['test','ood']:
                pred=fitted['estimators'][key].predict(values[split]);np.testing.assert_array_equal(pred,saved[key+'_'+split])
                expected_score=float((pred==labels[split][:,index]).sum()/len(pred))
                assert expected_score==result[split]['mean'] and len(pred)==4*result[split]['base_scenes']
                prediction_instances+=len(pred)
        print('verified matched probe',row['kind'],row['steps'],row['seed'],flush=True)
    lines=['# 3,000회와 6,000회 학습 표현의 같은 자료 비교','',
        '원래 가중치를 바꾸지 않고, 두 학습 길이의 12개 checkpoint를 동일 train/validation/test/OOD에서 CPU로 읽었다. 표준화와 판독기는 train으로만 학습했고 C는 validation 정확도로 선택했다. 각 표본의 네 회전은 같은 장면으로 묶는다. 모델·초기화·full 표현 공간과 판독기 설정을 맞췄다.','',
        '표는 세 초기화 평균 정확도(%)와 차이(%p)다. 독립 데이터 생성 seed는 하나이며, 연속 훈련 중간 checkpoint를 저장해 비교한 것은 아니다. 이 표의 6,000회 값도 같은 CPU 조건에서 새로 판독기를 맞춘 결과여서 이전 MPS 인코딩 기반 진단의 반올림 값과 소폭 다를 수 있다. 생성 성공률과는 다른 지표다.','',
        '| 모델 / 판독 항목 | 자료 | 3,000회 | 6,000회 | 차이(%p) |','| --- | --- | ---: | ---: | ---: |']
    bindings=[]
    for kind in ['aug','equivariant']:
        for metric in ['shape/linear','shape/rbf','color/linear','color/rbf','rotation/linear']:
            for split in ['test','ood']:
                groups={steps:[x['scores'][metric][split]['mean'] for x in summary['results'] if x['kind']==kind and x['steps']==steps] for steps in [3000,6000]}
                means={k:sum(v)/len(v)*100 for k,v in groups.items()};diff=means[6000]-means[3000]
                line=f"| {kind} / {metric} | {split} | {means[3000]:.2f} | {means[6000]:.2f} | {diff:+.2f} |";lines.append(line)
                bindings.append({'kind':kind,'metric':metric,'split':split,'per_seed':groups,'means_percent':means,'difference_pp':diff,'line':line})
    lines+=['','C4의 full 모양 판독은 일반 test에서 선형 41.67→45.54%, 비선형 71.22→75.00%였다. 미학습 조합에서는 비선형 32.81→49.22%였다. 더 오래 학습한 표현에 모양 정보를 읽어낼 단서가 늘었지만, 선형 판독이나 미학습 조합은 여전히 약하다. aug에서는 일반 test의 모양 판독이 높아지지 않았다.','',
        '단순 판독기의 성공은 속성을 독립적으로 바꾸거나 그림을 정확히 생성한다는 증거가 아니다. 앞서 완료한 6,000회 m0/m2/m13 진단은 그대로 보존했고, 이 추가 비교의 범위는 full 공간이다. 같은 평가 자료에서 수행한 탐색이며 새 독립 확인 실험으로 세지 않는다.','',
        '[전체 점수](summary.json) · [입력·선택 규칙](protocol.json) · [재생 검사](verification.json) · [기존 6,000회 부분 공간 진단](../probes_6000_v1/summary.json)']
    report=root/'RESULTS_KO.md';report.write_text('\n'.join(lines)+'\n')
    dump(root/'table_bindings.json',bindings)
    dump(root/'verification.json',{'all_passed':True,'checkpoints_verified':12,'all_four_split_encodings_replayed':True,
        'validation_candidates_replayed':validation_candidates,'classifier_prediction_instances_replayed':prediction_instances,
        'table_rows_checked':len(bindings),'summary_sha256':r.sha(root/'summary.json'),'report_sha256':r.sha(report),
        'verifier_sha256':r.sha(__file__),'completed_unix':time.time(),
        'scope':'All saved embeddings, all validation candidates and chosen test/OOD predictions checked, plus train-only scaler moments and fixed hyperparameter selection. Fitted probes replayed; original AE training not repeated.'})
    print('matched checkpoint comparison verified',validation_candidates,prediction_instances,flush=True)


if __name__=='__main__':main()
