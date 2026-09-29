"""Inventory the complete core figure set and its original numerical evidence."""
import json
import time
from PIL import Image
from common import HERE,dump,r


GROUPS=[
 ('생성 품질과 계산 비용','generation_cost_figure_v2',['generation_quality_cost.png','generation_quality_cost.svg'],['verification.json','figure_visual_review.json'],'같은 횟수/시간, 576장 생성 비용, 독립 확인과 AE+flow 총 시간.'),
 ('물체 분리와 실패','object_failure_audit_v1',['heldout_segmentation.png','heldout_segmentation.svg'],['verification.json','figure_manifest.json'],'물체 수·가림·합침·조각남; 독립 장면 구간, encoder 초기화 하나.'),
 ('실제 개입과 조합','object_image_edit_v1',['capability_curves.png','capability_curves.svg','edit_sequence.gif'],['verification.json','analysis_verification.json','video_manifest.json'],'대상 선택, 추정 상태 준수, 실제 전체 정답 성공을 분리; 고정 첫 사례 영상.'),
 ('상태와 환경 인식의 분리','dynamics_observation_eval_v1',['state_context_substitution.png','state_context_substitution.svg'],['verification.json','report_manifest.json'],'정답 상태/환경 교체 대조; 한 데이터 t=3 시작의 한 단계.'),
 ('긴 미래와 반대 행동','dynamics_autonomous_v1',['long_horizon_curves.png','counterfactual_response.png','stability_overview.png','autonomous_examples.gif'],['verification.json','aggregate_verification.json','report_manifest.json','video_manifest.json'],'실패율·유한 오차·대상/비대상 반응, 고정 첫 사례; 한 데이터.'),
 ('독립 데이터 반복','dynamics_repeats_v1',['independent_state_comparison.png','independent_rollout_stability.png'],['report_verification.json'],'데이터 세 묶음과 초기화 반복을 분리; 큰 유한 발산 보존.'),
 ('환경 정보 누락','dynamics_context_omission_v1',['context_information_effect.png'],['report_verification.json','figure_visual_review.json'],'108조건, 합력/저항/둘 모두 가린 모델을 재학습.'),
 ('같은 관측과 다른 미래','observation_identifiability_v1',['identical_past.png'],['verification.json'],'구성한 경계 쌍; 전체 자료의 오차 하한이 아님.'),
 ('관측 길이와 모든 예측 시간','dynamics_observation_length_v1',[f'report_v1/{x}.png' for x in ['observation_length_effect','length_rollout_stability','length_horizon_by_data','length_paired_contact','length_action_response']],['report_verification.json','figure_visual_review.json'],'같은 t=7, 1/2/4/8장, 공통 유한 장면의 짝 비교, 모든 520블록 연결.'),
 ('외부 SVIB 공개 예제','svib_preview_shape_swap_v1',['report_v1/preview_comparison.png','report_v1/examples_alpha_0p0.png','report_v1/examples_alpha_0p6.png'],['report_verification.json','figure_visual_review.json'],'전체 오차·변경 영역·복사 기준·계산 비용과 고정 실패 예제.')]


def main():
    rows=[];assets=[]
    for label,folder,files,receipts,note in GROUPS:
        base=HERE/'reports'/folder;evidence={}
        for name in receipts:
            p=base/name;d=json.loads(p.read_text())
            if 'all_passed' in d:assert d['all_passed']
            evidence[str(p.relative_to(HERE))]=r.sha(p)
        for name in files:
            p=base/name;entry={'path':str(p.relative_to(HERE)),'sha256':r.sha(p),'bytes':p.stat().st_size,'group':label}
            if p.suffix in ['.png','.gif']:
                with Image.open(p) as im:
                    entry.update(width=im.width,height=im.height,frames=getattr(im,'n_frames',1))
                    for frame in range(entry['frames']):im.seek(frame);im.load()
            assets.append(entry)
        rows.append({'group':label,'files':files,'folder':folder,'evidence_sha256':evidence,'scope':note})
    lines=['# 핵심 그림과 영상 목록','',
        '계획 D02의 원인 분리·계산 비용·조건 변화·조합·장기 예측을 아래 자료로 연결한다. 그림마다 실험의 분모와 범위가 다르므로 한 모델의 연속 성능 상승으로 읽지 않는다. PNG와 일부 SVG, GIF는 로컬 파일이다.','',
        '| 질문 | 대표 그림 / 영상 | 수치 근거와 범위 |','| --- | --- | --- |']
    for row in rows:
        links=' · '.join(f"[{name.split('/')[-1]}]({row['folder']}/{name})" for name in row['files'])
        lines.append(f"| {row['group']} | {links} | {row['scope']} |")
    lines+=['','관측 길이 그림 5개와 생성 비용 그림은 이번 최종 검토에서 직접 확인했다. 개입 곡선·개입 마지막 장면·정보 교체 그림·장기 영상의 16단계 장면도 다시 확인했다. 기존 문서의 수치 검증과 별도의 시각 확인은 같은 검사가 아니다. 모든 래스터 파일과 GIF 프레임은 이미지 디코더로 열어 검사했고 SHA-256을 남겼다. 색각 시뮬레이션이나 모든 SVG의 별도 브라우저 렌더링은 수행하지 않았다.','',
        '[파일 목록과 근거 해시](figure_catalog_verification.json) · [연구 메모](RESEARCH_MEMO_KO.md)']
    report=HERE/'reports/FIGURE_CATALOG_KO.md';report.write_text('\n'.join(lines)+'\n')
    dump(HERE/'reports/figure_catalog_verification.json',{'all_passed':True,'groups':rows,'assets':assets,'group_count':len(rows),
        'file_count':len(assets),'all_raster_frames_decoded':True,'report_sha256':r.sha(report),'script_sha256':r.sha(__file__),
        'checked_unix':time.time(),'scope':'Complete D02 coverage inventory, source receipts, hashes and image decoding. Original numerical audits retained. Visual review is limited to the views explicitly described in the catalog and earlier recorded reviews.'})
    print('core figure catalog',len(rows),'groups',len(assets),'assets',flush=True)


if __name__=='__main__':main()
