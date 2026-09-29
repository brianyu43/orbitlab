# 후속 연구의 선행 방법 확인

2026-09-29 공식 논문·저장소와 보존된 소스 파일을 다시 확인했다. 이전 `followup/reports/NEAREST_METHODS_REVIEW_KO.md`에도 이미 Dreamweaver 본문/부록/코드 검토가 있었으므로, 완료-v2 다음 계획의 ‘초록만 확인’이라는 표현은 이전 전체 기록을 정확히 반영하지 못했다. 원문은 보존하고 여기서 바로잡는다.

## Dreamweaver와 R2의 정보 예산

[논문 v5](https://arxiv.org/html/2501.14174v5)의 §2–3, App. C.1–C.5를 확인했다. 영상의 recurrent block-slot, 블록별 prototype memory, 미래 프레임 토큰 예측이 핵심이다. 표현 학습과, 편집할 블록을 알아내기 위한 정답 factor/mask 기반 probe를 구별해야 한다. 논문 표는 400,000회 학습, 5슬롯, 8블록, context 2–3프레임을 보고한다. 작은 R2 지도학습 비교를 이 논문의 재현이라고 부르지 않는다.

[공식 저장소](https://github.com/ahn-ml/dreamweaver-release)의 `binder.py`, `train.sh`, `LICENSE`를 내려받아 해시를 기록했다. 코드는 MIT, 저작권자는 2025 MLML이다. 실행 예시는 64px, 5슬롯, slot size 768, context/prediction 2/2이며 일부 학습률·gradient clip은 논문 표와 다르다. 따라서 ‘논문 설정’과 ‘공개 실행 예시’를 동일하다고 가정하지 않는다. 이번 단계에서 이 코드를 설치하거나 전체 모델을 재현한 것은 아니다.

R2는 계획대로 6슬롯/64px/4프레임 조건에서 단일 프레임과 다중 프레임, mask 감독 유무를 구분한다. 새로운 object/slot 인식기 구현은 OrbitLab 통제 실험이며 Dreamweaver의 완전 재현 또는 성능 대체물이 아니다.

## SVIB 다음 과제

[공식 자료 페이지](https://systematic-visual-imagination.github.io/)와 [공식 저장소](https://github.com/systematic-visual-imagination/svib)를 확인했다. Hard dSprites의 기존 압축에는 추가 규칙 과제가 있으므로 R5에서 먼저 로컬 파일을 확인한다. 저장소는 CC0-1.0으로 표시되어 있다. CLEVR/CLEVRTex 자료는 사용 시 다운로드 단위와 포함 라이선스를 별도로 확인해야 한다. 저장소의 라이선스 표시만으로 모든 하위 자산에 동일한 조건이 있다고 추정하지 않는다.
