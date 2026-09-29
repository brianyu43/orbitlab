# 원본 코드 감사 — 2026-09-23

## 회수·범위

iCloud Drive 최상위의 `OrbitLab_starter.zip`(31,141 bytes), `OrbitLab_Mac_실험계획서.docx`(53,491 bytes)를 복사했다. 원본 SHA-256과 ZIP 11개 파일 목록은 `source_manifest.json`에 있다. 압축 내부 경로·CRC 검사를 통과했다. 원본은 `sources/extracted/66642ae290950e82/orbitlab_starter`, 실행 복사본은 `work`이다. Word XML의 299개 문단을 `sources/word_plan_extracted.txt`에 추출하고 ZIP의 PLAN과 15회 일정·512개 이상 생성·16/32/64 step 요구를 대조했다.

원본 Python 3개 파일 전체, README, PLAN, 후속 구현 지시문, requirements, validation JSON 3개를 읽었다. 원본은 데이터/가중치 다운로드나 업로드 없이 합성 데이터를 생성한다. 선택 DINO 스크립트는 별도 가중치 다운로드를 포함하며 핵심 실험에는 실행하지 않는다.

## 확인한 구조

- Encoder는 `h(R_-k x)`를 네 방향에서 공유한다. group convolution 논문의 레이어를 구현한 것은 아니다.
- Decoder는 `mean_k R_k d(z_k)`로 대칭화한다. 공통 decoder의 sigmoid 뒤 평균이므로 대칭은 정확하나 이미지가 흐려질 수 있다.
- B1 총 latent=64, B2=4×16이지만 독립 encoder/decoder 잠재 head 크기는 다르다. 기본 파라미터는 B1 315,075, B2 118,419로 다르다.
- Flow는 shape/color one-hot을 조건으로 받아 속도장을 C4 평균한다. train latent의 batch+group 공통 채널 정규화는 rho와 교환한다.
- 원본은 train/val/test/ood RNG namespace와 조합 holdout을 구분하지만 pixel/orbit 중복은 제거하지 않는다.
- 원본 학습은 마지막 고정-step checkpoint를 저장한다. validation은 중간 선택에 사용하지 않는다. probe는 train 표준화와 val regularization 선택을 사용한다.
- 원본 AE checkpoint에는 optimizer/RNG/step이 없어 정확한 학습 재개 기능은 없다. 원본 분석은 배치별 지표 평균이고 per-scene 통계가 없다.
- 원본 `sample`의 B1 고정-rho GIF는 의미 있는 회전 작용으로 보장되지 않는다. B1의 회전 제어는 학습된 A90 결과로 판단해야 한다.

## 원본 실제 실행

원본 doctor/smoke를 MPS에서 통과했다. forward parity 최대 오차 1.19e-7, gradient parity 7.45e-9. encoder 오차 0, decoder 5.96e-8, flow 0. 원본 32×32, n=256, 200-step pilot의 foreground MAE=0.2335이며 실제 복원 grid가 흐렸다. 실행 성공과 품질 성공은 구분한다.

## 후속 수정

원본 orbitlab.py는 변경하지 않고 `research.py`, `probes.py`, `generation.py`로 후속 기능을 추가했다.

- 최대 train pool 4,096개를 먼저 고정하고 orbit SHA-256으로 중복을 제거한다. 독립 ID val/test와 조합 OOD 각각 256개를 그 뒤에 만든다. train 중복 4개, test 중복 1개를 제거했다. nested n=256/1024/4096은 동일 frozen pool의 prefix다.
- seed에 따른 미니배치 샘플러를 모델 초기화와 분리했다. B1/B2는 같은 원본 장면과 외부 회전 minibatch를 받는다. B2의 구조적 정확성 때문에 외부 회전은 중복되지만 비용은 측정된다.
- MSE/PSNR/foreground MAE/IoU/중심 오차를 표본별 CSV에 저장하고 C4 4개를 base-scene으로 묶어 bootstrap한다.
- parameter match는 B2 CNN width=31로 정한다(318,699개, B1 대비 +1.15%). latent 64차원은 유지한다.
- final-step 규칙을 유지하고 벤치마크와 run별 시간·노출 수·코드/데이터/체크포인트 해시를 저장한다. 시간 예산 일치 비교는 별도 실행으로 분리한다.
- full/m0/m2/m13 probes, 학습된 A90 및 실제 decoded action, 성분 제거와 증폭, 독립 renderer-state 평가기, raw Gaussian/16/32/64-step flow 생성, exact pixel train-neighbor 검색을 추가했다.
- 평가기는 별도 seed의 512개 clean base scene×4 회전에서 shape/color 모두 100%를 얻었다. 흐린 generated image의 정확도가 보장되는 것은 아니므로 template IoU와 실패 grid도 보고한다.

## 확정 실험 전 품질 기준

64×64 예비 학습 1,000-step에서 B1/B2의 validation 전경 MAE가 각각 0.0915/0.1015였다. 모양 보존은 개선됐으나 B2의 윤곽이 더 흐렸다. 이 validation 근거로 최종 본실험은 양쪽 모두 3,000 steps로 고정한다. test/OOD는 이 결정에 사용하지 않았다.

생성 진입은 validation에서 전체 MSE가 black baseline의 절반 미만, foreground MAE≤0.10, mean mask IoU≥0.70, 독립 평가기의 복원 shape/color 일치 각각≥0.90, 고정 예제의 시각적 모양 보존을 확인해야 한다. 기준 실패 시 생성 결과를 품질 성공으로 주장하지 않으며 원인을 기록한다.

## 최종 수행 결과

29개 AE 비교와 6개 생성용 보완 AE, 8개 flow를 실제 완료했다. 3,000-step B1 seed2의 복원 color gate 실패(88.28%)는 보존하고, 양쪽 모두 6,000-step 보완한 뒤 여섯 모델이 기준을 통과했다. 최종 latent shape linear probe 90%를 달성했다는 주장은 하지 않는다. 실제 생성의 조건 일치는 B2가 높았으나 흐림과 모양 오류가 남았다. 원본 코드 3개의 byte 동일성 및 실행 checkpoint/CSV/분할은 최종 감사에서 확인했다.
