# 15회차 요구사항과 완료 증거

2026-09-23. 원본 PLAN_KO.md 08절 및 Word 계획서와 대조했다. 15회차는 작업 슬롯 기준이다. 실제 수행·산출물 기준이며, 예정 시간을 실제 노동/학습 시간으로 환산하지 않았다.

| 회차 | 수행과 판정 | 증거 |
| --- | --- | --- |
| 1 | 원본 MPS forward/backward parity와 smoke 통과 | original_doctor_mps.json, original_smoke_mps.json |
| 2 | 4,864 base scenes의 orbit 중복 제거, split/factor 검증 | ../data/manifest.json, input_32.png, c4_orbits.png, 단위검사 |
| 3 | 원본 32px, n256, batch16, 200-step pilot 실행. 전경 MAE 0.2335로 흐림 확인 | ../runs/original_pilot_eq/, ../runs/original_pilot_eq_analysis/ |
| 4 | C4 encoder/decoder, 합성, DFT, 보장 범위 설명 및 unit tests | C4_MATH_KO.md, ../tests/test_research.py, unit_tests.log |
| 5 | B1 n1024 3,000-step baseline 학습·고정/최악 복원 검사 | ../runs/primary_aug_n1024_s0/ |
| 6 | B2 n1024 3,000-step 비교, foreground 지표와 시각 결과 | ../runs/primary_equivariant_n1024_s0/, ae_results.csv |
| 7 | B3 invariant-only, Fourier energy와 제거/증폭 개입 및 probes | ../runs/diagnostic_invariant_n1024_s0/analysis/, B2 band_*.png, band_interventions.json |
| 8 | train 표준화/적합, validation regularization, 독립 test/OOD와 A90 closure | representation_results.csv, 8개 analysis/probes.json, learned_action.png |
| 9 | 주 비교 seed 1/2 반복. 세 seed 원자료와 CI | ../configs/experiment_matrix.json, ae_summary.json, 각 *_samples.csv |
| 10 | n256/4096 각 3시드, 파라미터 통제 3회, 시간 통제 6회 | benchmark.json, ae_results.csv, figures/fairness.png |
| 11 | 3,000-step B1 seed2 진입 실패. 양 모델 6,000-step 보완 전부 통과 후 frozen AE flow 6회 | quality_gate.json, quality_gate_repair.json, repair_visual_review.json, flow run.json |
| 12 | 각 모델·seed의 동일 noise 576개를 Gaussian/16/32/64-step 생성. B2 GIF 회전 제어 | 8개 flow의 samples/samples_*.png, samples.pt, latent_rotation_*.gif |
| 13 | shape/color, occupancy, centroid, coverage, 정확한 train NN, 최악 grid 검사 | generation_results.csv, generation_summary.json, nn_reference.json, result_visual_review.json |
| 14 | 원본/데이터/checkpoint 해시, 표본 수, AE 동결, noise pairing, 실제 자원·시간·제약 감사 | completion_checks.json, final_audit.json, ../RESULTS_KO.md 7절 |
| 15 | 설명·결론·재현 명령·6장 발표·6분 원고·보관용 결과 패키지 | ../README.md, ../REPRODUCE.md, ../deliverables/, ../RELEASE_MANIFEST.json |

진행 완료와 가설 성공을 구분한다. H1 보편적 데이터 효율 우위는 미지지, H2 불변 pooling의 위치·방향 손실은 관측, H3 생성 조건 일치의 상대 개선과 불완전한 품질, H4 파라미터 공유와 실제 비용의 차이는 관측했다. 두 번째 Mac의 전체 재학습, flow peak RSS, 생성물에 대한 사람 판정, 여러 data seed 반복은 미수행이며 완료 근거로 세지 않았다.

3,000-step 본실험과 6,000-step 생성 보완 결과는 별도 보존했다. 보완 config에 남아 있던 상위 설명 3000 fixed steps는 6000으로 정정했고, 실제 실행 task는 모두 6000으로 동일했다. 수정 전 파일과 config_metadata_correction.json을 함께 보존했다.
