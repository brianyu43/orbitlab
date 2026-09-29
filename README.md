# OrbitLab

C4 회전 표현, 복원, 조건부 latent flow를 비교한 로컬 실험. 2026-09-23에 원래 계획의 15회차 산출물을 완료했다.

**같은 시간의 복원은 B1 회전 증강 AE가 좋았고, flow의 모양·색 조건 일치는 B2 C4 모델이 높았다. 생성물의 흐림과 모양 오류는 남아 있다.**

- [결과 보고서: 설계·실측·실패·한계](RESULTS_KO.md)
- [15회차 완료 증거표](reports/SESSION_COMPLETION_KO.md)
- [재현 명령과 저장 결과 재생](REPRODUCE.md)
- [6분 발표 PowerPoint](deliverables/OrbitLab_6min.pptx) · [발표 원고](deliverables/TALK_6MIN_KO.md)
- [원본·코드·결과·가중치 ZIP](deliverables/OrbitLab_15sessions_2026-09-23.zip) · [내용 해시 목록](RELEASE_MANIFEST.json)
- [전체 실행 계획](EXECUTION_PLAN_KO.md) · [원본 코드 감사](reports/CODE_AUDIT_KO.md) · [C4 수식](reports/C4_MATH_KO.md)

실제 실행: AE 비교 29회 + 생성용 AE 보완 6회 + flow 8회 = 43회. 별도의 pilot/calibration과 진단 검사는 이 수에 넣지 않았다. 15회차는 작업 목록이며 실제 소요 시간을 15×90분으로 주장하지 않는다.

| 3시드 평균 | B1 증강 AE | B2 C4 AE |
| --- | --- | --- |
| 같은 시간의 test 전경 MAE ↓ | 0.0446 | 0.0727 |
| 64-step 생성 seen 모양·색 동시 일치 ↑ | 34.0% | 57.0% |
| 64-step 생성 조합 OOD 동시 일치 ↑ | 29.5% | 54.5% |

생성 정확도는 clean renderer로 검증한 template 평가기 판정이다. 흐린 생성 이미지에서의 오류 가능성과 최악 사례를 보고서에 함께 제시한다. AE 시간 통제와 생성 비교는 서로 다른 실험이다.

`sources/originals/`에는 iCloud에서 회수한 원본 ZIP과 Word가 있다. `work/`는 실행 코드, `data/`는 orbit 중복 제거한 고정 분할, `configs/`는 실행 설정, `runs/`는 가중치·CSV·이미지·GIF, `reports/`는 감사·집계·검증 기록이다. 원본 Python 3개 파일은 그대로 보존했다.

```bash
cd /Users/xavier/Documents/dev/orbitlab
.venv/bin/python scripts/verify_artifacts.py
```

학습은 로컬 MPS에서 실제 실행했다. 선택 확장인 DINO·세계 모델은 수행하지 않았다. 큰 사전학습 모델의 성능이나 실사진 일반화를 주장하지 않는다.
