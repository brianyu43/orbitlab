# 3,000회와 6,000회 학습 표현의 같은 자료 비교

원래 가중치를 바꾸지 않고, 두 학습 길이의 12개 checkpoint를 동일 train/validation/test/OOD에서 CPU로 읽었다. 표준화와 판독기는 train으로만 학습했고 C는 validation 정확도로 선택했다. 각 표본의 네 회전은 같은 장면으로 묶는다. 모델·초기화·full 표현 공간과 판독기 설정을 맞췄다.

표는 세 초기화 평균 정확도(%)와 차이(%p)다. 독립 데이터 생성 seed는 하나이며, 연속 훈련 중간 checkpoint를 저장해 비교한 것은 아니다. 이 표의 6,000회 값도 같은 CPU 조건에서 새로 판독기를 맞춘 결과여서 이전 MPS 인코딩 기반 진단의 반올림 값과 소폭 다를 수 있다. 생성 성공률과는 다른 지표다.

| 모델 / 판독 항목 | 자료 | 3,000회 | 6,000회 | 차이(%p) |
| --- | --- | ---: | ---: | ---: |
| aug / shape/linear | test | 45.70 | 43.29 | -2.41 |
| aug / shape/linear | ood | 25.36 | 26.79 | +1.43 |
| aug / shape/rbf | test | 79.13 | 78.87 | -0.26 |
| aug / shape/rbf | ood | 44.34 | 52.21 | +7.88 |
| aug / color/linear | test | 99.48 | 98.99 | -0.49 |
| aug / color/linear | ood | 99.54 | 99.48 | -0.07 |
| aug / color/rbf | test | 98.40 | 98.08 | -0.33 |
| aug / color/rbf | ood | 99.48 | 99.51 | +0.03 |
| aug / rotation/linear | test | 35.32 | 33.07 | -2.25 |
| aug / rotation/linear | ood | 39.32 | 37.63 | -1.69 |
| equivariant / shape/linear | test | 41.67 | 45.54 | +3.87 |
| equivariant / shape/linear | ood | 18.75 | 21.48 | +2.73 |
| equivariant / shape/rbf | test | 71.22 | 75.00 | +3.78 |
| equivariant / shape/rbf | ood | 32.81 | 49.22 | +16.41 |
| equivariant / color/linear | test | 94.76 | 96.94 | +2.18 |
| equivariant / color/linear | ood | 97.01 | 99.48 | +2.47 |
| equivariant / color/rbf | test | 93.23 | 95.83 | +2.60 |
| equivariant / color/rbf | ood | 95.70 | 98.57 | +2.86 |
| equivariant / rotation/linear | test | 32.42 | 37.76 | +5.34 |
| equivariant / rotation/linear | ood | 38.80 | 42.84 | +4.04 |

C4의 full 모양 판독은 일반 test에서 선형 41.67→45.54%, 비선형 71.22→75.00%였다. 미학습 조합에서는 비선형 32.81→49.22%였다. 더 오래 학습한 표현에 모양 정보를 읽어낼 단서가 늘었지만, 선형 판독이나 미학습 조합은 여전히 약하다. aug에서는 일반 test의 모양 판독이 높아지지 않았다.

단순 판독기의 성공은 속성을 독립적으로 바꾸거나 그림을 정확히 생성한다는 증거가 아니다. 앞서 완료한 6,000회 m0/m2/m13 진단은 그대로 보존했고, 이 추가 비교의 범위는 full 공간이다. 같은 평가 자료에서 수행한 탐색이며 새 독립 확인 실험으로 세지 않는다.

[전체 점수](summary.json) · [입력·선택 규칙](protocol.json) · [재생 검사](verification.json) · [기존 6,000회 부분 공간 진단](../probes_6000_v1/summary.json)
