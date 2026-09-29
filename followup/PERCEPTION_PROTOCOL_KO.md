# RGB에서 물체를 찾는 다음 실험

2026-09-23. B04의 개발 pilot이다. 코드와 단위 검사를 준비했으며 pilot 완료만으로 B04 전체를 완료 처리하지 않는다.

원형은 Slot Attention의 반복적 경쟁 attention, 공유 GRU, 물체별 spatial broadcast decoder이다. 최종 그림은 slot별 RGB와 softmax 마스크의 혼합으로 복원한다. 논문 §2.1–2.2와 공식 `model.py`를 확인했다. [원문](https://arxiv.org/html/2006.15055v2), [공식 구현](https://github.com/google-research/google-research/tree/master/slot_attention).

| 항목 | 이번 소형 실험 |
| --- | --- |
| 입력 | 64×64 RGB만 사용 |
| 물체 용량 | 양쪽 모델 모두 slot 5개, slot당 32차원; 최대 네 물체와 배경을 담을 용량 |
| 비교 | Slot Attention / 같은 encoder와 decoder를 쓰는 flat-to-slots 모델 |
| 훈련 정보 | RGB 복원만 사용; 전경 가중치는 입력 밝기에서 계산하며 정답 물체 마스크·속성은 사용하지 않음 |
| 평가 | 전경 ARI, 전체 ARI, 최적 일대일 대응 마스크 IoU, 복원 오류를 별도 기록 |
| 첫 예산 | 각 1,000 steps, batch 16, 초기화 1개, validation만 확인 |
| 기록할 차이 | flat 모델의 파라미터가 더 많음; 같은 파라미터/시간 비교라고 주장하지 않음 |

원문 대비 encoder 폭·해상도·decoder·손실·데이터 규모와 학습 예산을 줄이거나 바꾼 개발 실험이다. 논문의 벤치마크 재현으로 부르지 않는다. 논문과 코드 원본은 `references/slot_attention/`에 URL·해시와 함께 보존했다.

실행 전 단위 검사로 attention의 입력 순서 불변성·slot 순서 일관성, 유한 gradient, RGB 혼합, mask 합, 분리 지표의 완벽/병합 예제를 확인했다. pilot은 계산 시간과 붕괴 여부를 확인하는 용도다. 이후 학습 예산·seed·확인용 새 평가 데이터를 고정한 전체 실험으로 넘어간다.

전체 B04–B05에서는 마스크를 입력하지 않는 물체 할당, 가림, 세·네 물체, 미관측 조합을 평가한다. 속성을 읽기 위한 별도 supervised probe를 사용한다면 비지도 표현 학습과 명확히 구분한다. 개입은 목표 변경·비대상 보존을 함께 채점한다. 현재 데이터는 물체마다 색이 달라 색 분할만으로 쉬워질 수 있으므로, B07 실패 분석에서는 같은 색 물체를 추가해야 한다.

물체 기준 좌표로 기하학적 대칭을 다루는 아이디어는 이미 Invariant Slot Attention에 있다. 단순히 slot과 기하 정보를 함께 쓴 것을 새로운 기여로 주장하지 않는다. 본문과 코드의 세부 비교는 D01에서 계속한다. [공식 논문](https://proceedings.mlr.press/v202/biza23a.html).
