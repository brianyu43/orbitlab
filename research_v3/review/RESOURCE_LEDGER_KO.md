# 실제 학습 비용 장부

**사용자 요청으로 실험을 중단한 시점의 장부다. 미완료 학습은 그대로 남긴다.**

본 개발 학습은 측정기까지 포함해 63개 구성 요소·876,000업데이트가 계획되어 있다. 현재 학습 종료 파일이 있는 구성 요소는 48개, 업데이트는 336,000회다. 모든 종료 파일의 체크포인트 해시를 확인했지만 추론 재생·연구 판정은 별도 검증 파일로 확인해야 한다.

| 단계 / 구분 | 구성 요소 | 업데이트 | 각 학습 구간의 벽시계 초 합 |
|---|---:|---:|---:|
| r1/discarded_pilot | 6 | 600 | 9.2 |
| r1/completed_main_training | 6 | 36,000 | 547.6 |
| r3/discarded_pilot | 6 | 600 | 6.4 |
| r3/completed_main_training | 6 | 12,000 | 125.0 |
| r2/discarded_pilot | 8 | 800 | 50.2 |
| r2/completed_main_training | 8 | 30,000 | 2029.2 |
| r2/archived_discarded_pilot | 1 | 100 | 8.3 |
| r4/completed_main_training | 14 | 100,000 | 3282.3 |
| r4/discarded_pilot | 14 | 1,400 | 53.5 |
| r5/disks/discarded_pilot | 5 | 500 | 14.9 |
| r5/disks/completed_main_training | 5 | 14,000 | 500.1 |
| r5/external/completed_main_training | 3 | 108,000 | 6365.6 |
| r5/measurement/completed_reader_training | 6 | 36,000 | 4928.1 |
| r5/measurement/discarded_pilot | 6 | 600 | 80.5 |
| r5/external/discarded_pilot | 9 | 900 | 74.7 |

R5 기존 시작 시각 이후 1.78시간이 지났고 현재 논리 파일 크기는 13.30GB다. 초기 상한은 12시간/30GB이며 진행 파일을 읽은 시각에 따라 값이 달라진다.

현재 잠긴 양성 후보가 요구하는 확인 업데이트는 648,000회다. 모든 과제의 후보 선택이 끝나지 않았다면 이 값은 최종 확인 예산이 아니다.

숫자를 읽을 때의 구분:

- 동시 프로세스의 초를 더해 전체 경과 시간으로 사용하지 않는다. GPU 스케줄링 시험 중 일시 정지와 자원 경쟁도 각 실행의 시간에 반영될 수 있다.
- discarded_pilot은 초기 가중치를 버린 시간 측정이며 본 학습과 합쳐 성공 업데이트 수로 세지 않는다. archived_discarded_pilot도 비용에서 숨기지 않는다.
- 학습이 끝나지 않은 progress는 아래에 별도로 남긴다. 진행 파일만 보고 프로세스가 살아 있다고 판단하지 않는다.
- 데이터 준비·검증·추론·출력 재생·패키징·작업 대기는 위 학습 시간 합에 포함되지 않는다. GPU 병렬 벤치마크 및 CPU/MPS 일치 확인용 업데이트는 별도 기록에 있다.
- 학습량·파라미터·입력 감독·물리 사전 지식이 다른 비교는 같은 계산/정보 예산이라고 부르지 않는다.

| 미완료 학습 파일 | 저장된 업데이트 | 저장된 학습 초 |
|---|---:|---:|
| r5/external/dsprites_hard/runs/Multiple_Atomic/c4_s0/progress.json | 9,000 | 697.0 |
| r5/external/dsprites_hard/runs/Multiple_Atomic/plain_s0/progress.json | 9,000 | 695.8 |

전체 구성 요소별 파라미터/시간/체크포인트 출처는 `resource_ledger.csv`와 `resource_ledger.json`에 있다. 현재 환경 버전도 JSON에 보존한다. 새 환경에서 의존성을 설치하고 전부 재학습한 증거로 사용하지 않는다.
