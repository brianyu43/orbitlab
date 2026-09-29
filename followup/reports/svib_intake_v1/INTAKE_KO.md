# SVIB 대표 과제 회수 및 프로토콜 검사

공식 프로젝트가 링크한 공개 예제 저장소에서 dSprites / Single Atomic(Shape-Swap)만 선택했다. 원본 이미지는 128×128이며 이 preview에는 물체 마스크가 없다. 정답 메타데이터와 이미지 쌍은 있다.

공식 코드 commit: `7eec57ef3c2ea7a2c7389cb6cd470eb3600fa57c`. 공개 예제 commit: `23586681bf0d79fba7e2f1e997964a3edd6d666e`.

| 분할 | 예제 수 | 입력 조합 수 | 목표 조합 수 | 입력이 test 입력 조합과 겹침 | 목표가 test 입력 조합과 겹침 | 입력=목표 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test | 100 | 12 | 40 | 12 | 12 | 26 |
| Train/alpha-0.0 | 100 | 4 | 16 | 0 | 2 | 25 |
| Train/alpha-0.2 | 100 | 16 | 47 | 0 | 9 | 32 |
| Train/alpha-0.4 | 100 | 28 | 55 | 0 | 8 | 29 |
| Train/alpha-0.6 | 100 | 40 | 55 | 0 | 11 | 29 |

500개 메타데이터 쌍 모두 두 물체의 모양만 서로 바꾸고 색·위치·크기를 유지하는 규칙과 일치했다. 일부 예제는 두 물체의 모양이 같아 입력과 목표 이미지가 동일하다. 따라서 아무것도 바꾸지 않는 기준선을 포함하고, 실제 변경이 있는 예제의 성능도 따로 보고해야 한다.

입력 조합의 학습/test 분리는 이 preview에서 확인했다. 다만 학습 목표 이미지에는 test 입력과 같은 조합이 일부 나타난다. 이는 이 과제가 입력 조합 일반화를 평가한다는 범위와 함께 기록할 노출 조건이다. 이미지-to-image 학습과 source-only 표현 학습을 구분하고, 모든 입력·목표 이미지를 섞어 비지도 pretraining을 수행하는 경우는 별도 설정으로 표시해야 한다. 이 관찰만으로 공식 벤치마크 전체에 누수가 있다고 단정하지 않는다.

공식 생성 코드는 RGB renderer 배열을 cv2.imwrite로 저장하고 공식 읽기 코드는 PIL RGB를 사용한다. 메타데이터 색과 저장 PNG 색을 비교하는 평가기를 만들 때 채널 순서를 확인해야 한다. 입력·목표 PNG를 그대로 학습·평가하는 원래 프로토콜을 임의로 바꾸지 않는다.

다음 작업은 이 형식의 데이터 어댑터와 identity 기준선, 축소한 예측 모델을 연결하는 것이다. 이 preview의 각 100개 예제를 전체 64,000개 학습 / 8,000개 test 벤치마크 결과로 보고하지 않는다.

[공식 벤치마크 설명](https://systematic-visual-imagination.github.io/) · [공식 코드](https://github.com/systematic-visual-imagination/svib) · [공식 예제](https://github.com/systematic-visual-imagination/svib-samples)
