# 결과 읽기와 재현

이 폴더의 실험은 생성, 물체 편집, 운동 예측을 각각 비교한다. 하나의 통합 모델의 정확도로 합치지 않는다. `PLAN_KO.md`의 전체 범위와 `status.json`의 현재 상태를 함께 본다. 과거 날짜의 진행 문서는 당시 기록이며 현재 완료 여부를 대신하지 않는다.

## 결과를 읽는 순서

1. `reports/CURRENT_CAPABILITY_KO.md`: 각 과제의 실제 능력과 한계.
2. `reports/STAGE_A_CONCLUSIONS_KO.md`: 생성 차이의 원인과 평가 기준의 한계.
3. `reports/OBJECT_FAILURE_SYNTHESIS_KO.md`: 물체 인식부터 실제 편집까지의 실패 위치.
4. `reports/dynamics_repeats_v1/RESULTS_KO.md`, `reports/dynamics_context_omission_v1/RESULTS_KO.md`, `reports/dynamics_observation_length_v1/RESULTS_KO.md`: 반복·정보 누락·관측 길이.
5. `reports/svib_preview_shape_swap_v1/RESULTS_KO.md`: 외부 공개 예제의 부정적 결과까지 포함한 비교.
6. `reports/NEAREST_METHODS_REVIEW_V2_KO.md`: 기존 연구와 겹치는 부분 및 아직 주장할 수 없는 독창성.

## ZIP의 파일 무결성 확인

ZIP을 풀면 최상위 `orbitlab/` 아래에 기존 연구와 `followup/`이 함께 있다. 원래 `RELEASE_MANIFEST.json`은 이전 배포 목록이고, 새 `BUNDLE_MANIFEST.json`은 ZIP에 담긴 전체 목록이다. Python 표준 라이브러리만으로 아래 명령을 실행한다. 출력 파일은 새 경로를 지정해야 하며 기존 파일을 덮어쓰지 않는다.

```sh
python3 orbitlab/followup/verify_bundle_integrity.py orbitlab --output bundle-integrity.json
```

이는 바이트가 보존됐다는 검사다. 숫자가 과학적으로 옳거나 모델이 유용하다는 별도 증거는 각 결과의 재생·집계 검증에 있다.

## 다른 경로에서 CPU 예측 재생

실험 당시 Python과 패키지 버전은 환경 기록 및 `requirements.lock.txt`를 참고한다. 아래는 새로운 가상환경을 만드는 예시이며, 패키지 설치는 사용 환경의 네트워크와 해당 버전 배포 여부에 달려 있다. 봉인된 파일을 수정하지 않고 가상환경과 검증 출력을 바깥에 둔다.

```sh
python3.13 -m venv replay-env
replay-env/bin/python -m pip install -r orbitlab/requirements.lock.txt
PYTHONDONTWRITEBYTECODE=1 replay-env/bin/python orbitlab/followup/replay_bundle_sample.py --output replay-result.json
```

재생 범위는 길이별 추정기 72개의 첫 validation batch 32개씩, SVIB 예측기 24개의 첫 외부 test batch 16개씩이다. 합계 96개 모델·예측 2,688건이며 모든 seed·모델을 포함한다. 같은 batch 크기와 기존 허용 오차를 사용한다. 전체 학습을 새로 한 검사도, 모든 예측을 다시 만든 검사도 아니다. 원래 작업 중 수행한 전체 재생 검증은 각 `verification.json`과 실행 로그에 별도로 남아 있다.

기존 checkpoint·로그 중 일부의 경로 문자열은 원래 작업 위치를 기록한다. 위 재생 도구는 파일을 현재 묶음의 상대 위치에서 읽어 원래 절대 경로를 필요로 하지 않는다. 모든 과거 실행 스크립트가 임의 경로에서 그대로 작동한다고 보증하지 않는다.

## 전체 실험과 변경 실험

각 `planning_*.md`와 `configs/`는 학습 횟수·seed·입력 정보·비교군·평가 구간을 고정한다. `logs/`와 실행 세션 기록에 실제 실행 명령과 종료 여부가 있다. 과거 실험의 producer·verifier는 설정에 코드 해시로 묶여 있으므로 코드나 checkpoint를 직접 바꾸면 검증이 실패하도록 했다.

새 학습을 할 때는 기존 `runs/`에 덮어쓰지 않는다. 별도 복사본과 새 실험 이름에서 입력·설정·원본 해시를 다시 고정한다. 일부 verifier는 결과 파일을 기록하므로 완전 재검산도 작업용 복사본에서 수행한다. 보관 묶음 자체에 대한 무결성 검사는 읽기 전용 도구를 쓴다. CPU 예측 재생과 MPS 재학습의 수치 동일성은 다른 주장이다.

## 사람 판정과 외부 자료

`reports/human_review_v1/review.html`은 출처를 가린 240장 판정 도구다. 실제 사람이 제출한 CSV가 없으면 사람 검증이 완료되지 않는다. 내부 정답 연결표 `private_key.csv`는 판정자에게 보여주지 않는다. 공개 화면의 브라우저 시각·상호작용 검사는 도구의 로컬 URL 접근 제한 때문에 수행하지 못했다. 이 제한을 우회하지 않았다.

SVIB 예제는 공개 preview의 축소 실험이다. 전체 공식 데이터나 LPIPS 평가를 재현하지 않았다. 외부 저장소·논문은 출처 확인과 개인 보관을 위해 포함하며, 원래 라이선스와 저작권이 유지된다. ZIP을 만들었다는 사실이 전체 자료의 공개 재배포 허가를 뜻하지 않는다. 네트워크 업로드나 연구실 전달은 이 작업에서 수행하지 않는다.
