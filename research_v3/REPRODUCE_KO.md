# 후속 연구 v3를 검토하고 재생하는 순서

이 문서는 **사용자 요청으로 중단한 시점의 재현 안내**다. 명령을 제공했다는 사실과 새 환경에서 실제로 전부 재학습했다는 사실은 다르다. 전체 연구는 미완료이며 [중단 기록](STOPPED_KO.md)에 보존 범위가 있다. 현재 공개 Git 커밋만으로 대용량 체크포인트·데이터가 모두 제공되는 것은 아니다.

모든 명령은 `orbitlab` 프로젝트 루트에서 실행한다. 다른 연구자가 검토할 때는 검토 묶음을 별도 폴더에 풀고 그 복사본을 사용한다. 원본 실행 폴더의 잠긴 소스·threshold·분할·가중치를 바꾸지 않는다.

## 1. 실행 환경과 확인 수준

실제 환경은 Apple M5 Pro, 통합 메모리 64GB, Python 3.13.14, PyTorch 2.14.0이다. 정확한 설치 버전은 루트 `requirements.lock.txt`, 실행 시각별 환경·계산 비용은 `review/resource_ledger.json`에 있다. 외부 R5 primary 모델은 **MPS float32, batch 32**, CPU 보조 작업은 주로 2스레드를 사용했다. R4 latent pool 생성은 6스레드였으며 별도 재생 수정 기록이 있다.

새 환경의 설치 예시는 다음과 같다. 이 안내 작성 시점에 별도 가상환경의 새 설치와 전체 재학습을 실행한 것은 아니다.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

| 확인 수준 | 무엇을 확인하는가 | 확인하지 않는 것 |
|---|---|---|
| ZIP 모든 파일 해시 | 받은 파일이 묶음 목록과 같은가 | 연구 결론·모델 성능 |
| 보고서·지표 재집계 | 전체 셀, 선택 규칙, 저장된 수치와 증거 연결 | 새 모델 학습 |
| 저장 가중치의 전체 추론 재생 | 같은 입력에서 출력과 지표를 다시 얻는가 | 다른 구현·연구자의 독립 재현 |
| 새 폴더의 동일 seed 재학습 | 같은 설정에서 다시 학습한 결과 | 새로운 데이터 seed 확인 실험 |

MPS/CPU 또는 라이브러리·스레드·batch를 바꾸면 부동소수점 결과가 달라질 수 있다. 비트 일치가 실패하면 환경 차이를 기록하고 조사한다. 기존 허용 오차를 넓혀 원래 검증을 통과한 것처럼 처리하지 않는다.

## 2. 검토 묶음의 범위와 별도 의존 자료

`review_package.py plan`은 현재 포함할 파일의 경로·크기와 제외 자료를 `review/package_inventory.json`에 기록한다. 실험 중에는 잠정 목록이다. 현재 설계는 v3 소스·사전 등록·데이터 배열·가중치·개별 결과·그림·검증 기록 및 필요한 역사적 참조 파일을 포함한다. 전체 이전 프로젝트 아카이브와 대화·사람 응답 원문은 포함하지 않는다.

```sh
.venv/bin/python research_v3/review_package.py plan
```

SVIB 원본 payload 약 12.2GB, 공식 전체 압축 파일, 제삼자 Git 저장소 복사본은 묶음 밖의 의존 자료다. payload를 제외해도 **원본 SQLite index와 manifest는 그대로 포함**한다. `package_inventory.json`의 `external_payloads`와 `upstream_dependency`에 복원 위치·해시가 있다.

데이터는 [공식 SVIB 페이지](https://systematic-visual-imagination.github.io/)의 Hard dSprites/CLEVR/CLEVRTex 자료다. 원래 다운로드 출처와 식별자는 `r5/references/sources.json`, 실제 받은 전체 압축 파일의 해시는 각 intake manifest에 기록되어 있다. 3D 자료는 원 실행에서 전체 스트림 해시를 계산했고 필요한 과제의 원본 바이트를 보존했다. 독립 두 번째 다운로드나 게시자가 제공한 별도 checksum 검증을 수행한 것은 아니다.

| 공식 Hard 압축 파일 | 기록된 SHA-256 |
|---|---|
| dSprites | `ef600992ffcdb59feb7be41ab1577477b54e3f453a3aae20a16379e46b373f5b` |
| CLEVR | `de9c99be908b7234fdcce88299b07cbb48022216d19a97b5e4267a9c97a050a8` |
| CLEVRTex | `16487ca660bd9c21d54d2ddedec2387a634a00a7f89ee72cca6b4d90172e2c0c` |

규칙·속성 목록과 dSprites raster 진단에 필요한 공식 코드의 고정 버전은 다음과 같다. **별도 검토 폴더에서 해당 경로가 없는 경우** 실행한다. 기존 checkout을 덮어쓰는 명령이 아니다.

```sh
git clone https://github.com/systematic-visual-imagination/svib.git followup/external/svib
git -C followup/external/svib checkout 7eec57ef3c2ea7a2c7389cb6cd470eb3600fa57c
```

SVIB 코드의 CC0 및 포함된 Spriteworld의 Apache-2.0 고지·라이선스를 유지한다. 원 실행에서 사용한 파일별 해시와 일치하는지도 묶음 manifest의 `upstream_dependency.files`로 확인한다. 이 checkout은 3D 장면을 Blender로 새로 생성하기 위한 완전한 asset 배포가 아니다. 재생은 보존된 공식 PNG를 사용한다.

## 3. 원본 payload만 복원하기

공식 압축 파일을 로컬에 확보한 뒤 아래 명령으로 계획을 확인한다. 예시 archive 경로는 자신의 로컬 파일로 바꾼다. 이 도구는 다운로드를 수행하지 않는다.

```sh
.venv/bin/python research_v3/restore_external_payload.py \
  --family dsprites --task Single_Atomic \
  --archive /absolute/path/to/dsprites_hard.archive \
  --output-dir research_v3/r5/data/dsprites_hard/Single_Atomic
```

실제로 복원하려면 같은 명령 끝에 `--execute`를 붙인다. 출력 폴더의 `payload.bin`이 **없는 검토 복사본**에서만 실행한다. 도구는 기존 payload나 부분 복원 파일을 덮어쓰지 않는다. 전체 archive, 432,000개 원본 항목의 크기·해시, 연속 offset, 완성 payload 해시를 검사한 뒤 이름을 바꾼다. 원래 manifest와 index의 바이트를 그대로 유지한다.

dSprites는 네 task 각각에 반복하고, CLEVR/CLEVRTex는 `--family clevr` 또는 `clevrtex`, `--task Single_Atomic` 및 해당 archive/output 경로를 쓴다. 원래 intake의 `--verify`는 시간 필드를 다시 기록하여 잠긴 후속 해시 연결을 바꿀 수 있으므로 **기존 결과의 복원에 사용하지 않는다**.

실제 복원 검증: dSprites Single Atomic의 432,000개 항목, 727,174,022바이트를 새 폴더에 복원하여 기존 payload와 전체 SHA-256 일치를 확인했다. 검증 후 새로 만든 중복 payload만 정리했으며 `review/payload_recovery_check/`의 로컬 증거를 보존했다. 3D archive에서의 동일 복원 경로는 아직 전체 실행 검증을 하지 않았다.

## 4. 저장된 전체 결과와 실제 추론 재생

현재 셀 누락을 포함해 다시 집계한다. 아래 명령은 `research_v3` 안의 보고서/스냅샷을 갱신하므로 검토 복사본에서 실행한다.

```sh
.venv/bin/python research_v3/r5_external_aggregate.py
.venv/bin/python research_v3/resource_ledger.py
.venv/bin/python research_v3/review_audit.py
```

`r5_external_aggregate.py --require-complete`는 외부 필수 개발/평가/조건부 확인 셀이 하나라도 빠지면 실패한다. 이 검사는 외부 R5에 한정되며 사람 평가·최종 다음 계획·전체 연구 종료를 대신하지 않는다. `review_audit.py`는 현재 원계획 항목별 스냅샷이고 최종 종료 판정은 별도다.

외부 R5의 **새로운 전체 forward pass**는 다음 명령으로 실행한다. 기존 `verification.json`이 있어도 추론을 생략하지 않으며, 원래 결과를 덮어쓰지 않고 새 영수증 한 개를 쓴다. 출력 파일은 새 이름이어야 한다.

```sh
.venv/bin/python research_v3/replay_external_unit.py \
  --family dsprites --task Single_Atomic --kind c4 \
  --mode validation_selected --split-seed 890101 --init 0 \
  --output research_v3/review/replays/my_single_atomic_c4.json --execute
```

`--execute`를 빼면 계획만 출력한다. 실제 실행은 MPS로 8,000장·250개 batch를 새로 추론하고 원본 float32 출력 해시 모두와 8,000개 지표 행을 비교한다. 원래 source/model/evaluator를 재사용하는 체크포인트 재생이다. **속성 판독이나 새 학습을 수행하는 명령은 아니다.** 이 명령은 dSprites Single Atomic C4에서 실제로 검증했다.

가족·과제·모델·checkpoint 방식·partition seed·init을 바꾸어 각 완료 셀을 재생한다. 필요한 셀 목록은 `r5/external_report/all_test_cells.csv`에 있으며 진행 중인 셀은 아직 재생할 수 없다. 원래 재개용 `r5_evaluate.py --verify` 및 `r5_verify_external.py`는 기존 검증 기록이 있으면 일부 작업을 생략할 수 있으므로, 단순 재호출을 새로운 전체 추론 증거로 세지 않는다.

R1/R3/R2와 통합 원형 물체의 원래 전체 검증 명령도 검토 복사본에서 사용할 수 있다. 이들은 검증 JSON을 다시 쓰며, 각각의 검사 범위는 영수증에 명시된다.

```sh
.venv/bin/python research_v3/r1_verify.py
.venv/bin/python research_v3/r3_verify.py
.venv/bin/python research_v3/r2_verify.py
.venv/bin/python research_v3/r2_diagnose.py --verify
.venv/bin/python research_v3/r2_legacy.py --verify
.venv/bin/python research_v3/r4_pool_replay_v2.py
.venv/bin/python research_v3/r4_verify.py training
```

이 목록만으로 R2 decoder/R4 생성/통합 원형 물체의 모든 조건을 다시 계산했다고 주장하지 않는다. 각 세부 조건은 다음 프로그램의 `--help`와 보존된 evaluation protocol의 전체 조건 목록을 따른다.

| 조건 | 프로그램과 검증 옵션 | 전체 조건 근거 |
|---|---|---|
| R2 신경 decoder | `r2_decoder_evaluate.py --arm … --count … --verify` 및 필요한 `--identity` | `r2/decoder/verification.json`의 36조건 |
| R4 복원·flow·생성 | `r4_evaluate.py reconstruction/flow/generation/reference … --verify` | `r4/evaluation_verification.json` |
| 통합 원형 물체 | `r5_disk_evaluate.py perceive/controls/evaluate … --verify` | `r5/disks/verification.json`의 57조건 |

R4 pool은 수정된 `r4_pool_replay_v2.py`를 사용한다. 원래 `r4_verify.py pools`의 스레드 차이에 대한 보존된 실패 기록과 수정 근거를 지우지 않는다.

## 5. 실제 새 학습과 최종 묶음

동일 seed의 원형 물체 전체 재학습·교체 실험은 새 경로로 실행하도록 준비했다.

```sh
.venv/bin/python research_v3/r5_disk_reproduce.py --name independent_disk_run
```

이 역시 기본은 계획만 출력하고 `--execute`를 붙여야 실제 학습한다. 기존 결과는 유지하며 `fresh_reproductions/independent_disk_run/`에 새 데이터를 만들고 14,000회 본 학습·500회 pilot 및 전체 평가를 수행한다. 동일 seed 재실행이므로 새로운 독립 데이터 확인으로 세지 않는다. 현재 이 경로의 계획 확인은 끝났고 두 번째 전체 재학습은 실행하지 않았다.

전체 v3를 처음부터 다시 학습하는 단일 명령의 새 설치 재현은 아직 검증하지 않았다. 원 실행 driver는 기존 시간 상한·해시·PID에 연결되어 있으므로 보존된 폴더에서 무작정 재시작하지 않는다. 추가 확인은 원계획의 통과 후보와 seed에 한정하고, 자원 상한 때문에 남은 셀은 미완료로 유지한다.

최종 묶음은 모든 과학적 필수 셀, 사람 평가 적용 여부, 다음 연구 계획, 비용, 마지막 역사적 보존 감사를 연결한 `review/final_scope_decision.json`이 있어야 만들 수 있다. 이 파일을 단순히 수동으로 `true`로 바꾸어 미완료를 면제하지 않는다.

```sh
.venv/bin/python research_v3/review_package.py build --name final_review_v1
.venv/bin/python research_v3/review_package.py verify \
  --archive research_v3/release/final_review_v1/orbitlab-research-v3-review.zip
```

현재는 첫 명령이 최종 판정 부재로 거부되는 것이 맞다. 완성된 ZIP은 모든 member의 바이트를 다시 읽어 해시를 검사한다. ZIP 검증 통과는 과학적 능력의 성공 판정과 다르다. 전체 81,297개 역사적 파일의 원본 보존 검사는 원래 프로젝트 아카이브가 필요하며, 선택된 검토 묶음만으로 그 검사를 재실행할 수 있다고 주장하지 않는다.
