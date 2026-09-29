# OrbitLab 재현과 결과 검사

실측 환경은 macOS 27 arm64, M5 Pro/64GB, Python 3.13.14, PyTorch 2.14.0 MPS FP32이다. 버전은 requirements.lock.txt, 실행별 장치는 run.json에 있다. CPU에서 검사/추론할 수 있지만 학습 행렬은 mps를 명시한다. 다른 장치의 시간·미세 수치까지 같다고 보장하지 않는다.

## 저장된 결과 확인

```bash
cd /Users/xavier/Documents/dev/orbitlab
.venv/bin/python scripts/verify_artifacts.py
.venv/bin/python scripts/final_metrics.py
.venv/bin/python scripts/plot_results.py
.venv/bin/python scripts/build_report.py
```

집계 명령은 학습을 다시 하지 않는다. CSV/보고서/figure 집계 산출물을 갱신한다. runs/의 원본 체크포인트와 측정 CSV는 변경하지 않는다. release manifest 검사를 먼저 하거나 별도 복사본에서 집계하면 배포본을 보존할 수 있다.

## 새 폴더·새 환경에서 전체 학습

결과 패키지를 새 폴더에 푼 뒤 새 실행 폴더를 만든다. 기존 runs/, 완료 marker, 시각 검토 승인 기록은 복사하지 않는다. 원래 결과를 지우는 명령 대신 새 디렉터리를 사용한다.

```bash
mkdir ../orbitlab-reproduction
cp -R work scripts tests configs data requirements.lock.txt ../orbitlab-reproduction/
cd ../orbitlab-reproduction
mkdir reports runs
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip check
.venv/bin/python work/orbitlab.py doctor --device mps --out reports/doctor_mps.json
.venv/bin/python work/orbitlab.py smoke --device mps --out reports/smoke_mps.json
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_matrix.py
.venv/bin/python scripts/quality_gate.py
.venv/bin/python scripts/run_matrix.py --manifest configs/generation_ae_repair.json
.venv/bin/python scripts/quality_gate.py configs/generation_ae_repair.json
```

run_matrix.py는 config의 정확한 task 설정과 완성 marker의 해시를 검사한다. 일부만 남은 실패 폴더에 덮어쓰지 않는다. 실패 시 로그를 확인하고 새 output 경로와 새 config를 사용한다. optimizer/RNG를 저장하지만 중간 step부터 이어가는 resume CLI는 구현하지 않았다. 첫 quality_gate.py는 원 실행에서 B1 seed2가 실패해 종료 코드 1이었다. 이는 기준 검사 결과이므로 로그를 보존하고 예정된 6,000-step 보완을 실행한다. 보완 gate 실패 시 생성으로 넘어가지 않는다.

새 실행에서도 생성 진입 수치 기준을 모두 통과했는지 확인하고, **새로 생성된 여섯** repair_*/validation/val_reconstruction.png를 직접 본다. 모양과 위치가 보존되고 배경뿐인 출력이 아닌지 검사한다. 실제 검토 후에만 아래 새 기록을 작성한다. 원 실행의 통과 기록을 재사용하지 않는다.

```bash
# 여섯 새 복원 grid를 실제로 검토하고 모두 통과한 경우에만 실행
.venv/bin/python - <<'PY'
import json
from pathlib import Path
Path('reports/repair_visual_review.json').write_text(json.dumps({
    'passed': True,
    'review_method': 'Reviewed all six newly generated repair validation reconstruction grids',
    'images': [f'runs/repair_{m}_n1024_s{s}/validation/val_reconstruction.png'
               for s in range(3) for m in ['aug','equivariant']]
}, indent=2))
PY
.venv/bin/python scripts/postprocess_matrix.py
.venv/bin/python scripts/run_generation.py
.venv/bin/python scripts/nn_reference.py
.venv/bin/python scripts/final_metrics.py
.venv/bin/python scripts/plot_results.py
```

새 실행은 새 CSV/JSON/그림으로 해석한다. build_report.py의 설명 문장과 발표 원고에는 2026-09-23 원 실행의 관측과 판단이 포함되어 있으므로, 새 장비/결과에 맞춰 검토하기 전에는 이를 새 실행 보고서로 쓰지 않는다.

주 AE 비교는 3,000 steps, 생성 자격용 AE는 6,000 steps다. flow는 4,000 steps, 초기 noise seed는 53000+training_seed, n576, Euler 0/16/32/64다. 시간 통제 AE는 고정 26.938초이며 새 장비에서는 step 수가 달라진다. 탐색 flow 시간 통제는 새 실행의 B2 seed0 학습 시간으로 설정된다. 정확한 step 기준으로 비교하려면 저장된 run.json의 실제 step 수도 함께 참조한다.

학습과 시간 측정 동안 다른 GPU 학습을 겹쳐 실행하지 않는다. 데이터 전체를 새로 만들려면 빈 새 실행 폴더에 data를 복사하지 않고 work/research.py prepare를 실행한다. NPZ의 파일 해시는 재압축 metadata에 따라 달라질 수 있으므로 이미지/orbit 해시와 split metadata도 검사한다.

## 저장 생성 모델을 다른 경로에서 재생

원 flow checkpoint는 AE의 절대 경로를 기록한다. 아래 helper가 원본을 바꾸지 않고 AE 해시를 검증한 뒤 경로만 고친 flow 복사본과 별도 표본을 만든다.

```bash
.venv/bin/python scripts/replay_generation.py \
  --flow runs/flow_equivariant_s0/flow.pt \
  --ae runs/repair_equivariant_n1024_s0/ae.pt \
  --out replay_equivariant_s0 --device mps --seed 53000 --n 576
```

MPS가 없으면 --device cpu를 쓴다. 작은 경로 검사에는 --n 24를 쓸 수 있으나 576개 평가와 동등한 품질 검증은 아니다. 이번에는 24개를 CPU로 재생해 경로 변경과 같은 가중치로 새 표본을 생성하는 경로를 실제 검증했다. reports/portability_replay/를 참조한다.

## 기록을 읽는 규칙

- AE train n1024와 latent probe fitting n256은 다르다. probe train/val/test/OOD는 각각 256 base scenes×4 회전이다.
- AE bootstrap은 base scene 단위다. 생성의 base_scene_ci95 키는 공통 helper의 이름을 사용하며, 이 경우에는 생성 noise 표본을 재표집한 CI다. 회전 복제 표본을 독립 생성물로 세지 않았다.
- 고정-rho 오차와 GIF는 B1에서 정답 회전으로 보장되지 않는다. B1은 learned A90의 residual/decoded error로 비교한다.
- flow_time_*는 seed0 한 번의 flow 학습 시간 통제다. 세 seed 주 생성 평균과 섞지 않는다.
- peak_process_rss_bytes는 AE 프로세스 peak, mps_driver_allocated_bytes는 마지막 시점 allocation이다. flow/시스템 전체 peak로 읽지 않는다.
- code_hashes에는 학습 중 사용하지 않은 보조 파일도 포함된다. AE의 orbitlab.py/research.py는 현재본과 일치한다. AE 학습 중 별도 개발한 generation 보조 파일의 이전 해시를 최종 flow 실행 버전으로 오해하지 않는다. flow run.json의 code_sha256는 최종 work/generation.py와 일치한다.

## 발표 자료

발표 원고는 deliverables/TALK_6MIN_KO.md, 생성 코드는 .slides-build/build.mjs다. 슬라이드 3/5는 편집 가능한 native chart와 embedded workbook이다. chart 값만 소수점 6자리로 반올림하고 원 통계는 JSON/CSV에 보존했다. 재생성은 Codex Presentations artifact-tool 런타임을 사용한다. 일반 Python 실험 환경에 이 전용 런타임을 요구하지 않는다. 최종 PPTX, 노트, PNG, 패키지 검증 receipt를 보존했다.

이 기록은 명시한 로컬 검증 범위다. 새 환경에서 전체 43회 재학습을 두 번 완료했다는 주장이나 GPU별 bitwise 일치 보장이 아니다.
