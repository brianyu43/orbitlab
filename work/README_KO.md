# OrbitLab — 맥 로컬 표현 분석·생성 실험 스타터

이 패키지는 계획서의 **작은 C4 표현 실험과 latent flow 생성**을 바로 시작하기 위한 코드입니다. RAE / JEPA / Flow Equivariant World Models의 논문 재현이 아닙니다. 원 논문의 아이디어 중 대칭 구조와 표현 공간 생성만 작은 통제 환경으로 가져왔습니다.

## 검증 상태

Linux CPU, Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5에서 다음을 확인했습니다.

- C4 encoder/decoder/flow의 equivariance, 군 합성, invariant pooling, Fourier 왕복, 유한한 역전파와 parameter update: 통과.
- AE 3 step → flow 3 step → 4-step ODE sampling → PNG/GIF → representation analysis → linear probe/learned-action fit: 실행 통과.
- **Mac MPS에서 직접 실행하지 않았습니다.** 속도·메모리·충분히 학습된 생성 품질은 검증하지 않았습니다. 3-step 시험 이미지는 유의미한 결과가 아니므로 포함하지 않습니다.
- DINOv3 선택 스크립트는 구문 검사만 했으며 가중치 다운로드/추론은 실행하지 않았습니다. 접근 승인과 네트워크가 필요할 수 있습니다.

검증 기록은 `validation_cpu_smoke.json`, `validation_cpu_doctor.json`, `validation_e2e.json`입니다. CPU doctor의 오차 0은 CPU 자기 비교 결과이지 MPS 검증이 아닙니다.

## 1. 설치와 장비 확인

Apple Silicon용 **arm64 Python 3.12 이상**의 새 가상환경을 사용하세요. Python 설치 여부는 이 패키지가 확인하지 못합니다.

```bash
cd orbitlab_starter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip freeze > installed-versions.txt
python orbitlab.py doctor --device mps --out doctor_mps.json
python orbitlab.py smoke --device mps --out smoke_mps.json
```

`--device mps`는 MPS가 없으면 실패하도록 설계했습니다. CPU로 조용히 바뀌지 않습니다. 의도적으로 CPU 검사할 때만 `--device cpu`를 지정하세요. 기본값 `auto`는 MPS가 없으면 CPU입니다.

처음에는 FP32를 유지하세요. unsupported operator가 나오면 먼저 해당 연산만 CPU/MPS 최소 예제로 검사하세요. CPU fallback을 켠 결과는 별도 실행으로 기록하고 MPS benchmark와 섞지 마세요. 메모리 한도 해제 환경변수는 사용하지 않습니다.

## 2. 최초 실행 — 작은 데이터에서 파이프라인 확인

```bash
python orbitlab.py train-ae --device mps --model equivariant \
  --size 32 --channels 8 --n 256 --batch 16 --steps 200 \
  --out runs/pilot_eq
python orbitlab.py analyze --device mps --ae runs/pilot_eq/ae.pt \
  --n 64 --batch 16 --out runs/pilot_eq_analysis
```

`reconstruction.png` 첫 줄은 입력, 둘째 줄은 재구성입니다. `intervention.png`는 입력 / 재구성 / latent를 90도 변환해 decode / 실제 입력을 90도 회전한 정답 순서입니다. 회전은 **이미지 중심을 기준으로 한 장면 전체 회전**이며, 물체 중심을 고정한 독립 회전이 아닙니다.

## 3. 주 비교 — augmentation과 구조적 equivariance

```bash
python orbitlab.py train-ae --device mps --model aug \
  --size 64 --channels 16 --n 1024 --batch 32 --steps 1000 \
  --seed 0 --data-seed 42 --out runs/aug_n1024_s0
python orbitlab.py train-ae --device mps --model equivariant \
  --size 64 --channels 16 --n 1024 --batch 32 --steps 1000 \
  --seed 0 --data-seed 42 --out runs/eq_n1024_s0
```

다음 seed는 `1`, `2`를 사용하되 `--data-seed 42`는 고정합니다. initialization/randomness와 데이터 구성을 섞지 않기 위한 것입니다. `plain`은 augmentation 없는 모델, `invariant`는 group mean만 남긴 정보 제거 대조군입니다.

**스타터는 parameter 수나 계산량이 정확히 맞춰진 벤치마크가 아닙니다.** equivariant 모델은 공유 encoder/decoder를 네 방향에서 평가합니다. `run.json`의 parameter 수와 시간을 보고, 본실험에서는 parameter-matched와 wall-clock-matched 비교를 별도로 구현하세요. 학습은 지정한 step 수의 최종 checkpoint를 저장합니다. best-validation checkpoint 선택 및 자동 다중 seed 집계는 구현되어 있지 않습니다.

## 4. 표현론적 분석

```bash
python orbitlab.py analyze --device mps --ae runs/eq_n1024_s0/ae.pt \
  --n 256 --batch 32 --out runs/eq_n1024_s0_analysis
python -m pip install scikit-learn
python probe_latents.py --input runs/eq_n1024_s0_analysis \
  --out runs/eq_n1024_s0_analysis/probes.json
```

`analyze`는 train/val/test/ood 각각에서 네 회전의 latent를 저장합니다. `probe_latents.py`는 train으로만 probe와 90도 선형 작용을 학습하고, val로 regularization을 고른 다음 test/ood에 한 번 적용합니다.

`fixed_rho_equivariance_relative_mse`는 **C4 모델에서만 그 의미가 설계로 보장**됩니다. plain/aug는 잠재 좌표계가 임의적이므로 이 수치만으로 나쁘다고 결론 내리면 안 됩니다. 이 모델들은 `probe_latents.py`의 학습된 A90 잔차와 closure도 비교하세요.

Fourier band energy는 group 축의 DFT입니다. m=0은 회전 불변 성분이며, m=1/3은 켤레 쌍, m=2는 90도 회전에서 부호가 바뀌는 성분입니다. **이것이 자동으로 모양/색/위치별 disentanglement라는 뜻은 아닙니다.** invariant 대조군의 병목 용량도 다르므로 주 경쟁 모델이 아니라 정보 제거 실험입니다.

## 5. 노이즈에서 새 이미지 생성

```bash
python orbitlab.py train-flow --device mps --ae runs/eq_n1024_s0/ae.pt \
  --n 1024 --batch 128 --steps 2000 --out runs/eq_flow_s0
python orbitlab.py sample --device mps --flow runs/eq_flow_s0/flow.pt \
  --n 48 --ode-steps 64 --out runs/eq_flow_samples_s0
```

`new_samples.png`는 학습 이미지 단순 회전/보간이 아니라 Gaussian noise에서 latent flow를 적분하여 얻은 이미지입니다. `latent_rotation_*.gif`는 그 latent를 C4 변환한 제어 실험입니다. GIF가 회전한다고 원본 생성 품질까지 좋다는 뜻은 아닙니다.

조건은 모양 4종 / 색 6종으로, 이 조건 labels는 **flow 학습에 제공**합니다. AE는 reconstruction만 학습합니다. label 없이 개념을 발견했다는 주장을 하면 안 됩니다. diagonal factor pair `(0,0), (1,1), (2,2), (3,3)`은 생성·분석의 OOD 시험이며 성공을 보장하지 않습니다.

생성 평가의 조건 일치율, 최근접 이웃, 분포 coverage, seed confidence interval, 16/32/64 ODE-step 비교는 계획서에 명세했지만 이 스타터에서 완전히 자동화하지 않았습니다. 표본 일부만 골라 결과를 제시하지 마세요.

## 6. 선택: DINOv3의 표현 관찰

```bash
python -m pip install 'transformers>=4.56' torchvision
# 필요한 경우 공식 모델 페이지에서 약관 동의/접근 승인을 먼저 완료합니다.
python extract_dino.py --images ./my_images --device mps \
  --limit 200 --out runs/dino_features.npz
```

기본 모델은 `facebook/dinov3-vits16-pretrain-lvd1689m`입니다. 모델 가중치를 다운로드하지만 사용자의 이미지를 외부로 업로드하지 않습니다. CLS와 patch 평균을 층별로 저장하며, register token은 patch 평균에서 제외합니다. 기본 레이어 3/6/9/12가 존재하는 경우에만 추출합니다. model/config/processor 변경은 별도 실험으로 취급하세요.

## 포함 / 미포함

포함: 생성기, 4개 AE 조건, group Fourier 기초 진단, train/val/test/ood split, latent intervention, conditional flow, PNG/GIF, linear probes, 학습된 A90, checkpoint와 환경 기록, MPS 전방/역전파 parity 검사.

미포함: 정식 JEPA 학습, 동역학·planning, 실제 RAE 재현, learned latent의 의미적 disentanglement 증명, 모든 생성 품질 평가, train/test 간 raster duplicate hash 제거, parameter/compute matched 실험 자동화, Streamlit 대시보드. 이것들은 계획서의 후속 과제입니다.

모든 output은 비어 있는 새 경로를 요구하므로 기존 실험을 덮어쓰지 않습니다. `torch.load`에는 패키지가 만든 로컬 checkpoint만 사용하세요.
