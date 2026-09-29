# OrbitLab — 맥 로컬 표현 분석·생성 실험 계획서

작성 기준: 2026-09-23

## OrbitLab

맥에서 시작하는 표현 분석 · 제어 · 생성 실험 계획서

> 핵심 질문  |  같은 물체임을 알아보면서도 회전 정보를 잃지 않는 표현은, 데이터가 적을 때 더 잘 복원하고 더 일관되게 생성할까?

작성 기준: 2026년 9월 23일. 이전에 알려준 M5 Pro · 통합메모리 64GB 맥을 기준으로 설계했다. 현재 장치에 직접 접속한 것은 아니므로 실제 환경과 처리속도는 첫 실행에서 측정한다.

### 권고하는 선택

하나의 저장소에서 작은 실험을 완성한다. 주 실험은 C4 대칭 표현 → Fourier 성분 분석 → 잠재변수 개입 → conditional latent flow 생성이다. DINOv3 관찰은 선택 모듈, 월드모델은 핵심 결과를 얻은 뒤의 확장 모듈로 둔다.

### 왜 이 주제인가

최근 관심사인 기하 딥러닝, 불변량과 가변 표현, 월드모델의 표현을 한 실험 안에서 연결한다. 동시에 회전·생성 장면을 눈으로 확인하고, 어떤 가정이 어떤 결과를 만드는지 수치로 설명할 수 있다. 주제를 계속 넓히는 대신 실험 하나를 끝까지 완주하는 선택이다.

| 항목 | 기본 계획 |
| --- | --- |
| 범위 | 64×64 합성 이미지 / 64차원 잠재표현 / 소형 CNN·MLP |
| 기간 | 15회차 × 90–120분의 작업 슬롯. 실제 학습 대기시간은 별도 측정 |
| 자원 | 로컬 CPU·MPS. 핵심 실험은 모델·데이터 다운로드 및 유료 API 불필요 |
| 최종 산출물 | 재현 가능한 코드, 비교표, latent 개입 GIF, 새 생성 샘플, 6분 발표 |
| 제외 | 거대 이미지·영상 모델 사전학습, 4K 영상 생성, 실제 단백질 설계·실험 검증 |

### 이 문서의 상태

연구 계획과 작동하는 스타터를 제공한다. 과학적 결과를 이미 얻었다는 문서가 아니다. Linux CPU에서 구조적 검사와 초소형 실행은 통과했지만, Mac MPS의 성능과 충분히 학습된 모델의 품질은 아직 측정하지 않았다. 모든 목표 수치는 사전 정의한 판단 기준이지 관측 결과가 아니다.

읽는 순서: 2쪽 연구 흐름 → 3–7쪽 실험 설계 → 8–9쪽 실행·일정 → 10쪽 확장 → 11쪽 패키지 범위 → 12쪽 출처.

## 01  |  최근 연구 흐름

최신 모델 이름보다, 가져올 수 있는 연구 질문을 선택한다

아래는 2026년 9월 23일까지 확인한 원 논문·공식 구현의 관련 흐름이다. 전체 분야의 완전한 목록이나 “가장 최신인 단 하나의 방법”이라는 뜻은 아니다.

| 연구·시기 | 중요한 변화 | 맥에서 가져올 부분 |
| --- | --- | --- |
| DINOv3 · 2025-08 공개 | 학습된 시각 표현을 고정한 채 전역·패치 수준 특징을 재사용한다. ViT-S/16은 21M parameters. [S1,S2] | 작은 모델의 층별 특징을 추출하고, 회전·위치 변화에 무엇이 보존되는지 조사 |
| RAE · 2025-10 / T2I 확장 2026-01 | 표현 encoder와 decoder를 결합하고, 그 잠재공간에서 생성한다. T2I 후속 연구는 0.5B–9.8B 범위를 실험했다. [S3,S4] | 거대 DiT를 재현하지 않고 작은 encoder를 고정한 뒤 소형 flow만 학습 |
| V-JEPA 2.1 · 2026-03 | 영상·이미지에서 공간 구조와 시간 일관성이 있는 dense feature를 강화한다. 공개 ViT-B는 80M. [S5] | 큰 영상 모델 학습 대신, 2D 영상에서 feature prediction 문제를 작게 구성 |
| Flow Equivariant World Models · 2026-01 | 자기 움직임과 물체 움직임을 Lie-group flow로 구조화하여 메모리와 장기 rollout을 다룬다. [S6] | 알려진 작은 대칭군에서 일관성·장기 예측의 이점을 통제 실험 |
| LeJEPA · 2025-11 | 표현 분포와 collapse 방지를 더 명시적으로 다루는 self-supervised 학습 원리를 제안한다. [S8] | 평균 loss만 보지 않고 분산·effective rank 등 퇴화 진단을 추가 |

### 중요한 구분

C4 group lifting이나 Fourier 분해 자체는 새로운 2026년 기술이 아니다. 이번 기획은 오래된 정확한 수학 구조를 최근의 “표현을 이해와 생성에 함께 사용”하는 질문과 연결한 축소 실험이다. 논문적 새로움은 구조를 썼다는 사실이 아니라, 언제 도움이 되고 언제 실패하는지를 통제하여 얻어야 한다. [S3,S4,S9]

FEWM의 “flow”는 시간에 따른 변환군의 흐름을 뜻한다. 여기서 생성에 쓰는 flow matching은 noise 분포에서 data 분포로 이동하는 벡터장을 학습한다. 관련될 수 있지만 같은 개념이나 같은 논문 알고리즘이 아니다. [S6,S10]

## 02  |  질문과 가설

알아보기 · 보존하기 · 조작하기 · 생성하기를 분리한다

### 일상적인 말로 표현하면

“오른쪽을 향한 화살표와 위쪽을 향한 화살표가 같은 종류라는 것은 알아야 한다. 그러나 그림을 다시 만들 때는 어느 방향을 향했는지도 알아야 한다.” 분류에 좋은 불변성과 재구성에 필요한 정보 보존이 어떻게 공존하는지를 실험한다.

### 표현학습과 수학적 표현론을 연결하는 최소 수식

```text
E(R_k x) = rho_k E(x)
D(rho_k z) = R_k D(z)
rho_a rho_b = rho_(a+b mod 4)
```

E는 이미지 encoder, D는 decoder, R_k는 전체 이미지의 90k도 회전, rho_k는 잠재공간에 작용하는 변환이다. 불변 표현은 rho_k가 항등작용인 경우다. 이 관계 자체는 군 equivariance의 기본 개념이며, 이번 실험에서는 유한군 C4로 정확하게 검사한다. [S9]

| 가설 | 검증 방법 | 허용하지 않을 결론 |
| --- | --- | --- |
| H1 데이터 효율 | 같은 base-scene 수에서 augmentation AE와 equivariant AE의 테스트 재구성·개입 오차를 비교 | equivariance가 정확하므로 모든 문제에서 더 좋다 |
| H2 정보 보존 | 전체 C4 표현과 group mean만 남긴 표현의 복원·방향 probe를 비교 | 불변 성분은 곧 “의미”, 나머지는 곧 “자세”다 |
| H3 생성과 제어 | 고정된 AE 표현에서 flow를 학습해 새 표본·조건 일치·회전 제어·최근접 이웃을 평가 | GIF가 잘 회전하므로 새 이미지 생성도 성공했다 |
| H4 계산비용 | 동일 step 실험과 동일 wall-clock 실험을 분리하여 비교 | 공유 weight이므로 계산량도 같다 |

### 성공의 정의

어느 모델이 이기든, 데이터·계산량·표현 구조를 구별하고 결과를 재현하면 프로젝트 목표를 달성한다. 사전 기대와 반대인 결과도 loss, capacity, 데이터의 방향 편향, domain shift를 검사할 수 있는 연구 결과다.

### 명시적으로 제공하는 정보

회전군과 회전 변환은 사람이 지정한다. 기본 AE에는 factor label을 주지 않지만 reconstruction 목표를 준다. 생성 flow에는 모양·색 조건을 제공한다. 따라서 “AI가 아무 가정 없이 군이나 개념을 발견했다”는 주장을 하지 않는다.

## 03  |  데이터와 분할

정답을 아는 작은 세계를 먼저 만든다

| 설계 요소 | 명세 |
| --- | --- |
| 관측 | 64×64 RGB, 검은 배경, 하나의 도형. pilot만 32×32 |
| 도형 | L / T / 화살표 / zigzag 4종, 색 6종, 연속 위치·크기 |
| 대칭군 | C4 = {0°,90°,180°,270°}. canonical raster 생성 후 rot90 사용 |
| 기본 크기 | 주 실험 base scenes 1,024. 추가 데이터 효율 실험 256 / 4,096 |
| 분할 단위 | 원본 장면·동일 원본의 모든 회전을 하나의 묶음으로 유지 |
| 조합 OOD | (shape,color)=(0,0),(1,1),(2,2),(3,3)은 학습에서 제외 |
| 동일 분포 평가 | 학습에 허용된 나머지 20개 조합으로 독립 val/test 장면 생성 |

### 누출 방지 규칙

train/val/test는 분리된 RNG namespace와 base_id를 사용한다. augmentation은 분할 이후에만 적용한다. 최종 보고 전에는 canonical image와 회전 orbit의 hash로 raster 중복까지 검사한다. 스타터는 seed와 factor-pair 분할을 구현했지만 raster hash deduplication은 후속 구현이다.

회전 holdout과 조합 holdout은 서로 다른 시험이다. 구조적 모델이 90도 회전을 처리하는 것은 설계상 보장되지만, 보지 못한 모양·색 조합을 만들어내는 것은 별도의 일반화 문제다. 두 결과를 합쳐 “OOD 성공”이라고 쓰지 않는다.

### 대칭을 잘못 주장하지 않기

회전은 화면 중심에 대한 장면 전체 변환이다. 화면 밖으로 나가는 임의 translation, crop, resize까지 같은 정확한 군으로 취급하지 않는다. 45도는 C4 밖의 변환이며, 별도 SO(2) 확장을 하지 않은 모델에 정확한 equivariance를 기대하지 않는다.

zigzag처럼 도형 자체에 180도 대칭이 있을 수 있다. “모양 방향”을 평가할 때는 그 대칭에 따른 동치 방향을 허용한다. 화면 전체의 위치까지 포함한 C4 orbit과 물체 자체의 pose는 구별한다.

### 보관할 metadata

base_id, split, shape, color, 중심 좌표, 크기, rotation, data_seed, 코드 버전, canonical hash를 보관한다. 실제 사진을 선택 모듈에 사용할 때는 공개 허가 또는 본인 촬영 데이터를 쓰고 개인정보가 드러나는 사진은 제외한다.

## 04  |  모델과 표현론적 분석

정확한 작은 구조를 직접 확인한다

| 모델 | 구성 | 역할 |
| --- | --- | --- |
| B0 Plain AE | 일반 CNN → 64D → decoder, 기본 방향 학습 | augmentation 없는 진단 대조군 |
| B1 Augmented AE | B0와 동일 구조, C4 augmentation | 주 비교 baseline |
| B2 C4 AE | 공유 encoder를 네 회전에 적용 → 4×16D; equivariant decoder | 주 실험 모델 |
| B3 Invariant-only | B2의 네 group slot을 평균하고 복제 | 방향 정보 제거 대조군; 주 성능 경쟁 모델 아님 |

### 외부 group-convolution 라이브러리 없이 시작하는 구현

```text
z_k = h(R_(-k) x),  k = 0,1,2,3
(rho_r z)_k = z_(k-r mod 4)
D(z) = (1/4) sum_k R_k d(z_k)
```

이 구현은 공유 encoder h와 decoder d를 여러 방향에서 평가하는 정확한 C4 구성이다. group-convolution 논문의 전체 architecture를 재현한 것이 아니다. 같은 weight을 공유해도 연산은 네 배 방향 평가를 포함하므로 parameter 수와 실제 시간을 모두 기록한다. [S9]

### DFT로 group 축을 분해한다

```text
z_hat_m = (1/2) sum_(k=0..3) z_k exp(-2*pi*i*m*k/4)
```

m=0은 회전에 변하지 않는 성분이다. m=1과 m=3은 실수 입력에서 켤레 쌍이며, 실수 좌표로 묶으면 2차원 회전 성분이 된다. m=2는 한 번의 90도 회전에서 부호가 바뀐다. DFT는 CPU에서 수행해 첫 실험에서 MPS complex 연산 의존성을 피한다.

### 세 가지 조작

첫째, rho_1로 group slot을 한 칸 옮겨 decode하고 실제 90도 회전 정답과 비교한다. 둘째, m=0만 남겨 어떤 정보가 사라지는지 확인한다. 셋째, 각 band에 대해 shape/color/방향 probe와 에너지를 측정한다. band 제거는 학습 분포 밖의 latent를 만들 수 있으므로 decoder artifact와 정보 제거 효과를 구분한다.

일반 AE의 잠재 좌표계에는 정해진 rho가 없다. B0/B1에는 train에서 A90을 ridge regression으로 적합하고 val로 정규화를 정한 뒤 test 잔차·A90^4≈I를 평가한다. 고정된 slot shift를 적용한 오차만으로 일반 AE가 열등하다고 결론 내리지 않는다.

## 05  |  생성 실험

복원 · 개입 · 새 표본 생성을 혼동하지 않는다

### 실험 G1: 잠재변수 개입

입력 x를 encode하고 rho_k를 적용한 뒤 decode한다. 결과를 실제 R_k x와 비교한다. 이 실험은 제어 가능한 재구성이며, noise에서 새 장면을 생성하는 실험은 아니다.

### 실험 G2: 고정된 표현에서 conditional flow 학습

AE 학습을 끝내고 encoder와 decoder를 모두 고정한다. 학습 장면을 latent z1으로 바꾼다. z0는 Gaussian noise, c는 모양·색 조건이다. 작은 MLP가 아래의 직선 경로 벡터장을 학습하고, 생성 시 ODE를 적분해 z1에 해당하는 표본을 만든다. 이는 flow matching 원리를 단순화한 설계다. [S10]

```text
z_t = (1-t) z0 + t z1
L_flow = E ||v_theta(z_t,t,c) - (z1-z0)||^2
x_new = D(ODEsolve(v_theta, z0, c))
```

C4 모델에서는 v_G(z,t,c) = (1/4) sum_g rho_g^-1 v(rho_g z,t,c)로 벡터장을 대칭화한다. 조건 c는 회전에 불변인 모양·색만 포함한다. 위치·방향 조건을 추가하면 그 조건도 함께 변환하도록 다시 설계해야 한다.

### 잠재변수 정규화 주의

group slot별로 서로 다른 평균·표준편차를 적용하면 원래의 단순한 rho 작용을 깨뜨릴 수 있다. 스타터는 group 축과 batch 축을 함께 평균하여 채널별 공통 정규화를 쓴다. pretrained DINO 특징을 사용하는 확장에서도 정규화와 decoder 학습을 별도로 검증한다.

### 필수 비교와 판정

G2는 “Gaussian latent를 decoder에 바로 넣기”와 “학습된 flow”를 비교한다. ODE 16/32/64 steps를 분리하고, 같은 생성 seed 목록에서 512개 이상을 평가한다. 네 회전만 반복한 GIF, 학습 표본을 거의 복사한 출력, 배경만 안정적인 출력은 성공으로 판정하지 않는다.

RAE의 핵심 아이디어와 연결되지만 정식 RAE 재현은 아니다. 본 실험의 encoder는 작은 합성 데이터에서 reconstruction으로 학습된다. RAE는 사전학습된 표현 encoder를 사용하는 설계이며, 그 차이를 보고서 제목과 방법에 명시한다. [S3,S4]

## 06  |  측정과 결론 규칙

수학적 보장과 경험적 성과를 분리한다

| 측정 | 무엇을 말하는가 | 주의 |
| --- | --- | --- |
| E(Rx)−rho E(x) | 구조 구현이 맞는가 | 정확한 B2에서는 correctness test이지 새 발견이 아님 |
| 재구성 MSE / PSNR / foreground MAE | 위치·색·모양 정보를 얼마나 보존하는가 | 배경 픽셀이 많은 데이터의 낮은 전체 MSE만 믿지 않음 |
| D(rho E(x))와 Rx의 차이 | 개입 후 정답 장면을 복원하는가 | D(rho z)=R D(z)만 맞아도 원본이 흐릴 수 있음 |
| linear probe / effective rank | 어떤 정보가 읽히고 latent가 퇴화했는가 | probe가 정보를 읽는다고 인과적으로 분리된 표현은 아님 |
| 조건 일치 / centroid / occupancy | 새 생성 샘플이 요구한 조건과 형태를 만족하는가 | 평가기 자체를 독립 holdout에서 검증 |
| NN 거리 / coverage / 다양성 | 학습 표본 암기·mode collapse가 있는가 | 품질이 낮은 noise의 높은 다양성을 성공으로 세지 않음 |

### 최소 실험표

확정 비교는 B1/B2 × n=1,024 × initialization seed 0/1/2, 총 6회다. B0/B3는 먼저 각 1회 진단한다. n=256/4,096의 추가 실험은 seed 0으로 방향만 확인한 뒤 의미 있는 차이가 보이면 seed를 늘린다. exploratory 결과와 confirmatory 결과를 표에 구분한다.

### 비교 공정성

같은 base scenes, 입력 해상도, 데이터 분할, latent 전체 차원, optimizer 계열, validation 사용 규칙을 맞춘다. parameter-matched와 wall-clock-matched 결과를 분리한다. 동일 step 결과만 있으면 정확히 그 조건에서의 비교라고 쓴다. 스타터는 parameter 수가 자동으로 같아지지 않는다.

### 통계와 중단 조건

세 seed의 원자료 점을 표시하고 평균±표준편차를 보고한다. bootstrap은 회전 이미지 단위가 아니라 원본 장면 단위로 수행한다. 생성에는 base noise seed를 고정한다. 1,024 장면에서 재구성이 제대로 되지 않으면 flow 단계로 넘어가지 않는다.

자체 진입 기준: held-out 재구성이 all-black baseline보다 개선되고, foreground 위치·모양이 육안과 수치 모두에서 보존되어야 한다. 예비 운영 목표로 shape/color probe ≥90%, foreground MAE ≤0.10을 둘 수 있으나 데이터 난이도에 따라 calibration 후 본실험 전에 고정한다. 이는 관측된 성능이 아니다.

FID 하나로 합성 도형의 품질을 주장하지 않는다. 표준 자연영상 특징과 작은 synthetic 분포가 맞지 않을 수 있으므로 정답 factor 기반 지표를 주 평가로 사용한다.

## 07  |  맥 자원과 환경

큰 사양을 전부 쓰기보다, 반복 가능한 작은 실험을 만든다

장비 가정은 M5 Pro / 64GB unified memory다. 현재 여유 RAM·저장공간·macOS/Python 버전은 확인되지 않았다. PyTorch는 MPS backend를 제공하지만 CUDA용 코드가 자동으로 같은 경로에서 동작한다는 뜻은 아니다. [S11,S12]

| 영역 | 시작 설정·운영 예산 |
| --- | --- |
| 주 모델 | 64×64, latent 4×16, trainable parameters 수백만 이하로 제한 |
| batch | AE 16 또는 32부터, flow 128부터. 실측 후 증감 |
| 메모리 | 주 프로세스 8GB 안쪽을 우선 목표로 삼는 운영 예산; 실측값 아님 |
| 저장공간 | 전체 프로젝트 10–20GB 이내를 목표. 최초 20GB 여유 확보 권고 |
| dtype | FP32 correctness 기준부터 시작; 혼합정밀도는 후속 ablation |
| 데이터 메모리 | 4,096 × 3 × 64 × 64 × FP32 = 약 0.188 GiB; activation·runtime은 별도 |
| 피할 것 | CUDA/Triton/xFormers 종속 코드, 모든 layer의 모든 patch를 무제한 캐시 |

### 설치

```text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip freeze > installed-versions.txt
python orbitlab.py doctor --device mps --out doctor_mps.json
python orbitlab.py smoke --device mps --out smoke_mps.json
```

기존 환경에 일괄 업그레이드하지 말고 새 arm64 환경을 쓴다. 현재 설치된 Python이 없으면 먼저 공식 Python 배포판을 준비해야 한다. 스타터의 auto mode는 CPU로 전환할 수 있으므로 성능 측정은 --device mps를 명시한다.

### 시간을 추정하는 방법

warm-up 20 step 이후 100 step을 세 번 측정하고 median을 기록한다. 측정 전후 torch.mps.synchronize()로 비동기 큐를 동기화한다. 총 소요 예산 = runs × steps × median seconds/step + 평가·저장 시간. 이 문서에는 Mac에서 측정하지 않은 분당 이미지 수나 “몇 분이면 끝남” 수치를 넣지 않는다.

메모리 한도를 해제하지 않는다. unsupported operator 발생 시 CPU/MPS 최소 예제로 위치를 찾는다. fallback을 켠 실행은 별도 기록하고 MPS-only benchmark와 섞지 않는다. 메모리 압박이 커지면 batch, 해상도, 캐시 순서로 줄인다. [S12]

## 08  |  15회차 실행 일정

탐색을 늘리지 말고, 회차마다 증거 하나를 남긴다

한 회차는 집중 작업 90–120분을 위한 계획 슬롯이다. 구현 경험과 장비 실측에 따라 조정하며 학습 완료 시간을 보장하지 않는다. 1–4회차만 완료해도 최소 결과물은 남게 설계한다.

| 회차 | 실행 | 완료 산출물 |
| --- | --- | --- |
| 1 | 환경·MPS 전방/역전파 parity와 smoke 검사 | doctor_mps.json / smoke_mps.json |
| 2 | 합성 데이터·분할 확인, 같은 원본 회전 보기 | 데이터 명세와 32개 입력 grid |
| 3 | 32×32 소규모 AE pilot, 시간·메모리 측정 | 첫 reconstruction + step time |
| 4 | C4 encoder/decoder·DFT·군 합성 설명 | 내 손으로 설명한 1쪽 수식과 unit tests |
| 5–6 | B1/B2 n=1,024 실행, foreground 복원 검사 | 첫 baseline 비교와 실패 사례 |
| 7 | invariant-only 제거 실험과 Fourier band probe | band energy / 정보 제거 그림 |
| 8 | A90 적합, val 선택, test·OOD 분석 | linear probe / closure / 잔차 표 |
| 9–10 | 주 비교 seed 1/2 반복, 계산량·parameter 기록 | 세 seed 비교표 |
| 11 | encoder/decoder 고정, conditional flow pilot | flow loss와 Gaussian baseline |
| 12 | 새 표본 생성, 16/32/64-step 비교 | 동일 seed 생성 grid / latent GIF |
| 13 | 조건 일치·NN·coverage 평가와 오류 분석 | 대표 성공·실패를 함께 제시한 평가표 |
| 14 | 메모리·시간·split 누출·결론 감사 | 최종 metrics와 검증 체크리스트 |
| 15 | README·6분 발표·재현 명령 정리 | 다른 사람이 실행 가능한 release |

### 오늘 바로 할 일

```text
python orbitlab.py train-ae --device mps --model equivariant \
  --size 32 --channels 8 --n 256 --batch 16 --steps 200 \
  --out runs/pilot_eq
```

파일 하나만 목표로 삼으면 runs/pilot_eq/reconstruction.png다. 첫 출력이 흐리거나 배경뿐이어도 “실험 실패”가 아니라 진단 시작점이다. 생성 모델이나 새로운 논문을 추가하기 전에 데이터, loss, shape, gradient가 맞는지 먼저 확인한다.

### 발표의 중심 문장

“회전을 알아보는 능력, 회전 정보를 잃지 않는 능력, 그 정보를 제어하여 생성하는 능력을 분리해 비교했다.” 숫자가 아직 없으면 이 문장은 목표로 쓰고, 성과 문장으로 쓰지 않는다.

## 09  |  선택 확장

핵심 결과를 얻은 뒤 하나만 선택한다

### 확장 A — DINOv3 표현 현미경

공식 ViT-S/16 21M 모델을 고정하고 200–500개의 이미지에서 원본·90·180·270도 변환을 본다. 3/6/9/12층의 CLS와 patch mean을 비교한다. dense patch는 register token을 제거하고 격자의 공간 회전을 정렬한 뒤 비교한다. 공개 구현과 모델의 접근 조건을 확인한다. [S1,S2]

질문은 “미리 학습된 표현은 무엇에 둔감하고 무엇을 보존하는가?”다. synthetic 도형과 본인 촬영 자연사진을 별도 분포로 보고, domain shift를 표시한다. 낮은 cosine similarity만으로 “회전을 모른다”고 결론 내리지 말고 learned A90 또는 정렬된 patch correspondence를 함께 본다.

기본 패키지 extract_dino.py는 CLS/patch mean을 저장한다. dense correspondence heatmap, layerwise CKA, 생성 decoder는 아직 구현하지 않았다. DINO 접근 승인이 늦어져도 주 C4 실험은 그대로 진행한다.

### 확장 B — 작은 동역학 모델과 반사실적 미래

64×64 장면 안에서 1–2개의 물체가 움직이도록 생성기를 확장한다. 첫 버전은 square boundary, 무중력·등방 마찰로 C4 대칭을 유지한다. 최소 두 프레임으로 속도를 추론하고, action을 바꾸면 어떤 미래가 나오는지 비교한다. DINO-WM은 고정된 visual feature를 예측하는 접근의 관련 선행연구다. [S7]

```text
z_(t+1) = F(z_t, z_(t-1), a_t)
F(rho z_t, rho z_(t-1), R a_t) ≈ rho F(z_t,z_(t-1),a_t)
```

constant-velocity baseline, 비구조적 latent predictor, C4 predictor를 비교한다. train 16-step / test 32·64-step rollout, centroid 오차, collision timing, 가림 뒤 재등장 위치를 평가한다. 궤적 단위 분할을 하고, 알려진 renderer로 그린 영상과 neural decoder로 생성한 영상을 분명히 구분한다.

중력 방향을 고정하면 회전 대칭이 깨진다. 중력 벡터도 입력으로 두고 함께 회전하거나, 대칭군을 줄여야 한다. 단일 이미지에서 관측할 수 없는 속도·숨겨진 상태를 맞추는 일을 “표현만 좋으면 해결”된다고 가정하지 않는다. 불확실한 미래는 점 예측과 조건부 분포를 따로 평가한다.

### 이번 회차에 하지 않을 확장

단백질·분자, LLM SAE, NeRF/3D Gaussian, 로봇 제어를 동시에 추가하지 않는다. 각각 별도 데이터·대칭·평가 기준이 필요하다. 첫 프로젝트의 가치는 적용 분야 수가 아니라 한 질문에 대한 검증 깊이에서 만든다.

## 10  |  실행 패키지와 검증 범위

바로 실행 가능한 부분과 후속 구현을 구분한다

| 파일 | 용도 |
| --- | --- |
| orbitlab.py | doctor / smoke / train-ae / analyze / train-flow / sample |
| probe_latents.py | train 적합·val 선택·test/ood 평가, linear probes와 learned A90 |
| extract_dino.py | 선택 DINOv3 특징 추출. 별도 transformers·torchvision 필요 |
| README_KO.md | 설치·정확한 실행 명령·실패 시 확인점 |
| IMPLEMENTATION_PROMPT_KO.md | 후속 코딩 회차용 구현 요구사항 |
| validation_*.json | CPU smoke·doctor·초소형 전체 경로 실행 기록 |

### 확인한 것

Linux CPU / Python 3.13.5 / torch 2.10.0+cpu에서 C4 encoder·decoder·flow, 군 합성, invariant pooling, DFT 왕복, 역전파·parameter update를 확인했다. encoder max absolute error는 0, decoder는 약 5.96×10^-8인 smoke 결과를 얻었다. 이는 구조 구현 확인이며 학습된 성능이 아니다.

AE 3 step → flow 3 step → 4-step ODE sampling → PNG/GIF → representation metrics → linear probe/learned-action fit의 경로도 실행했다. 품질을 평가할 만큼 학습하지 않았으므로 그 결과 이미지를 우수 사례로 제공하지 않는다.

### 아직 확인하지 않은 것

Mac MPS 실행·속도·메모리, 충분한 학습 이후의 품질, 모든 생성 평가 지표, 최종 여러 seed 통계는 미확인이다. DINO 스크립트는 구문 검사만 했으며 가중치를 받거나 inference를 실행하지 않았다. CPU doctor의 오차 0은 CPU 자기 비교이므로 MPS 검증으로 읽으면 안 된다.

### 최종 연구 보고 전 추가할 기능

raster/orbit hash 중복 제거, parameter/compute-matched 설정, per-sample CSV와 base-scene bootstrap, 조건 정확도·centroid·coverage·NN evaluator, best-validation checkpoint 규칙 또는 명시적인 고정-step 비교, 실행별 lockfile·코드 commit 기록을 완성한다. 후속 구현 프롬프트에 이 순서를 넣었다.

### 결과표는 지금 채우지 않는다

최종 표의 열은 model / data_n / seed / parameters / train_seconds / reconstruction / foreground_MAE / learned_action_error / condition_accuracy / coverage로 한다. 아직 측정하지 않은 칸은 미측정으로 남긴다. 문서의 목표값이나 architecture에서 따라오는 사실을 성능 측정치로 쓰지 않는다.

> 완료 기준  |  다른 사람이 새 환경에서 같은 명령으로 데이터를 만들고, 같은 분할에서 결과를 얻고, 설명과 숫자의 차이를 확인할 수 있는 상태.

## 11  |  출처와 읽기 순서

원 논문·공식 구현 중심 / 확인일 2026-09-23

아래 출처는 연구 흐름과 구현 인터페이스의 근거다. 실험 크기·일정·진입 기준·모델 비교표는 이 계획서가 제안한 설계이며, 원 논문의 실험 결과를 그대로 옮긴 것이 아니다.

[S1] Meta FAIR. DINOv3 공식 저장소. 2025-08-14 공개; 모델 크기·가중치 접근·라이선스 및 2026 업데이트.
https://github.com/facebookresearch/dinov3

[S2] Hugging Face Transformers. DINOv3 문서. CLS/register/patch 구분 및 공식 모델 사용 예제.
https://huggingface.co/docs/transformers/model_doc/dinov3

[S3] Zheng et al. Diffusion Transformers with Representation Autoencoders. 2025-10.
https://arxiv.org/abs/2510.11690

[S4] Tong et al. Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders. 2026-01-22.
https://arxiv.org/abs/2601.16208

[S5] Mur-Labadia et al. V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning. 2026-03-15 제출, 공식 구현 03-16 공개.
https://arxiv.org/abs/2603.14482
https://github.com/facebookresearch/vjepa2

[S6] Lillemark et al. Flow Equivariant World Models: Memory for Partially Observed Dynamic Environments. 2026-01-03.
https://arxiv.org/abs/2601.01075

[S7] Zhou et al. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning. 2024-11 / 2025-02 개정.
https://arxiv.org/abs/2411.04983

[S8] Balestriero & LeCun. LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. 2025-11.
https://arxiv.org/abs/2511.08544

[S9] Cohen & Welling. Group Equivariant Convolutional Networks. ICML 2016.
https://arxiv.org/abs/1602.07576

[S10] Lipman et al. Flow Matching for Generative Modeling. 2022 / ICLR 2023.
https://arxiv.org/abs/2210.02747

[S11] PyTorch. MPS backend documentation.
https://docs.pytorch.org/docs/2.14/notes/mps.html

[S12] PyTorch. MPS Environment Variables.
https://docs.pytorch.org/docs/2.14/mps_environment_variables.html

권장 읽기: 첫 실험 전에 S9의 equivariance 정의만 읽는다. 재구성 후 S3의 encoder/decoder 구분, flow 구현 전에 S10의 경로·벡터장 정의를 읽는다. S6·S7·S8은 확장할 때 읽고, 모두 읽기 전까지 구현을 미루지 않는다.
