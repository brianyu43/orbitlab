# 가까운 연구의 본문·구현 비교 기록

2026-09-23. D01 진행 기록이다. 아래 방법의 핵심 절과 공식 구현을 확인했다. 전체 문헌에 대한 독창성 부재/존재 판정을 완료한 것은 아니다. 후기 기하학적 편향·편집 연구와의 겹침은 추가 검토가 남아 있다.

| 방법 | 확인한 핵심과 입력 | OrbitLab에 주는 제약 |
| --- | --- | --- |
| Slot Attention | RGB feature를 경쟁 attention과 공유 GRU로 slot에 할당하고, slot별 RGB·mask를 합쳐 복원한다. 논문 §2.1–2.2와 공식 `model.py` 확인. | 물체별 표현, slot 순서 일관성, slot decoder 자체는 기존 방법이다. 이번 B04는 작은 데이터·축소 구조·다른 손실의 개발 실험이다. |
| Invariant Slot Attention | 물체 위치·크기·방향에 상대적인 좌표를 attention과 decoder에 쓴다. §4, §5.3, App. C.2와 `invariant_attention.py`, `decoders.py` 확인. | “물체와 기하학적 대칭을 결합”하는 설명만으로 차별화할 수 없다. 회전 추정은 mask의 주축을 쓰는 근사이며, 논문에서도 회전 추가 효과는 데이터별로 섞였다. |
| Neural Systematic Binder | slot 내부를 여러 factor block으로 나누고 block별 GRU·prototype memory로 갱신한다. §2.1–2.3과 공식 `sysbinder.py` 확인. | 물체 분리와 속성 분리는 서로 다른 평가다. 일반 slot을 얻거나 선형 probe가 잘 된 것만으로 독립 속성 조작을 입증할 수 없다. |
| Dreamweaver | 영상의 recurrent block-slot을 유지하고 미래 프레임 토큰을 예측한다. §2–3, App. C.5와 공식 `binder.py`, `model.py` 확인. | 물체·속성·운동의 재조합은 이미 핵심 목표다. 표현 학습과 편집할 block을 찾는 후처리를 구별해야 한다. |
| SEN | 관측을 알려진 군 작용을 갖는 표현으로 보내고 equivariant transition과 함께 학습한다. §4–5와 공식 `sen/net/model.py`, `transition.py` 확인. | 대칭 표현을 세계 모델에 넣었다는 사실 자체는 독창성이 아니다. 일반 latent에 임의 군 작용을 붙여 평가하는 비교도 피해야 한다. |

## 구현에서 직접 확인한 부분

- **Slot Attention:** attention의 softmax는 slot 축에 적용하고, 입력 축으로 다시 정규화한 가중 평균을 GRU에 넣는다. 학습 가능한 공통 Gaussian 초기화와 공유 업데이트를 사용한다. 이를 작은 PyTorch 구현의 순서 일관성 검사에 반영했다. [원문](https://arxiv.org/html/2006.15055v2), [공식 코드](https://github.com/google-research/google-research/tree/master/slot_attention)
- **ISA:** slot 위치·scale·회전 행렬을 별도로 다루고 상대 좌표를 decoder에도 전달하는 분기가 있다. 논문의 회전 주축 추정은 대칭 모양에서 모호할 수 있고, 회전 범위를 제한한다. OrbitLab의 정확한 전역 C4 작용과 같은 보장으로 취급하지 않는다. [본문·부록](https://proceedings.mlr.press/v202/biza23a/biza23a.pdf), [공식 코드](https://github.com/google-research/google-research/tree/master/invariant_slot_attention)
- **SysBinder:** `BlockGRU`, `BlockLinear`, `BlockPrototypeMemory`를 사용하고, 이미지 decoder는 block coupling과 토큰 transformer를 사용한다. 현재의 작은 독립 물체 상태 MLP는 그 방법의 재현 모델이 아니다. [본문](https://arxiv.org/html/2211.01177v3), [공식 코드](https://github.com/singhgautam/sysbinder)
- **Dreamweaver:** 코드에서 시간별 binder 출력과 마지막 상태를 이용한 미래 토큰 예측, 미래 시점 선택과 step indicator를 확인했다. 논문 App. C.5에서는 정답 factor와 mask matching을 이용한 probe 및 feature importance 검토로 편집할 block을 찾는다. 이를 포함한 조작 전체를 정답 정보가 전혀 없는 과정으로 서술하지 않는다. [본문·부록](https://arxiv.org/html/2501.14174v5), [공식 코드](https://github.com/ahn-ml/dreamweaver-release)
- **SEN:** 구현은 latent transition 오차와 음성 state에 대한 hinge를 결합한 contrastive loss를 사용한다. `TransitionGNN_C4`는 회전 구조를 가진 edge/node 함수를 쓴다. 현재 OrbitLab의 pixel 생성 손실과 수치를 직접 비교할 수 없다. [논문](https://proceedings.mlr.press/v162/park22a/park22a.pdf), [공식 코드](https://github.com/jypark0/sen)

## 앞으로 검증할 질문의 위치

후속 연구의 중심 후보는 **환경의 방향성, 환경 정보를 관측할 수 있는 정도, 미관측 조합과 긴 미래에서의 오차를 한 통제 실험에 함께 두는 것**이다. 이 조합이 기존 연구에 없다고 아직 확인한 것은 아니다. 상태만 돌릴 때와 상태·행동·환경을 함께 돌릴 때를 나누고, 조건 제공/추론/비공개를 같은 입력 정보의 모델끼리 비교해야 한다.

현재의 상태 oracle 실험은 독립적인 개입에서 물체별 공유의 예상 이점을 확인했다. 원래의 C4 생성 실험은 flow 제약과 decoder의 효과를 분리했다. 어느 쪽도 이 환경 조건 가설을 검증한 것은 아니다. 조건에 따른 대칭 실험, 외부 SVIB 모델 평가와 더 가까운 최신 방법 검토를 마친 뒤 주장 범위를 정한다.

## 자료 보존과 범위

`references/slot_attention/manifest.json`과 `references/nearest_methods_v1/manifest.json`에 실제 받은 URL, 바이트 수, SHA-256을 저장했다. 일부 Hugging Face markdown 조회는 404였으므로 버전이 지정된 arXiv 원문으로 보완했다. 원본 공개 코드는 참조만 했고 실행하거나 모델 전체를 재학습하지 않았다. 배포 시 각 원본의 라이선스를 따로 확인하며, 논문 원문을 연구 배포물에 임의로 재배포하지 않는다.
