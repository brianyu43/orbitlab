# 후속 계획 전체 항목 감사

31개 항목 중 29개는 명시한 범위에서 완료했다. 남은 항목: A04, D04. 전체 목표는 미완료다.

| ID | 요구한 증거 | 현재 판정과 실제 범위 |
| --- | --- | --- |
| A01 | 기존 release manifest의 전체 파일 해시 검사, MPS 확인, 분리된 출력 경로 | **complete_with_stated_scope**. 원본 전체 1,324파일 해시·크기 확인. 후속 출력 분리. 기존 실행 환경·MPS 학습 기록은 원본과 후속 run에 보존. |
| A02 | train/val/test/OOD orbit 중복 검사, 설정 파일·코드 해시 | **complete_with_stated_scope**. 원래 train만 fitting에 사용하고 새 validation/test/OOD의 회전 궤도 중복을 제거. 잠긴 설정·자료 해시 확인. |
| A03 | 변형별 모양·색 판정, 유효 판정률, 음성 예제 오검출, 판정 기준 민감도 | **complete_with_stated_scope**. 8종 변형의 모양/색/통과율과 4종 음성 대조의 임계값 민감도. 원래 판정기가 변형과 잡음을 받아들이는 한계를 공개. |
| A03b | 새 calibration에서 기준 선택, 독립 test와 기존 생성물에서 재평가, 사람 검증과 구분 | **complete_with_stated_scope**. calibration에서만 임계값 선택, 독립 test와 기존 생성물 재평가. 사람 검증 아님. |
| A04 | 모델·seed·seen/OOD 층화 240장, 출처 비공개 화면, 응답 CSV, 실제 사람 응답 여부 | **incomplete_requires_actual_human_responses**. 240장과 출처 비공개 HTML·연결표 준비. 실제 사람 응답 0. 브라우저 시각/상호작용 검사도 로컬 URL 정책으로 미실시. 준비를 실제 수집으로 세지 않음. |
| A05 | train으로만 probe fitting, validation 선택, 새 test/OOD 결과와 3,000-step 비교 | **complete_with_stated_scope**. train fitting/validation 선택, 새 test/OOD, 기존 6k 부분공간 및 3k/6k 같은 입력의 full 공간 비교. 누락됐던 직접 비교는 최종 감사 중 보완. |
| A06 | 같은 평가기의 복원/생성 비교, 방향별 출력 불일치, 평균 출력과 구성 출력의 오류 | **complete_with_stated_scope**. 같은 평가기의 복원/생성 8조건 및 방향별 decoder 분산 분해. 불일치가 곧 속성의 인과 분리라는 주장 없음. |
| A07 | AE 3개 각각 일반/증강/정확한 C4 flow, 가중치·샘플링·평가 noise 짝맞춤, 각 4,000 steps | **complete_with_stated_scope**. 같은 frozen C4 AE별 일반/증강/C4 flow, 9회×4,000 steps; 초기 가중치·표본·평가 noise 짝맞춤. |
| A08 | 같은 시간 flow 3종×3 seed, 전체 AE+flow 비용 및 추론 시간 표 | **complete_with_stated_scope**. 같은 시간 flow 9회, 실제 수행 step·AE+flow 비용·576장 추론 시간 기록. 같은 파라미터와 같은 연산량은 구분. |
| A09 | 데이터 seed 총 3개×초기화 3개에서 잠금 설정으로 확인, 새 test와 장면별 불확실성 | **complete_with_stated_scope**. 새 데이터 3개×초기화 3개, AE 9/flow 27개와 새 장면 중복·결과 재생. 데이터 3개와 noise 조건부 구간을 구분. |
| A10 | equivariance를 보존한 대안 decoder, 정답 상태 oracle, 같은 입력·학습 예산 비교 | **complete_with_stated_scope**. 같은 예산의 두 새 decoder와 oracle/학습 latent 12회 비교, 생성 전이 9개. 기존 pool의 탐색이며 독립 확인 아님. |
| A11 | 일반 MSE 항등식 유도, 도형별 수치 검증, invariant-only 결과와 분리된 해석 | **complete_with_stated_scope**. 표준 MSE 항등식 유도, 512장 검사, invariant-only 복원과 full C4 구분. 새로운 무한 정리 또는 모든 표현의 하한 아님. |
| A12 | 효과가 유지/소멸하는 조건, 평가기 한계, 다음 단계에 가져갈 모델 선정 이유 | **complete_with_stated_scope**. 유지되는 효과·평가기/decoder 한계·다음 단계 선정 이유 기록. 당시 남은 작업 문구는 최신 연구 메모와 종합 결과로 갱신해 해석. |
| B01 | 재현 가능한 생성, 개별 물체 ID, 충돌하지 않는 분할, 전역 회전 검증 | **complete_with_stated_scope**. 정답 물체 ID·마스크·상태·RGB를 재현 생성, 분할과 전역 회전 검사. |
| B02 | 변경한 변수 외 보존 검사, 조작 합성/순서에 대한 정답 검사 | **complete_with_stated_scope**. 색/회전/이동, 합성·순서와 변경 외 속성 보존을 정답 쌍에서 확인. |
| B03 | oracle 표시, 평평한 장면 표현과 물체별 표현의 동등 조건 비교 | **complete_with_stated_scope**. 정답 상태·정답 대상 표시, 평평한/물체별 공유 모델 동일 표본·파라미터 대조. 실제 RGB 인식 점수와 구분. |
| B04 | 정답 마스크를 입력하지 않는 평가, slot 할당·가림·재구성 결과 | **complete_with_stated_scope**. 평가 입력 RGB만 사용, slot 배정·복원·가림·물체 수 포함. 축소 고정 모델의 실패 진단이며 원논문 성능 재현 아님. |
| B05 | 모든 모델의 충분한 slot 용량, 목표 변경 정확도와 비목표 보존, 물체 수별 결과 | **complete_with_stated_scope**. 충분한 slot 용량, 새 640장·2,961명령, 115조건·68,103출력, 목표 변경/비목표 보존/전체 정답을 따로 검증. |
| B06 | 공식 자료와 split 확인, 축소 탐색/공식 프로토콜 구분, 데이터 노출 기록 | **complete_with_stated_scope**. 공식 preview 500쌍·24모델·공유 test100. 공식 fold·원본 픽셀·중복·노출 감사. 전체 공식 SVIB 및 LPIPS 미실행을 공개. |
| B07 | 가림·물체 수·조합 복잡도를 각각 바꾼 곡선, 오류 유형별 예제 | **complete_with_stated_scope**. 가림 정도·물체 수·합성 명령·오류 유형별 표/곡선, 고정 첫 예제 영상. 기술 통계를 인과 효과로 일반화하지 않음. |
| C01 | 상태와 영상 일치, 동역학 검증, 궤적 단위 split | **complete_with_stated_scope**. 이산 충돌 simulator, 상태/영상과 궤적 split·에너지·겹침 수치 검사. 연속/실제 물리 검증 아님. |
| C02 | T(Rs,Ra,Rc)=R T(s,a,c) 검사와 조건 고정시의 반례 | **complete_with_stated_scope**. 상태·행동·외력을 함께 회전한 정답 동치와 외력을 고정할 때의 반례. 정확한 적용 범위 공개. |
| C03 | 등속도/일반/물체별/대칭/완화/조건 포함 모델, 동일 입력 정보 | **complete_with_stated_scope**. 63개 모델과 등속도·알려진 물리 참조; 정답 상태·환경의 한 단계만 비교. 초기화·표본·재사용 감사. |
| C04 | 같은 과거 관측 길이, oracle와 분리된 지각·조건 추론 오차 | **complete_with_stated_scope**. 동일 과거 4장, RGB/측정 모델과 정답 상태·환경 교체, 307,200개 다음 예측. 인식과 정보 추정 분리. |
| C05 | 학습보다 긴 rollout, 물체 identity·위치·속도·충돌 시점 | **complete_with_stated_scope**. 61단계 자율 예측, 미학습 힘/속성/물체 수, 위치·속도·접촉 후보/실패 평가. 물체 색/존재는 복사하므로 학습 추적 성공으로 주장하지 않음. |
| C06 | 같은 초기 상태에서 개입한 정답 궤적, 비목표 물체의 물리적 반응도 포함 | **complete_with_stated_scope**. 같은 과거의 반대 행동 정답과 자율 반응을 대상/비대상별 검사. 없는 네 물체 반사실을 만들어내지 않음. 비유한 분모 공개. |
| C07 | 방향 효과·관측 길이·조합 난도별 효과, 누락 조건의 식별 불가능성 구분 | **complete_with_stated_scope**. 세 독립 데이터 반복·108개 정보 누락 조건·72개 길이 모델, 520개 전체 집계, 고정 접촉 집단·정보 모호성 사례와 종합 결론. |
| D01 | 실제 독창성 후보와 기존에 해결된 부분을 구분한 표 | **complete_with_stated_scope**. 9개 가까운 방법 본문·가능한 공식 코드 비교. 공개 코드 없는 방법은 미검토 표시. 타 논문의 전 학습 재현이나 독창성 증명 아님. |
| D02 | 원인 분리, 계산 비용, 조건 변화, 조합/장기 예측의 결과 | **complete_with_stated_scope**. 원인 분리·계산 비용·조건·개입/조합·긴 미래: 10묶음 25파일. 원자료·수치 검사·이미지 디코딩·명시된 시각 검토 범위 연결. |
| D03 | 가설·방법·결과·실패·한계, 수치와 원자료의 연결 | **complete_with_stated_scope**. 가설·방법·결과·실패·한계를 담은 짧은 연구 메모와 쉬운 설명. 수치/근거/미실행 제안/사람 미수집을 분리. |
| D04 | 모든 항목의 실제 증거, 재생 검사, 기존 release 불변 검사, 미완료 항목 공개 | **incomplete_archive_and_relocated_replay**. 원본 불변·전체 항목 감사·별도 ZIP·모든 ZIP 바이트 재검사·다른 경로에서의 CPU 표본 재생. 마지막 두 검사는 봉인 뒤 별도 영수증으로 확인. |

검사 범위: 파일이 있다는 사실만으로 완료를 판정하지 않았다. 원래 보고서·verifier 범위를 검토하고 항목별 수치·재생·집계 근거를 연결했다. 이 감사 도구가 전체 모델을 다시 학습한 것은 아니다. 각 자료의 해시와 실제 검사 필드는 연결 JSON에 있다.

[scope_audit_before_archive_v1.json](scope_audit_before_archive_v1.json)
