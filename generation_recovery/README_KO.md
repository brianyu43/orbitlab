# OrbitLab 생성 성공률 회복 실험

원래 프로젝트와 `followup/` 결과를 보존하고, 새 실험·자료·보고서를 이 폴더에 분리했다. **최종 확인까지 실행 완료.** 생성 성공률은 뚜렷하게 개선됐지만 사전 정의한 성공 그림의 크기 다양성 기준을 통과하지 못했으므로 최종 연구 이정표는 미달이다.

읽는 순서:

1. [최종 결과와 한계](confirmation_v1/FINAL_REPORT_KO.md) · [요약 그림](confirmation_v1/figures/confirmation_overview.png)
2. [P0 원인 진단](reports/P0_FINDINGS_KO.md)
3. [P1 조건·디코더 분리 실험](reports/P1_FINDINGS_KO.md)
4. [P2 직접 픽셀 기준선·P3 학습량 실험](reports/P2_P3_FINDINGS_KO.md)
5. [최종 설정 고정](confirmation_v1/lock.json) · [출처 필드 실행 수정](confirmation_v1/execution_amendment_01.json) · [결과 원자료](confirmation_v1/evaluation/summary.json) · [재계산 검증](confirmation_v1/evaluation/verification.json)

핵심 수치는 새 합성 데이터 seed 3개 × 초기화 3개에서 대응 비교한 엄격 통과율이다. 4천→1만6천 단계 flow의 본 조합 평균은 22.01→40.20%, 제외 조합은 16.02→32.70%였다. 모든 생성물의 다양성은 통과했지만, 엄격 통과물의 크기 분포는 세 seed 모두 기준을 넘었고 모양 2·색 0의 성공 수는 조건별 최소 표본에 못 미쳤다. 사람 판독·현실 데이터 일반화는 주장하지 않는다.

기록된 점검은 [P1](evaluation/p1_v1/verification.json), [P3](evaluation/p3_steps16k_v1/verification.json), [확인 학습 전](confirmation_v1/pre_evaluation_training_audit.json), [최종 확인](confirmation_v1/evaluation/verification.json)에 있다. 실행 환경은 CPU 4스레드이며 독립적인 패키지 설치나 외부 계산 자원을 사용하지 않았다.
