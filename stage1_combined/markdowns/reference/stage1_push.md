# Stage1 PUSH 구현 메모

- `stage1_v2_fixed_push`를 별도 환경·학습 config와 train/test/VNC wrapper로 추가했다. 기존 설정은 기존 실행 경로를 사용한다.
- 기본 2명·2048환경·4물체 토큰. 환경군을 기존 1024/PUSH 1024로 나누어 transition을 50:50으로 확보한다. 기존 과제 안의 비율은 SIT/CLIMB/AT/ON_TOP=10/25/32.5/32.5%를 유지한다.
- PUSH는 1.3m 정육면체·25kg·마찰 0.6. 로컬 −Y 면을 접촉면으로 고정하고 초기 +Y 법선 방향 2~5m에 목표를 둔다. 목표는 이후 고정하며 물리 움직임은 자유롭다. 기존 관측 차원을 유지한다. 4물체 토큰을 위해 담당 상자 2개와 경로 밖 실제 distractor 2개를 둔다.
- PUSH 접촉/AT 보상·0.9/0.1 공유·root 근접 페널티·상자 속도 페널티 제외는 PUSH 환경에만 적용한다. 유효 접촉은 제어 step 내 물리 tick의 OR로 집계한다.
- fixed60000 가중치를 전이하고 task 임베딩 4→5, AMP family 3→4와 discriminator 입력을 확장한다. 기존 tensor/입력 열을 보존하고 새 optimizer/epoch으로 시작한다. 학습률은 1e-5다.
- 원본 teacher를 동결하고 기존 과제에만 KL(기본 0.001)을 적용한다. 정책·AMP 입력 정규화는 전이 통계를 고정한다. 가치 정규화도 전이 후 현재 return으로 학습한다.
- PUSH AMP는 보행 4clip 33%/PUSH 21clip 67%. 기존 과제 expert는 유지한다. 보행/interaction RSI는 50:50이며 선별 bank의 clip·해시·물성을 검사한다. 팔 보정은 reset과 초기 AMP history에만 적용한다.
- `OLD_TASK_FRACTION`, `TEACHER_KL_COEF`, `STAGE1_CHECKPOINT`로 최초 전이를 조절한다. PUSH resume는 저장 설정과 optimizer를 복원하며 비율/KL만 변경할 수 있다.
- TensorBoard의 `relation/retention/*/current_success`, `relation/push/*`, `losses/teacher_old_task_kl`로 과제별 상태를 구분한다. 현재 step 성공과 최종 에피소드 성공은 별도다.

명령: [config](../config.md). 장기 수렴·기존 과제 성공률 보존은 아직 평가하지 않았다.

검증: CPU 167개 통과. 2048환경 epoch2/frame262144 학습·저장, 4환경 resume(epoch3/비율·KL override), headless 평가, 4환경 학습·1환경 테스트 noVNC의 실제 렌더·움직임·HTTP200/WebSocket101/RFB를 확인했다. 기존 정책 출력은 전이 직후 수치 오차 범위에서 유지되고 teacher·정규화 불변, 모델/loss/gradient finite, PUSH 임베딩 갱신을 확인했다. 결과: [검증 기록](../../output_etc/check/stage1_v2_fixed_push_setup/validation.json).

2026-10-19 질량 변경: PUSH 기본 질량을 25kg으로 올리고 전용 `mass25_bank.json`을 연결했다. 기존 20kg bank와 선별 이력은 보존한다. 11개 RSI 후보는 재사용하며, 25kg 전 후보 PD 재선별은 수행하지 않았다.

25kg 변경 검증: CPU12개 및 2048환경 짧은 학습·저장 통과. simulator 질량 약25.000004kg, 모델/loss/gradient finite·teacher/RMS 불변을 확인했다.

테스트 모션 로딩: 선택한 task와 양수 확률의 시작 동작만 로드한다. PUSH 보행 시작은 `push_loco` 4개, PUSH RSI 시작은 bank에서 사용하는 7개 clip만 로드하고 원본 motion ID를 내부 ID에 매핑한다. 보행만 쓰는 테스트는 PUSH RSI bank/21clip 해시 검사를 생략한다. 학습 모션 구성은 유지한다.
