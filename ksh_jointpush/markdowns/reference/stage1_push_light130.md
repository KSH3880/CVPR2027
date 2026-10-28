# Cuboid RSI uniform:10–15kg·20cm·130°

`stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130`은 기존 cuboid RSI uniform에서 다음을 변경한다.

- 같은 크기와 부피 선형 대응을 유지하며 질량 범위를10–15kg으로 낮춘다.
- PUSH의 목표 성공·성공 보상 포화·평가는 수평 거리≤0.20m, 기존 높이 오차≤0.001m를 사용한다. 연속 거리 보상 식은 유지한다.
- 보행 spawn은 지정 밀기 면의 정면을 중심으로±65°(총130°)다. RSI 자세는 원본을 유지한다.

KL0·원본 fixed78000 전이·새 optimizer/epoch·2048환경·기존/PUSH50:50·PUSH 내 RSI/보행50:50·RSI2–12m·보행2–10m·PUSH1200step을 유지한다. 마찰·크기·모션·RSI 후보는 유지하며 새 물성 bank는 후보 재사용을 명시한다. 새 질량에 대한 후보별 RSI screening 결과로 간주하지 않는다.

## 18K 감시와 조건부 전환

- 현재 학습: `tokenhsi-push-rsi-uniform-kl0-20261024-194136.service`.
- 감시: `tokenhsi-push-rsi-uniform-18k-monitor-20261025.service`, 상태 확인 간격30분.
- 설정·상태·로그: `output_etc/check/push_uniform_18k_monitor_20261025/{plan.json,state.json,monitor.log}`.
- 18000 저장본을 동일 초기512건·seed42·최대200초로 평균/확률 행동 각각 평가한다. 성공은 원래 에피소드에서 한 번이라도 목표를 달성한 경우다.
- 평균 행동 성공≥115/512(22.4609%)와 정체≤255/512(49.8047%)를 모두 회복하면 기존 학습을 유지한다. 기준은 같은 조건의13000 저장본이다.
- 정체는 접촉 후·첫 성공 전·목표까지0.5m 초과에서 상자 XY속도<0.05m/s와 사람 XY속도<0.1m/s가10초 연속인 경우다.
- 하나라도 미달이면 기존 서비스만 종료하고 `tokenhsi-push-cuboid-m10to15-r20cm-a130-20261025.service`로 새 variant를 fixed78000에서 시작한다. 확률 행동 결과는 보조 지표다.
- 판단은 `decision.json`에 저장한다. PID/cwd/명령 또는 고정한 입력 파일이 바뀌면 자동 전환을 수행하지 않고 `state.json`에 오류를 남긴다.

두 학습과 감시기는 사용자 systemd 서비스로 실행하며 `Linger=yes`를 확인한다.18K 평가는 기존 variant의 기존 성공 반경을 사용하고, 다음 variant만20cm를 사용한다. 상태 확인 지연과 평가 시간 동안 기존 학습은18000을 넘길 수 있다.
