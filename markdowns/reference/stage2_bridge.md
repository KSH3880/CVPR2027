# Stage2 Bridge

센서 없이 객체 토큰만으로 "다리가 없으면 기다리고, 연결되면 건너기"를 학습하는지 보는 실험이다. 설계 배경은 [bridge_prerequisite_plan](../bridge_prerequisite_plan.md)에 있다. 무대·gap·외력 다리·추락 종료는 참고 repo `TokenHSI_square_sensor_dev`의 `HumanoidMABridge`를 센서 없이 옮겼다. 실행은 [config](../config.md)를 따른다.

## stage2_bridge_scripted (phase 1)

- 2명·2048환경·4물체 토큰(각자 과제 상자 2, 다리 1, distractor 1)이며 600step(20초)이다. 두 사람은 모두 무대 A(4~8m, 높이1.5m) 위에서 시작하고, 과제는 gap 건너편 무대 B(4~5m) 위 자기 상자의 SIT/CLIMB(50/50)다. 참고 repo의 단독 무대 장면은 센서 없이는 가장자리를 볼 수 없어 제외했다.
- 다리는 중력·회전 없는 상자를 PD 외력으로 채널을 따라 착지 위치까지 옮긴다. 20%는 시작부터 연결되어 있고 나머지는 도착 시각1~18초(로그균등)·평균 속도0.3~1.5m/s로 움직인다. 다리는 edge 없는 object 토큰으로 관측되며 센서는 없다.
- 크기: gap1.5~2.5m, 다리 길이 gap−2×여유, 폭1.0~2.0m, 높이 무대−1cm이다. 다리와 무대 벽 사이 여유는 한쪽5cm다(참고 repo1cm). phase 2의 push 상자와 같은 틈 조건을 쓰기 위한 값이다.
- 시작 위치·상자 배치·near start(0.5→0.2, 300000step)는 참고 repo와 같다. 상자는 무대 B 가장자리에서1~2m 띠에 세 칸으로 놓는다. 비활성 goal 토큰은 무대 B 위 띠 중앙에 둔다.
- 사람 node의 root 높이는 사람마다 발밑 지형(무대 A/B·다리, 과제 상자 제외) 높이를 뺀 값이다. GTA pose는 실제 env-local 위치를 유지한다.
- 무대에서 시작한 사람은 무대 밖 바닥에 몸이 닿으면 종료하고, 그 외에는 지형 기준 기존 낙상 조건을 쓴다(`terrain_fall`). 한 명이 종료되면 장면 전체를 reset한다. 바닥 시작 예외(`_floor_start`)는 phase 2용이며 phase 1에서는 모두 꺼져 있다.
- 관측 끝에 사람별 선행 index(M개, 없으면 −1)를 붙이고 human SA4는 이 값으로 SELF/NONE/BEFORE를 만든다. phase 1은 모두 −1이며 phase 2와 같은 network 구조를 유지한다.
- Stage1은 PUSH raw(정육면체) 13500을 전이한다. 과제 embedding5칸·AMP family4(133차원/step)를 그대로 받는다. actor encoder·관측 RMS는 동결, coordination은 항등 초기화다. 보상·PPO·AMP 설정은 `stage2_before_ground`와 같다. `bridge_contract`로 다른 실험과 checkpoint를 분리한다.
- 지표: `relation/bridge/{waiting_near_gap_fraction, falls_before_arrival_per_agent, falls_after_arrival_per_agent, crossed_agent_fraction, far_stage_step_fraction, arrived_episode_fraction, approach_arrived_fraction, episodes}`.

## stage2_bridge_push (phase 2)

- 2명·2048환경·4물체·1200step(40초)이다. 매 에피소드 한 명(crosser)은 무대 A, 다른 한 명은 바닥에서 시작하며 역할은 무작위다. crosser 과제는 phase 1과 같은 무대 B 위 SIT/CLIMB다.
- 장면: 의존 push 60%(바닥 사람이 정육면체 O2를 채널로 밀어 다리를 만든다), 다리 있음 20%(O2가 이미 gap에 있고 바닥 사람은 바닥 상자 SIT/CLIMB), 의존 없는 push 20%(O2가 이미 gap에 있고 바닥 사람은 다른 정육면체 O3를 바깥쪽으로 민다). 의존 없는 push는 "동료가 밀고 있으면 기다린다"와 선행 관계 label을 분리하기 위한 장면이다.
- 토큰: crosser 상자, 바닥 사람의 과제 물체, 나머지 정육면체(O2 또는 O3, 다리 있음 장면은 O2), 무대 B distractor. 의존 장면에서만 crosser의 선행 index가 바닥 사람이고, crosser 목표 edge는 바닥 사람의 AT edge를 `prereq_mask` gate로 기다린다.
- O2·O3는 PUSH raw와 같은 정육면체(1.2~1.7m, 12~20kg, 본체0.9·밑판0.3/1cm)다. 무대 높이는 O2 한 변+1cm, gap은 한 변+2×여유(한쪽5cm)다. O2는 착지 위치에 가까운 채널 끝에서0.3~1.0m 바깥에 놓이고 채널 방향으로 민다(거리 약2~5.5m).
- 바닥 사람은 PUSH raw bank의 접촉 자세(raw RSI, 손 높이 보정 없음)로 시작한다(`rsi_probability`1.0). 다리 있음 장면의 바닥 사람과 crosser는 loco로 시작한다. 바닥 사람은 바닥 접촉 종료에서 제외한다.
- PUSH edge는 접촉(양손 면 XY≤18cm·힘), AT edge는 `phi_at≥0.9`·높이1mm로 판정하고 상자 속도 페널티를 제외한다(PUSH Stage1과 같음). AMP family3과 push/push_loco 데모를 쓴다. motion은 `dataset_stage1_push.yaml`이며 push motion 폴더는 `TokenHSI_stage1_push`로의 심링크다.
- 지표: phase 1의 `relation/bridge/*`(crosser 기준)와 장면별 `relation/bridge/{dependent,present,independent}_{crossed,crosser_falls,push_success,episodes}`.

## stage2_bridge_push4

- 4명·1024환경·8물체 토큰·1200step이다. 매 에피소드 역할을 섞어 다리 PUSH 1명, A→B crosser 1명, 독립 과제 2명을 배정한다. 다리 없음→의존 PUSH 연결 장면만 사용한다.
- crosser는 B 위 자기 상자에 SIT/CLIMB/CARRY를 각 1/3로 수행한다. CARRY는 B 위 상자를 들어 같은 B의 AT 목표로 운반한다. 독립 과제는 SIT/CLIMB/CARRY/PUSH 각 1/4이며 A·B·바닥 중 서로 다른 두 위치를 무작위 선택한다. 각 과제의 물체·목표는 같은 표면에 있다. 독립 PUSH에도 기존 raw 정육면체와 접촉 RSI를 사용하며, 무대 위 PUSH RSI는 해당 높이만큼 올린다.
- 두 무대의 XY 크기는 환경 생성 시 각각 독립적으로 4~8m(0.5m 간격)에서 뽑는다. 에피소드마다 A의 XY 이동과 B의 측면 offset을 새로 뽑으며 두 무대는 평행하다. gap은 다리 상자의 실제 폭+10cm로 양쪽5cm의 기존 여유를 유지한다. 상자 크기1.2~1.7m·질량12~20kg·밑판 물성은 유지한다. 무대 높이는 다리 상자 높이+1cm다.
- 두 무대가 겹치는 띠에서 다리 폭+양쪽0.8m를 확보하고 그 안에서 착지점을 균등 샘플링한다. crosser 시작점·B 상자·운반 목표는 별도로 샘플링한다. 사람은 기본0.8m, 상자 외곽은0.5m의 내부 여유를 두고 상자/사람 간 겹침을 거부한다. PUSH 사람의 샘플 기준 여유는0.5m이며 실제 raw RSI 자세는 별도로 검사한다. 앞선 배치 때문에 공간이 부족하면 같은 역할·과제·크기를 유지한 채 배치를 재추첨한다.
- 물체 토큰은 과제 물체4개와 바닥 distractor4개다. 물리 bank는 기존 단독 bank16개와 PUSH 정육면체4개이며 미사용 물리 상자는 멀리 격리한다. 과제 물체는 공유하지 않는다.
- crosser→다리 담당자에만 BEFORE를 두고 나머지 사람 쌍은 NONE, 대각은 SELF다. 선행 label은 에피소드 중 고정이며 gate는 상태/성공 보상에 적용하고 접근 보상은 유지한다. 별도 기다림 보상·다리 경유 waypoint는 추가하지 않는다.
- Stage1 PUSH raw13500에서 새로 전이한다. actor encoder·관측 RMS 동결과 coordination 항등 초기화, 기존 PPO/AMP를 유지한다. 사람 수에 맞춰 agent minibatch는16384(scene4096)다. 기존 2명 실험과 checkpoint 계약·config·output을 분리한다.
- `relation/bridge/independent_{sit,climb,carry,push}_fraction`과 `surface_{floor,stage_a,stage_b}_fraction`은 새 과제 추첨 수를 분모로 한다. bridge 건넘·추락 지표는 지정된 crosser 한 명 기준이다. 다리 연결 지표는 기존과 같이 목표 XY20cm 진입 이력이며 물리적 정렬 전체를 판정하지 않는다.
- CPU 검사는 `test_bridge_push4.py`, GPU 초기화·부분 reset·추론·10개 장면 촬영은 `check_bridge_push4.py`를 사용한다. 확인 산출물은 `output_etc/check/stage2_bridge_push4/`에 둔다.

## stage2_bridge_push4_mix50

- `stage2_bridge_push4`에서 시작 배치만 바꾸는 별도 실험이다. 다리 상자는 기존 무대 밖 시작50%, 목표까지 Y 방향 잔여 거리0.1~1.5m인 근접 시작50%를 섞는다. 밀기 방향과 접촉 RSI 사람의 배치를 함께 맞춘다.
- 근접의 절반은 crosser 시작점→다리 완성 목표→자기 과제 상자를 직선으로 배치한다. 전체 비율은 기존 무작위50%·근접 무작위25%·근접 직선25%다. 직선 방향과 사람/상자의 위치는 무대 내부 여유를 지키며 샘플링하고, 두 무대의 gap 경계에서 다리 측면까지 최소0.2m를 확보해 대각선이 다리 옆으로 벗어나지 않도록 한다. 운반 과제의 B 위 최종 AT 목표는 별도로 무작위 배치한다.
- 거리·직선 여부는 에피소드 graph를 만들 때 한 번 추첨하며 기하·물리 배치 재시도 중 유지한다. 기존 `stage2_bridge_push4`는 설정을 생략해 기존 샘플러와 난수 호출 순서를 유지한다.
- 0.1~0.2m 시작은 처음부터 기존 다리 도착 지표(목표 XY20cm 이내)를 만족할 수 있다. BEFORE 관계는 유지하고 실제 gate는 현재 AT 성공으로 계산한다. 별도 `bridge_present` 장면은 추가하지 않는다.
- `relation/bridge/start_{original,near_random,near_aligned}_fraction`은 새 에피소드 추첨 비율이다. 같은 접두사의 `episodes`는 종료 장면 수, `arrived`·`crossed`·`crosser_falls`는 해당 종료 장면 기준 비율이며 `start_near_fraction`은 두 근접 군의 합이다. 단계별 행동 평가는 첫 에피소드 궤적에서 군별로 구분한다.
- 같은 Stage1 PUSH raw13500에서 새로 전이하며 출력·checkpoint 계약을 분리한다. actor encoder·관측 RMS 동결, 4명·1024환경·8물체, 기존 보상·성공 조건·PPO/AMP·무대·상자 물성을 유지한다. 검증은 `test_bridge_push4.py`와 `check_bridge_push4.py`, 산출물은 `output_etc/check/stage2_bridge_push4_mix50/`를 사용한다.

이 머신의 GPU 실행은 `TOKENHSI_CONDA_ENV=tokenhsi_sm120`이 필요하다(`tokenhsi` 환경은 CPU torch). 검증 기록은 [changelog](../../changelog.md), 확인 산출물은 `output_etc/check/`에 둔다.
