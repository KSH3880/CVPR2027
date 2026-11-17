# Stage2 sensor 900

별도 실행 컨피그 `stage2_sensor_900`. 기준 구현은 `/home/user/jhh/Projects/stage_1_2_base_fuv1`의 square sensor·cliff/bridge/room·배치/초기화 규약이며 통합 Stage2의 BEFORE/COUPLED/mixed 그래프·보상·개인 PUSH 목표를 유지한다. 기존 `stage2_distill_900` 증류 코드는 그대로 둔다.

## 전이와 학습

- 원본: `output/stage2/Stage2Distill900_00001500.pth`, epoch1500, SHA256 `fcadf2297bedf8e3a8d010fd93d045f63dab412f0c85e65d7b89de4cc15a8a1e`.
- 원본279개 state tensor를 복사한 뒤 actor encoder·actor Stage2 coordination·sigma·입력 RMS만 동결한다. Action head, critic encoder/coordination/value head 전체, 센서 분기, AMP discriminator를 학습한다. `TokenHSI_square_sensor`의 학습 범위에 사용자 지정 actor coordination 동결을 추가한 구성이다.
- actor/critic 각각 센서 MLP `2304→256→128→64`와 `64→1024` 무편향 투영을 학습한다. 투영을0으로 초기화하여 기존 head 첫 선형층 출력에 더한다. 초기 action/value 네트워크 출력은 원본과 수치 허용오차 내 동일하며, 학습 후 SENSOR_MASK=1은 현재 head를 사용하므로 원본 정책 복원을 의미하지 않는다.
- 입력/AMP RMS는 원본에서 복사하되 입력 RMS만 고정한다. Value RMS는 square sensor처럼 새 통계로 시작하고 AMP/value RMS를 학습 중 갱신한다.
- PPO lr `2e-5`, horizon32, minibatch16384, mini_epochs6, task/AMP reward 각0.5. AMP 판별기도 업데이트한다. 이 실험에는 distillation loss/teacher 로딩이 없다.
- 새 run/checkpoint 이름은 `Stage2Sensor900Square`다. checkpoint에 원본 경로·hash·epoch, sensorScene·배치 계약, `sensor_training_contract=square_sensor_frozen_actor_coordination_v1`을 저장한다. 기존 전체 동결 checkpoint는 테스트만 가능하며 이 학습의 resume으로 사용하지 않는다. 새 계약의 resume은 모델·optimizer·통계를 복원한다.

## 센서와 시나리오

- 사람 root 중심, heading 정렬, ±7m(14×14m),48×48 높이 격자. 지면·상자·구조물 윗면 높이에서 root 높이를 빼고 ±3m clip 후3으로 나눈다. room 벽은 높이 센서와 CPA에 포함한다. 별도로 계산하는 발밑 지형 높이에서는 벽을 제외한다.
- 사람당2304개, 전체9216개를 기존1272 관측 뒤에 추가(총10488). 센서 suffix는 입력 RMS 정규화에서 제외한다.
- reset마다 cliff/bridge/room 각각1/3. 구조물은 고정 actor이며 scene origin을 바꿔 해당 장면을 활성화한다. root 높이는 발밑 지형 기준, feet mesh 최저점으로 초기 바닥 관통을 방지한다.
- cliff/room은 현재 통합 Stage2의 BEFORE/COUPLED/mixed 그래프 샘플러를 사용한다. PUSH 그룹은 구조물 밖 바닥에 배치하며 공유 상자·그룹 상대 배치·목표 경로를 함께 이동한다.
- cliff의 독립 비PUSH 사람은 도달 불가 과제(unreachable)를 유지한다. 정식 과제·interaction group 성공률에서 제외하고 별도 건수를 기록한다. 낙상은 전체 사람 기준으로 기록한다.
- room은 기준 repo의 방·벽·출입구 배치, pickup 경로 혼합, wall CPA와 지형 낙상 처리를 사용한다.

## Bridge 샘플링

- reset마다 pusher 수1/2/3을 균등 샘플링하고 별도의 crosser1명을 둔다. 나머지2명이 있을 때 BEFORE 쌍50%, 아니면 각각 독립; 나머지1명은 독립이다.
- PUSH native 크기에서 X만 pusher 수만큼 늘린다. Y/Z와 density3은 유지한다. 환경 생성 때 bridge1/2/3 상자와 고정 플랫폼 변형을 미리 만들며 reset에서 actor scale은 바꾸지 않는다.
- 실제 bridge 상자 X에0.2m를 더한 플랫폼 간격. 여러 pusher가 동일 physical box를 공유하고 기존 개인 구역 목표·gate를 사용한다.
- crosser SIT/CLIMB/AT/ON_TOP 각25%. original/near/aligned 시작50/25/25%; AT/ON_TOP은 carry-from-start50%. 원본 방식대로 별도 prerequisite/ENABLE_GAP은 추가하지 않는다.
- bridge 도착·crossing·crosser fall과 on-face 상태를 pusher 수별로 기록한다. 이 온라인 통계는 고정 평가 성공률이 아니다.

### 추론 전용 입구 모따기

`BRIDGE_ENTRY_CHAMFER=1`은 두 플랫폼에서 통로에 닿는 양 끝 모서리만 X0.20m·Y0.40m 사선으로 자른다. 정적 플랫폼은 같은 시각/충돌 convex mesh를 사용하고 높이맵·발밑 지형도 그 footprint를 계산한다. 최종 landing은 기존0.8m 여유 안에 있어 모따기 깊이0.4m보다 안쪽이며, 상자 크기·gap·시작 배치·보상은 그대로다. 기본0, 학습 사용은 금지한다. 생성 자산과 녹화 산출물은 `output_etc/check/bridge_chamfer_20261108/`에 저장한다. 현재 체크포인트로 형상 변경의 추론 동작을 확인하는 옵션이며 재학습이나 성능 향상을 의미하지 않는다.

### 원본 형상·끝선 정렬·초기 삽입 추론

`BRIDGE_ENTRY_INSERT_CM=1.5`는 모따기를 끈 원본 플랫폼의 진입 쪽(+Y 로컬) extent를 맞춘다. B의 Y 중심을 `half_a_y-half_b_y`로 두어 A/B의 +Y 끝선이 같아지고, 상자 중심은 gap 중앙 X와 `끝선+상자Y/2-0.015m`에서 시작한다. 로컬 -Y로 밀며 세계 좌표의 구조물 회전은 기존 샘플을 사용한다. pusher 배치·heading·목표·보상·gap 폭은 기존 규칙이다. 기존 near/aligned 표시는 crosser 샘플 선택용으로 유지하되 상자 시작은 항상 이 삽입 위치로 덮어쓴다. 기본0, 추론에서1–2cm만 허용하며 모따기와 동시에 사용할 수 없다. 실제 정적 actor 위치·extent·중앙 정렬·삽입 깊이 검증과 영상은 `output_etc/check/bridge_entry_insert_20261108/`에 저장한다.

## BEFORE 공통 선행 상자

세 후행 행동에 같은 범위를 적용한다. X/Y **0.40–0.60m**, Z **0.35–0.50m**, 간격0.05m. 별도 선행자 physical bank를 사용하므로 일반 solo AT와 PUSH 상자 분포는 유지한다. BEFORE 논리 토큰 alias도 새 선행 상자를 가리킨다.

## 실행과 확인

[실행 명령](../config.md), [resume·시나리오·센서 옵션](../USAGE.md#stage2-sensor-900). 기본 `tokenhsi_sm120`, GPU0,4명·1024환경·8토큰·900step. 현재 학습 범위 검증은 `output_etc/check/sensor_square_training_20261107/`, 과거 전체 동결 검증은 `output_etc/check/stage2_sensor_900/`에 저장한다. 본학습의 서비스·로그는 [USAGE](../USAGE.md#본학습-백그라운드-서비스)를 따른다.

CPU 전이·동결·샘플링 검사는 `tokenhsi/tests/test_sensor_stage2.py`, 실제1024환경 reset·부분 reset·구조물/센서 좌표·크기 검사는 `tokenhsi/tests/check_sensor_stage2.py`다. 짧은 업데이트·재개·평가·화면의 검증 범위는 [보고서](../../output_etc/check/stage2_sensor_900/report.md)에 기록한다.


## Bridge 시작 분포 실험

`stage2_sensor_900_bridge_init`은 같은 Distill9001.5k에서 fresh 학습한다. 모델·PPO·보상·동결 범위는 `stage2_sensor_900`과 같고, Bridge 초기화와 시나리오 샘플링 비율을 변경한다.

- 시나리오 reset 확률은 cliff5% / bridge55% / room40%, Bridge PUSH 인원은1/2/3명 각1/3이다. Bridge 내부20/40/40 분기는 전체 reset의11/22/22%에 해당한다.
- `sensorScene.bridge.start_distribution.probabilities`는 present/near/insert 순서로 `[0.2, 0.4, 0.4]`다. 배치 재시도 중에도 분기를 유지한다.
- 세 분기 모두 원본 직각 플랫폼의 로컬+Y 끝선을 맞추고 로컬-Y로 민다. 세계 좌표 회전, 좌우10cm 간격, 상판 높이 차이, 상자 크기·물성은 유지한다.
- Present는 동적 다리를 landing 위치·각도에 놓는다. Crosser는 gap 경계 안쪽0.8–2m, 반대편 첫 상호작용 목표로 향하는 직선이 다리를 통과하도록 배치한다. AT/ON_TOP의 출발 무대 상자 선택50%는 유지하되 이 분기의 해당 상자는 사람 뒤쪽 경로에서1–2m에 둔다. 물체를 들고 있는 자세는 강제하지 않는다.
- Near는 기존 다리-목표 거리0.1–1.5m를 유지하고 그중50%에 기존 crosser 정렬을 적용한다. Insert는 선두를1.5cm 삽입하고 crosser는 기존 일반 배치를 사용한다.
- PUSH 담당자는 현재 다리 위치에 맞춰 담당 구역·stand-off·heading을 배치한다. 목표와 성공 gate는 유지한다.
- `bridge/start/{present,near,insert}/pushers{1,2,3}/` 아래에 sampled·episodes·arrived·entered·crossed·crosser_falls를 기록한다. Entered는 살아 있는 root가 다리 footprint 위 및 상판+0.4m 이상에 들어간 진단이다. `bridge/push_required/arrived`는 near/insert만 집계해 선배치의 초기 도착을 PUSH 실적으로 혼합하지 않는다. 기존 all/pushers 도착 지표는 present도 포함하므로 분기별 로그와 함께 본다.
- 기존 config에 새 키가 없으면 기존 경로를 사용한다. 새 PTH는 자체 saved config로 추론/resume하며, 학습용 YAML 설정과 `BRIDGE_ENTRY_CHAMFER`/`BRIDGE_ENTRY_INSERT_CM` 추론 override는 동시에 사용할 수 없다.
- 새 train/train_vnc는 검증된 GPU별 MPS가 없으면 실행하지 않는다. 일반 CUDA로 자동 대체하지 않는다.

## Bridge 횡단 curriculum

`stage2_sensor_900_bridge_curriculum`은 cliff/bridge/room5/55/40, 4명·1024환경·8토큰·900step과 같은 Distill1.5k fresh 전이를 사용한다. 모델·관측·보상·PPO·동결 범위는 유지한다.

- Bridge reset의70%는 동적 다리 선배치·PUSH1명/crosser3명,30%는 기존 PUSH1/2/3명 균등·crosser1명 및20/40/40 초기화다. 다른 두 사람도 모두 독립적인 SIT/CLIMB/AT/ON_TOP 과제에서 반대편 첫 물체로 향한다. 역할은 매 reset 섞으며 retry에서는 분포·길이·시작 phase를 유지한다.
- 횡단 연습의 운반 물체는 목적지 무대에 둔다. 세 명 중 두 명은 입구0.8–2m/2.5–4m, 한 명은 다리25–45% 또는65–85%에서 각50%로 시작한다. 기존 locomotion RSI의 전신 자세·속도를 방향에 맞춰 변환하고 발 지지를 보정한다. 상자를 든 자세를 합성하지 않는다. 사람 간 최소1.2m, 작업 물체는 출구1.5m 이상 바깥에 배치한다.
- 다리 길이는 실제 PUSH 인원과 별도로 고른다. 완료된 PPO 업데이트0–1000에서 짧은/중간/긴 다리80/15/5%, 다음2000업데이트 동안 각1/3로 선형 전환한다. 기존 협력 분기는 인원과 길이의 기존 대응을 유지한다. checkpoint epoch에 따라 다음 rollout의 schedule을 복원한다.
- `crossing/{walk,cooperative}/{all,span1,span2,span3,entry,middle,exit}`에 agent별 episodes·entered·crossed·falls·other_agent_fall·arrived·timeout·interrupted_before_crossing을 기록한다. 진입/횡단은 ever, 낙상/중단은 종료 시점이다. 중간 시작의 진입값을 입구 시작 성공으로 해석하지 않는다. `scene_episodes`와 `all_crossed`는 장면 분모다.
- `crossing/exposure`의 실제 agent-step 비중과 Room 실내/Cliff 바깥 대상 접근 역할 수를 기록한다. 기존 `bridge/*` 통계는 첫 crosser 기준이므로 새3인 분기의 성공 비교에는 `crossing/*`를 사용한다. 일반 장면 종료 규칙은 유지한다.
- 기본 test/VNC/eval은 기존 협력 분기만 사용한다. 평가 전용 `BRIDGE_CURRICULUM_EVAL=walk_entry`는 세 명 모두 입구부터 시작하며, `training`은70/30과 중간 초기화를 재현한다. 두 옵션의 길이는 각1/3로 고정한다. 성능 비교는 기본/입구 시작으로 하고 training 옵션은 초기화 진단용으로만 사용한다.

## Room BEFORE 균형

`stage2_sensor_900_room_before`는 Bridge curriculum에서 Room의 샘플링만 변경한다. 시나리오5/55/40·Bridge 횡단70%/협력30%·길이 schedule·시작 분포·모델·관측·보상·PPO·동결 범위를 유지한다.

- Room BEFORE/COUPLED/MIXED 생성 분기 각1/3은 유지한다. BEFORE 분기는 pairing1.0으로 두 쌍을 만든다.
- Room의 독립 PUSH만 SIT/CLIMB/AT/ON_TOP으로 기존 비율을 정규화해 재배정한다.2/3/4인 COUPLED PUSH는 외부 바닥에 보존하며 다른 과제의 사람·상자·목표는 내부에 둔다.
- Room 인원 기대 비율은 독립12.1528%·BEFORE50%·COUPLED37.8472%, 내부62.1528%·외부37.8472%다.
- 전체 agent-reset 기대 비율은 절벽 외부 목표/Bridge 횡단33.9358%·그 외 독립19.0122%·BEFORE23.1458%·COUPLED23.9063%다. Room 내부 노출24.8611%는 이 과제들과 중복되며 실제 우회 강제 비율·성공률이 아니다.
- Train/train_vnc wrapper는 새 환경 YAML을 명시한다. 원래 curriculum checkpoint에서 `sensorScene.room.sampling` 두 설정만 추가되는 resume을 허용하며, 다른 계약 차이는 거부한다. 새 checkpoint에는 새 계약을 저장하고 이후 resume/추론은 해당 설정을 사용한다. 모델·optimizer·epoch/frame·입력/AMP/value 통계를 복원한다.
- 실행 진입점은 `stage2_sensor_900_room_before_{train,train_vnc,test,vnc,eval}.sh`다. GPU 확인·MPS 연결 후 실행하며 예시는 [config](../config.md), 운영은 [USAGE](../USAGE.md)에 둔다.
