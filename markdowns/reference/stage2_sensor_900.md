# Stage2 sensor 900

별도 실행 컨피그 `stage2_sensor_900`. 기준 구현은 `/home/user/jhh/Projects/stage_1_2_base_fuv1`의 square sensor·cliff/bridge/room·배치/초기화 규약이며 통합 Stage2의 BEFORE/COUPLED/mixed 그래프·보상·개인 PUSH 목표를 유지한다. 기존 `stage2_distill_900` 증류 코드는 그대로 둔다.

## 전이와 학습

- 원본: `output/stage2/Stage2Distill900_00001500.pth`, epoch1500, SHA256 `fcadf2297bedf8e3a8d010fd93d045f63dab412f0c85e65d7b89de4cc15a8a1e`.
- 원본279개 state tensor를 복사한 뒤 actor encoder·actor Stage2 coordination·sigma·입력 RMS만 동결한다. Action head, critic encoder/coordination/value head 전체, 센서 분기, AMP discriminator를 학습한다. `TokenHSI_stage12_teacher_sensor`에서 검증한 square sensor 학습 범위와 사용자 지정 actor coordination 동결을 동일하게 적용한다.
- actor/critic 각각 센서 MLP `2304→256→128→64`와 `64→1024` 무편향 투영을 학습한다. 투영을0으로 초기화하여 기존 head 첫 선형층 출력에 더한다. 초기 action/value 네트워크 출력은 원본과 수치 허용오차 내 동일하다. 학습 후 SENSOR_MASK=1은 학습된 head를 사용하므로 원본 정책 복원이 아니다.
- 입력/AMP RMS는 전이하되 입력 RMS만 동결한다. Value RMS는 fresh 초기화하고 AMP/value RMS는 학습 중 갱신한다.
- PPO lr `2e-5`, horizon32, minibatch16384, mini_epochs6, task/AMP reward 각0.5. AMP 판별기도 학습한다. 이 실험에는 distillation loss/teacher 로딩이 없다.
- checkpoint에 원본 경로·hash·epoch와 sensorScene·배치 계약, `sensor_training_contract=square_sensor_frozen_actor_coordination_v1`을 저장한다. 새 계약의 resume은 모델·optimizer·통계를 복원한다. 이전 전체 동결 checkpoint는 테스트만 가능하며 새 학습의 resume으로 사용하지 않는다.

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

## BEFORE 공통 선행 상자

세 후행 행동에 같은 범위를 적용한다. X/Y **0.40–0.60m**, Z **0.35–0.50m**, 간격0.05m. 별도 선행자 physical bank를 사용하므로 일반 solo AT와 PUSH 상자 분포는 유지한다. BEFORE 논리 토큰 alias도 새 선행 상자를 가리킨다.

## 실행과 확인

[실행 명령](../config.md), [resume·시나리오·센서 옵션](../USAGE.md#stage2-sensor-900). 기본 `tokenhsi_jhh`, GPU0,4명·1024환경·8토큰·900step. 확인 산출물은 `output_etc/check/stage2_sensor_900/`다. 2026-11-06 시작한 본학습의 서비스·로그는 [USAGE](../USAGE.md#본학습-백그라운드-서비스)를 따른다.

CPU 전이·동결·샘플링 검사는 `tokenhsi/tests/test_sensor_stage2.py`, 실제1024환경 reset·부분 reset·구조물/센서 좌표·크기 검사는 `tokenhsi/tests/check_sensor_stage2.py`다. 짧은 업데이트·재개·평가·화면의 검증 범위는 [보고서](../../output_etc/check/stage2_sensor_900/report.md)에 기록한다.

## 창고 확장 (`stage2_sensor_900_warehouse`)

[환경 YAML](../../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_warehouse.yaml)과 [학습 YAML](../../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_warehouse.yaml)을 별도로 사용한다. cliff·bridge·room·warehouse는 각각25%이며 기존 세 장면 config는 보존한다. Distill9001.5k 전이·위의 부분 동결/PPO·4명/1024환경/8토큰/900step을 사용한다. 새 run/checkpoint 이름은 `Stage2Sensor900WarehouseSquare`다. 관측10488차원과 네트워크 폭은 같다. 저장된 sensorScene 계약이 다르므로 세 장면 checkpoint를 네 장면 resume/eval에 사용하지 않는다.

- 기준은 `/home/user/jhh/Projects/stage_1_2_base_fuv1_roomopen`의 최종 창고 통합 구현이다. 폭·깊이4–6m(0.5m 간격), 바닥0.1m, 벽 높이2.5m·두께0.4m, 전면 중앙 입구를 사용한다.
- 무작위1명 PUSH와 나머지3명의 독립 또는 BEFORE 한 쌍+독립이다. `sample_roles`의 pairing_probability0.5를 유지하여 BEFORE 포함 비율은 약75%다. PUSH prerequisite·새 relation은 추가하지 않는다. 다른 장면의 통합 Stage2/bridge 공동 PUSH는 유지한다.
- 입구 차단 상자는 기존1인용 bridge bank의 native 크기·물성이다. 양쪽4cm overlap·외벽 간격4cm, 담당자는 측면1.6m에서 출발해 벽과 평행하게 좌우2–5m 민다. 도착 시 입구 여유는 최소0.35m다.
- 전원 외부 loco 시작, 작업 source는 내부, AT 목표·독립 ON_TOP 받침은 외부다. BEFORE 후행은 선행 AT 상자를 공유한다. source 깊이0.85–2.8m·옆 벽 여유0.2m, 접근/운반1–5m·후행 SIT/CLIMB 목표 거리1–10m·near50 경로 합 조건을 유지한다.
- GPU 상자 후보32개·사람별 후보32개, 내부16회·공통64회 재시도에서 역할/과제/near 모드를 유지한다. 벽·물체·사람·PUSH 이동 구간·목표 충돌을 검사하고 비과제 활성 상자는 구조물 밖3m 간격으로 주차한다. 부분 reset은 지정 환경만 바꾼다.
- 센서는 창고 바닥·벽·상자를 포함한다. 발밑 지형에는 벽을 제외하며 외부 시작은 바닥 낙상 조건을 사용한다. 벽 CPA 할인0.95, 기존 과제·AMP·BEFORE 보상/gate는 유지한다.
- `scene/warehouse/*`, `warehouse/{episodes,opened_fraction,entered_people_fraction,pusher_arrived_fraction}`은 온라인 진단이다. 실제 상자 회전을 반영한 열림·작업자 root 진입·PUSH 도착의 ever 값을 종료 에피소드별로 집계한다. 진입 분모는 종료 창고 에피소드×3명이다. 정식 평가에서는 기존 과제/interaction group 정의와 cliff unreachable 제외를 유지한다.

검증 범위와 산출물은 [검증 보고서](../../output_etc/check/stage2_sensor_900_warehouse/report.md)에 기록한다.

부분 동결 수정의 검증은 `output_etc/check/warehouse_square_training_20261107/`에 저장한다. 현재 본학습의 상태·로그·종료는 [USAGE](../USAGE.md#창고-포함-본학습-백그라운드-서비스)를 따른다.


## Warehouse Bridge 초기화 실험

`stage2_sensor_900_warehouse_bridge_init`은 Distill9001.5k fresh 전이·기존 모델/PPO/동결/보상을 유지한다. 시나리오는 cliff5% / bridge40% / room25% / warehouse30%다. Warehouse의 입구 PUSH1명·다른3명 구성, 출발 위치·BEFORE·physical bank·막힘/열림 조건은 변경하지 않는다.

- Bridge 내부 `start_distribution.probabilities`는 present/near/insert 순서 `[0.2, 0.4, 0.4]`; 전체 reset에서는8/16/16%다. PUSH 인원1/2/3명은 균등하고 분기는 배치 재시도 중 유지한다.
- Bridge 세 분기 모두 로컬+Y 진입 끝선을 맞추고 로컬-Y로 민다. 원본 직각 지형·좌우10cm 간격·상판1cm 차이·동적 상자와 물성은 유지한다.
- Present는 다리를 landing 위치/각도에 놓고 속도를0으로 초기화한다. Crosser는 gap 경계 안쪽0.8–2m에서 반대편 첫 상호작용 목표를 향해 다리 폭 안으로 정렬한다. AT/ON_TOP의 출발 무대 상자 선택50%를 유지하며, 이 분기의 해당 상자는 crosser 뒤쪽 경로1–2m에 둔다. 들고 있는 자세는 강제하지 않는다.
- Near는 상자-landing 거리0.1–1.5m와 그중50% crosser 정렬을 유지한다. Insert는 상자 선두를1.5cm 삽입하며 crosser는 기존 일반 배치를 사용한다. PUSH 담당자는 실제 다리 시작 위치를 기준으로 배치한다.
- Bridge 설정은 Warehouse 장면에 적용하지 않는다. Warehouse는 계속1인용 bank를 사용한다. 기존 config에 새 키가 없으면 기존 함수/RNG 경로를 유지한다.
- `bridge/start/{present,near,insert}/pushers{1,2,3}/`에 sampled·episodes·arrived·entered·crossed·crosser_falls를 기록한다. Entered는 살아 있는 root의 다리 footprint 진입 및 상판+0.4m 높이 진단이다. `bridge/push_required/arrived`는 near/insert만 집계한다. 기존 all/pushers 도착 통계에는 present도 포함되므로 함께 해석한다. Warehouse 열림/진입 지표는 기존대로 유지한다.
- 새 train/train_vnc는 검증된 GPU별 MPS 없이는 실행하지 않는다. saved config로 추론/resume하며 다른 실험의 초기화 설정으로 바꾸어 재개하지 않는다.

## Warehouse 진행 상태 RSI

`stage2_sensor_900_warehouse_bridge_init_rsi`는 현재 Bridge 초기화 실험에서 Warehouse의 시작 상태만 확장한다. 장면 비율5/40/25/30%, Bridge 완료/근접/삽입20/40/40%, PPO·900step·Distill9001.5k fresh 전이·freeze 범위는 유지한다. Actor encoder/coordination/sigma·입력 RMS는 동결하고 action head·critic 전체·센서·AMP discriminator·AMP/value RMS는 학습한다.

- Warehouse reset마다 closed/partial/open을50/25/25%로 고른다. 역할·과제·BEFORE near와 시작 모드는 배치 재시도 중 유지한다.
- Closed는 기존 중앙 입구 차단 상태다. Partial은 완전 개방에 필요한 최소 측면 이동량의25–75%만큼 상자를 옮긴 상태다. Open은 상자를 원래 PUSH 목표에 놓는다. 상자와 PUSH 담당자를 같은 방향·거리만큼 옮기고1.6m 간격을 유지한다. 속도는 기존 reset과 동일하게 상자는0·사람은 loco 참조다.
- 작업자3명은 모두 외부에서 시작한다. source 내부·AT 목표/독립 ON_TOP 받침 외부·BEFORE alias·거리/near 조건과 벽/물체/사람 충돌 검사는 유지한다. 구조물·센서 계산은 변경하지 않는다.
- `warehouse/start/{closed,partial,open}/`에 sampled·episodes·opened_fraction·entered_people_fraction·pusher_arrived_fraction을 기록한다. 기존 전체 opened 통계에는 처음부터 열린 샘플이 포함되므로 closed 분기와 구분한다. 진입 분모는 각 모드 종료 에피소드×작업자3명이다.
- 새 실험의 test/VNC/정식 평가는 기본 closed다. `WAREHOUSE_START=partial|open|mixed`는 추론 전용 상태 검사이며, 학습에서 override를 주면 오류다. 평가 시 초기 개방으로 인한 성공률 상승과 실제 개선을 혼동하지 않도록 기본 closed 평가를 유지한다.
- 기존 config에 새 `warehouse.start_distribution`이 없으면 기존 배치/RNG 경로를 보존한다. sensorScene 계약이 달라 기존 모델을 새 실험 resume으로 사용하지 않는다.

[실행 명령](../config.md), [옵션/운영](../USAGE.md#warehouse-rsi), [검증](../../output_etc/check/warehouse_rsi_20261108/report.md).

## Warehouse safety 샘플링

`stage2_sensor_900_warehouse_safety`는 기존 Warehouse RSI의 모델/PPO/보상/센서/상자 bank/기하/900step/동결 범위를 유지하며 reset 분포만 조정한다. 장면 reset 확률은 Cliff5% / Bridge43% / Room17% / Warehouse35%다.

| 장면 | 조건부 역할/초기화 |
| --- | --- |
| Cliff | 40% 전원4명 절벽 안전 역할, 60% 절벽2명+바닥 coupled PUSH2명. 독립 안전 과제는 SIT/CLIMB/AT/ONTOP, 높은 무대의 목표는 기존처럼 무대 밖. BEFORE 없음. |
| Bridge | 90% 다리 연결 완료·pusher1+crosser3; entry2명+middle/late1명 RSI. 10% 기존 협력(pusher1/2/3 균등), 해당 분기의 완료/근접/1.5cm 삽입20/40/40·완료 crosser0.8–2m 유지. |
| Room | BEFORE/COUPLED/MIXED 생성기30/40/30. BEFORE 생성기는 두 쌍 완전 pairing. 독립 PUSH는 다른 네 과제로 재추출하고 coupled PUSH는 바깥 바닥, 나머지는 안쪽. pickup 벽 가로지르기 요청75%(후반 placement retry에서 완화). |
| Warehouse | 입구 PUSH는 반드시1명, 크기/기하 동일. 작업자3명의 pairing 인자0.8 → BEFORE 한 쌍 존재 확률96%. 닫힘/부분/열림50/25/25 유지. |

다리 길이1/2/3의 확률은80/15/5에서 시작한다. 신규 curriculum 업데이트1000회 유지 후2000회에 걸쳐1/3씩으로 전환한다. 기존6000epoch에서 resume할 때 모델/optimizer/AMP·value RMS/global epoch/frame는 그대로 복원하고 새 curriculum anchor만6000으로 설정한다. 최초6001epoch의 schedule update는0이며 이후 checkpoint에 anchor를 저장하여 재개 시 이어간다.

전체 **agent-reset**의 상호 배타적 기대 비율은 절벽/횡단33.6000%, 기타 독립31.4292%, BEFORE24.8083%, COUPLED10.1625%다. Bridge 횡단30.1%+Cliff 안전3.5%이며, 일반 독립 PUSH인 bridge pusher는 COUPLED에 포함하지 않는다. 벽 관련 구조적 노출45.1292%는 다른 과제 비율과 중복된다. Warehouse 입구 통과가 필요한 역할20.65%, Room의 벽 경로 요청3.5328%는 전체 agent-reset 분모다. 배치 retry 수락률·에피소드 길이·실패 시점에 따라 실제 agent-step 노출은 다르며 요청75%를 실제 벽 충돌/경로 통과75%로 해석하지 않는다.

평가는 기본 기존 협력 분기다. `BRIDGE_CURRICULUM_EVAL=cooperative|walk_entry|training`은 평가 전용이다. `WAREHOUSE_START=closed|partial|open|mixed`는 기존 평가 옵션을 유지한다. 새 YAML을 train wrapper에서 명시하므로 resume 시 과거 relation_config의 장면 비율로 되돌아가지 않는다. 기존 RSI→safety 학습 전환만 계약 예외로 허용하며 reward/모델/물리 변경이나 기존 PTH의 새 환경 추론은 허용하지 않는다.

센서 agent의 actor encoder/Stage2 actor coordination/sigma·입력 RMS는 동결한다. Critic 전체·action head·센서·AMP discriminator 및 value/AMP 정규화는 학습한다. reset 실패64회마다 같은 역할/그룹의 RSI를 새로 추출하며 최대256회 시도한다. [검증·한계](../../output_etc/check/warehouse_safety_20261109/report.md), [운영](../USAGE.md#warehouse-safety).

### 현재 초기 가중치

2026-11-09 20:41 사용자 요청으로 safety 본학습을 Distill9001500 fresh 전이로 교체했다. 위6000epoch resume/anchor6000 설명은 보존된 이전 실험에 해당한다. 현재 모델·입력/AMP 통계는 Distill1500에서 전이하고 센서·optimizer·value RMS·epoch/frame/curriculum은 처음부터 시작한다. 장면/역할 비율과 동결 범위는 동일하다. [현재 서비스](../USAGE.md#warehouse-safety-fresh-본학습-서비스).

## Warehouse 공동 PUSH

`stage2_sensor_900_warehouse_coop`은 Safety의 네 장면 비율5/43/17/35%, Bridge/Cliff/Room curriculum·역할, 모델/PPO/보상/동결/900step을 유지하고 Warehouse만 확장한다.

- PUSH 인원1/2/3명 균등, 방 기준 좌/우/안/밖 네 방향 균등. 총4명이며 나머지3/2/1명은 비PUSH 작업자다. BEFORE worker pairing0.8은 가능한 작업자들 사이에만 적용한다.
- 승인한 기본 상자1.8×2.0×1.5m를 사람 수만큼 로컬X축으로 늘린다. 물리 bank3개를 미리 생성하고 방향에 맞게 회전한다. 같은 PUSH 그룹은 실제 상자 하나를 공유하며 기존 COUPLED 구역·개인 목표·접촉 gate를 사용한다. 센서는 공유 상자의 전체 물리 크기를 읽는다. 밀도/마찰은 기존 native PUSH 설정이다.
- 좌우형은 입구1.92m, 상자가 방 밖으로 길게 놓이며2.6m 옆으로 민다. 앞뒤형은 입구가 상자 폭보다0.12m 넓고3m 안/밖으로 민다. 바깥쪽 PUSH 담당자는 닫힌 초기 배치에서 방 안에 있다.
- 방은 깊이6m×폭8m로 고정해3명용5.4m 폭 상자 양옆에 각각1.3m 통로를 둔다.9m 전면 apron을 방 바닥과 같은10cm 높이로 만들어 앞뒤 PUSH의 문턱을 없앤다. 물체·목표·담당자 높이와 센서의 바닥 기하를 함께 맞춘다.
- 닫힘/부분/열림50/25/25를 유지한다. 부분은 최소 통로 개방 이동량의25~75%, 열림은 상자 목표에서 시작한다. 상자와 PUSH 담당자를 함께 이동한다. 작업자는 외부에서 시작하며 기존 개별 접근·운반 거리와 BEFORE near 조건을 유지한다.
- 입구/상자 count·방향·RSI 선택은 배치 재시도 중 유지한다. 정적 상자 sweep과 작업 물체/목표 간섭을 검사하며 네 방향×세 인원×세 RSI의 지표를 따로 기록한다. `warehouse/worker_episodes` 및 각 세부 `worker_episodes`가 실제 작업자 수에 따른 진입률 분모다.
- 기존 checkpoint/환경 계약은 보존한다. 새 실험은 기본 Distill1500 fresh 전이이며 이전 Safety500 또는 RSI6000을 새 분포에 그대로 resume하는 계약 예외는 추가하지 않았다.

## Warehouse 공동 PUSH Cliff·Room 조정 (`stage2_sensor_900_warehouse_coop_cliffroom`)

`stage2_sensor_900_warehouse_coop`에서 세 값만 바꾼 별도 실험이다. 모델·PPO·보상·동결·Bridge/Warehouse 역할·900step은 같고 Distill1500 fresh 전이다.

- 장면 reset 비율 Cliff/Bridge/Room/Warehouse를 5/43/17/35%에서 **20/30/17/33%**로 바꾼다. coop 3000epoch에서 Cliff는 에피소드가 약100step이라 실제 env-step의0.7%였다.
- Cliff `all_edge_probability`를 0.4에서 **1.0**으로 바꾼다. 4명 모두 무대 위에서 시작하고 각자의 물체는 가장자리 너머 바닥, 목표는 그보다 바깥에 놓인다. Cliff의 바닥 coupled PUSH는 없어진다.
- Room `size_range`를 10–14m에서 **7–14m**로 넓힌다. 벽은 env 생성 시 고정되므로 env마다 크기가 다르며 학습 중에는 바뀌지 않는다. 칸막이 길이는 3.4–10.4m다. 6×6m는 4명 배치가 첫 reset부터 소진되어 제외했고 7·8·9m 정사각형 방은 1024환경×8회 reset을 통과했다.

### Room 기하 참조와 혼잡 배치 복구

2026-11-11 Room 배치 재시도 소진에 대응해 창고 기하 사전 생성 후 `_room_half`와 `_wall_local`을 원래 Room 기하로 복원한다. 물리 actor·센서 기하와 Room 크기/벽 판정의 참조를 일치시킨다. 물리 기하·창고 변형과 그 사전 생성의 난수 소비는 유지한다.

기본 후보16개와 전체256회 재시도 상한은 유지한다.64회 실패한 후 남은 배치가 Room뿐이면 그룹 배치 순서를 섞고 그룹별 후보를64개로 늘린다. BEFORE 쌍 우선 배치·역할/관계·상자/방 크기·운반 거리·충돌 판정은 유지한다. 보상식/계수·모델·동결·학습 YAML은 바꾸지 않는다. [변경 diff](../../output_etc/check/room_sampling_fix_20261111/fix.patch).

## Room·Warehouse 전용 (`stage2_sensor_900_room_warehouse`)

`stage2_sensor_900_warehouse_coop_cliffroom`에서 Room/Warehouse만 50:50으로 샘플링하고, Distill900V4 2000(`output/stage2/Stage2Distill900V4_00002000.pth`, 2026-11-13 변경)에서 새로 전이한다. 고정 scene ID와 checkpoint 형식을 유지하기 위해 Cliff/Bridge 설정·자산은 보존하며 샘플 확률을0으로 둔다.

- `sensorScene.wall_collision_coefficient: 0.7`: Room·Warehouse 벽 CPA에만 적용한다. 사람 간 CPA 계수0.5·거리0.7m·사람 시간 할인0.99·벽 시간 할인0.95를 유지한다.
- 충돌 감점은 `-max(0.5 * human_risk, 0.7 * wall_risk)`다. 기존 max 합산 방식을 유지하며, 새 키가 없는 기존 실험은 기존 계산과 같다. task/AMP 혼합 가중치0.5/0.5도 유지한다.
- Room7–14m·BEFORE/COUPLED/MIXED30/40/30·반대편 집기 요청75%, Warehouse1/2/3명 PUSH·네 방향 균등·RSI50/25/25를 유지한다.
- 4명·1024환경·8토큰·900step·seed42, 네트워크/PPO·BEFORE 대기 보상·actor encoder/coordination·sigma·입력 RMS 동결을 유지한다. 이전4시나리오 checkpoint는 보존하며 resume 계약 예외를 추가하지 않는다.

실행 명령은 [config](../config.md), 백그라운드 운영은 [USAGE](../USAGE.md#room-warehouse-전용-본학습)를 따른다.

## 벽 접촉 종료 (`stage2_sensor_900_room_warehouse_wallterm`)

`stage2_sensor_900_room_warehouse`에 `sensorScene.wall_contact_termination: 0.3`만 더한 비교 실험이다. Run 이름은 `Stage2Sensor900RoomWarehouseWallTerm`이다. Room/Warehouse에서 손·발을 뺀 11개 몸체 중 하나가 접촉력(`|F|합>0.1`)을 받으면서, 그 MJCF 구·캡슐 형상이 벽 OBB(`_static_boxes` slot1–5)에서 3cm 이내이면 접촉으로 본다. 이 상태가 0.3초(30Hz 기준 9스텝) 연속되면 그 사람을 종료하고, 씬 reset·정식 평가 무reset 규칙은 넘어짐과 같다. 접촉력은 상대를 구분하지 않으므로 몸체 원점 거리 대신 형상 거리를 쓴다. 감지·로깅(`scene/<장면>/wall_contact_fraction`·`wall_terminations_per_episode`)은 키가 없는 기존 sensor 실험에도 동작하며 종료는 하지 않는다. 구현: `square_sensor.load_body_shapes`, `square_sensor_scene.body_box_contact`, `SensorStage2Observation._wall_contact_step`.

## Warehouse PUSH 상자 CPA·접촉 종료 (`stage2_sensor_900_room_warehouse_wallterm_boxterm`)

wallterm에 `push_box_collision_coefficient: 0.7`, `push_box_contact_termination: 0.3`만 더했다. Run 이름은 `Stage2Sensor900RoomWarehouseWallTermBoxTerm`이다. Warehouse에서 PUSH 담당자가 아닌 사람에게만 PUSH 상자(`_bridge_slot`)를 장애물로 적용한다. CPA는 상자 OBB에서 각 root에 가장 가까운 점을 상자 속도로 움직이는 agent로 두고, 벽과 같은 0.7m·할인0.95를 쓴다. 사람/벽 위험과는 max로 합친다. 종료는 벽과 같은 형상 3cm·접촉력 판정이 0.3초 연속일 때이며 카운터는 벽과 별도다. PUSH 상자 접촉 로깅(`warehouse/push_box_contact_fraction`·`push_box_terminations_per_episode`)은 키가 없는 실험에도 동작한다. 구현: `square_sensor_scene.compute_box_cpa_collision_penalty`, `warehouse_sensor.collision`·`push_box_contact_step`.

## V4 보상 (`stage2_sensor_900_room_warehouse_wallterm_boxterm_v4_reward`)

boxterm의 과제 보상을 dev checkout(`TokenHSI_stage12_teacher_sensor_dev`)의 `stage2_sensor_900_cliff_bridge_four_transfer_v4_reward`와 같게 맞췄다. 바뀐 키는 `stage2.reward_source: v4_teachers`, `before.reward_at_z_tolerance: 0.001`, `before.waiting_penalty`(`radius_buffer` 2.5→1.5, `ontop_distance_source` human→payload), `stage2JointPush.premature_motion_penalty`(0.5/0.05m/s/1e-6)다. 장면, 벽/PUSH 상자 CPA·접촉 종료, 전이 checkpoint, PPO 설정은 boxterm과 같다. Run 이름은 `Stage2Sensor900RoomWarehouseWallTermBoxTermV4Reward`다.

- AT 성공: BEFORE와 비공동 작업자(`_members==1`)는 Z20cm로 판정한다. 보상 포화·성공 보너스·paired HOLDING 포화는 Z1mm를 요구한다. 공동 PUSH AT는 성공과 보상 모두 Z1mm다.
- 대기 감점의 반경은 지지 상자 반대각+1.5m다. ON_TOP은 운반 상자 XY, SIT/CLIMB는 사람 XY로 진입을 판정한다.
- 공동 PUSH 조기 이동 감점: 그룹 전원의 두 손이 지정 면에 닿기 전에 상자 XY 속도가 0.05m/s를 넘으면 `-0.5*(speed-0.05)*손 반력 기여율`을 준다. 물리 step 평균으로 계산하고, 별도 reward term `premature_motion_penalty`로 기록한다. 1인 PUSH에는 적용하지 않는다.
- 진단 `relation/v4/`(AT 성공·정밀 포화, PUSH CONTACT, 대기 활성, 평균 조기 이동 감점)를 남긴다. AT 성공 기준이 다르므로 학습 로그 성공률을 boxterm과 직접 비교하지 않는다.
- 저장 계약(`stage2`/`before`/`stage2JointPush`)이 boxterm과 달라 boxterm checkpoint는 resume할 수 없다. Distill V4에서 새로 전이한다. dev의 외력 컨트롤러 CONTACT 연결은 Bridge 전용이어서 이식하지 않았다(roomopen에는 해당 경로가 없음).

구현: `stage2_reward.relax_solo_at`, `before_reward.reward_at_z_tolerance`, `stage2_wait.ontop_distance_source`, `humanoid_ma_coupled_push._update_push_feedback`, `coupled_push_contact.group_face_geometry`(`near`), `utils/premature_push.motion_penalty`, `edge_context_task`(감점 합산·진단), `stage2_spec.validate_env`(V4 native 비교).

## Warehouse 역할 고정 V4 (`stage2_sensor_900_warehouse_fixedroles_v4`)

V4 보상 config(`stage2_sensor_900_room_warehouse_wallterm_boxterm_v4_reward`)에서 Warehouse 키 6개만 바꿨다. 기존 옵션으로 표현되므로 새 코드 분기는 없다. Run 이름은 `Stage2Sensor900WarehouseFixedRolesV4`이고, Distill900V4 2000에서 새로 전이한다.

| 키 | 값 (V4 config 기준) |
| --- | --- |
| `sensorScene.mixture` | room 0.5→0, warehouse 0.5→1.0 (scene ID와 자산은 유지) |
| `warehouse.cooperative.pusher_probabilities` | [0, 1, 0]: 공동 PUSH 2명 |
| `warehouse.cooperative.direction_probabilities` | [0, 0, 1, 0]: 방 안쪽 |
| `warehouse.cooperative.frontal_distance` | 3.0→5.0m |
| `warehouse.cooperative.room_size` | [6, 8]→[8, 8]: 상자 끝면 5.85m, 뒷벽까지 여유 2.15m |
| `warehouse.worker_pairing_probability` | 0.8→1.0: 작업자 2명이 항상 BEFORE 쌍 |

작업자 2명은 `sample_roles`에 의해 AT 선행 1명과 그 선행을 따르는 후행 1명이 된다. 후행 과제는 크기가 맞는 SIT/CLIMB/ON_TOP 중 균등하게 정한다. 시작 상태는 닫힘/부분/열림 50/25/25를 유지한다. 부분 열림은 입구가 열리는 최소 이동량(1.5m)의 25–75%만 밀린 상태다. 900step 제한, 벽/PUSH 상자 CPA·접촉 종료, V4 보상·동결·PPO는 기반 config와 같다.

검증: [CHECK](../../output_etc/check/warehouse_fixedroles_v4_20261116/).

## Warehouse 역할 고정 V4 + BEFORE 2팀 (`stage2_sensor_900_warehouse_fixedroles_v4_beforepairs`)

fixedroles_v4에 `warehouse.cooperative.before_pairs_probability: 0.2` 하나만 추가했다. Run 이름은 `Stage2Sensor900WarehouseFixedRolesV4BeforePairs`이고, Distill900V4 2000에서 새로 전이한다. fixedroles_v4 2000epoch에서 후행이 대기 반경에서 멈춘 채 성공 경험을 얻지 못한 문제(정식 평가 SIT/CLIMB/ON_TOP 약 0%)를 겨냥한다.

- Warehouse reset의 20%는 **BEFORE 2팀**이다. AT 선행 2명이 각자 후행 1명(크기가 맞는 SIT/CLIMB/ON_TOP 중 균등)을 갖고, PUSH는 없다. 나머지 80%는 기존 PUSH 2 + BEFORE 1쌍이다.
- BEFORE 2팀은 일반 Stage2 BEFORE 기본 배치(거리·near 규칙 포함)를 사용한다. 그룹마다 Room과 같은 `_place_group` 강체 회전·이동으로 **방 앞 apron**에 배치한다. 영역은 앞벽 바깥 면부터 apron 끝까지, 폭은 방 폭 + 1m이고, 기존 사람 0.8m·물체 0.5m+반경 여유를 둔다.
- Default 초기화에서는 apron 배치 후 후행의 사용하지 않는 Target을 해당 사람의 초기 root 위치로 둔다. 원래 env에서 시작한 Default 사람과 선택된 Warehouse origin의 차이가 강체 이동에 섞여 중립 Target이 수백~수천 m로 벗어나는 문제를 막는다. 활성 AT 목표·사람/상자 배치·보상은 유지하며, Random/Start/Hybrid 초기화의 기존 Target은 바꾸지 않는다. 새로 생성한 배치는 이 좌표로 캐시된다.
- Stage2의 공유 상자 초기화는 Default와 loco 모두 같은 소유자 규칙을 쓴다. 선행/독립 담당자가 자기 상자를 샘플링하며, ON_TOP 후행은 자기 payload만 샘플링한다. 후행의 공유 받침을 다시 샘플링하지 않고 전체 Default 초기화 목록은 AMP 이력 초기화를 위해 복원한다.
- apron의 강체 배치는 기존 16개 회전·이동 후보를 한 번에 검사하고 첫 유효 후보를 선택한다(`place_rigid_batched`). 그룹 내 거리와 여유를 고려해 어느 회전에서도 들어갈 수 없는 경우는 충돌 검사 전에 탈락시킨다. 유효성·여유·최대 후보 수는 유지하며 난수 소비 순서는 달라진다. Room/Bridge/Cliff는 기존 순차 검사 경로를 사용한다.
- 방은 쓰지 않는다. PUSH가 없으므로 문 상자는 비활성 물리 상자처럼 env 밖(+1000m)에 남아 문이 열려 있다. 남는 distractor 상자는 apron 밖 지면으로 치운다. 사람은 기존 Warehouse 작업자처럼 바닥 시작으로 판정된다(apron에서 지면으로 내려가도 종료하지 않음).
- 변형 env는 기존 문 개방·PUSH 상자 진단에서 제외한다. `warehouse/variant/{push_before,before_pairs}/{sampled,episodes,leader_success,follower_success}`에 변형별 선행·후행 에피소드 성공을 기록한다.
- 학습 중에는 검사를 통과한 apron 시작 상태를 환경별 메모리 캐시에 보관한다(`warehouse_before_cache.py`). 두 후행의 `(과제, near/far)` 조합을 정렬한 조건과 상자 크기·선택된 Warehouse 기하·원점이 같을 때 재사용한다. 새로 뽑은 과제/near 조합은 유지하고, 사람별 역할·물체 매핑·그래프·자세/관절/속도·목표·AMP 이력은 캐시 항목과 함께 복원한다. 그래프 edge 순서는 매번 다시 섞는다. 조건당 최대 2개, 환경당 32개 조건을 보관하며, 적중 가능한 reset의 25%는 기존 생성·검사를 수행해 갱신한다. 미적중은 기존 reset 경로를 따른다. 평가·viewer는 사용하지 않고 checkpoint에는 캐시를 저장하지 않는다. 비활성화와 진단은 [운영 옵션](../USAGE.md#warehouse-역할-고정-v4--before-2팀-본학습)을 따른다.
- 키가 없거나 0인 기존 config에서는 새 분기를 실행하지 않고 추가 난수도 소비하지 않는다. 보상·모델·PPO·동결은 fixedroles_v4와 같다.

Default 초기화 학습은 첫 rollout 전에 32회 apron 배치로 캐시를 채운다. 준비 시간은 별도 진단으로 기록하며 실제 80/20 변형 비율·샘플링 계수·난수 상태를 복원한다. 준비 횟수와 비활성화는 위 운영 옵션을 따른다. constant LR의 PPO에서는 사용하지 않는 KL 스칼라 동기화와 동일 LR 재설정을 생략하며, adaptive/linear 스케줄은 유지한다.

검증: [배치·CHECK](../../output_etc/check/warehouse_beforepairs_20261116/).

## Warehouse 벽 CPA 박스 옵션 (`sensorScene.warehouse.wall_cpa`)

기본값(`segment`, 키 없음)은 벽 안쪽 면 선분 5개로 CPA를 계산한다. 이 방식에서는 방 밖에서 볼 때 벽이 두께(0.4m)만큼 뒤에 있는 것처럼 계산되고, 바깥 모서리와 문틀 옆면은 선분이 없다. 그래서 방 밖 감점이 약하다. 1m/s로 정면 접근하면 방 안 감점의 약 절반이고, 45° 이상 비스듬히 접근하면 1/3 이하다. `box`로 두면 물리 벽과 같은 static 박스 5개(`_static_centers/_static_quat/_static_half`의 Warehouse 슬롯 1–5)에 `compute_box_cpa_collision_penalty`를 정지 상자로 적용하고, 벽들 중 최댓값을 쓴다. 방 안에서는 선분 결과와 같고, 바깥 면·모서리·문틀은 실제 형상대로 계산한다. 계수 0.7, 거리 0.7m, 할인 0.95와 접촉 종료는 그대로다. 기존 config에는 이 키가 없다. 구현: `warehouse_sensor.wall_box_cpa`·`collision`. 테스트: `tests/test_warehouse_wall_box_cpa.py`.

## Warehouse 역할 고정 V4 + 선행 완료 시작 + 벽 박스 CPA (`stage2_sensor_900_warehouse_fixedroles_v4_leaderdone_wallbox`)

fixedroles_v4에 `warehouse.cooperative.leader_done_probability: 0.3`과 `warehouse.wall_cpa: box` 두 키만 더했다. Run 이름은 `Stage2Sensor900WarehouseFixedRolesV4LeaderDoneWallBox`이고 Distill900V4 2000에서 새로 전이한다.

- **선행 완료 시작**(학습 전용): Warehouse reset의 30%는 선행 AT가 끝난 직후 상태에서 시작한다. 역할과 배치는 기존 fixedroles와 같다.
  - 문 상태: 열림으로 고정한다. 문 상자는 landing에, PUSH 담당자는 그 뒤에 선다.
  - 선행: 상자를 apron의 AT 목표(`_tar_pos`) 위에 놓고, 선행은 상자 면에서 0.45–1.0m 떨어진 apron 위에 상자를 보고 선다. 다른 사람과는 0.85m, 다른 상자와는 반경+0.5m 이상 떨어뜨린다.
  - 후행: 기존 배치(apron, 선행 목표에서 1–10m)를 그대로 쓴다.
  - 사람 자세는 기존 loco 시작 그대로라 RSI 캐시를 쓰지 않는다. 선행 경로가 이미 끝났으므로 near 쌍의 경로 순서 조건은 해제한다.
  - 선행 성공과 후행 대기 해제는 상자 상태로 계산되어 첫 step부터 참이다. 선행이 받는 보상은 기존 open 시작의 PUSH 담당자와 같은 성공 유지 보상이고, 한 번만 주는 보너스는 없다.
  - 평가와 viewer에서는 문 시작 분포처럼 꺼진다. 그래서 정식 평가는 기존 fixedroles와 같은 닫힌 문 시작이다.
- **진단**: 완료 시작 env는 기존 `warehouse/*`의 문 열림·진입·문 상태 통계에서 뺀다. `warehouse/{main,leader_done}/{sampled,episodes,leader_success,follower_success}`로 따로 기록한다.
- **벽 CPA**: [벽 박스 옵션](#warehouse-벽-cpa-박스-옵션-sensorscenewarehousewall_cpa)을 켠다.
- 키가 없거나 0인 기존 config에서는 새 분기를 실행하지 않고 추가 난수도 쓰지 않는다. 구현은 `env/tasks/multi_agent/warehouse_done.py`에 있다. 설정 검증은 PUSH 2명·BEFORE 1쌍·문 시작 분포가 있는 Warehouse만 허용하고, BEFORE 2팀과 함께 쓰는 것은 막는다.

검증: [배치·CHECK](../../output_etc/check/warehouse_leaderdone_20261116/).

## Warehouse BEFORE 선행 문 입구 progress (`stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress`)

beforepairs 기반에 `sensorScene.warehouse.leader_door_progress: true`를 추가한 보상 비교 실험이다. 사용자 요청으로 이 config의 reset RSI도 비활성화했다. Run 이름은 `Stage2Sensor900WarehouseFixedRolesV4BeforePairsDoorProgress`이며, 기존과 같은 Distill900V4 2000에서 새로 전이한다. 별도 train/train_vnc/test/vnc/eval wrapper와 output 이름을 사용한다.

- reset은 `stateInit: Default`로 사람의 기본 자세·관절/속도를 사용한다. 문 시작 확률은 `[1.0, 0.0, 0.0]`으로 일반 Warehouse 장면은 항상 닫힘이다. BEFORE 2팀 20%는 기존대로 문 상자를 사용하지 않으며 배치 캐시도 유지한다. AMP 학습용 참조 모션은 유지하지만 reset 자세를 그 모션에서 뽑지 않는다.

- 대상은 Warehouse의 **2명 공동 PUSH + BEFORE 1쌍** 구성에서 후행이 참조하는 AT 선행 한 명뿐이다. PUSH 담당자·후행·BEFORE 2팀 변형·다른 장면에는 0이다.
- 선행의 actor root(`_humanoid_root_states[..., :2]`)에서 문 입구 중앙까지의 XY 거리 `d`에 기존 progress와 같은 `1 / (1 + d / sigma)`를 적용한다. `sigma`와 가중치는 기존 `relationReward.progress.sigma`·`progress_reward_weight`를 재사용한다(현재 1.0·0.2). 문은 앞벽 중앙선의 중앙점이며, 방 좌표로 `(-room_half_x - wall_thickness/2, 0)`이다.
- root가 문 폭 안에서 그 중앙선을 방 안쪽으로 넘으면 raw progress를 해당 에피소드 동안 1.0으로 유지한다. 이후 문에서 멀어지거나 다시 바깥으로 이동해도 감소하지 않는다. reset된 환경의 통과 기록만 지운다. Z는 거리·통과 판정 모두에 사용하지 않는다.
- 기존 edge progress와 별도로 더하며 `reward_terms/door_progress`에 가중치 적용 후 값을 기록한다. 최대 추가 보상은 선행 1명에게 step당 0.2다. 기존 성공 조건과 종료 조건은 유지한다. BEFORE apron 캐시는 기존과 같이 동작한다.

실행 명령은 [config](../config.md)의 각 목록 마지막 항목을 따른다. 보상 계산은 `warehouse_door_progress.py`, 합산은 `edge_context_task.py`, 통과 기록의 초기화는 `humanoid_ma_sensor_stage2.py`에 있다.

## 후속 검토: 상자를 운반하며 Bridge 횡단하기 (미구현)

사용자가 요청한 후속 보상 아이디어다. 임의의 이동량은 다리 횡단 신호가 되지 않으므로, 실제 다리의 입구·출구와 중심선 방향으로 진행도를 정의한다. 상자를 든 입구 접근, 다리 영역 안에서의 운반 전진, 사람과 상자가 모두 출구를 통과한 일회성 보너스를 구분한다. 횡단 항은 운반 상태와 다리 영역/발 지지를 확인해 제자리·왕복·다리 옆 평행 이동·상자만 던지는 동작으로 획득하지 못하게 한다. 상자를 놓고 전진한 구간을 다시 집었을 때 소급 보상하지 않는다. 입구·출구 root 상대좌표를 정책 관측에 주는 것은 별도 구조 변경으로 검토한다. 현재 doorprogress config는 Warehouse 100%이므로 실제 Bridge 학습에는 장면이 포함된 별도 config가 필요하다. 이 아이디어는 기록만 했으며 구현하지 않았다.
