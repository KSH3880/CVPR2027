# Changelog

최신 변경부터 기록한다. 현재 실행법은 [config.md](markdowns/config.md), 코드 위치는 [structure.md](markdowns/structure.md)를 참조한다. 실행 중인 GPU/PID는 이 파일에 고정하지 않고 실제 프로세스로 확인한다.

## 2026-10-19

### 연결다리 마찰계수 실행 옵션

- 사용자 요청으로 `HumanoidMABridge`에 `BRIDGE_FRICTION` 실행 옵션을 추가했다(기본 `1.0`, 유한한 0 이상 값). 연결다리 actor의 rigid shape 마찰에만 적용하며 fan bridge·fan/square mixed의 학습·평가·viewer가 공통으로 사용한다. checkpoint metadata는 유지하며 재개·평가 시 값을 다시 지정한다. `config.md`에 사용법을 추가했다.
- 검증: 관련 CPU 테스트 26개 통과. GPU 0에서 기본값·0.3·0·2.0 각각 2환경·16-step PhysX 확인: 실제 다리 마찰값 일치, 다른 모든 actor 마찰 불변, 기존 square checkpoint의 sensor metadata 일치. 음수·NaN·무한대·잘못된 문자열 거부 확인. 기존 checkpoint로 `SCENARIO=bridge BRIDGE_FRICTION=0.3` 1환경·16-step headless 평가 정상 종료. 짧은 실행 검증이며 보행 성능 비교는 수행하지 않았다.

## 2026-10-08

### 혼합 평가의 장면 선택 옵션

- 사용자 요청으로 `humanoid_ma_mixed_sensor.py`에 평가 전용 환경변수 `SCENARIO=cliff|bridge|room`을 추가했다. 지정하면 reset 장면 확률만 해당 장면 1.0으로 바꾼다. sensor metadata의 `scenario_mixture`는 유지하므로 기존 fan/square mixed checkpoint를 그대로 평가할 수 있다. 학습·watch(`--eval` 아님)이나 잘못된 이름은 에러로 종료한다.
- 검증: 학습 중인 square checkpoint(`ApproachStage2SquareMixed_00001000.pth`)로 `SCENARIO=room` headless 16환경·64-step 평가를 돌려 trial **16/16이 room**임을 확인했다. 학습 스크립트에 지정하면 `ValueError`로 종료되는 것도 확인했다. 증거: `output/approach_stage2_square_mixed_check/scenario_room/`. 실행 중인 본학습은 재시작하지 않았다.

### 정사각형 센서 범위 10m → 7m

- 사용자 요청으로 `approach_stage2_square_mixed.yaml`의 `sensor_range`를 10.0에서 7.0으로 줄였다. 이제 root 중심 한 변 14m이며 48×48·2304점을 유지해 점 간격은 0.298m다. 관측 5215차원·metadata version 8·스크립트·output 경로는 그대로다. 이전 10m checkpoint는 cliff 설정이 달라 resume·평가가 거부된다. 10m로 만든 `output/approach_stage2_square_mixed_watch`에는 저장된 checkpoint가 없다.
- 정사각형 CPU 테스트가 10m를 하드코딩하지 않고 config의 범위를 읽도록 고쳤다.
- 검증: CPU 24개 통과. PhysX 96환경×6 reset에서 앞·뒤 변 ±7m, 중심 오차는 최대 0.15mm였고 관측과 재계산 값이 일치했다. 10m checkpoint 거부도 확인했다. 증거: `output/approach_stage2_square_mixed_check/env_7m/`. 관측 차원과 학습 경로가 같아 학습·평가는 다시 실행하지 않았다.

### 3종 혼합 정사각형 센서 실험 추가

- 사용자 요청으로 부채꼴 대신 사방을 덮는 정사각형 센서를 별도 실험 `approach_stage2_square_mixed.yaml`로 추가했다. 정사각형은 root XY를 중심으로 root heading을 따라 회전하며, 한 변 20m(중심에서 앞·뒤·좌·우 10m), 48×48 = 에이전트당 2304점이다. 가림 없는 수직 조회와 장면·과제·보상은 fan mixed와 같다. 관측은 5215차원, sensor metadata는 8이다. 기존 fan 실험과 스크립트는 바꾸지 않았다.
- 코드: `humanoid_ma_cliff.py`에 `sensor_pattern: square` 분기를 추가했다(격자 생성, 머리 대신 root 원점). `vision_sensor_debug.py`에 정사각형 외곽선 표시를, `humanoid_ma_mixed_sensor.py`에 metadata 8을 추가했다. 전용 train/test/vnc/watch/watch_vnc 스크립트를 만들고 `config.md`·`structure.md`를 갱신했다.
- 검증: CPU 70개 통과(신규 2개). PhysX 96환경×6 reset에서 센서 중심과 root의 오차는 ≤0.12mm였고, 앞·뒤 변이 heading 방향 ±10m에 있었으며, 관측과 재계산 값이 일치했다. GPU 0에서 2048환경·`MAX_ITERATIONS=1` 전이 학습·저장(복사 169·신규 12, 센서 첫 층 256×2304), fan mixed checkpoint 거부, 96환경·64-step 평가를 완료했다. 실제 viewer 세 장면 캡처도 확인했다. 장기 학습 성능은 검증하지 않았다. 증거: `output/approach_stage2_square_mixed_check/`.
- 참고: GPU 실행 env는 `tokenhsi_sm120`이다(`tokenhsi` env의 torch는 CPU 빌드). 확인 스크립트에서 `env.reset()`을 반복했을 때 장면 비율이 cliff 약 65%로 나왔다. 같은 seed의 fan mixed config에서도 수치가 완전히 같아 이번 변경과는 무관하며, 원인은 조사하지 않았다. 실제 평가 경로의 장면 수는 29/37/30이었다.

## 2026-10-06

### 혼합 실험 기본 Stage 1 checkpoint 변경

- 사용자 요청으로 `approach_stage2_fan_mixed_train.sh`·`_watch.sh`·`_watch_vnc.sh`에서 `STAGE1_CHECKPOINT`를 생략하면 `output/stage1/8000.pth`(schema 9, `scenario_independent_stage1_self_sum_rescue`, epoch 8000)를 사용하도록 했다. 명시적 `STAGE1_CHECKPOINT`·`RESUME_CHECKPOINT`가 우선한다. 다른 실험 스크립트는 바꾸지 않았다.
- 검증: 셸 문법 확인. `STAGE1_CHECKPOINT` 없이 GPU 0에서 **2048환경·`MAX_ITERATIONS=1`** 전이 학습·저장을 완료했다. `stage2_transfer_report.json`에 source `output/stage1/8000.pth`, epoch 8000, 복사 169·신규 12 tensor가 기록됐다. 증거: `output/approach_stage2_fan_mixed_wallcpa_check/train.log`, `train/ApproachStage2FanMixed_06-13-06-49/`.

### Room 벽 CPA를 사람 CPA 재사용 방식으로 교체

- 원인: `output/approach_stage2_fan_mixed_cpa_static/ApproachStage2FanMixed_30-19-43-57` 로그 분석에서 room 에이전트가 정지하는 현상을 확인했다. 기존 선분 궤적 CPA는 진행 방향이 벽에 닿으면 예측 최근접 거리가 0이 되고, 할인율 0.99(반감기 2.3s) 때문에 닫힌 방에서는 어느 방향으로 걷든 −0.15~−0.31/step을 받았다. 진단 CSV에서 room의 접근 중 collision은 −0.134(bridge −0.014), 총보상은 −0.045(정지 시 +0.126)였다. 시작 거리 3.5m 이상 HOLDING 성공률은 room 8%, bridge 95%였다.
- 변경: `compute_wall_cpa_collision_penalty`는 벽 선분마다 root와 가장 가까운 점을 속도 0의 가상 에이전트로 두고 `compute_agent_cpa_collision_penalty`를 그대로 호출한다. 벽 전용 `wallCollisionTTCDiscount: 0.95`를 room·room partition·mixed config에 추가했다. 사람 간 0.99·계수 0.5·0.7m·단일 collision 항·관측·모델은 그대로이며 벽은 토큰에 넣지 않는다. 같은 7×7m 방에서 1m/s 무작위 보행의 평균 벽 패널티는 0.221에서 0.041로 줄었다(오프라인 계산).
- 기본 output: `_cpa_static` → `_cpa_static_wall`(mixed), `_cpa` → `_cpa_wall`(room, partition). watch·masked도 같다. checkpoint 형식은 바뀌지 않아 기존 checkpoint를 로드할 수 있지만, 새 벽 보상으로 학습된 가중치는 아니다.
- 검증: CPU 43개 통과(`test_room_scenario`·`test_room_partition`·`test_mixed_sensor`·`test_bridge_collision`·`test_bridge_scenario`). 선분 dense 참조 테스트는 정지 최근접점 사람 CPA와 일치·비스듬한 회피 방향 0 검사로 대체했다. GPU 0 PhysX: mixed 96환경 smoke 통과(`single_cpa_reward`). partition room 64환경 smoke 통과(collision 연결, 벽 위험도 128/128 양수). 증거: `output/approach_stage2_fan_mixed_wallcpa_check/`, `output/approach_stage2_fan_room_partition_wallcpa_check/`. 학습은 실행하지 않았으며, 정지 현상 해소는 아직 검증하지 않았다.

## 2026-09-30

### noVNC 현재 세션 해상도 변경·복원

- 사용자 요청으로 noVNC 5802의 디스플레이 `:2`를 XRandR로 **1600×900 → 1280×720**으로 변경했다. 현재 세션에 적용했으며 서비스 시작 설정은 유지한다. 실제 해상도 조회와 noVNC HTTP 200을 확인했다.
- 이후 사용자 요청으로 **1600×900**으로 복원했다. 실제 해상도와 noVNC HTTP 200을 확인했다.

### 혼합 시나리오 reset 직후 튕김 수정

- 무대·방 바닥/벽을 고밀도 동적 actor로 옮기던 구조를 고정 actor 9개로 교체했다. 세 장면을 32m 간격으로 만들고 reset 때 사람·상자·다리만 활성 공간으로 이동한다. 관측 원점·정규화·viewer 센서/목표/카메라 좌표를 맞췄다. 무대 B의 횡방향 위치는 환경 생성 때 랜덤으로 고정하며 나머지 배치·다리 일정은 reset마다 샘플링한다.
- 랜덤 RSI의 발 메시 최저점을 계산해 바닥 겹침을 최소 5mm 여유로 보정한다. 자세·속도·과제 25%·장면 각 1/3·센서·단일 CPA 보상은 유지한다. 혼합 실험에 적용하며 독립 무대/다리의 물리 구성은 유지한다. 새 checkpoint에 `physics_metadata.version: 1`, 새 출력에 `_cpa_static`을 사용한다. 같은 pickup 설정의 기존 혼합 checkpoint는 계속 사용할 수 있다.
- 검증: CPU **60개 통과**. PhysX **96환경×24 reset**의 9종 전환·부분 reset·센서/좌표·지지/칸막이 접촉 통과, 고정 지형 이동 **0m**. RSI 자세 유지 및 기존 500-epoch checkpoint 각각 **32환경×64 reset**에서 직후 0.2초의 튕김 **0/2048건**(상승 속도 >3m/s 또는 높이 증가 >0.4m 기준). 최대 상승 속도는 각각 **0.545/0.386m/s**, 최대 높이 증가는 **4.85/3.19cm**였다.
- 다리·작업 상자 위 지지 각각 **32/32** 통과, 하중 아래 다리 이동 최대 **1.1mm**. 실제 viewer 세 장면 촬영·센서 정렬 확인. **2048환경·2 epoch** 학습/저장 및 새 checkpoint **96환경·최대 64-step** 평가를 완료했다. 장기 학습 성능·모든 접촉 상태의 안정성을 입증하는 결과는 아니다. 기존 학습·viewer 프로세스는 재시작하지 않았다.
- 증거: `output/approach_stage2_fan_mixed_static_check/`의 `cpu.log`, `physics/physics.json`, `reset_hold/reset_physics.json`, `reset_policy/reset_physics.json`, `support.json`, `train.log`, `checkpoint.json`, `eval/metrics/`, `viewer/`. 실행법·재현 명령은 `markdowns/config.md`, 간단한 명령은 `markdowns/mixed_sensor_commands.md`에 갱신했다.

### Room 픽업의 랜덤 칸막이 우회

- 사용자 요청에 따라 partition room과 혼합 환경의 room에서 HOLDING/HOLDING_AT 에이전트별로 자기 상자와 같은 쪽/반대편 시작을 각각 50%로 선택한다. 반대편이면 사람–상자의 직선 경로가 칸막이에 막히도록 배치한다. 선택한 쪽은 겹침 재샘플링 중 유지한다. SIT/CLIMB은 같은 쪽, AT 목표는 상자의 반대편을 유지한다.
- 두 config에 `room.partition.pickup_cross_probability: 0.5`를 추가했다. 학습·평가 공통 reset에 적용하며 과제 25%·시나리오 각 1/3·loco RSI·센서·보상은 유지한다. 이 필드가 없는 이전 checkpoint는 기존 metadata 검사에서 거부되므로 새 배치는 Stage 1 전이로 시작한다. config·명령·구조 문서와 이전 화면 보고서의 배치 설명을 갱신했다.
- 검증: CPU **51개 통과**(최소 방·최대 상자 2048환경 포함). 실제 PhysX 혼합 **96환경×24 reset**에서 room의 반대편 픽업 HOLDING **189/359**, HOLDING_AT **221/425**, 단독 room **64환경×8 reset**에서 각각 **120/265**, **123/223**건을 확인했다. 반대편 픽업 모두 직선 경로가 칸막이에 막히고 SIT/CLIMB 반대편은 0건이었다. AT 목표·실내 배치·부분 reset·물리 접촉 검사도 통과했다.
- GPU 0의 **2048환경·2 epoch** 전이 학습·저장과 새 checkpoint의 **96환경·최대 64-step** 평가를 완료했다. 평가 장면은 단일 무대/다리/room **31/33/32**, 장면별 네 과제 모두 발생했다. actor encoder 75개·관측 RMS 6개 불변, 모델·scalar 256종 유한값, 이전 배치 checkpoint 거부를 확인했다. 짧은 실행 검증이며 우회 행동의 학습 성능은 검증하지 않았다.
- 증거: `output/approach_stage2_fan_mixed_pickup_check/`의 `cpu.log`, `mixed_physics/physics.json`, `room_physics/physics.json`, `train.log`, `checkpoint_audit.json`, `eval/metrics/`. Python 문법·diff·문서 경로 확인.

### 혼합 watch의 4환경 minibatch 오류 수정

- 사용자가 4환경 watch에서 보고한 `batch_size % minibatch_size` 실패를 수정했다. watch가 본학습용 PPO minibatch 16384를 사용해 4환경×2명×32-step=256인 rollout과 맞지 않았던 것이 원인이다.
- `approach_stage2_fan_mixed_watch.sh`를 기존 `amp_ma_cliff_sensor_watch.yaml`에 연결하고, watch/VNC 기본 환경 수를 4로 수정했다. PPO/AMP minibatch 64·AMP buffer 16384이며, 본학습 스크립트·2048환경 config는 유지한다. 명령 파일·config·structure도 4환경 사용법으로 갱신했다.
- 검증: 1/4/8환경 배치 나눗셈·PPO/AMP 크기와 네트워크 동일성, 셸/명령 문법·diff 통과. 실제 noVNC viewer에서 **4환경·2 epoch** 전이 학습·checkpoint 저장·정상 종료를 확인했다. 로그: `output/approach_stage2_fan_mixed_watch_4env_check/watch.log`.

### 혼합 실험 명령 파일·학습 viewer

- 요청한 학습·테스트·학습 시각화·테스트 시각화 명령만 `markdowns/mixed_sensor_commands.md`에 정리했다. 로컬 viewer와 서버 VNC 명령을 함께 제공한다.
- 혼합 실험의 `approach_stage2_fan_mixed_watch.sh`·`_watch_vnc.sh`를 추가했다. 본학습 config·2048환경·랜덤 샘플링을 유지하며 viewer와 센서 표시를 켜고 별도 `_cpa_watch` 출력에 저장한다. config·structure에 연결했다.
- 검증: 명령 블록·스크립트 셸 문법·경로·diff 확인. GPU 0의 실제 noVNC viewer에서 `MAX_ITERATIONS=1`, 2048환경·2 epoch 학습과 checkpoint 저장 후 정상 종료했다. 로그: `output/approach_stage2_fan_mixed_watch_check/watch.log`.

### 단일 무대·연결다리·Room 혼합 학습과 랜덤 평가

- 사용자 요청에 따라 `HumanoidMAMixedSensor`, `approach_stage2_fan_mixed.yaml`, 전용 train/test/VNC를 추가했다. 학습·평가 모두 reset마다 세 장면 각 1/3, agent별 HOLDING/SIT/CLIMB/HOLDING_AT 각 25%, loco 랜덤 RSI를 사용한다. 기존 개별 실험은 유지한다.
- 단일 무대의 바깥 상자는 과제 미완료·낙하 회피가 의도된 배치를 유지한다. AT 목표는 무대 A 위, 연결다리는 B 상자→A 목표, room은 칸막이 반대편이다. graph goal slot·실제 source box 높이를 사용한다.
- 동일 actor 구성에서 고밀도·중력 없는 지형을 reset 시 이동하며, 비활성 지형은 서로 떨어진 상공에 두고 센서/지형/벽 CPA에서 제외한다. floor·wall 물리 겹침을 없앴다. 사람 token의 높이는 활성 장면 바닥 기준으로 보정하고 관측 2655차원·단일 CPA 항·기존 종료를 유지한다. metadata 7로 개별 장면 checkpoint와 구분한다.
- TensorBoard에 시나리오별 경험량·종료·낙상·과제 샘플/성공 step을 기록하고, 평가 JSON에 최초 trial 장면/과제를 보존한 분리 결과를 추가했다. 실행법은 `markdowns/config.md`의 **Stage 2 세 시나리오 혼합** 참조.
- 검증: CPU **49개 통과**. PhysX 96환경×24 reset의 9종 전환·AT 목표 **1,163건**·부분 reset 48환경 불변·좌표/센서 제외·단일 CPA 확인. 실제 room→단일 무대→다리→room 전환에서 지지 **96/96**, room 칸막이 접촉 검사 각각 **96/96**, 지형 최대 이동 **0.057mm**.
- GPU 0에서 **2048환경·2 epoch** 전이 학습/저장 완료. actor encoder 75개·관측 RMS 6개 불변, 모델·scalar 256종 유한, sensor head 갱신, 이전 partition checkpoint 거부 확인. 저장 checkpoint로 96환경×3회·64-step 랜덤 평가를 완료했고, 단일 무대/다리/room **81/108/99**개 및 각 장면의 네 과제를 확인했다. AT 성공 0; 장기 학습 성능을 입증한 결과는 아니다.
- 증거: `output/approach_stage2_fan_mixed_cpa_check/`의 `cpu.log`, `physics/physics.json`, `checkpoint_audit.json`, `train.log`, `eval/metrics/`. Python/셸 문법·diff·문서 경로 확인.

### Room 내부 칸막이·HOLDING_AT·새 실제 화면

- 사용자 요청과 과제 비율 선택에 따라 `approach_stage2_fan_room_partition` config/train/test/VNC를 추가했다. HOLDING/SIT/CLIMB/HOLDING_AT 각각 25%, 중앙 칸막이와 양 끝 1.8m 통로, 상자와 반대편의 실내 운반 목표를 사용한다. 기존 plain room 실험·output과 구분했다.
- 기존 room 클래스에 칸막이 actor·센서 표면·4개 면 CPA를 추가하고, 충돌 보상은 기존 항 하나에 최댓값으로 통합했다. 목표는 graph binding에 따라 배치하며 Z는 바닥+대상 상자 높이/2다. marker 활성 판정도 owner 대신 실제 goal slot을 따라가도록 수정하고, 목표를 담당 에이전트 색으로 표시한다. metadata version 6으로 plain room checkpoint를 구분한다.
- 작은 방에 큰 상자가 몰릴 때 초기 샘플링 실패를 확인하고, 과제 대상 상자는 칸막이 길이 안쪽에 두되 비할당 상자의 실내 배치 범위를 넓혔다. 미래 목표에는 현재 사람이 서 있을 수 있도록 하고, 물리 객체끼리·목표와 초기 상자의 겹침을 배제했다.
- 검증: CPU 43개 통과(6×6m 방·최대 상자 발자국 2048개 포함). PhysX 64환경×8 reset에서 사람 1,024건·상자 1,536건 모두 실내, 활성 AT 목표 238건 모두 칸막이 반대편·높이/관측 일치, 그중 107건은 owner와 다른 goal slot. 외벽 충돌 64/64·칸막이 양면 충돌 64/64·사람 벽 접촉 비종료 64/64 통과.
- 수정 후 GPU 0의 2048환경·2 epoch 전이 학습/저장과 16환경·64-step 평가 완료. actor encoder 75개·관측 RMS 6개 불변, 모델·scalar 193종 유한, 센서 head 갱신 및 이전 room checkpoint 거부 확인. 실제 viewer 3개 장면·7개 PNG를 다시 촬영하고 직접 검수했다. Python/셸 문법·문서 링크·diff 검사 통과.
- [새 스크린샷 보고서](markdowns/room_partition_report_20260930.md), `markdowns/assets/room_partition_20260930/`, `output/approach_stage2_fan_room_partition_cpa_check/`에 증거를 보관했다. 성공한 학습은 `train_fixed/`; `train/`은 배치 수정 전 실패 기록이다. 우회 운반의 장기 학습 성능은 이번 smoke에서 검증하지 않았다.

### 4면 벽 Room·단일 사람/벽 CPA 충돌 보상

- 사용자 요청으로 `HumanoidMARoom`·`room_geometry.py`와 별도 `approach_stage2_fan_room` config/train/test/VNC를 추가했다. 내부 6–8m·벽 높이 2.5m·두께 0.4m의 닫힌 방 안에 사람 2명·작업 객체 3개·비활성 목표를 배치한다.
- 1024개·100도·10m·가림 없는 센서에 네 벽의 높이를 포함했다. 벽은 graph 토큰에 넣지 않으며 관측은 2655차원이다. 실제 벽 안쪽 선분과 root XY 속도로 CPA를 계산해 `-0.5 × max(사람 위험도, 벽 위험도)`를 기존 collision 항 하나에 기록한다. 안전거리 0.7m·시간 할인 0.99를 공유한다.
- 벽 접촉만으로 종료하지 않으며, 벽 상단을 낙상 판정 지형으로 쓰지 않는다. 방 바닥 기준 넘어짐·시간 제한 종료는 유지한다. room metadata version 5로 다리/무대 checkpoint를 구분하고 새 Stage 1 전이를 사용한다.
- 검증: CPU 37개 통과. PhysX 64환경×8 reset에서 에이전트 1,024건·객체 1,536건 모두 실내, 신체 중심 최소 벽 여유 0.406m·상자 외접원 0.2005m, 모든 에이전트가 벽 센서 샘플 최소 21개 관측. 4방향 상자 충돌 64/64, 서 있는 사람의 벽 접촉 후 비종료 64/64, 단일 collision 보상 연결·실제 actor 좌표 일치를 확인했다.
- GPU 0에서 2048환경·2 epoch 전이 학습/저장과 16환경·64-step 평가 완료. actor encoder 75개·관측 RMS 6개 불변, 모델·scalar 173종 유한, 센서 head 갱신, 기존 bridge checkpoint 거부 확인. Python/셸 문법·보고서 링크·diff 검사 통과.
- 실제 viewer의 3개 장면·7개 PNG를 촬영하고 대표 화면을 직접 열어 확인했다. [room 보고서](markdowns/room_scenario_report_20260930.md), `markdowns/assets/room_20260930/`, 실행 증거 `output/approach_stage2_fan_room_cpa_check/`. config·structure 문서에 실행·검증 경로를 추가했다. 장기 벽 회피 성능은 이번 smoke로 검증하지 않았다.

### 다리 실험의 충돌 패널티를 CPA로 교체

- 사용자 요청으로 `HumanoidMABridge`의 기존 root XY 거리 기반 `collision`을 `TokenHSI_gta_ontop_supp_weak`의 CPA 식으로 교체했다. 상대 XY 위치·속도의 연속 접근 정도, 예상 최근접 거리와 시간 할인을 사용한다. 0.7m 안에서도 서로 멀어지거나 상대속도가 0이면 감점은 0이다. 단일 무대·다리 장면 모두 기존 `collision` 항 하나에 기록하며, 바닥 접촉 실패와는 별개다.
- 계수 0.5·안전거리 0.7m를 사용하고 `agentCollisionTTCDiscount: 0.99`를 추가했다. train/test/watch의 기본 output을 `_cpa` 경로로 분리하고 config·실행 명령·구조 문서를 갱신했다. 관측·모델·sensor metadata는 같아 기존 다리 checkpoint의 metadata·tensor shape 호환을 확인했다. 실행 중인 프로세스에는 소급 적용되지 않는다.
- 검증: CPU 테스트 27개 통과(근거리 이탈·정지·동일 속도, 정면 접근·엇갈림·self 제외, 실제 relation 총 보상·로그 연결 및 기존 무대/다리 회귀). 원본 함수와 CPU/CUDA 각각 2048환경 비교의 최대 오차는 0이었다.
- GPU 0에서 2048환경·`MAX_ITERATIONS=1`(2 epoch) 전이 학습/저장과 16환경·최대 64-step 평가를 완료했다. Actor encoder 75개 tensor·관측 RMS 6개 불변, checkpoint 가중치·scalar 185종 유한값, `reward_terms/collision` 음수 기록을 확인했다. Python/셸 문법·diff 검사를 통과했다.
- 산출물: `output/approach_stage2_fan_bridge_cpa_check/`의 `reference_check.json`, `checkpoint_audit.json`, `train/`, `eval/`. 짧은 실행 검증이며 장기 충돌 회피·건너기 성능은 검증하지 않았다.

### 다리 장면 거리 축소·도착 일정 재설계·좌표계 검수

- **거리 축소:** 다리 장면의 에이전트–대상 거리를 Stage 1 범위로 줄였다.
  - 무대 B를 4–5m로 줄였다.
  - 상자와 목표를 무대 B의 간격 쪽 1–2m 띠에 모았다. 목표는 띠 중앙의 한 점이다.
  - 무대 A는 단일 무대 장면과 공유되므로 크기는 두고, 일반 시작 위치를 간격 쪽 0.8–3.0m와 두 무대가 겹치는 y 구간으로 제한했다.
  - CPU 샘플링 기준 에이전트–자기 상자 거리는 p50 5.9m, p90 7.0m, p99 7.7m가 됐다. Stage 1은 p99 7.8m다.
- **도착 일정:** 사용자가 보고한 "다리가 안 움직임"을 재현했다.
  - 원인은 무게가 아니었다. 0–6초 대기 후 이동하는 동안, 초기 정책의 에피소드가 1–2초에 끝나 다리가 출발 전에 reset됐다. 학습 확인에서 접근 에피소드 545개 중 출발 31%, 도착 0개였다.
  - 대기 학습도 유지하도록, 도착 시각을 1–18초 로그균등으로 정하고 평균 속도 0.3–1.5m/s 안에서 계속 움직이게 했다(`bridge_schedule`).
  - 늦으면 출발점에서 대기하고, 빠르면 경로 중간에서 움직이는 상태로 시작한다. 20% 즉시 연결과 단일 무대 50%는 유지한다.
  - 진단에 `approach_episodes/started/arrived`를 추가했다.
- **좌표계 검수:** `bridge_physics_smoke.py`(64환경)에 다음 검사를 추가해 모두 통과했다.
  - 무대 B 앞 벽과 간격 끝 일치, 다리 채널 위치·높이·유격, 초기 다리 상태의 궤적 일치.
  - 사람 루트가 무대 A 위 0.86–0.95m, 1 step 후 이동 4.9cm 이하. 상자 바닥이 무대 B 위 9mm. 목표 위치.
  - GTA 관측의 사람·상자·목표 자세, heading 오차 0.696rad 이하(jitter 0.7).
  - 무대·다리·간격 지면 높이. 이동 중인 다리 위 지점 3개 포함.
  - 1024개 센서를 독립 NumPy로 재계산: reset 경계 외 최대 오차 4e-8(상자 경계 1mm 이내 불일치 1/131,072), 다리 도착 후 8e-8.
  - 상자 192/192 무대 B 위, 다리 위 64/64 지지, 간격 바닥 64/64 실패.
- **시각 검수:** 실제 viewer로 두 경우를 시간순 촬영해 직접 확인했다.
  - 멀리서 9.5초에 걸쳐 채널로 들어와 간격을 채움.
  - 1.35초 도착을 위해 경로 중간에서 이동 중으로 시작.
  - 사람은 무대 A의 간격 쪽에, 상자는 무대 B의 간격 쪽 띠에 있었다. reset 직후 첫 프레임은 시뮬레이션 전 화면이라 1 step 뒤부터 촬영했다.
  - 산출물: `output/approach_stage2_fan_bridge_check/bridge_motion_v2/`.
- **학습·평가 확인:** CPU 테스트 31개 통과.
  - GPU 0에서 2048환경 2 epoch 전이 학습: encoder 75개·RMS 6개 불변, scalar 185종 유한.
  - 초기 정책의 에피소드는 여전히 0.8–1.8초지만, 접근 에피소드 454개 중 출발 452개, 도착 56개가 됐다.
  - 16환경·64-step 평가 경로를 확인했다.
  - 산출물: `output/approach_stage2_fan_bridge_check/{train_timing,eval_timing}/`.
  - 이전 `train_near_edge`, `bridge_motion`은 각각 일정 변경 전 결과, 시뮬레이션 전 첫 프레임을 포함한 결과다.

### 다리 장면 목표를 무대 B 위로

- 사용자 요청으로, 두 무대 장면의 목표(T 노드)를 env 원점에서 무대 B 윗면 중앙으로 옮겼다. 원점은 무대 A 중앙의 바닥 높이로, 무대 내부에 해당한다. 이 실험은 HOLDING_AT·ON_TOP을 샘플하지 않아 목표는 보상 없이 관측에만 쓰인다. 단일 무대 장면은 기존과 같다.
- carry의 `_reset_task`가 목표를 원점으로 덮어쓰므로, `HumanoidMABridge._reset_task`에서 그 뒤에 설정한다.
- 검증: PhysX 16환경 smoke를 통과했다. 다리 장면의 `_tar_pos`와 관측 목표 자세가 무대 B 윗면 중앙과 일치하고, 단일 무대 장면은 원점을 유지했다. 기존 smoke 항목도 모두 통과했다.
- 거리 확인: 에피소드 시작 시 에이전트–대상 거리는 Stage 1 진단 기준 p50 1.2m, p90 5.0m, p99 7.8m다. 다리 장면 배치를 CPU 샘플링하면 에이전트–자기 상자 거리가 p50 8.2m, p90 11.1m, p99 13.5m로 Stage 1 범위를 넘는다. 단일 무대는 p50 3.7m, p99 6.2m다. `envSpacing`은 이 실험의 관측(clean_scene·GTA, 미터 단위)에 쓰이지 않는다.

### 두 무대·연결다리 Stage 2

- 정지와 전진을 모두 학습하도록 새 실험 `approach_stage2_fan_bridge`를 추가했다. reset마다 기존 단일 무대 고립 장면과 두 무대 장면을 5:5로 샘플한다. 두 무대 장면에서는 외력으로 구동하는 다리 상자가 무작위 시각·위치에 간격을 채우며, 과제 상자는 건너편 무대 위에 놓인다.
  - 다리 구동은 `TokenHSI_gta_ontop_supp_weak`의 quintic 궤적·PD 힘 방식을 따랐다.
  - RSI는 loco만 쓰고, 다리 착지점 근처 시작 확률을 0.5에서 0.2로 줄여 간다.
  - Stage 1에서 새로 전이한다.
  - config, train/test/VNC/watch/watch_vnc 스크립트, `HumanoidMABridge`, `bridge_geometry.py`를 추가했다.
  - 명령은 `fan_sensor_commands.md`, 설계는 `config.md`에 정리했다.
- 사용자 결정으로 이 실험의 센서는 가림 판정 없이 수직 조회하며 1024개(32고리)를 쓴다. 관측은 2655차원, sensor metadata는 version 4다.
  - 가림이 있으면 멀리서 간격이 무대 B 높이로 채워져, 다리 유무를 구분할 수 없음을 CPU로 확인했다.
  - OBB 6개·2048환경에서 센서 함수의 CUDA event 중앙값은 가림 경로 13.9ms, 수직 조회 2.2ms(1024개)였다. 576개에서는 13.6ms와 1.9ms였다. 공유 GPU에서 측정했으며 전체 학습 속도 배율은 아니다.
- `humanoid_ma_cliff.py`에 `sensor_occlusion: none`, `_sensor_boxes`, `_sample_scene`, `_terrain_height` hook을 추가했다. 기존 fan 실험의 경로와 값은 바뀌지 않는다.
- `humanoid_ma_carry.py`에 `_extra_reset_actor_ids`를 추가했다. 추가 actor의 root state를 별도로 호출하면 carry의 상자 배치가 무효가 되는 문제를 확인했고, 이를 단일 호출로 합쳐 해결했다.
- **검증**
  - CPU 테스트: bridge·vision·cliff sensor 30개와 cliff 좌표 3개가 통과했다. 수직 조회는 서 있는 상자에서 기존 slab 표면과 1e-4 이내로 일치했다.
  - PhysX 16환경:
    - 다리 좌우 오차 최대 7.6µm(유격 1cm), 최종 위치 오차 0.08mm, 회전 0.
    - 무대 B 이동 0.04mm, 과제 상자 48/48개가 무대 B 위.
    - 다리 위에 선 사람 16/16 지지·비종료, 하중에 의한 다리 이동 1.0mm.
    - 간격 바닥 접촉 16/16 실패, 띄운 actor 이동 0.
  - GPU 0에서 2048환경·`MAX_ITERATIONS=1`(2 epoch) 전이 학습을 저장까지 완료했다.
    - Actor encoder 75개와 관찰 RMS 6개는 Stage 1과 같고, 센서 head는 갱신됐다.
    - scalar 182종이 모두 유한값이며 `cliff/bridge/*`가 기록됐다.
  - 16환경·64-step `BRIDGE_PROBABILITY=1` 평가와 1환경 viewer 촬영(다리 이동·연결, 상자 배치, 센서 표시)을 확인했다.
  - 산출물은 `output/approach_stage2_fan_bridge_check/{train_fixed,eval_fixed,screenshots_fixed}`, `output/approach_stage2_fan_bridge_physics_check/`다.
  - 같은 폴더의 `train`, `eval`, `screenshots`는 상자 배치 수정 전 결과이므로 사용하지 않는다.
  - 실행 경로 확인이며, 장기 학습의 정지·건너기 성능은 검증하지 않았다.

## 2026-09-29

### 10m 센서 학습·테스트 명령 모음

- `markdowns/fan_sensor_commands.md`에 공통 준비, 2048환경 본학습, 소수 환경 watch, checkpoint 선택, noVNC/로컬 테스트 시각화, headless 확인, resume 명령을 정리했다. 현재 출력 경로·checkpoint 호환 조건·향후 carryWith 상자 제외 요구를 명시하고 config/structure 문서에서 연결했다.
- Bash 코드 블록 10개 문법, 실행 스크립트 5개 경로, 문서 링크와 Stage 1 checkpoint 경로, diff를 확인했다. 문서만 변경했으며 학습·시뮬레이션은 실행하지 않았다.

### 부채꼴 가시거리 10m

- 사용자 요청으로 `approach_stage2_fan_sensor.yaml`의 `sensor_range`를 20m에서 10m로 줄였다. 576개·100도·고리 연결·가림 알고리즘은 유지한다. 고리 간격 약 0.417m, 접선 간격 약 0.364m이며 기본 train/test/watch output에 `_10m`를 붙여 이전 실험과 분리했다.
- 실행 문서에 2048환경 본학습 명령과 새 출력 경로를 반영했다. 기존 20m checkpoint는 거리 metadata가 달라 직접 resume할 수 없다. 센서 문서의 이전 학습·벤치마크·스크린샷은 20m 결과로 명시했다. 향후 carryWith 상자 제외 요구는 유지한다.
- 센서 CPU 테스트 14개 통과. 실제 YAML에서 생성한 576개 좌표가 이전의 정확히 절반이며 후방 연결이 같은 것과 20m metadata 거부를 확인했다. 셸 문법·diff 검사 통과. 이번 변경에서 시뮬레이션·본학습은 실행하지 않았다.

### 부채꼴 576개·100도 균일 배치와 중복 계산 제거

- 부채꼴 설정을 24개 고리의 1·3·…·47개 샘플(총 576개), 전방 100도·20m로 변경했다. 근거리 과밀을 줄이고 사용자 승인대로 다음 고리의 가장 가까운 방향을 연결해 첫 직접 관측 높이로 채운다. 무대/상자 모두 일반 OBB 표면·가림 판정을 유지하며, 현재 상자 포함과 향후 carryWith 제외 요구도 유지한다.
- 수직 ray 방향의 점별 중복 변환을 제거하고 1–4환경은 물체 최대 4개씩, 큰 환경은 하나씩 처리한다. viewer는 env 0의 학습 관측 캐시를 재사용한다. metadata version 3, 관측 1759차원으로 변경하고 기본 output을 `approach_stage2_fan_sensor_576_100` 및 `_watch`로 분리했다. 이전 센서 checkpoint는 호환되지 않는다.
- CPU/회귀 25개·Python/셸 문법 검사 통과. GPU 0에서 2048환경 2 epoch 학습/저장, 16환경·64-step 평가, noVNC 4환경 2 epoch 전이 및 1환경 resume 4 epoch를 확인했다. 세 실행의 actor encoder 75개 tensor/RMS 불변, head 갱신, 모든 scalar 유한값을 확인했다. 5개 가림 장면 촬영과 viewer의 센서 재계산 방지도 검증했다.
- 공유 GPU 센서 함수 측정 중앙값: 2048환경 12.131→4.825ms, 4환경 3.363→1.116ms (이전 1024개 대비). 전체 학습 가속 비율·장기 회피 성능 검증은 아니다. 산출물은 `output/approach_stage2_fan_sensor_576_check/`, 새 화면은 `markdowns/assets/fan_sensor_576_100_20260929/`이며 문서의 이전 1024개 결과는 과거 버전으로 구분했다.

### 부채꼴 센서 소수 환경 학습 watch/noVNC

- `approach_stage2_fan_sensor_watch.sh`·`watch_vnc.sh`를 추가했다. 부채꼴 config와 기존 작은 배치 PPO/AMP config를 사용해 기본 4환경·최소 1환경에서 실제 학습과 센서 표시를 함께 실행한다. Stage 1 새 전이와 부채꼴 checkpoint resume를 지원하고 기본 output은 `output/approach_stage2_fan_sensor_watch`다. 실행 가이드·구조 문서에 명령과 역할을 기록했다.
- GPU 0의 기존 noVNC 서비스에서 4환경 새 전이 2 epoch, 그 checkpoint의 1환경 resume 4 epoch 도달과 각각의 checkpoint 저장을 확인했다. 4환경 scalar 168종·1환경 scalar 151종 모두 유한값, resume action head 갱신을 확인했다. 셸 문법·문서 링크·diff 검사 통과. 검증 결과는 `output/approach_stage2_fan_sensor_watch_check/verification.json`이며 확인 실행은 종료했다. 기존 학습과 GUI 서비스는 유지했다.

### 머리 시점 1024차원 부채꼴 센서와 가림 검증

- 정사각형 실험과 분리한 `approach_stage2_fan_sensor` config/train/test/VNC를 추가했다. 머리 시점·root heading 기준 전방 120도/20m, 근거리 밀집 32×32 샘플로 무대·바닥·회전 상자 표면을 읽는다. 정확한 OBB 선분 교차로 가림을 판정하고 같은 방향의 가장 가까운 뒤쪽 직접 관측 높이로 채우며, 끝까지 미관측이면 정규화 뒤 2.0을 넣는다. 관측 2655차원과 네트워크 구조·기존 보상/종료는 유지하며 metadata version 2로 이전 센서 checkpoint 혼용을 차단한다.
- `markdowns/fan_sensor.md`에 관측 계약과 실제 화면을 정리했다. **현재 낙하 실험에만 상자를 포함하며 향후 carryWith에서는 상자를 제외해야 한다**는 사용자 요구와 높이/가림 제외 범위의 후속 결정 사항을 명시했다. 운반 높이 상자가 무대를 가리면 먼 낮은 바닥으로 잘못 채울 수 있는 한계도 기록했다.
- CPU/회귀 22개, Python/셸 문법·문서 링크·diff 검사 통과. GPU 0·2048환경·MAX_ITERATIONS=1(2 epoch) 학습/저장과 16환경·64-step headless, 1환경·16-step VNC 평가를 완료했다. Actor encoder 75개 tensor·관찰 RMS 불변, 센서 head 가중치 갱신, scalar 173종 유한값을 확인했다. 기존 GPU 작업과 GUI 서비스는 유지했다.
- 실제 viewer의 절벽·낮은 회전 상자·높은 벽·운반 높이 상자·heading 90도 장면에서 PNG 10개/NPZ 5개와 상태 개수를 저장했다. 원본은 `output/approach_stage2_fan_sensor_check/`, 대표 화면은 `markdowns/assets/fan_sensor_20260929/`다. 통제된 센서/실행 검증이며 장기 회피 성능 또는 실제 상자 운반 성공의 검증은 아니다.

### 센서 Stage 2 직접 시각 평가·한국어 실험 보고서

- 고정 epoch 1000 checkpoint로 1환경·시드 29/42의 정상/센서 마스킹 noVNC 평가 4회를 직접 확인하고 영상·root 궤적을 저장했다. 각 시드의 초기 관측 hash는 정상/마스킹 간 동일했다. 정상은 모두 599 step 시간 종료, 마스킹은 각각 76/57 step 실패 종료였다.
- 시드 42에서는 두 에이전트의 가장자리 추종 후 꼭짓점 정지를 재현했다. 시드 29에서는 직선 가장자리 정지도 관찰되어 보편적 꼭짓점 정지로 단정하지 않았다. `markdowns/stage2_cliff_sensor_experiment_report_20260929.md`에 동기·설계·종료 조건 변경·epoch 1000까지의 로그·ablation·한계와 후속 검증안을 정리하고 주요 화면·그래프를 문서 assets에 보존했다.
- 이번 산출물은 `output/cliff_visual_report_20260929/`다. MP4 4개 재열기, 상태 JSON·초기 관측 일치, 문서 링크·diff를 확인했다. 기존 학습 프로세스와 GUI 서비스는 유지했고 확인용 실행은 종료했다. 정책·보상 코드는 변경하지 않았다.

### 센서 Stage 2 평가 마스킹 ablation

- `SENSOR_MASK=1` 평가 옵션을 추가해 학습된 체크포인트의 actor·critic 센서 특징 64차원을 head 입력 직전에 0으로 만든다. 기본 평가는 기존 센서 경로를 사용한다. 물리 지형·원래 센서 진단·checkpoint 가중치는 유지하고, 마스킹 평가는 기본적으로 별도 output에 기록하며 JSON에 `sensor_masked`를 표시한다.
- CPU 센서·좌표 테스트 11개, Python/셸 문법·diff 검사를 통과했다. GPU 0에서 동일한 epoch 500 체크포인트·시드 29로 16환경·64-step 정상/마스킹 평가를 각각 완료했다. 정상 0/16, 마스킹 2/16 종료였으나 이는 실행 경로 확인용 소표본이며 성능 차이의 결론은 아니다. 결과는 `output/approach_stage2_cliff_sensor_mask_check_20260929/`에 분리 저장했다. 기존 학습 프로세스는 유지했다.

### 센서 Stage 2 바닥 접촉 실패

- 무대 밖 지면에 발을 포함한 신체 부위가 닿으면 정상 착지여도 실패하도록 종료 조건을 추가했다. 같은 부위의 접촉력·지형 상대 높이·무대 밖 위치를 함께 검사한다. 무대 위 정상 발 접촉, 공중 이탈 허용, 기존 무대 위 낙상 및 정상 동료 value bootstrap은 유지한다. `cliff/*/falls`에 바닥 접촉 실패도 집계한다.
- CPU 센서·좌표 테스트 10개와 Python 문법·diff 검사를 통과했다. PhysX 16환경에서 똑바로 선 발 착지 16/16 실패, 무대 위 32명 지지, 무대/지면 낙상 각각 8/8, 공중 이탈 비종료·정상 동료 truncation을 확인했다. 결과는 `output/approach_stage2_cliff_floor_touch_check_20260929/physics_smoke.json`에 있다. 기존 GPU 프로세스는 유지했으며 새 종료 조건은 프로세스를 새로 실행할 때 적용된다.

### 센서 Stage 2 소수 환경 학습 noVNC

- 실제 학습을 보면서 확인하도록 전용 `approach_stage2_cliff_sensor_watch.sh`·`watch_vnc.sh`와 작은 PPO/AMP minibatch config를 추가했다. 기본 4환경·최소 1환경이며 Stage 1 이식/센서 Stage 2 resume를 지원한다. 본학습 2048환경 config와 출력 경로를 분리하고 센서 표시를 기본 활성화했다.
- 실제 viewer에서 기존 목표 표시가 센서 선을 지우는 문제를 발견해 센서 task의 그리기 순서를 수정했다. noVNC 화면에서 두 에이전트·무대·바깥 객체와 높이 차를 따라 표시되는 센서 그리드를 확인했다.
- GPU 0에서 4환경 21 epoch, 1환경 9 epoch 학습과 checkpoint 저장을 완료했다. 두 실행의 TensorBoard scalar 173종은 모두 유한값이며 셸/Python 문법과 diff 검사를 통과했다. 로그·checkpoint·화면 캡처는 `output/approach_stage2_cliff_sensor_watch_check_20260929/`에 있다. 기존 GPU 프로세스와 noVNC 서비스를 재시작하지 않았고 확인용 실행은 종료했다. 작은 배치의 실행 검증이며 정지·우회 학습 수렴 검증은 아니다.

### 무대·높이 센서 Stage 2

- `HumanoidMACliff`와 전용 config/train/test/VNC를 추가했다. 환경별 4–8m 무대 위 두 에이전트, 무대 밖 세 객체, HOLDING/SIT/CLIMB 각 1/3, loco 전용 RSI를 사용한다. 기존 reward·AMP 모션 분포는 유지하고 지형 상대 높이로 무대 위 낙상도 종료한다.
- 기존 scene 607차원 뒤에 32×32 센서 두 개를 추가했다. Actor encoder/관찰 RMS를 동결하고 sensor encoder·128차원 입력 action head·critic을 학습한다. Stage 1 head는 `[W, 0]`으로 확장하며 sensor metadata로 checkpoint 혼용을 차단한다.
- 초기 검증에서 무대의 환경 원점 오프셋 누락을 발견해 actor 생성 world 좌표를 수정했다. 수정 후 PhysX 16환경에서 32명 모두 발 지지, 무대/지면 낙상 각각 8/8, 정상 동료 truncation, 공중 이탈 비종료를 확인했다. 물리 높이 probe 64개의 최대 오차는 0.0000135m, reset 센서와 독립 NumPy 계산의 최대 오차는 4e-8이었다.
- CPU/회귀 테스트 21개, Python 문법·셸 문법·diff 검사를 통과했다. 지정 Stage 1 epoch 14500에서 169개 tensor를 이식해 수정본 2048환경·`MAX_ITERATIONS=1`(2 epoch) 학습과 저장 checkpoint의 16환경·64-step 평가를 완료했다. Actor encoder tensor 75개와 관찰 RMS는 불변, 센서 입력 가중치는 갱신, scalar 173종은 모두 유한값이었다.
- 최종 검증 산출물은 `output/approach_stage2_cliff_sensor_check_20260929/`의 `train_verified`, `eval_verified`, `coordinate_final`, `checkpoint_audit_verified.json`이다. 이전 `train`/`eval`은 좌표 수정 전 결과이므로 사용하지 않는다. 기존 GPU 프로세스와 MPS 설정을 유지했다. 장기 정지·우회 습득과 실제 VNC 렌더링은 이번 스모크의 검증 범위가 아니다.

### 저장소 독립 VNC 기본 포트 변경

- `run-gui.sh`가 별도 Xvfb 화면을 띄울 때의 기본 포트를 noVNC `5803`, VNC `5903`으로 바꿨다. 저장소 밖 `tokenhsi-gui.service`는 그대로라 이 서버의 GPU 0 서비스 재사용 경로는 `5802`/`5902`를 유지한다. `bash -n`과 `git diff --check`를 통과했고, 이 서버에 Xvfb가 없어 독립 화면 실행은 확인하지 않았다.

### 기존 noVNC 서비스로 VNC wrapper 연결

- 이 서버에는 `Xvfb`가 없고 `tokenhsi-gui.service`가 이미 화면 `:2`와 noVNC `5802`를 제공한다. `run-gui.sh`가 GPU 0에서 해당 서비스를 재사용하도록 연결해 34 distill VNC 명령의 `Xvfb not found` 실패를 해결했다. 12,500 epoch checkpoint로 VNC wrapper 평가를 종료했고, Isaac Gym 1600×900 창과 실제 캐릭터·상자 렌더링 캡처를 확인했다.

## 2026-09-28

### 34번 원본 통합 teacher distillation

- 34번의 graph·보상·PPO·AMP를 유지한 별도 distillation config/train/test/VNC/output을 추가했다. 원본 TokenHSI Stage 1 checkpoint를 동결하고 agent별 graph template에 따라 Carry(HOLDING·AT·ON_TOP), Sit, Climb 관찰과 목표를 만든 뒤 학생 rollout의 `KL(teacher || student)`를 actor loss에 더한다. 원본 34번 및 Stage 2 실행 경로는 유지한다.
- CPU 집중 테스트 29개 통과. `tokenhsi` 환경은 CUDA 없는 PyTorch라 GPU 검증은 `tokenhsi_sm120`에서 했다. GPU 0·2048환경·`MAX_ITERATIONS=1` 확인은 2 epoch와 checkpoint 저장까지 종료했고, 첫 업데이트에서 KL/PPO actor gradient norm 비율 0.036, KL critic gradient 없음, actor 파라미터 변경, teacher 동결을 확인했다. TensorBoard scalar 196종 모두 유한값이었다. 저장 checkpoint의 1환경·32-step teacher 없는 평가와 Stage 2 SIT plane 2048환경·1 iteration 전이 학습도 종료했다. Stage 2는 모델 tensor 169개를 복사하고 협력 tensor 8개를 새로 만들었다.
- Teacher 직접 제어 16환경·599-step 첫 episode에서 agent 0의 목표를 한 번 이상 달성한 수는 HOLDING 7, SIT 13, CLIMB 16, HOLDING_AT 4, HOLDING_ON_TOP 6이었다. 이는 teacher label 적합성의 작은 표본이며 학생 정책 수렴·장기 성공률 증거는 아니다.

### 학습 데이터 심링크 복구

- 기존 데이터 디렉터리 심링크 12개가 없는 `CVPR2027/TokenHSI`를 가리켜, 현재 서버의 `/home/user/jhh/Projects/TokenHSI`로 다시 연결했다. 35번 학습의 motion 목록에서 참조하는 파일 136개와 모든 심링크 대상을 확인했다. 학습 실행은 하지 않았다.

### 뷰어 상자 색을 과제 배정에 연결

- Stage 1·2 로컬/VNC 뷰어에서 graph의 실제 대상 상자를 에이전트 색으로 칠한다. 공동 대상은 노란색, 비대상은 회색이다. 물리 상자 순서와 graph 배정이 달라도 색이 맞도록 변경했다.
- Stage 1 ON_TOP, Stage 2 place_climb/place_stack graph를 사용한 색 지정 호출 검증과 Python 문법 확인을 마쳤다. 실제 뷰어 화면 확인은 아직 하지 않았다.

## 2026-09-27

### Stage 2 SIT plane·34번 보상/독립 과제 변형

- 기존 29번과 분리한 `approach_stage2_coordination_sit_plane` config·train/test/VNC·output을 추가했다. SIT도 상판 안쪽 10%와 높이 오차 7cm로 판정하고, 자기 edge 합계·동료 edge 평균에 0.9/0.1 팀 보상을 적용한다. 에피소드는 place_climb/place_sit/place_stack/독립 20/20/40/20이며, 독립 과제 내부는 34번의 10/10/10/35/35와 RSI·가까운 시작 일정을 따른다. 협력 과제의 RSI는 기존 Stage 2 규칙을 유지한다.
- Stage 2 checkpoint 로드에서 reward config 불일치를 검사하도록 수정했다. 34번 epoch 500 checkpoint로 GPU 6에서 2048환경·1 iteration 이식 학습과 1환경 place_sit 평가를 완료했다. 물리 reset 실패 0, scalar 177개 모두 finite, 관찰·AMP 통계 이식 확인. 관련 CPU 테스트 22개 통과. 장기 수렴은 아직 확인하지 않았다.

### output_etc Git 제외

- `.gitignore`에 `/output_etc/`를 추가했다. `git check-ignore`로 내부 파일 제외와 `git status`에서 해당 폴더가 사라진 것을 확인했다.

### VNC 평가 결과 기본 경로 통일

- 현재 실험 9개의 VNC wrapper가 기본 `OUTPUT_PATH`를 각 실험의 `output/<실험명>`으로 전달하도록 바꿨다. 로컬 test와 같은 위치에 `metrics/`, `diagnostics/`가 생기며 학습 checkpoint의 run 디렉터리와 분리된다. 기존 `_vnc` 폴더와 실행 중인 학습은 변경하지 않았다. 셸 문법과 기본 경로를 확인했다.

### 34·35번 자기 보상 합계와 가까운 시작 일정 비교

- 33번의 AT reset 수정·state/progress/성공 식을 유지하고, 34번은 자기 edge 보상 합계·동료 edge 평균과 HOLDING/SIT/CLIMB/HOLDING+AT/HOLDING+ON_TOP 10/10/10/35/35 샘플링을 적용했다. 35번은 34번에서 가까운 시작 80%를 60만 per-env step까지 유지하고 120만 step까지 30%로 감소시킨다. 두 실험은 별도 config·train/test/VNC·output에서 scratch로 시작하며 33번과 기존 학습 프로세스는 변경하지 않았다.
- 관련 Stage 1·2 CPU 테스트 **30개 통과**, 스크립트 문법·diff 확인. GPU 6에서 각각 2048환경·1 iteration 학습과 checkpoint 저장을 완료했다. 두 실험 모두 물리 reset 실패 **0**, 공유 주 물체 **0**, scalar **177개 모두 finite**였고, 샘플 비율은 목표값에 근접했다. 짧은 진단 표본에서 AT 두 번째 목표 slot 높이도 정상이다. 저장 checkpoint의 1환경·32-step AT headless 평가와 JSON 생성도 각각 완료했다. 이는 실행 경로 검증이며 장기 수렴 검증은 아니다.

### 33번 AT 목표 reset 수정

- 독립 graph의 AT가 두 번째 목표 slot을 지정해도 reset이 첫 번째 slot에 목표를 배치하던 선택 오류를 수정했다. 32번의 보상·샘플링·RSI를 유지한 33번 config와 전용 train/test/VNC·output을 추가했다. 공통 reset 수정은 이후 새로 시작하는 27~32번에도 적용되며, 실행 중인 프로세스는 변경하지 않았다.
- 관련 CPU 테스트 **23개 통과**. GPU 6에서 2048환경·1 iteration scratch 학습과 checkpoint 저장 완료. 진단 CSV의 AT 목표 높이는 0보다 컸고, TensorBoard scalar **177개 모두 finite**, 공유 주 물체 **0**이었다. 이는 AT 목표 배치와 실행 경로 검증이며 정책 수렴 검증은 아니다.

## 2026-09-25

### 과거 실험 진입점 정리

- 실행 중인 27·28·30·32번과 Stage 2 29번, 비교용 31번 및 원본 1번을 남겼다. Git에 있던 9~26번 config 18개·전용 실행 스크립트 50개·전용 테스트 11개·오래된 설계 문서 10개·미사용 graph 예제 2개·미사용 학습 smoke config 1개를 제거했다. 현재 문서를 다시 작성하고 25·26번 전용 sampler 분기를 제거했다. 공유 graph·reward 함수는 현재 실험도 사용하므로 유지했다.
- 기존 학습 프로세스와 output은 변경하지 않았다. 삭제된 config·스크립트는 Git 이력에서 확인할 수 있다.
- 남긴 테스트 전체 **109개 통과**, 실행 스크립트 문법·Markdown 로컬 링크·diff 확인. 새 코드로 32번 2048환경·1 iteration 확인에서 checkpoint 저장, 물리 reset 실패 **0**·공유 주 물체 **0**, scalar **177개 모두 finite**였다. 이는 실행 경로 검증이며 장기 수렴 검증은 아니다.

### 32번 보상식 유지 커리큘럼

- 31번의 가까운 AT/ON_TOP 시작·후반 CLIMB RSI·과제/RSI 비율을 유지하고, AT/ON_TOP progress와 CLIMB state만 30번 식으로 되돌린 별도 schema 9 variant와 train/test/VNC를 추가했다. 31번 설정과 checkpoint는 그대로 두며 32번은 scratch로 시작한다.
- 관련 CPU 테스트 **17개 통과**. 2048환경·`MAX_ITERATIONS=1` GPU 1 확인에서 checkpoint 저장, 물리 reset 실패 **0**·재시도 **135회**, scalar **177개 모두 finite**, 템플릿 비율 약 **0.095/0.105/0.298/0.255/0.247**을 확인했다. 저장 checkpoint의 1환경·20-step ON_TOP headless 평가와 JSON 생성도 완료했다. 이는 실행 경로 검증이며 정책 수렴 검증은 아니다.

### 30번 SIT plane 정규화·31번 hard-skill 시작 상태 실험

- 28번을 보존하고 새 schema 9 variant 두 개를 분리했다. 30번은 SIT 성공을 root XY의 상자 윗면 안쪽 10%·목표 Z ±7cm로 바꾸고 agent별 active-edge 보상을 평균내 단일/두 edge 최대 task reward를 0.6으로 맞춘다. 31번은 AT·ON_TOP 가까운 시작 비율을 환경 step 0→300,000에서 80→30%로 낮추고, 장거리 progress·CLIMB root/발 높이 state·hard-skill sampling을 추가한다. 두 실험 모두 scratch이며 팀 공유 0.9/0.1은 유지한다.
- 기존 plane CSV 표본에서 AT/ON_TOP은 에피소드 첫 관측 3.33/4.27m에서 마지막 관측 5.70/6.36m로 목표에서 멀어졌다. CLIMB reference 7개·505프레임을 상자 높이 0.4m·XY 0.5m로 실측한 결과 성공 가능 29프레임 전부 원본 RSI 제외 구간에 있었다. 31번은 CLIMB RSI의 70%를 후반 65~98% 구간에 배정하며 기존 데이터셋과 27·28번 학습을 변경하지 않는다.
- 관련 CPU **34개 통과**, 스크립트 문법·diff 확인. GPU 1에서 두 실험 각각 2048환경·1 iteration scratch와 checkpoint 저장, 저장 checkpoint의 1환경·32-step headless 평가를 완료했다. 30번 물리 리셋 실패 0·재시도 121회, 31번 실패 0·재시도 156회, 각 scalar 177개 finite였다. 31번 CLIMB root/발 조건 동시 통과율은 후반 RSI 이전 짧은 확인 0%에서 1.26%로 바뀌었다. 이는 초기 상태·실행 경로 확인이며 장기 정책 수렴 검증은 아니다.

### 29번 Stage 2 coordination 전이 실험

- 28번 plane 성공 조건과 사용자 지정 전역 팀 보상 0.9/0.1을 유지하고, `place_climb/place_sit/place_stack` 협력 90%와 독립 bundle 10%를 샘플하는 schema 10 config·train/test/VNC를 추가했다. Stage 1 schema 9 semantic `.pth`를 `STAGE1_CHECKPOINT`로 선택해 actor/critic/AMP weight와 actor·AMP 관측 통계를 엄격히 복사하고, actor encoder·관측 RMS를 고정했다. 새 grounded-edge cross-attention과 Stage 1 head의 `[W,0]` 확장으로 협력 입력을 연결했다. Stage 2 resume는 별도 `RESUME_CHECKPOINT`로 분리했다.
- 공유 물체 SIT/CLIMB agent는 loco에서 시작하고, downstream 성공 또는 전체 graph 성공으로 시작한 reset은 재시도한다. 3인 공유 물체·4인 독립 쌍의 explicit 평가 graph와 가변 agent/edge 입력을 추가했다. Stage 2 taxonomy의 CLIMB 누락과 평가 시 학습 graph 동일성 검사를 수정했다.
- CPU 전체 **278개 통과**, 2048환경·1 iteration GPU 학습 및 checkpoint 저장 완료. 저장 checkpoint의 첫 head 협력 입력 weight norm은 **0.0556**, TensorBoard scalar **162개 모두 finite**, 물리 reset 실패 **0**이었다. 실제 Stage 1 plane checkpoint에서 전이 직후 동일 입력의 action mean·sigma 최대 차이는 각각 **0**이었다. 같은 Stage 2 checkpoint로 2·3·4인 1환경·20-step headless 평가와 JSON 저장을 완료했다. 2048환경 Stage 2 resume 뒤 협력 입력 weight norm은 **0.0817**로 변했고 attention weight도 갱신됐으며 actor encoder 75개 tensor와 actor RMS는 동일했다. Resume scalar **174개 모두 finite**, 물리 reset 실패 **0**이었다. 이는 실행·가변 입력 경로 검증이며 협력 성공률·장기 수렴 검증은 아니다.

## 2026-09-24

### 28번 stage1_plane 독립 시나리오 실험

- 27번의 독립 물체·goal 배정, 다섯 행동 각 0.2, RSI·AMP와 팀 보상 0.9/0.1을 유지했다. CLIMB·ON_TOP의 current-success만 원격 `ontop_climb_plane` 방식으로 변경했다. ON_TOP은 source 중심 XY의 support 윗면 안쪽 10% 영역 및 면 높이 오차 1mm, CLIMB은 root XY의 안쪽 5% 영역 및 root 높이 오차 20cm·평균 발 높이 오차 7cm를 요구한다. 연속 state·progress 보상은 기존 중심점 기준이며, 새 reward variant·config·train/test/VNC·output과 checkpoint 계약을 분리했다.
- CPU 전체 **274개 통과**, shell 문법·diff 공백 확인. GPU 3·2048환경·1 iteration scratch에서 다섯 template 비율 **0.194~0.205**, shared primary object **0**, 물리 reset 실패 **0**, TensorBoard scalar **174개 모두 finite**, checkpoint 저장을 확인했다. 저장 checkpoint로 1환경·32-step ON_TOP headless 평가와 JSON 저장도 종료했다. 이는 실행 경로 확인이며 장기 학습 수렴 검증은 아니다.
- GPU 3 본학습 로그 점검에서 CSV의 `self_task_contribution`/`other_task_contribution`이 Stage 1 보상을 자기 100%로 잘못 표시하는 문제를 발견해 config의 0.9/0.1 비율을 따르도록 수정했다. 실제 학습 reward와 TensorBoard 공유 지표는 이미 0.9/0.1로 정상이며, 실행 중 프로세스는 재시작하지 않아 해당 프로세스의 CSV 표시는 이전 방식으로 남는다. 관련 CPU 테스트 **17개 통과**·diff 공백 확인.

### 27번 독립 물체 binding + standalone CLIMB scratch 실험

- 26번의 5-template reward·0.9/0.1 팀 보상·AMP·template RSI를 유지하고, 물체 3개를 reset마다 무작위 순열로 두 agent의 서로 다른 primary object와 남은 ON_TOP support에 배정한다. AT goal도 비공유 무작위 배정하며 동시 ON_TOP은 제외한다. 유효한 template pair의 각 agent 주변 확률은 기존 0.2를 유지한다. 새 sampler/config/train/test/VNC/output과 별도 checkpoint variant를 연결했다.
- CPU 전체 **270개 통과**. GPU 2·2048환경·1 iteration 확인에서 5-template 비율 **0.196~0.203**, shared primary object **0**, 물리 reset 실패 **0**, TensorBoard scalar **172개 모두 finite**, GTA position p95 **4.4012m**, checkpoint 저장을 확인했다. 저장 checkpoint로 1환경·32-step ON_TOP headless 평가와 JSON 저장도 종료했다. 이는 실행 경로 검증이며 장기 수렴·물체 수 확대 추론은 미검증이다.
