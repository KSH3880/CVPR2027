# Warehouse 5명 · 2 PUSH + 독립 Carry AT 3명 — 실내 안쪽 상자 상세 명세

작성일: 2026-11-15. **상태: 기존 규약에서 운반 source3개를 실내로 더 이동하는 신규 평가 명세. 이 배치의 동적 성공 장면은 아직 탐색·검증하지 않았다.** 본 문서는 다른 파일 없이 LLM에 전달하도록 공통 규약을 모두 포함한다. repo/가중치/asset 데이터는 별도로 필요하다.

**명시적 설계 가정:** ‘조금 더 방 안쪽’을 입구의 실내 법선 +X로 0.75m 이동으로 수치화한다. `depth_offset_m=0.75`를 기본으로 두고 0.50/1.00m 등은 기록된 별도 profile로 조절할 수 있다. 세 상자 모두 같은 +X 이동을 적용한다. agent spawn·목표·크기·질량·PUSH 경로·독립 AT graph는 유지한다. 이 문서를 작성하며 새 탐색이나 학습은 실행하지 않았다.

## 1. 목적과 과제 의미

평가 전용 Warehouse 환경에서 5명의 자연스러운 전체 성공 장면을 최소 5개 확보한다. 두 명이 입구를 막은 큰 상자 하나를 공동으로 밀고, 나머지 세 명은 각자 방 안의 자기 상자를 들고 밖으로 운반한다. 세 Carry AT 중 둘은 같은 XY에 받침과 상단 상자를 순서대로 놓아 실제로 적재한다. 나머지 한 명은 별도 위치에 놓는다.

**독립 AT는 관계 그래프가 독립이라는 뜻이다.** 상단 운반자가 늦게 도착하도록 reset 거리와 사후 선별로 시간차를 만든다. BEFORE/enable, ON_TOP, SIT, CLIMB 과제를 추가하거나 가상의 대기 상자로 행동을 막지 않는다. 선행/후행이라는 레거시 변수명은 각각 받침/상단 운반자를 가리킬 뿐이다.

| Agent | 역할 | Native template | 자기 물체 | 실제 목표 |
|---|---|---:|---|---|
| A0 | 상단 상자 운반, 이하 payload | AT=3 | 0.30m 정육면체 | 받침과 같은 XY, 중심 Z=0.700m |
| A1 | 공동 pusher | PUSH=5 | A4와 공유하는 입구 상자 | 공동 PUSH 구역 목표 |
| A2 | 별도 Carry AT, 이하 solo | AT=3 | 0.40m 정육면체 | 별도의 실외 목표 |
| A3 | 받침 상자 운반, 이하 base | AT=3 | 0.45m 정육면체 | 중심 Z=0.325m인 실외 목표 |
| A4 | 공동 pusher | PUSH=5 | A1과 같은 입구 상자 | 공동 PUSH 구역 목표 |

실제 그래프 배열은 다음과 같다. 역할 ID는 논리 task ID와 별개다.

```json
{
  "num_agents": 5,
  "num_objects": 10,
  "templates": [3, 5, 3, 3, 5],
  "predecessor": [-1, -1, -1, -1, -1],
  "groups": [0, 1, 2, 3, 1],
  "roles": [1, 4, 3, 0, 2],
  "assigned_physical_boxes": [0, 39, 2, 3, 39],
  "visible_physical_boxes": [0, 2, 3, 39]
}
```

`num_objects=10`은 관측용 논리 object token 수다. 저장된 `boxes`의 실제 physical actor 수는 41개다. 토큰과 physical slot을 동일한 인덱스로 취급하지 않는다. 이 slot 번호는 해당 구현의 재현값이며, 다른 구현에서는 역할과 공유 관계를 보존한 뒤 새 번호를 명시한다.

## 2. 좌표계와 방·입구 기하

아래 좌표는 모두 저장된 `scene.json`/`initial.npz`와 같은 **방 중심 기준 warehouse-local 좌표 W**다. 오른손 좌표계, Z 위, 단위 m/rad, quaternion은 XYZW다. +X는 방 안쪽, 방 밖에서 입구를 바라볼 때 +Y는 왼쪽이다.

| 항목 | 중심 또는 범위 W | 크기/설명 |
|---|---|---|
| 방 내부 | X∈[-3,3], Y∈[-4,4] | 내부 깊이 6m, 폭 8m |
| 바닥 | 중심 (-4.5,0,0.05) | 15×17×0.10m, 윗면 Z=0.10m |
| 바닥 범위 | X∈[-12,3], Y∈[-8.5,8.5] | 좌우 PUSH 여유를 위해 평가에서 확장 |
| 뒤벽 | 중심 (3.2,0,1.35) | 0.4×8.8×2.5m |
| 양옆 벽 | 중심 (0,±4.2,1.35) | 각각 6×0.4×2.5m |
| 앞벽 두 조각 | 중심 (-3.2,±2.68,1.35) | 각각 0.4×3.44×2.5m |
| 앞벽 두께 구간 | X∈[-3.4,-3.0] | 안쪽 면 -3.0, 바깥 면 -3.4 |
| 출입구 | 앞벽의 Y∈[-0.96,0.96] | 폭 약 1.92m |

저장 좌표 W와 실제 GPU 병렬 환경의 world 좌표를 섞지 않는다.

```text
world_position = Rz(original_world_yaw) @ W_position + original_world_origin
W_position     = Rz(-original_world_yaw) @ (world_position - original_world_origin)
```

기존 `stack_layout()`의 sampler 좌표 L은 **입구 안쪽 면 X=0**이다. 따라서 `L.x=W.x+3.0`, `L.y=W.y`. profile JSON의 숫자는 W로 적지만 helper가 `FRONT_SHIFT=3.0`을 더해 L로 변환한다. 다른 머신에서 이미 L로 바꾼 좌표를 profile에 다시 넣어 3m를 중복 이동하지 않는다. 위치뿐 아니라 body/box quaternion과 선·각속도도 같은 회전으로 변환한다.

## 3. 물체 크기·질량·목표 높이

| 물체 | Physical slot | 크기 XYZ(m) | 밀도 kg/m³ | 질량 kg | 바닥에 놓인 중심 Z |
|---|---:|---|---:|---:|---:|
| payload | 0 | [0.30,0.30,0.30] | 100 | 2.7 | 0.250 |
| solo | 2 | [0.40,0.40,0.40] | 100 | 6.4 | 0.300 |
| base | 3 | [0.45,0.45,0.45] | 100 | 9.1125 | 0.325 |
| 입구 PUSH 상자 | 39 | [3.60,2.00,1.50] | 3 | 32.4 | 0.850 |

운반 상자 3개는 서로 다른 실제 물체이며 모두 처음부터 실내 바닥 위에 있다. 초기 box yaw는 0이다. native 재질·마찰·구동기·관측·센서·충돌·접촉 설정은 원래 repo/config를 유지한다. 상자를 가볍게 하거나 마찰을 없애 성공을 만들지 않는다.

**기존 BEFORE 초기 배치를 재사용할 때 중요한 변환:** 원본 reference의 A3 assigned slot은 33이며 크기는 0.45³다. 독립 AT로 바꾸면 A3 slot이 3이 되므로 자산 생성 전에 `SIZES[3]=SIZES[33]`을 수행하고 밀도에 맞춰 질량을 계산해야 한다. 그래프만 바꾸면 slot3의 원래 0.30³ 상자가 남을 수 있다. source profile의 키는 physical slot이 아니라 **agent ID**다. 기본 source 좌표도 원본 `ASSIGNED[3]=33`에서 읽어 새 slot3에 배치한다.

실외 목표는 다음을 유지한다.

```text
G_base    = [-6.000000,  1.000000, 0.325]
G_payload = [-6.000000,  1.000000, 0.700]
G_solo    = [-4.983032, -0.414795, 0.300]
```

상단 목표 Z는 상자 중심의 높이다. `Z_payload=Z_base+(H_base+H_payload)/2`이므로 0.325+(0.45+0.30)/2=0.700m다. 두 상자의 목표 XY에는 **같은 jitter 결과를 공유**한다. 각자 XY jitter를 독립 적용하지 않는다. 상단 목표를 바닥 높이로 바꾸거나 goal Z를 무시하는 성공 판정은 허용하지 않는다.

## 4. 공동 PUSH 규약

PUSH 대상은 큰 상자 **하나**다. A1/A4가 두 구역을 담당한다.

```text
초기 상자 중심 = [-5.239990, 0.000000, 0.850]
최종 상자 중심 = [-5.239990, 5.940430, 0.850]
이동 방향      = +Y / left
초기 box yaw   = 0 rad
pusher heading = +π/2 rad
```

입구를 조금 여는 것으로 끝내지 않고 약 5.94m 밀어 다른 세 명의 진입·반출 동선에서 벗어나게 한다. 좌우 PUSH는 바닥 경계까지 갈 필요가 없으며 이 과제에서는 약 7m 이내의 여유 있는 이동을 허용한다. 바닥은 위 표처럼 확장되어 있다. 현재 재현 profile은 +Y 방향이며, -Y 반전 탐색은 별도 profile로 기록한다.

pusher 명목 root는 X≈-6.43994/-4.64014, Y≈-1.79980이다. native region 배정에 따라 A1/A4의 좌우 역할이 바뀔 수 있다. 개인 구역 목표는 공동 중심 목표에서 X±0.9m인 token 목표다. 두 pusher의 `_tar_pos`를 큰 상자 중심의 목표와 혼동하지 않는다.

다른 세 명을 PUSH 진행 영역·상자에 치여 밀려나는 위치에 spawn시키지 않는다. PUSH 상자 뒤의 추진/작업 공간과 앞으로 밀리는 swept 영역을 확인하되, 단순히 방 외부라는 이유만으로 안전하다고 간주하지 않는다.

## 5. 안쪽 source 변형의 정확한 정의

root 위치는 기존과 같다.

| Agent | 명목 root XY W |
|---|---|
| A0 payload | [-5.800000,-4.200000] |
| A1 pusher | [-6.439941,-1.799805] 또는 반대 구역 |
| A2 solo | [-5.036865,-2.489746] |
| A3 base | [-3.900000,-2.600000] |
| A4 pusher | [-4.640137,-1.799805] 또는 반대 구역 |

세 운반 상자의 중심만 reset에서 다음처럼 이동한다. 입구 상자 slot39, root 위치, 목표, 방/바닥은 이동하지 않는다.

```text
source_deeper[a] = source_baseline[a] + [depth_offset_m,0,0]
a ∈ {0,2,3}; default depth_offset_m=0.75
```

| Agent | 기존 source XYZ W | 새 source XYZ W | 안쪽 면 X=-3에서의 새 중심 깊이 |
|---|---|---|---:|
| A0 payload | [-2.200000,-1.000000,0.250] | [-1.450000,-1.000000,0.250] | 1.550m |
| A2 solo | [-1.625488,1.026855,0.300] | [-0.875488,1.026855,0.300] | 2.125m |
| A3 base | [-2.600000,0.500000,0.325] | [-1.850000,0.500000,0.325] | 1.150m |

**적용 위치는 agent source profile**이다. `reference.INITIAL['boxes'][3]`만 이동하면 기존 base source가 slot33에서 읽히므로 원하는 변경이 적용되지 않는다. source dict의 `"0"`, `"2"`, `"3"` 값을 수정하고 `stack_layout`이 실제 새 assigned slot에 배치하도록 한다. 이미 +0.75m인 위 숫자에 offset을 다시 더하지 않는다.

완전 명시한 기본 profile은 다음과 같다. A-wide는 root_jitter만 0.08로 바꾼다. 실제 helper에서 쓰는 source dict 키는 agent ID다.

```json
{
  "root": {
    "0": [
      -5.8,
      -4.2
    ],
    "1": [
      -6.439941,
      -1.799805
    ],
    "2": [
      -5.036865,
      -2.489746
    ],
    "3": [
      -3.9,
      -2.6
    ],
    "4": [
      -4.640137,
      -1.799805
    ]
  },
  "source": {
    "0": [
      -1.45,
      -1.0
    ],
    "2": [
      -0.875488,
      1.026855
    ],
    "3": [
      -1.85,
      0.5
    ]
  },
  "goal": {
    "2": [
      -4.983032,
      -0.414795
    ],
    "3": [
      -6.0,
      1.0
    ]
  },
  "landing": [
    -5.23999,
    5.94043
  ],
  "source_jitter": 0.02,
  "root_jitter": 0.025,
  "goal_jitter": 0.025
}
```

source ±0.02m와 goal ±0.025m, root A ±0.025m/A-wide ±0.08m jitter는 기존과 같다. 세 source의 명목 +X 이동량은 같으며 각 source jitter는 기존대로 독립이다. 목표 XY/Z와 목표 공유는 바꾸지 않는다. source 변경에 따라 native heading을 다시 계산한다. 명목 yaw는 A0≈36.34°, A3≈56.52°, A2≈40.20°다. pose/속도0 규약은 유지한다.

### 배치 가능성과 아직 확인하지 않은 부분

| Agent | 기존→새 사람-source XY 거리 | 기존→새 source-goal XY 거리 |
|---|---:|---:|
| A0 | 4.817→5.400m | 4.294→4.970m |
| A3 | 3.362→3.717m | 3.437→4.180m |
| A2 | 4.899→5.448m | 3.654→4.353m |

명목 새 배치는 실내 범위와 reset <9.8m 조건을 만족하고, 세 source 사이의 상대 간격은 공통 이동이라 유지된다. 이는 **정적 계산**이며 정책이 집기/반출/적재까지 성공한다는 검증이 아니다. 특히 A0/A2 접근 거리는 이전 학습 sampler의 약5m 범위를 넘는다. 이 독립 AT 평가 reset은 <9.8m를 사용했지만 정책 성공률은 낮아질 수 있다.

base source에서 기존 목표로 잇는 명목 직선은 문 바깥 면에서 Y≈0.687m로 지나간다. 0.45m 축정렬 상자의 옆 여유가 기존 약11.7cm에서 약4.8cm로 줄고, 상자가 45° 회전하면 이 직선은 벽과 겹칠 수 있다. 따라서 직선 이동 성공을 가정하지 말고 실제 box OBB·yaw와 벽 간섭을 검토한다. 에이전트가 문 중앙을 거쳐 나올 수 있는지는 rollout으로 판단한다.

초기 실패가 많으면 먼저 reset 유효성·거리·문 통과를 분류한 뒤 offset=0.50m부터 비교한다. offset별 결과/seed/좌표를 기록하고 서로 다른 offset 후보를 같은 원본 조건의 성과라고 합치지 않는다. source Y를 미세 조정하는 후속 profile도 가능하지만 **기본안은 Y/root/goal 변경 없이 +X만** 적용하며 변경 profile을 별도로 표시한다. 벽을 끄거나 상자를 순간이동해 통과시키지 않는다.

## 6. 정지·완전 닫힘 시작과 reset 유효성

1. 다섯 명 전원의 **전신**이 방 밖에서 시작한다. root만 검사하지 않는다. MJCF capsule endpoints/반경과 발 형상을 사용해 앞벽 바깥 면 X=-3.4보다 신체 최대 X가 최소 0.10m 더 바깥인지 검사한다.
2. 입구 상자는 완전히 닫힘이다. 중간 진행 RSI, 이미 걷는 자세에서 시작하는 모션 crop, 이미 밀린 상자부터 시작하는 장면을 섞지 않는다.
3. root 선속도·각속도·DOF 속도 성분은 초기 절댓값 1e-6 이하이며 원래 기록은 0이다. 모든 상자도 정지 상태로 reset한다.
4. 자산 크기/높이와 발 바닥 정합을 유지한 정지 pose로 시작한다. 정확한 재현은 reference의 `dof_pos`/root tilt/Z를 사용한다. native standing fallback은 유사 조건이며 사용 여부를 기록한다.
5. 초기 root pair 최소 XY 간격은 >0.75m다. 이는 빠른 reset 조건일 뿐 전신 충돌을 보장하지 않으므로 전신 검사를 따로 한다.
6. sampler L에서 모든 root X<-0.43이고, 세 상자는 각 반쪽 크기와 최소 0.02m margin을 포함해 실내 X∈(0,6), Y∈(-4,4)에 있어야 한다.
7. 세 Carry AT 실제 목표 중심은 앞벽 바깥 면보다 0.20m 이상 밖, 즉 W에서 goal X<-3.6이어야 한다.
8. reset 시 사람→자기 목표, 세 운반자의 사람→자기 상자, 자기 상자→목표 XY 거리는 각각 <9.8m다. 수행 중 10m 제한을 위한 여유다. payload 초기 사람→자기 목표 XY 거리는 추가로 ≥4m다.
9. 배치 실패는 재샘플링한다. 고정 profile이 성립하지 않는데 tries만 무한히 늘리지 말고 실패 원인을 계수한다.
10. 처음 reset 직후 Isaac Gym의 refreshed body tensor는 stale일 수 있다. step0 전신 검사·기록은 reset에 배정한 kinematic body state 또는 일치 검증한 FK를 사용한다. 실제 시뮬레이션을 여러 step 돌린 뒤 그것을 초기 배치라고 저장하지 않는다.

**시작 프레임은 물리 step0**이다. `initial.npz`와 `motion.npz` 첫 frame의 root/DOF/body/box를 대조한다. preview의 첫 frame도 이 frame이어야 한다.

## 7. 체크포인트·추론 방식·관측 보존

기준 체크포인트는 다음과 같다. 다른 체크포인트로 바꾸면 새 실험으로 경로·epoch·SHA를 기록한다.

```text
repo-relative checkpoint:
output/stage2_sensor_900_room_warehouse/Stage2Sensor900RoomWarehouse_12-16-35-04/nn/Stage2Sensor900RoomWarehouse_00010000.pth

epoch: 10000
SHA256: f7a24bee523ad0696b755ad65af6f8e2ae0b0ac615d1bbc94a679704dcda67d2
```

- 원래 가중치와 observation RMS를 복원하고 **deterministic mean action**을 사용한다. 기존 호출은 `get_action(obs, True)`다. 학습 시 action sampling/noise, 수동 steering, root 이동, pose 보정, 접촉 후 재배치, 강제 grasp, 선행 완료 flag 조작을 추가하지 않는다.
- native square sensor가 실제 벽·운반 상자·큰 PUSH 상자를 반영하게 한다. 다른 사람/물체 attention mask나 task-slot 유효성을 임의로 줄이지 않는다.
- 학습은 4명/8token이었지만 여기서는 **실제 5명/10token tensors**로 평가했다. 검증 함수의 4/8 제한만 평가 wrapper에서 처리했으며, 4명 행동을 복제해 5명처럼 보여준 방식이 아니다. 모델/센서의 실제 인원·관측 shape와 finite action을 확인한다.
- PyTorch/Isaac Gym/물리 옵션·asset가 달라지면 동일 seed도 bitwise 동일 rollout을 보장하지 않는다. 기하·과제·정책·선별 규약의 재현과 원본 trajectory replay를 구분한다.
- 원래 runtime은 GPU physics/GPU policy, TF32 비활성, Isaac Gym을 torch보다 먼저 import한다. `runtime_env.sh`를 사용한다. 이 머신의 실사용 conda는 `tokenhsi_sm120`이었으며 다른 머신의 환경명/절대 경로는 현지 환경에 맞춘다.
- 기록 주기는 약 30Hz(`dt≈0.033333335s`)이고 `controlFrequencyInv=2`, `sim.substeps=2`인 기존 환경을 유지한다. config·checkpoint 부속 `relation_config.yaml`·실제 task.dt를 대조한다.

## 8. 탐색 런타임과 안전한 최적화

원래 채택본을 낸 주 profile은 A와 A-wide다. 둘 다 worker당 1024env·최대 32batch·최대 8자동후보·평가 episode 1200control step(약40초)이었다. seed는 각각 14611/14612다. **본학습의 900step 설정을 수정한 것이 아니라 평가 길이 override다.** A2 등 추가 profile도 시험했지만 최종 5편에는 A/A-wide만 포함된다.

다른 GPU에서는 적절한 env 수를 benchmark해도 되지만 5명/10token·reset 분포·성공 조건을 조용히 바꾸지 않는다. env 수를 바꾸면 RNG 소비도 바뀌므로 별도 실행으로 기록한다. max_saved는 후보 저장 상한이지 최종 확정 장면 수가 아니다.

- critic 생략 및 이미 끝난 env의 actor 생략은 **살아 있는 env의 원래 평균 행동과 max abs 오차≤1e-5**를 확인한 경우에만 사용한다. 기존 구현은 초기 및 rollout 150/300step에서 비교한다. 불일치하면 원래 inference로 돌아간다.
- actor 생략은 env 전체가 탐색 종료된 경우에만 적용한다. 한 agent가 과제를 끝냈다고 그 agent의 행동을 freeze하지 않는다. 완료한 사람도 자연스럽게 계속 움직이며 타인 간섭 검토 대상이다.
- 배치 크기와 worker 수는 env·physics-step/s 및 최종 후보당 시간을 실제로 측정해 정한다. GPU 메모리 여유만 보고 worker/env를 무조건 늘리지 않는다.
- GPU 번호는 `TOKENHSI_GPU`로 한 번 지정한다. 내부 `cuda:0`은 runtime이 선택한 visible GPU의 번호일 수 있다.
- 기존 학습·공유 MPS를 임의로 종료하지 않는다. 정말 대상 client를 종료해야 할 때만 service/PID/cwd/config/MPS mapping을 확인한 후 해당 server에 `terminate_client <server_pid> <client_pid>` 성공 응답 0을 받고 그 service만 종료한다. 이전 머신의 PID를 재사용하지 않는다.

## 9. rollout의 성공·거리·종료 규약

다음 상태는 서로 다르다.

| 상태 | 실제 의미 |
|---|---|
| `success[t,a]` / native done | 해당 frame에서 agent의 required goal edge가 현재 성공 |
| `streak[a]` | native success가 연속인 frame 수 |
| `completed[a]` | 6연속 native 성공을 한 번 달성한 뒤 유지하는 탐색용 latch |
| 최종 채택 | latch 이력 외에 마지막 동시 성공·물리 적재·동선·영상 검토까지 통과 |

이 모드의 Carry AT native 판정은 해당 구현/config에서 대략 **상자 중심의 3D 목표 오차≤0.20m이고 Z 오차≤0.001m**다. native 경로를 사용하며 아래 사후 필터 XY0.30/Z0.08로 대체하지 않는다. required goal만의 native 성공은 agent가 실제로 집어서 운반했다는 이력 증명이 아니다. 실제 집기·운반은 별도로 확인한다.

두 pusher도 native PUSH/개인 구역 AT 경로를 그대로 사용한다. 최종 목표에서 goal saturation으로 성공이 유지될 수 있으므로 손을 뗀 순간을 무조건 실패로 보지 않는다. 반대로 contact 지표만으로 이동 완료를 선언하지 않는다.

원래 search는 모든 agent가 각각 6연속 frame(약0.2초) 성공하면 완료를 latch하고, 전원 latch 이후 20step(약0.67초)을 더 기록한다. native termination 또는 길이 제한 등으로 더 일찍 끝날 수 있다. 성공 flag가 잠시 켜졌다는 이유만으로 마지막 장면을 보지 않고 확정하지 않는다.

매 control step마다 **미완료자**의 XY 사람→현재 자기 목표 및 사람→현재 자기 assigned box 중 하나라도 ≥10m이면 그 env를 종료한다. 이 모드는 전부 pre=-1이므로 future goal도 자기 goal과 같다. `completed`가 이미 latch된 사람만 10m 검사에서 제외한다. 이미 완료한 base가 멀리 이동하는 것은 허용하지만 낙상/충돌/다른 작업 방해까지 면제되는 것은 아니다.

`predecessor_at_success`라는 저장 필드가 있어도 이 모드에서는 **A3 base의 현재 native done**이다. BEFORE edge도 아니고 6frame 완료 latch도 아니다. 첫 true frame을 시간차 선별 기준 `t_base`로 사용한다.

## 10. base 완료 전 payload의 실제 접근 검사

단순 root 전진거리나 다른 사람에게 밀린 이동만으로 통과시키지 않는다. 기존 자동필터는 `pending_approach.measure(...,leader=3,follower=0)`를 직접 호출한다. **`REQUIRE_CONTINUOUS_PENDING_PICKUP=0`이어도 이 호출은 생략되지 않는다.**

첫 base native 성공 이전 구간을 대상으로 다음을 검사한다.

1. 사람과 자기 payload 상자 XY를 약0.2초 bucket으로 평균낸다.
2. bucket 간 이동속도≥0.2m/s, 이동 방향과 현재 자기 상자 방향의 cos≥0.6, 사람–상자 거리 증가≤0.03m인 interval을 walking으로 분류한다.
3. 속도≥0.2m/s인 interval의 방향 cos<0.6이거나, 사람–상자 거리가0.03m를 초과해 증가하면 즉시 window를 끊는다. 비보행 interval이 연속0.6초를 넘겨도 window를 끊는다. 3cm 이하의 작은 거리 변화나 저속 자세조정을 모두 즉시 탈락시키는 규칙은 아니다.
4. 한 유효 window에서 **root의 자기 상자 방향 투영 이동합≥1m**, **실제 사람–상자 거리 감소≥1m**, **root 순변위≥1m**, **walking interval의 시간 합≥1.5초**를 모두 요구한다. 1.5초 내내 무조건 보행만 해야 한다는 뜻은 아니다.
5. pending 마지막 평균 사람–자기 상자 거리가≤1m이면 이전 유효 window로 통과한다. 아직 >1m이면 유효 window 끝이 pending 끝의0.6초 이내까지 이어져야 한다.
6. 그 window의 발/다리 관절 변화를 기록한다. 자동필터는 이 값에 hard threshold를 걸지 않으므로 자력 보행 여부는 사람·상자 접촉 기하와 영상에서 재확인한다.

base 첫 성공이 실제로 관측되어야 하며 그 전 payload 보행 window가 너무 짧거나 없으면 탈락한다. **상단 운반자가 자기 상자를 일찍 집고 기다리는 장면은 허용한다.** spawn에서 가만히 서 있다가 base 완료 후 출발하는 장면은 제외한다.

추가 순서 조건:

- `t_payload_first_native_success > t_base_first_native_success`.
- payload 상자가 자기 실외 목표의 XY1m 안으로 처음 들어오는 시점은 `t_base`보다 앞서면 안 된다. 이 검사는 사람의 목표 접근 시점이 아니라 **상자**의 도착 시점이다.
- 초기 payload 사람–목표 거리는 ≥4m지만 rollout 동안 4m 반경 밖을 유지하는 조건은 **없다**. BEFORE 탐색에서 썼던 hard4m guard나 virtual wait를 복사하지 않는다.

## 11. 원래 자동 후보 필터의 나머지 hard 조건

아래는 실제 `independent_at_stack_quality.py`의 조건이다. 최종 시각 감사와 구분한다.

| 구분 | 자동 조건 |
|---|---|
| 관계 | predecessor에 0 이상 값이 하나라도 있으면 탈락 |
| 닫힘 | start_mode=closed, 초기 gate 중심 Y 오차≤0.005m |
| 초기 정지 | root 선/각속도·DOF 속도 각 성분 절댓값≤1e-6 |
| 초기 외부 | 전신 extent<앞벽 외측−0.10m |
| source/goal | 세 source 전면>앞벽 내측+0.02m, 세 goal X<외측−0.20m |
| PUSH 경로 | 전 frame 세 운반자의 root–gate 평면 L∞ 외곽 여유≥0.27m |
| 낙상 | 전 frame 전원 root Z≥바닥 윗면+0.45m |
| 심한 겹침 | 임의 root pair XY<0.28m가 연속0.19초 초과면 탈락 |
| 최종 세 AT | 자기 box 중심의 goal XY 오차≤0.30m, Z 오차≤0.08m |
| 최종 적재 XY | payload/base 중심 XY 간격≤0.20m |
| 최종 적재 높이 | 아래 정의한 상하 높이 오차≤0.08m |
| 최종 적재 속도 | 마지막0.5초 두 상자 선속도 norm 최대≤0.15m/s |

```text
stack_height_error = abs(
    (payload_center_z - payload_size_z/2)
  - (base_center_z    + base_size_z/2)
)

gate_local_xy = R_gate^-1 @ (worker_root - gate_center)
gate_gap_proxy = max(max(abs(gate_local_xy) - gate_half_xy, 0))
```

gate gap은 root 기준 **L∞ proxy**이며 정확한 전신 Euclidean gap이 아니다. 높이 오차도 sizeZ 기반 proxy라 회전 상자의 실제 상면 접촉을 증명하지 않는다. 이를 전신/물리 검토 대신 사용하지 않는다.

자동 순위 score는 `500 - 전원 root XY 총경로(m) - 15*stack_xy(m)`다. 회선 길이에 hard 상한은 없었고 과한 회선은 후순위로 처리했다. 자동필터는 마지막 native 전원 true가 아니어도 `requires_final_geometry_review`를 표시하고 후보를 저장할 수 있다. 따라서 **자동 후보 저장=최종 채택이 아니다.**

## 12. 최종 채택·수동/후속 기하 감사

각 candidate를 다음 기준으로 검토하고, 최소5개 충분히 다른 장면을 확정한다.

- 끝 frame에서 `success[-1]`가 5명 모두 true인지 확인한다. 원본 채택5개는 이를 만족한다. 이것은 원래 자동필터의 별도 hard reject가 아니라 최종 선별 기준이다.
- 각 Carry AT가 자기 상자를 실제로 집고 들어 운반했는지 확인한다. 다른 사람이 밀어 목표에 들어간 것만으로 AT 장면을 인정하지 않는다. native holding 상태를 추가 저장할 수 있으면 보조 증거로 쓰되, 기존 NPZ에 holding/contact force가 있었다고 가정하지 않는다.
- payload의 base AT 전 자기 상자 접근이 자발적인지 본다. 발/다리 관절·root/box 상대 운동과 사람/PUSH 상자 접촉 시점·간격을 함께 본다. 초기 충돌로 튀어나온 이동은 제외한다.
- 심한 몸통 밀기, 서로 엉켜 같은 곳에서 장기 정체, 넘어진 사람 끌기, 진행 경로를 지속 차단하는 장면은 제외한다. **사소한 스침·잠깐의 발/다리 접촉은 허용**한다. 임의의 모든 접촉=탈락 규칙을 새로 추가하지 않는다.
- 벽 표면 접근과 실제 벽 관통/충돌 정체를 구분한다. 실제 MJCF capsule/발 mesh와 회전 OBB를 사용해 몸통·팔다리·운반 중 상자의 벽 간격, 저속 정체 구간, 계속 벽을 누르는 장면을 본다. 얕고 짧은 스침만으로 좋은 후보를 모두 버리지 않는다.
- PUSH 상자가 세 운반자를 밀거나 진로를 크게 방해하지 않아야 한다. 완료한 pusher/base의 이후 이동도 남은 사람을 막지 않는지 확인한다.
- base와 payload 실제 회전 상면의 접촉/지지, payload COM의 지지 다각형 포함, 속도·시간 안정성과 상자 크기/높이를 확인한다. 단순히 허공의 높은 AT 목표에 도착한 frame을 적재로 인정하지 않는다. 과도한 overhang은 후순위지만 원래 임의의 지지 면적 비율 hard cutoff는 없었다.
- 불필요한 긴 대기·큰 회선은 후순위다. 사용자가 허용한 초기 접근 중의 정상 대기와, 충돌/실패 때문에 서 있는 정체를 구분한다. 실제 native 정책의 보행을 스크립트로 고치지 않는다.

`audit_motion.json`(자발 접근/집기/순서), `audit_body.json`(사람/벽/PUSH 간섭), `audit_geometry.json`(초기 기하/최종 목표/실제 지지)와 MP4 프레임 검토를 남긴다. geometry는 접촉력 측정과 다르며, 검토하지 않은 후보에 최종 승인 표시를 하지 않는다.

## 13. 샘플링 조절의 허용 범위

허용: reset의 root/source/실외 goal 미세 이동, jitter 폭/seed/agent 간 간격 조절, 목표를 보는 초기 heading, 평가 바닥의 충분한 여유, 여러 env 병렬 평가. 바꾸면 profile과 seed를 저장한다. 에이전트를 방 안에 놓거나, 운반 상자를 미리 밖에 꺼내 놓거나, 이미 열린 입구 상자부터 시작해 성공률을 높이지 않는다.

다음은 이 명세의 유지 조건이다: 5명/2PUSH/3AT, 공유 PUSH1개/운반3개, predecessor 전부−1, 원래 policy/센서/물리, 세 source 실내/전신 실외/목표 실외, 목표 높이와 적재 순서, 활성10m 제한, 자발 접근, 심한 간섭 제외, step0 원본 기록. 크기/질량/과제 종류/가중치 변경은 별도 실험이다.

회전·반전 배치가 필요하면 방과 입구·source·goal·PUSH 방향·heading을 일관되게 변환한다. 일부 source만 반전하거나 이미 변환한 world 좌표에 회전을 중복 적용하지 않는다. 원래 채택 profile A/A-wide는 left였다.

## 14. 다른 머신에서의 재현 절차와 의존 파일

문서만으로 가중치·모션 데이터가 생성되지는 않는다. 같은 repo snapshot, 체크포인트와 부속 config, humanoid MJCF/발 mesh, dataset YAML이 참조하는 실제 모션 데이터를 별도로 준비한다. 기존 symlink의 절대 target이 다른 머신에도 있는지 확인한다. 빠진 파일을 다른 모델/자산으로 조용히 대체하지 않는다.

기존 helper를 이식하려면 아래 파일과 reference를 같은 구조로 복사한다. root는 repo-relative 경로다.

```text
output_etc/check/warehouse_push2_before_ontop_at_10000_20261114/clear_pass/
  independent_at_stack_run.py
  independent_at_stack_reset.py
  independent_at_stack_quality.py
  reference_reset.py
  runtime.py
  search.py
  motion_export.py
  quality_filter.py
  pending_approach.py
  geometry.py
  preview.py
  render.py
  export_blend.py
  verify_export.py
  launch_independent_at_stack.sh
  independent_stack_profile_a.json
  independent_stack_profile_a_wide.json
  checkpoint.json
  manifest.json
  candidates/review_s_b0001_e936/scene.json
  candidates/review_s_b0001_e936/initial.npz
  assets/mjcf/phys_humanoid_v3.xml
  assets/smpl_foot_meshes/left_foot.obj
  assets/smpl_foot_meshes/right_foot.obj
```

추가로 repo의 `tokenhsi/data/cfg/train/rlg/stage2_sensor_900_room_warehouse.yaml`, 환경 config, `tokenhsi/data/dataset_stage1_jointpush.yaml`, `tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml` 및 해당 구현 모듈이 필요하다. 위 목록은 helper 진입점 의존 목록이지 repo 자체를 대체하는 독립 배포본이 아니다.

현재 공용 `manifest.json`은 이후 AT/SIT 탐색으로 갱신된 기록도 포함한다. **독립 AT 규약은 이 문서와 `independent_at_stack_*`/채택 scene을 기준으로 판단**하고, 공용 manifest의 현재 BEFORE SIT 항목을 복사하지 않는다. 체크포인트 SHA를 별도로 대조한다.

1. 새 eval 전용 작업/출력 폴더를 마련한다. `runtime.OUT`과 `motion_export.OUT`은 `--output_path`만으로 이동하지 않고 helper 파일의 위치에서 결정되므로 helper를 별도 폴더로 복사하거나 그 경로 결정을 명시적으로 맞춘다. 원래 채택 데이터를 덮어쓰지 않는다.
2. launcher의 예전 절대 `cd`, checkpoint/profile 경로 및 `search.py`/렌더 도구의 절대 Python·Blender·FFmpeg 경로를 새 머신에 맞춘다. 복사본의 경로만 조절하고 생산 학습 코드를 고치지 않는다.
3. 다음 핵심 환경값을 검토한다. `BEFORE_SIT_SEARCH=0`을 명시해 다른 탐색 shell의 상속 값을 제거한다. quality dispatch는 SIT 분기가 우선이다.

```bash
export INDEPENDENT_AT_STACK=1 BEFORE_SIT_SEARCH=0 FULL_START=1
export FOLLOWER_FACE_GOAL=0 REFERENCE_FLIP=0
export PENDING_MIN_GOAL_DISTANCE=0 REQUIRE_CONTINUOUS_PENDING_PICKUP=0
export LEADER_CUBE=1 CUBIC_PROPS=1 EXACT_CUBES=1 SMALL_CUBES=1
export SEARCH_ENVS=1024 SEARCH_BATCHES=32 MAX_SAVED=8 EPISODE_LENGTH=1200
# A: SEED=14611 / root_jitter=.025; A-wide: SEED=14612 / root_jitter=.08
```

`WAREHOUSE_START=partial`이 기존 launcher에 남아 있어도 `reference_reset`이 `FULL_START=1`로 실제 closed reset을 강제했다. 문자열 하나만 보고 partial 시작이라고 판단하지 말고 최종 `_warehouse_start_mode`, frame0 gate, whole-body/속도를 확인한다. `CHECK_INITIAL_POSE=0`은 한 보조 검사 옵션이며 전신 실외 검사 자체를 해제하는 값이 아니다.

4. 실행 경로는 `independent_at_stack_run.py → independent_at_stack_reset.py → reference_reset/runtime → search.py → tokenhsi/run.py`다. 주요 CLI는 다음을 보존한다.

```text
--task HumanoidMACoupledPush
--cfg_train tokenhsi/data/cfg/train/rlg/stage2_sensor_900_room_warehouse.yaml
--motion_file tokenhsi/data/dataset_stage1_jointpush.yaml
--num_agents 5 --num_objects 10 --num_envs 1024
--sim_device cuda:0 --pipeline gpu --rl_device cuda:0 --num_threads 4
--eval_repeats 1 --task_graph random_scenario --task_camera stack
--eval_skills loco --eval_skill_probs 1.0
--test --eval --headless --no_video --episode_length 1200
--coupled_assignment ordered --coupled_peers all --group_spawn_over4
```

checkpoint/output/profile/seed와 visible GPU는 현지 절대 경로로 연결한다. 기존 launcher를 경로 수정 없이 복사해 바로 실행 가능하다고 가정하지 않는다. `--no_video`는 native viewer 영상 비활성이며 CPU 후보 preview 생성과는 별개다.

5. 먼저 짧은 초기화/finite 검사와 reset 그림을 검토한다. 5명/10token/32DOF, active source slot, 닫힘·정지·전신 외부, 목표 Z, 관측/행동 finite를 확인한 뒤 본 탐색을 수행한다. 이 문서를 작성한 작업 자체에서는 새 물리 탐색을 실행하지 않았다.
6. persistent user service 등 세션과 독립적인 실행을 사용하고 명확한 log·status·STOP 경로를 기록한다. 기존 완료 STOP marker를 복사했으면 새 작업 marker만 구분한다. `STOP_INDEPENDENT_STACK`가 이미 있으면 원래 search는 새 batch를 시작하지 않는다.
7. candidate가 나오면 빠른 MP4를 우선 공유하고, 검토 통과한 것만 Blender/고비용 렌더로 보낸다. 저장 상한까지 나왔다고 모든 후보를 성공 시안으로 보고하지 않는다.

## 15. 원본 데이터·Blender·확인 영상 보존

각 최종 시안 폴더에 다음을 저장한다. HTML 모아보기만 주지 말고 실제 MP4/폴더 경로를 제공한다.

| 파일 | 내용 |
|---|---|
| `initial.npz` | 물리step0의 초기 root/DOF/body/상자/속도/성공 상태 |
| `motion.npz` | 초기 frame을 포함한 약30Hz 전체 구간의 원본 동작 |
| `scene.json` | 단위/좌표/quaternion 순서, DOF/body 이름, 목표 XYZ, 역할·할당, 물체 크기, 벽/바닥 transform, checkpoint/profile/seed 등 |
| `quality.json` | 자동 필터/거리/순서/최종 적재/순위 점수 |
| `audit_motion.json` 등 | 실제 집기/자력 접근·기하·사람/벽 간섭의 후속 검토 |
| `preview.mp4` | 목표와 역할·진행 상황이 표시된 빠른 확인 영상 |
| `warehouse.blend` | 전체 keyframe의 사람·상자·scene; 원본 pose와 비교한 재사용 파일 |
| `blend_verification.json`, `video.json` | 독립 재개방/비교와 전체 디코딩 확인 |
| `artifact_hashes.json` | 원본·scene·영상·Blender 무결성 SHA |

현재 export 주요 배열:

```text
root: (T,5,13) = position3 + quaternion4 + linear_velocity3 + angular_velocity3
dof_pos, dof_vel, actions: (T,5,32)
body_pos: (T,5,15,3); body_quat: (T,5,15,4)
boxes: (T,41,13)
success, entered: (T,5); opened: (T,)
predecessor_at_success: (T,) = base A3 현재 native success
```

초기 actions=0은 저장 placeholder이며 정책이 처음 실제로 출력한 action과 구분한다. body 순서는 `scene.json`의 이름 배열을 따른다. 기존 export가 모든 desired provenance 필드(예: seed/holding/contact force)를 직접 넣는 것은 아니므로 launch/profile/별도 실행 메타에서 보완한다. 없던 필드를 원본에 있었다고 간주하지 않는다.

preview는 1280×720 CPU capsule 확인 영상으로 약15fps였다. 실외 PUSH 목표 G0, base/payload 공통 XY의 G1, 별도 AT G2를 표시하고 두 AT 목표 Z=.325/.700도 표기한다. 이 모드에는 가상 대기 원이 없으므로 BEFORE4m 원이나 SIT 대기 원을 그리지 않는다. 영상 생성 성공 문구만 믿지 말고 FFmpeg 전체 decode 결과와 첫/끝 frame을 확인한다.

Blender는 전체 keyframe을 저장한 뒤 별도 재개방으로 원본 root/body/box 위치·quaternion과 비교한다. 저장 keyframe 전수 확인과 표본 frame pose 비교를 구분해 기록한다. 허용오차 예시는 위치1e-4m 이하이며 원본5개는 약3.1e-7m 이하였다. 관절을 새 IK로 보정하거나 raw trajectory를 smoothing한 파일을 원본이라고 표시하지 않는다. 정제본을 추가하면 원본과 별도로 명시한다.

## 16. 디스크·실패 후보·진행 보고

- 탈락 rollout은 기본적으로 RAM에서 폐기한다. 각 worker 진단은 대표1개를 덮어쓰는 등 개수를 제한한다. 자동 후보도 최종 미채택이면 작은 조건/감사 요약을 남기고 불필요한 NPZ/영상/Blender를 정리한다.
- 기존 채택5개/채택 원본 reference·checkpoint·공유 캐시는 보존한다. 원래 search는 디스크 여유30GiB 미만이면 새 batch를 중단했다. 이 값은 해당 머신의 운영 guard이며 대량 실패 데이터를 쌓아도 된다는 뜻은 아니다.
- 최종 승인 후보와 타깃이 아닌 파일·service/PID는 삭제/종료하지 않는다. raw 후보의 이름·profile·batch·env·seed·탈락 이유를 남긴 뒤 해당 후보만 정리한다.
- `시도 env 수 / 자동 후보 수 / 최종 검토 통과 수`를 따로 보고한다. 조건을 유리하게 고른 scene search이므로 채택5개를 전체 분포 성공률로 표현하지 않는다.
- 좋은 후보가 나오면 MP4 실경로와 남은 검토 항목을 즉시 제공한다. 동영상 파일의 존재·전체 decode·정확한 시작/목표 표시까지 확인한 상태인지 구분한다.

## 17. 실패 유형별 확인 순서

| 증상 | 먼저 확인할 것 |
|---|---|
| 사람이 이미 걷거나 상자가 열린 첫 frame | FULL_START 최종 적용, step0 포함 여부, stale body tensor, RSI/crop 혼입 |
| 받침이 작거나 적재 높이가 다름 | slot33→3 크기 복사, 질량, center Z 계산, 실제 assigned slot |
| payload가 spawn에서 계속 대기 | 잘못 남은 BEFORE/SIT flag·pre 배열, source/goal 상대 위치, 목표 거리와 자기 상자 접근 시간 |
| 초반 root는 움직이나 스스로 걷지 않음 | 최초 peer/gate 접촉, 발/다리 관절, root-source 거리 감소, 타인에 의한 source 이동 |
| 5명 성공 flag인데 물리 적재가 이상함 | native현재값과 latch 구분, 끝frame, 실제 회전 상면/COM/속도, goal Z |
| 벽·사람 싸움 | 초기 전신 간격, 입구의 실제 폭과 상자 yaw, 동시에 들어가는 동선, 완료자가 남은 사람을 막는지 |
| 뒤늦게10m로 잘림 | 실제 assigned box/goal인지, completed만 예외인지, 레거시 가상 목표를 검사하지 않는지 |
| candidate가 없고 reset만 반복 | 두 좌표계 +3 변환, root>.75 간격/전신10cm 여유, source 실내 여유, <9.8m, scene validation reason |
| 느리거나 OOM | env·worker 조합 실측, actor/critic 최적화 행동 동등성, PhysX+PyTorch 총VRAM, 렌더 중복 |

## 18. 근거와 재현 수준

확인 기준일은 2026-11-15다. 해당 repo의 실제 `independent_at_stack_reset.py`, `independent_at_stack_quality.py`, `pending_approach.py`, `search.py`, `motion_export.py`, `reference_reset.py`, `runtime.py`, A/A-wide profile 및 채택 초기 상태를 확인했다.

native success는 `tokenhsi/env/tasks/multi_agent/{edge_ontop_reward,edge_interaction_reward,edge_context_reward,before_reward,stage2_reward}.py`, PUSH의 개인 구역/goal gate는 `humanoid_ma_coupled_push.py` 및 native coupled PUSH 구현을 따른다. 학습 YAML 기본값만으로 이 평가의 closed/reset/길이/과제를 추정하지 않는다.

정확 재현에는 같은 소스·asset·reference·config·weights가 필요하다. 이 문서의 수치와 규약으로 새 helper를 구현하는 경우에는 먼저 정적/짧은 물리 검사를 통과시키고 유사 조건의 새 탐색으로 보고한다. 수치표의 소수점 반올림과 새 머신의 물리 차이 때문에 원본과 trajectory가 같아진다고 보장하지 않는다.

## 19. 검증된 것과 신규 작업의 경계

기존 source에서 확보한 비교 후보는 `ind_stack_a_wide_b0000_e916`, `ind_stack_a_b0000_e411`, `ind_stack_a_wide_b0000_e748`, `ind_stack_a_b0000_e800`, `ind_stack_a_b0000_e069`다. 경로는 `output_etc/check/warehouse_push2_before_ontop_at_10000_20261114/clear_pass/candidates/` 아래이며 069는 큰 운반 회선의 예비다.

**이 5개는 +0.75m 안쪽 배치의 성공 증거가 아니다.** 새 작업에서는 새 profile/tag/출력 폴더로 구분하고 최소5개를 다시 탐색·검토한다. 초기화/finite 검사만 성공한 상태, 자동 후보 저장 상태, 실제 최종 채택 상태를 구분해서 보고한다. 기존 raw/영상/Blender와 학습 프로세스는 보존한다.

## 20. reference 누락 시 pose 복원 참고

정확 재현은 위 reference NPZ와 native reset 함수를 사용한다. 파일을 옮기지 못할 경우 다음 수치는 원래 채택916의 DOF 정지 pose 참고값이다. 원래 MJCF/DOF 순서에서만 유효하며 root tilt·body FK·발 바닥 높이를 함께 맞춰야 한다. 유사 복원이라는 점을 기록한다. position/yaw는 위 profile과 native heading 규칙을 사용한다.

```json
{
  "dof_names": [
    "abdomen_x",
    "abdomen_y",
    "abdomen_z",
    "neck_x",
    "neck_y",
    "neck_z",
    "right_shoulder_x",
    "right_shoulder_y",
    "right_shoulder_z",
    "right_elbow",
    "left_shoulder_x",
    "left_shoulder_y",
    "left_shoulder_z",
    "left_elbow",
    "right_hip_x",
    "right_hip_y",
    "right_hip_z",
    "right_knee_x",
    "right_knee_y",
    "right_knee_z",
    "right_ankle_x",
    "right_ankle_y",
    "right_ankle_z",
    "left_hip_x",
    "left_hip_y",
    "left_hip_z",
    "left_knee_x",
    "left_knee_y",
    "left_knee_z",
    "left_ankle_x",
    "left_ankle_y",
    "left_ankle_z"
  ],
  "payload_A0_dof_pos_rad": [
    -0.01000507,
    0.10287612,
    -0.02922127,
    -0.0001054,
    0.04512224,
    -0.00925753,
    -0.27665487,
    0.08689304,
    0.16007796,
    -0.6947968,
    0.30384701,
    0.15324569,
    -0.11242739,
    -0.62782484,
    -0.21155883,
    0.21545391,
    -0.03849503,
    -0.0308322,
    0.00762166,
    0.13056785,
    0.26991278,
    -0.14948599,
    -0.02457136,
    0.09907724,
    0.2172505,
    0.05213093,
    0.01075125,
    -0.16131872,
    -0.17679015,
    -0.11534724,
    -0.07616252,
    0.15005438
  ],
  "A1_A2_A3_A4_dof_pos_rad": [
    0.01849028,
    0.14255312,
    -0.03387735,
    -0.00512938,
    0.00118534,
    -0.04367636,
    -0.29929823,
    0.03278892,
    0.29105762,
    -0.69131386,
    0.34381309,
    0.10308249,
    -0.3486923,
    -0.58085823,
    0.02041693,
    0.16678932,
    -0.00704005,
    -0.05372912,
    0.01414418,
    0.05641792,
    0.1547655,
    -0.14348657,
    -0.10431039,
    -0.0262986,
    0.20642664,
    0.00467496,
    0.06875803,
    -0.08421177,
    -0.19516928,
    -0.08274908,
    -0.1112383,
    0.16135305
  ],
  "root_z_m": [
    1.01513076,
    1.04934978,
    1.04934978,
    1.04934978,
    1.04934978
  ],
  "root_linear_velocity": [
    0,
    0,
    0
  ],
  "root_angular_velocity": [
    0,
    0,
    0
  ],
  "all_dof_velocity": 0
}
```
