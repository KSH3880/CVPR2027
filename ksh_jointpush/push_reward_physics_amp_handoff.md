# PUSH 보상·상자 물성·AMP 전달 명세

확인일: 2026-10-19. GPU6에서 확인한 이 저장소의 PUSH 학습(PID 2992970)을 기준으로 한다. 모델의 정책 입력과 네트워크 설계는 제외한다. 실행 중인 학습·데이터·설정은 변경하지 않았다.

## 1. 기준 설정과 파일

저장소 루트: `/home/hwanhee/koo_cvpr/stage1_fix`. 아래 코드·설정·데이터·출력 경로는 이 저장소 루트를 포함한 절대 경로로 표기한다.

| 항목 | 경로 / 값 |
| --- | --- |
| Task | `HumanoidMAOpenPushPoint` |
| 환경 YAML | `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/cfg/multi_agent/push_fixed130_point_zband_heightmatch_goal2to5.yaml` |
| 학습·AMP YAML | `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml` |
| AMP manifest | `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_push_only_push180.yaml` |
| 실행 wrapper | `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/scripts/multi_agent/push_fixed130_point_zband_heightmatch_goal2to5_train.sh` |
| 출력 | `/home/hwanhee/koo_cvpr/stage1_fix/output/push_fixed130_point_zband_heightmatch_goal2to5/` |
| 재개에 사용한 checkpoint | `/home/hwanhee/koo_cvpr/stage1_fix/output/push_fixed130_point_zband_heightmatch_goal2to5/resume/PushFixed130PointZBandHeightMatchGoal2To5_00000500.pth` |
| 학습 규모 | 2048환경, 환경당 사람2명·실제 PUSH 상자2개 |
| 에피소드 | 최대600 제어 step, 목표 달성만으로 종료하지 않음; 낙상 등 조기종료 활성 |
| 목표 | reset 시 초기 상자 중심에서 수평 반경2~5m 균등·방향360° 균등, 이후 고정 |

실행 인자의 `num_objects=4`는 내부 슬롯 계약이다. 현재 PUSH-only 경로는 문 actor를 생성하지 않으며 실제 PUSH 상자는2개다. wrapper의 GPU 기본값은2지만, 확인한 프로세스는 GPU6에서 실행 중이었다.

## 2. 상자와 바닥 물성

| 항목 | 설정 |
| --- | --- |
| 형상 | Isaac Gym `create_box`로 생성하는 정육면체 |
| X/Y/Z 길이 | 각각1.3m, 크기 무작위화 없음 |
| 부피 | 1.3³ = 2.197m³ |
| 질량 | 20kg 고정 |
| 생성 밀도 | `20 / (1.3 × 1.3 × 1.3)` ≈ **9.103323kg/m³** |
| 상자 마찰계수 | rigid shape `friction=0.6` |
| 상자 반발계수 | `restitution=0.0` |
| 선형 / 각 감쇠 | 각각0.01 |
| 바닥 정지 / 동마찰계수 | 각각1.0 |
| 바닥 반발계수 | 0.0 |

밀도는 별도로 샘플링하지 않고 질량/부피로 계산해 `AssetOptions.density`에 지정한다. YAML의 `box.build.randomSize=false`, `randomDensity=false`이며 실제 PUSH 생성 크기·질량·마찰의 기준은 `pushBox`다. 비활성 `scaleRangeZ` 범위를 현재 크기 변화로 해석하지 않는다. 마찰 값은 계수이며 고정된 N 단위 마찰력이 아니다.

물리 설정: `controlFrequencyInv=2`, `sim.substeps=2`, PhysX solver type1, position iteration4, velocity iteration0, contact offset0.02m, rest offset0, contact collection2. 이 문서의 값은 YAML/생성 코드 확인 결과이며 이번 문서 작업에서 simulator 질량 tensor를 다시 측정하지 않았다.

구현: [상자 생성·물성](/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/env/tasks/multi_agent/humanoid_ma_open_push.py).

## 3. 과제 보상

사람마다 담당 상자의 **접촉 보상 + 목표 배치(AT) 보상**을 더한다. 두 사람이 같은 상자를 공동 운반하는 보상은 아니다.

### 3.1 기호와 접촉 목표

- `b`: 현재 상자 중심, `g`: 고정 목표 중심, `r`: 사람 root 위치.
- `C`: 유효 양손 접촉, `A`: AT 성공, `S = C OR A`: 접촉 쪽 보상의 포화 조건.
- reset 시 최초 목표 반대쪽 상자 면을 정한다. 이후 목표 방향을 매번 다시 골라 면을 바꾸지 않고, 선택한 material face가 상자 회전을 따라간다.
- 기준점은 그 면 중앙·상자 바닥 기준 높이0.6175m이다.
- `delta`: 양손 중점과 현재 기준점의 차이.
- `d_face`: 각 손과 해당 유한 사각 면 사이의3D 거리 중 최댓값.
- `e_xy = max(norm(delta.xy), d_face)`.
- `e_z = max(abs(delta.z) - 0.0325, 0)`.

상자가 똑바로 바닥에 있을 때 중점 높이0.585~0.650m는 높이 감점이 없다. 실제 코드는 현재 상자 pose로 변환한 기준점에 대한 **월드 Z 오차**를 사용하므로 기울어진 상자에서도 절대 월드 높이 구간을 고정하는 방식은 아니다.

### 3.2 연속 점수와 성공 조건

```text
접촉 상태 phi_C = exp(-10 × e_xy² - (e_z / 0.04)²)
접근 점수 p_C = 1 / (1 + max(norm(r.xy - b.xy) - 0.5, 0) / 1.0)

AT 상태 phi_A = exp(-10 × norm(b - g)²)
AT 접근 점수 p_A = 1 / (1 + max(norm(b.xy - g.xy) - 0.1, 0) / 1.0)
```

`C`는 아래 조건을 모두 만족할 때 인정한다.

1. 양손 각각의 지정 면 거리 ≤0.18m.
2. 양손 중점의 기준점 XY 거리 ≤0.18m.
3. `abs(delta.z) ≤0.0325m` (float32 경계 허용1e-6m).
4. 각 손의 면 법선 반력 ≥0.5N.
5. 상자의 안쪽 법선 방향 합력 ≥0.5N.

각 제어 step에 속한 물리 tick 중 유효 접촉이 한 번이라도 있으면 `_push_hits`에 OR 누적한다. 손 간격이나 두 손 각각의 높이는 따로 강제하지 않는다.

`A`는 `phi_A ≥0.9 AND abs(b.z - g.z) ≤0.001m`이다. 높이 오차가0이면 수평거리 약0.102645m 이내다. **현재 contact_at 경로에는 AT 기울기·속도 성공 제한이나 기울기 감쇠가 없다.** YAML의 `push.max_tilt=0.35`, `goal_tolerance=0.15`, `max_speed=0.2` 등을 이 경로의 AT 성공 조건으로 사용하면 안 된다.

### 3.3 가중치와 합산

```text
R_state    = 0.2 × (S이면1 아니면phi_C) + 0.2 × (A이면1 아니면phi_A)
R_approach = 0.2 × (S이면1 아니면p_C)   + 0.2 × (A이면1 아니면p_A)
R_success  = 0.2 × float(S)             + 0.2 × float(A)
R_local    = R_state + R_approach + R_success
```

접촉·AT 각각 최대0.6, 합계 최대1.2다. YAML의 `state_reward_weight=0.4`는 두 상태 항목에 절반씩 적용하고, progress·success의0.2는 각 항목에 그대로 적용한다.

접촉 시 접촉 쪽 세 항목을 최대치로 포화한다. AT 성공 시 손을 떼도 접촉 쪽까지 포화한다. 성공은 현재 상태로 재판정하며 한 번 성공했다고 에피소드 끝까지 고정하지 않는다. `progress`라는 코드 이름을 쓰지만 현재 보상은 **현재 거리 기반 접근 점수**이며, 이전 step 대비 이동량 증가분이 아니다. 물리 접촉 이동 credit을 추적하더라도 이 contact_at 보상의 추가 변위 항으로 쓰지 않는다.

```text
R_shared_i = 0.9 × R_local_i + 0.1 × R_local_teammate
R_power_i = -0.0005 × sum(abs(dof_torque × dof_velocity))
R_collision_i = -0.5 × max(0.7 - 사람 root 간 XY거리, 0) / 0.7
R_env_i = R_shared_i + R_power_i + R_collision_i
R_train_i = 0.5 × R_env_i + 0.5 × R_AMP_i
```

사람 간 페널티는 실제 접촉 force가 아니라 root XY 근접도다. `box_vel_penalty=false`이고 현재 `_compute_reward`에 별도 상자 속도 페널티 항은 없다. 낙상은 조기종료로 처리한다.

구현: [접촉·AT 기본 합산](/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/env/tasks/multi_agent/open_push_reward.py), [점 보상으로 상태항 보정](/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/env/tasks/multi_agent/humanoid_ma_open_push_anchor.py), [현재 점 보상 runtime](/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/env/tasks/multi_agent/humanoid_ma_open_push_point.py), [점 기하·높이·접촉 조건](/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/utils/push_point.py).

## 4. AMP 설정

### 4.1 Expert 데이터 구성

manifest: `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_push_only_push180.yaml`.

| 그룹 | 모션 수 | expert 그룹 선택 비율 | 그룹 내 선택 |
| --- | ---: | ---: | --- |
| `loco` | 4 | 0.33 | 모든 clip `weight: 1.0` → clip 균등 |
| `push` | 21 | 0.67 | 모든 clip `weight: 1.0` → clip 균등 |

```yaml
skill: [loco, push]
skillInitProb: [1.0, 0.0]
skillDiscProb: [0.33, 0.67]
ampTaskConditioning: true
ampEnabled: true
ampExpertMapping:
  PUSH: [0.33, 0.67]
numAMPObsSteps: 10
```

실제 조건부 expert 생성은 `HumanoidMAOpenPush._fetch_conditioned_amp_demo()`가 `ampExpertMapping.PUSH`를 참조한다. PUSH 과제의 expert에 보행도 33% 포함하므로 접근 보행도 같은 과제 조건으로 학습한다. `skillInitProb`는 초기화 측의 지정이며 AMP expert 비율이 아니다.

AMP는10시점의 신체 특징을 사용한다. `phys_humanoid_v3`의 신체 특징129D/시점에 과제 family one-hot2D를 붙여 **131D×10 = 1310D**가 된다. family는 OPEN=0/PUSH=1의2개 슬롯을 유지하지만 이 실험의 과제는 PUSH뿐이다. 신체 특징에는 root 높이·자세·속도, 관절 자세/속도, 양손·양발 위치 등을 사용하고 과거 시점은 제어 주기 `dt` 간격으로 생성한다. AMP 정규화 후에도 과제 라벨을 유지한다.

### 4.2 AMP 보상과 discriminator 학습

```text
D = sigmoid(discriminator_logit)
R_AMP = 2 × [-log(max(1 - D, 0.0001))]
R_train = 0.5 × R_env + 0.5 × R_AMP
```

| 設定 | 値 |
| --- | ---: |
| `amp_enabled` / `normalize_amp_input` | true / true |
| `task_reward_w` / `disc_reward_w` | 0.5 / 0.5 |
| `disc_reward_scale` | 2 |
| `disc_coef` | 5 |
| `disc_logit_reg` | 0.01 |
| `disc_grad_penalty` | 5 |
| `disc_weight_decay` | 0.0001 |
| `amp_obs_demo_buffer_size` | 200000 |
| `amp_replay_buffer_size` | 200000 |
| `amp_replay_keep_prob` | 0.01 |
| `amp_batch_size` | 512 |
| `amp_minibatch_size` | 4096 |

0.5/0.5는 계수이며 실측 보상의 기여도가 반드시 반반이라는 의미는 아니다. `disc_reward_scale=2`를 적용한 후0.5를 곱한다.

구현: [AMP 보상](/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/learning/amp_agent.py), [과제 라벨 정규화 유지](/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/learning/multi_agent/ma_agent.py), [AMP demo 생성](/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/env/tasks/multi_agent/humanoid_ma_open_push.py), [MotionLib clip 선택·경로 해석](/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/utils/motion_lib.py).

## 5. AMP와 RSI 구분

AMP는 위25clip을 동작 품질의 expert로 사용한다. RSI는 에피소드 시작 자세 샘플링이며 별도 JSON bank를 사용한다.

| RSI設定 | 値 |
| --- | --- |
| bank | `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/push_fixed130_point_zband_heightmatch_goal2to5_bank.json` |
| `interaction_probability` | 0.5 |
| `late_probability` | 0.7 (interaction RSI 내 후반 요청 확률) |
| `late_fraction` | 0.3 |
| `stand_off` | 0.06m |
| 후보 | 높이 정합성·PD 안정성으로 선별한11자세 / 7모션 (PUSH ID0/1/2/5/6/19/20) |

RSI 후보를7모션으로 좁혔어도 **AMP expert의 PUSH21clip은 삭제하지 않았다**. 현재 bank는3~5m 실험의500회에서 목표만2~5m로 바꾼 전용 resume 계약에 대응한다.

## 6. 데이터를 다른 환경에 전달할 때의 경로

manifest의 `file:`은 manifest가 위치한 `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/` 기준 상대 경로로 해석한다. 실행 시 `--motion_file /home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_push_only_push180.yaml`을 지정한다.

- AMASS 링크: `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_amass_loco/motions` → `/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data/dataset_amass_loco/motions`.
- 변환된 PUSH: `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push/motions/`, `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/`.
- PUSH 선정 정보: `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/bones_push180_selection.json`. 모션 변환: `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/scripts/multi_agent/convert_bones_push.py`.

AMP 학습은 변환된 `ref_motion.npy`를 읽으며 원본 BVH를 직접 읽지 않는다. 아래 목록은 내부 사용을 위한 데이터 위치 명세이며 외부 재배포 허가가 아니다. 외부 협업자에게는 설정·코드·공식 다운로드 경로를 전달하고, 데이터는 상대가 자신의 사용 자격으로 취득하도록 한다. 변환 NPY도 임의 공개·제3자 전달 가능한 것으로 간주하지 않는다. 기존 심볼릭 링크를 옮기면 대상 환경에도 링크 원본이 존재해야 한다. 공유 데이터 원본은 읽기 전용으로 취급한다. 사용 조건은7절을 참조한다.

### 전체 AMP 모션 파일

아래는 `/home/hwanhee/koo_cvpr/stage1_fix`를 포함한 전체 절대 경로 목록이다.

#### loco (4개)

```text
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_amass_loco/motions/ACCAD/ACCAD+__+Female1Walking_c3d+__+B1_-_stand_to_walk_stageii/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_amass_loco/motions/ACCAD/ACCAD+__+Female1Walking_c3d+__+B2_-_walk_to_stand_stageii/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_amass_loco/motions/ACCAD/ACCAD+__+Female1Walking_c3d+__+B2_-_walk_to_stand_t2_stageii/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_amass_loco/motions/ACCAD/ACCAD+__+Female1Walking_c3d+__+B3_-_walk1_stageii/phys_humanoid_v3/ref_motion.npy
```

#### push (21개)

```text
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_001__A344_M/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_002__A342/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_002__A342_M/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_002__A344/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_002__A344_M/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push/motions/push_obstacle_180_003__A342/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_003__A342_M/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_003__A344/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_003__A344_M/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push/motions/push_obstacle_180_101__A341/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_101__A341_M/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_101__A343/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_101__A343_M/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_102__A341/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_102__A341_M/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_102__A343/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_102__A343_M/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_103__A341/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_103__A341_M/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_103__A343/phys_humanoid_v3/ref_motion.npy
/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_103__A343_M/phys_humanoid_v3/ref_motion.npy
```

검증: manifest의 보행4개·PUSH21개 총25개 `ref_motion.npy`가 모두 존재함을 확인했다. 로컬 문서 링크·경로와 diff를 점검했으며, 문서 작성만 수행하여 추가 학습/시뮬레이션은 실행하지 않았다.

## 7. BONES-SEED 출처와 사용 조건

### 7.1 어떤 데이터인가

BONES-SEED는 Bones Studio가 Vicon 광학식 모션캡처로 수집한 사람 동작 데이터다. 이 실험은 SOMA Uniform 골격의 BVH를 가져와 humanoid 골격으로 변환했다. 원본에는 이 실험에 사용하는 상자 pose·접촉력의 실측값이 없으며, 사람 참조 동작만 사용하고 상자 상태와 힘은 시뮬레이션에서 계산한다. 출처: [공식 데이터 소개](https://bones.studio/datasets/seed), [공식 뷰어](https://seed-viewer.bones.studio/).

### 7.2 현재 AMP/RSI 용도로 사용 가능한가

2026-10-19 공개 약관을 확인한 결과다. **사용 자격을 충족하면, 현재처럼 제어 정책을 학습하는 AMP와 그 초기화를 위한 RSI는 허용 범위로 해석된다.** 약관2절은 데이터 가공·변환과 ML 학습을 허용하고,6절은 제어 신호를 출력하는 로봇 제어 정책·모션 추적 모델의 학습을 명시적으로 금지 대상에서 제외한다. 이는 현재 사용 방식에 대한 해석이며, 사용자의 기관 자격이나 실제 다운로드 계정의 동의 이력을 확인했다는 의미는 아니다. 근거: [BONES 공식 라이선스 §2·6](https://bones.studio/info/seed-license).

| 구분 | 적용 조건 |
| --- | --- |
| 학술 사용 | 비영리 대학·연구기관의 비상업적 연구이며 연구 결과를 공개해야 함. 일반 기업 소속·겸직자는 학술 소속만으로 자격을 충족하지 않음 |
| 기업 사용 | 약관의 연 매출100만USD 미만 기업 자격 또는 별도 상업 라이선스 필요 |
| 데이터 취득 | 공식 Hugging Face의 접근 조건·약관에 동의하고 취득. 현재 로컬 파일 취득 계정/동의 기록은 이번 작업에서 검증하지 않음 |
| 논문·영상·모델 공개 | 출처 표시 필요. 모델에서 원본 데이터를 추출·복원할 수 없어야 함 |
| BVH·변환 NPY 공유 | 공개 저장소 업로드·제3자 재배포를 허용하는 것으로 해석하지 않음. 내부 직원/계약자 접근에도7절 비밀유지 조건 적용 |
| 모션 생성 모델 | 원본을 상업적으로 대체하는 모션 데이터 생성 용도는 제한됨 |

라이선스는 자체 조건의 계약이며 무제한 공개 데이터가 아니다. 원본의 가공·리타게팅만으로 재배포 권한이 생기지 않는다. 출처: [공식 라이선스 §1·3·5–8](https://huggingface.co/datasets/bones-studio/seed/blob/main/LICENSE.md), [공식 다운로드·접근 조건](https://huggingface.co/datasets/bones-studio/seed).

논문 Data/Acknowledgements에는 `Motion Data by Bones Studio`와 [Bones Studio](https://bones.studio/) 링크를 표시한다. 모델·소프트웨어를 배포할 때는 README/model card 등에 아래 취지의 출처·라이선스 안내를 넣는다. 정확한 요구 문구는 [공식 약관8절](https://bones.studio/info/seed-license)을 따른다.

> Training data includes Motion Data by Bones Studio.

이 문구에 Bones Studio 링크와 BONES 라이선스 적용 안내를 함께 기재한다. **이 확인은 BONES PUSH21clip에 관한 것이며 AMASS/ACCAD 보행4clip의 사용 조건은 별도다.** 보행 데이터 라이선스는 이번 문서 확장에서 확인하지 않았다.

## 8. 원본 모캡 → AMP expert 변환

### 8.1 변환 파일과 산출물

| 역할 | 절대 경로 |
| --- | --- |
| 원본 SOMA Uniform BVH | `/home/hwanhee/koo_cvpr/stage1_fix/bones_seed/soma_uniform/bvh/` 아래 날짜별 디렉터리 |
| 원본 시간 구간 metadata | `/home/hwanhee/koo_cvpr/stage1_fix/bones_seed/metadata/seed_metadata_v002_temporal_labels.jsonl` |
| 21clip 선정·crop 기록 | `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/bones_push180_selection.json` |
| PUSH 변환 진입점 | `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/scripts/multi_agent/convert_bones_push.py` |
| 공통 IK 변환 구현 | `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/scripts/multi_agent/convert_bones_door.py`의 `convert(..., profile='push')` |
| BVH parser/FK | `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/utils/bones_bvh.py` |
| 대상 골격 | `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml` |
| 모션별 산출물 |6절의 각 `phys_humanoid_v3/` 내 `ref_motion.npy`와 `conversion_report.json` |

`convert_bones_door.py`라는 파일명을 공유하지만 PUSH는 `profile='push'` 분기로 처리한다. 문 열기 모션으로 변환한 것이 아니다.

### 8.2 실제 전처리 순서

1. **선정·crop:** 정면 양손 PUSH인 `push_obstacle_180_*`21개를 선정한다. 신규19개는 temporal metadata에 따라0초부터 밀기 중단/팔 해제 직전까지 자른다. 기존2개 `push_obstacle_180_101__A341`(0~12.2초), `push_obstacle_180_003__A342`(0~11.733333초)는 기존 변환본을 보존한다. 정확한 clip별 구간은 selection JSON과 conversion report에 있다.
2. **BVH 복원:** Hips translation과 계층별 회전 채널을 읽어 forward kinematics로 관절 위치·회전을 복원한다. cm 단위를 m로 바꾼다.
3. **시간축 변환:** 원본 시각을 기준으로 **30FPS**로 재표본화한다. 위치는 선형 보간, 회전은 SLERP다. 원본 FPS는 각 report에 기록하며 모든 원본이 정확히120FPS라고 가정하지 않는다.
4. **좌표 정렬:** 원본 Y-up을 Z-up으로 바꾸고 초기 신체 전방을+X, 왼쪽을+Y로 정렬한다. root 초기 XY를 빼 시작 수평 위치를0으로 맞춘다.
5. **신체 비율:** 원본과 대상의 왼쪽 하퇴·발 골격 길이 합의 비율로 root 이동과 상대 관절 위치를 스케일링한다. 대상 humanoid의 골격 길이는 유지한다.
6. **제약 IK:** SOMA 관절을 대상15개 body로 매핑하고 대상 XML의 관절 한계를 적용한다. 대상의32DOF 관절값을 bounded least-squares로 맞춘다. 손·발 끝점 위치, 몸통/머리/발 방향, 기준 자세 및 직전 프레임과의 차이를 목적함수에 사용한다. 원본 관절각을 그대로 복사하는 방식은 아니다.
7. **회전·바닥 보정:** 인접 quaternion 부호를 연속화하고 발 mesh가 clip 전체에서 바닥 아래로 내려가지 않도록 root Z에 일정한 오프셋을 더한다. 가장 낮은 발 mesh 높이를0.005m로 맞추며 프레임마다 발을 강제로 붙이지는 않는다.
8. **저장:** local quaternion·root translation에서 Poselib `SkeletonMotion`을 만들어 `ref_motion.npy`로 저장한다. 속도 등은 이 motion 표현에서 계산·로드해 AMP 특징을 구성한다. report에는 원본 SHA256·crop·FPS·스케일·바닥 이동량·끝점 오차·관절 제한 오차를 기록한다.

구체적 예: `push_obstacle_180_001__A344_M`의 기존 report는 원본 약120.0048FPS →30FPS/320프레임, crop0~10.65초, 저장 구간 길이10.6333초, scale 약0.908961, root 바닥 보정 약+0.083007m, 관절 제한 위반0rad를 기록한다. 이는 한 clip의 변환 기록이며 전체 clip의 오차가 같다는 의미는 아니다.

예시 report: `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/dataset_bones_push180/motions/push_obstacle_180_001__A344_M/phys_humanoid_v3/conversion_report.json`.

### 8.3 AMP 학습 시 처리

manifest가 변환된 PUSH21clip과 기존 AMASS 보행4clip을 읽는다. MotionLib는 clip의 pose·속도를 보간하고, expert sampler는 보행33%/PUSH67%로 clip과 시간을 샘플링한다. 시간`t, t-dt, …, t-9dt`의 신체 특징과 PUSH family 라벨을 묶어 discriminator expert로 만든다. **AMP expert의 NPY에는 아래 RSI용 손 낮추기 보정을 덮어쓰지 않는다.** 여기서 expert가 원본이라고 할 때는 BVH 자체가 아니라, RSI 보정을 적용하지 않은 humanoid 리타게팅 모션을 뜻한다.

## 9. AMP 모션 → RSI 초기 상태 변환·선별

### 9.1 AMP와 다른 역할

| 구분 | AMP expert | RSI |
| --- | --- | --- |
| 목적 | 동작 품질의 discriminator 기준 | 실제 시뮬레이션 시작 상태 |
| PUSH 모션 범위 |21clip 유지 | 현재 bank의7clip·11자세 |
| 추가 팔 높이 보정 | 없음 | reset 때 어깨 회전 보정 |
| 저장 내용 | 전체 시계열 `ref_motion.npy` | `motion_id/time/phase/box_progress` 등 후보 JSON |
| 상자 상태 | 원본 실측 상자 궤적 없음 | 장면·목표·후보 진행률에서 시뮬레이션 상태 생성 |

### 9.2 후보 생성과 reset 변환

1. 변환된 humanoid 모션에서 root 높이·기립 방향·발 높이·손 전방 도달 등을 확인하고 AMP history를 만들 수 있는 프레임을 후보로 만든다. 기하 후보는 적격 프레임을4프레임 간격으로 추출하고 `phase`와 `box_progress=0.85×phase`를 기록한다. 이 `box_progress`는 원본 상자 움직임의 측정값이 아니라 초기 배치를 위한 합성 진행률이다.
2. 후보의 `motion_id/time`으로 사람 pose와 속도를 읽는다. reset 전용 `lower_push_arms()`가 각 어깨를 중심으로 팔 subtree를 회전해 기준 손 높이0.6175m 쪽으로 낮춘다. 몸통·발·팔 길이는 유지하며 도달 범위 때문에 모든 자세가 정확히 같은 높이가 되지는 않는다. 보정된 어깨 DOF 속도는0으로 설정하고 자식 body의 속도·각속도도 일관되게 변환한다.
3. 상자는 `초기 상자 위치 + box_progress × 목표 오프셋`에 둔다. 사람은 양손의 XY 방향/중점이 상자 접촉 목표 쪽을 향하도록 **yaw 회전+수평 이동**으로 정렬하고 stand-off0.06m를 사용한다. 이 정렬은 발 Z를 유지한다.
4. root·관절·속도와 기하 body 상태를 simulator에 설정한다. JSON은 전체 자세를 복제 저장한 것이 아니라 원본 motion과 변환 규칙을 참조하는 bank다.

구현: `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/utils/open_push_rsi.py`의 `geometry_candidates/lower_push_arms/align_push_pose`, `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/env/tasks/multi_agent/humanoid_ma_open_push.py`의 `_apply_joint_rsi`.

### 9.3 현재11개 후보가 만들어진 과정

- 기존 저높이 PD 검사 bank에서 물성·모션 hash·기하/팔 보정 계약을 대조해 point용408후보를 재사용했다. donor는 `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/push_fixed130_point_goal3to5_bank.json`이다.
- `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/scripts/multi_agent/build_push_height_matched_rsi.py`가 이408후보를 후보당 독립2장면·12제어 step으로 다시 검사했다.64평가환경에서32후보씩 처리하며 zero-action PD 안정성을 검사했다.
- 두 장면 모두 **초기 기하 참조 자세와 첫 물리 step의 양손 중점 Z가0.585~0.650m**인 후보만 남겼다.12step 동안 관측/보상 finite, reset 없음, root 높이>0.65m, root3D 이동<0.45m, 상자 중심 Z의0.65m 대비 오차<0.08m도 확인했다.
- step2 이후 발 높이<0.35m·수직 접촉력≥1N인 발 지지가 관측되고, 양손 지정 면 거리 최댓값<0.18m인 시점이 관측돼야 한다. 발 지지·손 근접 조건은 매 step 연속 유지 조건이 아니다.
- 최종11후보,7모션 ID0/1/2/5/6/19/20을 보존했다. 이후 step 전체에서 높이 구간을 유지한다거나 장기 정책 성공·상자 기울기5도 이하를 보장하는 검사는 아니다.
- 결과 bank: `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/push_fixed130_point_zband_heightmatch_goal3to5_bank.json`. 기존 검사 기록: `/home/hwanhee/koo_cvpr/stage1_fix/output/push_fixed130_point_zband_heightmatch_goal3to5_rsi_check/height_screen.json`.
- 현재2~5m bank는 위11후보를 유지하고 목표 범위 계약만 바꾼 것이다. 이관: `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/scripts/multi_agent/prepare_push_heightmatch_goal2to5_resume.py`. 현재 bank: `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data/push_fixed130_point_zband_heightmatch_goal2to5_bank.json`.2~5m로 바꾼 뒤 전후보 PD 검사를 다시 실행한 것은 아니다.

### 9.4 학습 중 RSI 샘플링과 AMP 이력

학습 reset의 interaction RSI 확률은50%이고 나머지는 보행 reference 시작이다. interaction RSI에서70% 확률로 각 생존 clip 후보의 시간순 마지막30% 풀을 요청한다. 이는 전체 원본 모션의 마지막30%와 동일하지 않다. 일반 풀과 후반 풀 모두 생존 clip의 원래 weight를 기준으로 clip별 확률을 배분하고 해당 clip의 후보에 분배한다. 후보가 많은 clip이 자동으로 더 큰 비중을 갖지 않는다.

PUSH RSI로 시작할 때 **초기 AMP history의 과거9시점에도 같은 어깨 회전 보정**을 적용한다. 현재 상태와 과거 이력이 서로 다른 팔 자세로 이어지지 않도록 하는 처리다. 이는 agent 측 초기 AMP 관측의 변환이며 expert sampler나 NPY를 수정하는 것은 아니다. 구현은 `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/env/tasks/multi_agent/humanoid_ma_open_push.py`의 `_init_amp_obs_ref`다.

추가 문서 검증: 변환 코드·예시 conversion report·현재 bank metadata를 대조했다. 이 문서 확장에서는 모션 재변환·bank 재생성·추가 GPU 검증을 실행하지 않았다.
