# 보정된 공동운반 RSI 연결 인계

기준일: 2026-10-07. 목표는 준비된 초기 상태를 기존 환경의 **두 사람·한 상자 공동운반 RSI reset**에 연결하는 것이다. 현재 학습/graph/reset 연결은 미구현이며, 인계 정리 중 학습·reward·checkpoint는 변경하지 않았다.

## 다음 Codex에게 전달할 요청

> 저장소 AGENTS.md와 joint_carry/HANDOFF.md를 읽고, joint_carry/carry_joint_corrected의 screened_rsi_snapshots.npz를 기준으로 두 사람·한 상자의 공동운반 RSI reset을 별도 실험에 구현해줘. 기존 독립 운반 실험은 유지해줘. 두 사람은 같은 skill/sample을 함께 뽑고 공유 상자는 한 번만 초기화해야 해. Graph/물체 binding, reset 뒤 덮어쓰기, AMP 이력과 skill conditioning까지 확인해줘. 현재 검증은 0.1초 초기 상태 검사이므로 연속 전문가 모션으로 간주하면 안 돼. 우선 CPU 테스트와 짧은 시뮬레이션·2048환경 확인 학습으로 검증하고, 본학습 시작은 이번 요청 범위에 포함하지 않아.

## 기준 데이터

| 동작 | RSI에 읽을 파일 | 선택 프레임 수 |
| --- | --- | ---: |
| pickUp | [snapshot](carry_joint_corrected/pickUp/screened_rsi_snapshots.npz) | 55 |
| carryWith | [snapshot](carry_joint_corrected/carryWith/screened_rsi_snapshots.npz) | 129 |
| putDown | [snapshot](carry_joint_corrected/putDown/screened_rsi_snapshots.npz) | 135 |

한 행은 **한 환경에 함께 넣을 두 사람과 공유 상자 한 개**다. 서로 다른 행에서 사람을 하나씩 뽑지 않는다. 총 319개는 독립 초기 상태이며 클립 사이를 연결한 연속 궤적이 아니다. 원본 B19/B20/B21 각각 한 클립에서 합성했다. 다른 pickUp 원본까지 모두 포함했다는 뜻은 아니다.

같은 폴더의 `cooperative_rsi_candidate.npz`는 보정된 전체 reference다. `screened_rsi_snapshots.npz`가 실제 선택 대상이며 pickUp 원본 제외 구간은 빠져 있다. `original_joints.npz`와 `archive/`는 비교·재생성 자료다.

`training_loader_integrated=false`, `continuous_motion_validated=false`가 현재 정확한 상태다. 전체 reference의 `rsi_ready/physics_validated=false`를 임의로 바꾸지 않는다. 선별 snapshot의 `short_rsi_screen_pass=true`는 아래 짧은 검사만 뜻한다.

## 배열 계약

`numpy.load(path, allow_pickle=False)`로 읽는다. 단위는 m·m/s·rad·rad/s·초, quaternion은 **xyzw**, 축은 **Z-up**이며 reference는 30fps다.

| key | shape | 의미 |
| --- | --- | --- |
| `root_state` | `[K,2,13]` | 위치 0:3, quaternion 3:7, 선속도 7:10, 각속도 10:13 |
| `dof_position`, `dof_velocity` | `[K,2,32]` | `phys_humanoid_v3` 순서의 관절값·속도 |
| `box_state` | `[K,1,13]` | 공유 상자 하나의 위치·회전·속도 |
| `box_size` | `[3]` | 로컬 XYZ 길이 `[0.52,0.80,0.40]` |
| `source_frame`, `source_time` | `[K]` | 전체 보정 reference의 프레임 번호·시간 |
| `short_rsi_screen_pass` | `[K]` | 선별된 행 모두 true |
| `screen_seconds` | scalar | `0.1` |

상자 밀도 100kg/m³·질량 16.64kg이다. 다른 random-size asset으로 대체하면 검증 결과를 재사용할 수 없다. Box 높이/속도를 기존 `_reset_boxes`의 바닥 clamp·속도 0 설정으로 바꾸어도 같은 초기 상태가 아니다. 원본 상자 높이에는 바닥과의 간격이 있을 수 있다.

Agent 0은 이동한 원본, agent 1은 공유 상자의 반대편 복제다. 내부 관절각은 보정 후 둘이 같고 root 회전이 다르다. Actor slot을 바꾸면 root/DOF/graph owner를 함께 바꾼다.

32-DOF 순서는 [humanoid.py](../tokenhsi/env/tasks/humanoid.py)의 `_setup_character_props`와 같다. `DOF_BODIES=[1,2,3,4,6,7,9,10,11,12,13,14]`, `DOF_OFFSETS=[0,3,6,9,10,13,14,17,20,23,26,29,32]`다. 3-DOF는 exponential map, 팔꿈치는 Y축 1-DOF다. Euler XYZ로 다시 해석하지 않는다.

## 보정·검증 범위

- 실제 1축 팔꿈치·고정 손목으로 변환하고 손 접촉·팔 capsule 충돌을 고려한 어깨/팔꿈치 IK를 적용했다. 손 중심 목표는 손 반지름을 고려해 상자 옆면에 배치하며 약 5mm의 접촉 관통을 허용했다.
- 상자 크기·15cm 안쪽 잡기 위치·다리/몸통/머리 관절각은 유지했다. Root는 클립별 상수 1.4~2.6cm 올렸다. 팔 DOF 최대 변경은 pickUp 42.0°·carryWith 29.4°·putDown 36.6°다. 원본 관절각을 전부 유지한 데이터가 아니다.
- Isaac Gym **CPU PhysX**, 동일 asset/PD gain/중력/마찰, 1/60초·2 substep이다. 매 프레임 reset 후 초기 관절 target을 고정하고 6 step을 진행했다.
- 기존 RSI 충격 기준: root 선속도 변화 ≤3m/s, 상자 속도 ≤5m/s, root/상자 변위 각각 ≤0.3m, finite. 추가로 형상 관통 ≤2cm와 원본 RSI 제외 조건을 적용했다. carryWith는 두 사람 모두 최소 3/6 step 손 접촉을 요구했고 실제로는 최소 5/6 step이었다.
- 55/129/135개가 통과했다. carryWith 손/팔 최대 관통은 11.51/10.10cm→약 0.50/0cm, 실제 FK와 렌더 FK 오차는 6.44e-7m 미만이다. [보고서](carry_corrected_physics/report.json), 동작별 `cooperative/frames.csv`가 근거다.
- 형상 계산은 구/capsule 중심선 표본을 사용한다. 사람 간 관통은 성긴 표본의 하한이며 mesh-상자와 self collision은 PhysX에서 처리한다. 2cm는 선별 기준이지 물리 법칙의 허용치를 뜻하지 않는다.
- **미완료:** GPU 학습 환경에서의 paired reset, 연속 발 미끄러짐·장기 균형, 학습 정책에 의한 성공, 두 사람용 AMP expert 타당성. Native PD 장기 추종은 원본·합성 모두 실패했다. 초기 상태 통과를 연속 동작 성공으로 해석하지 않는다.

## 연결할 코드와 주의점

| 파일·심볼 | 현재 동작 / 필요한 작업 |
| --- | --- |
| [humanoid_ma_carry.py](../tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py) `_reset_ref_state_init` | 사람별 skill/clip/time을 뽑는다. 같은 물체에 두 advanced skill이 붙으면 재샘플하는 conflict 루프가 있다. 공동운반 전용 분기에서 paired sample을 한 번 뽑아 함께 적용한다. 일반 독립 경로의 검사를 단순 삭제하지 않는다. |
| 같은 파일 `_reset_actors`, `_reset_boxes`, `_reset_envs` | 기존 object reset이 공유 상자를 다시 쓰거나 속도를 0으로 바꾸지 않게 순서를 설계한다. 두 사람과 상자에 동일한 환경 좌표 변환을 적용한다. |
| [edge_scenario_spec.py](../tokenhsi/utils/edge_scenario_spec.py), [edge_ontop_task.py](../tokenhsi/env/tasks/multi_agent/edge_ontop_task.py) | 현재 unified는 독립 과제 sampler다. 동일한 **물리 object ID**를 두 owner가 공유하는 graph/binding·goal을 정의한다. Stage 2라는 이름만 보고 공동운반이 이미 지원된다고 가정하지 않는다. |
| [task_role_spec.py](../tokenhsi/utils/task_role_spec.py), [edge_stage1_reward.py](../tokenhsi/env/tasks/multi_agent/edge_stage1_reward.py) | H0→O, H1→O와 O→goal/support 표현·owner별 보상·물체 보상 중복·성공 판정을 확인한다. 정책 packet/관측 계약도 함께 점검한다. |
| 환경 `_init_amp_obs_ref`, [unified_training.py](../tokenhsi/utils/unified_training.py) | 현재 AMP 과거 상태는 단일 인물 MotionLib에서 복원한다. 보정한 팔·root와 이력이 불일치하지 않도록 별도 초기화 정책을 결정·검증한다. 복제 reference를 검증된 AMP expert로 바로 추가하지 않는다. |
| [size_rsi_cache.py](../tokenhsi/env/tasks/multi_agent/size_rsi_cache.py), [size_rsi.py](../tokenhsi/utils/size_rsi.py) | 기존 캐시는 한 사람·크기별 초기 충격 검사다. 이번 paired 형상/접촉 검사를 대신하지 않는다. 0.80m 길이 상자도 기존 독립 asset 분포와 별도 취급한다. |

권장 구현 순서:

1. 별도 config/variant·checkpoint 계약을 정한다. NPZ를 최초 한 번 로드해 device tensor로 캐시한다. Reset마다 환경 단위로 한 행을 뽑고 사람 축은 함께 사용한다.
2. 먼저 carryWith 단일 scene에서 두 사람·상자 하나를 복원한다. Asset 크기·밀도와 root/DOF/상자 위치·회전·속도를 원본 행과 대조한다.
3. 공유 object graph와 actor indexing을 연결한다. Partial reset에서 다른 환경이 변하지 않는지, 상자를 한 번만 쓰는지 검사한다.
4. 공통 yaw/평행 이동은 `p'=Rz*p+t`, `q'=qz*q`, `v'=Rz*v`, `ω'=Rz*ω`로 두 사람·상자에 함께 적용한다. DOF는 유지하고 env origin을 중복 더하지 않는다.
5. pickUp/putDown, skill 비율, AMP 이력/conditioning을 연결한다. `source_frame`은 압축된 snapshot의 행 번호가 아니다. 과거 reference를 찾을 때 구분한다.
6. 관련 CPU 테스트 후 실제 GPU의 짧은 reset/접촉·부분 reset·finite·graph binding 검사를 한다. 추가 물체와 GPU PhysX 차이도 재검증한다.
7. 실행용 config를 추가하면 전용 train/test/VNC와 문서를 함께 만든다. 짧은 학습도 **2048환경**, `MAX_ITERATIONS` 축소·별도 `OUTPUT_PATH=output/<실험>_check`를 사용한다. 본학습은 이번 인계 범위가 아니다.

완료 판정은 **실제 환경에서 공유 물체·두 사람·속도·AMP 이력·graph가 일치하고 첫 물리 step이 검증된 상태**다. 기존 학습은 임의 종료/재시작하지 않는다.

## 확인·재생성

모든 명령은 저장소 루트에서 실행한다. 루트 AGENTS.md를 따른다. conda는 `tokenhsi`, GPU 실행 전 `nvidia-smi`로 점유를 확인한다. 공유 원본 모션은 읽기 전용이다.

이미 있는 보정본은 다음 확인만으로 시작한다. 재생성은 필수가 아니다.

```bash
source tokenhsi/scripts/multi_agent/runtime_env.sh
python joint_carry/scripts/check_handoff.py
```

아래는 기존 산출물을 덮어쓰는 **재생성 절차**다. 현재 전달 ZIP을 보존한 뒤 필요할 때만 실행한다.

```bash
source tokenhsi/scripts/multi_agent/runtime_env.sh
python joint_carry/scripts/carry_joint_preview.py --skill all --grip-inset 0.15 --output joint_carry/archive/carry_joint_preview_all_inset15
TOKENHSI_GPU=0 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
python joint_carry/scripts/validate_carry_physics.py --input joint_carry/archive/carry_joint_preview_all_inset15 --output joint_carry/archive/carry_physics_validation'
python joint_carry/scripts/correct_carry_candidates.py --no-render
TOKENHSI_GPU=0 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
python joint_carry/scripts/validate_carry_physics.py --input joint_carry/carry_joint_corrected --output joint_carry/carry_corrected_physics'
python joint_carry/scripts/correct_carry_candidates.py --finalize
python joint_carry/scripts/check_handoff.py --write-manifest --package
```

`--finalize`는 물리 보고서 SHA256과 일치하는 입력만 사용한다. Manifest 갱신은 데이터/코드를 의도적으로 바꾼 뒤 수행하며 과거 물리 검증을 새 데이터에 자동 부여하는 기능이 아니다.

## 전달 범위

`joint_carry_handoff.zip`을 같은 저장소 루트에 풀면 `joint_carry/`가 생긴다. 보정본·물리 결과·코드·이 문서를 포함하며 `archive/`와 중복 하위 ZIP은 제외한다. TokenHSI 코드·asset·conda·Isaac Gym은 별도 필요하다. 보정 NPZ 로딩에 원본 모션 재생성은 필요 없지만 전체 재생성에는 원본 모션 심링크도 필요하다.

공통 [config.md](../markdowns/config.md), [structure.md](../markdowns/structure.md), [changelog.md](../changelog.md)는 저장소 안내/이력을 유지한다. 공동운반 전용 코드는 모두 `joint_carry/scripts/`에 모았다. 저장소 전체가 이 폴더 하나로 독립 실행된다는 뜻은 아니다. 큰 데이터/영상은 Git 제외 대상이므로 새 checkout에는 **전달 ZIP도 함께 복사**한다.
