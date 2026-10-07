# 레포 구조

현재 실행 실험은 PUSH/DOOR Stage 1, 원본 multi-agent carry(1), 독립 Stage 1(27·28·30·31·32·33·34·35·36·37 및 paired AT/placement·unified 독립 5과제), 협력 Stage 2(29 및 SIT plane 변형)다. 실행 명령과 checkpoint 규칙은 [config.md](config.md)에 있다.

## 진입점

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/run.py` | 학습·평가 태스크와 알고리즘 등록 |
| `tokenhsi/box_cleanup_demo.py` | 지정 rescue checkpoint의 4명·16상자 정리 데모 평가 전용 진입점 |
| `tokenhsi/scripts/multi_agent/` | 현재 실험별 train/test/VNC, `runtime_env.sh`, `run-gui.sh` |
| `tokenhsi/data/cfg/multi_agent/` | 현재 실험 YAML과 평가 graph 예제 |
| `tokenhsi/data/cfg/train/rlg/` | PPO/AMP/Transformer 학습 설정 |
| `output/<실험>/<run>/` | checkpoint `nn/`, TensorBoard `summaries/`, 진단 `diagnostics/` |

## 현재 실험이 사용하는 코드

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py` | 물체 배정·RSI·물리 환경·관측 통합 |
| `tokenhsi/at_goal_demo.py`, `tokenhsi/env/tasks/multi_agent/at_goal_demo.py`, `tokenhsi/learning/multi_agent/at_goal_player.py` | Rescue checkpoint 평가 전용 AT 목표 중심 높이 0~2m·0.2m 간격과 세 XY 위치 순회·장면별 초기화·VNC/JSON 기록 (`scripts/multi_agent/at_goal_demo_test.sh`, `at_goal_demo_vnc.sh`) |
| `tokenhsi/env/tasks/multi_agent/box_cleanup_demo.py`, `tokenhsi/learning/multi_agent/box_cleanup_player.py`, `tokenhsi/utils/box_cleanup_spec.py` | 중앙 16상자 크기 혼합·불규칙 2단 더미·윗단 두 라운드 운반·회색 표시·바닥 안정 판정·옛 rescue checkpoint 평가 로드·기본 10회/1500 step·VNC 결과 저장 |
| `tokenhsi/tests/test_box_cleanup_demo.py` | 8개 상자 배정·100개 혼합 크기 배치의 초기 겹침·목적지 높이·checkpoint 계약·네 명 완료 barrier와 물리 상태 유지 검증 |
| `tokenhsi/env/tasks/multi_agent/edge_ontop_task.py`, `edge_context_task.py` | graph reset, 가까운 시작, 물리 타당성·진단 |
| `tokenhsi/env/tasks/multi_agent/edge_stage1_reward.py` | Stage 1 edge 보상·포화·팀 공유 |
| `tokenhsi/env/tasks/multi_agent/edge_ontop_reward.py`, `edge_interaction_reward.py` | AT·ON_TOP·SIT·CLIMB state/progress와 plane 성공 |
| `tokenhsi/utils/edge_scenario_spec.py` | 독립 물체·goal binding, 3물체·4물체 같은 과제 쌍 및 canonical 독립 5과제 sampler; size RSI의 GPU 배치 graph 생성 |
| `tokenhsi/utils/size_rsi.py`, `tokenhsi/env/tasks/multi_agent/size_rsi_cache.py` | 행동별 실제 asset 크기·조건부 과제 배정·프레임별 초기 충격 캐시·후반 RSI 선택 |
| `tokenhsi/utils/unified_training.py` | unified 과제별 AMP 전문가 분포·라벨·리플레이 매칭·설정 검증 |
| `tokenhsi/utils/edge_stage1_spec.py` | Stage 1 variant·semantic/owner-HOLDING packet·후반 CLIMB/ON_TOP putDown RSI 검증 |
| `tokenhsi/utils/edge_stage2_spec.py` | Stage 2 협력 graph·평가 preset |
| `tokenhsi/utils/relation_task_spec.py` | reward config·checkpoint 계약 검증 |
| `tokenhsi/learning/multi_agent/amp_network_builder_ma.py`, `edge_context_encoder.py`, `ma_agent.py` | GTA actor/critic, edge bias·HOLDING 상태 융합, AMP·PPO 학습 |
| `tokenhsi/learning/multi_agent/coordination_head.py`, `stage2_transfer.py` | 협력 attention·Stage 1 weight 이식 |
| `tokenhsi/tests/test_scenario_independent_with_climb.py`, `test_scenario_stage1_plane.py`, `test_scenario_stage1_sit_plane_curriculum.py`, `test_stage2_coordination.py` | 현재 실험 CPU 검증 |
| `tokenhsi/tests/test_size_rsi.py` | size RSI config·checkpoint 격리·크기/과제 분포·progress·프레임 pool·대체 skill·배치 graph/부분 reset 동등성 검증 |
| `tokenhsi/tests/test_stage1_unified.py` | 세 unified config 계약·공유 edge gradient·독립 분포·goal 회전·AMP family 매칭·토큰 순열의 actor/value 및 gradient 검증 |
| `tokenhsi/tests/test_typed_edge_bias.py` | Size RSI typed-bias 계약·전체 과제 쌍/goal slot/edge 방향·토큰/edge 순열의 출력/gradient·독립 표 업데이트·checkpoint 복원 검증 |
| `tokenhsi/tests/test_typed_edge_message.py` | 관계 메시지 설정·GTA 뒤 가중합·head 분할·NONE/SELF masking·순열 출력/gradient·독립 업데이트·checkpoint 복원 검증 |
| `tokenhsi/tests/test_graph_box_speed_penalty.py` | graph 담당 물체의 속도 패널티·부분 reset history·기존 할당 경로 회귀 검증 |

`edge_context_spec.py`, `edge_ontop_spec.py`, `edge_interaction_spec.py`와 관련 reward 함수는 과거에 도입됐지만 현재 Stage 1·2의 graph·관찰·보상 계산에서도 호출된다. 파일명만 보고 삭제하면 현재 실험이 깨진다.

Size RSI typed-bias의 전용 env/train YAML과 train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_typed_bias` 이름을 쓴다. `amp_network_builder_ma.py`의 `TypedEdgeBias`와 `edge_context_encoder.py`의 `PackedEdgeTypedBiasFusion`이 타입별 bias 표를 graph에 연결한다.

Typed bias + relation message의 전용 env/train YAML과 train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_typed_bias_message` 이름을 쓴다. `edge_context_encoder.py`의 `PackedEdgeRelationMessage`가 layer 공유 관계 표를 조회하고, `RelationTransformerLayer`가 GTA 복귀 후 같은 attention으로 가중한 관계 메시지를 더한다.

Task message의 전용 env/train YAML과 train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_task_message` 이름을 쓴다. `utils/task_role_spec.py`가 primitive graph에서 task·역할 packet과 정책 3연결을 만들고, `learning/multi_agent/task_role_encoder.py`의 `TaskRoleFusion`이 역할별 bias/message 표를 조회한다. `tests/test_task_role_message.py`는 16개 과제 쌍·물체/goal/owner 재배정·edge/task/token 순열·gradient·보상·크기/RSI/AMP binding·checkpoint를 검증한다.

## BONES 모션 데이터 준비

- `data/dataset_bones_dooropen/`, `data/dataset_bones_push/` (앞에 `tokenhsi/`): 원본 BVH와 `motions/<clip>/phys_humanoid_v3/` 변환 결과. 생성된 모션·보고서·프리뷰는 Git에서 제외한다.
- `tokenhsi/utils/bones_bvh.py`: SOMA 절대 Hips 위치·채널 순서에 맞춘 BVH 파서.
- `tokenhsi/scripts/multi_agent/convert_bones_amp.py`: 좌표·단위·FPS 변환, 관절 제한과 프레임 연속성의 CPU IK, SHA256·오차 보고서와 manifest 생성.
- `tokenhsi/scripts/multi_agent/validate_bones_amp.py`, `preview_bones_amp.py`: 실제 MotionLib·multi-agent AMP 관측 CPU 검증 및 standalone 재생·contact sheet.
- `tokenhsi/data/dataset_bones_door_amp.yaml`, `dataset_bones_push_amp.yaml`: 인간 참조 모션 목록; 새 task/RSI object binding은 별도 구현 대상.
- `tokenhsi/tests/test_bones_bvh.py`: 절대 위치·단위·intrinsic Euler·FK·비정상 입력 회귀.

## Door 물리 환경 준비

- `tokenhsi/data/assets/door/left_hinge_right_handle.urdf`: 고정 문틀·왼쪽 revolute 힌지·오른쪽 앞/뒤 손잡이·충돌 형상.
- `tokenhsi/utils/door_asset.py`: 문 치수·좌표·손잡이 기하·복원 스프링 spec와 재현 가능한 URDF 생성.
- `tokenhsi/env/tasks/multi_agent/door_scene.py`: 재사용 DoorFixture·asset 생성·자동 닫힘 drive·DOF/body 인덱스·문 관측·부분 reset.
- `tokenhsi/door_environment.py`: 독립 GPU 물리 접촉 개방·자동 닫힘·좌표/reset 검증 및 viewer 진입점. 학습 task는 아래 Push/Door 모듈에서 제공한다.
- `tokenhsi/scripts/multi_agent/door_environment_test.sh`, `door_environment_vnc.sh`: headless/로컬 viewer·서버 VNC 실행. `runtime_env.sh`의 tokenhsi 환경을 사용한다.
- `tokenhsi/tests/test_door_asset.py`: 좌우·개방 방향·복원 힘·URDF 생성/충돌·비정상 spec 검증.

## 작업별 확인

- **Stage 1 graph·보상:** 해당 YAML → `edge_scenario_spec.py` → `edge_ontop_task.py` → `edge_interaction_reward.py`·`edge_stage1_reward.py`; 위 Stage 1 테스트로 검증한다.
- **Stage 2 협력·전이:** `approach_stage2_coordination.yaml` → `edge_stage2_spec.py` → `coordination_head.py`·`stage2_transfer.py`; `test_stage2_coordination.py`로 검증한다.
- **실행·GPU:** [config.md](config.md), `runtime_env.sh`, `mps/`, `run-gui.sh`를 확인한다.
- **지표 해석:** [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 확인한다.

## Push/Door Stage 1 학습

- `tokenhsi/env/tasks/multi_agent/humanoid_ma_push_door.py`: 2-agent/2-box/2-door 배치·env-local reset·진행도/유지 보상·손잡이 접촉·단계별 AMP.
- `tokenhsi/utils/push_door_spec.py`: phase hysteresis·최고 진행도·연속 성공·접촉·AMP expert window·동적 관계 binding·checkpoint 계약.
- `tokenhsi/data/cfg/multi_agent/push_door_stage1.yaml`, `data/cfg/train/rlg/amp_ma_push_door_stage1.yaml` (둘 다 앞에 `tokenhsi/`): 전용 환경/학습 설정. `data/dataset_push_door_stage1.yaml`는 loco/BONES push/door 참조 목록이다.
- `tokenhsi/scripts/multi_agent/push_door_stage1_{train,test,vnc}.sh`: GPU 6 기본·2048환경 학습 및 유한 평가/서버 viewer.
- `tokenhsi/learning/multi_agent/push_door_eval.py`: task별 성공·현재 유지·각도·PNG/JSON 평가.
- `tokenhsi/tests/test_push_door_stage1.py`, `smoke_push_door_sim.py`: reward 악용·접촉·phase·AMP·checkpoint·Transformer 순열/gradient CPU 회귀 및 실제 물리 검증.
- `humanoid_ma.py`는 명시적으로 추가된 object DOF를 humanoid 제어에서 분리한다. `amp_network_builder_ma.py`는 task별 실제 object/goal 관계를 만들고, `ma_agent.py`/`ma_players.py`는 전용 checkpoint 계약을 검증한다. 기존 task의 DOF/control/관계 경로를 유지한다.


## Push/Door 랜덤 시작 (ZIP 서버 반영)

- `tokenhsi/utils/push_door_spec.py`의 `sample_start_layout()`: CPU 사용 가능한 독립 lane별 시작 배치 샘플러. `interaction_metadata()`는 랜덤 시작 설정이 있는 경우에만 분포 계약을 추가한다.
- `tokenhsi/env/tasks/multi_agent/humanoid_ma_push_door.py`의 `_reset_envs()`: 선택적 랜덤 시작·상자/목표/문 root·사람 및 loco rigid body/kinematic 변환. 기존 config는 고정 시작을 유지한다.
- `tokenhsi/data/cfg/multi_agent/push_door_stage1_random_start.yaml`, `tokenhsi/data/cfg/train/rlg/amp_ma_push_door_stage1_random_start.yaml`: 별도 실험 설정, 학습 2048환경·기존과 동일한 유지 AMP door_tail.
- `tokenhsi/scripts/multi_agent/push_door_stage1_random_start_{train,test,view,vnc}.sh`: 서버 GPU 6 기본·기존 runtime/MPS 연결. 출력은 `output/push_door_stage1_random_start`와 각 viewer/eval suffix로 분리한다.
- `tokenhsi/tests/test_push_door_stage1.py`: ZIP 랜덤 시작 간격·목표 거리·높이·재샘플링·checkpoint 격리 테스트 소스 포함. 사용자 요청으로 이번 반영에서는 테스트·문법 검사 및 GPU 실행을 하지 않았다.

- 랜덤 시작 Push/Door의 선택적 충돌 패널티는 `humanoid_ma_carry.py`의 `compute_agent_collision_penalty()`를 재사용한다. `push_door_spec.py`는 활성 충돌 보상 설정을 checkpoint 계약에 포함한다. 기존 고정 config는 기존 보상을 유지한다.
