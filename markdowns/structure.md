# 레포 구조

현재 실행 실험은 원본 multi-agent carry(1), 독립 Stage 1(27·28·30·31·32·33·34·35 및 34 distill), 협력 Stage 2(29 및 SIT plane 변형), 무대 센서 Stage 2, 두 무대·연결다리 Stage 2, 4면 벽 room Stage 2다. 실행 명령과 checkpoint 규칙은 [config.md](config.md)에 있다.

## 진입점

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/run.py` | 학습·평가 태스크와 알고리즘 등록 |
| `tokenhsi/scripts/multi_agent/` | 현재 실험별 train/test/VNC, `runtime_env.sh`, `run-gui.sh` |
| `tokenhsi/data/cfg/multi_agent/` | 현재 실험 YAML과 평가 graph 예제 |
| `tokenhsi/data/cfg/train/rlg/` | PPO/AMP/Transformer 학습 설정 |
| `output/<실험>/<run>/` | checkpoint `nn/`, TensorBoard `summaries/`, 진단 `diagnostics/` |
| [stage2_cliff_sensor_experiment_report_20260929.md](stage2_cliff_sensor_experiment_report_20260929.md) | 무대 센서 실험의 동기·구성·학습 결과·직접 시각 평가·ablation 분석 |
| `markdowns/assets/cliff_sensor_20260929/` | 보고서의 실제 화면과 학습·이동 궤적 그래프 |
| [room_scenario_report_20260930.md](room_scenario_report_20260930.md), `markdowns/assets/room_20260930/` | 4면 벽 room의 실제 Isaac Gym 화면·배치·센서·CPA·물리 및 학습 실행 검증 |
| [room_partition_report_20260930.md](room_partition_report_20260930.md), `markdowns/assets/room_partition_20260930/` | 내부 칸막이·HOLDING_AT room의 실제 화면·목표 binding·통로·물리/학습 검증 |

## 현재 실험이 사용하는 코드

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py` | 물체 배정·RSI·물리 환경·관측 통합 |
| `tokenhsi/env/tasks/multi_agent/edge_ontop_task.py`, `edge_context_task.py` | graph reset, 가까운 시작, 물리 타당성·진단 |
| `tokenhsi/env/tasks/multi_agent/edge_stage1_reward.py` | Stage 1 edge 보상·포화·팀 공유 |
| `tokenhsi/env/tasks/multi_agent/edge_ontop_reward.py`, `edge_interaction_reward.py` | AT·ON_TOP·SIT·CLIMB state/progress와 plane 성공 |
| `tokenhsi/utils/edge_scenario_spec.py` | 독립 물체·goal binding과 과제 sampler |
| `tokenhsi/utils/edge_stage1_spec.py` | Stage 1 variant·semantic packet·후반 CLIMB RSI 검증 |
| `tokenhsi/utils/edge_stage2_spec.py` | Stage 2 협력 graph·평가 preset |
| `tokenhsi/utils/relation_task_spec.py` | reward config·checkpoint 계약 검증 |
| `tokenhsi/learning/multi_agent/amp_network_builder_ma.py`, `ma_agent.py` | GTA actor/critic, AMP·PPO 학습 |
| `tokenhsi/learning/multi_agent/coordination_head.py`, `stage2_transfer.py` | 협력 attention·Stage 1 weight 이식 |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_cliff.py`, `cliff_geometry.py` | 센서 Stage 2의 무대 생성·좌표 변환·RSI·지형 높이·낙상·진단; `sensor_pattern: square`는 root 중심·heading 회전 정사각형 격자로 부채꼴의 수직 조회 경로 재사용 |
| `tokenhsi/env/tasks/multi_agent/vision_sensor.py`, `vision_sensor_debug.py` | 균일 고리·100도 부채꼴 표면 높이·OBB 가림·고리 연결 채움, 가림 없는 수직 조회(`vertical_height`) 및 관측 캐시 기반 viewer 표시(부채꼴 호·정사각형 외곽선) |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_bridge.py`, `bridge_geometry.py` | 두 무대·연결다리 Stage 2: 무대 B·다리 actor, 5:5 장면 샘플링, 간격 쪽 배치, 로그균등 도착 일정, quintic 궤적 PD 힘, 지형·센서 확장; root 상대 XY 위치·속도의 CPA 위험도로 단일 `collision` 보상 계산 |
| `tokenhsi/tests/test_bridge_scenario.py`, `bridge_physics_smoke.py` | 수직 조회·배치·도착 일정 CPU 검증; 다리 도착·유격·하중·간격 바닥 실패·상자 배치 PhysX 검증과 좌표계 검수(독립 NumPy 센서 재계산, GTA 자세, heading, 지면 높이, 1 step 후 시작 위치) |
| `tokenhsi/tests/test_bridge_collision.py` | CPA 정면 접근·안전한 엇갈림·근거리 이탈·정지·동일 속도·self 제외와 실제 relation 총 보상·로그 연결 CPU 검증 |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_room.py`, `room_geometry.py` | 4면 벽 room: 실내 사람·객체 배치, 고정 벽 actor·센서 표면, 벽 최근접점을 정지 가상 에이전트로 둔 사람 CPA 재사용(벽 전용 할인)과 사람 CPA를 단일 collision 항으로 통합 |
| `tokenhsi/data/cfg/multi_agent/approach_stage2_fan_room_partition.yaml`, `tokenhsi/tests/test_room_partition.py` | 같은 room 환경에 내부 칸막이·반대편 HOLDING_AT 목표·goal slot별 표시 추가; 4개 과제 각 25%, HOLDING/AT 픽업 반대편 시작 50%, 배치·CPA·goal 높이/binding CPU 검사 |
| `tokenhsi/tests/test_room_scenario.py`, `room_physics_smoke.py`, `room_capture.py` | 벽 CPA·배치·센서·종료 CPU 검사; PhysX 배치/접촉/보상 연결 검사; 실제 viewer PNG·센서 NPZ 캡처 |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_mixed_sensor.py`, `mixed_sensor_geometry.py` | 세 고정 공간의 reset별 1/3 혼합(평가 전용 `SCENARIO` 장면 고정)·활성 원점, 네 과제 각 25%, room 픽업 반대편 50%, 실제 발 메시 기준 RSI 높이 보정·센서·CPA·AT 목표 |
| `tokenhsi/tests/test_mixed_sensor.py`, `mixed_sensor_physics_smoke.py`, `mixed_reset_physics_smoke.py` | 혼합 확률·부분 reset·좌표/목표·고정 actor·정사각형 센서 root 중심/heading 회전·PhysX 지지/칸막이 충돌, RSI 자세 유지/실제 checkpoint의 리셋 직후 튕김 회귀 검사 |
| `tokenhsi/data/cfg/multi_agent/approach_stage2_square_mixed.yaml`, `tokenhsi/scripts/multi_agent/approach_stage2_square_mixed_{train,test,vnc,watch,watch_vnc}.sh` | 3종 혼합과 같은 장면·과제·보상에 root 중심 14×14m·48×48 정사각형 센서를 쓰는 별도 실험(sensor metadata 8) |
| `tokenhsi/tests/test_vision_sensor.py`, `fan_sensor_capture.py` | 부채꼴 좌표/가림·정사각형 viewer 외곽선 CPU 검증, 5개 통제 장면의 실제 viewer PNG·센서 NPZ 저장 |
| [fan_sensor_commands.md](fan_sensor_commands.md) | 두 무대·다리 실험과 576개·100도·10m 센서 실험의 본학습·watch·테스트 시각화·resume 명령 |
| [mixed_sensor_commands.md](mixed_sensor_commands.md) | 현재 3종 혼합 실험의 학습·테스트·학습/테스트 시각화 명령만 모은 파일 |
| `tokenhsi/scripts/multi_agent/approach_stage2_fan_mixed_watch.sh`, `approach_stage2_fan_mixed_watch_vnc.sh` | 기본 4환경·기존 작은 배치 watch config를 사용하는 로컬/서버 학습 viewer, 별도 watch output |
| [fan_sensor.md](fan_sensor.md) | 부채꼴 관측 계약, 미관측 특수값, 향후 carryWith에서 상자 제외 요구 |
| `tokenhsi/learning/multi_agent/cliff_sensor.py` | 센서 encoder·64→128 head 전이·checkpoint 계약 |
| `tokenhsi/scripts/multi_agent/approach_stage2_cliff_sensor_watch.sh`, `approach_stage2_cliff_sensor_watch_vnc.sh` | 센서 Stage 2 소수 환경 학습의 로컬/noVNC viewer |
| `tokenhsi/scripts/multi_agent/approach_stage2_fan_sensor_watch.sh`, `approach_stage2_fan_sensor_watch_vnc.sh` | 부채꼴 센서 소수 환경의 실제 학습·로컬/noVNC viewer; 기존 작은 배치 config 공유 |
| `tokenhsi/data/cfg/train/rlg/amp_ma_cliff_sensor_watch.yaml` | watch 전용 작은 PPO/AMP minibatch·buffer; 본학습 config와 분리 |
| `tokenhsi/tests/test_cliff_sensor.py`, `test_cliff_coordinates.py`, `cliff_physics_smoke.py` | 센서 전이/gradient·좌표계 CPU 검증 및 PhysX 지지·낙상·높이 probe 검증 |
| `tokenhsi/learning/multi_agent/stage1_unified_teacher.py`, `distillation.py` | 34번 distillation의 원본 통합 teacher 관찰·분포와 KL/gradient 검사 |
| `tokenhsi/tests/test_scenario_independent_with_climb.py`, `test_scenario_stage1_plane.py`, `test_scenario_stage1_sit_plane_curriculum.py`, `test_stage2_coordination.py` | 현재 실험 CPU 검증 |

`edge_context_spec.py`, `edge_ontop_spec.py`, `edge_interaction_spec.py`와 관련 reward 함수는 과거에 도입됐지만 현재 Stage 1·2의 graph·관찰·보상 계산에서도 호출된다. 파일명만 보고 삭제하면 현재 실험이 깨진다.

## 작업별 확인

- **Stage 1 graph·보상:** 해당 YAML → `edge_scenario_spec.py` → `edge_ontop_task.py` → `edge_interaction_reward.py`·`edge_stage1_reward.py`; 위 Stage 1 테스트로 검증한다.
- **Stage 2 협력·전이:** `approach_stage2_coordination.yaml` → `edge_stage2_spec.py` → `coordination_head.py`·`stage2_transfer.py`; `test_stage2_coordination.py`로 검증한다.
- **Stage 2 센서·무대:** `approach_stage2_cliff_sensor.yaml` → `humanoid_ma_cliff.py`·`cliff_geometry.py` → `cliff_sensor.py`·`amp_network_builder_ma.py`. 기존 schema-9 reward와 별도의 `sensor_metadata`를 사용한다. 평가의 `SENSOR_MASK=1`은 `ma_players.py`에서 정책 센서 특징만 0으로 마스킹한다. 무대 생성 world 좌표와 센서/종료의 terrain 조회 좌표를 함께 검증한다.
- **Stage 2 두 무대·다리:** `approach_stage2_fan_bridge.yaml` → `humanoid_ma_bridge.py`·`bridge_geometry.py` → `humanoid_ma_cliff.py`의 `_sample_scene`·`_sensor_boxes`·`_terrain_height` hook과 carry의 `_extra_reset_actor_ids`. `test_bridge_scenario.py`와 `bridge_physics_smoke.py`로 검증한다.
- **Stage 2 room:** `approach_stage2_fan_room.yaml`(외벽만, metadata 5), `approach_stage2_fan_room_partition.yaml`(칸막이·HOLDING_AT, metadata 6) → `humanoid_ma_room.py`·`room_geometry.py`; 각 전용 train/test/VNC 스크립트. cliff 센서·reset 경로와 bridge의 사람 CPA 함수를 재사용한다. HOLDING_AT은 partition에서 허용하며 goal marker는 graph의 목적지 slot을 따른다.
- **Stage 2 3종 혼합:** `approach_stage2_fan_mixed.yaml` → `humanoid_ma_mixed_sensor.py`·`mixed_sensor_geometry.py`; 전용 `approach_stage2_fan_mixed_{train,test,vnc}.sh`, sensor metadata 7·physics metadata 1. 실제 고정 지형은 생성 위치에 유지하고 학습/평가 모두 활성 공간을 재샘플링한다. 이동 다리만 동적 지형이며, RSI 발 겹침은 초기 높이로 보정한다. `edge_context_eval.py`는 최초 trial 장면과 과제를 보존해 장면별 성공/종료를 집계한다.
- **Stage 2 3종 혼합 · 정사각형 센서:** `approach_stage2_square_mixed.yaml` → 같은 `humanoid_ma_mixed_sensor.py`; 센서 차이는 `humanoid_ma_cliff.py`의 `_square_sensor` 분기(격자·root 원점)와 `vision_sensor_debug.py`의 외곽선뿐이다. sensor metadata 8.
- **34번 distillation:** `approach_scenario_stage1_self_sum_distill.yaml` → `stage1_unified_teacher.py` → `ma_agent.py`; `test_stage1_unified_distillation.py`로 관찰·label 정렬을 검증한다.
- **실행·GPU:** [config.md](config.md), `runtime_env.sh`, `mps/`, `run-gui.sh`를 확인한다.
- **지표 해석:** [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 확인한다.
