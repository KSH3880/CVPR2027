# 레포 구조

유지하는 컨피그는 `stage1`, `stage2_before_nonemask_rsi433_near50`, `stage2_before`, `stage2_before_rsi433`, `stage1_push`, `stage2_before_push`, `stage2_coupled_push`, `stage2_before_push_near50`, `stage2_coupled_push_goal5`, `stage2_coupled_push_individual_goal`, `stage2_before_push_near50_pair50_source5`, `stage2_coupled_push_individual_goal_solo5`, `stage2_coupled_push_individual_goal_solo5_carry5`, `stage2_coupled_push_individual_goal_solo5_countnorm_coupled`, `stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none`, `stage2_coupled_push_individual_goal_solo5_keep1`, `stage2_coupled_push_individual_goal_solo5_shrink1`, `stage2_coupled_push_individual_goal_solo5_carry5_ground`, `stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg`, `stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg`, `stage2`, `stage2_distill`, `stage2_900`, `stage2_distill_900`, `stage2_sensor_900`, `stage2_sensor_900_warehouse`, `stage2_sensor_900_warehouse_bridge_init`, `stage2_sensor_900_warehouse_bridge_init_rsi`, `stage2_sensor_900_warehouse_safety`, `stage2_sensor_900_warehouse_coop`, `stage2_sensor_900_warehouse_coop_cliffroom`, `stage2_sensor_900_room_warehouse`이다. 실행 명령은 [config](config.md), 설치·추가 옵션은 [USAGE](USAGE.md)에 있다.

## 실행과 설정

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/run.py`, `utils/parse_task.py` | MA 학습/평가·Stage1/BEFORE 모델 등록과 환경 생성 |
| `tokenhsi/utils/config.py`, `experiments.py` | CLI, 스물다섯 실험 계약, 학습/추론/resume env/sim 로딩 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/` | 환경·학습 YAML 각 스물다섯 개 |
| `tokenhsi/scripts/multi_agent/*_{train,test,vnc}.sh` | 실험별 학습·viewer·VNC 진입점 75개 |
| `tokenhsi/scripts/multi_agent/{stage2_distill_900,stage2_sensor_900}_train_vnc.sh` | 4명·1024환경 학습 시각화, noVNC6080 |
| `tokenhsi/scripts/multi_agent/*_eval.sh`, `evaluate.sh` | 일반 정식 평가25개, Stage1/Stage1 PUSH 전용 1명 단독 평가2개·공통 runtime |
| `tokenhsi/scripts/multi_agent/stage2_teachers_{test,vnc,eval}.sh` | enable/coupled 두 PTH의 관계별 제로샷 viewer·공간 채점, 5명 이상 그룹 배치 |
| `tokenhsi/evaluate.py`, `rescore.py` | 독립 seed 실행·통계, 저장 trace CPU 재채점 |
| `tokenhsi/utils/evaluation_protocol.py`, `learning/multi_agent/formal_eval.py` | 학습 인원/단독 구분·점/영역/BEFORE·개인/공유 중심 공동 PUSH·interaction group 성공률과 seed 통계·개인 낙상 제어 |
| `tokenhsi/scripts/multi_agent/launch.sh`, `runtime_env.sh` | 공통 인자·출력 경로·conda·GPU |
| `tokenhsi/scripts/multi_agent/{gui_gpu_env,run-gui,mps_auto_env}.sh` | viewer GPU, VNC, 기존 MPS 연결 |
| `tokenhsi/utils/checkpoint_compat.py` | 기존 checkpoint의 이름·RSI 경로를 현재 표기로 변환 |
| `checkpoints/`, `pth/` | 기본 전이·teacher 체크포인트 |
| `output/`, `output_etc/check/` | 본학습·공용 캐시, 확인 학습·평가·viewer |

## 환경

아래 경로는 `tokenhsi/env/tasks/multi_agent/` 기준이다.

| 파일 | 역할 |
| --- | --- |
| `humanoid_ma.py`, `humanoid_ma_carry.py` | 다중 사람·제어·상태와 공통 환경·물체 생성 |
| `scene_observation.py`, `scene_reset.py`, `reference_motion.py` | 관측, reset/배치, 모션·RSI·AMP |
| `edge_ontop_task.py`, `edge_context_task.py` | 과제 그래프·물리 재시도·보상·진단 |
| `humanoid_ma_push.py`, `jointpush_reward.py` | 1/2명 PUSH 자산·배치·접촉 RSI·CONTACT+AT 보상 |
| `humanoid_ma_before_push.py`, `before_push_reward.py` | BEFORE solo PUSH 자산·배치·접촉 보상, 기존 BEFORE gate 유지 |
| `humanoid_ma_stage2.py`, `stage2_layout.py`, `stage2_reward.py`, `stage2_wait.py` | 통합 Stage2의 관계별 배치·AT/접촉/선후행 gate·대기 속도 감점·V4 AT 완화/정밀 포화 분리 |
| `humanoid_ma_coupled_push.py`, `coupled_push_reward.py` | 공동 상자·가상 구역·그룹 gate·goal saturation |
| `humanoid_ma_sensor_stage2.py`, `sensor_stage2_{geometry,placement,observation}.py` | Stage2 센서 시나리오·고정 bridge 자산·BEFORE/COUPLED 배치·관측·진단 |
| `warehouse_sensor.py` | 창고 RSI 50/25/25·고정 벽/입구·단독/공동 PUSH 배치·열림/진입 진단 |
| `square_sensor.py`, `square_sensor_scene.py` | 기준 repo의 48×48 높이 센서·cliff/bridge/room 기하 |
| `push_goal_markers.py` | PUSH 목표의 viewer/VNC 선 표시만 추가; actor·reset·scale 수정 없음 |
| `humanoid_ma_before.py` | BEFORE 역할·공유 물체·AT progress·후행 초기화 |
| `scene_layout.py`, `scene_validation.py` | 활성 물체·distractor·loco 참조·물리 유효성 |
| `before_rsi_cache.py`, `native_rsi_cache.py` | 후행 pickUp/carryWith native 크기별 screening |
| `before_pickup_carry_rsi.py`, `before_relative_sampling.py` | RSI433 적용/fallback, near50 배치 |
| `size_rsi_cache.py` | Stage1 크기별 RSI 캐시 |
| `*_reward.py` | 과제 보상·BEFORE gate·CPA·물체 속도 |

## 정책과 과제 정의

- `tokenhsi/tests/{test_bridge_training_start,check_bridge_training_start}.py`: 네 장면5/40/25/30·Bridge20/40/40 초기화·Warehouse 회귀·실제 actor/센서/부분 reset 및 초기 화면 검증.

- `tokenhsi/learning/multi_agent/sensor_stage2_{network,agent}.py`: actor encoder/coordination·sigma·입력 RMS 동결; 센서·action head·critic 전체·AMP 및 AMP/value RMS 학습, resume 학습 계약 검사. 두 센서 실험의 공통 구현.
- `tokenhsi/utils/warehouse_sensor.py`: 창고 역할 샘플링·거리/충돌 조건부 GPU 배치·입구 기하.
- `tokenhsi/tests/{test_warehouse_sensor,check_warehouse_sensor,capture_warehouse_sensor}.py`: 창고 CPU 계약·실제 reset·닫힘/열림 화면 검수.
- `tokenhsi/tests/test_room_warehouse.py`: Room/Warehouse 50:50·벽 CPA 계수 분리·기존 설정 보존·checkpoint 계약 검사.
- `tokenhsi/tests/test_room_warehouse_wallterm.py`: 벽 접촉 종료 config 차이·형상 기반 접촉 판정·연속 스텝 종료/초기화 검사.
- `tokenhsi/tests/test_room_warehouse_boxterm.py`: Warehouse PUSH 상자 CPA·접촉 종료의 비 PUSH 담당자 한정·상대속도·연속 스텝 검사.
- `tokenhsi/tests/test_room_warehouse_v4_reward.py`: V4 보상 config 차이·AT 성공/정밀 포화 높이 분리·payload 대기·공동 PUSH 조기 이동 감점 검사.
- `tokenhsi/tests/test_warehouse_fixedroles_v4.py`: 역할 고정 Warehouse config 차이·PUSH 2명+AT 선행/후행 역할·방 안쪽 5m 배치 검사.
- `tokenhsi/tests/test_warehouse_wall_box_cpa.py`: Warehouse 벽 CPA `wall_cpa: box` 옵션의 방 안 선분 동일성·바깥 면/문틀 대칭·설정 검증.
- `tokenhsi/env/tasks/multi_agent/warehouse_done.py`: 학습 전용 선행 완료 시작(`leader_done_probability`)의 샘플링·선행 상자/선행 재배치·모드별 성공 진단. `tests/test_warehouse_fixedroles_v4_leaderdone_wallbox.py`·`tests/check_warehouse_leaderdone.py`에서 CPU 계약과 실제 1024환경 배치를 검사한다.
- `tokenhsi/env/tasks/multi_agent/warehouse_before.py`: Warehouse BEFORE 2팀 변형(`before_pairs_probability`)의 역할 샘플링·apron 배치(`_place_group`의 `region`)·남는 상자 치우기·변형별 성공 진단. `tests/test_warehouse_fixedroles_v4_beforepairs.py`·`tests/check_warehouse_beforepairs.py`에서 CPU 계약과 실제 1024환경 배치를 검사한다.
- `tokenhsi/tests/test_warehouse_before_targets.py`, `test_warehouse_evaluate_entry.py`: Default 후행 중립 Target의 좌표·부분 reset·활성 목표/보상 보존, 공유 상자 소유자/초기화 목록 복원, doorprogress 평가 진입점 회귀 검사.
- `tokenhsi/tests/test_sensor_stage2_v4_gate.py`: 실제 센서 과제 샘플러의 AT 높이 완화 연결·부분 갱신·후행 해제와 정밀 보상 분리, 비 V4 경로 보존 검사.
- `tokenhsi/env/tasks/multi_agent/warehouse_before_cache.py`: 학습 전용 BEFORE apron 검증 배치·역할/그래프·자세/AMP 이력 캐시와 Default 초기화의 첫 rollout 전 캐시 준비. `tests/test_warehouse_before_cache.py`·`tests/check_warehouse_before_cache.py`에서 조건 일치·복원·충돌·reset 시간을 검사한다.
- `tokenhsi/env/tasks/multi_agent/square_sensor_scene.py`의 `place_rigid_batched`: apron 강체 배치 후보의 일괄 검사·불가능한 기하 조기 탈락. `tests/test_warehouse_reset_speedup.py`에서 경계·충돌 조건과 불가능한 그룹을 검사한다.
- `tokenhsi/env/tasks/multi_agent/warehouse_door_progress.py`: 2 PUSH + BEFORE 1쌍에서 선행 root의 문 입구 XY progress와 통과 후 포화. `tests/test_warehouse_door_progress.py`에서 적용 대상·좌표계·포화를 검사한다.
- `tokenhsi/qualitative/warehouse_waypoint{,_policy,_task,_viewer}.py`: 원본 DistillV4 추론 전용 입구 가상 상자/root 거리 전환·집기 후 입구 가상 목표/payload 거리 복귀·선행 행동 교체·뒤쪽 좌우 스폰·선행 상자 벽 여유 1.2m·탑뷰 viewer. `scripts/multi_agent/warehouse_waypoint_{env,test,vnc}.sh`가 GPU5/6/7·MPS/직접 CUDA와 실행을 담당한다. `tests/test_warehouse_waypoint.py`·`check_warehouse_waypoint.py`에서 관측 보존·스폰·실제 정책 출력을 검증한다.
- 같은 추론 viewer의 `--follower-ontop-waypoint`는 ON_TOP 후행의 진입용 상자와 운반용 목표를 입구에 두는 GTA 장면을 추가한다. `--record`로 한 에피소드를 MP4로 저장한다.
- `warehouse_waypoint_task.py`와 `warehouse_waypoint_network.py`는 1.2m 선행·ON_TOP 후행의 단계 저장, 역할별 GTA 배치 인코딩과 기존 1.3m 선행 전용 관측을 지원한다. 실행 계약은 [waypoint 학습 명세](reference/warehouse_waypoint_training.md#12m-선행on_top-후행)를 따른다.
- `tokenhsi/utils/sensor_stage2_spec.py`: 4명·8토큰 센서 계약, bridge 1/2/3명 PUSH·crosser·나머지 사람 샘플링.
- `tokenhsi/tests/{test_sensor_stage2,check_sensor_stage2}.py`: CPU 전이/동결/샘플링·실제 1024환경 배치/센서 검사.

- `tokenhsi/learning/multi_agent/amp_network_builder_ma.py`, `task_role_encoder.py`: 공통 GTA·task bias·사람 action MLP·actor/critic.
- 같은 디렉터리의 `before_network.py`, `human_coordination.py`: BEFORE 사람 attention·object packet·Stage1 전이·actor encoder 동결.
- `stage2_push_network.py`: BEFORE PUSH와 coupled의 동일 5과제/4family 정책·기본6N task/group suffix(keep1/shrink1은 선택 상대 ID N개를 더한7N)·4종 사람 relation·packet 기반 alias 전처리·Stage1 PUSH 전이. `human_coordination.py`는 선택한 관계의 query별 유효 key 수로 log-count 보정하며 `test_human_attention_count_normalization.py`에서 관계 비중·선호 보존·mask·전이/계약을 검증한다. `before_push_network.py`는 같은 builder의 호환 import다.
- `stage2_relation_fusion.py`, `utils/stage2_spec.py`, `utils/before_goal_sampling.py`: BEFORE/COUPLED 독립 임베딩과 공통 fusion MLP·세 장면 유형 1:1:1·충돌 없는 alias 배정·goalroute 배치. `test_stage2.py`에서 관계·토큰·보상·전이·gradient·재로드를 검사한다.
- `stage2_teachers.py`, `before_relation_fusion.py`: 원본 teacher별 네트워크·RMS 복원과 사람별 행동 선택. `utils/stage2_group_sampling.py`, `env/tasks/multi_agent/stage2_group_layout.py`는 추론 전용 순차 그룹 선택·5m 중심 배치·BEFORE 중심 복원·distractor 제외·카메라를 담당한다. `test_stage2_teachers.py`, `tests/check_stage2.py`에서 가중치/담당 선택·관측 제외·물리/부분 reset을 검사한다.
- `coupled_relation_fusion.py`: ground A/B의 COUPLED 64차원 공유 임베딩·레이어/헤드 bias 투영·output projection 전 residual fusion. SELF/NONE만 A는 lookup 학습, B는 0 고정이다. `test_coupled_relation_fusion.py`에서 계산 동치·단독 생략·인원/셔플·gradient·전이/재로드를 검사한다.
- `stage2_distill_agent.py`, `distill_replay.py`, `utils/stage2_distill_spec.py`: 고정 두 teacher의 사람별 평균 행동 모방·scene 단위 teacher/student 실행 전환·누적 replay·actor coordination/action head 업데이트·체크포인트/optimizer/replay 복원. [distillation 명세](reference/stage2_distill.md).
- `viewer_diagnostics.py`: viewer 전용 native 보상 성공 진단, 정식 평가와 별도.
- `ma_agent.py`, `ma_players.py`, `scene_normalizer.py`: 장면 단위 PPO/AMP·rollout 경계 done 처리·저장/복원·평가·정규화.
- `tokenhsi/utils/{edge_*,task_role_spec,before_spec}.py`: 그래프·역할·packet·checkpoint 계약.
- `tokenhsi/utils/{coupled_push_spec,coupled_push_scene}.py`: 공동 그룹·인원별 물리 bank·직선 목표·충돌 검사.임의 N≥1·M≥N과 물체 부족 장면의 과제 재샘플링을 지원한다.
- `tokenhsi/utils/coupled_push_contact.py`: 개인 목표 실험의 공통 4면 선택·담당 구역 CONTACT·개인 도착 gate.
- `tokenhsi/utils/premature_push.py`: V4 보상의 공동 PUSH 준비 전 상자 이동 감점(손 반력 기여율 배분).
- `tokenhsi/utils/coupled_push_peer_selection.py`: viewer의 keep3/shrink3과 학습·추론의 keep1/shrink1 접촉 기반 상대 선택·독립 RNG·부분 reset. 선택되지 않은 COUPLED는 NONE으로 바꾸며, `test_coupled_peer_selection.py`·`test_single_coupled_peer.py`가 관계 전환·셔플·PPO 선택 재사용·전이를 검사한다.
- `tokenhsi/utils/coupled_push_assignment.py`: viewer 전용 좌우 정렬·Hungarian·교차 우선 구역 배정. reset 후 개인 목표/관측/접촉에 같은 배정을 연결한다. `test_coupled_assignment.py`는 배정 목적함수·회전/그룹·접촉/목표 연결·9명 이상 근사 탐색을 검사한다.
- `tokenhsi/utils/{before_push_spec,before_push_layout}.py`: solo PUSH 샘플링·계약·다중 사람 경로 배치.
- `tokenhsi/utils/{before_sampling,before_relative_sampling,before_object_alias}.py`: mix226/near50·토큰 alias.
- `tokenhsi/learning/multi_agent/stage1_push_network.py`: 5과제 정책·4 family AMP·선택적78k 전이.
- `tokenhsi/utils/{stage1_jointpush,jointpush_scene,jointpush_geometry,jointpush_rsi,jointpush_rsi_geometry}.py`: PUSH 계약·1/2명 장면·bank/모션/기하 검증.
- `tokenhsi/utils/stage1_sampling.py`: 기본 목표 정사각형 샘플링과 carry5 전용 상자 기준 거리/방향 샘플링.
- `tokenhsi/utils/{stage1_geometry,stage1_config,stage1_platforms,size_rsi}.py`: Stage1/BEFORE 공통 기하·RSI·물리 설정과 coupled carry5_ground 지면 배치 계약.
- `lpanlib/poselib/`: 기존 모션 배열 로딩 라이브러리.

## 검증과 문서

`tokenhsi/tests/`는 보상·배치·GTA·mask·RSI·전이·재로드·출력 경로를 검사한다. `before_test_utils.py`, `stage1_test_utils.py`는 CPU fixture다. `check_before_object_alias.py`·`check_before_push.py`·`check_coupled_push.py`는 실제 물리·부분 reset·추론 검사 진입점이다.

`test_coupled_scene_sampling.py`는 비PUSH 사람의 경로 장애물 유지·공동 상자의 전체 크기·distractor 재시도와 ON_TOP 받침 보존을 검사한다.

[Stage2 sensor](reference/stage2_sensor_900.md), [통합 Stage2](reference/stage2.md), [PUSH](reference/stage1_push.md), [BEFORE PUSH](reference/stage2_before_push.md), [공통 Stage2/coupled](reference/stage2_coupled_push.md), [4명 초과 그룹 스폰](reference/group_spawn_over4.md), [평가 규약](reference/evaluation.md), [Stage1 단독](../evaluation_single.md), [공통 정책](reference/ma_clean_scene_gta.md), [BEFORE](reference/stage2_before.md), [지표](reference/relation_diagnostics.md), [이관](reference/PORTING_GUIDE.md)을 참고한다. 이전 실험은 [legacy](legacy/README.md)에 보존한다.

## Warehouse safety 진입점

- `tokenhsi/utils/sensor_safety_spec.py`: Cliff/Room 역할 비율 및 기존 Warehouse RSI→safety의 제한된 학습 resume 계약.
- `tokenhsi/utils/bridge_curriculum.py`, `env/tasks/multi_agent/sensor_stage2_curriculum.py`: 길이별 schedule·다중 crosser 초기화·횡단 노출/진단.
- `tokenhsi/learning/multi_agent/sensor_stage2_agent.py`: 기존 optimizer/RMS 복원, 신규 curriculum 시작 epoch 저장/복원.
- `tokenhsi/tests/test_sensor_safety.py`: 승인한 config 차이·역할 노출·토큰 budget·resume 제한. 실제 검증은 `output_etc/check/warehouse_safety_20261109/`.
- `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage2_sensor_900_warehouse_safety.yaml`, `scripts/multi_agent/stage2_sensor_900_warehouse_safety_{train,train_vnc,test,vnc,eval}.sh`: 전용 실행 설정과 wrapper.

`tokenhsi/tests/test_ma_rollout_reset.py`는 실제 play_steps/wrapper의 rollout 경계 reset·초기 done·보상 중복 방지·개인 terminate bootstrap을 검사한다.

## Warehouse 공동 PUSH 진입점

- `output_etc/check/warehouse_five_before_ontop_6000_20261113/`: 5명/10토큰·2 PUSH/BEFORE ON_TOP1쌍/독립 AT 평가 helper·6000/6500epoch reset 탐색·기하 선별·Blender/영상 저장 서비스. CPU PhysX/GPU 정책 추론이며 본학습 변경 없이 진행 중이다.
- `output_etc/check/warehouse_seven_before_ontop_20261113/`: 7명 CPU reset/탐색 helper·6000epoch 감시·[중간 후보](../output_etc/check/warehouse_seven_before_ontop_20261113/previews/candidate_01/index.html). 중간 후보는 충돌/후행 대기로 최종 선별 탈락이며, 후속 탐색은 사용자 승인5명 구성으로 전환했다.
- `output_etc/check/warehouse_outside_det4500_20261113/`: RoomWarehouse4500epoch·deterministic·선후행 외부 출발5장면(닫힘2/부분 RSI3), 관절/장면/Blender·30fps 영상과 native CLIMB/양발 지지 판정 차이·[모아보기](../output_etc/check/warehouse_outside_det4500_20261113/index.html)·[조건·재사용](../output_etc/check/warehouse_outside_det4500_20261113/report.md).
- `output_etc/check/warehouse_push2_carry_climb_20261113/`: RoomWarehouse4000epoch·5504episode 탐색·2 PUSH/실내 반출/후행 보행·대기·CLIMB 성공 영상5편, 관절/장면/Blender 묶음·[모아보기](../output_etc/check/warehouse_push2_carry_climb_20261113/index.html)·[조건·재사용](../output_etc/check/warehouse_push2_carry_climb_20261113/report.md).
- `output_etc/check/warehouse_self_walk_20261112/`: 후행 비접촉 보행 검사·576episode 재탐색·영상5편과 관절/장면/Blender 묶음, [조건·검증](../output_etc/check/warehouse_self_walk_20261112/report.md).
- `output_etc/check/warehouse_showcase_20261112/`: 5500epoch Warehouse 선택 영상3개와 후행 활동 추가 영상, 관절/장면/Blender 묶음·[재사용](../output_etc/check/warehouse_showcase_20261112/report.md)·[후행 활동 비교](../output_etc/check/warehouse_showcase_20261112/follower_activity/report.md).
- `tokenhsi/utils/warehouse_coop.py`: 1~3명 역할·네 방향·RSI 배치와 이동 경로 간섭 검사.
- `tokenhsi/env/tasks/multi_agent/warehouse_coop.py`: 방향/인원별 입구 기하 선택·열림 판정·실제 작업자 분모 진단.
- `tokenhsi/tests/{test_warehouse_coop,test_warehouse_room_geometry,check_warehouse_coop}.py`: CPU 계약·36조합·Room 기하 참조 일치 및 실제 1024환경 reset·센서·화면 검사.
- `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage2_sensor_900_warehouse_coop.yaml`, `scripts/multi_agent/stage2_sensor_900_warehouse_coop_{train,train_vnc,test,vnc,eval}.sh`: Safety를 기반으로 한 별도 실행 설정.
- `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage2_sensor_900_warehouse_coop_cliffroom.yaml`, `scripts/multi_agent/stage2_sensor_900_warehouse_coop_cliffroom_{train,train_vnc,test,vnc,eval}.sh`: coop 기반 Cliff·Room 샘플링 변경 실행 설정. 코드 변경은 실험 이름 등록뿐이다.

- `tokenhsi/env/tasks/multi_agent/warehouse_waypoint_task.py`·`warehouse_waypoint_cache.py`: 80% PUSH+BEFORE 선행의 고정 1.3m 단계 전환·PPO 관측 suffix, 두 변형의 GPU 배치 캐시·초기 준비. `warehouse_waypoint_placement.py`는 최적화 전 배치의 성능 대조용이다.
- `tokenhsi/learning/multi_agent/warehouse_waypoint_network.py`·`utils/warehouse_waypoint.py`: 저장 관측으로 실제/선행용 GTA를 독립 batch 행에 묶고 출력 담당을 분리한다. 동결 actor 특징을 PPO mini-epoch 간 재사용하며 센서 분기는 공유한다. `tests/test_warehouse_waypoint_training.py`·`check_warehouse_waypoint_training.py`가 정책·캐시·1024환경을 검증한다.
