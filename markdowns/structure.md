# 레포 구조

유지하는 컨피그는 `stage1`, `stage2_before_nonemask_rsi433_near50`, `stage2_before`, `stage2_before_rsi433`, `stage1_push`, `stage2_before_push`, `stage2_coupled_push`, `stage2_before_push_near50`, `stage2_coupled_push_goal5`, `stage2_coupled_push_individual_goal`, `stage2_before_push_near50_pair50_source5`, `stage2_coupled_push_individual_goal_solo5`, `stage2_coupled_push_individual_goal_solo5_carry5`, `stage2_coupled_push_individual_goal_solo5_countnorm_coupled`, `stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none`, `stage2_coupled_push_individual_goal_solo5_keep1`, `stage2_coupled_push_individual_goal_solo5_shrink1`, `stage2_coupled_push_individual_goal_solo5_carry5_ground`, `stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg`, `stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg`, `stage2`, `stage2_distill`, `stage2_900`, `stage2_distill_900`, `stage2_sensor_900`, `stage2_sensor_900_bridge_init`, `stage2_sensor_900_bridge_curriculum`, `stage2_sensor_900_room_before`, `stage2_sensor_900_cliff_bridge`, `stage2_sensor_900_cliff_bridge_surface`, `stage2_sensor_900_cliff_bridge_compact`, `stage2_sensor_900_bridge_before_fixed`, `stage2_sensor_900_bridge_before_fixed_carry_rsi`, `stage2_sensor_900_bridge_before_fixed_layout_cache`이다. 실행 명령은 [config](config.md), 설치·추가 옵션은 [USAGE](USAGE.md)에 있다.

## 실행과 설정

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/run.py`, `utils/parse_task.py` | MA 학습/평가·Stage1/BEFORE 모델 등록과 환경 생성 |
| `tokenhsi/utils/config.py`, `experiments.py` | CLI, 실험 계약, 학습/추론/resume env/sim 로딩 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/` | 환경·학습 YAML 각 서른한 개 |
| `tokenhsi/scripts/multi_agent/*_{train,test,vnc}.sh` | 실험별 학습·viewer·VNC 진입점 93개 |
| `tokenhsi/scripts/multi_agent/{stage2_distill_900,stage2_sensor_900}_train_vnc.sh` | 4명·1024환경 학습 시각화, noVNC6080 |
| `tokenhsi/scripts/multi_agent/*_eval.sh`, `evaluate.sh` | 일반 정식 평가31개, Stage1/Stage1 PUSH 전용 1명 단독 평가2개·공통 runtime |
| `tokenhsi/scripts/multi_agent/stage2_teachers_{test,vnc,eval}.sh` | enable/coupled 두 PTH의 관계별 제로샷 viewer·공간 채점, 5명 이상 그룹 배치 |
| `tokenhsi/evaluate.py`, `rescore.py` | 독립 seed 실행·통계, 저장 trace CPU 재채점 |
| `tokenhsi/utils/evaluation_protocol.py`, `learning/multi_agent/formal_eval.py` | 학습 인원/단독 구분·점/영역/BEFORE·개인/공유 중심 공동 PUSH·interaction group 성공률과 seed 통계·개인 낙상 제어 |
| `tokenhsi/scripts/multi_agent/launch.sh`, `runtime_env.sh` | 공통 인자·출력 경로·conda·GPU |
| `tokenhsi/qualitative/`, `scripts/multi_agent/bridge_before_qual.sh` | 추론 전용 5명 Bridge BEFORE 정성 장면 탐색(런타임 패치)·순위/미리보기 렌더 |
| `tokenhsi/scripts/multi_agent/{gui_gpu_env,run-gui,mps_auto_env}.sh` | viewer GPU, VNC, 기존 MPS 연결 |
| `tokenhsi/utils/checkpoint_compat.py` | 기존 checkpoint의 이름·RSI 경로를 현재 표기로 변환 |
| `checkpoints/` | 외부 checkpoint 심링크 |
| `output/`, `output_etc/check/` | 본학습·공용 캐시, 확인 학습·평가·viewer |
| `output_etc/check/bridge_enable_push2_20261111/` | 추론 전용 공동 PUSH2명·AT→SIT 배치 탐색, 실제 관절/장면 내보내기와 Blender 영상; [재사용](../output_etc/check/bridge_enable_push2_20261111/README.md) |
| `output_etc/check/bridge_enable_push3_cap{,04}_20261111/` | 5명·공동 PUSH3·AT→SIT, 다리 상면 마찰1.0/0.4의 분리된 추론·선별·관절/Blender 내보내기 |
| `output_etc/check/bridge_push3_stability_20261111/` | 동일 초기 상태의 마찰10종 비교·상면0.2 안정 횡단/AT→SIT10개·전체/확대 영상·관절/scene/Blender·ZIP |
| `output_etc/check/bridge_wait_resume_20261112/` | 추론 초기 PUSH6–9m 후퇴·AT 정지/다리 도착/재출발 탐색,128개 전체 관절/scene 보관·CPU 재분석; [미확보 결과](../output_etc/check/bridge_wait_resume_20261112/README.md) |

## 환경

- `humanoid_ma_sensor_compact.py`, `compact_geometry.py`, `compact_rsi.py`, `utils/compact_bridge.py`: 인원별 소형 무대·5cm 접합 틈·거리 커리큘럼·보행 경계 RSI. [명세](reference/stage2_sensor_900_cliff_bridge_compact.md).
- `humanoid_ma_sensor_bridge_before.py`, `utils/bridge_before_fixed.py`: Bridge 고정 역할(PUSH2·BEFORE2), 같은 출발 무대 20%/crossing 80% 배치·crossing 선행 payload의 다리 중심 progress 보상·Default 후행 비활성 Target 좌표 보정. [명세](reference/stage2_sensor_900_bridge_before_fixed.md).
- `humanoid_ma_sensor_bridge_layout_cache.py`, `utils/episode_layout_cache.py`: fixed Bridge의 최종 거리 무작위 배치 사전 검사·환경당 36개 캐시 복원·이전 거리 커리큘럼 캐시 호환. [명세](reference/stage2_sensor_900_bridge_before_fixed_layout_cache.md).
- `test_bridge_neutral_targets.py`, `test_bridge_cache_saved_config.py`: 후행 Target의 좌표·부분 reset·활성 목표/보상 보존과 저장 sampler의 평가·재개 계약·wrapper 설정 선택 회귀 검사.
- `humanoid_ma_sensor_bridge_before_carry_rsi.py`, `utils/bridge_before_carry_rsi.py`: 고정 Bridge에 운반 경계 RSI(다리 놓임 강제·carryWith 자세+상자)와 carryWith AMP 이력 수정을 더한 subclass. 배치·캐시는 참고 repo와 같은 `bridge_carry_boundary_rsi.py`·`bridge_carry_rsi_cache.py`·`bridge_carry_rsi_probe.py`, 캐시 생성은 `scripts/multi_agent/prepare_bridge_carry_rsi.py`. 본인 상자 센서 제외(hidebox)는 `sensor.exclude_own_payload` 키가 있을 때만 `sensor_stage2_observation`·`square_sensor`에서 동작한다. [명세](reference/stage2_sensor_900_bridge_before_fixed_carry_rsi.md).
- `utils/premature_push.py`와 `reward_source: v4_teachers` 분기(`before_reward`·`stage2_reward`·`stage2_wait`·`humanoid_ma_coupled_push`·`cliff_bridge_controller`·`edge_context_task`): dev V4 보상. 현재 `stage2_sensor_900_bridge_before_fixed`만 사용한다.

- `bridge_surface.py`: 선택적5mm 상면 충돌 형상·기존 외형/질량/관성 보존. 일반/외력 다리 생성에서 공유한다. [전용 variant](reference/stage2_sensor_900_cliff_bridge_surface.md).
- [Bridge 학습 개선 우선순위](reference/bridge_learning_improvement_priorities.md): 거리·경계 RSI·접합 틈 등 개선 후보와 비교 원칙. 구현 전 제안 문서다.

아래 경로는 `tokenhsi/env/tasks/multi_agent/` 기준이다.

| 파일 | 역할 |
| --- | --- |
| `humanoid_ma.py`, `humanoid_ma_carry.py` | 다중 사람·제어·상태와 공통 환경·물체 생성 |
| `scene_observation.py`, `scene_reset.py`, `reference_motion.py` | 관측, reset/배치, 모션·RSI·AMP |
| `edge_ontop_task.py`, `edge_context_task.py` | 과제 그래프·물리 재시도·보상·진단 |
| `humanoid_ma_push.py`, `jointpush_reward.py` | 1/2명 PUSH 자산·배치·접촉 RSI·CONTACT+AT 보상 |
| `humanoid_ma_before_push.py`, `before_push_reward.py` | BEFORE solo PUSH 자산·배치·접촉 보상, 기존 BEFORE gate 유지 |
| `humanoid_ma_stage2.py`, `stage2_layout.py`, `stage2_reward.py`, `stage2_wait.py` | 통합 Stage2의 관계별 배치·RSI 재추출(선택적으로 COUPLED 포함)·AT/접촉/선후행 gate·대기 속도 감점 |
| `humanoid_ma_coupled_push.py`, `coupled_push_reward.py` | 공동 상자·가상 구역·그룹 gate·goal saturation |
| `humanoid_ma_sensor_stage2.py`, `sensor_stage2_{geometry,placement,observation}.py` | Stage2 센서 시나리오·고정 bridge 자산·BEFORE/COUPLED 배치·관측·진단 |
| `square_sensor.py`, `square_sensor_scene.py` | 기준 repo의 48×48 높이 센서·cliff/bridge/room 기하 |
| `bridge_entry_chamfer.py` | 추론 전용 bridge 입구 모따기 convex mesh/URDF·학습 사용 방지 |
| `push_goal_markers.py` | PUSH 목표의 viewer/VNC 선 표시만 추가; actor·reset·scale 수정 없음 |
| `humanoid_ma_before.py` | BEFORE 역할·공유 물체·AT progress·후행 초기화 |
| `scene_layout.py`, `scene_validation.py` | 활성 물체·distractor·loco 참조·물리 유효성 |
| `before_rsi_cache.py`, `native_rsi_cache.py` | 후행 pickUp/carryWith native 크기별 screening |
| `before_pickup_carry_rsi.py`, `before_relative_sampling.py` | RSI433 적용/fallback, near50 배치 |
| `size_rsi_cache.py` | Stage1 크기별 RSI 캐시 |
| `*_reward.py` | 과제 보상·BEFORE gate·CPA·물체 속도 |

## 정책과 과제 정의

- `tokenhsi/env/tasks/multi_agent/humanoid_ma_sensor_cliff_bridge.py`, `cliff_bridge_{controller,placement,metrics}.py`, `tokenhsi/utils/cliff_bridge_{spec,layout}.py`: Cliff/Bridge 전용 variant·외력 다리·BEFORE 횡단 배치와 분기별 진단. [설정](reference/stage2_sensor_900_cliff_bridge.md).
- `tokenhsi/tests/{test,check}_cliff_bridge.py`: 샘플링·토큰·경로·외력 CPU 검사와 실제 추론/센서/부분 reset/초기 렌더 검사.

- `tokenhsi/utils/bridge_curriculum.py`, `tokenhsi/env/tasks/multi_agent/sensor_stage2_curriculum.py`: 횡단/협력 샘플링·길이 schedule·다중 crosser 배치와 agent별 횡단 진단.
- `tokenhsi/tests/test_room_before_sampling.py`: Room 관계 비율·실내/외 역할·COUPLED 보존·resume 변경 범위 검사.
- `tokenhsi/tests/{test_bridge_curriculum,check_bridge_curriculum}.py`: curriculum CPU 계약·1024환경 실제 reset/센서/초기 화면 검사.

- `tokenhsi/tests/{test_bridge_training_start,check_bridge_training_start}.py`: 새 Bridge 20/40/40 시작 분포·선배치 crosser/상자 근접·실제 actor/센서/부분 reset 및 초기 화면 검증.

- `tokenhsi/learning/multi_agent/sensor_stage2_{network,agent}.py`: actor encoder/coordination·sigma·입력 RMS 동결; 센서·action head·critic 전체·AMP 및 AMP/value RMS 학습, resume 학습 계약 검사.
- `tokenhsi/utils/sensor_stage2_spec.py`: 4명·8토큰 센서 계약, bridge 역할 샘플링·Room BEFORE 증가/독립 PUSH 제외 및 제한된 sampler-only resume 계약.
- `tokenhsi/tests/{test_sensor_stage2,check_sensor_stage2}.py`: CPU 전이/동결/샘플링·실제 1024환경 배치/센서 검사.
- `tokenhsi/tests/{test_bridge_entry_chamfer,test_bridge_entry_alignment,capture_bridge_chamfer}.py`: 모따기/끝선 정렬·초기 삽입 기하 검사와 고정 checkpoint의2/3인 PUSH 영상·실제 초기 좌표 검증.
- `output_etc/check/bridge_crosser_compare_20261108/{capture,analyze,package}.py`: 다리 선배치의 crosser 정렬/무작위 시작 비교·초기 상태/궤적 검증·좌우 비교 영상 생성용 진단 산출물.

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
- `tokenhsi/utils/coupled_push_peer_selection.py`: viewer의 keep3/shrink3과 학습·추론의 keep1/shrink1 접촉 기반 상대 선택·독립 RNG·부분 reset. 선택되지 않은 COUPLED는 NONE으로 바꾸며, `test_coupled_peer_selection.py`·`test_single_coupled_peer.py`가 관계 전환·셔플·PPO 선택 재사용·전이를 검사한다.
- `tokenhsi/utils/coupled_push_assignment.py`: viewer 전용 좌우 정렬·Hungarian·교차 우선 구역 배정. reset 후 개인 목표/관측/접촉에 같은 배정을 연결한다. `test_coupled_assignment.py`는 배정 목적함수·회전/그룹·접촉/목표 연결·9명 이상 근사 탐색을 검사한다.
- `tokenhsi/utils/{before_push_spec,before_push_layout}.py`: solo PUSH 샘플링·계약·다중 사람 경로 배치.
- `tokenhsi/utils/{before_sampling,before_relative_sampling,before_object_alias}.py`: mix226/near50·토큰 alias.
- `tokenhsi/learning/multi_agent/stage1_push_network.py`: 5과제 정책·4 family AMP·선택적78k 전이.
- `tokenhsi/utils/{stage1_jointpush,jointpush_scene,jointpush_geometry,jointpush_rsi,jointpush_rsi_geometry}.py`: PUSH 계약·1/2명 장면·bank/모션/기하 검증.
- `tokenhsi/utils/stage1_sampling.py`: 기본 목표 정사각형 샘플링과 carry5 전용 상자 기준 거리/방향 샘플링.
- `tokenhsi/utils/{stage1_geometry,stage1_config,stage1_platforms,size_rsi}.py`: Stage1/BEFORE 공통 기하·RSI·물리 설정과 coupled carry5_ground 지면 배치 계약.
- `tokenhsi/utils/motion_lib.py`: 모션 파일·보간·RSI 샘플링, 프레임 일괄 관절 속도 계산과 필요한8개 텐서의 GPU 캐시.
- `lpanlib/poselib/`: 기존 모션 배열 로딩 라이브러리.

## 검증과 문서

`tokenhsi/tests/`는 보상·배치·GTA·mask·RSI·전이·재로드·출력 경로를 검사한다. `before_test_utils.py`, `stage1_test_utils.py`는 CPU fixture다. `check_before_object_alias.py`·`check_before_push.py`·`check_coupled_push.py`는 실제 물리·부분 reset·추론 검사 진입점이다.

`test_coupled_scene_sampling.py`는 비PUSH 사람의 경로 장애물 유지·공동 상자의 전체 크기·distractor 재시도와 ON_TOP 받침 보존을 검사한다.

[Stage2 sensor](reference/stage2_sensor_900.md), [통합 Stage2](reference/stage2.md), [PUSH](reference/stage1_push.md), [BEFORE PUSH](reference/stage2_before_push.md), [공통 Stage2/coupled](reference/stage2_coupled_push.md), [4명 초과 그룹 스폰](reference/group_spawn_over4.md), [평가 규약](reference/evaluation.md), [Stage1 단독](../evaluation_single.md), [공통 정책](reference/ma_clean_scene_gta.md), [BEFORE](reference/stage2_before.md), [지표](reference/relation_diagnostics.md), [이관](reference/PORTING_GUIDE.md)을 참고한다. 이전 실험은 [legacy](legacy/README.md)에 보존한다.

`tokenhsi/tests/test_bridge_surface.py`·`check_bridge_surface_material.py`는 Bridge 고정 높이 관측·checkpoint 격리·자산 기하/질량과 CPU 미끄러짐을 검사한다. `check_cliff_bridge.py`는 새 variant의 실제 GPU actor 재질/질량 검사도 지원한다.

`tokenhsi/tests/test_ma_rollout_reset.py`는 실제 play_steps/wrapper의1/4명 환경에서 마지막 step 종료가 다음 rollout 행동 전에 reset되는지, 종료 보상 중복·개인 bootstrap·초기 done을 검사한다.
