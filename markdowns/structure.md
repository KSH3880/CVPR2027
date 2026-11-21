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

- `utils/config.py`의 `--pusher_assignment`와 `launch.sh`의 `PUSHER_ASSIGNMENT`는 rear90 추론 전용 담당 배정 override다. `humanoid_ma_bridge_follow_cap.py`에서 저장 cfg와 분리한 runtime 사본에 적용한다.
- 같은 실행기의 `--pusher-gaps`/`PUSHER_GAPS`는 `utils/bridge_pusher_sampling.py:set_pusher_gaps`로 연결된다. reset의 simulator 상태 전송 전 밀기 사람의 root/kinematic body XY를 이동하고 이후 담당 배정을 수행한다. 검증은 `output_etc/check/bridge_rear90_view_gaps_20261119/`에 있다.
- `utils/bridge_pusher_sampling.py`: `_rear90` 전용 상자 기준 후방 90°·외곽 거리 0.5~5m 후보 생성과 플랫폼/상자/사람 여유 검사. `humanoid_ma_bridge_follow_cap.py`에서 배치 후 호출하며 relation history 초기화 전 담당 영역을 정렬한다. CPU 검증은 `tests/test_bridge_pusher_sampling.py`.
- `humanoid_ma_bridge_follow_cap.py`, `bridge_follow_cap.py`: 기존 layout-cache 샘플러의 학습용 추종 캡·선행 waypoint·단계별 progress. `learning/multi_agent/bridge_waypoint_single_pass_network.py`는 후행 alias를 원래 목표에 둔 뒤 선행 GTA 위치만 바꾸어 actor/critic 각각 한 번 계산한다. `bridge_waypoint_network.py`는 이전 두 번 계산한 실행 재현용이다. [세 설정 명세](reference/bridge_follow_cap_training.md).

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

`tokenhsi/qualitative/bridge_rear90_zeroshot.py`는 `_rear90`의 학습 배치 캐시를 사용해 원본 V4 정책을 추가 학습 없이 실행한다. `stage2_sensor_900_bridge_follow_cap_rear90_zeroshot_{test,vnc}.sh`가 진입점이며, 정책 strict 로드와 관측 정규화는 `bridge_waypoint_policy.py`를 공유한다. [명령](USAGE.md#follow-cap-5cm-학습-비교).

`tokenhsi/qualitative/bridge_wait_{viewer,layout}.py`는 학습된 follow-cap 모델의 선행·상자·목표 좌우 위치와 공동밀기 삽입 방향/거리를 지정하는 추론 전용 viewer다. 실행은 `scripts/multi_agent/bridge_wait_{test,vnc}.sh`이며 [위치 지정 인자](USAGE.md#학습-모델의-위치-지정-viewer)를 따른다.

`output_etc/check/bridge_training_view_20261118/view_training.{sh,py}`는 기존 학습 배정 viewer다. 시작 모드 선택과 캐시 복원 직후 밀기 사람의 삽입축 거리 변경을 지원하며 담당 영역·개인 목표는 보존한다.

`output_etc/check/bridge_rear90_eval_1000_ordered_20261118/`는 rear90 정렬/랜덤 학습 1k checkpoint를 새 rear90 insert fixture와 동일한 최종 좌우 정렬로 비교한다. checkpoint 검증 후 첫 평가 reset 전 배정 override와 배치/담당 영역 검증을 포함한다.

`output_etc/check/bridge_rear90_eval_3500_random2500_ordered_20261118/`는 위 fixture·평가 판정을 재사용하여 rear90 3500/random 2500을 비교한다. `launch_eval.py`는 GPU5 전용 최대 4개 동시 실행·85000MiB 감시·완료 후 자동 집계를 담당한다.

`output_etc/check/bridge_rear90_eval_4500_random3500_ordered_20261119/`는 rear90 4500을 GPU5에서 평가하고 random 3500 seed 0/1/2는 GPU5/6/7로 분산한다. 지정 checkpoint가 없으면 저장 완료·epoch 검증까지 기다린 뒤 자동 실행하며 완료 결과는 재사용한다.

`output_etc/check/bridge_rear90_eval_7000_ordered_20261119/`는 동일 7k의 GPU5 rear90/GPU7 random 평가다. 기존 fixture와 평가 판정을 유지하고 GPU별 메모리 예약·상한 감시·checkpoint epoch 확인·완료 후 집계를 수행한다.

위 폴더의 `approach_analysis/`는 선행 pickup 추정–다리 진입 XY 경로, `failure_analysis/`는 씬 실패 단계·후행 활성 후 성공·선행 낙상 위치와 다리 준비 타이밍의 CPU 분석이다. 각각 `analyze.py`, `summary.json`, `episodes.csv`를 보관한다. 준비 판정은 기존 중심 거리/각도 기준이며 발 접촉 분석과 구분한다.

`output_etc/check/bridge_follower_diagnostics_7000_20261119/`는 같은 7k/fixture의 후행 진단 재생이다. `eval_crossing.py`가 원 evaluator를 감싸 후행 root/손/발·상자·native 오차를 기록하고, `analyze_followers.py`가 원 평가와 초기 상태/관측·재생 편차를 검사하며 정체·목표 오차·상자 상승을 분석한다. `follower_analysis.json`과 `follower_episodes.csv`가 결과다.

같은 폴더의 `view_failure.sh`/`view_failure.py`는 7k 실패 장면을 모델·seed·환경 번호로 선택하는 VNC다. `single_scene.py`가 512환경 fixture 중 선택한 한 행의 물리 크기·초기 상태를 추출하고 월드 좌표를 이동해 **1환경**으로 복원한다. 카메라는 도착 플랫폼을 향하고 후행은 초록색이다. `HEADLESS=1 VIEW_CHECK_STEPS=16`은 GUI 없는 짧은 경로 검사이며 원 평가 파일을 덮어쓰지 않는다.

`output_etc/check/bridge_push3_follow_cap_view_20261119/{view.sh,view.py,scene.py,pair_layout.py}`는 기존 rear90 checkpoint의 1환경 추론 전용 경로다. `--pushers 2|3|4`, `--before-pairs N`, `--bridge-width <m>`로 밀기 인원·같은 방향의 선행/후행 쌍 수·보행로 폭을 조절한다. `scene.py`는 프로세스 내 역할·물리 크기를 확장하고 `pair_layout.py`는 기존 횡단 sampler와 점유 검사를 이용해 각 선행의 사람·상자·목표를 배치한다. 배치 후 담당 정렬·follow_cap과 쌍별 상자/AT 연결을 검증한다.

`output_etc/check/bridge_random15k_push4_enable2_eval_20261120/`는 random15k·밀기4명/Enable2세트의 8명 평가다. viewer adapter를 복사해 고정하고 `evaluate.py`가 900/1500step의 세 협업 과제·전체 씬 성공과 gate 제외 native 행동 성공을 기록한다. `launch.py`는 GPU5/6/7의 3seed를 관리하고 `summarize.py`는 프레임 재집계/검증 및 결과표를 생성한다.


`output_etc/check/bridge_missing_8k9k11k_ordered_20261119/`는 미평가 rear90 8000·9000 및 random 11000의 보충 평가·집계다. 기존 4명 evaluator·fixture로 각각 3seed를 GPU5/6에 분산한다.

`output_etc/check/bridge_rear90_random_12000_ordered_20261119/`는 rear90/random 동일 12k 평가다. GPU5/6에서 각각 3seed를 실행하며 미저장 checkpoint는 기다린다. `verify.py`로 이전 초기 상태·성공 집계를 확인하고 `history.py`로 7k~12k의 1k 단위 표를 만든다.

`output_etc/check/bridge_rear90_13000_ordered_20261119/`는 GPU5/6의 rear90 13k 평가와 기존 12k 결과 비교다. 이전 평가 결과를 읽기 전용으로 연결하고 동일 초기 상태 검증·전체 지표·7k~13k 표를 생성한다.

`output_etc/check/bridge_random_13000_18000_ordered_20261120/`는 GPU7의 random 13k~18k 평가다. 6개 checkpoint×3seed를 메모리 한도 내 병렬 실행하고 기존 fixture 대비 검증·전체 지표·7k~18k 표를 생성한다.

`output_etc/check/bridge_random_at_failure_20261120/analyze.py`는 random 10k~18k 저장 궤적에서 AT-ever/최종 실패, 횡단·낙상 분류와 후행 성공을 CPU 분석한다. `summary.json`에 정의·집계, `scenes.csv`에 해당 장면을 저장한다.

`output_etc/check/bridge_rear90_11000_random10000_ordered_20261119/`는 rear90 11000·random 10000의 평가·집계다. 두 모델의 seed0/1/2를 GPU5/6/7에 나누고 이전 evaluator·fixture를 재사용한다.

`output_etc/check/bridge_rear90_10000_random9000_ordered_20261119/`는 rear90 10000·random 9000의 평가·집계다. 두 모델의 seed0/1/2를 GPU5/6/7로 나누고 이전 evaluator·fixture를 재사용한다.

`output_etc/check/bridge_rear90_9500_random8000_ordered_20261119/`는 rear90 9500·random 8000을 GPU6/7에서 평가한다. 기존 7k evaluator와 초기 fixture를 재사용하고, 실행기·집계·로그·결과는 분리한다.

`output_etc/check/bridge_crossing_eval_6000_training_assignment_20261118/`는 가상목표 없는 6k의 기존 학습 담당 배정 평가와 동일 checkpoint·물리 배치의 ordered 결과 비교를 보관한다. 접촉 probe 없이 행동/씬 지표·제어 스텝 궤적을 저장한다.

`output_etc/check/bridge_crossing_eval_6000_ordered_20261118/`는 두 6k 모델의 최종 배치 후 좌우 정렬 평가 실행·행동/씬 SR 집계·접촉 진단을 보관한다. `assignment_verification.json`은 실제 reset 완료 후 담당 순서와 개인 목표 검증이다.

`output_etc/check/bridge_crossing_eval_3500_20261118/{eval_crossing.py,launch_eval.py,summarize_eval.py}`는 follow-cap 두 모델의 동일 초기 상태 insert 횡단 평가·GPU5/7 실행·집계를 담당한다. 모델별 seed 폴더에 초기 상태·trajectory·행동/씬 SR을 저장하며 [평가 규약](reference/bridge_follow_cap_training.md#insert-횡단-평가)을 따른다.

`output_etc/check/bridge_crossing_eval_4500_20261118/`은 같은 초기 상태의 4500 평가다. 동일한 평가·집계 스크립트를 사용하고 `launch_eval.py`는 GPU당 세 seed를 병렬 실행하며 전체 메모리 85000MiB를 감시한다.

`output_etc/check/bridge_crossing_eval_5000_contact_20261118/`은 같은 초기 상태의 5000 평가다. `contact_probe.py`는 물리 step별 발·캡 자세와 발 순접촉력을 기록하고, `analyze_contact.py`는 플랫폼 끝 접근·발 내딛기·캡 접촉 추정·낙상 시점의 밀기 진행도와 대기 후보를 집계한다. 학습 코드에는 적용하지 않는다.

같은 폴더의 `view_eval.sh`는 모델·seed를 받아 기존 5000 평가 실행기를 GUI로 호출한다. 저장된 512환경 초기 fixture를 복원하며 별도 viewer 결과 폴더를 사용한다. [VNC 실행법](USAGE.md#follow-cap-5cm-학습-비교).

`tokenhsi/utils/bridge_eval_assignment.py`는 저장된 Bridge 초기 배치에서 구조 좌표계의 좌우 위치로 학습 담당 영역과 개별 목표를 복원한다. 3500·4500·5000 headless 평가가 학습 복원을 사용한다. 같은 파일의 `restore_ordered_assignment`는 5000 eval VNC에서 최종 배치 후 실제 상자 좌표계의 사람 좌우 순서로 담당 영역·개별 목표를 정렬한다. `randomize_pusher_assignment`는 `_rear90_random` 학습·평가에서 새 배치 및 캐시 복원 후 담당 영역을 무작위 배정한다. 세 함수 모두 물리 상태는 변경하지 않는다. 평가 실행·집계·접촉 분석의 수정 후 결과는 각 평가 폴더의 `training_assignment/`에 저장한다.

`output_etc/check/bridge_training_view_20261118/`의 `view_training.{sh,py}`는 새 학습 배치 캐시·확률적 행동·native reset으로 5000 모델을 재생한다. `compare_assignment.py`와 `launch_assignment.py`는 기존 seed0 평가의 물리 상태를 보존하고 공동밀기 담당 영역·개별 목표만 학습 방식으로 복원하는 원인 분석용 실행기다. [학습 규칙 viewer](USAGE.md#학습-규칙-그대로-보는-viewer).

`tokenhsi/qualitative/bridge_waypoint{,_task,_policy,_viewer}.py`는 DistillV4 원본 제로샷의 Bridge present·직선 배치·선행 GTA 중간 목표 전환을 담당한다. `--no-waypoint`는 가상목표 없이 처음부터 원래 목표를 사용한다. 실행은 `scripts/multi_agent/bridge_waypoint_{test,vnc}.sh`, CPU/GPU 검사는 `tests/{test,check}_bridge_waypoint.py`이며 [실행법](USAGE.md#distillv4-bridge-중간-목표-제로샷-viewer)을 따른다.

`tokenhsi/qualitative/bridge_waypoint_physics.py`는 추론 전용 다리의 baseline/fixed/carry/fixed_carry/platform/baseline_nocap/carry_nocap/follow_cap 비교를 담당한다. `bridge_waypoint_cap.py`는 동적 본체를 따라가는 별도 고정 상면 캡을 생성·갱신한다. 반복 reset의 고정 자세·무대 지지·물성 검사는 `tests/check_bridge_waypoint_physics.py`, 캡의 물리적 이동·회전 추종과 본체 동역학 검사는 `tests/check_bridge_follow_cap.py`다. 사용자 확인용 영상은 `video/bridge_ablation_20261117/`, 원본 진단은 `output_etc/check/bridge_contact_20261117/`에 둔다.

캡 높이(`--cap-lift`)와 시작 배치(`--bridge-start`)의 새 영상은 `video/bridge_follow_cap_lift2cm_20261117/`, 검증 원본은 `output_etc/check/bridge_cap_lift_20261117/`에 둔다. 새 영상은 별도 폴더·개별 MP4로 제공하며 전체 목록은 `video/index.html`이다.

`video/bridge_cap_height_compare_20261118/`에는 가상목표 없이 0·1·2·5cm를 동일 장면 96회 비교한 MP4·CSV·결과 그림·JSON이 있다. 원본 trace·초기 상태와 측정/실행/집계 스크립트는 `output_etc/check/bridge_cap_height_compare_20261118/{measure.py,worker.sh,summarize.py}`에 있다.

`tokenhsi/tests/`는 보상·배치·GTA·mask·RSI·전이·재로드·출력 경로를 검사한다. `before_test_utils.py`, `stage1_test_utils.py`는 CPU fixture다. `check_before_object_alias.py`·`check_before_push.py`·`check_coupled_push.py`는 실제 물리·부분 reset·추론 검사 진입점이다.

`test_coupled_scene_sampling.py`는 비PUSH 사람의 경로 장애물 유지·공동 상자의 전체 크기·distractor 재시도와 ON_TOP 받침 보존을 검사한다.

[Stage2 sensor](reference/stage2_sensor_900.md), [통합 Stage2](reference/stage2.md), [PUSH](reference/stage1_push.md), [BEFORE PUSH](reference/stage2_before_push.md), [공통 Stage2/coupled](reference/stage2_coupled_push.md), [4명 초과 그룹 스폰](reference/group_spawn_over4.md), [평가 규약](reference/evaluation.md), [Stage1 단독](../evaluation_single.md), [공통 정책](reference/ma_clean_scene_gta.md), [BEFORE](reference/stage2_before.md), [지표](reference/relation_diagnostics.md), [이관](reference/PORTING_GUIDE.md)을 참고한다. 이전 실험은 [legacy](legacy/README.md)에 보존한다.

`tokenhsi/tests/test_bridge_surface.py`·`check_bridge_surface_material.py`는 Bridge 고정 높이 관측·checkpoint 격리·자산 기하/질량과 CPU 미끄러짐을 검사한다. `check_cliff_bridge.py`는 새 variant의 실제 GPU actor 재질/질량 검사도 지원한다.

`tokenhsi/tests/test_ma_rollout_reset.py`는 실제 play_steps/wrapper의1/4명 환경에서 마지막 step 종료가 다음 rollout 행동 전에 reset되는지, 종료 보상 중복·개인 bootstrap·초기 done을 검사한다.
