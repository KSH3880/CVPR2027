# 레포 구조

Multi-agent 실험은 `stage1`, `stage1_v2`, `stage1_v2_fixed`, `stage2`와 공유 받침 성공 기준 비교 두 개다. 컨피그 목록·기본 실행 명령·추론 옵션은 [config.md](config.md), 추가 옵션·checkpoint 규칙은 [USAGE.md](USAGE.md)를 따른다.

## 실행과 설정

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/utils/push_eval_sizes.py`, `tests/{test,check}_push_eval_sizes.py`, `humanoid_ma_five_task.py` | 평가 전용 실제XYZ 크기·mass/friction 보존·bbox관측 검사·명시적PUSH배정; [명령](USAGE.md#push-크기-제로샷) |
| `tokenhsi/utils/push_box_switch.py`, `tests/test_push_box_switch.py` | 목표 고정·평가 중 상자를 목표 반대편/사람 뒤로 이동·겹침 회피·목표/사람 불변 확인; [명령](USAGE.md#push-중간-상자-이동-제로샷) |
| `tokenhsi/utils/push_goal_switch.py`, `tests/test_push_goal_switch.py`, `learning/multi_agent/edge_context_eval.py` | 평가 전용90°/180° 목표 전환·이전 성공 이력 분리·전환 좌표/성공 JSON; [명령](USAGE.md#push-중간-목표-전환-제로샷) |
| `*five_task_depth{180_20kg,200_20kg_friction04}_train.sh`, `humanoid_ma_carry.py` | `TASK_GRAPH=push_at`으로 5과제 Stage1 checkpoint의 PUSH 100% 재개; [명령](reference/push100_resume1000.md) |
| `tokenhsi/utils/push_motion_metrics.py`, `tests/test_push_motion_metrics.py`, `scripts/multi_agent/resume_five_task_metrics.py` | PUSH 물리 tick별 접촉/이동/목표 진척 관측·에피소드 집계·고정 PID의 500회 재개; [명세](reference/push_training_metrics.md) |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_face_xy_depth200_20kg_friction04.yaml`, 동명 train/test/VNC·전용 RSI cache | Y2m·20kg·상자마찰0.4 공통면CONTACT·물성별 RSI 재검증 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_face_xy_depth{180_20kg,200_25kg}.yaml`, 동명 train/test/VNC·`tests/{recheck,audit}_push_face_rsi.py`·`scripts/multi_agent/render_push_face_rsi.py` | 마찰0.6·공통면CONTACT 두물성/전수RSI검사·실제충돌체/발메시영상 |
| `tokenhsi/utils/push_point.py::common_face_xy`, `*_phase_face_xy` env/train/test/VNC, `tests/{test,check}_push_common_face_xy.py` | 동적 공통 유한 면 XY CONTACT·state/반력 success·독립 보상 계약 |
| [PUSH RSI 기준](../../../../markdowns/push_rsi_guidelines.md) | 향후 PUSH RSI의 원본 보존·높이/밀기 구간 선별·loco 분리·상자 동반 검수 기준 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_depth180_{h100_rsi1m_at5deg,h080_rsi1m}_phase.yaml`, 동명 train/test/VNC | 새 RSI5모션·loco50%·목표1~4m(커리큘럼 없음)의 두 새 학습 설정 |
| `tokenhsi/scripts/multi_agent/select_push_rsi_phase.py`, 동명 `tests/test_push_rsi_phase_selection.py`·`data/stage1_push_depth180_rsi_le1m_push_phase_*` | 원본 손 높이≤1m·양팔 밀기 구간만 RSI 재선별; 기존 학습 미적용 |
| `tokenhsi/utils/push_goal_curriculum.py`, `scripts/multi_agent/migrate_push_goal_curriculum.py`, `tests/{test,check}_push_goal_curriculum.py` | 목표1~5→1~7→1~10m·epoch 기준 단계/상태보존 이관 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_depth180_{h100_rsi1m_at5deg,h080_rsi1m}.yaml`, 동명 train/test/VNC·`tests/filter_push_rsi_height.py`, `test_push_rsi_height_selection.py` | Y1.8m/20kg·원본RSI손중점≤1m 선별·1.0m/AT5°와0.8m/무제한 비교 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_depth180_safe_h100_relaxed.yaml`, 동명 train/test/VNC | Y1.8m·20kg·손 목표1.0m 고정·완화state·성공state>0.9+0.5N |
| `tokenhsi/scripts/multi_agent/analyze_push_depth180_high.sh`, `report_push_depth180_high.py` | Y1.8m/정방형 대조·20/25kg·고위치 접촉체/21AMP PD 전도 검사·trace 재검증 보고서 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_depth180_safe_{h065,hamp}_relaxed.yaml`, 동명 train/test/VNC·`tests/test_push_depth180_relaxed.py` | Y1.8m·0.65m/AMP 높이 비교, 연속 state 완화·성공 state>0.9+0.5N |
| `tokenhsi/tests/test_push_midpoint_force.py` | safe PUSH 3종 0.5N 기본값·법선 반력/위치 동시성·부분 reset·checkpoint 계약 CPU 검사 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_midpoint_safe_at5deg.yaml`, 동명 train/test/VNC·`tests/test_push_at5deg.py` | GPU5 safe 기반 AT 성공 기울기≤5°·독립 output/checkpoint |
| `tokenhsi/tests/{audit_push_topple,analyze_push_topple_geometry,test_push_reset_geometry}.py` | 초기학습전도사건추적·발mesh내부정점분석·collider/바닥회귀검사 |
| `tokenhsi/utils/push_reset_geometry.py`, `stage1_push_{midpoint,depth180}_safe` config/train/test/VNC/cache | 발mesh·sphere/capsule의상자/바닥초기관통방지·GPU5/6별도계약; [명세](reference/push_safe_start.md) |
| `tokenhsi/data/cfg/multi_agent/stage1_push_depth180_native_rsi.yaml`, 동명 train/test/VNC·cache 생성기 | X만1.2~1.5가변·Y1.8/Z1.3고정·20kg·−Y면/+Y목표1~4m; [명세](reference/stage1_push_depth180_native_rsi.md) |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_midpoint_native_rsi.yaml`, 동명 train/test/VNC·전용cache 생성기, `tests/test_stage1_push_midpoint.py` | 중점만 CONTACT/state>0.9·원본AT/paired포화·20kg/X=Y/Z고정/목표1~4m; [명세](reference/stage1_push_midpoint_native_rsi.md) |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_rect_point.yaml`, 동명 `_{train,test,vnc}.sh`, `data/stage1_push_rect_point_bank.json`, `tests/{test_stage1_push_rect_point,check_stage1_push_rect_rsi}.py` | 독립XYZ 1.2~1.5m/20kg·높이0.65m 양손중점·전수선별3 RSI; [명세](reference/stage1_push_rect_point.md) |
| `tokenhsi/scripts/multi_agent/{analyze_push_rect_stability.sh,analyze_push_rect_contacts.py,analyze_push_rect_humanoid.py,push_stability_common.py,report_push_rect_stability.py}`, `tokenhsi/tests/test_push_stability_metrics.py` | XYZ독립가변·동일질량·양손중점의전구간기울기/전진분석과측정검증; [명세·결과](reference/push_rect_stability.md) |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_no_force.yaml`, 동명 `_{train,test,vnc}.sh` | 힘 gate만 제거한 CONTACT 비교·전용output/계약; 위치/높이·원본AT 유지 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push.yaml`, `tokenhsi/scripts/multi_agent/stage1_push_{train,test,vnc}.sh` | 기존 Stage1의2명·4상자 입력 유지·PUSH100%·직선1~5m 별도 실행; [명세](reference/stage1_push.md) |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_stage1_push.py`, `tokenhsi/utils/stage1_push.py` | 실제CONTACT·기존AT·고정물성/목표·11자세RSI·PUSH AMP 연결, checkpoint분리 |
| `tokenhsi/tests/{test,check}_stage1_push.py` | CPU 계약/원본회귀·GPU4상자/634D/부분reset/물리 검사 |
| `tokenhsi/run.py` | 학습·평가 등록과 실행 |
| `tokenhsi/data/cfg/multi_agent/stage1*.yaml` | 세 실험의 환경 설정 |
| `tokenhsi/data/cfg/train/rlg/stage1*.yaml` | 세 실험의 PPO/AMP/모델 설정 |
| `tokenhsi/scripts/multi_agent/stage1*_{train,test,vnc}.sh` | 실험별 학습·로컬 평가·서버 viewer, 총9개 |
| `tokenhsi/scripts/multi_agent/{runtime_env,mps_auto_env,gui_gpu_env,run-gui}.sh` | 공통 conda·GPU·MPS·VNC 준비 |
| `output/<실험>/<run>/` | 본학습 checkpoint·TensorBoard·진단, Git 제외 |
| `output/rsi_cache*` | 운영용 공용 RSI 캐시 |
| `output_etc/check/<실험>/{train,eval}/` | 확인 학습·평가·viewer 결과 |
| `tokenhsi/utils/output_paths.py`, `tokenhsi/scripts/multi_agent/output_paths.sh` | 확인 경로 검사·구 output 경로 전환 |
| `output_etc/check/stage1_cleanup_20261015/` | 실험 정리의 동등성·실행 검증·이동 경로표 |
| `output_etc/check/output_routing_20261015/` | 문서/출력 정리의 CPU·학습·경로 전환 검증 |

## 메인 구현

| 경로 | 역할 |
| --- | --- |
| `env/tasks/multi_agent/humanoid_ma.py`, `humanoid_ma_carry.py` | `tokenhsi/` 아래의 다중 사람·상자·선반 물리, scene 관측·RSI·AMP 통합 |
| `env/tasks/multi_agent/edge_ontop_task.py`, `edge_context_task.py` | graph/배치 reset·물리 재추첨·보상·trace·진단 |
| `env/tasks/multi_agent/edge_stage1_reward.py` | 자기 edge 합산·placement 성공 시 HOLDING 보상 포화·task packet |
| `env/tasks/multi_agent/edge_interaction_reward.py`, `edge_ontop_reward.py` | SIT/CLIMB/AT/ON_TOP 기하·상판 영역 성공 |
| `env/tasks/multi_agent/collision_reward.py`, `relation_reward.py` | 사람 CPA·graph 담당 물체 속도 패널티 |
| `env/tasks/multi_agent/size_rsi_cache.py` | 크기별 물리 screening과 프레임 cache |
| `utils/edge_scenario_spec.py`, `task_role_spec.py` | canonical 독립 과제·크기 조건 sampler·가변 인원·task packet |
| `utils/edge_context_spec.py`, `edge_ontop_spec.py`, `edge_interaction_spec.py` | graph 자료형·배치/부분 선택·순열·관계 ID |
| `utils/edge_stage1_spec.py`, `relation_task_spec.py` | 남은 Stage1 reward·checkpoint 계약 |
| `utils/size_rsi.py`, `unified_training.py` | 실제 크기·RSI pool/fallback·과제별 AMP family와 모델 계약 |
| `utils/stage1_platforms.py`, `stage1_sampling.py` | 선반 높이·graph binding·원본 거리 분포·구 stage1 snapshot 호환 |
| `utils/stage1_v2.py`, `stage1_v2_fixed.py` | v2 기하·실제 높이 phase·fixed env/sim 복원 |
| `learning/multi_agent/amp_network_builder_ma.py`, `task_role_encoder.py` | 공유 task bias·사람 residual MLP·GTA actor/critic |
| `learning/multi_agent/ma_agent.py`, `ma_players.py`, `scene_normalizer.py` | scene 단위 PPO·AMP·checkpoint·평가·노드별 정규화 |
| `learning/multi_agent/edge_context_eval.py` | 현재/누적 task·scene 성공 평가 |

표의 구현 경로는 `tokenhsi/` 기준이다. `edge_ontop_*`, `edge_context_*`, `task_role_*`는 세 실험이 사용하는 공통 구현 이름이다. 파일명만 보고 제거하지 않는다.

## 검증

- `test_output_paths.py`: 본학습/확인 분리·기존 경로 전환·경로 이탈 거부.
- `tokenhsi/tests/stage1_test_utils.py`: 과거 실험에 의존하지 않는 CPU fixture.
- `test_stage1.py`, `test_stage1_sampling.py`, `test_stage1_v2.py`, `test_stage1_v2_fixed.py`: 세 config·성공·RSI·선반·snapshot 계약.
- `test_stage1_policy.py`, `test_stage1_gta.py`, `test_stage1_wrapper.py`, `test_ma_scene_features.py`: task/edge/token 순열·gradient·optimizer·GTA 수식·정규화.
- `test_generic_task_scenario.py`, `test_graph_box_speed_penalty.py`, `test_agent_cpa_collision.py`, `test_gui_gpu.py`: 가변 인원·담당 물체·CPA·GPU 선택.
- `smoke_stage1_sim.py`, `smoke_stage1_v2_training.py`: 실제 reset/선반/CPA 및 학습 중 RSI/AMP 정렬 검사.

## 문서와 별도 자료

원본 단일인 TokenHSI의 `basic_interaction_skills/`, `adapt_interaction_skills/`, `comp_interaction_skills/`, `multi_task/`, `longterm_task_completion/` 설정·코드와 데이터 전처리는 이번 multi-agent 실험 정리 범위 밖이다. [원본 태스크 참고](legacy/tokenhsi_original_carry_sit_climb.md), [v2 비교](legacy/tokenhsi_vs_stage1_v2_train_eval.md)를 참조한다.

설치·추가 실행 옵션은 [USAGE.md](USAGE.md), 머신 이관은 [PORTING_GUIDE.md](reference/PORTING_GUIDE.md), 지표는 [relation_diagnostics.md](reference/relation_diagnostics.md)를 확인한다.

일상 안내는 `README.md`, `changelog.md`, `AGENTS.md`와 `markdowns/{config,structure,USAGE}.md`다. 상세 모델·진단·포팅은 `markdowns/reference/`, 과거 변경·분석은 [legacy](legacy/README.md)에 보관한다. MPS 운영 안내는 [mps/README.md](../mps/README.md)를 따른다.

## 다섯 과제 통합

`utils/push_random_spawn.py`는 전신 collider→상자 표면 거리1~5m·360° 시작과 전체 상자 무관통 검사를 제공한다. `*five_task_depth{180_20kg,200_20kg_friction04}` env/train·train/test/VNC는 GPU6 물성별 실행이며, `tests/test_push_random_spawn.py`·`check_five_task.py`가 CPU/GPU 검증, `render_push_face_rsi.py`가 실제 자세·발메시·목표/4상자 탑뷰 영상을 담당한다. [명세](reference/five_task_gpu6_variants.md).

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/utils/{five_task,five_task_stage2}.py` | 기존 네 ID 보존·PUSH_AT=4 검증/분포·공동 source/goal graph |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_five_task{,_stage2}.py` | Stage1 혼합5과제·Stage2 PUSH 전용 coupled·기존 보상/RSI와 분리 적용 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/*five_task.yaml`, 동명 train/test/VNC·`data/dataset_five_task.yaml` | 기존 Stage1/2 기반 별도 실행·원래 모션에 PUSH 추가 |
| `tokenhsi/tests/{test,check}_five_task.py` | ID/정책/AMP/전이 CPU 회귀·GPU reset/부분reset/물리/PUSH RSI 검사 |
| `tokenhsi/tests/check_five_task_eval_rsi.py` | 5과제 Stage1 평가의 PUSH/loco/혼합 시작·원본 관절/속도·목표거리·부분 reset·12step 검사 |

[설정과 검증 범위](reference/five_task_push_at.md)를 따른다.

## Stage2 추가 구현

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_stage2.py`, `stage2_scene.py` | fixed 기반 4명 그룹·8개 장면 상자·공동 파지·GraspView·CPA 예외 |
| `tokenhsi/utils/stage2_spec.py`, `stage2_amp.py`, `stage2_rsi.py` | 그룹 graph·task별 AMP·15cm 공동 RSI 캐시와 누락 시 자동 생성 |
| `tokenhsi/env/tasks/multi_agent/stage2_solo_rsi_cache.py` | 단독 RSI의 4명 asset bank 연결 |
| `tokenhsi/learning/multi_agent/stage2_network_builder.py`, `human_coordination.py`, `grasp_view.py` | 현재 stage1 가중치 전이·동결·human SA4·사람별 앵커 관측 |
| `joint_carry/` | 복사한 공동 reference·보정·크기별 RSI 생성과 물리 screening |
| `joint_carry/render_size_rsi.py` | 원본/대표27크기의 native asset RSI 영상·검사 통과 프레임 표시 |
| `joint_carry/single_source.py`, `single_source_manifest.json` | 1인 원본 B19/B20/B21과 공통 다리 보정·공동 배치 생성 |
| `joint_carry/audit_size_rsi.py`, `tokenhsi/utils/grasp_geometry.py` | 생성 범위의 데이터 검사·RSI/GTA/리워드 공통 앵커 |
| `tokenhsi/tests/test_stage2.py`, `check_stage2.py` | CPU 계약·전이·gradient와 GPU reset/추론·부분 reset 검증 |

세부 계약과 데이터 준비는 [Stage2 참고](reference/stage2.md)에 있다.

## 공유 받침 비교 실험

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage2_shared_{full_success,region_success}.yaml` | 전체 받침/선택 구역 성공 비교, 4명·1024환경 |
| `tokenhsi/scripts/multi_agent/stage2_shared_*_{train,test,vnc}.sh` | 두 실험 전용 실행, 기본 GPU5 |
| `tokenhsi/utils/shared_support.py` | 순차 그룹 sampler·최소 토큰 수·구역/distractor 압축 배치·강체 변환·checkpoint 계약 |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_shared_support.py` | 물리 bank·공유 관측·loco reset·현재 최근접 구역 선택 |
| `tokenhsi/env/tasks/multi_agent/shared_support_{reward,scene}.py` | 성공 영역 비교·원본 물리 높이·실제 물체 배치 검사 |
| `tokenhsi/learning/multi_agent/shared_support_network.py` | padding 없는 공통 장면 encoder·물리 ID 기반 coordination |
| `tokenhsi/tests/{test,check}_shared_support.py` | CPU 계약·보상·전이와 GPU 추론/부분 reset 검사 |

설정·관측·초기화 계약은 [공유 받침 참고](reference/shared_support.md)에 있다.
