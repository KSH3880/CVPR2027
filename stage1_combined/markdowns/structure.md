# 현재 의자 FT 진입점

기존 Chair 본학습 결과는 `output/chair_hold_at_78k_20261026_135444/`와 `output/chair_bbox35_20k_20261022_142206/`에 모았다. 구 경로는 호환 링크로 유지한다. [공통 결과 경로](../../docs/OUTPUT_LAYOUT.md).

`../chair_assets/`: 로컬 의자/테이블 에셋과 catalog. `../checkpoints/Stage1V2Fixed_00073000.pth`: 초기 weight. `../tokenhsi/env/tasks/multi_agent/humanoid_chair_ft.py`: 환경. `../tokenhsi/learning/multi_agent/chair_ft_network.py`: 35D MLP. `../tokenhsi/learning/multi_agent/chair_ft_agent.py`: 동결/검증. 아래는 상속한 전체 코드 구조 참고.

# 레포 구조

Multi-agent 실험은 `stage1`, `stage1_v2`, `stage1_v2_fixed`, `stage2`와 공유 받침 성공 기준 비교 두 개다. 컨피그 목록·기본 실행 명령·추론 옵션은 [config.md](config.md), 추가 옵션·checkpoint 규칙은 [USAGE.md](USAGE.md)를 따른다.

## 실행과 설정

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/run.py` | 학습·평가 등록과 실행 |
| `tokenhsi/data/cfg/multi_agent/stage1*.yaml` | 세 실험의 환경 설정 |
| `tokenhsi/data/cfg/train/rlg/stage1*.yaml` | 세 실험의 PPO/AMP/모델 설정 |
| `tokenhsi/scripts/multi_agent/stage1*_{train,test,vnc}.sh` | 실험별 학습·로컬 평가·서버 viewer, 총9개 |
| `tokenhsi/scripts/multi_agent/train_vnc.sh` | 컨피그별 4환경 학습 noVNC, minibatch·확인 출력 설정 |
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
| `utils/evaluation_motion.py` | 테스트 task·시작 skill별 모션 로딩 선택 |
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
| `tokenhsi/scripts/multi_agent/stage2_shared_*_{train,test,vnc}.sh` | 두 실험 전용 실행, 기본 GPU0 |
| `tokenhsi/utils/shared_support.py` | 순차 그룹 sampler·최소 토큰 수·구역/distractor 압축 배치·강체 변환·checkpoint 계약 |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_shared_support.py` | 물리 bank·공유 관측·loco reset·현재 최근접 구역 선택 |
| `tokenhsi/env/tasks/multi_agent/shared_support_{reward,scene}.py` | 성공 영역 비교·원본 물리 높이·실제 물체 배치 검사 |
| `tokenhsi/learning/multi_agent/shared_support_network.py` | padding 없는 공통 장면 encoder·물리 ID 기반 coordination |
| `tokenhsi/tests/{test,check}_shared_support.py` | CPU 계약·보상·전이와 GPU 추론/부분 reset 검사 |

설정·관측·초기화 계약은 [공유 받침 참고](reference/shared_support.md)에 있다.

## Stage1 PUSH fine-tuning

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_v2_fixed_push.yaml` | fixed 기반 기존/PUSH 50:50 전용 설정 |
| `tokenhsi/env/tasks/multi_agent/{humanoid_ma_stage1_push,push_reward,push_rsi}.py` | 환경군·수직 목표 배치·접촉 보상·RSI |
| `tokenhsi/learning/multi_agent/stage1_push_{network,agent}.py` | 5-task/4-family 전이·기존 과제 teacher KL |
| `tokenhsi/utils/stage1_push.py`, `tokenhsi/data/dataset_stage1_push.yaml` | checkpoint/RSI 계약·설정 복원·AMP 목록 |
| `tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_{train,test,vnc}.sh` | 학습·평가·noVNC 진입점 |
| `tokenhsi/tests/test_stage1_push.py` | 기존 정책 보존·AMP·접촉·KL 검사 |

구현 메모: [Stage1 PUSH](reference/stage1_push.md).

## Stage1 PUSH 360도 접근

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_v2_fixed_push_360.yaml` | 360도 접근·장거리 목표 전용 설정 |
| `tokenhsi/utils/push_navigation.py` | 단계별 범위·겹침 없는 초기 배치·지정 면까지의 우회 거리 |
| `tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_{train,test,vnc}.sh` | 기존 PUSH 모델 전이·학습·평가·viewer |
| `tokenhsi/tests/test_push_navigation.py` | 배치·접근 보상·단계/시간 제한·전이 계약 검증 |

구현 메모: [PUSH 360](reference/stage1_push_360.md).

추론 RSI 원본/보정 선택: [USAGE](USAGE.md#push-추론-rsi-자세).

구현 전 설계: [PUSH 초기 충돌과 크기별 RSI 보정](reference/stage1_push_size_rsi_plan.md).

참조 모션 영상: [25개 MP4·RSI 후보 목록](../output_etc/check/push_ref_motion_mp4_20261020/videos/README.md).

정밀 진단: [PUSH RSI 보정 전후·바닥·충돌 대조 영상과 분석](reference/stage1_push_rsi_collision_audit.md).

## Stage1 PUSH 원본 RSI·가변 정육면체

- `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_v2_fixed_push_360_raw.yaml`: 원본 RSI·1.04m 접촉점·1.2~1.7m 큐브·25~30kg 전용 설정.
- `tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_{train,test,vnc}.sh`: 기존 모델 전이·학습·추론 진입점.
- `tokenhsi/data/push_raw_cube120to170_mass25to30_bank.json`: 원본 후보 시각·물성 계약.
- `push_rsi.py`의 충돌 형상 support와 `face_retreat`: 원본 자세를 유지하는 최소 수평 후퇴.
- `tokenhsi/tests/test_push_raw.py`: 크기/질량·접촉 높이·후퇴·checkpoint 호환 검사.

구현·검증: [PUSH raw](reference/stage1_push_raw.md).

## Stage1 PUSH 원본 RSI·직육면체

- `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_v2_fixed_push_360_raw_cuboid.yaml`: raw 기반 push 상자 축별 크기(y≥z)·15~20kg·마찰0.3·1200step 설정.
- `tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_{train,test,vnc}.sh`: 학습·추론 진입점.
- `tokenhsi/data/push_raw_cuboid_mass15to20_friction03_bank.json`: raw 후보 재사용·직육면체 물성 계약.
- `push_navigation.py`의 `sample_box_properties`·`max_box_edge`: 축별 크기 샘플링과 장면 간격.
- `tokenhsi/tests/test_push_raw_cuboid.py`: 크기/질량·y≥z·미끄럼 조건·push 상자 한정 마찰·계약 검사.

구현·검증: [PUSH raw](reference/stage1_push_raw.md#직육면체-variant).

## 의자 토크나이저 FT

- `env/tasks/multi_agent/humanoid_chair_ft.py`: 2인 의자·주변 물체/좌판 목표/형상 관측.
- `utils/chair_geometry.py`: 일반·등받이·좌판 표본과 semantic descriptor.
- `learning/multi_agent/chair_ft_{network,agent}.py`: PointNet 토크나이저/전이/동결 검증.
- `chair_assets/catalog.json`: train49/test26 의자와 주변 테이블 descriptor.
- 상위 [README.md](../../README.md): 실행/검증/파일럿 정보.

`../evaluation/climb_mixed/`: 원본 category-balanced CLIMB 혼합 샘플링, metadata 목표, 두 frozen 정책 paired 평가 런처. 에셋과 실제 평가 backend/산출물은 프로젝트 runs/climb_mixed_eval_20261022/에 준비.

현재 기본 물체 평가는 `../evaluation/object_comparison/`의 종류별 독립 SIT/CLIMB 비교다. `climb_mixed/`는 이전 혼합 방식 코드로 보존한다.

## Object comparison 목표 전체FT

`chair_table_eval_full_ft` YAML/train/test/VNC와 `test_chair_table_eval_goal.py`: Chair좌판/Table metadata의 CLIMB 목표를 별도로 선택해 object_comparison과 일치시키며, SIT는 기존 좌판/상판 목표를 유지한다.

## Fine-tuned 정책 점/영역 평가

`evaluation/finetuned/`는 현재35D tokenizer checkpoint를1명/1native물체 평가에 연결한다. Chair26/Table8·SIT/CLIMB·512환경×3seed, root점/영역ever 기준과 trace재채점을 사용한다. 기존public/73k backend와 paired 비교는 하지 않는다.

## Native chair/table RSI 후보 검사

`rsi_native/`는 물체별 retarget, 관절한계 보정·FK, native mesh 정적 검사와 PhysX 짧은 hold 검사를 제공한다. 정적/단독 물리 audit 후 보정 AMP history를 만들어 혼합 FT reset에 연결한다. 절차와 한계는 [README](../rsi_native/README.md)를 따른다.

## Box·Chair·Table 혼합 전체 FT

`box_chair_table_mixed_full_ft` YAML/train/test/VNC는 Box·Chair·Table 목표를 각1/3로 혼합한다. `HumanoidChairMixedFT`가 native 환경의 두 목표를 chair/table 하나씩 배정하고 Box 환경의 크기·reward·RSI는 기존 경로로 유지한다. [설정 상세](reference/box_chair_table_mixed_full_ft.md).

`utils/chair_geometry.py::table_box_token_inputs`는 Table의 관측 bbox 윗면을 metadata로 보정하고 Box tokenizer로 라우팅한다. 물리·reward용 geometry와 Chair 추가 입력은 유지한다.

`rsi_native/view_table_sit.py`는 native Table의 검사 통과 착석 RSI와 SIT 목표점/20cm 구를 로컬 viewer에 표시한다. 조작·명령은 [RSI README](../rsi_native/README.md#로컬-table-sit-착석-확인).

`utils/native_rsi.py`, `rsi_native/build_training_bank.py`, `rsi_native/view_reset_snapshot.py`는 native RSI의 실제 물체 pose 배치·보정 AMP history·두 사람 실제 reset viewer를 제공한다.

`rsi_native/{repair_support,support}.py`와 `utils/rsi_support.py`는 source 발 지면 보정·수평 지지 근접·Box 부유 RSI 재추출을 담당한다. 최신 실행 bank/검증은 [혼합 FT 상세](reference/box_chair_table_mixed_full_ft.md#지지-상태를-반영한-rsi-수정-2026-10-23)를 따른다.

`rsi_native/{front_filter,foot_surface,validate_reset}.py`, `utils/{chair_front,native_foot_guard}.py`는 정면 제한·발/표면 교차·무학습 reset 검증을 담당한다. 최신 review는 [RSI README](../rsi_native/README.md#최신-검증review-학습-보류).

`rsi_native/{ik,ik_preview}.py`는 탈락한 완료 자세의 다리 접촉 IK 보정과 물리 통과 후보의 보정 전/후 snapshot을 제공한다. 기존 실행 RSI cache는 자동 변경하지 않는다.

`tokenhsi/env/tasks/multi_agent/humanoid_chair_sit_only_ft.py`: 78k 기반 전체 FT용 chair SIT 전용 환경. `chair_sit_only_full_ft.yaml` 및 동명train/test/vnc wrapper 사용. 상세 [chair_sit_only_ft.md](reference/chair_sit_only_ft.md).

原본 TokenHSI 학습/공식eval 설정 감사: [reference/tokenhsi_train_eval_settings.md](reference/tokenhsi_train_eval_settings.md). 실제모드별초기화·360도배치·10초Short horizon·공식반복집계 구분.

현재 기본 학습: `tokenhsi/env/tasks/multi_agent/humanoid_tokenhsi_sit_ft.py`, 원본 보상 `tokenhsi/utils/tokenhsi_sit_reward.py`, CLIMB 없는 데이터 `tokenhsi/data/dataset_tokenhsi_sit_only.yaml`. 상세 [reference/tokenhsi_sit_ft.md](reference/tokenhsi_sit_ft.md).

Chair HOLDING+AT: `humanoid_chair_hold_at_ft.py`는 원본 동적 ArmChair 배치, 기존 30D 관측/관계 보상을 사용한다. `hold_at_ft_agent.py`와 `utils/hold_at_ft.py`는 78k strict 전이, 동명 YAML/train/test/vnc는 전용 실행 경로다. [상세](reference/chair_hold_at_ft.md).

`evaluation/hold_at/`는 native Chair HOLDING+AT의30초·512×3 단독평가다. `run_eval.py` 독립사본/seed실행, `backend.py` 첫episode deterministic rollout, `metrics.py` AT위치/ever/재채점/통계를담는다. [실행·명세](../evaluation/hold_at/README.md). 학습/기존viewer와분리된 `chair_hold_at_full_ft_eval.sh` 사용.

`evaluation/table_hold_at/`와 `table_hold_at_full_ft_eval.sh`는 Chair와 같은 AT20cm/30초/512×3 첫episode ever 기준의 Table 전용 평가·최신번호checkpoint 선택·독립사본·trace 재채점을 제공한다. [평가 안내](../evaluation/table_hold_at/README.md).

Table carry RSI: `rsi_native/{table_carry,check_table_carry,merge_table_carry,check_table_rsi_reset}.py`는 B20 기반 native32 팔 IK/mesh 접촉/물리 선별/부분reset 검증을 담당한다. `tokenhsi/utils/table_carry_rsi.py`와 `humanoid_table_carry_rsi_ft.py`는 HOLDING_AT 보상·30D 입력을 유지하며 보정 자세와 AMP history를 적용한다. [상세](reference/table_carry_rsi.md).

`humanoid_table_tokenhsi_ft.py`와 `utils/tokenhsi_table_kernels.py`는 원본Table 보상·종료함수를 우리78k 장면에 연결한다. `table_tokenhsi_ft_agent.py`는 actor 물체토크나이저 adaptation/critic·새판별자 학습과 동결을 관리한다. 전용 `table_tokenhsi_ft.yaml`/train/test/vnc 사용. [설정](reference/table_tokenhsi_ft.md).

`table_adaptation_network.py`/`table_adaptation_agent.py`는 기존78k encoder·action head·scene RMS를동결하고action residual 또는복사초기화한새물체tokenizer만정책학습한다. `table_adapt_{residual,tokenizer}.yaml` 및전용train/test/vnc, CPU검증 `check_table_adaptation_cpu.py`. [설계·실행](reference/table_adaptation_compare.md).

Chair residual: `humanoid_chair_tokenhsi_ft.py`는 Chair 정면 reset과 Table에서 공유하는 원본 carry 보상·종료를 연결한다. `chair_adapt_residual.yaml`과 전용 train/test/vnc/eval은 기존 adaptation network/agent를 재사용한다. CPU 검증은 `check_chair_residual_cpu.py`, 정식 평가는 adapter를 복원하는 `evaluation/hold_at/`이다. [설정](reference/chair_adapt_residual.md).

공개 TokenHSI Table 평가는 `evaluation/table_hold_at/{prepare,worker}.py`에서 원본 policy/에셋 사본과 공통 장면을 연결한다. `evaluation/hold_at/public_policy.py`의 원본265D carry policy를 재사용하며 RTX5090 math attention 및 CPU/GPU action 대조를 제공한다. [명령·기준](../evaluation/table_hold_at/README.md).

`tokenhsi/scripts/multi_agent/chair_after_table.py`는 지정 Table 정상종료·10k checkpoint·GPU 여유를 확인하고 Chair4096 2PPO검증→fresh78k/10k 본학습을 순차 실행하는 단발 예약 supervisor다. 예약 상태는 `chair_residual_active_run.json`, 고정 사본과 로그는 `runs/chair_residual_after_table_20261027/`. [예약·상태 확인](reference/chair_adapt_residual.md#table-종료-후-자동-실행-예약).

Chair 예약 복구는 `runs/chair_residual_after_table_20261028_recovery/`에서 실행한다. scheduler는 실제 출력 resolver로 사전 경로 검사하며 검증 출력은 고정 checkout 내부 `code/output_etc/check/chair_preflight/train`이다. 회귀 검증은 `tokenhsi/tests/test_chair_after_table.py`. 현재 PID/경로는 `chair_residual_active_run.json`이 기준이다.

`evaluation/tokenhsi_native_table/`은 원본 TokenHSI Table 환경·20초/직립/연속3repeat player loop에 우리10k정책을 연결한다. native→우리280D bridge/strict복원은policy.py,실행은worker.py,독립재채점은validate.py. [명세·실행](../evaluation/tokenhsi_native_table/README.md).
