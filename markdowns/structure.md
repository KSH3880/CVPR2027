# 레포 구조

Multi-agent 실험은 `stage1`, `stage1_v2`, `stage1_v2_fixed`, `stage2`와 공유 받침 성공 기준 비교 두 개다. 현재 raw PUSH의 실행 명령·추론 옵션은 [config.md](config.md), 다른 실험과 추가 옵션·checkpoint 규칙은 [USAGE.md](USAGE.md)를 따른다.

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
| `output_etc/check/push_mass_sweep_20261021/view_15kg.sh`, `view.py` | 기존 raw checkpoint의15kg 비교용 noVNC 실행기 |
| `output_etc/check/push_attention_six_20261022/` | 실제1환경·6명·5과제 attention 진단 실행기, H/O/G 전체 행렬·레이어/head/시간 뷰어·영상 |
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
| `tokenhsi/tests/{test_stage1_push,test_push_teacher}.py` | 기존 정책 보존·AMP·접촉·KL·계수0의 teacher 생략 검사 |

구현 메모: [Stage1 PUSH](reference/stage1_push.md).

## Stage1 PUSH 360도 접근

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_v2_fixed_push_360.yaml` | 360도 접근·장거리 목표 전용 설정 |
| `tokenhsi/utils/push_navigation.py` | 크기/질량 샘플링·단계별 범위·겹침 없는 초기 배치·지정 면까지의 우회 거리 |
| `tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_{train,test,vnc}.sh` | 기존 PUSH 모델 전이·학습·평가·viewer |
| `tokenhsi/tests/test_push_navigation.py` | 배치·접근 보상·단계/시간 제한·전이 계약 검증 |

구현 메모: [PUSH 360](reference/stage1_push_360.md).

추론 RSI 원본/보정 선택: [USAGE](USAGE.md#push-추론-rsi-자세).

구현 전 설계: [PUSH 초기 충돌과 크기별 RSI 보정](reference/stage1_push_size_rsi_plan.md).

참조 모션 영상: [25개 MP4·RSI 후보 목록](../output_etc/check/push_ref_motion_mp4_20261020/videos/README.md).

정밀 진단: [PUSH RSI 보정 전후·바닥·충돌 대조 영상과 분석](reference/stage1_push_rsi_collision_audit.md).

## Stage1 PUSH 원본 RSI·가변 정육면체

- `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_v2_fixed_push_360_raw.yaml`: fixed78000 전이·원본 RSI·고정 면 XY 접촉 보상·1.2~1.7m 큐브·크기에 선형 대응하는12~20kg 설정.
- `tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_{train,test,vnc}.sh`: 기존 모델 전이·학습·추론 진입점.
- `tokenhsi/data/push_raw_cube120to170_mass12to20_linear_plate09_03_full_bank.json`: 원본 후보 시각·크기별12~20kg·본체0.9/밑판0.3·상자와 같은 가로/세로 물성 계약. 이전 bank는 기존 checkpoint용으로 보존한다.
- `tokenhsi/utils/push_box_asset.py`: 한 강체·두 충돌 형상의 두께1cm 밑판 URDF 생성. 현재 밑판은 상자와 같은 가로/세로이며 이전 안쪽 밑판도 지원한다. 생성 파일은 실행 출력의 `push_assets/`에 저장한다.
- `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_v2_fixed_push_360_raw_mass20to25.yaml`: raw PUSH의 크기별 질량만20~25kg으로 높인 비교 설정. 마찰·보상·RSI·1800step을 유지한다.
- `tokenhsi/data/push_raw_cube120to170_mass20to25_linear_plate09_03_full_bank.json`: 같은 원본 RSI 후보를 사용하는20~25kg 물성 계약.
- `tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_mass20to25_{train,test,vnc}.sh`: 질량 variant의 독립 학습·평가·VNC 진입점.
- `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_v2_fixed_push_360_raw_cuboid.yaml`, 전용 `*_cuboid_{train,test,vnc}.sh`: X1.2~1.7m·Y/Z1.45~1.7m·Y≥Z·부피별15~20kg·1200step 설정과 실행 진입점.
- `tokenhsi/data/push_raw_cuboid_x120to170_yz145to170_mass15to20_volume_plate09_03_bank.json`: 같은 원본 RSI 후보와 직육면체 물성 계약.
- `tokenhsi/tests/test_push_cuboid.py`: 축별 범위·Y≥Z·부피별 질량·직육면체 관성·명시적 checkpoint 전이 검사.
- `push_rsi.py`의 충돌 형상 support와 `face_retreat`: 원본 자세를 유지하는 최소 수평 후퇴.
- `push_reward.py`의 `face_xy_distances`: 유한 지정 면의 XY 투영 거리·양손 최대 거리 state·접촉력 판정.
- `tokenhsi/tests/test_push_raw.py`: 크기/질량·면 XY 거리/접촉력·회전/높이 불변성·후퇴·checkpoint 호환 검사.
- `tokenhsi/tests/test_push_box_asset.py`: 밑판의 경계·질량/중심/관성·단일 강체 계약 검사.

구현·검증: [PUSH raw](reference/stage1_push_raw.md).

현재 cuboid 물성·PUSH/AT 보상식·페널티·AMP 합산: [Cuboid PUSH 보상 설계](reference/push_cuboid_reward_design.md).

RSI 균등 거리 실험은 `stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform` 환경/학습 YAML·전용 train/test/VNC를 사용한다. 초기화·전이 계약 검사는 `test_push_cuboid.py`에 있으며, 변경 범위는 [PUSH RSI 균등 거리](reference/stage1_push_rsi_uniform.md)를 따른다.

## Cuboid RSI uniform10–15kg·20cm·130°

- `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130.yaml`와 전용 `*_{train,test,vnc}.sh`: 새 variant 설정과 실행 진입점.
- `tokenhsi/data/push_raw_cuboid_x120to170_yz145to170_mass10to15_volume_plate09_03_bank.json`: 같은 원본 후보를 사용하는10–15kg 물성 계약.
- `tokenhsi/scripts/multi_agent/push_uniform_18k_monitor.{py,sh}`: 기존 서비스30분 감시·18K 평가·조건부 종료/새 서비스 시작.
- `tokenhsi/tests/test_push_light130.py`: 질량·spawn 면/각도·PUSH 성공 보상·기존 과제 보존 검증.
- [설정·전환 기준](reference/stage1_push_light130.md).

## Stage1 jointpush

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_jointpush.yaml` | 기존 Stage1 + KOO PUSH 스크래치 설정 |
| `tokenhsi/env/tasks/multi_agent/{humanoid_ma_stage1_jointpush,jointpush_reward}.py` | 혼합 과제·학습 RSI/평가 loco·면 CONTACT/AT 보상 |
| `tokenhsi/utils/{stage1_jointpush,jointpush_scene,jointpush_geometry,jointpush_rsi,jointpush_rsi_geometry}.py` | 크기/과제·360° 거리/간섭·KOO RSI 연결 |
| `tokenhsi/data/{dataset_stage1_jointpush.yaml,stage1_jointpush_rsi_bank.json}` | 모션 경로·원본 KOO 선별 후보 |
| `tokenhsi/data/jointpush_motions/dataset_bones_push{,180}` | 로컬 PUSH 데이터 심링크; 원본 NPY는 읽기 전용으로 사용하며 Git 제외 |
| `tokenhsi/scripts/multi_agent/stage1_jointpush_{train,test,vnc}.sh` | 스크래치 학습·기존 PTH 평가·VNC |
| `tokenhsi/tests/test_stage1_jointpush.py` | 샘플링·보상·기존 PTH 호환 검사 |
| `tokenhsi/tests/check_stage1_jointpush.py` | 실제 reset·물성·AMP·PTH 정책·스크래치 gradient 검사 |

`stage1_jointpush_density3_cone90`는 같은 task를 사용하는 밀도3·목표±45° variant다. 환경/학습 YAML과 전용 `{train,test,vnc}.sh`가 같은 이름으로 분리되며, `test_stage1_jointpush_density3_cone90.py`가 질량·cone 거부·기존 설정 보존·RSI 후보 재사용을 검사한다.

`stage1_jointpush_density3_cone90_finetune` 환경/학습 YAML·전용 train/test/VNC는78k 전이와 PUSH50% 혼합을 사용한다. `ma_agent.py`가 `stage1_push_network.transfer()`로 초기화하며, `test_stage1_jointpush_transfer.py`가 기존 과제 출력 보존·50/50 혼합을 검사한다.

세부 설정과 원본 출처는 [Stage1 jointpush](reference/stage1_jointpush.md)를 따른다.

`stage1_jointpush_density3_cone90_finetune_atprogress_gate` 환경/학습 YAML·전용 train/test/VNC는 PUSH AT progress에 step당 수평 목표 거리 감소 1mm 초과 gate를 적용한다. `jointpush_reward.gate_push_at_progress`와 task의 직전 상자 위치를 사용하며, `test_stage1_jointpush_atprogress_gate.py`가 임계값·혼합 과제·성공 포화·reset 경계를 검사한다.

## Stage2 jointpush

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/utils/stage2_jointpush.py` | 순차 그룹·단독 과제 샘플링, 구역 할당/상태, 공동 뒤쪽 면 gate |
| `tokenhsi/learning/multi_agent/stage2_jointpush_network.py` | PUSH 가중치 전이·동결, 실제 공유 그룹 기반 coordination |
| `tokenhsi/tests/test_stage2_jointpush.py` | 그룹 분포·구역/GTA용 상태·gate·전이/identity 검사 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage2_jointpush.yaml` | 4명·1024환경·8 object 토큰, Stage1 PUSH 전이 |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_stage2_jointpush.py`, `stage2_jointpush_reward.py` | 24개 후보 bank·미사용 격리, 물리 공유/구역 관측, gate 적용 AT |
| `tokenhsi/utils/stage2_jointpush_scene.py` | 사람 고정·32후보 배치, coupled 뒤쪽/목표 cone·swept rectangle, 단독 Stage1 경로 검사 |
| `tokenhsi/env/tasks/multi_agent/push_goal_markers.py` | jointpush PUSH 목표20cm 구형 테두리·바닥 링/세로선, 공동 노란색·그룹당1개 |
| `tokenhsi/scripts/multi_agent/stage2_jointpush_{train,test,vnc}.sh` | GPU5 학습·Stage1 제로샷/Stage2 추론 |
| `tokenhsi/tests/check_stage2_jointpush.py` | 학습 업데이트 없는 전이·1024환경 reset/추론·부분 reset·viewer 검사 |

사용자가 미사용 후보 격리를 승인했다. 상세 계약은 [Stage2 jointpush](reference/stage2_jointpush.md), 실행법은 [config](config.md)에 있다.

## Stage2 jointpush RSI

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage2_jointpush_rsi.yaml` | 단독 Stage1 RSI·coupled 그룹 RSI 학습, 평가 loco100% |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_stage2_jointpush_rsi.py` | 기존 Stage2의 RSI 초기화 연결·물리 상자/목표 유지 |
| `tokenhsi/env/tasks/multi_agent/jointpush_solo_rsi_cache.py` | 4명·24 actor bank에 맞춘 기존 과제 RSI 캐시/후반 프레임 샘플링 |
| `tokenhsi/utils/stage2_jointpush_rsi.py` | 그룹별 RSI 선택·구역 뒤쪽 손 목표·사람 간 충돌 검사 |
| `tokenhsi/scripts/multi_agent/stage2_jointpush_rsi_{train,test,vnc}.sh` | GPU5 학습·Stage1 제로샷/Stage2 평가·VNC, `PUSH_RSI=1` 진단 옵션 |
| `tokenhsi/tests/{test,check}_stage2_jointpush_rsi.py` | CPU 계약·실제 초기화/AMP/평가·optimizer 업데이트 없는 학습 연결 검사 |

초기화·캐시·검증 범위는 [Stage2 jointpush RSI](reference/stage2_jointpush_rsi.md)를 따른다.

## Stage2 jointpush 목표 포화 비교

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage2_jointpush{,_rsi}_goal_saturation.yaml` | 일반/RSI 각각 목표 도착 시 gate 면제, 기존 설정과 출력 분리 |
| `tokenhsi/scripts/multi_agent/stage2_jointpush{,_rsi}_goal_saturation_{train,test,vnc}.sh` | 두 설정의 학습·평가·VNC 실행 |
| `tokenhsi/tests/test_stage2_jointpush_goal_saturation.py` | 목표 도착·손 떼기·목표 이탈·높이 오차 및 기존 설정 동등성 검사 |

상세 규칙은 [Stage2 jointpush](reference/stage2_jointpush.md#목표-도착-시-gate-면제-설정)를 따른다.

## Stage2 jointpush 내부 action adapter

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage2_jointpush_goal_saturation_adapter.yaml` | 일반 loco·목표 포화 보상에 동결 action head와 두 병렬 adapter 적용 |
| `tokenhsi/learning/multi_agent/stage2_jointpush_network.py` | 선택적 내부 adapter 생성·전이·동결·추론 |
| `tokenhsi/scripts/multi_agent/stage2_jointpush_goal_saturation_adapter_{train,test,vnc}.sh` | 별도 학습·평가·VNC 출력 |
| `tokenhsi/tests/test_stage2_jointpush_adapter.py` | 설정 동등성·14k 전이 초기 출력·동결/gradient·모델 복원 검사 |

상세 구조는 [Stage2 jointpush](reference/stage2_jointpush.md#내부-action-adapter-비교-설정)를 따른다.

## Stage2 jointpush 밀도3·단독 목표 ±45°

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage2_jointpush_goal_saturation_density3_cone90.yaml` | 기존 목표 포화 설정에 밀도3·단독 cone90·Stage1 8000 전이 적용 |
| `tokenhsi/scripts/multi_agent/stage2_jointpush_goal_saturation_density3_cone90_{train,test,vnc}.sh` | 전용 학습·로컬 추론·서버 VNC 실행 |
| `tokenhsi/tests/test_stage2_jointpush_density3_cone90.py` | 설정 차이 제한·단독/공동 질량·checkpoint 계약 검사 |

상세 설정은 [Stage2 jointpush](reference/stage2_jointpush.md#밀도3단독-목표-45-설정)를 따른다.
