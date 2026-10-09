# 레포 구조

현재 실행 실험은 원본 multi-agent carry(1), 독립 Stage 1(27·28·30·31·32·33·34·35·36·37 및 paired AT/placement·unified 독립 5과제), 협력 Stage 2(29 및 SIT plane 변형)다. 실행 명령과 checkpoint 규칙은 [config.md](config.md)에 있다.

## 진입점

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/scripts/prepare_joint_carry.py`, `joint_carry/runtime_manifest.json` | Clone 후 기존 데이터 링크·필수 파일·현재 AMP/RSI 해시 검사; 두 LocoAMP train/test 및 VNC 경로에서 실행 |
| `tokenhsi/run.py` | 학습·평가 태스크와 알고리즘 등록 |
| `tokenhsi/box_cleanup_demo.py` | 지정 rescue checkpoint의 4명·16상자 정리 데모 평가 전용 진입점 |
| `tokenhsi/scripts/multi_agent/` | 현재 실험별 train/test/VNC, `runtime_env.sh`, `run-gui.sh` |
| `tokenhsi/data/cfg/multi_agent/` | 현재 실험 YAML과 평가 graph 예제 |
| `tokenhsi/data/cfg/train/rlg/` | PPO/AMP/Transformer 학습 설정 |
| `output/<실험>/<run>/` | checkpoint `nn/`, TensorBoard `summaries/`, 진단 `diagnostics/` |
| `output_etc/` | 비활성 검증·분석·이전 영상 보관; `archive_20261009_cleanup.json`에 이동 전/후 경로 기록. `output/`은 본학습4개·RSI cache·현재 전체 보정 영상 유지 |
| `joint_carry/scripts/`, `joint_carry/HANDOFF.md` | 공동운반 RSI 합성·팔 IK·CPU PhysX 검사·인계 안내; 재현 결과는 `output/joint_carry_rsi_rebuild_20261007/`의 `verification.json`, `corrected/before_after/`, `pd_videos/` |
| `joint_carry/teamhoi_reference/`, `joint_carry/scripts/preview_teamhoi_reference.py` | 공식 TeamHOI 후진9개·옆걸음3개와 출처/해시; CPU 원본 FK 영상 생성 → `output_etc/teamhoi_reference_preview/` (AMP 미연결) |
| `joint_carry/teamhoi_retarget_pilot/`, `joint_carry/scripts/{retarget_teamhoi_reference,check_teamhoi_retarget,preview_teamhoi_retarget}.py` | 후진·옆걸음 각1개를 현재 asset에 맞춘 IK/접지·속도 보정, AMP/2048환경 GPU 물리 검사 및 비교 영상 → `output_etc/teamhoi_retarget_pilot_check/`; 학습 미연결 |
| `joint_carry/teamhoi_retarget_all/` | 과도한 무릎 굽힘이 있는 이전12개 보정본(현재 학습 미사용); 위 스크립트의 `--all` → `output_etc/teamhoi_retarget_all_check/` |
| `joint_carry/teamhoi_retarget_posture_pilot/`, `joint_carry/scripts/compare_teamhoi_gait.py` | 원본 무릎 자세를 보존하는 재보정2개·기존 걷기5개 대조; 위3개 스크립트 `--posture-first`, 결과/4열 영상 → `output_etc/teamhoi_retarget_posture_pilot_check/`; 학습 미연결 |
| `joint_carry/teamhoi_retarget_posture_all/` | 현재 LocoAMP용 자세 보존12개·해시·AMP9/3개 목록; 보정/검사/비교 영상 `--all --posture-first`, 모아보기 추가 `--montage`, 결과 → `output/teamhoi_retarget_posture_all_check/` |
| `output/joint_carry_gpu5_check_20261007/check_gpu.py`, `report.json`, `per_env.npz` | GPU PhysX·2048환경의 paired RSI 319개 초기 충격·공통 yaw·부분 reset 검사; 학습 환경 연결·AMP 검증과 구분 |

## 현재 실험이 사용하는 코드

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py` | 물체 배정·RSI·물리 환경·관측 통합 |
| `tokenhsi/env/tasks/multi_agent/scene_features.py` | clean-scene 관측 보조·공유 물체/목표 소유자·노란색·ON_TOP 큐브/SIT 원/CLIMB 십자 마커 |
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
| `tokenhsi/utils/joint_carry_spec.py`, `tokenhsi/learning/multi_agent/task_coordination.py`, `tokenhsi/tests/test_joint_carry.py` | Joint AT/ON_TOP sampler·config 계약·task CA·NONE/SELF/COUPLED bias·전이·순열 검증 |
| `tokenhsi/tests/test_joint_carry_alignment.py`, `approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding` env/train·train/test/VNC | 공동 HOLDING 중 farthest-human heading 공유 보상·포화 거리·좌표/순열·checkpoint 계약 검사; 구현은 `joint_carry_reward.py`, 통합은 `edge_context_task.py` |
| `tokenhsi/utils/joint_carry_amp.py`, `tokenhsi/tests/{test_joint_carry_amp_ablation,check_joint_carry_amp_runtime}.py` | AMP 전용 후진/옆걸음 라이브러리·80/10/10·데이터 해시·CA/head-only checkpoint 격리; CPU 및 실제 Mixed80 reset/AMP 검증 |
| `tokenhsi/utils/joint_carry_rsi.py`, `tokenhsi/env/tasks/multi_agent/joint_carry_reset.py`, `joint_carry_reward.py`, `tokenhsi/tests/test_joint_carry_runtime.py` | 보정 paired snapshot 로드·공통 좌표 변환·공동/부분 reset·AMP 초기 이력·반대 앵커 placement gate·agent CPA 보상 검증 |
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

Task MLP의 전용 env/train YAML과 train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_task_mlp` 이름을 쓴다. Task message와 `task_role_spec.py`의 task packet·3연결·RSI/AMP 경로를 공유하고, `task_role_encoder.py`의 `TaskRoleMLPFusion`은 메시지 없이 task·출발 역할·도착 역할을 MLP bias로 인코딩한다. `tests/test_task_role_mlp.py`는 env 설정 동일성·메시지 부재·MLP 출력/gradient·순열·보상·checkpoint 분리를 검증한다.

Task MLP split의 전용 env/train YAML과 train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_task_mlp_split` 이름을 쓴다. `TaskRoleMLPFusion(split_tasks=True)`는 역할 embedding을 공유하고 task별 네 MLP·projection을 적용한다. `tests/test_task_role_mlp_split.py`는 task별 weight/gradient 독립성·공유 weight 복제 시 출력/gradient 동등성·순열·RSI/AMP/보상 동일성·checkpoint 격리를 검증한다.

Task embedding의 전용 env/train YAML과 train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_task_embedding` 이름을 쓴다. `task_role_encoder.py`의 `TaskTypeEmbeddingBias`는 task/NONE/SELF의 6개 embedding을 H/O/G 타입 쌍·layer/head별 공유 projection으로 변환한다. `TaskTypeEmbeddingFusion`은 canonical task 연결에 bias를 적용하며 이후 공통 토큰/GTA 순열 경로를 따른다. `tests/test_task_type_embedding.py`는 카테고리·타입 공유, 배경/역방향/padding, endpoint 재매핑·출력/gradient 순열, RSI/AMP/보상 동일성·checkpoint 격리를 검증한다.

Carry-only distillation은 `approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill` env/train YAML과 train/test/VNC를 쓴다. `task_role_spec.py`의 carry-only 조건부 확률과 `humanoid_ma_carry.py`의 carry 크기·AMP family 분기를 사용한다. `learning/multi_agent/stage1_unified_teacher.py`는 graph를 원본 teacher 관측으로 변환하고, `distillation.py`는 Gaussian KL·gradient 검사를 제공한다. `ma_agent.py`는 rollout label·minibatch·loss를 연결한다. `tests/test_carry_distillation.py`가 carry 샘플링·RSI/보상 유지·AMP family·목표/물체/goal binding·label 정렬·teacher 동결·학생 gradient·checkpoint 분리를 검증한다.

네 과제 전체 distillation은 `approach_scenario_stage1_unified_size_rsi_task_embedding_distill` env/train YAML·train/test/VNC를 쓴다. 원래 task embedding의 과제·크기·RSI·AMP를 유지하고 위 teacher/KL 경로를 공유한다. `tests/test_task_embedding_distillation.py`가 원본 설정 동일성·checkpoint 분리·16개 과제 쌍의 teacher 라우팅/목표·SIT 방향·네 task gradient·AMP family를 검증한다.

## 작업별 확인

- **Stage 1 graph·보상:** 해당 YAML → `edge_scenario_spec.py` → `edge_ontop_task.py` → `edge_interaction_reward.py`·`edge_stage1_reward.py`; 위 Stage 1 테스트로 검증한다.
- **Stage 2 협력·전이:** `approach_stage2_coordination.yaml` → `edge_stage2_spec.py` → `coordination_head.py`·`stage2_transfer.py`; `test_stage2_coordination.py`로 검증한다.
- **실행·GPU:** [config.md](config.md), `runtime_env.sh`, `mps/`, `run-gui.sh`를 확인한다.
- **지표 해석:** [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 확인한다.

Joint carry Stage 2는 `approach_stage2_joint_carry_task_embedding` env YAML·train/test/VNC와 `amp_ma_stage2_joint_carry_task_embedding` train YAML을 쓴다. `humanoid_ma_carry.py`가 joint 전용 reset·reward·carry AMP를 연결하고 `stage2_transfer.py`가 Stage 1 actor/RMS 동결과 head 확장을 담당한다. GPU 연결 검사는 `output/approach_stage2_joint_carry_task_embedding_check/`의 `check_reset.py`, `reset_check.json`, `check_saved.py`, `saved_check.json`에 있다.

Mixed80 공동/단독 운반은 `approach_stage2_joint_carry_mixed80_task_embedding` env YAML·train/test/VNC와 `amp_ma_stage2_joint_carry_mixed80_task_embedding` train YAML을 쓴다. `utils/mixed_carry_spec.py`는 고정 환경 분할·독립 graph를, `env/tasks/multi_agent/mixed_carry_reset.py`는 공동/단독 reset의 단일 commit과 Stage 1 크기별 RSI cache 재사용을 담당한다. `joint_carry_reward.py`가 공유 여부에 따라 반대 앵커 또는 자기 HOLDING gate를 적용한다. CPU 검사는 `tests/test_mixed_carry.py`, GPU 결과는 `output/approach_stage2_joint_carry_mixed80_task_embedding_check/`에 있다.

BEFORE Stage 2는 `approach_stage2_before_task_embedding` env YAML·train/test/VNC와 `amp_ma_stage2_before_task_embedding` train YAML을 쓴다. `utils/before_spec.py`는 공통 크기·고정 80/20 환경·세 dependency graph·AMP family 분포를, `env/tasks/multi_agent/before_reward.py`는 현재 AT prerequisite과 gated success/포화를 담당한다. `before_reset.py`는 공유 물체 reference 정렬·속도/AMP history 변환·단일 물리 commit을, `before_rsi_cache.py`는 실제 source/support 크기의 물리 RSI 선별을 담당한다. 기존 `task_coordination.py`의 별도 BEFORE 모드에 비대칭 bias를 추가하며 joint checkpoint 형식은 유지한다. CPU 검증은 `tests/test_before.py`, GPU 초기 물리/부분 reset·AMP 검사는 `tests/check_before_runtime.py`이며 최종 결과는 `output/approach_stage2_before_task_embedding_check_final/`에 저장한다. 물리 안정성 진단은 pass 수를 별도로 기록하며 모든 장면 통과를 의미하지 않는다.
