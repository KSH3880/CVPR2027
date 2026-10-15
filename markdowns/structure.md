# 레포 구조

현재 실행 실험은 원본 multi-agent carry(1), 독립 Stage 1(27·28·30·31·32·33·34·35·36·37 및 paired AT/placement·unified 독립 5과제), 협력 Stage 2(29 및 SIT plane 변형)다. 실행 명령과 checkpoint 규칙은 [config.md](config.md)에 있다.

## 진입점

원본 단일 사람 Stage 1의 Carry·Sit·Climb 설정은 [TokenHSI 원본 상세](tokenhsi_original_carry_sit_climb.md)를 참조한다. 초기 배치·크기·보상·성공·평가 집계와 논문/로컬 데이터 차이, 평가 asset별 치수를 정리한 참고 문서다.

새 `stage1_v2`의 적용 범위와 원본과의 학습/평가 차이는 [비교 문서](tokenhsi_vs_stage1_v2_train_eval.md)에 정리한다. 전용 env/train `stage1_v2.yaml`, `stage1_v2_{train,test,vnc}.sh`를 사용하며 기존 stage1과 checkpoint/RSI 캐시를 분리한다.

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/run.py` | 학습·평가 태스크와 알고리즘 등록 |
| `tokenhsi/evaluate_action_tokenhsi.py` | 원본 1인 Carry/Sit/Climb 환경의 Action 관측 bridge·평가·물리 호환·출발 거리 제한 옵션; ON_TOP 확장, 학습/VNC 없음 |
| `tokenhsi/box_cleanup_demo.py` | 지정 rescue checkpoint의 4명·16상자 정리 데모 평가 전용 진입점 |
| `tokenhsi/scripts/multi_agent/` | 현재 실험별 train/test/VNC, `runtime_env.sh`, `run-gui.sh` |
| `joint_carry/scripts/carry_joint_preview.py` | CPU 관절 FK로 pickUp/carryWith/putDown 원본·공동운반 MP4/NPZ 후보·모음 영상 생성, 관절각 유지한 안쪽 배치 옵션; 물리 검증·학습 reset 연결 없음 |
| `joint_carry/scripts/validate_carry_physics.py` | 원본/합성 carry의 실제 32-DOF FK·형상 관통·CPU PhysX 0.1초 RSI 충격·free-root PD 추종 검사, 프레임별 CSV/마스크·보고서 저장 |
| `joint_carry/scripts/correct_carry_candidates.py` | 실제 32-DOF arm IK·바닥 침범 보정, 검사 SHA256 확인 후 독립 RSI snapshot 추출·보정 전후 영상·ZIP; 학습 reset 연결 없음 |
| `joint_carry/scripts/check_handoff.py`, `joint_carry/HANDOFF.md` | 보정본 319개 RSI snapshot의 계약·물리 입력 hash·전달 manifest 검사, 보정본 중심 ZIP 및 다음 구현 인계 |
| `tokenhsi/data/cfg/multi_agent/` | 현재 실험 YAML과 평가 graph 예제 |
| `tokenhsi/data/cfg/train/rlg/` | PPO/AMP/Transformer 학습 설정 |
| `output/<실험>/<run>/` | checkpoint `nn/`, TensorBoard `summaries/`, 진단 `diagnostics/` |
| `output_etc/` | 완료된 짧은 검증·평가·분석 보관본; `archive_manifest_20261010.json`은 이번 14폴더의 원래/현재 경로·파일 SHA256 |
| `joint_carry/` | 전용 코드·인계 문서·보정 NPZ·물리 검사·전달 ZIP; `archive/`에는 보정 전 자료, `README.md`에는 최종 파일 안내 |

## 현재 실험이 사용하는 코드

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py` | 물체 배정·RSI·물리 환경·관측 통합 |
| `tokenhsi/env/tasks/multi_agent/box_cleanup_demo.py`, `tokenhsi/learning/multi_agent/box_cleanup_player.py`, `tokenhsi/utils/box_cleanup_spec.py` | 중앙 16상자 크기 혼합·불규칙 2단 더미·윗단 두 라운드 운반·회색 표시·바닥 안정 판정·옛 rescue checkpoint 평가 로드·기본 10회/1500 step·VNC 결과 저장 |
| `tokenhsi/tests/test_box_cleanup_demo.py` | 8개 상자 배정·100개 혼합 크기 배치의 초기 겹침·목적지 높이·checkpoint 계약·네 명 완료 barrier와 물리 상태 유지 검증 |
| `tokenhsi/env/tasks/multi_agent/edge_ontop_task.py`, `edge_context_task.py` | graph reset, 가까운 시작, 물리 타당성·진단 |
| `tokenhsi/env/tasks/multi_agent/edge_stage1_reward.py` | Stage 1 edge 보상·포화·팀 공유 |
| `tokenhsi/env/tasks/multi_agent/edge_ontop_reward.py`, `edge_interaction_reward.py` | AT·ON_TOP·SIT·CLIMB state/progress와 plane 성공 |
| `tokenhsi/utils/edge_scenario_spec.py` | 독립 물체·goal binding, 3물체·4물체 같은 과제 쌍 및 canonical 독립 5과제 sampler; size RSI의 GPU 배치 graph 생성, task-role 가변 인원 추론·받침 부족 시 과제 재추첨 |
| `tokenhsi/tests/test_generic_task_scenario.py` | M≥1·O≥M graph/과제 packet·받침 중복 방지·크기 적격·부분 reset·1물체 높이 검사 |
| `tokenhsi/utils/size_rsi.py`, `tokenhsi/env/tasks/multi_agent/size_rsi_cache.py` | 행동별 실제 asset 크기·조건부 과제 배정·프레임별 초기 충격 캐시·후반 RSI 선택 |
| `tokenhsi/utils/stage1_v2.py`, `tokenhsi/tests/test_stage1_v2.py` | v2 전용 RSI/기하 계약·역할별 적층 높이 검사·새 크기/AT 성공·AMP/토큰 셔플 검증; 구버전 범위는 유지 |
| `tokenhsi/utils/stage1_v2_fixed.py`, `tokenhsi/tests/test_stage1_v2_fixed.py` | v2 고정 선반 opt-in 검증·저장 env/sim 복원·legacy v2 sim 보충 및 기존 실험 격리 검사 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_v2_fixed.yaml`, `tokenhsi/scripts/multi_agent/stage1_v2_fixed_{train,test,vnc}.sh` | 고정 선반 전용 실행 설정·2048환경 train·로컬/서버 viewer, 별도 output |
| `output_etc/v2_fixed_validation_20261015/` | 운영 변경 전 후보 물리/RSI/AMP/학습·resume·속도 검사와 최종 실행 경로 검증; `validation.json`에 결과/제한 기록 |
| `tokenhsi/tests/smoke_stage1_v2_training.py`, `output_etc/stage1_v2_check_20261014/` | 2048환경 실제 학습/resume에서 reset 모션·AMP 이력·정규화 라벨·PPO 행별 family 정렬 검사, 실행/캐시/시뮬레이션 검증 JSON |
| `output_etc/stage1_rsi_height_audit_20261014/` | Climb 0.20~0.70m 확대 전 격리 물리 감사: `audit.py`, 독립 기하 `analyze.py`, 운영 helper 대조 `validate.py`, `validation.json`; 운영 config/캐시 미변경·정책 성능 실험 아님 |
| `output_etc/action_climb_height_eval_20261014/` | Frozen Action 55k의 실제 Climb 높이 0.40~0.70m 평가: `evaluate.py`, 독립 trace 재집계 `analyze.py`, `summary.json`·`validation.json`; 높이당 XY 81조합·486시도, 출발 1~2m·운영 설정 미변경 |
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

Task MLP의 전용 env/train YAML과 train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_task_mlp` 이름을 쓴다. Task message와 `task_role_spec.py`의 task packet·3연결·RSI/AMP 경로를 공유하고, `task_role_encoder.py`의 `TaskRoleMLPFusion`은 메시지 없이 task·출발 역할·도착 역할을 MLP bias로 인코딩한다. `tests/test_task_role_mlp.py`는 env 설정 동일성·메시지 부재·MLP 출력/gradient·순열·보상·checkpoint 분리를 검증한다.

Task MLP split의 전용 env/train YAML과 train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_task_mlp_split` 이름을 쓴다. `TaskRoleMLPFusion(split_tasks=True)`는 역할 embedding을 공유하고 task별 네 MLP·projection을 적용한다. `tests/test_task_role_mlp_split.py`는 task별 weight/gradient 독립성·공유 weight 복제 시 출력/gradient 동등성·순열·RSI/AMP/보상 동일성·checkpoint 격리를 검증한다.

Task embedding의 전용 env/train YAML과 train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_task_embedding` 이름을 쓴다. `task_role_encoder.py`의 `TaskTypeEmbeddingBias`는 task/NONE/SELF의 6개 embedding을 H/O/G 타입 쌍·layer/head별 공유 projection으로 변환한다. `TaskTypeEmbeddingFusion`은 canonical task 연결에 bias를 적용하며 이후 공통 토큰/GTA 순열 경로를 따른다. `tests/test_task_type_embedding.py`는 카테고리·타입 공유, 배경/역방향/padding, endpoint 재매핑·출력/gradient 순열, RSI/AMP/보상 동일성·checkpoint 격리를 검증한다.

Task embedding action의 전용 env/train YAML과 train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_task_embedding_action` 이름을 쓴다. `TaskTypeEmbeddingBias(zero_background=True)`는 4개 task만 인코딩하고 SELF/NONE bias를 0으로 고정한다. `TaskActionFusion`은 GTA 복귀·head concat 뒤 사람 토큰에 상태/task concat residual MLP를 적용한다. Actor endpoint를 토큰 순열에 맞게 재매핑하고 네 layer가 MLP를 공유한다. `tests/test_task_embedding_action.py`는 샘플링/RSI/AMP 계약, 주입 위치·padding·사람별 task binding·출력/gradient 순열·독립 업데이트·checkpoint 격리를 검증한다.

Task/Role gate + Entmax의 전용 env/train YAML·train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax` 이름을 쓴다. `learning/multi_agent/task_role_gate.py`는 task embedding·layer별 4D block gate·sparse K/V correction·정확한 Entmax 1.5와 attention 진단을 제공한다. Gate 조회는 학습 시 one-hot GEMM gradient 합산, no-grad/frozen/FP32 TF32 시 compact index lookup을 사용하며 16D 조회 뒤 block을 확장한다. `RelationTransformerLayer`는 GTA 뒤 K/V 보정과 전체 scene 정규화를 적용하고, `ma_agent.py`는 task embedding/gate gradient norm을 추가 기록한다. `utils/task_role_spec.py`·`unified_training.py`는 bias/message 비활성화와 AMP/RSI 계약을, `relation_task_spec.py`는 checkpoint 격리를 검증한다. `tests/test_task_role_gate_entmax.py`는 dense/reference 수식·gradient·GTA 순서·role/padding·토큰/edge/task 순열·독립 업데이트·기존 runtime 계약을 확인한다.

Action CPA의 전용 env/train YAML·train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_task_embedding_action_cpa` 이름을 쓴다. `env/tasks/multi_agent/collision_reward.py`는 사람 간 CPA 위험·config 검증·checkpoint collision 계약을 제공하며 `humanoid_ma_carry.py`에서 static/CPA를 선택한다. `ma_agent.py`/`ma_players.py`는 저장·복원 시 계약을 확인하고, `tests/test_agent_cpa_collision.py`는 기하·보상 교체·기존 static 호환·실험 격리를 검증한다. 모델은 기존 Action과 같다.

Stage1 전용 env/train YAML은 각각 `data/cfg/multi_agent/stage1.yaml`·`data/cfg/train/rlg/stage1.yaml`, 실행은 `scripts/multi_agent/stage1_{train,test,vnc}.sh`다. stage1/v2 test와 이를 호출하는 VNC는 checkpoint run의 `relation_config.yaml`을 기본 env로 선택하며 전용 `STAGE1_ENV_CONFIG`/`STAGE1_V2_ENV_CONFIG`로 재정의할 수 있다. `TaskTypeEmbeddingBias(shared_types=True)`와 기존 Action MLP/CPA를 사용한다. `utils/stage1_platforms.py`는 엄격한 실행 계약·TokenHSI 높이 추첨·graph에 따른 carry payload/AT 목적지 선반 reset을 담당하며 `edge_ontop_task.py`의 물리 장면 검사 전에 호출한다. `tests/test_stage1.py`는 공유 bias 수식/gradient·순열·checkpoint·RSI/선반 범위·부분 reset을, `tests/smoke_stage1_sim.py`는 저장 모델의 과제/RSI별 실제 reset·rollout·CPA 대조·GPU 선반 접촉/재배치를 검증한다. `utils/stage1_sampling.py`는 선택형 TokenHSI 접근 거리·공통 목표 XY·ON_TOP 받침 배치와 구/신 RSI resume 검사를 제공한다. `tests/test_stage1_sampling.py`는 분포·graph binding·부분 reset·구버전 계약을, simulator smoke는 실제 초기 거리까지 검사한다.

`stage1_v2_fixed`는 `humanoid_ma_carry.py`의 opt-in asset/첫 reset 분기와 `ma_agent.py`의 sim snapshot 저장을 사용한다. `utils/stage1_v2_fixed.py`가 새 wrapper의 저장 env/sim 입력을 해석하며, 기존 v2 checkpoint를 새 고정 선반 환경에서 평가할 때는 고정 선반 선택을 명시적으로 적용한다. 기존 stage1/v2 config의 동작은 유지한다.

## 작업별 확인

- **Stage 1 graph·보상:** 해당 YAML → `edge_scenario_spec.py` → `edge_ontop_task.py` → `edge_interaction_reward.py`·`edge_stage1_reward.py`; 위 Stage 1 테스트로 검증한다.
- **Stage 2 협력·전이:** `approach_stage2_coordination.yaml` → `edge_stage2_spec.py` → `coordination_head.py`·`stage2_transfer.py`; `test_stage2_coordination.py`로 검증한다.
- **실행·GPU:** [config.md](config.md), `runtime_env.sh`, `mps/`, `run-gui.sh`를 확인한다.
- **지표 해석:** [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 확인한다.
