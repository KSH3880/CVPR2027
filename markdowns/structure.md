# 레포 구조

현재 실행 실험은 원본 multi-agent carry(1), 독립 Stage 1(27·28·30·31·32·33·34·35·36·37 및 paired AT/placement·unified 독립 5과제), 협력 Stage 2(29·SIT plane·unified semantic/owner HOLDING·rescue KLClimb50·Shared9 변형)다. 실행 명령과 checkpoint 규칙은 Stage 1 [config.md](config.md), Stage 2 [config_stage2.md](config_stage2.md)에 있다.

## 진입점

| 경로 | 역할 |
| --- | --- |
| `markdowns/config.md`, `markdowns/config_stage2.md` | Stage 1·Stage 2별 실험 목록과 학습·로컬 평가·서버 VNC 명령 |
| `tokenhsi/docs/stage2_diagnostics.md` | Stage 2 실측·RSI 충돌 검사·attention 분석 재현 명령; 본학습 실행 가이드와 분리 |
| `tokenhsi/run.py` | 학습·평가 태스크와 알고리즘 등록 |
| `tokenhsi/scripts/multi_agent/` | 현재 실험별 train/test/VNC, `runtime_env.sh`, `run-gui.sh` |
| `tokenhsi/scripts/multi_agent/measure_shared_support.py` | 같은 받침의 ON_TOP/SIT/CLIMB 6조합 정적 공간 측정·별도 CPU PhysX 접촉 확인; 정책 rollout과 구분 |
| `tokenhsi/scripts/multi_agent/measure_shared_rectangles.py` | 운반 결과의 XYZ 목록으로 공유 6조합 정적 공간 측정; source 높이·0/4cm 여유·별도 CPU PhysX witness 확인 |
| `tokenhsi/scripts/multi_agent/audit_shared_scenarios.py` | 공유 9조합 RSI·Stage 1-only rollout, 독립 AT 운반 grid·4과제 독립 평가·공유 6조합 진단; override는 진단 프로세스에 한정 |
| `tokenhsi/scripts/multi_agent/summarize_transport_sharing.py` | 운반 실측의 후속 검사 크기 목록 생성·공유 6조합 공간 교집합 JSON/CSV 집계; Python 표준 라이브러리만 사용 |
| `tokenhsi/scripts/multi_agent/analyze_stage2_attention.py`, `render_stage2_attention.py` | Rescue checkpoint별 실제 rollout·동일 관측 bank의 CA 추세·추론 개입 비교 및 PNG/HTML 보고서; 진단 subprocess에만 적용 |
| `tokenhsi/data/cfg/multi_agent/` | 현재 실험 YAML과 평가 graph 예제 |
| `tokenhsi/data/cfg/multi_agent/graphs/stage2_three_agent_place_two_climb.yaml` | Rescue Stage 2 3인 평가: H0 HOLDING+AT, H1·H2가 같은 O0에 CLIMB |
| `tokenhsi/data/cfg/train/rlg/` | PPO/AMP/Transformer 학습 설정 |
| `output/<실험>/<run>/` | checkpoint `nn/`, TensorBoard `summaries/`, 진단 `diagnostics/` |

## 현재 실험이 사용하는 코드

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py` | 물체 배정·RSI·물리 환경·관측 통합 |
| `tokenhsi/env/tasks/multi_agent/edge_ontop_task.py`, `edge_context_task.py` | graph reset, 가까운 시작, 물리 타당성·진단 |
| `tokenhsi/env/tasks/multi_agent/scene_features.py`, `tokenhsi/tests/test_ma_viewer_markers.py` | 공통 좌표/pose helper·AT destination 기준 표시·물체 공유 owner·같은 행동/물체의 목표 노란색·SIT 링/ON_TOP 상자/CLIMB 십자·CPU viewer 회귀 검증 |
| `tokenhsi/env/tasks/multi_agent/edge_stage1_reward.py` | Stage 1 edge 보상·포화·팀 공유 |
| `tokenhsi/env/tasks/multi_agent/collision_reward.py`, `tokenhsi/tests/test_agent_cpa_collision.py` | 사람–사람 정적 거리/CPA 위험도·config·checkpoint 계약·CPU 보상 검증; Shared9 CPA 전용 YAML·train/test/VNC는 기존 설정/스크립트 폴더 |
| `tokenhsi/env/tasks/multi_agent/edge_ontop_reward.py`, `edge_interaction_reward.py` | AT·ON_TOP·SIT·CLIMB state/progress와 plane 성공 |
| `tokenhsi/utils/edge_scenario_spec.py` | 독립 물체·goal binding, 3물체·4물체 같은 과제 쌍 및 canonical 독립 5과제 sampler |
| `tokenhsi/utils/unified_training.py` | unified 과제별 AMP 전문가 분포·라벨·리플레이 매칭·설정 검증 |
| `tokenhsi/utils/edge_stage1_spec.py` | Stage 1 variant·semantic/owner-HOLDING packet·후반 CLIMB/ON_TOP putDown RSI 검증 |
| `tokenhsi/utils/edge_stage2_spec.py` | Stage 2 협력 graph·평가 preset·4물체 canonical sampler·Rescue 평가용 M인/O물체 범용 그룹/독립 sampler·Shared9의 9조합 평가 전용 인원 확장 |
| `tokenhsi/utils/stage2_shared_spec.py` | Rescue Shared9 조합·크기별 고정 환경 풀·공유 받침/source 논리→물리 배정·초기 FK body sphere 검사 |
| `tokenhsi/utils/stage2_shared_eval.py` | Shared9 기본 평가: reset별9조합·2인 그룹/홀수 잔여자·자원 제한·역할 호환 고정 크기·그룹별 조합 집계; 학습 샘플러와 분리 |
| `tokenhsi/utils/stage2_evaluation_layout.py` | 범용 Stage 2 평가의 실제 상자 반경·AT 목표 여유 공간을 고려한 초기 XY 배치 |
| `tokenhsi/utils/relation_task_spec.py` | reward config·checkpoint 계약 검증 |
| `tokenhsi/learning/multi_agent/amp_network_builder_ma.py`, `edge_context_encoder.py`, `ma_agent.py` | GTA actor/critic, edge bias·HOLDING 상태 융합, AMP·PPO 학습 |
| `tokenhsi/learning/multi_agent/coordination_head.py`, `stage2_transfer.py`, `ma_players.py` | 협력 attention의 semantic/owner-state edge 입력·Stage 1 weight 이식·학습 전 Stage 1 baseline 평가 |
| `tokenhsi/learning/multi_agent/attention_probe.py`, `tokenhsi/tests/test_attention_probe.py` | CA head별 가중치 복원·edge/전체 CA/균등 routing/동료 edge 개입과 PyTorch 원 계산 대조 검증 |
| `tokenhsi/tests/test_scenario_independent_with_climb.py`, `test_scenario_stage1_plane.py`, `test_scenario_stage1_sit_plane_curriculum.py`, `test_stage2_coordination.py` | 현재 실험 CPU 검증 |
| `tokenhsi/tests/test_stage1_unified.py` | 두 unified config 계약·독립 분포·goal 회전·AMP family 매칭·토큰 순열의 actor/value 및 gradient 검증 |
| `tokenhsi/tests/test_stage2_rescue_klclimb50.py` | 3→4물체 전이·AMP 1290·baseline·범용 인원/물체·홀수/자원 제약 샘플링·확장 모델/RMS 로딩·활성 협력 가중치에서 human/edge 순열·endpoint grounding·action/gradient 동등성 검증 |
| `tokenhsi/tests/test_stage2_rescue_shared9.py` | 9조합·평가1환경/범용 자원/비율·부분 reset의 크기/배정·all-loco RSI/AMP 계약·공유 ON_TOP source 고유성·활성 협력의 human/token/edge 순열 검증 |
| `tokenhsi/tests/test_stage2_unified.py` | 일반 unified 설정 계승·5필드 관측·전이 동등성·고정 encoder·협력 gradient·다른 variant 거부 |
| `tokenhsi/tests/test_stage2_unified_owner_holding.py` | Owner Stage 2 샘플링·Stage 1 설정 계승·checkpoint 전이·초기 action 동등성 검증 |
| `tokenhsi/tests/test_graph_box_speed_penalty.py` | graph 담당 물체의 속도 패널티·부분 reset history·기존 할당 경로 회귀 검증 |

`edge_context_spec.py`, `edge_ontop_spec.py`, `edge_interaction_spec.py`와 관련 reward 함수는 과거에 도입됐지만 현재 Stage 1·2의 graph·관찰·보상 계산에서도 호출된다. 파일명만 보고 삭제하면 현재 실험이 깨진다.

## 작업별 확인

- **Stage 1 graph·보상:** 해당 YAML → `edge_scenario_spec.py` → `edge_ontop_task.py` → `edge_interaction_reward.py`·`edge_stage1_reward.py`; 위 Stage 1 테스트로 검증한다.
- **Stage 2 협력·전이:** `approach_stage2_coordination.yaml`·`approach_stage2_unified.yaml`·`approach_stage2_unified_owner_holding.yaml`·`approach_stage2_rescue_klclimb50.yaml`·`approach_stage2_rescue_shared9.yaml` → `edge_stage2_spec.py` → `coordination_head.py`·`stage2_transfer.py`; `test_stage2_coordination.py`·`test_stage2_unified.py`·`test_stage2_unified_owner_holding.py`·`test_stage2_rescue_klclimb50.py`·`test_stage2_rescue_shared9.py`로 검증한다.
- **실행·GPU:** Stage 1 [config.md](config.md), Stage 2 [config_stage2.md](config_stage2.md), `runtime_env.sh`, `mps/`, `run-gui.sh`를 확인한다.
- **지표 해석:** [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 확인한다.
