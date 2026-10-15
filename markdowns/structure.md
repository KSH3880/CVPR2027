# 레포 구조

Multi-agent 실험은 `stage1`, `stage1_v2`, `stage1_v2_fixed` 세 개다. 컨피그 목록·기본 실행 명령·추론 옵션은 [config.md](config.md), 추가 옵션·checkpoint 규칙은 [USAGE.md](USAGE.md)를 따른다.

## 실행과 설정

| 경로 | 역할 |
| --- | --- |
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
