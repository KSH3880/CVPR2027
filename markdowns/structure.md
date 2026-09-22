# 레포 구조

TokenHSI의 multi-agent carry 실험 저장소. 현재 비교는 distance progress의 **성공 시 edge 포화 on/off**와 포화 on 기준 **Holding k=5/10**이며 세부 설정과 명령은 [config.md](config.md)에 모은다.

## 코드 위치

| 경로 | 역할 |
| --- | --- |
| `mps/` | GPU별 MPS 단축 함수·운영 안내. 이 서버의 개인 설치본은 `~/.local/share/gpu-mps/shell.sh`, 새 Bash에서 자동 로드 |
| `tokenhsi/scripts/multi_agent/mps_auto_env.sh` | runtime에서 지정 GPU의 내 MPS를 검증·자동 연결. 데몬 시작·종료는 하지 않음 |
| `tokenhsi/run.py` | 학습·평가 진입점, 환경·알고리즘 등록 |
| `tokenhsi/scripts/multi_agent/` | 실험별 train/test/VNC, 19~24번 Stage 1 전용 스크립트, `runtime_env.sh`, 공통 `run-gui.sh`·`gui_gpu_env.sh` |
| `tokenhsi/data/cfg/multi_agent/` | 실행 가능 환경·reward YAML 17개(ID 1, 9~24); 24번은 CLIMB 전용 semantic-only schema 7 scratch 실험 |
| `tokenhsi/data/cfg/train/rlg/` | PPO/AMP/Transformer 학습 설정. 현재 relation 학습은 `amp_ma_carry_relation.yaml` |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py` | 박스 배정·reset·물리 환경·관측 통합 |
| `tokenhsi/env/tasks/multi_agent/relation_task.py` | 환경 상태를 relation reward/diagnostics에 연결 |
| `tokenhsi/env/tasks/multi_agent/relation_reward.py` | state/progress/reward 계산, 성공 조건, history runtime |
| `tokenhsi/env/tasks/multi_agent/relation_diagnostics.py` | TensorBoard 표시 순서(`RELATION_TB_GROUPS`)·배치 유지·CSV 진단 |
| `tokenhsi/env/tasks/multi_agent/ontop_task.py` | OnTop reset·역할 배정·면 좌표 연결·시나리오별 진단 |
| `tokenhsi/utils/edge_ontop_spec.py` | 17번 batched graph·reset 샘플러·preset·7-field packet·schema 3 검증 |
| `tokenhsi/env/tasks/multi_agent/edge_ontop_reward.py` | 회전 bbox의 월드 Z extent·OnTop 상태·같은 환경 task 공유 |
| `tokenhsi/env/tasks/multi_agent/edge_ontop_task.py` | dynamic Ox·AT-only 플랫폼·물리 reset 제약·sampling/기하학 진단·viewer |
| `tokenhsi/utils/edge_interaction_spec.py` | 18번 7-pattern sampler·SIT/CLIMB relation·preset·schema 4 graph 검증 |
| `tokenhsi/env/tasks/multi_agent/edge_interaction_reward.py` | SIT tarSitPos target, CLIMB root target·feet success 검증 |
| `tokenhsi/utils/edge_stage1_spec.py` | 19~22번 7-pattern sampler, 23번 단일 H/S/C, 24번 CLIMB-only sampler·5-field semantic packet·relation RSI 검증 |
| `tokenhsi/env/tasks/multi_agent/edge_stage1_reward.py` | 19번 own-success-only 포화·sharing 없는 task 합·constant START/KEEP packet |
| `tokenhsi/utils/edge_context_spec.py` | 새 schema 검증, explicit graph compile, PRE/TERM 참조, resume task 계약 |
| `tokenhsi/env/tasks/multi_agent/edge_context_reward.py` | Holding·At binding 기반 평가, raw context, edge-local live 성공·포화 |
| `tokenhsi/env/tasks/multi_agent/edge_context_task.py` | 새 mode의 reset/reward/관측·CSV·성공률 연결 |
| `tokenhsi/learning/multi_agent/edge_context_encoder.py`, `edge_context_eval.py` | scalar context/fusion bias와 required-goal 평가 |
| `tokenhsi/data/cfg/multi_agent/graphs/` | 평가용 explicit graph override 예제 (환경/실험 YAML과 구분) |
| `tokenhsi/utils/ontop_task_spec.py` | 혼합 시나리오 검증·환경 배분·환경별 그래프·회전된 면 중심 계산 |
| `tokenhsi/learning/multi_agent/transfer.py` | carry 가중치·기존 relation 행 복사와 OnTop 행 확장 검증 |
| `tokenhsi/utils/relation_task_spec.py` | relation graph, config 검증, checkpoint 호환 계약 |
| `tokenhsi/env/tasks/multi_agent/scene_features.py` | scene token/pose feature 구성 |
| `tokenhsi/learning/multi_agent/amp_network_builder_ma.py` | A2 relation bias와 GTA actor/critic |
| `tokenhsi/learning/multi_agent/ma_agent.py` | multi-agent PPO/AMP, rollout·학습·저장 |
| `tokenhsi/learning/multi_agent/ma_players.py` | checkpoint 평가·시각화 |
| `tokenhsi/learning/multi_agent/scene_normalizer.py` | 타입별 관측 정규화 |
| `tokenhsi/tests/` | relation/scene/policy CPU 테스트 및 simulator smoke |
| `tokenhsi/data/dataset_*/` | dataset YAML과 공유 데이터 심링크 |
| `tokenhsi/data/assets/`, `body_models/`, `lpanlib/` | 물리 asset, body model 로더, 공통 모션 라이브러리 |
| `output/<실험>/<run>/` | checkpoint `nn/`, TensorBoard `summaries/`, config·diagnostics |

## 작업별 최소 탐색

- **18번 SIT/CLIMB interaction:** [edge_context_interaction.md](edge_context_interaction.md), `approach_distance_edge_context_interaction_{train,test,vnc}.sh`, CPU `test_edge_interaction.py`.
- **19~24번 Stage 1 relation skill:** [edge_context_stage1.md](edge_context_stage1.md), `approach_distance_edge_context_stage1_{train,test,vnc}.sh`, `_ground_sit_climb_{train,test,vnc}.sh`, `_common_boxes_{train,test,vnc}.sh`, `_fixed_boxes_sit_fix_{train,test,vnc}.sh`, `_primitives_rsi_{train,test,vnc}.sh`, `approach_distance_stage1_climb_rsi_{train,test,vnc}.sh`, CPU `test_edge_stage1.py`.

- **17번 sampled OnTop:** [edge_context_ontop.md](edge_context_ontop.md), `approach_distance_edge_context_ontop_{train,test,vnc}.sh`. CPU `test_edge_ontop.py`, 실제 PPO/GAE 저장 경로 `smoke_edge_ontop_train.py`, 물리 접촉·preset·replay `smoke_edge_ontop_sim.py`.

- **16번 Holding·At edge context:** [edge_context_success.md](edge_context_success.md), `approach_distance_edge_context_success_{train,test,vnc}.sh`. CPU 검증 `test_edge_context_success.py`, 실제 실행 검증 `smoke_edge_context_sim.py`. 새 학습과 기존 실험을 분리한다.

- **보상/성공 조건:** 실험 YAML → `relation_task_spec.py` → `relation_reward.py` → `relation_task.py`. 회귀 테스트는 `test_relation_distance.py`와 관련 `test_relation_*.py`.
- **Holding k=10 수식·입력 좌표:** [approach_distance_success_holding_k10_reward.md](approach_distance_success_holding_k10_reward.md).
- **OnTop 혼합학습·전이·실행:** [ontop_mixed_config.md](ontop_mixed_config.md). 512/768/768 동시학습, epoch 18000 전이, 전용 `approach_distance_success_ontop_mixed_{train,test,vnc}.sh`. CPU 테스트 `test_ontop_mixed.py`, 실제 시뮬레이터 확인 `smoke_relation_sim.py`.
- **reset/박스/좌표:** `humanoid_ma_carry.py`, `scene_features.py`, 해당 환경 YAML.
- **네트워크/관측:** `amp_network_builder_ma.py`, `scene_normalizer.py`, `ma_agent.py`; 현재 구조 상세는 [ma_clean_scene_gta.md](ma_clean_scene_gta.md).
- **학습 실행/VNC:** [config.md](config.md)와 대상 train/test, `runtime_env.sh`, `run-gui.sh`.
- **지표 해석:** [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md).
- **설치/서버 이관:** [USAGE.md](USAGE.md), [PORTING_GUIDE.md](PORTING_GUIDE.md)를 필요한 부분만 확인한다.
- **과거 설계/분석:** 오래된 문서는 삭제했다. 역사 비교가 필요한 작업에서만 Git 이력을 확인한다.

## CPU 테스트 예시

저장소 루트에서 `tokenhsi` conda 환경으로 실행한다. 변경 범위에 맞춰 파일을 선택한다.

```bash
PYTHONPATH=tokenhsi conda run -n tokenhsi python -m pytest -q \
    tokenhsi/tests/test_relation_distance.py \
    tokenhsi/tests/test_relation_reward.py \
    tokenhsi/tests/test_relation_success_saturation.py \
    tokenhsi/tests/test_relation_runtime.py \
    tokenhsi/tests/test_relation_diagnostics.py
```
