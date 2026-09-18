# 레포 구조

TokenHSI의 multi-agent carry 실험 저장소. 현재 비교는 distance progress의 **성공 시 edge 포화 on/off**이며 세부 설정과 명령은 [config.md](config.md)에 모은다.

## 코드 위치

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/run.py` | 학습·평가 진입점, 환경·알고리즘 등록 |
| `tokenhsi/scripts/multi_agent/` | 실험별 train/test, 현재 실험 VNC 단축 실행 `approach_distance_success_vnc.sh`, `runtime_env.sh`, 범용 `run-gui.sh`와 GPU 선택 `gui_gpu_env.sh` |
| `tokenhsi/data/cfg/multi_agent/` | 환경·reward 실험 YAML 6개: 기존 ID 1, 9~13 유지 |
| `tokenhsi/data/cfg/train/rlg/` | PPO/AMP/Transformer 학습 설정. 현재 relation 학습은 `amp_ma_carry_relation.yaml` |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py` | 박스 배정·reset·물리 환경·관측 통합 |
| `tokenhsi/env/tasks/multi_agent/relation_task.py` | 환경 상태를 relation reward/diagnostics에 연결 |
| `tokenhsi/env/tasks/multi_agent/relation_reward.py` | state/progress/reward 계산, 성공 조건, history runtime |
| `tokenhsi/env/tasks/multi_agent/relation_diagnostics.py` | TensorBoard 표시 순서(`RELATION_TB_GROUPS`)·배치 유지·CSV 진단 |
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

- **보상/성공 조건:** 실험 YAML → `relation_task_spec.py` → `relation_reward.py` → `relation_task.py`. 회귀 테스트는 `test_relation_distance.py`와 관련 `test_relation_*.py`.
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
