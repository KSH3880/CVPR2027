# Multi-Agent Carry 실험 가이드

`tokenhsi/data/cfg/multi_agent`의 **12개 config 전체**와 각각의 학습·시각화 명령을 정리한다. 모든 명령은 저장소 루트에서 실행한다.

```bash
cd /home/cvlab/Desktop/CVPR2027/edge_a2_gta_state_near_approach_0
```

- 학습 인자: `[num_agents] [num_envs] [num_objects]`
- 시각화 인자: `<checkpoint.pth> [num_agents] [num_envs] [num_objects] [eval_repeats]`
- 아래 예시는 학습 `2 / 2048 / 3`, 시각화 `2 / 1 / 3 / 10`이다.
- `HEADLESS=0`은 viewer, `HEADLESS=1`은 화면 없는 평가다.

## Config 전체 목록

| 번호 | Config | 핵심 차이 | Train / Test script |
| ---: | --- | --- | --- |
| 1 | [amp_humanoid_ma_carry.yaml](../tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml) | 원본 TokenHSI 방식, state-relation 미사용 | `ma_carry_train.sh` / `ma_carry_test.sh` |
| 2 | [amp_humanoid_ma_carry_relation.yaml](../tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry_relation.yaml) | relation v0, state 가중치 1.0, 성공 +5 | `ma_carry_relation_train.sh` / `ma_carry_relation_test.sh` |
| 3 | [amp_humanoid_ma_carry_relation_state02.yaml](../tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry_relation_state02.yaml) | relation v0에서 state 가중치 0.2 | `ma_carry_relation_state02_train.sh` / `..._test.sh` |
| 4 | [amp_humanoid_ma_carry_relation_state02_near.yaml](../tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry_relation_state02_near.yaml) | At state를 XYZ box-near Gaussian으로 변경 | `ma_carry_relation_state02_near_train.sh` / `..._test.sh` |
| 5 | [amp_humanoid_ma_carry_relation_state02_near_dir.yaml](../tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry_relation_state02_near_dir.yaml) | raw progress를 direction cosine으로 변경 | `ma_carry_relation_state02_near_dir_train.sh` / `..._test.sh` |
| 6 | [amp_humanoid_ma_carry_relation_state02_near_dir_success10.yaml](../tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry_relation_state02_near_dir_success10.yaml) | 첫 성공 보너스 +5 → +10 | `ma_carry_relation_state02_near_dir_success10_train.sh` / `..._test.sh` |
| 7 | [amp_humanoid_ma_carry_relation_state02_near_dir_success10_approach.yaml](../tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry_relation_state02_near_dir_success10_approach.yaml) | At progress에 XY approach blending | `ma_carry_relation_state02_near_dir_success10_approach_train.sh` / `..._test.sh` |
| 8 | [approach_rsi.yaml](../tokenhsi/data/cfg/multi_agent/approach_rsi.yaml) | 7번 + reset skill RSI 웜업 | `approach_rsi_train.sh` / `approach_rsi_test.sh` |
| 9 | [approach_rsi_all_edges.yaml](../tokenhsi/data/cfg/multi_agent/approach_rsi_all_edges.yaml) | Holding과 At 모두 XY approach blending | `approach_rsi_all_edges_train.sh` / `..._test.sh` |
| 10 | [approach_rsi_all_edges_success_sat.yaml](../tokenhsi/data/cfg/multi_agent/approach_rsi_all_edges_success_sat.yaml) | 9번 + 현재 H/At/Z 성공 조건 + 성공 후 edge 0.8 | `approach_rsi_all_edges_success_sat_train.sh` / `..._test.sh` |
| 11 | [approach_distance.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance.yaml) | 10번 기반 통합 XY distance progress + RSI 웜업 제거. 기존 +10/latch 유지 | `approach_distance_train.sh` / `approach_distance_test.sh` |
| 12 | [approach_distance_success.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_success.yaml) | 11번의 distance progress + 현재 At/Z edge saturation + 매-step success 0.2 | `approach_distance_success_train.sh` / `approach_distance_success_test.sh` |

`..._test.sh`는 같은 행의 train script 이름에서 `_train.sh`를 `_test.sh`로 바꾼 전체 이름이다.

## 새 학습 명령 — 전체

각 명령 중 원하는 **하나만** 실행한다.

```bash
# 1. Original TokenHSI-style carry
bash tokenhsi/scripts/multi_agent/ma_carry_train.sh 2 2048 3

# 2. Relation v0
bash tokenhsi/scripts/multi_agent/ma_carry_relation_train.sh 2 2048 3

# 3. State weight 0.2
bash tokenhsi/scripts/multi_agent/ma_carry_relation_state02_train.sh 2 2048 3

# 4. State02 + box-near At state
bash tokenhsi/scripts/multi_agent/ma_carry_relation_state02_near_train.sh 2 2048 3

# 5. State02 + near + direction progress
bash tokenhsi/scripts/multi_agent/ma_carry_relation_state02_near_dir_train.sh 2 2048 3

# 6. 위 설정 + success bonus 10
bash tokenhsi/scripts/multi_agent/ma_carry_relation_state02_near_dir_success10_train.sh 2 2048 3

# 7. 기존 approach
bash tokenhsi/scripts/multi_agent/ma_carry_relation_state02_near_dir_success10_approach_train.sh 2 2048 3

# 8. Approach + RSI 웜업
bash tokenhsi/scripts/multi_agent/approach_rsi_train.sh 2 2048 3

# 9. RSI + all-edge approach blending
bash tokenhsi/scripts/multi_agent/approach_rsi_all_edges_train.sh 2 2048 3

# 10. RSI all-edge + success saturation
bash tokenhsi/scripts/multi_agent/approach_rsi_all_edges_success_sat_train.sh 2 2048 3

# 11. 기존 unified distance 실험
bash tokenhsi/scripts/multi_agent/approach_distance_train.sh 2 2048 3

# 12. 현재 toy: unified distance + current-success saturation/reward
bash tokenhsi/scripts/multi_agent/approach_distance_success_train.sh 2 2048 3
```

재현 가능한 seed가 필요하면 명령 앞에 `SEED=42`를 붙인다. 2~12번 relation 학습은 같은 config에서 만든 checkpoint로 재개할 수 있다.

```bash
RESUME_CHECKPOINT='/absolute/path/checkpoint.pth' SEED=42 \
    bash tokenhsi/scripts/multi_agent/approach_distance_train.sh 2 2048 3
```

1번 `ma_carry_train.sh`는 현재 `RESUME_CHECKPOINT` 환경변수를 처리하지 않으므로 위 resume 형식을 그대로 사용할 수 없다.

## 시각화 명령 — 전체

각 `CKPT`를 해당 config로 학습한 실제 checkpoint 경로로 바꾼다. 다른 reward config의 checkpoint를 섞으면 relation metadata 검사에서 거부된다.

```bash
# 1
CKPT='/path/to/ma_carry.pth'
HEADLESS=0 bash tokenhsi/scripts/multi_agent/ma_carry_test.sh "$CKPT" 2 1 3 10

# 2
CKPT='/path/to/relation_v0.pth'
HEADLESS=0 bash tokenhsi/scripts/multi_agent/ma_carry_relation_test.sh "$CKPT" 2 1 3 10

# 3
CKPT='/path/to/state02.pth'
HEADLESS=0 bash tokenhsi/scripts/multi_agent/ma_carry_relation_state02_test.sh "$CKPT" 2 1 3 10

# 4
CKPT='/path/to/state02_near.pth'
HEADLESS=0 bash tokenhsi/scripts/multi_agent/ma_carry_relation_state02_near_test.sh "$CKPT" 2 1 3 10

# 5
CKPT='/path/to/state02_near_dir.pth'
HEADLESS=0 bash tokenhsi/scripts/multi_agent/ma_carry_relation_state02_near_dir_test.sh "$CKPT" 2 1 3 10

# 6
CKPT='/path/to/state02_near_dir_success10.pth'
HEADLESS=0 bash tokenhsi/scripts/multi_agent/ma_carry_relation_state02_near_dir_success10_test.sh "$CKPT" 2 1 3 10

# 7
CKPT='/path/to/approach.pth'
HEADLESS=0 bash tokenhsi/scripts/multi_agent/ma_carry_relation_state02_near_dir_success10_approach_test.sh "$CKPT" 2 1 3 10

# 8
CKPT='/path/to/ApproachRSI.pth'
HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_rsi_test.sh "$CKPT" 2 1 3 10

# 9
CKPT='/path/to/ApproachRSIAllEdges.pth'
HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_rsi_all_edges_test.sh "$CKPT" 2 1 3 10

# 10
CKPT='/path/to/ApproachRSIAllEdgesSuccessSat.pth'
HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_rsi_all_edges_success_sat_test.sh "$CKPT" 2 1 3 10

# 11
CKPT='/path/to/ApproachDistance.pth'
HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_distance_test.sh "$CKPT" 2 1 3 10

# 12
CKPT='/path/to/ApproachDistanceSuccess.pth'
HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_distance_success_test.sh "$CKPT" 2 1 3 10
```

화면 없이 평가하려면 `HEADLESS=1`로 바꾸고 필요하면 환경 수를 `1`에서 `64`로 늘린다. 결과 JSON은 각 output 폴더의 `metrics/`에 저장된다.

## 현재 실험: Unified Distance Progress

12번 `approach_distance_success.yaml`은 기존 11번 파일을 수정하지 않고 분리한 새 toy 실험이다.

1. direction·velocity·approach blending·progress pinning을 제거하고 모든 directed edge에 동일한 현재 XY 거리식을 사용한다.
2. `skillInitCurriculum`을 제거하고 reset 비율을 처음부터 `0/0.5/0.1/0.3/0.1`로 고정한다.
3. 현재 At/Z success가 유지되는 step에만 해당 agent의 모든 edge reward를 최댓값으로 바꾸고 success reward 0.2를 추가한다.

\[
P_e=\frac{1}{1+\max(d_e-0.5,0)/1.0},
\qquad
R_e=G_e\left(0.2\phi_e+0.2P_e\right)
\]

- Holding: 사람 root → 할당된 박스 중심의 XY 거리
- At: 박스 중심 → 목표의 XY 거리
- relation/token type과 Z는 progress에 사용하지 않는다.
- relation-specific state \(\phi_e\)와 prerequisite \(G_e\)는 그대로다.
- 현재 success는 `At phi ≥ 0.9 AND |goal Z error| ≤ 1 mm`다. Holding은 조건에 포함하지 않는다.
- success가 참인 동안 각 edge는 `0.2 + 0.2 = 0.4`, agent의 두 edge 합은 `0.8`이다.
- 별도 current-success reward `0.2`를 매 성공 step 추가하므로 relation task reward는 최대 `1.0`이다.
- success에서 벗어난 다음 step에는 success 0.2와 saturation이 모두 사라지고 즉시 원래 \(G_e(0.2\phi_e+0.2P_e)\) 계산으로 돌아간다.
- first-success 보너스는 `0`이며, 과거 성공 `done`이 reward saturation을 유지하지 않는다.
- 실제 phi/gate/satisfied/achieved/done 관측은 덮어쓰지 않는다.
- Power·collision·box-speed 페널티는 saturation 밖에서 그대로 더해지고, AMP와 네트워크도 그대로다.

기존 `approach_distance` 파일·script·output은 변경하지 않았다. 11번 checkpoint는 reward config가 달라 12번에 직접 resume할 수 없다. 새 학습 output은 `output/approach_distance_success`이며 이후에는 12번 config로 만든 checkpoint끼리 resume한다.

## TensorBoard

```bash
source tokenhsi/scripts/multi_agent/runtime_env.sh
tensorboard --logdir output --host 127.0.0.1 --port 6006
```

추천 pin은 다음 10개다.

| 태그 | 의미 |
| --- | --- |
| `relation/holding/satisfied` | Holding ≥ 0.9인 step 비율 |
| `relation/at/satisfied` | At ≥ 0.9인 step 비율 |
| `relation/current_success_state` | 현재 At ≥ 0.9 및 Z ≤ 1 mm인 agent-step 비율 |
| `relation/saturation_active` | 실제 edge reward saturation이 적용된 agent-step 비율 |
| `relation/current_success_reward` | 매 step 추가된 success reward 평균. 최대 0.2 |
| `relation/placement/episode_ever_rate` | 배치를 한 번이라도 달성한 완료 에피소드 비율 |
| `relation/placement/episode_final_rate` | 종료 시에도 배치된 완료 에피소드 비율 |
| `relation/placement/post_first_retention` | 첫 배치 이후 배치 유지 비율 |
| `relation/placement/first_seconds` | 첫 배치까지 걸린 시간 |
| `relation/placement/longest_hold_seconds` | 최장 연속 배치 시간. 손으로 잡은 시간이 아님 |

공통 배치 판정은 XY ≤ 10 cm, Z 오차 ≤ 1 mm이며 reward 성공 조건과 별개다. `done/achieved`는 에피소드 성공률이 아니라 이력 flag의 step 평균이다. 세부 정의와 매-step CSV는 [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 참고한다.

## 실행 주의사항

- 본학습 전에 셸에 남아 있는 `SMOKE`, `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT`를 확인한다.
- test 스크립트의 `HEADLESS=0`은 viewer를 켜지만 영상 녹화는 `--no_video`로 끈다.
- 평가 reset 비율은 loco/pickUp/carryWith/putDown = `0.5/0.1/0.3/0.1`이다.
- RSI 웜업은 8~10번 기존 재현 config에만 남아 있다. 11~12번에는 없다.
- 빠른 점검은 별도 output을 사용한다.

```bash
SMOKE=1 MAX_ITERATIONS=3 RESUME_CHECKPOINT= OUTPUT_PATH=output/smoke_distance_success \
    bash tokenhsi/scripts/multi_agent/approach_distance_success_train.sh 2 32 3
```
