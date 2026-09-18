# Multi-Agent Carry Config 가이드

`tokenhsi/data/cfg/multi_agent`의 **6개 config 전체**와 각각의 학습·시각화 명령을 정리한다. 모든 명령은 저장소 루트에서 실행한다.

```bash
cd /home/hwanhee/ksh/approach_distance_success
```

- 학습 인자: `[num_agents] [num_envs] [num_objects]`
- 시각화 인자: `<checkpoint.pth> [num_agents] [num_envs] [num_objects] [eval_repeats]`
- 아래 예시는 학습 `2 / 2048 / 3`, 시각화 `2 / 1 / 3 / 10`이다.
- `HEADLESS=0`은 viewer, `HEADLESS=1`은 화면 없는 평가다.

## 빠른 확인

학습 환경 수는 RTX PRO 6000 기준 **2048**로 통일한다. 평가·VNC의 환경 수는 화면 구성과 평가 목적에 따라 별도로 지정한다.

- 현재 비교: **12번 `approach_distance_success`**(성공 중 edge 포화) / **13번 `approach_distance_success_no_sat`**(포화 없음).
- 둘 다 현재 `At phi ≥ 0.9 AND |Z error| ≤ 1 mm`일 때 매 step `+0.2`; 포화 여부만 다르다.
- 11번 `approach_distance`는 과거 +10/latch 실험이다. 12·13번과 혼동하지 않는다.
- 기본 학습: 에이전트 2 / 환경 2048 / 물체 3. GPU는 `TOKENHSI_GPU`, conda는 `TOKENHSI_CONDA_ENV=tokenhsi`로 지정한다.
- 실행: `tokenhsi/scripts/multi_agent/<실험명>_{train,test}.sh`; 결과: `output/<실험명>/`.
- 학습 YAML: `tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation.yaml`; 환경 YAML: `tokenhsi/data/cfg/multi_agent/<실험명>.yaml`.
- config가 다른 checkpoint는 resume/evaluate 호환되지 않는다. checkpoint의 config와 test 스크립트를 맞춘다.
- VNC: 현재 12번은 `TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_distance_success_vnc.sh "$CKPT"`로 실행한다.
- 본학습 전 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여 설정을 확인한다.
- 목차에서 필요한 부분만 읽는다: [전체 목록](#config-전체-목록), [학습](#새-학습-명령--전체), [시각화](#시각화-명령--전체), [VNC](#원격-서버에서-vnc로-시각화), [현재 보상](#현재-실험-unified-distance-progress), [포화 비교](#비교-실험-성공-시-포화-끄기), [TensorBoard](#tensorboard).

## Config 전체 목록

번호는 기존 실험 ID를 유지한다. 2~8번은 config·스크립트·전용 연산과 함께 삭제했다.

| 번호 | Config | 핵심 차이 | Train / Test script |
| ---: | --- | --- | --- |
| 1 | [amp_humanoid_ma_carry.yaml](../tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml) | 원본 TokenHSI 방식, state-relation 미사용 | `ma_carry_train.sh` / `ma_carry_test.sh` |
| 9 | [approach_rsi_all_edges.yaml](../tokenhsi/data/cfg/multi_agent/approach_rsi_all_edges.yaml) | Holding과 At 모두 XY approach blending + RSI 웜업 | `approach_rsi_all_edges_train.sh` / `..._test.sh` |
| 10 | [approach_rsi_all_edges_success_sat.yaml](../tokenhsi/data/cfg/multi_agent/approach_rsi_all_edges_success_sat.yaml) | 9번 + 현재 H/At/Z 성공 조건 + 성공 후 edge 0.8 | `approach_rsi_all_edges_success_sat_train.sh` / `..._test.sh` |
| 11 | [approach_distance.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance.yaml) | 10번 기반 통합 XY distance progress + RSI 웜업 제거. 기존 +10/latch 유지 | `approach_distance_train.sh` / `approach_distance_test.sh` |
| 12 | [approach_distance_success.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_success.yaml) | 11번의 distance progress + 현재 At/Z edge saturation + 매-step success 0.2 | `approach_distance_success_train.sh` / `approach_distance_success_test.sh` |
| 13 | [approach_distance_success_no_sat.yaml](../tokenhsi/data/cfg/multi_agent/approach_distance_success_no_sat.yaml) | 12번에서 edge saturation만 해제, 현재 성공 +0.2 유지 | `approach_distance_success_no_sat_train.sh` / `..._test.sh` |

`..._test.sh`는 같은 행의 train script 이름에서 `_train.sh`를 `_test.sh`로 바꾼 전체 이름이다.

## 새 학습 명령 — 전체

각 명령 중 원하는 **하나만** 실행한다.

```bash
# 1. Original TokenHSI-style carry
bash tokenhsi/scripts/multi_agent/ma_carry_train.sh 2 2048 3

# 9. RSI + all-edge approach blending
bash tokenhsi/scripts/multi_agent/approach_rsi_all_edges_train.sh 2 2048 3

# 10. RSI all-edge + success saturation
bash tokenhsi/scripts/multi_agent/approach_rsi_all_edges_success_sat_train.sh 2 2048 3

# 11. 기존 unified distance 실험
bash tokenhsi/scripts/multi_agent/approach_distance_train.sh 2 2048 3

# 12. 현재 toy: unified distance + current-success saturation/reward
bash tokenhsi/scripts/multi_agent/approach_distance_success_train.sh 2 2048 3

# 13. Current-success +0.2 유지, edge saturation 해제
bash tokenhsi/scripts/multi_agent/approach_distance_success_no_sat_train.sh 2 2048 3
```

재현 가능한 seed가 필요하면 명령 앞에 `SEED=42`를 붙인다. 9~13번 relation 학습은 같은 config에서 만든 checkpoint로 재개할 수 있다.

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

# 13
CKPT='/path/to/ApproachDistanceSuccessNoSat.pth'
HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_distance_success_no_sat_test.sh "$CKPT" 2 1 3 10
```

화면 없이 평가하려면 `HEADLESS=1`로 바꾸고 필요하면 환경 수를 `1`에서 `64`로 늘린다. 결과 JSON은 각 output 폴더의 `metrics/`에 저장된다.

## 원격 서버에서 VNC로 시각화

위 시각화 명령을 [`run-gui.sh`](../tokenhsi/scripts/multi_agent/run-gui.sh)로 감싸면 서버의 IsaacGym viewer를 로컬 브라우저에서 볼 수 있다. 이 스크립트가 임시 Xvfb 화면과 x11vnc/noVNC를 띄우고 `HEADLESS=0`을 설정한다.

### 1. 서버에서 실행

12번 실험은 전용 실행 스크립트로 GPU와 checkpoint만 지정한다. `TOKENHSI_GPU=5` 한 곳에서 GPU를 정하면 CUDA와 viewer 렌더링에 모두 적용된다.

```bash
cd /home/hwanhee/ksh/approach_distance_success

# 해당 실험에서 저장된 실제 .pth 파일로 변경한다.
CKPT='output/approach_distance_success/ApproachDistanceSuccess_18-14-22-43/nn/ApproachDistanceSuccess.pth'

TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_distance_success_vnc.sh "$CKPT"
```

conda `tokenhsi`, 서버 VNC 설치 경로, 포트 `6080`, 출력 `output/approach_distance_success_vnc`, 평가 인자 `2 1 3 10`을 기본으로 사용한다. 필요할 때만 `PORT=6081` 같은 환경변수나 checkpoint 뒤의 평가 인자를 덮어쓴다.

`CKPT`는 이미 저장된 파일이어야 한다. 본학습 설정은 500 epoch마다 정기 저장하므로, 학습 초반에 파일이 없다면 저장 후 실행한다. 저장된 파일 목록은 다음 명령으로 확인한다.

```bash
find output/approach_distance_success -type f -name '*.pth'
```

인자 `2 1 3 10`은 에이전트 2개, 환경 1개, 물체 3개, 평가 반복 10회다. 다른 실험은 해당 `*_test.sh`와 checkpoint를 `run-gui.sh`로 감싼다:

```bash
TOKENHSI_GPU=5 VNC_DIR="$HOME/opt/vnc" \
    bash tokenhsi/scripts/multi_agent/run-gui.sh \
    bash tokenhsi/scripts/multi_agent/approach_distance_success_no_sat_test.sh "$CKPT_NO_SAT" 2 1 3 10
```

GPU는 명령 앞의 `TOKENHSI_GPU` 한 곳에서 지정한다. 같은 터미널에서 계속 쓸 때는 `export TOKENHSI_GPU=5`로 먼저 설정해도 된다. 뒤쪽 명령에 GPU 인자를 붙일 필요가 없다.

### 2. 로컬 PC에서 브라우저 연결

**VS Code Remote-SSH 사용 시:** 서버에 연결한 VS Code의 **Ports(포트)** 탭에서 **6080**을 포워딩한 뒤 로컬 브라우저로 접속한다. 로컬 포트가 다른 번호로 배정되면 URL도 그 번호로 바꾼다.

```text
http://localhost:6080/vnc.html?autoconnect=1&resize=remote
```

**일반 SSH 사용 시:** 로컬 PC의 별도 터미널에서 아래 터널을 유지하고 같은 URL에 접속한다. `SERVER_HOST`는 실제 서버 주소나 SSH 설정의 호스트 별칭으로 바꾼다.

```bash
ssh -N -L 6080:127.0.0.1:6080 hwanhee@SERVER_HOST
```

noVNC는 서버의 `127.0.0.1`에서만 수신하므로 포트 포워딩을 통해 접속한다. viewer는 지정한 checkpoint를 로드하며, 학습 중 새로 저장되는 checkpoint를 자동으로 다시 읽지는 않는다. 최신 모델을 보려면 시각화 명령을 다시 실행한다.

### 종료 및 문제 확인

- 평가가 끝나면 시각화 프로세스와 임시 VNC 서비스가 종료된다. 중간에 종료하려면 서버의 시각화 터미널에서 `Ctrl+C`를 누른다.
- `port 6080 is already in use`가 나오면 `PORT=6081` 등으로 바꾸고, 포트 포워딩과 브라우저 URL도 같은 번호로 맞춘다.
- `x11vnc not found`가 나오면 `VNC_DIR="$HOME/opt/vnc"`를 확인한다. 이 서버의 실행 파일은 `$HOME/opt/vnc/usr/bin/x11vnc`다.
- `websockify not found`가 나오면 명령 앞에 `WEBSOCKIFY="$HOME/anaconda3/envs/tokenhsi/bin/websockify"`를 추가한다.
- 검은 화면이면 먼저 서버 터미널에서 모델·모션 로딩이 끝났는지, checkpoint 경로 오류가 없는지 확인한다. VNC 서비스 로그는 `/tmp/tokenhsi_gui_6080_{xvfb,x11vnc,websockify}.log`에 있으며, 포트를 바꾸면 파일명의 `6080`도 바뀐다.

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

## 비교 실험: 성공 시 포화 끄기

13번 `approach_distance_success_no_sat.yaml`은 12번에서 `success.saturate_edge_rewards_while_current`만 `false`로 바꾼 실험이다. 성공 조건과 매-step 성공 보상 `+0.2`, distance progress, reset 비율, 학습 설정은 같다.

| 항목 | 12번: 포화 켜기 | 13번: 포화 끄기 |
| --- | --- | --- |
| 현재 성공 조건 | At phi ≥ 0.9 및 Z 오차 ≤ 1 mm | 동일 |
| 성공 중 edge reward | 각 edge를 0.4로 고정 | 원래 `G × (0.2φ + 0.2P)` 유지 |
| 성공 중 추가 보상 | 매 step +0.2 | 매 step +0.2 |
| 성공 중 relation task reward | agent당 1.0 | 실제 edge 합 +0.2, 최대 1.0 |
| `relation/02_reward/01_saturation_active` | 현재 성공 시 1 | 항상 0 |

두 실험 모두 성공에서 벗어나면 즉시 `+0.2`가 사라진다. 13번은 성공 후에도 Holding state와 prerequisite gate에 따라 edge reward가 달라지므로, 포화가 배치 유지와 손을 놓는 행동에 미치는 영향을 비교할 수 있다. `current_saturation_z_tolerance`라는 기존 키는 포화 여부와 별개로 현재 성공 보상에도 사용하는 Z 허용 오차다.

```bash
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi \
    bash tokenhsi/scripts/multi_agent/approach_distance_success_no_sat_train.sh 2 2048 3
```

결과는 `output/approach_distance_success_no_sat`에 저장된다. reward config가 다르므로 12번 checkpoint로 13번을 직접 resume/evaluate할 수 없다. 새로 학습하고, 시각화에는 13번의 test 스크립트와 checkpoint를 사용한다. VNC 실행 방법은 위와 같고 `OUTPUT_PATH`도 `output/approach_distance_success_no_sat_vnc`로 바꾸면 된다. 같은 seed로 두 실험을 새로 학습하려면 양쪽 명령 앞에 동일한 `SEED` 값을 지정한다.

## 정리한 범위와 유지하는 기능

- 삭제: 기존 2~7번 relation/state02/near/direction/+10/At-only approach와 8번 `approach_rsi`. 환경 YAML 7개, train/test 14개, 해당 실험 전용 테스트 2개를 제거했다.
- 전용 코드 삭제: Gaussian velocity progress, near/putdown 혼합 At state, At-only approach와 state 자체 gate로 progress를 포화시키는 경로. schema 검증과 simulator smoke에서도 해당 경로를 제거했다.
- 유지: 1번 원본 carry, 9~10번 all-edge RSI, 11번 distance + latch, 12~13번 현재 성공 포화 on/off.
- 9~10번이 사용하는 direction progress·all-edge blending·RSI curriculum, 10~11번의 latched reward, 모든 relation 실험의 관측용 achieved/done history는 유지한다.
- 기존 1·9~13번 YAML의 설정값과 checkpoint metadata는 이번 정리에서 바꾸지 않았다.

추가 후보는 소규모 전용 `amp_ma_carry_relation_smoke.yaml`, `amp_ma_carry_watch.yaml`과 `ma_carry_watch.sh`, legacy 관측/Geo 분기, upstream의 다른 태스크들이다. 이번 삭제 범위에는 포함하지 않았다. `Humanoid` 기반 기능·AMP·모션 처리처럼 현재 carry가 사용하는 공통 코드는 유지한다.

## TensorBoard

```bash
source tokenhsi/scripts/multi_agent/runtime_env.sh
tensorboard --logdir output --host 127.0.0.1 --port 6006
```

추천 pin은 **`relation/00_main`의 5개**다.

| 태그 | 의미 |
| --- | --- |
| `relation/00_main/01_placement_episode_final_rate` | 종료 시에도 배치된 완료 에피소드 비율. 최종 성과 |
| `relation/00_main/02_placement_episode_ever_rate` | 배치를 한 번이라도 달성한 완료 에피소드 비율. 도달 능력 |
| `relation/00_main/03_placement_post_first_retention` | 첫 배치 이후 배치 유지 비율. 도달한 에피소드만 집계 |
| `relation/00_main/04_current_success_state` | 현재 reward 성공 조건인 At ≥ 0.9 및 Z ≤ 1 mm의 agent-step 비율 |
| `relation/00_main/05_holding_satisfied` | Holding ≥ 0.9의 agent-step 비율. 잡기 단계 진단 |

relation 내부 순서는 다음처럼 고정한다. 각 그룹 안에서도 숫자 순으로 중요 지표를 먼저 둔다.

| 그룹 | 참고할 내용 |
| --- | --- |
| `00_main` | 위 핵심 5개 |
| `01_placement` | XY·Z 오차 → At 만족 비율 → 첫 배치 시간 → 최장 유지 시간 → 도달 후 관측 시간 |
| `02_reward` | 포화 적용 비율 → 현재 성공 보너스 → relation 총보상 → edge별 실제·포화 전 보상 |
| `03_samples` | 유효 완료 수 → 도달 수 → 초기 배치로 제외된 표본. 유지율·시간의 표본 수 확인 |
| `04_state` | Holding/At phi·gate → achieved·현재 조건 → done·상태 전환 |
| `05_motion` | 사람-물체 거리 → 손 거리 → 물체 속도 → progress/blending |
| `90_debug` | 거리대별 정체 진단, 관측된 최초 성공 시간 등 나머지 지표 |

태그 정렬을 오름차순으로 두면 이 순서로 표시된다. `current_success_reward`는 현재 성공 비율에 0.2를 곱한 값이라 pin에서는 빼고 보상 그룹에 둔다. `saturation_active`도 포화 on/off 확인용으로 보상 그룹에서 본다.

새 태그는 다음 학습 시작/resume부터 적용된다. 실행 중인 프로세스와 과거 event 파일은 바꾸지 않는다. 예전 구간은 기존 태그에 남으며 새 태그로 이력을 복사하지 않는다. 브라우저의 기존 pin은 자동 변경되지 않으므로 기존 pin을 해제하고 `00_main`의 5개를 선택한다. 도달 표본이 없으면 유지율·시간 태그는 0으로 채우지 않고 생략한다.

공통 배치 판정은 XY ≤ 10 cm, Z 오차 ≤ 1 mm이며 reward 성공 조건과 별개다. `done/achieved`는 에피소드 성공률이 아니라 이력 flag의 step 평균이다. 세부 정의와 매-step CSV는 [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 참고한다.

## 실행 주의사항

- 본학습 전에 셸에 남아 있는 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT`를 확인한다.
- test 스크립트의 `HEADLESS=0`은 viewer를 켜지만 영상 녹화는 `--no_video`로 끈다.
- 평가 reset 비율은 loco/pickUp/carryWith/putDown = `0.5/0.1/0.3/0.1`이다.
- RSI 웜업은 9~10번 기존 재현 config에만 남아 있다. 11~13번에는 없다.
- 빠른 점검도 2048 환경을 사용하고, `MAX_ITERATIONS`로 반복 횟수만 줄인다. `SMOKE` 자동 축소는 제거했다.

```bash
MAX_ITERATIONS=3 RESUME_CHECKPOINT= OUTPUT_PATH=output/smoke_distance_success \
    bash tokenhsi/scripts/multi_agent/approach_distance_success_train.sh 2 2048 3
```
