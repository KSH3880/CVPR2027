# TokenHSI-masteer Sequential Stack 평가 코드

이 문서는 `TokenHSI-masteer`의 **기본 sequential-stack 정책 평가기**를 다른 사람이
코드와 함께 이해하고 실행할 수 있도록 정리한 문서다. Learned coordinator를 붙이는
`eval_coord_sequential_stack.sh`나 `TokenHSI-coord/stack_planner` 평가는 범위에 포함하지
않는다.

## 1. 무엇을 평가하나

한 환경에 humanoid 두 명과 각자 소유한 상자 하나씩을 둔다. 정책은 아래 의존 관계를
순서대로 수행해야 한다.

```text
A1이 아래 상자를 바닥 목표에 배치
  -> 아래 상자가 안정적인지 확인
  -> A1이 상자에서 후퇴
  -> A2의 목표를 실제 아래 상자 위로 공개
  -> A2가 자기 상자를 운반해 위에 배치
  -> 상자가 안정적으로 유지되면 종료
```

핵심은 A1과 A2의 독립 carry 성공률이 아니라, **A1 배치 → 공간 확보 → A2 stack**이라는
순차 의존성을 한 에피소드 안에서 끝내는지 보는 것이다. 정책 입력 ABI는 기존
MA-steer와 같은 340-D이며, 이 평가기는 별도 planner를 사용하지 않는다.

## 2. 코드 위치와 역할

| 파일 | 역할 |
|---|---|
| `scripts/masteer/eval_sequential_stack.sh` | 실행 진입점, 평가 설정, 결과 집계 |
| `TokenHSI-masteer/tokenhsi/env/tasks/adapt_interaction_skills/humanoid_ma_sequential_stack_carry.py` | sequential phase machine, 성공·종료 판정, 단계별 metric 기록 |
| `TokenHSI-masteer/tokenhsi/learning/transformer/trans_players.py` | deterministic evaluation을 3회 반복하고 완료 episode를 flush |
| `TokenHSI-masteer/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py` | 공통 episode metric 누적 및 `.npy` 저장 |
| `TokenHSI-masteer/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml` | humanoid, carry, box test size와 기본 환경 설정 |

실행 호출 관계는 다음과 같다.

```text
eval_sequential_stack.sh
  -> 임시 YAML 생성: numAgents=2, numEnvs=STACK_EVAL_ENVS
  -> tokenhsi/run.py --task HumanoidMASequentialStackCarry --test --eval
  -> TransPlayer.run_eval(): deterministic policy evaluation 3회
  -> 환경이 episode별 agent-row metric을 MA_METRICS에 저장
  -> shell 안의 Python 집계기가 warm-up 제거, A1/A2 row 결합, 결과 출력
```

## 3. 실행 방법

저장소 루트에서 다음처럼 실행한다.

```bash
MA_GPU=7 STACK_EVAL_ENVS=270 \
bash scripts/masteer/eval_sequential_stack.sh \
  /absolute/path/to/Humanoid_00010000.pth \
  my_eval_tag
```

두 번째 인자인 tag는 생략할 수 있다. 생략하면 checkpoint 파일명에서 `.pth`를 뺀 값을
사용한다. tag에는 영문자, 숫자, `_`, `.`, `-`만 쓸 수 있다.

필수 입력은 다음 두 checkpoint다.

- positional argument 1: 평가할 MA-steer/sequential-stack policy
- `MS_CKPT`: stage-1 backbone. 생략 시
  `TokenHSI-masteer/output/ckpt_stage1.pth`

conda 환경 기본값은 `tokenhsi118`이다. 다른 환경을 쓸 때는
`TOKENHSI_CONDA_ENV`를 지정한다.

### 주요 실행 옵션

| 환경변수 | 기본값 | 의미 |
|---|---:|---|
| `MA_GPU` | `7` | physical GPU 번호. 내부에서는 `cuda:0`으로 remap |
| `STACK_EVAL_ENVS` | `270` | 병렬 환경 수. box grid 사용 시 9의 배수여야 함 |
| `STACK_EVAL_SEED` | `0` | 평가 seed |
| `STACK_EVAL_BOX_GRID` | `1` | A1/A2 상자 크기의 3×3 조합을 환경 전체에 균등 배치 |
| `STACK_EVAL_BOX_SIZE_IDS` | `0,4,7` | YAML `testSizes`의 index. 기본 cube 변 길이는 0.22/0.42/0.57 m |
| `STACK_EVAL_SUCCESS_MODE` | `box_radius` | 성공 판정. `box_radius` 또는 `tokenhsi_carry` |
| `STACK_EPISODE_LENGTH` | `900` | 한 sequential episode의 최대 action step |
| `STACK_BOTTOM_Z_TOL` | `0.05` | A1 아래 상자의 목표 높이 오차 허용치(m) |
| `STACK_BOTTOM_DISPLACE_TOL` | `0.50` | commit 뒤 아래 상자 이동으로 비정상 종료하는 거리(m) |

평가 스크립트는 `MA_TOKEN=mask`, `STACK_TASK_MODE=stack`,
`STACK_CARRY_REHEARSAL_PROB=0`, `STACK_END_ON_A2_RESUME=0`으로 고정한다. 즉 native-carry
rehearsal이나 A1-only curriculum이 아니라 전체 stack sequence만 평가한다.

## 4. Phase machine

| ID | phase | 동작과 다음 전환 |
|---:|---|---|
| 0 | `A1_PLACE` | A1이 아래 상자를 바닥 목표로 운반. XY와 Z가 허용 범위에 들면 1 |
| 1 | `VERIFY_BOTTOM` | 아래 상자의 선·각속도와 안정 시간을 확인. 조건을 잃으면 0, 통과하면 top pose를 commit하고 2 |
| 2 | `A1_RETREAT` | A1이 생성된 후퇴 목표로 이동. 목표 0.30 m 이내 또는 진행 거리 0.60 m 이상이면 3 |
| 3 | `A2_RESUME` | 이때 처음 A2에게 아래 상자 위의 top 목표를 공개. top 위치 범위에 들어오면 4 |
| 4 | `VERIFY_STACK` | top 상자가 지지면 위에서 느린 상태를 유지하는지 확인. 조건을 잃으면 3, 20 step 안정 시 5 |
| 5 | `DONE` | 정상 episode 종료 |

A1이 아래 상자를 안정화하면 top 목표는 **nominal target이 아니라 실제 아래 상자 pose**로
계산한다. A2는 A1이 후퇴를 끝내기 전까지 이 목표를 보지 못하고 자기 상자 위치에서
대기한다.

A1이 놓은 상자가 commit 위치에서 기본 0.50 m 넘게 이동하면 물리적 지지대가 사라진
것으로 보고 즉시 실패 종료한다. 양손은 정상적인 상자 조작을 위해 허용 contact body에
포함되지만, 몸통·머리·골반·무릎 등의 낙상 접촉은 계속 종료 조건이다.

## 5. 성공 판정

평가 성공은 `DONE` 도달과 동일하지 않다. A2 상자가 허용 위치에 **한 번이라도** 들어오면
`_seq_top_reached`를 latch하고, 이후 episode가 끝날 때 이 값을 성공으로 기록한다.
단, 아래 상자가 과도하게 움직여 비정상 종료되면 latch를 다시 false로 만든다.

### `box_radius` — 기본

A2가 활성화된 phase 3 또는 4에서 다음 두 조건을 만족하면 성공이다.

```text
XY: top box 중심이 bottom box 중심 기준 지지 반경 안에 있음
    radius = 0.5 * sqrt(bottom_size_x^2 + bottom_size_y^2)
Z : committed top target과의 높이 오차 < STACK_TOP_Z_TOL (기본 0.08 m)
```

상자 크기에 따라 XY 허용 범위가 달라진다. 현재 sequential 구현에서 top의 phase 전환과
기본 평가 성공은 `STACK_TOP_XY_TOL`의 고정값이 아니라 이 지지 반경을 사용한다.

### `tokenhsi_carry`

원본 TokenHSI carry 평가와 같은 조건을 쓴다.

```text
norm(top_box_xyz - top_target_xyz) <= successThreshold
```

현재 config의 `successThreshold`는 0.20 m다. A2 대기 목표가 처음부터 자기 상자 위치와
같기 때문에, 잘못된 즉시 성공을 막도록 phase 3/4에서만 판정한다. 이 옵션은 **보고되는
성공 판정만** 바꾸며, phase 3→4→5 전환은 계속 `box_radius` 기반 조건을 사용한다.

## 6. 표본 구성과 warm-up 제거

`TransPlayer.run_eval()`은 전체 평가를 3회 반복한다. `MA_EVAL_ALL_AGENT_ROWS=1`이므로
각 반복에서 모든 환경의 A1/A2 row가 수집되고, `MA_EVAL_FLUSH_DONE=1`이 마지막 완료
episode도 metric 파일에 기록되게 한다.

metric row는 episode가 끝난 순서대로 비동기 저장된다. 따라서 전체 배열의 앞부분을
잘라 warm-up을 제거하면 빨리 끝나는 상자 조합에 편향된다. 집계기는 다음 순서를 강제한다.

1. row ID로 각 환경의 A1/A2 기록을 모은다.
2. **각 환경별** 첫 episode를 warm-up으로 버린다.
3. 같은 환경의 두 번째와 세 번째 episode만 사용한다.
4. A1/A2 row를 같은 episode pair로 묶는다.

기본 270 env에서는 최종 표본이 `270 × 2 = 540 episodes`다. 3×3 box grid라면 각
bottom/top 크기 조합에 60 episode가 있어야 한다. 하나라도 부족하거나 조합별 개수가
불균형하면 집계기가 실패하고 결과를 유효한 것으로 출력하지 않는다.

## 7. 출력 파일

```text
runs/results/sequential_stack/eval_<tag>.npy
runs/results/sequential_stack/eval_<tag>.log
```

기존 파일은 덮어쓰지 않는다. 같은 checkpoint를 다시 평가하더라도 새 tag를 사용해야
한다.

로그에는 전체 요약 한 줄과 box 조합별 요약이 남는다.

```text
SEQ_STACK_EVAL ...
SEQ_STACK_BOX_COMBO ...
```

### 핵심 출력 지표

| 지표 | 의미 |
|---|---|
| `episodes` | warm-up 제거 후 균형 표본 수 |
| `success_n`, `success_rate` | latch된 sequential stack 성공 횟수와 비율 |
| `root_cum`, `box_cum` | A1+A2의 경로 횡오차 누적합 평균 |
| `root_mae`, `box_mae` | 누적 횡오차를 해당 step 수로 나눈 episode 평균 |
| `place_*` | A1 place + bottom verify 구간의 A1 오차 |
| `retreat_*` | A1 retreat 구간의 A1 오차 |
| `a2_*` | A2 resume + stack verify 구간의 A2 오차 |
| `success_root_*`, `success_box_*` | 성공 episode만 제한한 전체 경로 오차 |
| `terminate_rate` | timeout/DONE을 제외한 비정상 종료 비율 |
| `term_fall_a1_n`, `term_fall_a2_n`, `term_fall_both_n` | agent별 낙상 종료 수 |
| `term_bottom_displaced_n` | commit된 아래 상자가 허용 거리 밖으로 이동한 종료 수 |
| `term_multiple_n` | 같은 step에 복수 비정상 원인이 발생한 수 |
| `term_unknown_n` | 상속된 terminate였지만 원인을 분류하지 못한 수 |

`cum`은 episode 길이에 따라 커지므로 정책 비교의 주 지표로는 `*_mae`가 더 직접적이다.
전체 평균만 보지 말고 `SEQ_STACK_BOX_COMBO`에서 큰 상자/작은 상자 조합별 성공률과 종료
원인도 같이 확인한다.

## 8. `.npy`에서 평가기가 직접 읽는 열

`.npy`는 agent-row 단위 2-D 배열이며 현재 최소 55열이어야 한다. 이 스크립트가 직접
사용하는 열은 다음과 같다.

| 열 | 내용 |
|---:|---|
| 0 | global agent row ID: `2 * env + agent` |
| 26, 27 | 전체 root/box lateral-error 누적합 |
| 30 | 전체 error 누적 step 수 |
| 40 | sequential 성공 latch |
| 41 | 종료 시 최종 phase ID |
| 42–44 | place 구간 root 누적, box 누적, step 수 |
| 45–47 | retreat 구간 root 누적, box 누적, step 수 |
| 48–50 | A2 구간 root 누적, box 누적, step 수 |
| 51–53 | 해당 agent가 소유한 상자의 실제 X/Y/Z 크기 |
| 54 | env-level 비정상 종료 reason code |

종료 reason code는 `0=없음`, `1=A1 fall`, `2=A2 fall`, `3=both fall`,
`4=bottom displaced`, `5=multiple`, `6=unknown`이다. 성공과 종료 reason은 env-level
값이라 A1/A2 두 row에 동일하게 기록되며, 집계에서는 A1 row의 값을 대표로 읽는다.

## 9. 재현 시 주의점

- 이 스크립트는 학습 run의 env sidecar를 자동으로 읽지 않는다. checkpoint를 학습시킨
  commit과 평가 commit, stage-1 checkpoint, 별도로 override한 `STACK_*`/`MS_*` 변수를
  함께 기록해야 정확히 재현할 수 있다.
- shell에 남아 있는 환경변수 중 스크립트가 기본값 방식으로 export하는 값은 평가 동작을
  바꿀 수 있다. 공유 결과에는 실행 명령과 실제 환경변수를 같이 보낸다.
- `ObjectSet_test_0/1/2`는 서로 다른 object set이 아니라 동일 평가의 세 반복이다.
  첫 반복은 환경별 warm-up으로 버리고 뒤의 두 반복만 사용한다.
- `success_rate`는 top box가 허용 영역에 한 번 들어간 비율이다. 20-step 안정화까지 끝낸
  엄격한 `DONE` 비율과 같지 않다.
- `STACK_EVAL_SUCCESS_MODE`가 다르면 success의 의미가 다르므로 서로 직접 비교하지 않는다.
- 기본 box grid를 쓸 때 `STACK_EVAL_ENVS`는 반드시 9의 배수여야 한다.

## 10. 전달할 때 함께 보내면 좋은 것

최소한 아래 항목을 같이 전달한다.

```text
1. 이 문서와 사용한 source commit hash
2. 평가 대상 policy checkpoint
3. stage-1 checkpoint 경로 또는 파일
4. 실행 명령과 override 환경변수
5. eval_<tag>.log
6. 필요하면 원시 eval_<tag>.npy
```

가장 먼저 볼 값은 `SEQ_STACK_EVAL`의 `success_mode`, `episodes`, `success_rate`,
`terminate_rate`다. 그다음 `SEQ_STACK_BOX_COMBO`로 크기 일반화 실패인지 확인하고,
`place_*`, `retreat_*`, `a2_*`를 비교해 어느 단계에서 경로 추종이 무너졌는지 진단한다.
