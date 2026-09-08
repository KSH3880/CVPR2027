# Coordinator → masteer sequential-stack 연결

`juan`의 기존 sequential-stack을 상속하는 별도 실행 경로다.
기존 `run.py`, task registry, 평가·viewer 스크립트 및 정책 네트워크는 수정하지 않았다.

```text
현재 simulator의 A1/A2 root·box·goal·phase
 → frozen coordinator PTH: joint 33점 (x,y,v) 후보
 → phase를 고려한 후보 선택
 → 0.1m/320점 경로·속도 버퍼
 → root-local 6점/12-D steer
 → 기존 masteer PTH의 340-D 관측 → full-body action
```

## Phase별 적용

| 기존 phase | coord 출력을 적용하는 agent | 기존 동작을 유지하는 agent |
|---|---|---|
| A1_PLACE / VERIFY_BOTTOM | A1 | A2 대기 |
| A1_RETREAT | 없음 | A1 후퇴, A2 대기 |
| A2_RESUME / VERIFY_STACK | A2 | A1 후퇴 목표 유지 |
| DONE | 없음 | 기존 종료 처리 |

순서 전환, box의 XYZ 목표·높이, bottom을 따라가는 top 목표,
A1의 virtual retreat carry 관측, 보상과 성공 판정은 부모 환경이 담당한다.
coord 입력에는 GT 경로나 stack phase ID를 추가하지 않는다. 기존 0..3 Carry phase만
실제 hand proximity와 lift 상태로 구성하며, 별도 stack phase는 적용할 agent를 고르는 데 쓴다.

기본 6 action step마다 재계획하며, stack/Carry phase 변경이나 5cm 초과 목표 이동은
즉시 재계획한다. 속도 명령은 0.375..1.5m/s, step간 변화는 기본 0.75m/s²로 제한한다.
waypoint의 XY와 속도를 동일한 경로 호길이로 보간한다. endpoint clipping은 기존 ms18 규칙이다.

후보 평가는 HH/BB/HB XY 거리와 완료시간을 사용한다. 적용하지 않는 agent와 box는
현재 위치에 정지한 것으로 근사해, 실제로 실행하지 않을 learned Carry 경로를 평가하지 않는다.
후퇴의 미래 동작과 쌓기 높이를 예측하는 모델은 아니므로 이 근사는 stack 안전성 보장이 아니다.
유효한 후보가 모두 unsafe이면 비용이 가장 낮은 유효 후보를 선택한다.

NaN, anchor 불일치, buffer 초과, 속도 범위 위반, 급곡선은 거부한다.
동일 phase/goal에서 invalid replan은 이전 계획을 유지한다. 첫 계획이나 변경된
phase/goal에서 유효 계획이 없으면 그 agent의 native steering 경로로 fallback한다.

## 실행

masteer와 coordinator PTH는 별도 파일이다. `MS_CKPT`는 stage1 backbone이며,
기본값은 `TokenHSI-masteer/output/ckpt_stage1.pth`다.
활성 conda 환경을 사용하거나 `TOKENHSI_PYTHON`에 Isaac Gym 환경의 Python을 지정한다.
활성 환경이 없으면 `CONDA_BASE` 또는 `$HOME/anaconda3`, `$HOME/miniconda3` 아래의
`TOKENHSI_CONDA_ENV`(기본 `tokenhsi118`)를 사용한다.

아래 `COORD_CKPT`는 **실제로 보유한 C13 파일 경로로 교체**한다.

```bash
COORD_CKPT=/path/to/c13_path10_s0/coord_c2_000300.pth \
MA_GPU=0 STACK_EVAL_ENVS=270 \
bash scripts/masteer/eval_coord_sequential_stack.sh \
  TokenHSI-masteer/output/sequential_stack/anti_feat_top_s2/Humanoid_00010000.pth \
  c13_s2_10000
```

결과는 `runs/results/coord_sequential_stack/eval_<tag>.npy`와 `.log`로 분리한다.
기존 평가와 동일하게 환경별 첫 episode를 버리고 이후 2회, box 3×3 조합을 집계한다.
54열 metric 형식은 유지한다. 로그의 `COORD_STACK_SUMMARY`에서 실제 설치한
agent 경로 수(`installed_agent_rows`), invalid/unsafe, fallback을 확인할 수 있다.
`root_mae` 등의 경로 오차는 이제 coord가 선택한 경로를 기준으로 한 오차다.

데스크톱 또는 기존 X 서버의 `DISPLAY`가 있는 곳에서 native Isaac Gym viewer를 연다.
이 새 스크립트는 VNC/Xvfb 서버를 생성하거나 기존 서버를 종료하지 않는다.

```bash
COORD_CKPT=/path/to/c13_path10_s0/coord_c2_000300.pth \
MA_GPU=0 STACK_VIEW_BOX_IDS=0,7 \
bash scripts/masteer/view_coord_sequential_stack.sh \
  TokenHSI-masteer/output/sequential_stack/anti_feat_top_s2/Humanoid_00010000.pth 1
```

기존 viewer의 경로 띠와 steering 점이 실제 교체된 버퍼를 읽는다.
`STACK_TOP_FOLLOWS_BOTTOM=0` 등의 기존 task 옵션은 두 새 스크립트에도 적용할 수 있다.

## C13 체크포인트

`traj` commit `5f39199f7ea9396532fe251c46aa634734ee9f10`에는
`scripts/coord/c13_pathguard.sh`와 `c13_path10_s0`, `c13_path30_s0` 설정이 있다.
두 설정은 C2 schema의 joint waypoint/speed-cap MLP이며 경로 residual loss 계수만 다르다.
2026-09-08 현재 이 작업 머신에서 학습된 C13 PTH는 찾지 못했다.
Git으로 옮긴 코드에 학습 가중치가 포함된 것은 아니다.

체크포인트의 schema/config와 `extras.random_priority`를 자동으로 읽는다.
C13의 episode-random priority agent는 원본 계약대로 직선·평속 경로를 사용하며,
반대 agent의 learned 경로·속도를 사용한다. 이는 stack phase의 순서/대기 gate를 바꾸지 않는다.
따라서 해당 phase의 활성 agent가 priority인 경우에는 nominal 경로가 적용된다.

`COORD_CKPT`를 생략하면 `runs/coord/c13*/coord_c2_latest.pth`가 정확히 1개 있을 때만
자동 선택한다. 없거나 여러 개면 경로를 지정하도록 실패하며 무작위 모델로 대체하지 않는다.

## 검증

```bash
cd TokenHSI-coord
OMP_NUM_THREADS=1 python -m unittest discover -s coordinator/tests -v
```

92개 CPU 테스트 통과. 새 검사는 C13와 같은 architecture의 임시 체크포인트 round-trip,
공간 위치에 맞는 속도 보간, 기존 ms18 `_steer_obs`를 이용한 실제 12-D 창 변화,
부분 env reset, phase gate, 목표 변경, invalid fallback, 우선권 보존을 포함한다.
runtime 검사의 simulator 부모는 tensor fixture이며 physics rollout 성공을 의미하지 않는다.
실제 학습된 C13 및 GPU simulator를 이용한 성공률 검증은 아직 수행하지 않았다.
