# Carry 고정 경로 속도 회피

두 agent의 직선 XY 경로는 episode reset 때 한 번 설치한다. 속도 정책은
경로를 바꾸지 않고 `0, 0.375, 0.75, 1.125, 1.5 m/s` 중 명령을 선택한다.
실제 이동 속도는 동작 정책의 응답이며 명령과 같다고 가정하지 않는다.

첫 실험에서는 매 episode 무작위로 우선 agent를 선택해 1.5m/s 명령으로
진행시킨다. 다른 agent가 감속·정지·재출발을 학습한다. 두 agent를 함께
학습하는 최종 통합 모델에 앞서 속도로 회피 가능한지 확인하는 실험이다.
시작은 `carryWith` 모션으로 상자를 든 상태이며 직각 교차를 만든다.
20%는 3m 떨어진 평행 경로로, 불필요한 정지를 비교하는 대조 장면이다.
목표는 교차점에서 4m 떨어져 있어 서로 가까운 목표를 만들지 않는다.

## 실행

로컬 기본 executor는 읽기 가능한 `/home/visitor/koo_cvpr`의
`ms18_carry_steer50_s0/.../Humanoid_00012000.pth`다. 서버에서는 executor와
stage1 파일 경로를 명시한다. 로컬 코드가 서버에 반영됐다고 가정하지 않는다.
실험 이름은 새 이름을 사용한다. 같은 launcher와 생성된 config를 학습·평가에 쓴다.

```bash
# 감속/정지/재출발의 실제 응답 확인: 충돌 없는 평행 장면
CARRY_SPEED_ENVS=8 CARRY_SPEED_FREE_PROB=1 CARRY_SPEED_ITERS=1 \
  CARRY_SPEED_HORIZON=40 MA_GPU=0 bash carry_speed/launch.sh stop_probe_01 probe

# 동일 seed·환경에서 정속 대조군과 기하학적 양보 규칙 비교
CARRY_SPEED_ITERS=2 MA_GPU=0 bash carry_speed/launch.sh crossing_nominal_01 nominal
CARRY_SPEED_ITERS=2 MA_GPU=0 bash carry_speed/launch.sh crossing_rule_01 rule

# 속도 정책 학습
MA_GPU=0 bash carry_speed/launch.sh crossing_train_01 train

# 정책 평가 / 학습 재개 (새 태그 필요)
MA_GPU=0 bash carry_speed/launch.sh crossing_eval_01 policy "$EXECUTOR" "$SPEED_POLICY"
MA_GPU=0 bash carry_speed/launch.sh crossing_resume_01 train "$EXECUTOR" "$SPEED_POLICY"
```

위 명령의 작업 디렉터리는 `TokenHSI-coord`다. 실행 전 GPU 점유를 확인한다.
서버 경로는 `MS_CKPT=/path/to/stage1.pth`, 세 번째 인자 executor로 지정한다.
기존 Carry planner 체크포인트는 속도 정책 체크포인트와 호환되지 않는다.
재개는 정책·optimizer·iteration을 복원하며 시뮬레이터 episode/RNG는 새로 시작한다.

## 설정과 기록

- 기본값: 256환경, 200 iterations, rollout 32 × low-level 6스텝, episode 360스텝.
- `CARRY_SPEED_ENVS/ITERS/HORIZON/LOW_STEPS/SEED/FREE_PROB`로 제어한다.
- `COORD_ACCEL_UP/DOWN` 기본값 0.75/1.0 m/s². 이 환경은 최소 속도 clamp 없이 0 명령을 보낸다.
- 보상은 기존 물리 태스크 보상 + 진행량×2 − 근접 cost×10 − 0.01,
  두 상자 배송 완료 시 +6. 정지 자체에는 보너스를 주지 않는다.
- `CARRY_SPEED_COLLISION_COEF`, `CARRY_SPEED_PROGRESS_COEF`로 보상 계수를 설정한다.
- PPO: LR 3e-4, clip 0.2, entropy 0.003, KL 조기 중단 0.02,
  gradient norm 1.0. Gaussian std를 사용하지 않는다.
- 산출물: `runs/carry_speed/<tag>/env.yaml`, `run.env`, `run.log`,
  `metrics.jsonl`, `summary.json`, TensorBoard, 학습 체크포인트.
- 평가/probe의 `trace.npz`: 요청/전송/실제 속도, root, 상자, held,
  done, 역할, episode age, 고정 경로를 매 결정 시점 기록한다.

`proxy_episode_fraction`과 `proxy_step_fraction`은 기존 거리 기반 근접
proxy이며 물리 접촉률이 아니다. 완료 episode가 0이면 episode 비율을
성능으로 해석하지 않는다. 요청 속도 0만으로 실제 정지가 입증되지 않으므로
trace에서 전송 속도 0, 실제 속도 감소, 상자 유지, 재출발을 함께 확인한다.
단기 smoke test는 회피 성능이나 수렴의 검증이 아니다.

## 2026-10-01 실제 검증

- 단위 테스트 7개 통과. 8환경 × rollout 4 × 6스텝의 GPU smoke에서
  새 frozen executor 로딩, PPO 6회 업데이트, 속도 체크포인트 저장까지 통과했다.
- 평행 경로 8환경에서 6초간 정지 요청한 probe:
  전송 명령이 0.05m/s 미만인 204개 결정 표본의 실제 평균 속도는 0.571m/s,
  실제 0.2m/s 미만 표본은 0개, held 평균은 0.995였다.
  4~6.5초 구간에서도 평균 0.546m/s였고, 재출발 8~9초 구간은 1.129m/s였다.
  명령에 따른 감속·재가속은 보였지만 완전한 중간 정지는 확인하지 못했다.
  현재 frozen executor로 완전 정지를 전제로 하는 회피 성능을 보장할 수 없다.
- reset 없는 인접 결정 시점의 XY 경로는 trace에서 동일함을 확인했다.
- seed 0, 8환경, 384스텝의 교차 장면 비교:
  정속은 완료 8개/배송 3개, proxy episode 6/8, proxy step 510/3060;
  양보 규칙은 완료 9개/배송 4개, proxy episode 7/9, proxy step 214/3054.
  step 근접은 줄었지만 episode 근접 회피는 확인되지 않았다. 단기 n=1이며
  성공률 개선을 주장하는 평가가 아니다. 비교 trace와 summary는 각각
  `eval_speed_cross_nominal_20261001_a`, `eval_speed_cross_rule_20261001_a`에 있다.
- 초기 3개 smoke 실패(config task 인덱스, 초기 버퍼 생성 순서,
  IET 비활성 시 부모 reset의 필수 버퍼)와 성공 로그를 모두 보존했다.
