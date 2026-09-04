# TokenHSI TensorBoard 지표

TokenHSI 학습 중 `output/<run>/summaries/`에 기록되는 스칼라 태그 전부.
stage1 학습 기준 28~40개. 학습 파이프라인 전체 구조는 [TOKENHSI_STAGE1.md](TOKENHSI_STAGE1.md).

기본 하이퍼파라미터 (`amp_imitation_task_transformer_multi_task.yaml`):
`num_envs 4096` / `horizon_length 32` / `minibatch_size 16384` / `mini_epochs 6` /
`lr 2e-5 constant` / `e_clip 0.2` / `critic_coef 5` / `bounds_loss_coef 10` /
`entropy_coef 0` / `task_reward_w 0.5` / `disc_reward_w 0.5`

---

# x축 3종

같은 값이 접미사만 다르게 여러 번 기록된다.

| 접미사 | x축 | 의미 |
|---|---|---|
| `/frame` | 누적 env 스텝 수 | **기본으로 이걸 봐라.** 샘플 효율 비교에 맞다 |
| `/iter` | epoch 번호 | 몇 번 업데이트했나 |
| `/time` | 벽시계 초 | 실제 걸린 시간 비교용 |

한 epoch = `horizon_length × num_envs` = 32 × 4096 = **131,072 frame**.
즉 `iter 100` ≈ `frame 13.1M`. `losses/*`와 `info/*`는 `frame` 축 하나만 있다.

# rewards0/* — 가장 중요한 곡선

```python
self.current_rewards += rewards              # env가 준 보상만 누적
game_rewards.update(current_rewards[done])   # 에피소드 끝날 때 기록
```

**함정 두 개:**

1. **에피소드 총합(return)이지 스텝당 평균이 아니다.** 에피소드가 길어지면 자동으로 올라간다.
   항상 `episode_lengths`와 같이 봐야 한다.
2. **disc reward가 안 들어있다.** `current_rewards`는 `env.step()`이 반환한 값만 더한다. AMP style
   reward는 나중에 advantage 계산할 때만 합쳐진다
   ([amp_agent.py:131](../TokenHSI-steer/tokenhsi/learning/amp_agent.py#L131) vs
   [:164](../TokenHSI-steer/tokenhsi/learning/amp_agent.py#L164)).
   즉 `rewards0`는 **순수 task reward + power penalty**의 에피소드 합.

뒤의 `0`은 value head 인덱스. `value_size=1`이라 항상 `rewards0` 하나뿐.

## reward_traj/*, reward_sit/*, reward_carry/*, reward_climb/*

stage1처럼 multi-task env일 때만 생긴다. env가 `extras[task_name] = 마스크`를 내보내고
([:1211](../TokenHSI-steer/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py#L1211)),
agent가 과제별로 따로 집계한다.

**stage1 진단은 사실상 이 4개로 한다.** `rewards0`만 보면 "carry는 무너졌는데 sit이 좋아져서 평균은
유지"인 상황을 놓친다. 과제마다 보상 스케일이 달라 절대값 비교는 무의미하고, **각각의 추세**만 본다.

## episode_lengths/*

평균 에피소드 길이(스텝). 상한 600(=20초).

- **오르면**: 안 넘어지고 오래 버틴다 → 대체로 좋음
- **내리는데 `rewards0`은 오른다**: IET가 자주 발동 = **목표를 빨리 달성하고 조기 종료** → 좋음
- **둘 다 내린다**: 넘어지고 있다 → 나쁨

# losses/*

전부 6 mini_epoch × 8 minibatch = **48회 gradient step의 평균**.

| 태그 | 의미 |
|---|---|
| `a_loss` | PPO actor loss. **0 근처에서 진동하는 게 정상.** advantage가 정규화돼 평균 0이라 loss도 0 근처. 절대값이 계속 커지면 정책 급변 신호 |
| `c_loss` | `(V(s) − return)²`. `normalize_value: True`라 정규화 스케일. `critic_coef: 5`로 곱해짐. 계속 크거나 튀면 value가 return을 못 따라감 |
| `bounds_loss` | `Σ clamp(mu-1,min=0)² + clamp(mu+1,max=0)²`. mu가 `[-1,1]` 안이면 정확히 0. `bounds_loss_coef: 10`으로 강하게 눌림. **커지면 정책이 관절 한계를 넘어서려 밀어붙이는 중** |
| `entropy` | sigma가 상수라 **엔트로피도 상수.** `entropy_coef: 0`이라 loss 반영도 없음. **볼 필요 없는 직선** |
| `disc_loss` | BCE + logit_reg + 5×grad_penalty + weight_decay. grad_penalty가 대부분을 차지하므로 절대값보다 아래 acc를 보는 게 낫다 |
| `compensate_composer_reg_loss` | stage2 adapt 계열에서만. stage1엔 없음 |

# info/* — AMP 진단 (핵심)

## disc_agent_acc / disc_demo_acc
```python
agent_acc = mean(logit < 0)   # 정책 궤적을 "가짜"로 맞춘 비율
demo_acc  = mean(logit > 0)   # 모션캡처를 "진짜"로 맞춘 비율
```

**GAN 균형 게이지. 0.5~0.8 사이가 건강하다.**

| 상황 | 의미 |
|---|---|
| 둘 다 **1.0에 붙음** | 판별기 압승. 정책이 뭘 해도 가짜로 찍혀 style gradient가 무의미해진다 |
| 둘 다 **0.5 근처** | 판별기가 구분 못 함. 정책이 잘 모사 중이거나 판별기가 죽었거나 |
| `agent_acc` 높고 `demo_acc` 낮음 | 판별기가 편향됨 (드묾) |

`disc_grad_penalty: 5`, `disc_logit_reg: 0.01`을 크게 잡은 게 전부 **판별기가 이기지 못하게 하려는 것**.

## disc_agent_logit / disc_demo_logit
판별기의 원시 출력(sigmoid 전). agent는 음수, demo는 양수여야 정상. **둘의 간격이 벌어질수록 판별기가
이기고 있다.** acc가 포화(1.0)돼도 logit은 계속 벌어지므로, 포화 후에는 이쪽이 더 민감하다.

## disc_reward_mean / disc_reward_std
**`disc_reward_mean`이 0으로 붕괴하면 style 학습이 죽은 것.** 최종 보상이 `0.5·task + 0.5·disc`니까
절반이 사라진다. 캐릭터가 목표는 달성하는데 동작이 로봇처럼 변하는 실패 모드가 여기 대응한다.
`std`가 크면 env마다 자연스러움 편차가 크다 — 특정 과제만 스타일이 무너지고 있을 수 있다.

## disc_grad_penalty
진짜 데이터 지점에서의 `‖∂logit/∂obs‖²` 평균. **치솟으면 판별기가 과적합**되어 곧 acc가 1.0으로
포화된다. 선행 지표로 쓸 수 있다.

## disc_logit_loss
`‖disc_logits.weight‖²`. 마지막 선형층 가중치 크기. 보통 완만하게 증가하다 안정된다.

# info/* — PPO 진단

| 태그 | 의미 |
|---|---|
| `kl` | 직전 정책과의 근사 KL. `lr_schedule: constant`라 **lr을 조정하지 않는다.** 순수 관찰용. 급등하면 한 업데이트에서 정책이 너무 크게 바뀐 것 |
| `clip_frac` | PPO ratio가 `[0.8, 1.2]` 밖으로 나가 clip된 비율. **0.1~0.3이 건강.** 0에 가까우면 업데이트가 너무 소심, 0.5 초과면 학습 신호가 잘려나감 |
| `last_lr` / `lr_mul` / `e_clip` | `lr_schedule: constant` → **`lr_mul`은 항상 1.0.** 셋 다 평평한 직선. 볼 필요 없다 |
| `epochs` | epoch 번호를 frame 축에 그린 것. 완벽한 직선. 진행률 확인용 |

# performance/*

| 태그 | 의미 |
|---|---|
| `step_fps` | **롤아웃만**의 처리량. 순수 시뮬레이션 속도 |
| `total_fps` | 롤아웃 + 학습 포함 전체 처리량 |
| `play_time` | epoch당 롤아웃에 쓴 초 |
| `update_time` | epoch당 gradient 업데이트에 쓴 초 |

`play_time` : `update_time` 비율로 병목을 본다. num_envs를 늘리면 `play_time`이,
minibatch/mini_epochs를 늘리면 `update_time`이 커진다.

# 볼 순서

```
1. reward_carry/frame    ← 목표 과제가 실제로 나아지나
2. episode_lengths/frame ← 오래 버티나 / IET로 일찍 끝나나 (1번 해석의 전제)
3. disc_agent_acc        ← 0.5~0.8 유지? 1.0 포화면 style 죽음
4. disc_reward_mean      ← 0으로 붕괴 안 했나
5. losses/bounds_loss    ← 0 근처 유지? 커지면 action이 한계 밖으로
6. info/clip_frac        ← 0.1~0.3?
7. losses/c_loss         ← 발산 안 하나
```

**무시해도 되는 것**: `losses/entropy`(상수), `info/last_lr`·`lr_mul`·`e_clip`(상수),
`info/epochs`(직선), `info/kl`(참고용).

# 자주 겪는 패턴

| 증상 | 해석 |
|---|---|
| `rewards0`↑ 인데 `episode_lengths`↓ | IET 조기 종료가 늘어난 것. **좋은 신호** |
| `rewards0`↓ 이고 `episode_lengths`↓ | 넘어지고 있다 |
| `disc_agent_acc` → 1.0, `disc_reward_mean` → 0 | 판별기 압승. 동작이 부자연스러워지는 중 |
| `disc_grad_penalty` 급등 후 acc 포화 | 판별기 과적합. grad penalty를 올리거나 disc lr을 낮춰야 |
| `bounds_loss` 지속 증가 | action saturation. 정책이 관절 한계에 붙어 있다 |
| `reward_carry`만 정체, 나머지 상승 | 과제 간 간섭. `taskInitProb` 조정 검토 |
| `c_loss` 급등 후 `rewards0` 하락 | value 추정 붕괴 → 정책 붕괴 순서 |
