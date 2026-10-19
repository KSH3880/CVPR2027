# Lift — TeamHOI full/masked AMP

현재 권장 lift AMP 설정은 `lift_stage1_teamhoi.yaml`이다. 기존 `lift_stage1.yaml` 단일 masked AMP는 비교용으로 보존한다. 두 설정은 checkpoint 호환이 없으며 새 설정은 scratch로 시작한다. SUPPORT/LIFT 보상, 상자0.55×0.65×0.35m·5kg, 고정 목표 중심 높이1.6m는 동일하다.

## 공식 코드에서 가져온 구조

참조 commit: [`sail-sg/TeamHOI@6fdc9885f8a9c82854be99adca6f2570e67ac63a`](https://github.com/sail-sg/TeamHOI/tree/6fdc9885f8a9c82854be99adca6f2570e67ac63a).

- [trans_agent.py](https://github.com/sail-sg/TeamHOI/blob/6fdc9885f8a9c82854be99adca6f2570e67ac63a/teamhoi/learning/transformer/trans_agent.py): 두 demo buffer, 정규화 전 mask, 별도 RMS, 거리 history의 sigmoid 혼합, outside-only full updates, paired replay, 각 판별기의 BCE/logit reg/gradient penalty/weight decay.
- [teamhoi_transformer.py](https://github.com/sail-sg/TeamHOI/blob/6fdc9885f8a9c82854be99adca6f2570e67ac63a/teamhoi/learning/transformer/teamhoi_transformer.py): 독립 초기화된 full/masked MLP와 logits.
- [reward_functions.py](https://github.com/sail-sg/TeamHOI/blob/6fdc9885f8a9c82854be99adca6f2570e67ac63a/teamhoi/env/tasks/reward_functions.py): root XY에서 물체 하단 perimeter 후보까지의 거리.
- [teamhoi_stage1.yaml](https://github.com/sail-sg/TeamHOI/blob/6fdc9885f8a9c82854be99adca6f2570e67ac63a/cfg/teamhoi_stage1.yaml): distance threshold1.0m.

MIT 출처·라이선스는 `third_party/teamhoi/`에 보관한다. 전체 코드/actor를 그대로 복제하는 대신 아래 AMP 경로를 현재 TokenHSI scene PPO에 이식했다.

## 현재 구현

| 경로 | Full discriminator | Masked discriminator |
|---|---|---|
| Reference | 기존 loco12개만 | loco0.33 + OMOMO train lift2개0.67 |
| 입력 | raw129×10 | 같은 raw history의 팔38feature를 매frame0으로 만든129×10 |
| 정규화 | 독립1290-D window RMS | 마스킹 **후** 독립1290-D window RMS |
| 정책 negative | history 모든 시점에서 거리>1m인 live/replay | 모든 live/replay |
| 업데이트 | live와 replay 각각 outside501개 이상일 때 | 매 업데이트 |

마스크는 native129-D의 `25:49`(양 어깨·팔꿈치6D), `91:99`(해당8DOF 속도), `117:123`(양손 위치)다. 공식125-D에서 양손 위치113:119인 것을 그대로 복사하지 않았다. 기존 정책 관측/행동은 바꾸지 않으며 원본 AMP history는 보존한다.

각 agent의 자기 상자 하단 local 사각 perimeter에64개 점을 균등 샘플링하고 world XY에서 root와 가장 가까운 점의 거리를 d로 쓴다. 상자 회전을 반영한다. 각 environment reset에서는 새 현재 거리로 전체10시점 history를 채우며, 이후 raw AMP history와 같은 순서로 이동한다.

`w_mask = mean_history(sigmoid((1.0m - d) × 15))`

`r_style = (1-w_mask) × r_full + w_mask × r_masked`

각 r은 공식 코드와 같은 `-log(max(1-sigmoid(logit),1e-4)) × disc_reward_scale`다. 구현은 같은 포화를 갖는 softplus 식으로 수치적으로 안정화했다. 가까이서도 full 보상을 hard-off하지 않고 sigmoid로 섞는다.

두 discriminator loss는 같은 계수로 합산한다. Full은 모든 history 시점이1m 밖에 있는 raw live와 raw replay를 각각501개 이상 확보했을 때만 업데이트한다(공식 `>500`). 샘플이 부족하면 full loss는0이며 masked loss와 PPO는 계속 학습한다. Full demo 수는 현재 outside 샘플 수에 맞춰 무작위 선택한다.

Replay는 raw `amp_obs`와 거리 `amp_cond`를 한 행으로 함께 저장·샘플링한다. keep-probability와 capacity truncation에도 같은 index를 적용한다. 우리 `(T,N×M,...) → (N×T,M,...)` scene grouping에도 두 필드가 같이 이동한다.

두 모델과 두 RMS, dual AMP 설정 metadata를 checkpoint에 저장하고 학습 재개·평가에서 복원한다. 단일 AMP checkpoint, masked RMS 누락, 다른 dual 설정은 거부한다. 기존 agent의 기본 확장 경로는 이전 정규화 순서·replay 순서를 유지한다.

고정2-agent 환경에서는 두 agent가 항상 활성화되므로 TeamHOI의 variable-team padding/비활성 agent mask는 쓰지 않는다. Actor/critic은 기존550-D clean-scene GTA 구조를 유지한다. 공식 reference의 ACCAD pickup 대신 요청한 OMOMO train lift를 masked branch에 사용한다.

## 실행

`stage1_fix/`에서 실행한다. GPU6만 허용하고 기존 MPS 자동 연결만 사용한다. MPS daemon 시작/종료/설정 변경은 하지 않는다.

```bash
# 새 본학습 (현재 자동 시작하지 않음), scratch2048env
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/lift_stage1_teamhoi_train.sh

# bounded 학습 경로 점검
TOKENHSI_GPU=6 MAX_ITERATIONS=1 OUTPUT_PATH=../runs/lift_teamhoi/check \
  bash tokenhsi/scripts/multi_agent/lift_stage1_teamhoi_train.sh

# 평가, 로컬 viewer, 서버 VNC
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/lift_stage1_teamhoi_test.sh CHECKPOINT.pth 16 3
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/lift_stage1_teamhoi_test.sh CHECKPOINT.pth 1 1
TOKENHSI_GPU=6 PORT=6081 bash tokenhsi/scripts/multi_agent/lift_stage1_teamhoi_vnc.sh CHECKPOINT.pth 1 1

# 재개
TOKENHSI_GPU=6 RESUME_CHECKPOINT=CHECKPOINT.pth \
  bash tokenhsi/scripts/multi_agent/lift_stage1_teamhoi_train.sh
```

평가에서 `CFG_ENV=<run>/relation_config.yaml`로 checkpoint의 저장 설정을 쓸 수 있다. 기존 기본 `lift_stage1_train.sh`는 여전히 단일 masked AMP 비교용이다. 이번 구현은 새 `lift_stage1_teamhoi_*` 실행기를 사용한다.

## 검증

- CPU37 tests 통과: 마스크/원본 비변경, 서로 다른 RMS, history sigmoid 평균과 outside-min, full 활성/skip 두 경로의 loss·backward, paired replay·scene grouping, 두 discriminator checkpoint 복원/단일checkpoint 거부, 기존 단일 AMP hook 순서, 기존 lift/관계/속도보상 회귀.
- GPU6 기존 MPS에서4env reset/8step/부분reset·raw demo/full demo·거리 history 연결 검증 통과.
- 2048env `MAX_ITERATIONS=1` 유한 점검에서 기존 rl_games 종료 계약에 따라 실제2update 후 정상 저장. 두 update 모두 `full_updated=1`, full/masked loss 유한값을 확인했다. mean masked blend는 약0.097/0.199였다.
- Checkpoint의 full/masked RMS 모두1290-D이며 각각 count>1, masked 팔 feature의 mean은0, full 팔 mean은nonzero를 확인했다.
- 4env×1회 유한 평가의 checkpoint·두 RMS 로딩 통과(40steps/8trials, lift 성공0). 구현 연결 검사이며 학습 성능으로 해석하지 않는다.
- 2048env에서 optimizer·두 RMS를 포함한 checkpoint 재개도 유한1update(epoch3) 후 정상 종료·저장했다.
- 첫 학습 점검은 실행기 복사 중 모션 목록 이름이 잘못 바뀌어 시작 전에 실패했다. 실제 기존 `dataset_lift_stage1.yaml` 경로로 수정한 후 위 검증을 완료했다. 실패 로그도 보존한다.

산출물은 저장소 루트 `runs/lift_teamhoi/`에 있다. 점검 checkpoint는 `check/LiftStage1TeamHOI_16-17-20-06/nn/LiftStage1TeamHOI.pth`, 재개본은 `resume_check/LiftStage1TeamHOI_16-17-22-29/nn/LiftStage1TeamHOI.pth`, 평가는 `eval/lift_eval.json`이다. 본학습은 시작하지 않았다.

## 파일

- `tokenhsi/utils/dual_amp.py`: mask/blend/rim distance/loss/checkpoint 계약.
- `tokenhsi/learning/multi_agent/dual_amp_{model,agent,player}.py`: 두 discriminator·학습·평가.
- `tokenhsi/env/tasks/multi_agent/humanoid_ma_lift.py`: opt-in raw AMP와 거리 history.
- `tokenhsi/learning/multi_agent/ma_agent.py`: 기존 동작을 기본값으로 유지하는 확장 지점.
- `tokenhsi/data/cfg/multi_agent/lift_stage1_teamhoi.yaml`, `tokenhsi/data/cfg/train/rlg/amp_ma_lift_stage1_teamhoi.yaml`: 별도 설정.
- `tokenhsi/tests/test_dual_amp.py`: dual AMP와 기존 단일 AMP 확장 경로 회귀.
