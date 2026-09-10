# Carry state-relation v0 구현 결과

2026-09-10 · `edge_a2_gta_state` · 기존 HEAD `9b68fca`에서 구현. 명세 원문과 이후 대화 합의의 차이는 명세 상단에 기록했다.

## 바로 학습

저장소 루트에서:

```bash
bash tokenhsi/scripts/multi_agent/ma_carry_relation_train.sh
```

기본값: **2048 environments, PPO minibatch 16384, mini-epochs 6**, M=2 humans / O=3 boxes. Horizon 32, learning rate 2e-5, gamma .99, task/AMP weight 각각 .5. AMP minibatch는 기존 4096이며 PPO minibatch와 다른 설정이다.

이 코드의 PPO minibatch 16384는 **agent sample 수**다. M=2이면 8192 scene rows를 한 minibatch로 처리하고 각 scene의 두 agent를 함께 유지한다. 한 rollout은 `2048*32=65536` scenes / `131072` agent samples다.

새 mode는 scratch 학습이 기본이며 기존 20k checkpoint를 자동으로 사용하지 않는다. 학습 config의 기본 max_epochs는 기존과 같은 1,000,000이므로 원하는 상한이 있으면 `MAX_ITERATIONS`를 지정한다.

```bash
# 동일 기본값을 명시
bash tokenhsi/scripts/multi_agent/ma_carry_relation_train.sh 2 2048 3

# 선택: seed, 최대 epoch, GPU 지정
SEED=42 MAX_ITERATIONS=20000 TOKENHSI_GPU=0 \
  bash tokenhsi/scripts/multi_agent/ma_carry_relation_train.sh

# 새 모드 checkpoint에서 optimizer/epoch와 함께 재개
RESUME_CHECKPOINT=/absolute/path/to/new_relation_checkpoint.pth \
  bash tokenhsi/scripts/multi_agent/ma_carry_relation_train.sh
```

기존 trainer는 종료를 `epoch > max_epochs`로 검사한다. 따라서 `MAX_ITERATIONS=1`은 실제 2 epoch를 실행하며, resume에서 값은 추가 epoch 수가 아니라 절대 epoch 상한이다. legacy trainer의 이 동작은 수정하지 않았다.

결과는 `output/ma_carry_relation_v0/CarryRelationV0_<timestamp>/`에 저장된다. `nn/`, `summaries/`, 실제 설정 snapshot `relation_config.yaml`, `diagnostics/relation_samples.csv`가 생성된다. `OUTPUT_PATH`로 상위 출력 폴더를 바꿀 수 있다. 스크립트가 기존 `runtime_env.sh`를 이용해 conda/라이브러리/데이터 경로를 준비한다.

## 평가 / 뷰어

### State delta ×2 비교 실험

기존 v0 YAML은 그대로 두고 `amp_humanoid_ma_carry_relation_state2.yaml`을 추가했다. 두 env YAML의 유일한 설정 차이는 `relationReward.state_delta_weight: 1.0 → 2.0`이다. Holding/At delta의 양·음 성분이 함께 2배이며, 속도 .2 / bonus 5 / AMP / penalty / 물리 설정은 동일하다. PPO/network train YAML도 공유한다.

```bash
RELATION_VARIANT=state2 bash tokenhsi/scripts/multi_agent/ma_carry_relation_train.sh
```

기본 env2048 / MB16384 / ME6 / M2 / O3, seed는 현재 v0 run과 같은 **9896**이다. scratch 학습이며, 별도로 `RESUME_CHECKPOINT`를 지정하지 않는다. weight=1의 v0 checkpoint는 reward metadata가 다르므로 이 변형으로 resume할 수 없다.

출력은 **`output/ma_carry_relation_state2/CarryRelationState2_<timestamp>/`**로 분리된다. 체크포인트·TensorBoard·config snapshot·sampled diagnostics가 이 run 아래에 저장된다. 기존 `output/ma_carry_relation_v0/`는 건드리지 않는다. `OUTPUT_PATH`를 명시하면 그 값이 우선한다. `RELATION_VARIANT`를 생략하면 기존 v0 동작을 유지한다. `SEED`로 seed를 명시적으로 변경할 수도 있다.

state2 모델의 평가/뷰어에도 같은 variant를 지정해야 한다:

```bash
RELATION_VARIANT=state2 HEADLESS=0 \
  bash tokenhsi/scripts/multi_agent/ma_carry_relation_test.sh /path/to/state2_checkpoint.pth 2 4 3 3
```

아래 기존 평가 명령은 variant를 생략한 **v0 checkpoint용**이다.

```bash
# 순서: checkpoint, humans, environments, boxes, repeats
bash tokenhsi/scripts/multi_agent/ma_carry_relation_test.sh /path/to/checkpoint.pth 2 16 3 3

# GUI 디스플레이가 있는 터미널
HEADLESS=0 bash tokenhsi/scripts/multi_agent/ma_carry_relation_test.sh /path/to/checkpoint.pth 2 4 3 3

# 필요 시 기존 noVNC 실행 도우미
HEADLESS=0 bash tokenhsi/scripts/multi_agent/run-gui.sh \
  bash tokenhsi/scripts/multi_agent/ma_carry_relation_test.sh /path/to/checkpoint.pth 2 4 3 3
```

평가는 유한한 반복이며 각 scene의 첫 episode만 집계한다. agent success, 모든 subgoal의 누적 완료, 현재 모든 target의 유효 만족을 구별하고, 기존처럼 agent 0만 보고 scene success라고 집계하지 않는다. 결과는 output 상위 폴더의 `metrics/relation_<timestamp>.json`이다. 시험 중인 미학습 smoke checkpoint는 금방 넘어질 수 있다. 새 policy의 성공적인 운반 영상을 검증했다는 뜻은 아니다.

TensorBoard:

```bash
/home/cvlab/anaconda3/bin/tensorboard \
  --logdir output/ma_carry_relation_v0 --port 16007
```

기존 16006 baseline 서버와 포트를 분리한다. 데이터/뷰어 asset는 앞서 연결한 TokenHSI symlink를 그대로 사용한다. 이번 작업에서 원본 데이터나 기존 학습 모델은 수정하지 않았다.

## 구현 내용

- `utils/relation_task_spec.py`: Carry-only graph compiler, directed Holding/At IDs 6/7, schema/config validation, checkpoint 계약.
- `env/tasks/multi_agent/relation_reward.py`: pure PyTorch evaluator, 공통 progress, soft prerequisite, signed delta, 일회성 bonus 및 history runtime.
- `relation_task.py` + `humanoid_ma_carry.py`: assignment에 따른 logical object slot, kinematic reset seed, bounded all-initial-success resampling, reward → history commit → next observation 순서. 성공만으로 scene reset/terminate하지 않는다.
- `amp_network_builder_ma.py`: 8-type static semantic edge encoder 및 batch별 5D dynamic edge MLP `5→32→64`, layer/head별 zero-init projection. actor/critic 독립, NONE pair도 attention에 남고 dynamic bias만 해당 directed edges에 들어간다.
- `scene_normalizer.py` / MA wrapper: H/O RMS 유지, Target/pose/history passthrough. 새 모드에서 optional flat clipping도 Target/pose/history를 바꾸지 않는다.
- `ma_agent.py` / `ma_players.py`: suffix가 저장된 scene PPO batch를 모델에 전달, scale/보상 로그, strict checkpoint metadata, scene-aware 평가.
- 새 env/train/smoke YAML 및 train/test scripts. 기존 legacy YAML/script의 기본값은 바꾸지 않았다.

관측은 `H223*M | O30*O | Target1*M | pose7*(2M+O) | relation_state(2M,4) | done(M)`, 총 `247M+37O`이다. M1/O1=284, M2/O2=568, M2/O3=605, M3/O4=889. suffix 순서는 relation별 `phi,gate,satisfied,achieved`, 이후 owner별 done. Canonical edges는 `[Holding0,At0,Holding1,At1,...]`이며 physical box assignment를 바꿔도 assigned object가 logical slot 앞쪽에 정렬된다.

Holding은 두 손 평균과 object center 사이 거리의 `exp(-5*d²)`. At은 XYZ near score와 기존 XY 10cm / Z 1mm putdown 판정을 결합한다. 속도는 source displacement / control dt의 XY 성분을 **post-step source→target 방향**으로 투영한다. 이동 target의 속도를 빼지 않는다. gate는 이전 phi, success prerequisite는 이전 achieved를 사용한다.

완료 전 task는 Holding/At signed delta와 soft-gated velocity, 첫 valid success bonus 5의 합이다. 완료 후 해당 task 합은 0이며 power/collision/box-speed penalty와 AMP는 계속 적용된다. 성공 후 phi는 물리 상태를 따라 계속 관측되지만 achieved/done history와 bonus는 재활성화되지 않는다.

## 기록 항목

- `reward_terms/*`: Holding/At delta와 velocity를 분리, bonus, power, collision, box speed, total, AMP, combined.
- `relation/holding/*`, `relation/at/*`: phi/gate/current satisfaction/achieved, threshold 상하 crossing, 각 signed 성분.
- `relation/goal_xy_error`, `goal_z_error`, 손 midpoint/좌우 hand center 거리, root–box XY, box speed, near/put, target XY crossing.
- `relation/near_{0.1,0.25,0.5}m/*`: active agent의 근접 fraction, 속도/velocity reward, stationary-unplaced fraction. 거리 band는 **로그용이며 reward cutoff가 아니다**. 이 band 평균은 step별 조건부 평균을 rollout 동안 평균한 값이므로 fraction과 함께 해석한다.
- `relation/near_unplaced_slow/*`: XY<0.5m, box speed<0.05m/s, 미배치/미완료 조건의 AMP·combined 및 sample count. AMP는 해당 rollout의 동일 discriminator로 계산하며 전체 sample을 합친 조건부 평균이다. count=0이면 평균 로그의 0은 관측값이 아님에 주의한다.
- first holding/valid time 및 관측 비율, current-all-valid와 ever-all-done. 시간 mean은 현재 episode에서 해당 사건이 관측된 agent에 한정한 snapshot 평균이며 episode duration 평균이 아니다.
- `relation_attention/{actor,critic}/*`: static/dynamic row-centered RMS, layer별 GTA-QK row-centered RMS와 attention entropy. 100 encoder forwards마다 재측정한다. 처음 forward의 zero bias만 보고 scale을 평가하지 않는다.
- CSV: 기본 2 env, 30 control step마다 episode ID/step 및 agent별 시계열. 모든 env를 매 step CPU로 복사하지 않는다. `box_bottom_height_proxy`는 회전을 반영하지 않은 center-minus-half-height proxy다.

## 실제 실행 검증

CPU tensor/network tests **48 passed**:

```bash
source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python -m pytest -q \
  tokenhsi/tests/test_relation_reward.py \
  tokenhsi/tests/test_relation_runtime.py \
  tokenhsi/tests/test_ma_relation_policy.py \
  tokenhsi/tests/test_ma_scene_policy.py \
  tokenhsi/tests/test_ma_scene_features.py
```

velocity target/stop/reverse/vertical/invariance, 1mm/10cm 경계, prerequisite 방향, signed delta, 이전 achievement 순서, bonus once, 부분 reset, history suffix, agent permutation, batch isolation, 두 optimizer step 이후 dynamic MLP gradient, entity-count-independent parameters, RMS bypass, strict schema/legacy 경로를 검증했다. 알려진 exploit fixture는 예방했다고 assert하지 않고 실제 허용 결과를 검사한다.

Isaac Gym / CUDA:

1. M1/O1 및 M2/O3, 32 env에서 PPO/AMP 3 epoch 실행·저장.
2. M2/O3 checkpoint strict resume, optimizer/epoch 복원 후 epoch 5까지 실행·저장.
3. M1/O1 및 M2/O3의 실제 64-transition 검사: kinematic reset phi, 부분 reset 격리, history와 next observation 일치, simulator reward와 pure reference 일치, clipping bypass, success-only 비종료. `tests/smoke_relation_sim.py`로 재현 가능.
4. 기존 sibling `edge_geo_2`의 MB16K/ME6/seed42 20k 모델 strict load 후 finite simulator rollout 통과. legacy의 velocity 성분도 pinning/anti-kick 바깥에서 shadow 비교했다.
5. 새 checkpoint의 유한 평가 완료, agent/scene 분리 JSON 저장. 거의 미학습인 3-epoch 모델의 8-scene 평가 성공률은 0이며 동작 품질을 평가할 단계가 아니다.
6. **실제 production 기본값 env2048 / MB16384 / ME6 / M2/O3**로 scratch 2 epoch 실행·저장. 모든 model tensor 및 TensorBoard scalar finite. 115 scalar tags와 sampled CSV 생성 확인. actor/critic dynamic projection이 0에서 학습된 것도 확인했다.

Production 설정 검증 실행:

```bash
MAX_ITERATIONS=1 SEED=42 OUTPUT_PATH=output/diagnostics/relation_v0_production_settings \
  bash tokenhsi/scripts/multi_agent/ma_carry_relation_train.sh
```

검증 산출물: `output/diagnostics/relation_v0_production_settings/CarryRelationV0_10-14-47-16/`. checkpoint epoch=2, frame=262144, obs=605, dt≈1/30. 이번 짧은 실행의 total throughput은 약 21k agent samples/s였으나 공유 GPU의 초기 2 epoch 수치이며 안정적인 성능 benchmark로 보지 않는다. 본 학습은 계속 실행해 두지 않았다.

작은 smoke 재실행:

```bash
SMOKE=1 MAX_ITERATIONS=2 bash tokenhsi/scripts/multi_agent/ma_carry_relation_train.sh 1 32 1
SMOKE=1 MAX_ITERATIONS=2 bash tokenhsi/scripts/multi_agent/ma_carry_relation_train.sh 2 32 3
```

`SMOKE=1`만 PPO minibatch=1024, AMP minibatch=512, demo/replay buffers=4096인 별도 train YAML을 선택한다. mini-epochs는 6 그대로다. 작은 env를 production YAML로 실행하면 rollout보다 minibatch가 커질 수 있으므로 반드시 smoke 옵션을 사용한다.

## 남은 한계와 비교 기준

- **학습이 실행됨과 운반 품질이 좋아짐은 다르다.** 새 policy의 장기 성공률/정지/실제 grasp·release·안정 배치는 아직 검증하지 않았다. hand contact/grasp/release는 별도 확정 센서가 없어 unknown이며 영상 관찰이 필요하다.
- 1mm tolerance와 Holding scale 유지에는 기존 20k 정책의 실측을 참고했다. 이전 baseline probe에서 384 agent episodes 중 264가 1mm putdown 조건을 한 번 이상 달성했고 263은 8 consecutive steps 달성했다. 이는 새 정책의 결과가 아니다. 원자료와 해석은 `output/diagnostics/carry_reference_review/REPORT.md`에 있다.
- gate는 정확한 자동 감속 장치가 아니다. putdown 미성립 시 phi_At≤.5, gate_At≤약 .0001234이므로 속도 항이 거의 안 줄어드는 경우가 남는다. 이번에는 합의대로 식을 유지하고 로그로 확인한다.
- midpoint-only Holding false positive, 한 번 잡은 뒤 kick/throw, goal-first-then-touch가 success 이력 조건상 허용될 수 있다. stable contact/lift/continuous carry/release 조건을 새로 강제하지 않았다.
- gate-toggle closed cycle은 state reward 합 약 +.399011, gamma .99 discounted 합 약 +.379310이다. potential-based/cycle-free 보상이라고 주장하지 않는다.
- reset 시 이미 만족한 relation을 seed하며 초기 유효 완료에 bonus를 지급하지 않는다. 모든 subgoal이 처음부터 완료인 scene은 최대 16회 재표본화 후 계속 발생하면 오류를 낸다.
- 구 checkpoint → 새 mode의 warm-start 변환은 구현하지 않았다. legacy checkpoint는 legacy 모드에서 그대로 쓰고, 새 checkpoint는 schema/보상 설정을 확인한 뒤 strict load한다. diagnostics-only 변경은 허용한다.
- 기존 trainer의 resume는 weight/optimizer/epoch를 복원하고 물리 episode와 relation history는 새 reset에서 시작한다. simulator state/RNG/replay까지 포함하는 bitwise mid-episode resume를 구현한 것은 아니다.
- OnTop/Beside/Push/Pull, relation/operator 범용 입력, planner/world model은 구현 범위 밖이다.

이후 비교에서는 동일 seed/초기화 분포/frame budget/MB16K/ME6를 맞추고, reward 평균뿐 아니라 history-qualified success, 현재 실제 배치, stall/drop/overshoot 및 영상을 함께 확인한다.
