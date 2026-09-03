# Multi-Agent Carry 좌표계 및 평가 reset 수정 명세

이 문서는 다른 TokenHSI 계열 repo에서 다음 두 문제를 동일하게 찾아 수정하기 위한 이식 명세다.

1. env origin 불일치 + distractor 공유 정규화 오염
2. 현재 test의 reference 비정렬 box reset

보상, 정책 네트워크, Transformer, AMP/PPO 계산은 수정 범위가 아니다.

## 1. 필수 좌표 불변식

Isaac Gym의 actor root state는 world 좌표다. Motion dataset과 agent spawn offset은 env-local 좌표이므로 모든 env에서 다음 식을 지켜야 한다.

```text
world_position = env_origin[env_id] + agent_spawn_offset[agent_id] + reference_local_position
```

`envSpacing` 숫자를 위치에 직접 더하면 안 된다. `envSpacing`은 Gym이 병렬 env를 배치할 때 사용하는 설정이고, 실제 변환값은 `gym.get_env_origin()`으로 얻은 env별 origin이다.

환경 생성 후 origin을 저장하는 구현이 없다면 다음과 같은 텐서를 만든다.

```python
origins = [gym.get_env_origin(env_ptr) for env_ptr in envs]
env_origins = torch.tensor([[o.x, o.y, o.z] for o in origins], device=device)
```

## 2. Issue 1 수정: train/test 좌표계 통일

### Reference humanoid reset

`_reset_ref_state_init()` 또는 이에 해당하는 함수에서 reference root position에 agent offset과 env origin을 모두 더한다. test일 때만 origin을 더하는 조건문이 있으면 제거한다.

```python
offset = agent_spawn_offsets[curr_agent] + env_origins[curr_env]
root_pos = reference_root_pos + offset
```

### Reference assigned box reset

`pickUp`, `carryWith`, `putDown` motion과 연결된 box도 humanoid와 동일한 env/agent offset을 사용한다.

```python
offset = agent_spawn_offsets[curr_agent] + env_origins[curr_env]
box_pos = reference_box_pos + offset
```

Ground 보정, box rotation, velocity 초기화는 기존 로직을 유지한다. origin을 더한 뒤 z ground 보정을 수행해도 되며, 일반적인 평면 env의 origin z는 0이다.

### Target reset

Reference putdown target과 무작위 target 중심도 world 좌표로 만든다.

```python
# Reference putdown target
offset = agent_spawn_offsets[curr_agent] + env_origins[curr_env]
target_pos = reference_target_pos + offset

# loco/pickup/carry용 무작위 target 중심
center_xy = agent_spawn_offsets[curr_agent, :2] + env_origins[curr_env, :2]
```

### Distractor box

미할당 distractor는 해당 env origin을 중심으로 arena-local 반경 안에서 생성한다.

```python
center_xy = env_origins[env_ids, :2]
distractor_xy = sample_arena_xy(center_xy)
```

Assigned box와 distractor를 같은 normalizer에 넣는 구조 자체는 유지할 수 있다. 모든 object가 같은 좌표계로 표현될 때 shared `RunningMeanStd`는 slot permutation에 독립적인 정상 설계다. 이 문제를 normalizer 분리로 우회하지 않는다.

## 3. Issue 2 수정: reference-aligned test reset

기본 test에서는 humanoid와 assigned box가 같은 motion ID/time을 사용해야 한다.

```yaml
box:
  reset:
    testRandomArenaSpawn: false
```

코드 fallback도 수정할 수 있다면 안전한 기본값을 `False`로 둔다.

```python
random_arena_box_spawn = bool(
    args.test and box_cfg["reset"].get("testRandomArenaSpawn", False)
)
```

동작 기준은 다음과 같다.

- `pickUp`, `carryWith`, `putDown`: humanoid와 동일 motion ID/time의 reference box 사용
- `loco` 또는 default state: 기존 학습과 동일하게 owner 주변에서 box 생성
- unassigned distractor: 해당 env arena 안에서 독립 생성
- 독립 random arena test: 난이도/OOD stress test가 필요할 때만 명시적으로 활성화

`_reset_all_boxes_random_arena()` 같은 기능은 삭제하지 않아도 된다. 기본 평가 경로에서 reference box를 덮어쓰지 않게 하는 것이 핵심이다.

## 4. 기존 체크포인트 처리

수정 전 다중 env 학습 체크포인트는 잘못된 object x/y 통계가 `RunningMeanStd`에 누적됐을 수 있다. 모델 weight만 정상이어도 오염된 RMS를 함께 불러오므로 좌표 수정 효과를 제대로 얻을 수 없다.

- 수정 효과 검증은 신규 학습 체크포인트를 기준으로 한다.
- 기존 체크포인트 resume은 기본 권장 경로가 아니다.
- RMS 초기화 resume을 별도로 실험할 수는 있지만 정책 입력 분포가 즉시 바뀌므로 신규 학습과 같은 결과를 보장하지 않는다.

## 5. 검증 절차와 합격 기준

GPU 사용 시 대상 머신의 지정 GPU만 노출한다. 이 repo에서는 GPU 0만 사용한다.

```bash
CUDA_VISIBLE_DEVICES=0 sh tokenhsi/scripts/multi_agent/ma_carry_train.sh 2 2048 3
```

먼저 작은 multi-env 진단을 수행한다. 권장 조건은 `M=2`, `O=3`, `num_envs>=16`이다.

1. `world humanoid position - env_origin`이 모든 env에서 arena 크기 수준인지 확인한다.
2. `world box position - env_origin`이 assigned/distractor 모두 arena 크기 수준인지 확인한다.
3. Reference humanoid와 assigned box의 실제 위치를 `reference + agent offset + env origin`과 비교한다.
4. Reference 위치 최대 오차가 `1e-4 m` 이하인지 확인한다.
5. Fresh object RMS의 local x/y 표준편차가 한 자릿수 m 범위이며 수십~수백 m로 커지지 않는지 확인한다.
6. test에서 `testRandomArenaSpawn=False`이고 reference box가 reset 후에도 동일 motion/time 위치에 있는지 확인한다.
7. headless 학습, headless test, viewer/noVNC test를 각각 smoke test한다.

현재 repo의 수정 후 진단값은 다음과 같았다.

```text
env origin xy max:          70.000 m
humanoid arena-local max:    5.217 m
box arena-local max:         4.988 m
reference root max error:    0.000007629 m
reference box max error:     0.000007629 m
object local xyz std:       [1.9643, 1.7183, 0.4304] m
```

수정 전 체크포인트에서 관찰된 object local x/y 표준편차 약 `155/140 m`가 재현되면 좌표계가 아직 섞여 있는 것이다.

## 6. 수정 시 금지사항

- `envSpacing` 값을 actor position에 직접 더하지 않는다.
- env origin을 이미 포함한 world position에 다시 더하지 않는다.
- 문제를 숨기기 위해 observation clipping 범위나 RMS epsilon을 변경하지 않는다.
- shared entity normalizer를 임의로 slot별 normalizer로 바꾸지 않는다.
- 보상 가중치, AMP scale, 모델 구조, reference motion dataset을 함께 변경하지 않는다.
- 수정 전 체크포인트 성능으로 좌표 수정의 성공 여부를 판단하지 않는다.

## 7. 다른 repo의 LLM 에이전트에게 줄 작업 지시

이 문서와 대상 repo를 제공한 뒤 다음처럼 지시한다.

> Multi-agent carry 구현에서 이 명세의 두 버그가 존재하는지 관련 reset 함수와 config만 확인하라. 존재하면 최소 diff로 env origin 좌표계를 통일하고 test 기본값을 reference-aligned로 수정하라. 보상, 모델, 학습 파이프라인은 변경하지 말라. 기존 체크포인트를 resume하지 말고 지정 GPU에서 multi-env 좌표 진단과 학습/test/viewer smoke test를 수행해 수치와 결과를 보고하라.
