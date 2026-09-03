# Carry viewer 박스 reset 문제와 수정 명세

## 문제 현상

`carryWith` 또는 `putDown` RSI로 GUI 테스트를 시작하면 humanoid의 손과 팔은 박스를 든 자세지만, 실제 박스는 바닥에 나타났다. 바닥의 박스 위치는 에피소드가 바뀌어도 다음 좌표로 반복됐다.

- object 0: 약 `(1.0, 0.0, 0.185)`
- object 1: 약 `(-1.5, 0.0, 0.11)`

이 좌표는 무작위 reset 결과가 아니라 `_build_box()`에서 actor를 생성할 때 사용하는 초기 pose다.

## 원인

`HumanoidMACarry.render()`와 `_capture_video_frame()`이 `_update_marker()`를 호출하고 있었다. `_update_marker()`는 마커와 플랫폼을 움직이기 위해 `set_actor_root_state_tensor_indexed()`를 물리 step 직전에 다시 호출했다.

Isaac Gym GPU pipeline에서는 reset 직후 일부 actor만 대상으로 root-state setter를 다시 호출하면, 대상에 포함되지 않은 동적 박스가 actor 생성 pose로 되돌아갔다. 따라서 단순한 viewer 표시 문제가 아니라 실제 PhysX 상태가 변경됐다.

레퍼런스 repo는 `_update_marker()`의 setter 대상에 박스도 포함해 이 현상을 우회하고 있었다. 현재 repo에서 viewer 경로를 read-only로 만들기 위해 박스를 대상에서 제외하면서 문제가 드러났다.

분리 실험 결과는 다음과 같다.

| 조건 | 결과 |
|---|---|
| headless | RSI 박스가 손에 정상 배치됨 |
| GUI, marker-only setter | 박스가 생성 pose로 롤백됨 |
| GUI, platform-only setter | 박스가 생성 pose로 롤백됨 |
| GUI, `_update_marker()` no-op | headless와 동일 |
| GUI, 첫 physics step까지 setter 지연 | headless와 동일 |
| GUI, 레퍼런스처럼 박스도 setter에 포함 | headless와 동일 |

## 영향 범위

- 일반 headless 학습: video recording이 꺼져 있으면 `_update_marker()`를 호출하지 않으므로 직접적인 영향 없음
- `ma_carry_test.sh` GUI 테스트: 영향 있음
- `ma_carry_watch.sh` 시각화 학습: 영향 있음
- headless 학습/테스트에서 video recording 활성화: `_capture_video_frame()` 경로를 통해 영향 가능

네트워크, policy, reward, AMP, RSI 표본 추출 자체에는 문제가 없었다.

## 적용한 수정

파일: `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py`

1. `render()`에서 `_update_marker()` 호출을 제거했다.
2. `_capture_video_frame()`에서 `_update_marker()` 호출을 제거했다.
3. 매 프레임 root-state를 변경하던 `_update_marker()`를 제거했다.
4. `_reset_env_tensors()`에서 해당 env의 다음 actor를 하나의 indexed setter로 함께 적용하도록 변경했다.
   - 동적 박스
   - 목표 마커
   - 시작 플랫폼
   - 목표 플랫폼
5. 목표 마커 위치는 reset 시 `_tar_pos`에서 복사한다.

목표와 플랫폼 위치는 에피소드 reset에서만 결정되고 에피소드 중에는 변하지 않으므로, 렌더마다 갱신할 필요가 없다. 이 방식은 viewer와 video capture 경로를 물리 상태에 대해 read-only로 유지한다.

## 검증 결과

단일 conda 환경에서 GPU 한 장(`TOKENHSI_GPU=0`)만 사용해 검증했다.

- Python 문법 검사 통과
- GUI `carryWith` 전용, 2 agents, 1 env, 2 objects, seed 123, 3 episodes 통과
- 첫 렌더 이후에도 박스가 손에 유지됨
- GUI와 headless의 박스 궤적이 수치적으로 동일함
- 에피소드별 agent 0 초기 박스 위치:
  - episode 0: `(2.410, 0.422, 1.135)`
  - episode 1: `(2.113, 0.102, 1.104)`
  - episode 2: `(2.449, 0.511, 1.136)`
- reset 직후 박스-손 거리: 약 `0.008~0.023 m`
- 30 step 후 박스-손 거리: 약 `0.021~0.059 m`

따라서 에피소드마다 같은 바닥 좌표에 나타나던 현상은 제거됐고, RSI의 motion phase와 박스 위치가 정상적으로 달라진다.

## 다른 repo에 적용할 때

1. viewer 또는 camera capture 함수에서 actor root-state setter를 호출하는지 검색한다.
2. 시각화용 actor 갱신이 동적 actor와 같은 root-state tensor를 공유하는지 확인한다.
3. 에피소드 동안 고정된 marker/target은 reset setter에 포함하고 렌더에서는 갱신하지 않는다.
4. 한 env의 reset에 필요한 동적 actor와 시각화 actor를 가능한 한 한 번의 indexed setter로 적용한다.
5. headless와 GUI에서 같은 seed를 사용해 다음 값을 비교한다.
   - RSI motion phase
   - box assignment
   - 박스 위치
   - 손과 박스 사이 거리
6. 첫 번째 렌더 전후 프레임을 각각 캡처한다. 첫 프레임만 보면 setter 적용 전 상태가 찍혀 문제를 놓칠 수 있다.

