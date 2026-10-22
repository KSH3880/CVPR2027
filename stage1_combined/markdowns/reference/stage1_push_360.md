# PUSH 360도 접근 구현 메모

`stage1_v2_fixed_push_360`은 기존 PUSH 구현에 `stage1Push.navigation` 옵션을 추가한다. 기존 설정과 실행 중 학습에는 적용되지 않는다.

- 최종 범위: 박스 중심→목표 2~10m, 보행 root→박스 중심 1.65~6m, 박스 주변 360도. 반경과 방위각을 각각 균등 샘플링하고 처음에는 박스 중심을 바라본다. 사람·박스 간 초기 겹침은 root 반경 0.45m를 기준으로 재샘플링한다.
- 박스→목표는 초기 지정 면의 법선 방향을 유지한다. RSI는 기존 50%와 자세·상자 진행률·손 정렬을 유지한다. 기존 과제/PUSH transition 비율 50:50, 25kg·마찰·AMP·관측 차원도 유지한다.
- 두 담당 박스의 원래 중심 간격은 `2 × (현재 목표 상한 + 1.3m)`로 확보한다. 10m 목표의 RSI 진행률을 적용해도 두 박스와 각 목표 구간이 겹치지 않게 한다.
- 접근 보상은 박스 중심 거리에서 지정 면 앞 1m 접근 지점까지의 거리로 바꾼다. 상자에 여유 폭 0.45m를 더하고 모서리를 경유하는 우회 거리를 사용해 반대편에서 벽을 뚫고 접근하는 경로를 피한다. 접촉·목표 성공 조건은 유지한다.
- `contact_midpoint_only: true`: 접촉 상태 점수는 양손 중점과 기준점의 XY 오차 및 기존 Z 허용 구간만 사용한다. 각 손과 면 사이 거리는 상태 점수에서 제외하고, 실제 유효 접촉의 거리·힘 판정에는 유지한다. 옵션이 없는 기존 checkpoint/config는 원래 점수를 사용한다.
- 학습 단계는 아래 표를 따른다. 평가는 epoch과 무관하게 최종 단계다. 재개 시 첫 reset부터 저장 epoch의 단계를 사용한다.
- 추론의 보행 시작은 `PUSH_SPAWN_MODE=random`(기본 360도) 또는 `straight`(미는 면 뒤쪽 직선)로 선택한다. 직선 모드는 방위각만 0도로 제한하고 반경 1.65~6m·목표 2~10m·RSI·학습 커리큘럼은 유지한다. 직접 실행 인자는 `--push_spawn_mode`이며 checkpoint 계약과 별개의 추론 옵션이다.

| 새 실험 epoch | 박스→목표 | root→박스 중심 | 미는 면 뒤쪽 기준 방위각 |
| --- | --- | --- | --- |
| 0~999 | 2~5m | 1.65~2.65m | ±45도 |
| 1000~2999 | 2~7m | 1.65~4m | ±90도 |
| 3000~ | 2~10m | 1.65~6m | ±180도 |

- PUSH 제한 시간은 1800step(60초), 기존 과제는 600step(20초)이다. 낙상 종료는 유지한다.
- `PUSH_CHECKPOINT`는 기존 PUSH 모델과 정규화를 복사하고 새 optimizer/epoch으로 시작한다. 원본 fixed teacher를 유지한다. 일반 resume는 동일 설정만 허용하며 물성·AMP·RSI 변경을 전이 경로로 우회할 수 없다.
- TensorBoard의 `relation/push/curriculum_stage`, `goal_max_m`, `spawn_radius_max_m`, `spawn_half_angle_degrees`로 현재 단계를 확인한다.

실행: [config](../config.md). 전이/재개: [USAGE](../USAGE.md#push-360도-접근-전이). 배치·짧은 실행 검증은 연결 확인이며, 360도 우회 접근과 10m 밀기의 정책 수렴을 보장하지 않는다.

검증: CPU190개, 기존 PUSH1500 전이 가중치/정규화 일치,2048환경 짧은 학습/저장과teacher/RMS 불변·finite,64환경 보행/RSI의 각16step 평가,4환경 학습 재개/1환경 평가 noVNC의 실제 화면·움직임을 확인했다. [결과](../../output_etc/check/push360_setup/validation.json).

후속 중점 보상 변경: CPU191개·PUSH1500 전이 계약을 확인했다. 해당 변경의 GPU 재학습은 수행하지 않았다. [결과](../../output_etc/check/push_midpoint_reward/validation.json).
