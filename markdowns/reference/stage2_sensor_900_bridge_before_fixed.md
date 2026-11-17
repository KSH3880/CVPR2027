# Fixed BEFORE Bridge

`stage2_sensor_900_bridge_before_fixed`는 compact(`cliff_bridge_compact_v1`)를 상속한 Bridge 전용 실험이다. variant는 `bridge_before_fixed_v1`이다. 실행 명령은 [config](../config.md)에 있다. 전이는 compact와 같은 `output/stage2/Stage2Distill900V4_00002000.pth`다.

## 장면

모든 episode에 같은 역할을 고정한다. 1 env에 4명이다.

| 역할 | 내용 |
| --- | --- |
| pusher | 2명. 바닥에서 다리를 민다(span 2 기하). |
| 선행 | BEFORE 선행 AT. 사람과 운반 payload는 출발 무대에 둔다. 80%는 건너편 무대가 AT 목표이고, 20%는 출발 무대가 AT 목표다. |
| 후행 | 선행 상자에 SIT/CLIMB/ON_TOP 중 균등으로 하나를 한다. 80%는 건너편, 20%는 선행과 같은 출발 무대에 spawn한다. ON_TOP의 자기 payload도 후행과 같은 무대에 둔다. 선행 상자 크기 [.4,.6]×[.4,.6]×[.35,.5]는 세 과제를 모두 지원한다. |

20% 같은 무대 구성에서는 사람 두 명·선행 payload·BEFORE 목표·후행 ON_TOP payload를 출발 무대에 함께 배치한다. pusher 두 명은 두 구성 모두 기존처럼 다리를 민다. 그룹별 인원에 따른 compact 무대 크기 선택은 유지한다.

fixed와 layout-cache의 새 학습 기본값은 `stateInit: Default`다. 모든 사람을 기본 자세·0 초기 속도로 시작하며 모션 RSI를 사용하지 않는다. 이 장면의 경계 RSI도 실행되지 않는다. AMP 전문가 학습은 유지하며, reset된 사람의 AMP 과거 프레임은 현재 기본 자세로 모두 채운다. 부분 배치 재시도로 reset 대상 목록이 줄어들어도 전체 reset 대상의 AMP 이력을 초기화한다. Default 슬롯도 loco와 같은 BEFORE follower 필터를 거쳐 공유 상자/목표를 별도 ON_TOP 배치로 덮어쓰지 않는다. carry-RSI 실험의 `stateInit: Random`과 운반 RSI 설정은 유지한다.

YAML에서 compact와 다른 값은 다음뿐이다.
- `sensorScene.mixture`: bridge 1.0
- `bridge.branches`: cooperative 1.0
- `variant`
- `stateInit: Default`
- 실험 이름
- 아래 V4 보상 키
- `bridge.before_training`: 같은 무대 확률 0.2와 다리 중심 progress 설정

다리 시작(이미 놓임 0.2 / 가까이 0.4 / insert 0.4), 거리 커리큘럼, 무대 크기는 compact와 같다. `pusher_counts`와 `carry_from_start_probability`는 compact 검증 계약을 위해 남겼지만 이 variant에서는 쓰지 않는다. compact boundary RSI는 기존 규칙상 carry_from_start를 제외하므로 적용되지 않는다.

Default 배치에서는 모든 강체 이동과 발 높이 보정 후 비활성 후행 Target만 자기 root로 맞춘다. 초기 사람 위치와 scene origin의 좌표 차이가 비활성 Target에 누적돼 GTA 입력에 수 km 좌표가 들어가던 오류를 방지한다. 활성 AT/PUSH 목표·사람·payload·보상·Random RSI 경로는 유지한다. layout-cache도 이 경로를 거친 교정 상태를 저장한다.

## 보상 (V4)

기본 보상은 dev 저장소 `stage2_sensor_900_cliff_bridge_four_transfer_v4_reward`의 V4 보상과 같다. relationGraph, relationReward, stage2, before, stage2JointPush, jointPush YAML 섹션은 유지하고 다음 progress 항을 더한다.

- **80% crossing 구성의 선행에게만** `0.2 × r`을 추가한다. `d`는 선행 payload 중심과 현재 연결 다리 상자 중심의 XY 거리이며, `r = 1 / (1 + d / 1m)`이다. 다른 사람과 20% 같은 무대 구성은 항상 0이다.
- 출발 무대→도착 무대 방향으로 `(payload_xy - bridge_xy)`를 투영한 값이 0 이상이면 중심 통과로 판정한다. 한 번 통과하면 이후 거리가 멀어지거나 되돌아가도 해당 에피소드에서 `r=1`을 유지한다.
- 실제 부분 reset과 캐시 복원에서 통과 상태를 초기화한다. 보상은 `reward_terms/bridge_center_progress`, 구성 비율은 `relation/bridge_before/same_platform_fraction`에 기록한다.
- 이 변경은 fixed와 layout-cache YAML의 `bridge.before_training`이 있을 때만 적용한다. carry-RSI 실험에는 이 블록이 없어 기존 배치와 보상을 유지한다. 변경 전 fixed/cache checkpoint와는 환경 계약이 다르므로 새 학습에서 적용한다.

| 키 | 값 |
| --- | --- |
| `stage2.reward_source` | `v4_teachers` |
| `before.reward_at_z_tolerance` | 0.001 |
| `before.waiting_penalty` | radius_buffer 1.5, ontop_distance_source payload |
| `stage2JointPush.premature_motion_penalty` | weight 0.5, speed_threshold 0.05, force_epsilon 1e-6 |

- AT 성공 판정: BEFORE 관계이거나 공동 PUSH가 아닌 모든 작업자는 Z 20cm까지 성공으로 본다.
- 보상 포화와 성공 보너스: Z 1mm 이내여야 준다.
- 대기 감점: 반경은 지지 상자 반대각+1.5m다. ON_TOP은 운반 상자 XY로 진입을 판정한다.
- 공동 PUSH 조기 이동 감점: `premature_motion_penalty` reward term에 기록된다.
- 다리 외력 루프에서도 물리 step마다 CONTACT와 감점을 누적한다.
- 모든 코드 분기는 `reward_source: v4_teachers`일 때만 켜지므로 다른 config는 기존 보상을 유지한다.
- `relation/v4/*`에 AT 성공·포화, PUSH CONTACT, 조기 이동 감점, 대기 활성을 기록한다.

## 구현

- `_sample_bridge_curriculum`: pusher 수를 2로 고정한다.
- `_draw_episode_graph`
  - compact 샘플링 결과에 `utils/bridge_before_fixed.fixed_roles`로 역할을 덮어쓴다.
  - 에피소드마다 같은 무대 20%/crossing 80%를 뽑고 선행 목표와 후행의 무대를 정한다. carry_from_start=True는 유지한다.
- `_place_bridge`: crossing은 기존 배치 뒤에 후행을 `_place_group`으로 건너편에 단독 배치한다. 같은 무대 구성은 pusher만 기존 경로로 배치하고 BEFORE 두 명을 하나의 그룹으로 출발 무대에 배치한다.
  - 같은 occupancy를 쓰므로 AT 목표와 충돌하지 않고, 상자는 다리 착지점 근처를 피한다.
  - 이후 기존 `invalid_before_layout`이 후행과 목표 사이 거리 1m 이상을 검사한다.
- 다리 진입 side는 기존처럼 +y다. 출발 무대 A/B를 50%로 뽑으므로 crosser 기준 좌/우 진입이 반반이다.
- 학습 로그 `bridge_before/entry_left|entry_right|follower_sit|follower_climb|follower_ontop`에 이 비율과 후행 과제 수를 기록한다.
- 평가 환경변수는 `SCENARIO=bridge`, `BRIDGE_MODE=cooperative`, `BRIDGE_PUSHERS=2`만 허용한다(미지정 가능).
