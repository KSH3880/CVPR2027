# Bridge 고정 역할 + 운반 경계 RSI: `stage2_sensor_900_bridge_before_fixed_carry_rsi`

2026-11-17. [`stage2_sensor_900_bridge_before_fixed`](stage2_sensor_900_bridge_before_fixed.md)에 crosser 운반 경계 RSI와 hidebox를 더한 별도 실험이다. 보상(V4)·모델·동결·전이 원본(V4 Distill2000)은 같다. 구현은 참고 repo `TokenHSI_stage12_teacher_sensor_2only`의 hidebox carry RSI/carry focus를 4명 고정 시나리오에 맞춘 것이다.

## 설정 차이 (env YAML)

| 키 | 값 |
| --- | --- |
| `sensorScene.sensor.exclude_own_payload` | `true`: AT/ON_TOP 소유자(선행 AT·ON_TOP 후행 포함)의 본인 상자를 본인 높이 센서에서 뺀다 |
| `sensorScene.bridge.carry_boundary_rsi` | `version: 1`, `episode_probability: 0.5`, `boundary_probabilities: [0.375, 0.3125, 0.3125]`, `cache_directory: output/rsi_cache_bridge_carry_v1` |

`variant`는 `bridge_before_fixed_v1` 그대로다. `parse_task`는 `carry_boundary_rsi` 블록이 있을 때만 새 클래스 `HumanoidMASensorBridgeBeforeCarryRsi`를 만든다. `checkpoint_contract`가 `sensorScene` 전체를 포함하므로 fixed checkpoint와 계약이 다르다.

## 초기화 분포 (학습만)

- 에피소드의 50%는 운반 RSI 에피소드다. 이때 다리 시작 상태를 '놓임'(start mode 0, aligned)으로 강제하고, crosser(AT 선행)에 입구 직전/다리 진입 직후/출구 직전 = 37.5/31.25/31.25%를 요청한다(기존 경계 비율 30/25/25를 정규화).
- 나머지 50%는 fixed와 같다(다리 놓임/가까이/insert = 20/40/40, 출발 무대에서 상자 집기부터).
- 배치: 일반 장면을 먼저 완성한 뒤, 같은 carryWith clip/time의 사람·상자를 같은 yaw/평행이동으로 옮긴다. 거리 0.35–0.65m, 발 지지·충돌·초기 성공 없음·9.9m 검사(`bridge_carry_boundary_rsi.valid`)와 BEFORE 쌍 layout 검사(`invalid_before_layout`)를 통과해야 한다. 32회 안에 실패하면 일반 시작으로 되돌린다.
- 캐시: 실제 payload 크기(5cm 단위)로 찾는다. crosser 상자 100개 크기 중 72개만 지원한다(가로 0.6m 28개는 선별 통과 프레임 없음). 미지원 크기는 일반 시작이다.
- 평가·viewer(`--eval`)는 운반 RSI와 캐시 로딩이 없고, 초기 상태와 난수 소비가 fixed와 같다(센서 hidebox만 다름).

## AMP 이력 수정

`HumanoidMACoupledPush._init_amp_obs_ref`는 skill 이름과 상관없이 loco/push_loco 라이브러리로 과거 9프레임을 만든다. 운반 RSI 대상의 AMP 이력이 loco로 채워지는 문제가 있어, 새 subclass에서만 `carryWith`를 `ReferenceMotionMixin._init_amp_obs_ref`로 보낸다. 대상자는 loco 이력 목록에서 빼고 carryWith clip/time으로 다시 채운다. 공통 부모는 바꾸지 않았다.

## 진단 scalar

`relation/carry_rsi/<entry_before|entry_after|exit_before>/<requested|executed|fallback>`, `relation/carry_rsi/episodes/<rsi|ordinary>`, `relation/carry_rsi/fallback_cache`, `relation/carry_rsi/fallback_placement`, `relation/carry_rsi/before_layout_rejects`(시도 단위).

검증 결과는 [changelog](../../changelog.md), 캐시 준비는 [USAGE](../USAGE.md#bridge-고정-운반-경계-rsi-캐시).
