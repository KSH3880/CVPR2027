# Compact Cliff/Bridge

`stage2_sensor_900_cliff_bridge_compact`는 surface variant에서 분리한 신규 실험이다. 현재 학습을 바꾸거나 새 학습을 시작하지 않는다. 실행 진입점은 [config](../config.md)에 있다. 기본 전이는 원본 `output/stage2/Stage2Distill900_00001500.pth`이며 확인용 초기화 checkpoint를 본학습 시작점으로 사용하지 않는다.

## 무대와 물리

| 무대 이용 인원 | X/Y 각 변 범위 |
| --- | --- |
| 1 | 5–6m |
| 2 | 5.5–6.5m |
| 3 | 6.5–7.5m |
| 4 | 7–8m |

생성 때 X/Y를 0.5m 간격으로 샘플링한 A/B 각4개 자산을 준비한다. reset에서는 출발·목적 무대 방문자의 합집합으로 등급을 선택한다. 바닥 PUSH 담당은 제외한다. Cliff는4명 전원7–8m이며, 기존처럼 목표/객체는 무대 밖이다. 지형 actor는 환경당9개(Cliff1+A4+B4)이고 비활성 자산은 지하로 격리한다. GPU reset은 사람·상자·무대 상태를 해당 환경의 actor batch로 함께 반영한다.

양 끝 접합 틈은 각각5cm이며 gap은 다리 길이+0.10m이다. 삽입측 모서리 정렬·1.5cm 걸침 시작·다리 profile/질량/높이·상면1.0/본체0.4 마찰을 유지한다. 센서와 발 지지는 실제 활성 geometry를 사용하며 틈을 가상으로 메우지 않는다.

## 거리 커리큘럼

새 학습 update0–500은 초기 단계,500–2500은 선형 보간,2500이후는 최종 단계다. Bridge 장면별 초기거리20%/현재단계60%/최종거리20%를 샘플링한다. 원본 Distill1500의 epoch을 새 커리큘럼에 더하지 않는다. 동일 variant resume은 복원한 학습 epoch을 사용한다.

| 항목 | 초기 | 최종 |
| --- | --- | --- |
| 첫 출발 행의 입구 거리 | 0.8–1.2m | 0.8–2m |
| 출발 행 간격 | 1.2m | 1.5m |
| 첫 목표 행의 출구 여유 | 객체 반경+0.8–1.5m | 객체 반경+1.5–3m |
| 운반 payload→목표 | 1–2m | 1–4m |

목표는 사람/객체 점유에 따라1.2m 행 간격과 무대 경계를 고려한다. BEFORE 선행 목표는 객체 반경+2.5m 대기 buffer+0.8m를 우선한다. 운반 payload는 해당 목표 주변에서 배치한다. 운반 거리 상한은 같은 무대 안의 구간에 적용하고, 출발측 payload를 다른 무대로 운반하는 기존 샘플은 물리적 gap을 유지한다. 출발측 payload가 있는 crosser는 상자 크기·사람 간격·접근 공간을 더한 최소 출발 거리를 우선한다. 따라서 빈손 횡단의0.8–1.2m 범위를 그대로 강제하지 않는다. Cliff 거리, 기존 다리 길이 스케줄, 시나리오/관계/인원 비율은 유지한다.

## 입출구 RSI

다리가 연결된 장면에서 적격 crosser 최대1명을 고른다. 선택자 비율은 일반20%/입구직전30%/진입직후25%/출구직전25%다. 경계 거리는0.35–0.65m다. 기존 free 분기의 강제 중간/후반 초기화는 이 규칙으로 교체한다. 나머지 crosser는 일반 거리 초기화다.

기존 loco 모션에서 속도0.3–1.5m/s·전진 방향 오차30도 이내 상태를 재추출한다. 기존 AMP 모션 id/time에 같은 상태를 기록한다. 실제 발 메시의 지지·침투와 사람/객체 충돌을 확인한다. 모션/배치 재추출 상한은32회이며 소진되면 같은 과제·관계의 일반 초기화로 복구한다. 다리 미도착 또는 출발측 payload를 먼저 집어야 하는 crosser는 경계 RSI에서 제외한다. 선행 성공·파지 상태를 인위적으로 생성하지 않는다.

일반 test/VNC는 최종거리·일반출발을 사용한다. 기존 `SCENARIO=cliff|bridge`, `BRIDGE_MODE=controlled|free|cooperative` 선택을 지원한다. 검증기의 `_compact_test_kind`와 `_compact_test_alpha`는 eval에서만 사용할 수 있다.

## 진단과 호환

`compact/reset/<full|entry_before|entry_after|exit_before>/{requested,executed,fallback}`와 사유별 fallback, `compact/crossing/<mode>/...`, 무대 크기·거리 alpha를 기록한다. 기존 분기/길이/역할별 지표도 유지한다. 다리 위 RSI를 진입 성공으로 계산하지 않는다. 일반 출발 횡단은 별도로 비교하며 온라인 횡단 지표는 최종 과제 완료율이 아니다.

네트워크·입력 차원·actor encoder/coordination/입력RMS 동결과 critic 전체/value RMS 학습은 surface와 같다.4명/1024환경/8토큰/900step,Cliff/Bridge40/60%,Bridge 외력/완료/협력40/30/30%다. 신규 설정은 sidecar와 checkpoint 계약에 저장되고 다른 variant의 resume/추론은 거부한다.

검증 산출물은 [bridge_compact_20261113](../../output_etc/check/bridge_compact_20261113/)에 둔다. GPU 검사는 여유8GiB 이상에서만8환경 단기 reset/추론으로 수행한다. 이번 구현에는 본학습·PPO update·현재 학습 중단이 포함되지 않는다.
