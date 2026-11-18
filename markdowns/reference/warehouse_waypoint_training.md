# Warehouse waypoint 센서 학습

실행·운영 옵션은 [USAGE](../USAGE.md#warehouse-waypoint-센서-학습), 기본 명령은 [config](../config.md)를 따른다.

- 실험: `stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint`, Run `Stage2Sensor900WarehouseFixedRolesV4BeforePairsWaypoint`.
- 기존 doorprogress의 보상·AT 높이 마스크·BEFORE 게이트·20% apron BEFORE 2쌍을 유지한다. 나머지 80% 선행만 고정 1.3m의 입구 접근/집기/입구 운반/최종 운반으로 전환한다.
- 관측은 기존 scene/group/sensor 뒤 5개 값 `[leader, phase, door_local_x, door_local_y, door_local_z]`를 붙인다. phase는 비활성 -1, 진입0, 집기1, 복귀2, 최종3이다. 입력 RMS가 추가 값을 통과시키며 actor/critic은 활성 환경의 선행 임베딩만 교체한다. 사람 coordination을 거친 뒤 선택하므로 비선행 입력은 원래 장면을 유지한다.
- 실제 장면과 활성 선행용 GTA 장면을 서로 다른 batch 행에 넣어 encoder/coordination을 각각 한 번 호출한다. 행 간 attention은 없으며 PUSH 두 명·후행은 실제 장면 행에서, 선행만 가상 장면 행에서 출력한다. 선행의 입구 payload/Target과 후행의 최종 목표 alias는 별도 토큰이며, 후행 alias를 원래 최종 목표로 먼저 확정한다. 단일 공통 가상 장면으로 네 사람을 조종하는 방식은 사용하지 않는다.
- PPO에서는 동결된 actor encoder/coordination 출력을 rollout당 한 번 계산해 6회 mini-epoch에서 재사용한다. 센서·action head·critic·AMP는 매 minibatch 갱신하며 특징은 checkpoint에 저장하지 않는다. 기존 checkpoint의 파라미터/optimizer 형태는 같다.
- 물리 step의 관계 보상/HOLDING 갱신 뒤 다음 관측을 만들 때 전환한다. 각 PPO 표본과 bootstrap 관측이 자신의 전환 상태를 갖는다. 현재 task 상태나 mutable network 속성으로 minibatch를 조종하지 않는다.
- 초기 배치는 선행 x=[-3.5,-2.5], |y|=[2.4,3.2], 상자 쪽 부호와 상자 외곽의 벽 여유1.2m를 검사한다. 사람0.85m·물체 외곽0.5m·기존 접근/near/운반 조건을 유지한다. 후보 생성 중 직접 검사하여 중복 작업자 배치와 배치 후 전체 거절을 줄인다.
- 학습 전용 GPU 캐시는 변형별로 조건을 분리한다. 20%의 키는 기존 후행 과제/near의 순서 없는 키를 유지하며, 80%는 templates/predecessor/near/regions를 포함한다. 역할 목록의 내부 나열 순서는 키에서 제외하되 실제 actor/predecessor/region 연결은 유지한다. 상자 크기·env origin·방 기하·방향·PUSH 인원·시작 모드도 비교한다. 키당2개 초기 장면을 순환 저장하고 최대32키는 LRU로 교체한다. 캐시 적중도25%는 새로 만든다. 고정 후보 풀로 역할 샘플링을 대체하지 않는다.
- 캐시에 물리/목표/graph/owner/AMP 초기 이력을 함께 저장한다. 매 에피소드 성공/대기/접촉/waypoint 상태는 기존 reset 경로에서 새로 초기화한다. 평가·viewer는 캐시를 끄고 새 장면을 생성한다.
- 별도 network builder는 기존 센서 모델의 state tensor 이름/크기를 유지하므로 원본 DistillV4 전이와 부분 동결을 재사용한다. 환경의 waypoint/cache/version 계약은 `coupled_push_contract.sensorScene`에 저장되어 기존 모델과 혼용하지 않는다.

역할 분리·PPO 특징 재사용의 검증은 [측정](../../output_etc/check/warehouse_waypoint_roles/benchmark.json)·[확인 학습](../../output_etc/check/warehouse_waypoint_roles/training_verification.json)을 따른다.

초기 GPU 검증 및 속도 측정 산출물: `output_etc/check/warehouse_waypoint_training/`. 짧은 측정은 장기 학습 성공률이나 안정적인 최고 FPS를 보장하지 않는다.

## 1.2m 선행·ON_TOP 후행

- `stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint_ontop`은 DistillV4 2000에서 새로 전이하는 waypoint v2 실험이다. Warehouse의 PUSH 2명 + BEFORE 1쌍에서는 선행과 ON_TOP 후행만 독립 상태를 진행한다. BEFORE 2쌍은 20% 확률로 샘플링하고 양쪽 유도를 끈다.
- 한 입구 좌표를 공유한다. root XY가 1.2m 이내면 실제 상자로, HOLDING 뒤에는 입구 목표로, payload XY가 1.2m 이내면 원래 목표로 바뀐다. 후행 목표는 별도 ON_TOP 받침 토큰의 GTA 좌표다.
- 기존 5개 waypoint 값 뒤에 후행 ID·phase를 추가해 총 10495차원 관측을 사용한다. 실제 장면과 활성 선행·후행 장면을 한 배치에 넣고 역할별 결과를 각자의 출력에만 사용한다. v1과 v2는 각각 저장한 관측 계약으로 실행한다.
- PPO 동결 actor 특징 재사용과 초기 배치 캐시는 v1 경로를 공유한다. 초기 배치 캐시는 물리·그래프·AMP 상태를 저장하고 reset 시 두 waypoint phase를 초기화한다. [기본 명령](../config.md) · [추가 옵션](../USAGE.md#warehouse-ontop-후행-waypoint-센서-학습).

## 가상상자 전용 변형

- 별도 waypoint v3 실험은 7개 waypoint 관측 값을 유지한다. 두 역할의 phase 0은 가상상자, 1은 실제 상자 접근, 3은 첫 집기 이후를 뜻한다. 1.2m 진입 또는 첫 집기에 가상상자가 해제되며 에피소드 중에는 다시 켜지지 않는다. 두 역할 모두 처음부터 원래 목표 관측을 사용한다.
- 선행의 문 통과와 집기 포화 상태는 분리된다. 첫 HOLDING에서도 문 통과 후와 같은 상수 보상을 지급하며, 부분 리셋에서 해당 환경의 두 역할 phase와 보상 포화 상태를 초기화한다.
- 평가의 출구 통과는 첫 HOLDING 이후 root와 물리 상자 중심이 각각 입구 바깥쪽 평면을 실내에서 실외로 통과한 경우다. 평면을 지나는 지점의 횡좌표가 실제 문 폭 안이어야 한다. 최종 목표 성공은 기존 point·region 평가 결과를 사용한다. [실행과 결과 형식](../USAGE.md#warehouse-가상상자-전용-학습).
