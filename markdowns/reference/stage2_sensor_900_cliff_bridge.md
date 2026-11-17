# Cliff / Bridge 전용 센서 실험

`stage2_sensor_900_cliff_bridge`는 기존 실험과 별도 config·task subclass·출력 경로를 사용한다. 4명/1024환경/8토큰/900step, seed42다. 기본 전이는 `output/stage2/Stage2Distill900_00001500.pth`이며 센서 없이 증류한1500epoch에서 새로 시작한다. 기존 학습의 optimizer나 센서 가중치를 이어받지 않는다.

## 분포

| 환경 | 전체 reset 비율 | 역할 |
| --- | ---: | --- |
| Cliff | 40% | 무대 위4명, 모든 과제 객체·목표는 무대 밖 바닥 |
| Bridge controlled | 24% | 외력 다리 +4명 crosser;50%는 BEFORE 두 쌍,50%는 독립 |
| Bridge free | 18% | 연결 완료 자유 다리 +1명 pusher/3명 crosser; crosser BEFORE 한 쌍 |
| Bridge cooperative | 18% | 기존1/2/3명 PUSH 균등 +1명 crosser;1명 PUSH일 때 나머지2명 BEFORE |

전체 agent-reset 기대값은 횡단42%, Cliff 위험40%, BEFORE24%, coupled7.5%다. 횡단과 관계 비중은 중복된다. 실제 agent-step 비율은 에피소드 길이·조기 종료에 따라 다르다. Cliff의 네 과제와 controlled 독립 과제는 SIT/CLIMB/HOLDING_AT/HOLDING_ON_TOP 균등이다. controlled 독립에서는 다리 토큰을 위해 전원 ON_TOP 조합(조건부 추출 전 확률1/256)만 재추출한다.

## 배치와 제어

Cliff는 전원 무대 위에서 시작하여 밖의 객체·목표를 향하게 한다. 사람에게 외력을 주지 않는다. 기존 낙상 종료/보상을 유지하고 객체 충돌과 최종 BEFORE 경로를 검사한다.

controlled 다리는 같은 크기의 전용 자산을 사용한다. 밀도500, 중력 해제, 회전 속도0이며 질량을 반영한PD 외력을 물리 substep마다 적용한다.2Hz/감쇠비1/가속도 상한6m/s², 연결 완료20%/이동80%, 도착1–18초 log-uniform, 속도0.3–1.5m/s다. 비활성 자산은 멀리 주차하고 외력은 현재 controlled 자산에만 적용한다. free/cooperative 다리의 기존 물성·플랫폼 간격은 바꾸지 않는다.

controlled의4명은 같은 출발 무대에서 시작한다. free의 BEFORE 두 사람은 출발 무대, 독립 crosser는 다리 중간25–45%/후반65–85%에 균등 초기화한다. 연습 분기의 길이는1000 update 동안80/15/5, 다음2000 update 동안 균등으로 전환한다. cooperative는 연결 완료/근접/1.5cm 삽입20/40/40을 유지한다.

BEFORE는 AT→SIT/CLIMB/ON_TOP을 샘플링한다. 선행 상자 크기는X/Y0.40–0.60,Z0.35–0.50이다. 연습 분기의 객체와 목표는 도착 무대에 배치하고, BEFORE 대기 반경이 다리까지 닿지 않게 목표를 안쪽에 둔다. 객체→목표 거리는1–5m, near50%는 지형 배치 후 실제 경로 합으로 검증한다. 출발 무대→객체 접근거리는 횡단 길이에 따라 평지5m 상한을 넘을 수 있다.

외력 다리는 과제·BEFORE alias에 쓰이지 않는 여유 토큰 한 개를 사용하여 entity와 높이맵에 함께 포함한다. 모델 폭·토큰 수·센서 융합 위치·보상·PPO는 기존과 같다. actor encoder/coordination/sigma/입력RMS 동결, 센서/action head/critic 전체/AMP/value 및AMP 정규화 학습 규칙을 유지한다.

## 추론과 진단

전용 test/VNC/eval은 새 checkpoint의 저장 config를 사용한다. `SCENARIO=cliff|bridge`, `BRIDGE_MODE=training|controlled|free|cooperative`를 추론에서 지정할 수 있다. 기본은 학습 분포다. `BRIDGE_CURRICULUM_EVAL` 대신 `BRIDGE_MODE`를 사용한다. 기존 실험 checkpoint를 새 variant로 resume하는 것은 허용하지 않는다.

`sampling/bridge/*`는 분기별 reset·BEFORE·crosser·coupled 인원 수다. `crossing/<branch>/<all|spanN|entry|middle|exit>`와 `crossing/role/<independent|before_leader|before_follower>`는 종료된 crosser-agent episode 분모를 함께 기록한다. 동료 낙상 중단과 본인 낙상을 구분한다. `cliff/*`는 종료 장면·사람 분모, 무대 이탈/높이 하강/낙상/timeout을 구분하며 `controller/*`는 추종 오차를 기록한다. 온라인 횡단 지표를 고정 평가 프로토콜의 성공률로 해석하지 않는다.

운영 명령은 [USAGE](../USAGE.md#cliffbridge-전용-variant), 기본 명령은 [config](../config.md)를 따른다.
