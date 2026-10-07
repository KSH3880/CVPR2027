# 독립 Stage 1 실험

현재 비교 대상은 [config.md](config.md)의 27·28·30·31·32번이다. 두 에이전트가 세 물체에서 서로 다른 주 물체를 사용하고 AT goal도 공유하지 않는다. 한 장면에서 ON_TOP 과제를 둘 다 동시에 뽑지 않는다.

| 번호 | 달라지는 점 |
| ---: | --- |
| 27 | 독립 물체·goal binding, 점 기준 성공 |
| 28 | ON_TOP·CLIMB 성공을 상판 안쪽 영역으로 변경 |
| 30 | SIT도 상판 영역으로 변경하고 agent별 유효 edge 보상을 평균 |
| 31 | 30번 + AT/ON_TOP 가까운 시작·CLIMB 후반 RSI·어려운 과제 증량·거리/높이 shaping |
| 32 | 31번의 시작 상태·RSI·샘플링 유지, AT/ON_TOP progress·CLIMB state는 30번 식 |

27~32번은 모두 schema 9 semantic graph를 사용하며 모델 구조는 같다. 각 config는 reward 계약과 output 경로가 달라 학습 checkpoint를 임의로 resume하지 않는다. 30~32번의 팀 task 보상은 자기 90%, 동료 10%이고 단일/두 edge 모두 최대 0.6이다.

30번과 32번은 현재 학습 중이다. 성공률 비교에서는 같은 epoch/checkpoint 단계와 평가 preset을 맞춰야 한다. 32번의 초기 CLIMB 성공에는 후반 RSI에서 성공에 가까운 자세로 시작한 효과가 포함될 수 있다.

실행 명령은 [config.md](config.md), 구현 위치는 [structure.md](structure.md), 지표 정의는 [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 따른다.
