# Stage1 전용 1명 단독 평가

2026-10-29 900step 개정 기준 `stage1_single` 프로토콜이다. **일반 Stage1/Stage2 평가는 학습 인원을 따르며 이 평가와 별개다.** BEFORE의 solo 역할도 이 단독 평가를 의미하지 않는다.

- Stage1 및 PUSH 확장 Stage1을 지원한다. AT/ON_TOP/SIT/CLIMB를 각각1명으로 독립 실행하고 PUSH 확장 모델은 PUSH도 추가한다.
- AT/SIT/CLIMB/PUSH 물체1개, ON_TOP payload+받침2개이며 distractor는 없다.
- AT는 기존 지정9종 크기, PUSH는 학습167종 크기이며 나머지 크기·배치·물리는 학습 설정을 따른다. SIT/CLIMB도 학습처럼 동적 상자다.
- loco 시작, 최대 **900step(30초)**, seed0·1·2 각각512환경 첫 episode. 행동당1,536시도다.
- 같은 궤적으로 **점 ever·영역 ever**를 계산한다. 평균 ± 표본 표준편차 및 네 행동 macro를 보고한다.
- 낙상 frame까지 채점하고 이전 성공을 보존한다. 다인 일반 평가에서도 한 명 낙상으로 전체 종료하지 않는다.

```bash
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_single_eval.sh \
  checkpoints/Stage1_00078000.pth \
  --output output_etc/check/stage1_78k_single
```

기하 수식·BEFORE 두 사람 성공·일반 평가와 차이·샘플링·낙상 제어·저장/재채점은 [공통 평가 규약](markdowns/reference/evaluation.md)에 있다. 원래900step/고정 SIT·CLIMB 상자 문서는 [과거 규약](markdowns/legacy/evaluation_single_20261022.md)에 보존했다.

PUSH 모델은 `stage1_push_single_eval.sh`를 사용한다. [PUSH 설정](markdowns/reference/stage1_push.md), [전체 실행 명령](markdowns/config.md)을 따른다.
