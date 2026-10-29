# 유지하는 BEFORE 실험

세 설정은 바닥 배치·AT progress·역할별 object 토큰을 공유한다. 기본4명·1024환경·8토큰이며 N명에 최소2N 토큰이다. 선행은 상자를 AT 목표로 운반하고 후행은 그 상자를 SIT/CLIMB/ON_TOP 대상으로 사용한다.

## 역할별 object 토큰

`before_object_alias.py`는 distractor 슬롯을 후행 target 별도 토큰으로 대체한다. 두 토큰은 같은 물리 상자·pose·관측을 공유한다. encoder packet만 변경하며 사람 attention·보상·gate·RSI는 원래 packet/물리 ID를 쓴다.

## 거리와 초기화

| 설정 접미사 | 거리 | 사람 attention | 후행 ON_TOP 초기화 |
| --- | --- | --- | --- |
| `stage2_before_nonemask_rsi433_near50` | 원래/상대 거리 합 조건50/50 | NONE 차단·SELF bias0·BEFORE bias 학습 | loco/pickUp/carryWith40/30/30 |
| `stage2_before` | 기존/중간/짧음20/20/60 | SELF/NONE/BEFORE bias 학습 | loco |
| `stage2_before_rsi433` | 20/20/60 | SELF/NONE/BEFORE bias 학습 | 40/30/30 |

near50은 후행 사람→공유 상자 거리(ON_TOP은 후행 사람→payload 거리 추가)의 합이 선행 사람→공유 상자 거리보다 작도록 해당 모드에서 재추첨한다. mix226 범위는 YAML과 `before_sampling.py`를 따른다.

RSI433는 후행 ON_TOP만 적용한다. 크기·몸 관통·native 물리 유효성을 검사하고 carryWith는 HOLDING·바닥 위8cm 조건도 검사한다. 프레임이 없거나8회 배치 소진 시 해당 후행만 loco로 대체한다. 선행·단독·SIT/CLIMB 초기화는 loco다.

## 보상과 학습

현재 선행 AT 성공이 후행 state/success gate를 연다. progress는 gate와 독립적으로 선행 AT 목표를 향하며 실제 state/success 기하는 원래 대상 상자를 사용한다. 기존 가중치·CPA·AMP·부분 reset을 유지한다.

fixed Stage1에서 전이하며 actor encoder·입력 RMS는 동결, action head·사람 attention·critic은 학습한다. 기본 episode1200·학습률2e-5다. checkpoint 계약은 세 설정 혼용을 거부한다. 실행은 [config](../config.md), 추가 옵션은 [USAGE](../USAGE.md)를 따른다.
