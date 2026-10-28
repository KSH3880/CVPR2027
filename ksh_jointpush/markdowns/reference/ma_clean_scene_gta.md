# Stage1 clean-scene GTA 정책

현재 세 실험은 같은 모델을 사용한다. 실행 설정은 [config.md](../config.md), 구현 위치는 [structure.md](../structure.md)를 따른다.

## 관측과 정규화

하나의 환경에 사람 M명·물체 O개·목표 M개가 있다. 노드는 canonical `[H | O | G]` 순서이며 각 사람223-D, 물체30-D, 목표1-D다. 노드 뒤에 각 token의 환경 로컬 위치3+xyzw quaternion4, 사람마다 task packet5를 붙인다. 2명·4물체의 관측은634-D다.

사람과 물체의 intrinsic feature에만 타입별 RMS를 적용한다. 목표 상수·pose7·task packet은 원값을 통과시킨다. AMP 정규화도 family one-hot은 그대로 둔다.

## 모델

- 타입별 tokenizer `[256,128,64]`, 타입 embedding, GTA Transformer4층·2head·64-D를 사용한다. Actor와 critic은 독립이다.
- task embedding은 SIT/CLIMB/AT/ON_TOP 네 종류·64-D다. Layer/head마다 공유 projection64를 하나씩 사용해 방향성 task edge bias를 만든다. H/O/G 물리 타입·역할에 따라 projection을 나누지 않는다. NONE/SELF bias는0이다.
- Carry는 actor→payload, actor→target, payload→target 연결에 같은 task bias를 적용한다. SIT/CLIMB는 actor→target이다.
- GTA는 token pose에서 `g=T^-1`, `g^-1=T`를 만들고 head feature를4-D block으로 나눠 Q에 `g^T`, K/V에 `g^-1`를 적용한다. Softmax attention의 출력을 `g`로 각 token 좌표계에 되돌린다.
- Head concat 뒤 사람 token에 `z += MLP([z,e_task])`를 적용한다. MLP는128→64→64 ReLU이며4층이 공유하고, actor/critic은 독립이다. 마지막 선형층은0으로 초기화한다.
- Task/token/edge 순열은 explicit endpoint를 재매핑하고, 마지막에는 사람 token을 canonical 순서로 복원해 action/value head로 전달한다.

## 학습과 checkpoint

PPO 관측은 장면 단위 `(T,N,scene)`로 저장한다. Action·reward·AMP는 사람별로 유지하고 loss에서만 `(N*T*M)`으로 편다. AMP demo/replay는 셔플된 PPO 행의 실제 family와 일치시킨다.

모델의 parameter 이름·shape는 정리 전 세 실험의 checkpoint와 같다. Entity 수를 바꾸면 비학습 buffer와 packet 폭을 다시 만들며 모델 weight는 유지한다. 다만 reward·크기·샘플링·물리 설정의 호환은 별도 checkpoint 검사와 wrapper 규칙을 따른다.

수식·정규화·출력/gradient 순열은 `test_stage1_gta.py`, `test_stage1_policy.py`에서 검사한다. 이번 정리에서는 변경 전 세 모델의 state_dict strict load와 동일 입력의 actor/critic 출력·gradient도 별도로 비교했다.

## 환경·checkpoint 계약

SIT/CLIMB/AT/ON_TOP task prior는 10/25/32.5/32.5%다. Carry 크기는 XYZ 0.2~0.6m, Sit은 XY 0.25~0.8·Z 0.35~0.5m다. 크기별 과제 적격·RSI 물리 screening·fallback·AMP family 연결을 유지한다. 자세를 수직 retarget하지 않는다. 명목 RSI 비율과 유효 프레임/fallback 이후 실제 분포는 구분한다.

현재 기본 접근 반경은 SIT U(1,5m), 나머지 U(1,10m)다. AT 목표와 ON_TOP 받침 XY는 환경 원점 기준 각 축 U(-4.5,4.5m)이며 사람·payload와 최소 1m를 유지한다. 다중 장면의 간섭 재추첨 때문에 원본 1인 태스크와 최종 수락 분포가 완전히 같지는 않다. 운반 RSI는 loco/pickUp/carryWith/putDown 50/10/30/10이다.

AT/ON_TOP loco 출발 payload와 AT loco/pickUp/carryWith 목적지에 선반 높이를 추첨한다. 50%·U(0,1m) 바닥 높이, payload 상판 1.2m clamp, 출발 5cm 여유 규칙을 사용한다. putDown AT 바닥 목표, ON_TOP 받침, SIT/CLIMB 배치는 각 과제 규칙을 따른다.

- `stage1`과 v2는 크기·성공 조건이 달라 checkpoint를 혼용하지 않는다.
- `stage1_v2` checkpoint는 fixed에서 명시적으로 평가·resume할 수 있다. 물리 환경을 변경하는 후속 실험이다.
- checkpoint env snapshot·명시적 override 규칙은 [USAGE](../USAGE.md)의 checkpoint 절을 따른다.
- 구 `stage1`의 1~2m·40/10/40/10 snapshot도 지원한다. 새 기본 설정으로의 암묵적 resume는 거부한다.
- checkpoint의 모델 parameter 이름·shape·관측·reward metadata 계약은 유지한다. 제거된 다른 multi-agent 실험의 실행과 checkpoint 호환은 지원하지 않는다.
- 저장 YAML의 `hard_skill_training`은 세 실험의 checkpoint 계약에 남아 있다. near-start 확률과 legacy Climb late 비율은 모두 0으로 검증하며, 비활성 curriculum 실행 코드는 제거했다. Size RSI 자체의 late-frame 선택은 유지한다. 비활성 경로의 불필요한 난수 소비도 제거되어 같은 seed의 새 scratch run이 과거와 동일한 궤적을 재현한다는 보장은 없다.
