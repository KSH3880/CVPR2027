# 다리 상자 선행조건(prerequisite) 실험: 문제 정의와 구현 방향

상태: 설계 논의 단계. 구현 없음.

## 1. 목적

센서 없이, 객체 token과 사람 간 attention만으로 에이전트가 다음 두 가지를 학습할 수 있는지 확인한다.
- 다리가 없으면 멈추기(출발 보류)
- 다리가 생기면 건너가기

## 2. 출발점: 현재 repo 측정 결과

현재 repo의 Stage 1(`output/stage1/8000.pth`) encoder를 동결한 상태에서, 다리를 edge 없는 객체 token으로 추가해 측정했다(probe, repo 밖 scratchpad).

| 측정 | H64 변화 |
|---|---|
| 다리 도착 전 ↔ 도착 후 | ~4e-6 |
| 다리 위치 +1m | ~1.4e-6 |
| edge 없는 과제 상자 +1m | ~9e-7 |
| 자기 과제 상자 +1m | 0.25 (장면 간 std 0.62) |

- Stage 1 head의 출력(mu) 변화는 ~7e-8로, 정책 σ 0.055에 비하면 무시할 수준이다. "도착 여부", "남은 시간" probe는 모두 chance 수준이다.
- 원인은 relation bias다. 사람 → 자기 상자는 +10.4, 사람 → edge 없는 객체는 −18.4이다. 그래서 edge 없는 객체에는 attention이 사실상 0이다.
- 결론: 동결 encoder, edge 없는 다리 token, 학습 가능한 head 조합으로는 다리 정보를 쓸 수 없다. 병목은 상자 크기가 아니라 edge 없는 객체에 대한 마스킹이다.

## 3. 새 시나리오와 가설

### 시나리오
- H1: gap이 있는 무대 A 위에 있다. 건너편 무대 B의 O1 위로 올라가야 한다(climb, H1–O1–G1).
- H2: 무대 아래 바닥에 있다. 큰 상자 O2를 gap 자리 G2로 밀어 다리를 만든다(push, H2–O2–G2).

### 가설: O2 → H2 → H1 전달 경로
1. Stage 1(동결)에서 O2는 H2의 payload다. 따라서 O2 정보가 H2 토큰에 들어간다.
   - action head는 사람 토큰만 읽는다. 그러니 push 동작에 필요한 O2 정보는 H2 최종 토큰에 반드시 있다.
   - 2절의 "자기 과제 상자 +1m → 0.25"가 이 첫 단계의 근거다.
2. Stage 2의 사람 간 attention에서 H1이 H2를 참조한다. 그러면 H1이 O2를 간접적으로 알게 된다.
3. H1에게 O2는 distractor다. 그래서 H1이 O2를 직접 보는 경로는 약하다고 본다.

## 4. 구현 방향

### Stage 1
- stage1_v2 push 구조와 checkpoint를 사용한다.
  - 큰 상자(1.2–1.7m)를 학습했다.
  - 배경 relation bias가 구조적으로 0이다(`TaskTypeEmbeddingBias`).
- encoder와 관측 RMS는 동결한다.

### Stage 2
base repo(`TokenHSI_stage1_stage2_base`)의 파이프라인을 따른다.
- `HumanCoordination`: 사람끼리만 self-attention하며 항등으로 초기화한다.
- 학습 대상: SA, critic, action/value head, discriminator
- 추락은 early termination으로 처리한다.

### relation 확장
- 지금 label은 0 none, 1 self, 2 coupled다. 여기에 **3 prerequisite**를 추가한다.
- 이유: coupled는 같은 task·payload·target을 공유할 때만 성립해서, climb와 push 조합은 none이 된다.

### prerequisite 정보 입력
5필드 packet `(valid, task, actor, payload, target)`에는 의존 관계 정보가 없다. 방법은 셋이다.
1. packet에 필드를 추가한다. 동결된 Stage 1이 packet을 해석하는 방식이 깨지므로 비권장이다.
2. **Stage 2 전용 입력을 준다**(사람 쌍별 의존 행렬). 권장안이다.
3. env 규칙으로 유추한다(push target이 다리 자리이면 prerequisite).

### 보완 고려
- **방향성**: H1 → H2는 requires다. H2 → H1은 none으로 둘지 별도 enables label로 둘지 정해야 한다.
- **내용 주입**: bias는 attention 경로만 바꾼다. label 자체를 내용으로 쓰려면 relation embedding을 V/K에 더하거나 relation 잔차를 둔다(0으로 초기화).
- **시나리오 혼합**: 사람 2명, 단일 시나리오에서는 prerequisite label이 학습 가능한 none-bias와 같아진다. label이 의미를 가지려면 다음을 섞는다.
  - prerequisite 에피소드
  - 독립 과제 에피소드
  - 다리가 이미 있는 에피소드
- **탐색**: H1과 H2의 시작 거리를 무작위로 준다. 그래야 "H1 도착 전에 push 완료"인 성공 에피소드 비율이 확보된다.

## 5. 예상되는 한계

- **좌표계 변환 불가**: `HumanCoordination`은 pose와 edge 정보를 쓰지 않는다. 따라서 H1은 H2 좌표계 정보를 자기 좌표계로 변환할 수 없다.
  - 잘 전달될 것: 좌표계와 무관한 진행도(O2–G2 거리, 완료 여부)
  - 잘 전달되지 않을 것: 다리의 상대 위치
- **멈추기의 형태**: gap 가장자리는 entity가 아니다. 그래서 "가장자리에서 멈추기"보다 **"출발 보류 → 신호가 오면 출발"**로 학습될 가능성이 크다.
- **크기 분포**: O2 높이와 무대 높이의 정합, push 상자 크기 분포 범위를 확인해야 한다.

## 6. 검증 계획

- **사전 측정**: push checkpoint에서 H2 토큰을 probe한다.
  - O2–G2 진행도와 완료 여부의 R² / 정확도
  - 비교용: edge 없는 큰 상자 +1m일 때 H1 토큰 변화(직접 경로)
- **학습 후 측정**
  - O2 위치만 바꾸는 반사실 개입 → H1 행동이 바뀌는지
  - prerequisite label ablation
  - 추락률, 성공률

## 7. 미결정 사항

- 구현 위치: 현재 repo에 stage1_v2를 이식할지, base repo 사본에 다리 장면을 얹을지
- push checkpoint 선택: cuboid(11500) / full plate(12000)
- prerequisite 정보 입력 방식, 방향성, 내용 주입 여부
- 무대 높이, gap 폭, O2 크기 분포
