## 결론

보상 공백은 원본과 현재 구현에 공통입니다. 성공/실패를 가르는 가장 강한 차이는 멀티에이전트 코드의 env 좌표계 불일치가 object 정규화 통계를 오염시킨 것입니다.

그 결과 현재 정책은 박스의 x/y 위치와 크기 변화를 원본보다 약 100배 둔감하게 봅니다. 반면 z축 정보는 정상입니다. 따라서:

> 박스까지 이동 → 박스 높이에 맞춰 쭈그리기 → 정확한 수평 손 접촉 실패 → 정지
> 

라는 현재 증상과 정확히 일치합니다.

## 핵심 버그

현재 학습은 2048 env / 2 agents / 3 boxes입니다.

- 사람과 담당 박스의 reference reset에는 env origin이 더해지지 않습니다.
    - tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py:819
    - tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py:1030
- 하지만 미할당 distractor 박스는 _env_origins 주위에 배치됩니다.
    - tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py:962
- 이후 담당 박스와 distractor 박스 모두 같은 RunningMeanStd(39)에 들어갑니다.
    - tokenhsi/learning/multi_agent/ma_agent.py:24

즉 env 0 이외에서는 담당 개체들은 원점 근처, distractor만 수십~수백 m 떨어진 env origin 근처에 놓입니다. 이 비정상적인 distractor 좌표가 모든 object token의 정규화 통계를 오염시킵니다.

체크포인트에서 실제로 확인된 값입니다.

object local position    원본 Carry    현재 MA Carry
━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━━━━
x 표준편차                   1.31 m         155.08 m
───────────────────────  ────────────  ───────────────
y 표준편차                   0.86 m         139.94 m
───────────────────────  ────────────  ───────────────
z 표준편차                  0.371 m          0.365 m
───────────────────────  ────────────  ───────────────
x 평균                       0.65 m         -18.64 m
───────────────────────  ────────────  ───────────────
y 평균                       0.04 m         -12.50 m

현재 humanoid의 arena x/y도 평균 약 (-44, -45), 표준편차 약 (26, 26)입니다. 원래 arena-local feature라면 대략 한 자릿수 이내여야 하므로, 사람 reset에도 env origin이 빠졌다는 별도 증거입니다.

0.1 m의 박스 이동은 정규화 후:

- 원본 x: 약 0.076
- 현재 x: 약 0.00065

밖에 되지 않습니다. 체크포인트의 첫 object tokenizer도 이 축의 weight를 크게 증폭하지 않았습니다. 첫 레이어 민감도 proxy는 다음과 같습니다.

- box/bbox x·y: 원본의 약 0.6%
- z: 원본의 약 100%

z가 정상이라 쭈그리는 높이 제어는 되고, x/y 박스 중심·모서리 정밀도가 사라져 grasp contact를 만들지 못하는 것으로 해석됩니다. 현재 관찰과 매우 강하게 부합합니다.

## 원본과 동일해서 원인이 아닌 것

- 보상 함수는 수치적으로 거의 동일합니다.
    - 원본: tokenhsi/env/tasks/basic_interaction_skills/humanoid_carry.py:681
    - 현재: tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py:653
- 명시적인 loco→pickup switch는 양쪽 모두 없습니다.
- handheld, lift 전 0.2 m 공백, power penalty도 같습니다.
- skillInitProb=[0, 0.5, 0.1, 0.3, 0.1]과 dataset RSI 구간도 같습니다.
- 현재는 2048×2=4096 agent rows, 원본은 4096×1=4096이므로 epoch당 pickup 초기화의 기대 개수도 같습니다.
- box 크기·density, humanoid asset, PD 제어도 같습니다.

따라서 보상 공백은 “공통으로 존재하는 어려움”이지 차이를 발생시킨 직접 원인은 아닙니다. 현재판에서는 손-박스 정밀 관측이 약해져 그 공백을 넘어갈 탐색이 훨씬 어려워진 것입니다.

## 추가로 큰 차이

현재 --test는 모든 박스를 reference 상태와 무관하게 arena에 다시 배치합니다.

- 설정: tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml:87
- 실행 조건: tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py:88
- 실제 reset: tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py:900

원본 carry_test.sh에는 이 test 전용 reset이 없습니다. 따라서 현재 viewer는 학습 때의 pickup/carry reference-aligned box 상태보다 훨씬 어려운 분포를 평가합니다.
1-env에서는 물리 좌표 불일치는 피하지만, 오염된 normalization과 train/test 분포 차이는 그대로 남습니다.

정책 구조도 동일하지 않습니다.

- 원본: 265D 전체 관측을 [1024, 512] MLP에 직접 입력
- 현재: humanoid/object/goal을 각각 64D로 압축한 뒤 relation Transformer 사용

“1인 관측이 원본과 동일”하다는 설명은 223D humanoid 부분에만 해당합니다. 전체 정책 입력과 네트워크는 동일하지 않습니다.

또 현재 체크포인트는 확인 시점에 epoch 13,500, 약 17.69억 frames였고, 원본 ckpt_carry.pth는 epoch 100,000, 약 131.07억 frames입니다. 현재 보상 곡선도 아직 상승
중입니다. 다만 좌표/정규화 문제가 구조적이므로 단순히 더 오래 학습하는 것만으로 해결될 가능성은 낮습니다.

## 판단

우선순위는 다음과 같습니다.

1. env origin 불일치 + distractor 공유 정규화 오염 — 가장 유력
2. 현재 test의 reference 비정렬 box reset — 평가 난이도 증가
3. 새 Transformer의 64D entity 병목과 처음부터 M=2 학습
4. 학습량 차이
5. 보상 공백·pickup 10%·power — 공통 요소이므로 비교 원인으로는 낮음

확정적인 causal proof에는 M=2/O=2처럼 distractor가 없는 짧은 신규 학습과, 좌표계를 맞춘 M=2/O=3 학습의 비교가 필요합니다. 그 전에 reward나 pickup 확률부터 바꾸
면 핵심 관측 버그를 가릴 가능성이 큽니다.

이번 검토는 코드·체크포인트·TensorBoard·실행 명령을 읽기 전용으로 확인했습니다. 파일을 수정하거나 현재 학습/viewer 프로세스를 중단하지 않았습니다.

---

결론부터 말하면:

1. env-origin 불일치는 학습에서 실제로 발생하며, M=2, O=3, envs=2048에서는 pickup 학습을 심각하게 훼손합니다.
2. reference 비정렬 test reset은 AMP 문제가 아닙니다. 테스트 초기상태의 OOD 문제이며, 주원인보다는 평가 교란 요인입니다.
3. 64D Transformer와 처음부터 M=2 학습은 난도를 높이지만, 현재 실패의 치명적인 원인으로 보이지는 않습니다.

## 1. env-origin 불일치는 학습에서도 발생하는가?

네. 오히려 학습에서 만들어진 문제가 1-env 테스트 체크포인트에 그대로 남습니다.

학습 reset 흐름은 다음처럼 서로 다른 좌표계를 섞습니다.

- reference humanoid: reference root + agent offset, env origin은 더하지 않음
tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py:819
- 담당 box: reference box + agent offset, 역시 env origin 없음
tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py:1014
- 남는 distractor box: env origin을 중심으로 배치
tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py:962
- humanoid arena 좌표: root - env_origin으로 계산
tokenhsi/env/tasks/multi_agent/humanoid_ma.py:233

따라서 현재 M=2, O=3에서는 object 세 개 중:

- 담당 box 두 개는 원점 근처
- distractor 한 개는 각 env grid origin 근처

에 놓입니다.

더 큰 문제는 세 object slot을 하나의 RunningMeanStd에 합쳐 정규화한다는 점입니다. tokenhsi/learning/multi_agent/ma_agent.py:24

앞서 체크포인트에서 확인한 object 위치 표준편차가:

- x: 약 155 m
- y: 약 140 m
- z: 약 0.365 m

였습니다. 이 상태에서는 pickup에 필요한 10 cm의 x/y 변화가 정규화 후 약 0.0006~0.0007밖에 되지 않습니다. 반면 z 변화는 정상적으로 크게 보입니다.

그래서 현재 행동과 정확히 대응됩니다.

- goal이나 coarse 위치 정보로 box까지 걸어감
- z 정보는 살아 있어서 box 높이까지 쭈그림
- 손과 box 사이의 정밀한 x/y 오차는 거의 보이지 않아서 grasp를 완성하지 못함

num_envs=1 테스트는 실시간 env-origin 불일치는 대부분 피합니다. 하지만 테스트 player가 같은 EntityRunningMeanStd를 만들고, 체크포인트의 오염된 통계를 다시 로드
합니다. tokenhsi/learning/multi_agent/ma_players.py:26, tokenhsi/learning/amp_players.py:55

즉:

> 1-env 테스트는 물리 좌표 문제는 피하지만, 2048-env 학습에서 망가진 observation scaling은 피하지 못합니다.
> 

다만 전체 학습이 완전히 무효인 것은 아닙니다. 담당 box와 agent 자체는 같은 잘못된 좌표계에 있어서 coarse navigation은 학습할 수 있습니다. pickup처럼 정밀한
object-relative 제어가 특히 치명적으로 손상됩니다.

## 2. test의 reference 비정렬 box reset은 왜 중요한가?

사용자 판단이 맞습니다. AMP discriminator 때문에 중요한 것은 아닙니다.

Reference motion에는 두 역할이 있습니다.

- 학습에서 AMP demo 제공
- 학습과 테스트에서 RSI 초기 자세·속도 생성

현재 설정은 stateInit: Random이므로 테스트에서도 humanoid는 loco/pickUp/carryWith/putDown reference 중 하나의 중간 자세로 초기화됩니다. tokenhsi/data/cfg/
multi_agent/amp_humanoid_ma_carry.yaml:30

그런데 --test에서는 testRandomArenaSpawn=True가 활성화되고, _reset_boxes()가 reference box state를 사용하지 않고 모든 box를 독립적으로 랜덤 배치한 뒤 바로 반환
합니다. tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py:88

예를 들면 다음 초기상태가 생길 수 있습니다.

- humanoid: box를 들고 있는 carryWith 중간 자세
- 실제 box: 손과 무관한 먼 위치

원본은 pickup/carry/putdown RSI일 때 같은 motion ID와 time의 reference box 위치를 사용합니다.
`tokenhsi/env/tasks/basic_interaction_skills/humanoid_carry.py:723`

따라서 이 문제의 의미는:

- AMP 점수 문제가 아님
- 초기 몇 프레임의 물리 상태가 학습 분포와 다름
- 원본 viewer와 현재 viewer의 비교 조건이 동일하지 않음

입니다.

하지만 이것을 pickup 실패의 핵심 원인으로 놓는 것은 과합니다. agent가 이미 정상적으로 box까지 걸어간 뒤 쭈그리고 멈춘다면 reset 직후의 reference 비정렬 영향은
대부분 지나간 상태입니다.

따라서 이전 분석에서 이 항목을 세 핵심 원인 중 하나로 같은 비중으로 놓았다면 수정해야 합니다. 이는 주로 테스트 공정성과 초기 안정성 문제이고, 반복되는 grasp 실
패의 주원인은 아닙니다.

## 3. 64D Transformer와 처음부터 M=2 학습은 치명적인가?

현재 증거로는 치명적이지 않습니다.

네트워크는 각 entity를 독립적으로 tokenize한 뒤 Transformer에서 결합합니다.

- humanoid: 230D → 256 → 128 → 64D
- object: 39D → 256 → 128 → 64D
- goal: 6D → 256 → 128 → 64D
- M=2, O=3: 총 7개 token
- 4 Transformer layers, 2 heads
tokenhsi/learning/multi_agent/amp_network_builder_ma.py:139, tokenhsi/data/cfg/train/rlg/amp_ma_carry.yaml:14

64D가 실제 압축인 부분은 주로 230D humanoid token입니다. Object는 39D에서 64D로 확장되므로, box geometry 자체가 차원 수 때문에 직접 잘리는 구조는 아닙니다.

M=2 학습도 sample 수 관점에서는 크게 불리하지 않습니다.

- 현재: 2048 env × 2 agents = 4096 agent rows/step
- 원본: 4096 env × 1 agent = 4096 agent rows/step

따라서 pickup 초기화 10%라면 agent-level pickup 샘플 수는 대략 동일합니다. 다만 두 agent rollout이 같은 env 안에서 상관되어 있고, 한 agent의 reset/충돌이 다른
agent에 영향을 주며, teammate와 distractor에 대한 attention까지 배워야 하므로 학습 난도와 sample efficiency는 나빠질 수 있습니다.

특히 grasp는 humanoid–hand–box의 정밀한 cross-token 상관관계가 필요하므로 원본의 큰 dense MLP보다 늦게 배울 가능성은 있습니다. 하지만 다음 사실 때문에 현재의 절
대적인 pickup 실패를 64D/M=2 탓으로 보기는 어렵습니다.

- teammate ownership과 navigation은 이미 학습됨
- 모델 크기는 원본보다 작지 않음
- 실패가 x/y만 극단적으로 약하고 z는 정상인 축별 패턴을 보임
- 그 패턴이 체크포인트의 정규화 통계와 정확히 일치함

즉 M=2 설계를 포기할 이유는 없습니다. 현재 문제는 M=2 자체가 아니라:

> origin 없는 RSI agent/assigned box + origin 기반 distractor + 모든 object slot을 공유하는 RMS
> 

의 조합입니다. 특히 O=3 > M=2이기 때문에 distractor 한 개가 모든 담당 box의 정규화까지 오염시킵니다.

최종적으로 원본과 현재 모델의 차이는 이렇게 정리할 수 있습니다.

- 보상 공백: 양쪽 모두 존재
- 원본: 정밀 box-relative observation이 살아 있어서 공백을 넘어 lift를 학습할 수 있음
- 현재: 정밀 x/y 정보가 정규화 과정에서 거의 사라져, 보상 공백 앞의 “접근 후 쭈그린 상태”가 훨씬 강한 local optimum이 됨
- 64D/M=2: 이 상황을 더 어렵게 만들 수 있지만 근본 원인으로 확인되지는 않음
- test reference 비정렬: 부차적인 초기상태·비교 공정성 문제