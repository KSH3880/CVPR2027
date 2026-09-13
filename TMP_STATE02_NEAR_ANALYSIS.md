# state02_near 임시 분석 메모

작성: 2026-09-13. 토론용 문서이며, 변경 명세나 확정된 구현 계획이 아니다.
이 문서 작성으로 학습 코드·config·실행 중인 학습을 변경하지 않는다.

## 진행 순서와 제약

- 먼저 `dense relation reward`의 의미를 설명한다.
- 다음 대화에서 1번부터 하나씩 검토한다. 여러 변경을 한꺼번에 결정하지 않는다.
- Raw velocity progress는 사용자의 기존 방침대로 유지하는 것을 우선한다.
- At state를 예전 `N(0.5+0.5 I_put)`로 되돌리는 것은 확정된 제안이 아니다.
- 하나의 relation gate를 pinning·prerequisite 등에 공통 사용하고 싶다는 요구를 고려한다.
- 아래 대안 중 별도 pinning gate나 success 판정을 사용하는 것은 그 통일성과의 명시적인 절충이다.

## 0. 용어: dense relation reward

여기서 dense relation reward는 **매 simulation/control step마다 계산하는 Holding 및 At의 state·progress reward 합계**를 뜻한다. 이전 step의 reward를 더해 저장한 값이 아니다.

현재 state02_near에서:

\[
R_H=0.2\phi_H+0.2\bar P_H
\]

\[
R_A=g_H[0.2\phi_A+0.2\bar P_A]
\]

\[
R_{\mathrm{dense}}=R_H+R_A,
\qquad \bar P_e=(1-g_e)P_e+g_e.
\]

- state reward: 현재 상태 만족도에 대한 보상.
- progress reward: 목표 방향 velocity progress를 gate로 soft-pinning한 보상.
- dense relation reward: 위 네 항의 합. agent별·step별 값.
- success bonus: 최초 유효 성공 시 별도로 +5. dense 합계에는 포함하지 않는다.
- task/environment reward: dense + success bonus + power/collision/box-speed penalty.
- PPO 학습 reward: 현재 설정에서는 `0.5 × task/environment reward + 0.5 × AMP reward`.

네 dense 항은 각각 최대 0.2이므로 dense 합의 상한은 0.8이다. sigmoid가 유한한 값이므로 실제로는 정확한 0.8보다 조금 작다. `dense`라는 말 자체가 시간 누적이나 모든 지점에서의 수학적 연속성을 뜻하지는 않는다.

TensorBoard `relation/task_relation_total`은 success bonus까지 포함한다. `holding/state_reward`, `holding/progress_reward`, `at/state_reward`, `at/progress_reward`를 더하면 dense 합이다. `at/*_reward`에는 이미 prerequisite와 가중치가 포함돼 있다.

구현의 시간 인덱스: state에는 업데이트된 phi를, gate와 prerequisite에는 직전 step의 phi를 사용한다. 아래 가상 상태 계산은 상태가 유지되는 경우를 가정한다.

## 현재 수식과 TokenHSI 비교 기준

\[
\phi_H=\exp(-5\|\tfrac12(h_L+h_R)-O\|^2),
\qquad N=\exp(-10\|O-G\|^2).
\]

\[
\phi_A=N,\qquad g_e=\sigma(30(\phi_e-0.8)).
\]

\[
u_A=\mathbf v_{O,xy}\cdot\widehat{(G-O)_{xy}},
\qquad
P_A=\begin{cases}\exp[-5(1.5-u_A)^2],&u_A>0\\0,&u_A\le0.\end{cases}
\]

H progress는 human root의 XY 속도를 상자 방향으로 투영해서 같은 방식으로 계산한다.

| 항목 | 기존 TokenHSI, onlyVelReward=true | 기존 v0/state02 relation | state02_near |
|---|---|---|---|
| box-near Gaussian | N=exp(-10 XYZ 거리 제곱) | 동일한 N을 사용 | 동일한 N을 사용 |
| At에 해당하는 dense state | 0.2 N이 별도 가산 | lambda_s N(0.5+0.5 I_put), gH 곱함 | 0.2 N, gH 곱함 |
| transport velocity pinning | XY 거리 0.5m 미만에서 1 | gA로 soft-pinning | gA로 soft-pinning |
| transport prerequisite | handheld soft gate를 곱하지 않음 | gH | gH |
| geometric put 보상/성공 | 조건 만족 중 매 step 0.2 I_put | phiA>=0.9와 Holding 달성 이력으로 최초 +5 | N>=0.9와 Holding 달성 이력으로 최초 +5 |

TokenHSI carry velocity는 낮은 상자에 height mask도 적용하지만, 이후 XY 0.5m pinning으로 velocity가 다시 1이 될 수 있다. N에는 이 height mask를 적용하지 않는다.

**N 자체는 현재도 원래 TokenHSI와 같다. 문제는 N을 사용하는 주변의 gate·pinning·success 구조까지 동일하지 않다는 점이다.** N을 dense state로 사용할 수 없다는 결론은 아니다.

## 1. 목표 근처 감속과 pinning: XY progress / XYZ gate의 차이

### 확인한 사실

- P_A는 수평 목표 방향 속도를 본다.
- phiA와 gA는 XYZ 위치 오차를 본다.
- XY로 목표 근처에 도착해도, 높이가 남아 있으면 gA가 작다.
- 그 상태에서 감속하면 P_A가 감소하고, gA가 이를 충분히 보완하지 못할 수 있다.

예: gH≈1, dxy=0.2m, dz=0.5m이면 N≈0.055, gA≈0.

| 행동 | At reward |
|---|---:|
| 목표 방향 1.5m/s, P_A=1 | 약 0.211 |
| 정지, P_A=0 | 약 0.011 |

이는 같은 위치·Holding 상태에서 속도만 달리한 순간 reward 비교다. 미래 state 개선까지 포함한 정책 가치 비교는 아니다.

직전 분석에서 읽은 학습 sampled CSV의 `미완료, XY<0.5m, Z오차>0.2m` 구간은 평균 gH≈0.978, gA≈0.00008, 상자 속도≈0.96m/s였다. 목표 근처에서도 높이가 남고 pinning이 거의 없는 상황이 존재한다.

### 가설

사용자가 관찰한 목표 통과·재접근의 한 원인일 수 있다. 모든 통과 행동의 원인이라고 확정하지 않는다.

### 검토할 대안 — 미확정

Raw P_A는 유지하고, progress의 pinning만 XY 접근 완료 정도 b_A로 바꾼다.

\[
\bar P_A=(1-b_A)P_A+b_A.
\]

예를 들어 XY 0.8→0.5m 사이에서 b_A를 부드럽게 0→1로 만들고, 0.5m 안에서 1로 유지한다. 이 반경은 실험 시작안이지 검증된 최적값이 아니다.

phiA=N과 strict gA는 유지할 수 있다. 단, pinning과 relation satisfaction의 gate를 분리한다는 설계 비용이 있다. 단일 gA를 유지할 경우 어떤 절충이 가능한지 먼저 토론한다.

평가: 최초 XY 0.5m 진입 후 높이 오차 감소, 감속, 영역 재이탈, geometric put까지 걸리는 시간.

## 2. 목표 몇 cm 위와 실제 placement 사이의 보상 차이

손 중앙 오차 10cm, 정지, 같은 나머지 조건을 가정한 dense 합:

| 상태 | dense relation reward |
|---|---:|
| 목표 바로 위 5cm | 0.778 |
| 목표 바로 위 3cm | 0.781 |
| 목표 높이 오차 0 | 0.783 |

5cm 위에서도 N>=0.9라 현 success bonus를 이미 받을 수 있다. 정확히 놓는 행동의 추가 이득이 작다.

이것은 **N을 state로 쓰기 어렵다는 뜻이 아니라**, `정확한 placement까지 원할 때 N의 높은 값만으로 완료를 정의하면 목표가 느슨해진다`는 뜻이다. 원래 TokenHSI는 동일한 N 외에 별도 put reward를 더했다.

검토안: N을 dense state로 유지하고, success만 `Holding 달성 이력 AND placement 조건`으로 분리. 우선 기존 geometric put 조건으로 비교 가능하다. 필요할 때 속도·유지시간·지지 상태로 정의를 보강한다.

기존 geometric put은 XY<=10cm 및 목표 중심 높이 오차<=1mm인 기하학적 proxy다. 실제 지지 접촉이나 손을 떼었는지는 판정하지 않는다. 성공 보너스만 바꿔도 1번의 감속 문제까지 해결된다고 보장할 수 없다.

## 3. 놓은 뒤 손을 떼면 Holding과 At reward가 함께 감소

상자가 목표에 정확히 정지해 있을 때:

| 손 중앙–상자 거리 | dense relation reward |
|---|---:|
| 10cm | 0.783 |
| 20cm | 0.546 |
| 30cm | 0.132 |

현재 success 이후에도 dense reward를 계속 계산하고, At의 prerequisite는 현재 Holding gate다. 따라서 손을 떼면 Holding reward뿐 아니라 At reward도 감소한다.

먼저 원하는 행동이 `바닥에 놓기`인지 `놓고 손을 떼고 일어서기`인지 구분한다. 후자까지 원한다면 완료 뒤 prerequisite/보상 생명주기를 별도로 설계할 필요가 있다. 운반 중 prerequisite를 영구 달성 이력으로 바꾸는 것은 박스를 떨어뜨려도 At reward가 계속 열리는 부작용이 있어, 바로 적용하지 않는다.

## 4. 먼 목표 앞에서 들고 정지 / 큰 상자에서 정지

At state가 거의 0인 먼 위치, pH=0을 가정한 순간 dense 합:

| 손 중앙 오차 / 움직임 | dense 합 |
|---|---:|
| 10cm / 정지 | 0.388 |
| 10cm / 목표 방향 1m/s | 0.445 |
| 20cm / 목표 방향 1m/s | 0.328 |

운반하면서 Holding 오차가 커지면, 앞으로 이동해도 순간 reward가 정지보다 낮아질 수 있다. power 비용과 AMP, 미래 도달 가치까지 포함한 비교는 추가 측정이 필요하다.

현재 P_A는 목표 방향 속도 0.5m/s에서 약 0.00674, 1m/s에서 약 0.2865, 1.5m/s에서 1이다. 낮은 속도에서 시작하는 움직임의 추가 보상은 작다.

gamma=0.99, 제어 30Hz이므로 10초 뒤 보너스 5의 현재 할인값은 약 0.245다. 이는 보너스 단독 계산이며 미래 dense reward도 존재한다. `5가 작다/크다`를 단순 누적합만으로 판단하지 않는다.

큰 상자는 동일 밀도에서 질량·관성이 커지지만 target_speed는 동일하다. 크기별 실제 속도·gH 변동·힘 비용을 기록해야 이 가설을 확인할 수 있다.

**기존 추천 수정:** Holding state weight를 무조건 0.2→1로 올리면 초반 학습에는 도움이 될 수 있으나, 움직임 중 Holding 오차 비용도 커진다. 성숙한 near의 정지 문제 해결책으로 확정하지 않는다.

## 5. 우회·초기 역방향 이동·작은 상자 회전

P_A는 목표 방향 속도 성분을 보상한다. 반대 방향 이동은 보상하지 않지만, 같은 목표 방향 성분이면 옆방향 속도가 달라도 같은 P_A를 받을 수 있다. 속도 penalty는 상자 속도 2.5m/s 초과에서 발생하며 별도 감속 penalty는 없다.

따라서 넓은 회전으로 속도·Holding을 유지하는 전략은 가능한 가설이다. 원래 TokenHSI raw velocity 식에도 같은 특성이 있어, 이 사실만으로 이전 모델과의 차이를 설명하지 않는다.

샘플상 `Holding 강함, 미완료, 목표>1m, 상자 속도>0.3m/s` 구간에서도 raw P_A=0인 프레임이 약 55.6%였다. 방향 벡터와 회전·장애물 정보가 없어 원인은 확정할 수 없다.

Holding phi는 두 손의 중앙만 본다. 양손 접촉, 손의 상대 미끄러짐, 상자 회전 안정성을 보장하지 않는다. 상자 회전 자체를 무조건 벌주면 정상적인 운반 회전도 억제할 수 있다.

추가 진단: 초기 skill, box size/mass, 목표 방향 속도, 옆방향 속도, 몸과 목표 사이 각도, 몸 yaw rate, 상자 angular velocity, gH, per-agent power/AMP, 장애물 근접도.

## 관측·통계 해석의 한계

- 위 CSV 분석은 학습의 2개 sampled env에서 마지막 16,000행을 읽은 결과다. 전체 환경의 추정치나 독립 episode 성공률이 아니다.
- 뷰어 CSV와 학습 CSV는 별도 파일이다. 확률적 학습 rollout과 평가 동작도 구분해야 한다.
- TensorBoard at/gate는 전체 env·agent·수집 step의 평균이다. 최대 개별 gate가 아니다.
- at/satisfied, put은 현재 조건을 만족하는 프레임 비율이다. episode당 성공률과 다르다.
- near_0.5m/fraction 등의 구현은 `거리 조건 AND 미완료`의 전체 평균이다. 완료 샘플을 제외하므로 전체 목표 접근률처럼 비교하면 안 된다. 버전별 완료 정의도 달라 주의가 필요하다.
- 평균 AMP reward가 평균 task reward보다 크다는 사실만으로 AMP가 정책 학습을 지배한다고 확정할 수 없다. 행동별 차이와 advantage를 봐야 한다.
- 한 seed의 다른 reward 정의·학습량 비교는 원인을 확정하는 실험이 아니다. 특히 조기 정체를 최종 실패로 판단했던 기존 표현은 장기 결과에 따라 수정한다.

## 변경 실험 후보와 판단 기준 — 아직 미승인

1. 현재 near를 기준으로 1번 XY pinning만 변경: 목표 통과와 감속 문제 확인.
2. success만 geometric placement로 변경: 목표 위 정지와 실제 놓기 차이 확인.
3. 각각의 효과가 확인되면 결합.
4. 먼 거리 정지/우회는 크기·초기 skill별 진단 후 gate 또는 제어 쪽 변경 결정.

Raw progress 변경, Holding weight 증가, curriculum, AMP weight 변경, 방향/회전 penalty를 한꺼번에 넣지 않는다. 현재 목표는 무엇이 어떤 문제를 개선하는지 구분하는 것이다.

평가는 동일 checkpoint 단계/초기화 조건/상자 크기에서 진행하고, success 정의가 다른 run도 같은 geometric placement 기준을 함께 측정한다. reward config가 달라지면 현 checkpoint 검사에서 거절되므로, 추후 구현 시 명시적인 warm-start/새 실험 경로를 설계해야 한다.

## 코드 근거

- `tokenhsi/env/tasks/multi_agent/relation_reward.py`: phi, gate, P, dense reward, history/bonus.
- `tokenhsi/env/tasks/multi_agent/relation_task.py`: reward 구성 및 TensorBoard/CSV 진단 집계.
- `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py`: legacy TokenHSI reward, reset, 상자 물성, 관측.
- `tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry_relation_state02_near.yaml`: near 실험 설정.
- `tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation.yaml`: gamma, task/AMP 가중치.
- `tokenhsi/learning/multi_agent/ma_players.py`: evaluation 성공 및 geometric put 집계.

## 토론 진행 상태

- [x] 0. dense relation reward 용어 설명
- [ ] 1. 기존 TokenHSI / 기존 relation / near pinning 비교와 대안
- [ ] 2. N state와 placement success의 역할
- [ ] 3. 손을 떼는 동작까지 원하는지와 완료 후 처리
- [ ] 4. 먼 거리/큰 상자 정지
- [ ] 5. 우회와 작은 상자 불안정
- [ ] 변경안 확정 후에만 코드 구현

## 후속 토론: 현재 선호와 미확정 후보

이 절이 앞선 실험 후보보다 최신 논의를 반영한다. 학습 코드에는 아직 적용하지 않았다.

사용자가 선택한 방향:

- 성공 판정은 gA보다 phiA의 임계값으로 표현한다.
- At dense state는 우선 Z를 빼고 XY Gaussian으로 검토한다.
- Z는 성공 indicator 안에서 별도 허용 오차로 확인한다.
- 목표 근처에서 감속할 때 progress reward가 감소하는 문제는 별도로 해결해야 한다.

Holding 달성 이력은 현재 episode에서 phiH>=0.9를 한 번이라도 만족했는지 기억하는 boolean이다. 유효한 reference reset에서 이미 Holding을 만족하면 초기부터 true로 설정된다. 실제 접촉이나 정책이 스스로 pickup했음을 증명하는 값은 아니다.

이력의 목적은 Holding prerequisite를 성공 판정에 반영하면서, 내려놓는 순간 현재 gH가 떨어져도 성공을 인정하는 것이다. 다만 한 번 Holding을 만족한 후 놓치거나 밀어서 목표에 보내는 경로까지 막지는 못한다. 유지 여부는 아직 토론 중이다.

조건부 성공 후보:

\[
S_t=a_{H,t-1}\mathbf1[\phi_{A,t}\ge\tau_{\mathrm{success}}\land |z_O-z_G|\le\epsilon_z],
\quad R_{\mathrm{success},t}=5(1-D_{t-1})S_t.
\]

감속 보상 후보 — 미확정:

\[
\phi_A=e^{-k_{xy}d_{xy}^2},\quad
c_A(\phi_A)=\min(1,\phi_A/\tau_{\mathrm{brake}}),\quad
\bar P_A=P_A+c_A(\phi_A)(1-P_A).
\]

같은 phiA를 쓰되 감속 보완의 포화 임계값과 성공 임계값을 다르게 둔다. 원래 progress의 gA(1-P_A) 보완을 c_A(1-P_A)로 바꾸는 안이다. strict gA를 별도 prerequisite에 사용하는 것과 구분해야 하며, 이는 gate 하나로 모든 역할을 통일하는 설계와는 다르다.

예시 kxy=10, 감속 보완 반경 0.5m이면 tau_brake=exp(-2.5)≈0.082085. phiA>=tau_brake에서 barP_A=1이라 같은 위치·gH에서 감속에 따른 progress 손실이 정확히 0이다. 0.5m는 비교 실험 시작안이지 최적값이 아니다.

이 방식은 거리 경계에서 reward가 점프하지 않는 연속 함수지만, 포화점의 미분은 연속이 아니다. 보상항이라는 표현을 사용해도 기능적으로는 state 기반 pinning/포화이며 그 사실을 숨기지 않는다. 총 reward에는 상태 변화·Holding 변화·penalty도 영향을 주므로, 감속 중 전체 reward까지 반드시 일정하다는 보장은 없다.

성공 XY 허용 반경을 0.1m로 원하면 tau_success=exp(-10*0.1^2)≈0.904837. tau_success=0.9를 쓰면 허용 반경은 약 0.1026m다. 감속과 성공 임계값을 구분해 gate를 느슨하게 만들었다고 성공 범위까지 넓어지지 않도록 한다.

### 보완항 가중치에 대한 후속 검토 — 미확정

현재 실행 중인 near는 여전히 XYZ state와 기존 gate 설정을 사용한다. 아래는 XY-only 변경을 가정한 수식 분석이며 실제 정책의 거리별 속도를 측정한 표가 아니다.

사용자 제안인 보완항 증폭은 다음처럼 표현할 수 있다.

\[
Q_A=\min\{1,\ P_A+\alpha g_A(1-P_A)\},\qquad
R_A=g_H[0.2\phi_A+0.2Q_A].
\]

alpha=1은 기존 식과 같다. alpha>1에서 상한을 두지 않으면 정지가 정상 이동보다 더 높은 progress 점수를 받을 수 있다. 상한 1을 유지하면 최대 progress reward는 계속 0.2다.

alpha*gA>=1인 구간에서는 P_A에 관계없이 Q_A=1이다. alpha*gA<1에서는 감속 손실이 남는다. 따라서 기존 center=0.8, kxy=10에서 alpha=2만 적용하면 완전 보완이 시작되는 반경은 약 0.149m다. 0.2m에서 정지까지 완전 보완하려면 alpha≈50이 필요하고, 0.4m에서는 약 6200만이 필요하다. gate가 거의 0인 구간은 작은 가중치 증가로 해결되지 않는다.

철회한 후보 조합: kxy=10, betaA=30, centerA=0.2, alpha=2. 이때 phiA>=0.2 즉 dxy<=약 0.401m에서 Q_A=1이다. 계산 자체는 맞지만, 같은 gA를 relation satisfaction 및 신뢰할 수 있는 prerequisite로 공통 사용하려는 사용자 설계 의도와 충돌한다. 감속을 위해 state gate를 느슨하게 만드는 이 조합은 현재 추천안에서 제외한다.

가까운 영역에서 Q_A가 포화되는 것은 기능적으로 pinning이다. gate를 느슨하게 하면 gA 자체를 엄격한 placement prerequisite로 해석할 수 없다. XY-only state에서는 Z도 별도로 성공 판정에 필요하다.

보수적인 비교 기준 추천: Holding 만족 phiH>=0.9, 성공 phiA>=0.9(XY 약 10.26cm), Z 오차<=1mm, state/progress weight 각각 0.2, success bonus 5. 이 값들은 최적값이라는 뜻이 아니라 기존 실험과 비교를 유지하기 위한 기준이다. betaA/centerA/alpha의 변경은 별도 실험으로 검증한다.

성공을 늦출수록 반드시 유리한 것은 아니다. 현재는 성공 이후에도 dense reward가 지속되고 gamma=0.99이므로 성공 보너스를 먼저 받는 것이 할인 측면에서 유리하다. 다만 placement 과정에서 Holding 및 At dense reward가 크게 감소하면 지연 유인이 생길 수 있다. success bonus 증가만으로 감속/보상 전환 문제를 모두 해결한다고 단정하지 않는다.

이번 확인 당시 near는 약 22.8k이며 최근 200 iteration 평균 Holding gate≈0.848, At progress reward≈0.0889/0.2, goal XY error≈1.72m였다. 장기 학습이 개선되고 있으므로 초기 정체나 reward 평균 크기만으로 특정 가중치가 최적/실패라고 판정하지 않는다.

### 설계 제약 재확인 — gate의 의미 유지

- phi_e는 relation state satisfaction, g_e는 그 satisfaction에 기반한 soft activation이다. g_e를 정지 허용도로 재해석하여 prerequisite 의미를 바꾸지 않는다.
- gate center=0.2 제안은 철회했다. center=0.8, beta=30은 현재 기준으로 유지하며, 감속 문제는 미해결로 남긴다.
- 원래 식과 동일한 위치·상태·prerequisite에서는 progress 감소량이 lambda_p G_e (1-g_e) DeltaP다. strict g_e가 낮고 raw P가 감속 시 낮아진다면, 그 감소는 state gate를 그대로 둔 원래 식에서 자동으로 사라지지 않는다.
- 감속을 해결하려면 progress의 감속 처리 또는 보완 방식에 대한 별도 설계 선택이 필요하다. 그 변경은 공통 R_e=G_e(lambda_s phi_e+lambda_p Pbar_e) 구조 안에서 검토할 수 있지만, 현재 raw progress 유지 방침과의 절충을 먼저 명확히 한다. 코드 변경은 아직 없다.
- XY-only phi_A를 선택하면 g_A도 XY 관계만 대변한다. center=0.8을 유지해도 실제 placement(Z 포함)를 보장하지 않는다. 후속 relation이 요구하는 prerequisite가 XY 도달인지 placement 완료인지 구분해야 한다. 성공 조건을 prerequisite로 대체하는 것도 명시적 설계 변경이며 자동 적용하지 않는다.

### 다음 검토 후보: 목표 도착을 위한 속도 추종 — 아직 미확정

gate 의미를 유지하면서 적절한 감속에 보상을 주려면 progress가 요구하는 목표 속도를 거리에 따라 낮추는 방법이 있다. 이는 raw progress를 전혀 바꾸지 않겠다는 기존 제약을 일부 완화하는 제안이며, signed distance-progress로 변경하는 제안은 아니다.

\[
\delta=(G-O)_{xy},\quad d=\|\delta\|,\quad
\mathbf v^*(\delta)=v_{\max}\frac{\delta}{\max(r_{\mathrm{slow}},d)},\quad
P_A=\exp[-k_v\|\mathbf v_{O,xy}-\mathbf v^*(\delta)\|^2].
\]

기존 P_A는 목표 방향 속도 성분의 1.5m/s 오차를 사용한다. 이 후보는 XY 목표 속도 벡터와의 오차를 사용한다. 따라서 감속 스케줄뿐 아니라 옆방향 속도에도 오차가 생기고, 기존 `목표방향 속도<=0이면 P=0` mask도 그대로 유지하지 않는다. 목표 위치에서는 정지 시 P=1이 되어야 하기 때문이다. 이러한 변경 차이를 숨기지 않는다.

예시 vmax=1.5m/s, r_slow=1m, kv=5: 거리 1m/0.5m/0.2m/0.1m/0m에서 목표 속도는 1.5/0.75/0.3/0.15/0m/s. 목표 속도와 방향을 맞추면 각 거리에서 P_A=1이다. 원래 g_A와 Pbar=(1-g_A)P_A+g_A, relation 가중치는 그대로 둘 수 있다.

이 안은 아무 거리에서나 무조건 정지를 만점으로 만드는 방식이 아니다. 현재 거리에서 필요한 속도로 감속하면 만점을 유지하고, 목표에서는 정지가 만점이다. 실제 가감속 능력에 맞는 r_slow는 검증이 필요하다. 초기 검증은 At에 한정해 Holding 학습 영향과 분리한다.

XY state에서 Z를 빼면 중간 높이 오차에 대한 dense 유인은 빠지지만, 자세 흔들림이나 숙임이 반드시 사라진다는 보장은 없다. 위 progress 변경도 실제 운반·정착 성능 개선이 입증된 것은 아니다.

### 확정 및 구현: state02_near_dir — raw progress만 교체

사용자가 별도 TokenHSI-steer 실험에서 방향 cosine progress로 개선을 관찰했다고 보고했다. 그 실험은 기존 0.5m hard pinning도 포함하므로, 우리 soft gate만 사용하는 구조의 정지 성능까지 입증한 것은 아니다.

이번 새 config는 `amp_humanoid_ma_carry_relation_state02_near_dir.yaml`이다. Holding과 At 모두 raw progress만 다음으로 교체했다.

\[
P_e=\operatorname{clip}\left(
\frac{\mathbf v_{s,xy}\cdot\widehat{\mathbf d}_{st,xy}}
{\max(\|\mathbf v_{s,xy}\|,\epsilon)},0,1\right),\qquad
\bar P_e=(1-g_e)P_e+g_e.
\]

- 실제 위치 변화로 계산한 XY 이동 방향을 사용하며 몸의 heading은 사용하지 않는다.
- 정지는 raw P=0, 목표 XY 거리가 epsilon 이하이면 방향이 정의되지 않으므로 raw P=0이다. epsilon은 1e-6이며 극소 속력에서는 분모 clamp가 적용된다.
- 별도 0.5m 거리 pinning, 높이 mask, 목표 속도 스케줄, 추가 보상은 넣지 않았다.
- At state는 기존 XYZ `exp(-10 * ||goal-box||^2)`를 유지한다. XY-only 및 별도 Z 성공 조건 후보는 이번에 적용하지 않았다.
- state/progress 가중치 0.2/0.2, gate beta=30/center=0.8, prerequisite minimum, 성공 상태 임계값 0.9 및 Holding 달성 이력, 최초 보너스 5, dense 지속, 벌점과 AMP 설정 모두 유지한다.
- 기존 config에서 progress.kind를 생략하면 Gaussian이 그대로 실행된다. 새 config에서만 `kind: direction`을 지정한다. checkpoint의 config 일치 검사는 유지하여 기존 near checkpoint를 dir로 잘못 resume/test하는 것을 막는다.
- 최대 보상 범위는 같지만 느린 전진의 보상이 커지므로 평균 reward 자체를 기존 실험과 성능으로 직접 비교하지 않는다. 성공률, 목표 오차, 도달 시간도 함께 확인한다.

학습: `bash tokenhsi/scripts/multi_agent/ma_carry_relation_state02_near_dir_train.sh`

출력: `output/ma_carry_relation_state02_near_dir/CarryRelationState02NearDir_*/`

뷰어: `HEADLESS=0 bash tokenhsi/scripts/multi_agent/ma_carry_relation_state02_near_dir_test.sh <새 dir checkpoint.pth> 2 1 3 10`

### 추가 비교 실험: dir_success10

- 새 config: `tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry_relation_state02_near_dir_success10.yaml`
- 기존 dir config와 비교하여 `subgoal_success_bonus: 5.0 -> 10.0`만 변경했다.
- XYZ state, raw direction progress, soft pinning, gate, prerequisite, state/progress 0.2/0.2, 성공 조건은 그대로다. 기존 실험 및 실행 중인 학습은 변경하지 않았다.
- 학습: `bash tokenhsi/scripts/multi_agent/ma_carry_relation_state02_near_dir_success10_train.sh`
- 출력: `output/ma_carry_relation_state02_near_dir_success10/CarryRelationState02NearDirSuccess10_*/`
- 뷰어는 대응하는 `ma_carry_relation_state02_near_dir_success10_test.sh`를 사용한다. 기존 bonus=5 checkpoint와는 reward config가 다르므로 그대로 resume/test하지 않는다.
- config의 유일한 차이, 최초 보너스 10/재지급 0, 성공 후 dense 지속, checkpoint config 검사를 포함한 전체 테스트 66개 통과. 실제 학습은 실행하지 않았다.

이 실험은 정지/정착 중 dense 보상 감소를 미래 성공 보너스로 보완할 수 있는지 확인하는 것이며, 보상 감소 자체를 제거하는 수식 변경은 아니다. 성공은 여전히 Holding 달성 이력 + XYZ phi_At>=0.9이며 실제 지지, 손 놓기, 1mm PutDown 조건과 동일하지 않다. 높은 목표에서의 통과 성공 증가와 낮은 목표에서의 정착 개선을 구분해야 한다.

전체 설계의 미해결 쟁점: Z는 최종 배치 및 후속 OnTop prerequisite에는 필요한 정보이지만, 접근 중 보상과 정지 시 progress 보완까지 같은 완료 gate에 결합되어 있다. 따라서 Z 전체를 노이즈로 단정하거나 XY-only로 바꾸면 해결된다고 단정하지 않는다. 이후 검토에서는 접근 유도, 정착, 최종 상태 판정, 후속 edge 활성화를 함께 보되, 이번 success10 비교에는 추가 변경을 섞지 않는다.

### 구현: At-only approach progress, r=0.5 m

직전 `state02_near_dir_success10`을 대조군으로 삼아, At의 progress 보완만 변경한 별도 실험을 추가했다. 성공 보너스는 10으로 유지한다.

\[
a_A=\frac{1}{1+(d_{xy}/0.5)^2},\qquad
\bar P_A=a_A+(1-a_A)P_A,\qquad
R_A=g_H[0.2\phi_A+0.2\bar P_A].
\]

- `a_A`는 post-step 박스-목표 XY 거리로 계산한다. 0.5m에서 0.5, 0.1m에서 약 0.9615, XY 일치 시 1이다.
- 기존 At self-pinning의 `g_A`를 `a_A`로 교체한다. `g_A`를 다시 더하거나 곱해 이중 보완하지 않는다.
- Holding은 기존 `Pbar_H=(1-g_H)P_H+g_H`를 유지한다. 두 raw progress는 모두 XY 방향 cosine이다.
- At 전체 prerequisite에는 이전 상태에서 계산한 `g_H`를 그대로 곱한다. XYZ state, gate 관측, 달성 이력, 성공 판정, state/progress 0.2/0.2, 벌점 및 AMP 설정은 변경하지 않는다.
- 접근 달성도는 성공 판정 및 prerequisite에 쓰지 않는다. 목표 위에 들고 정지해 At progress가 1이어도 XYZ state가 부족하면 성공하지 않는다.
- 구현 옵션: `relationReward.progress.at_approach_radius: 0.5`. 이 옵션이 없는 모든 기존 config는 기존 gate pinning으로 실행된다.
- 새 config: `tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry_relation_state02_near_dir_success10_approach.yaml`
- 학습: `bash tokenhsi/scripts/multi_agent/ma_carry_relation_state02_near_dir_success10_approach_train.sh`
- 출력: `output/ma_carry_relation_state02_near_dir_success10_approach/`
- 뷰어: `HEADLESS=0 bash tokenhsi/scripts/multi_agent/ma_carry_relation_state02_near_dir_success10_approach_test.sh <새 approach checkpoint.pth> 2 1 3 10`
- TensorBoard: `relation/at/approach`가 a_A, `relation/at/progress_bar`가 가중치 전 Pbar_A, `relation/progress_at`이 raw dir P_A다. 기존 `relation/at/gate`는 XYZ 완료 gate로 유지한다. `relation/at/progress_reward`에는 0.2 및 prerequisite가 이미 적용돼 있다.
- 새 checkpoint와 기존 reward config의 혼용은 계속 차단한다. 새 실험은 새 학습으로 시작한다.

검증: 전체 CPU 테스트 80개 통과. 거리별 수치, 보상 범위/단조성, At-only 변경, Holding/상태/성공/관측 불변성, 실제 mixin의 거리 전달, 높은 기존 g_A를 추가 적용하지 않음, config/checkpoint 계약을 확인했다. 실제 simulator 학습은 실행하지 않았다.

남은 검증 사항: 가까이 들고 머무르는 행동, 먼 거리의 완만한 접근 보상, 정착 중 Holding 손실. 현재 식은 감속 자체를 선호하거나 물리적 배치/손 놓기를 보장하지 않는다. 성공 10 대조군과 비교하여 목표 접근 및 낮은 목표 정착이 실제로 개선되는지 확인한다.
