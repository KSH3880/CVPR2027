# Stage2 BEFORE PUSH

환경은 [stage2_before_push.yaml](../../tokenhsi/data/cfg/multi_agent/stage2_before_push.yaml), 학습은 [동명 YAML](../../tokenhsi/data/cfg/train/rlg/stage2_before_push.yaml)이다. 실행 명령은 [config](../config.md), 전이·resume는 [USAGE](../USAGE.md)를 따른다. 기존 `stage2_before`는 유지한다.

## 행동과 BEFORE 쌍

기본 4명·1024환경·8개 관측 물체 토큰·1200step이다. 기존 BEFORE처럼 남은 사람을 순서대로 골라 0.8 확률로 둘씩 묶는다. 4명일 때 쌍 수의 이론적 확률은 0쌍 0.8%, 1쌍 35.2%, 2쌍 64%다.

쌍에 속하지 않은 사람만 현재 `stage1_push`의 행동 비율을 사용한다.

| 독립 행동 | 확률 |
| --- | --- |
| AT | 25% |
| ON_TOP | 20% |
| SIT | 10% |
| CLIMB | 15% |
| PUSH | 30% |

이 30%는 전체 사람 중 PUSH 비율이 아니다. 기본 그룹 확률에서 독립 역할은 사람의 평균 18.4%이므로 전체 PUSH 기대 비율은 약 5.52%다. 로그 `sampling/solo_*_fraction`은 실제 독립 역할만 분모로 사용한다. 선행 AT는 분모에서 제외한다.

BEFORE 쌍은 선행 AT → 후행 SIT/CLIMB/ON_TOP이다. 후행 가족은 선행 AT 상자 크기를 수행할 수 있는 가족 사이에서 균등하게 고른다. 세 가족 모두 가능하면 각각 1/3이며, 크기 조건 때문에 전체 후행 집계가 항상 1/3인 것은 아니다. PUSH는 선행·후행에 들어가지 않는다. 기존 공유 물체 alias·현재 AT 성공 gate·AT 목표 기준 후행 progress를 유지한다. 성공 이력으로 gate를 고정하지 않는다.

## 크기와 배치

기존 BEFORE의 AT/SIT/CLIMB/받침 크기 bank, ground 초기화, original/medium/short 20/20/60 거리 모드를 유지한다. 독립 PUSH는 Stage1 PUSH와 같은 167종 큰 상자, 밀도3, 마찰0.4를 사용한다. 출발 상자는 사람에서2–10m, 목표는 상자에서2–10m, 방향은 상자 local +Y 기준 ±45도다. 목표와 본인 거리는1m 이상, rear 여유1.5m, PUSH 경로 사이3.5m, 다른 사람·활성 물체와 여유0.5m를 검사한다.

사람마다 기존 네 종류와 PUSH 한 종류, 총 5N개 물리 후보 상자를 미리 만든다. 토큰 수를4N보다 크게 지정하면 추가 일반 distractor 후보도 만든다. 기본 후보20개 중 관측은8토큰이며 BEFORE 공유 alias당 활성 물리 물체가 한 개 줄어든다. 사용하지 않는 후보는 멀리 숨긴다. 큰 PUSH 후보는 PUSH 역할에만 배정하며 일반 distractor나 BEFORE 받침으로 뽑지 않는다. distractor는 기존 네 bank에서 뽑는다.

시작 자세는 모두 loco다. PUSH 사람은 PUSH 전문가의 `push_loco` 모션, 나머지는 기존 loco를 쓴다. PUSH 접촉 RSI bank는 생성·사용하지 않는다. 네 AMP family(carry/sit/climb/push)를 사용한다.

## 정책·보상·평가

`stage2_coupled_push`와 동일한 `Stage2PushBuilder`를 사용한다. 관측은 task packet5N+그룹 ID N이며 사람 relation은 NONE/SELF/BEFORE/COUPLED 네 종류다. BEFORE 공유 물체 alias와 directed 의존 관계는 유지한다. Stage1 PUSH의 5개 task embedding과 4family AMP discriminator를 그대로 전이한다. `PUSH`도 동일 task bias·task embedding/action MLP·action head 경로를 사용한다. 기본 전이는21k Stage1 PUSH 가중치이며 기존 tensor를 그대로 복사하고 새 사람 attention을 identity로 초기화한다. 공통 Stage2처럼 actor encoder와 입력 정규화를 동결하고 coordination·action head 등은 학습한다.

PUSH는 Stage1 PUSH의 CONTACT+AT 보상이다. CONTACT의 state/progress/success 가중치는 각각0.2, 나머지 행동·BEFORE gate는 기존 설정을 유지한다. AT 도달 시 같은 사람 CONTACT도 포화한다. 접촉은 양손 뒤쪽 기하·손 힘·상자 힘 조건을 사용하며 정식 평가 성공과 구분한다.

일반 평가는 학습 인원·샘플링·물체·배치를 유지하고 loco·1200step·512환경×3seed다. AT/PUSH 상자 중심과 고정 목표의 3D 거리≤0.20m를 점/영역 두 리포트에 같은 값으로 반영한다. 다른 행동의 점/영역 기준, 개인 낙상 처리, BEFORE 개인·둘 다 ever·동시 성공 지표는 [공통 평가 규약](evaluation.md)을 따른다. `all_solo`와 `push`는 viewer/검사 preset이며 Stage1 전용 1명 단독 평가와 별개다.

공통 모델과 coupled의 차이는 [Stage2 coupled PUSH](stage2_coupled_push.md)를 따른다. 기존6N 그룹 관측이 없는 BEFORE PUSH 검사 저장본은 새 모델의 resume/viewer에 사용할 수 없다.

공통 Stage2 encoder의 Alias target 리다이렉션은 config와 무관하게 packet의 BEFORE 의존 관계를 따른다. 기본 설정은 복제 토큰의 feature/pose를 유지하고, 아래 Virtual Object 설정에서만 Alias XYZ를 변경한다. 원본 packet은 보존한다. 새 coordination의 Linear bias는 원본처럼0으로 초기화하며, 기존 PTH 로드는 저장된 bias를 그대로 복원한다.

## Near50 거리 변형

[stage2_before_push_near50.yaml](../../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50.yaml)은 기본 `stage2_before_push`에서 `before.distance_mixture`만 `relative_sum` near50으로 바꾼 별도 실험이다. 모델·독립 행동 비율·PUSH 크기/배치·loco 초기화·1200step·Stage1 PUSH21k 전이는 같다. 참고 설정 `stage2_before_nonemask_rsi433_near50`의 RSI433과 attention mask는 가져오지 않는다.

각 BEFORE 쌍을 reset할 때50% 확률로 near 모드를 뽑는다. 가까운 모드에서 선행 사람 A·공유 받침 O·후행 사람 B·후행 ON_TOP payload P의 XY 거리는 다음 조건을 만족한다.

- SIT/CLIMB: `distance(B,O) < distance(A,O)`.
- ON_TOP: `distance(B,O) + distance(B,P) < distance(A,O)`.

나머지50%는 기존 original 범위에서 뽑고 위 부등식을 강제하지 않는다. 따라서 전체에서 우연히 가까운 경우까지 합친 비율은50%보다 높을 수 있다. mix226의20/20/60 거리 구간이나 절대거리 단축과 다르다. 독립 PUSH에는 near 조건을 적용하지 않는다.

mix226·near50의 학습/추론 checkpoint 계약은 구분한다. 실행 명령은 [config](../config.md)에 있으며 새 학습은 기본 Stage1 PUSH21k에서 시작한다.


## 쌍 선택50·접근 최대5m 변형

`stage2_before_push_near50_pair50_source5`은 `stage2_before_push_near50`에서 쌍 생성 확률을0.8→0.5로 바꾸고 `before.source_max_distance: 5.0`을 추가한 별도 실험이다. 일반 단독/BEFORE의 사람→대상 상자, BEFORE 후행 ON_TOP의 사람→공유 상자와 사람→payload는1–5m다. SIT의 기존1–5m는 유지하며 단독 PUSH 접근은2–5m다. 4명 기준 이론적 쌍 수 확률은0쌍12.5%·1쌍62.5%·2쌍25%, 사람 수 기대 비율은BEFORE56.25%·단독43.75%다.

near50의 상대 거리 부등식과50% 선택은 유지한다. AT 목표와 단독 ON_TOP 받침은 기존 환경 원점±4.5m 정사각형 샘플링을 유지한다. BEFORE 후행 ON_TOP의 payload→공유 상자 거리에는 별도 상한을 추가하지 않는다. 사람→상자 제한과 near 조건으로 운반 거리가 간접적으로 달라질 수 있다. 단독 PUSH 이동도 기존2–10m를 유지한다.

source5·virtual·virtual_asym의 초기 PUSH 후보128개를16개씩 최대8묶음으로 검사한다. 기존 `sample_layout(..., max_attempts=1)`의 거리·각도·경로·충돌 검사를 그대로 재사용하고 환경마다 첫 유효 후보를 선택한다. 성공 환경은 다음 묶음에서 제외하며 마지막 묶음도 남은 후보 예산만 사용한다. 난수 소비 순서가 바뀌므로 동일 seed의 실제 배치는 이전 버전과 달라질 수 있다.128개가 소진되면 성공한 환경의 배치를 보존하고 실패 환경에만 기존 순차 경로 배치를 적용한다. 사람별 후보128개를 묶어 검사하고 유효한 경로를 유지하며, 나머지 경로가 막히면 배치 순서를 바꿔 다시 시도한다. 보완도 소진되면 실패 환경만 기존 장면 재샘플링 루프로 넘긴다. 과제 그래프·near 모드·거리 범위·최종 경로/충돌 조건은 유지하지만 보완 장면의 공간적 배치 분포는 기존 동시 샘플링과 다르다. 시뮬레이터 상태 반영의 좌표 반올림을 고려해 이 설정의 후보 생성에만 PUSH 경로 사이·사람·장애물 여유와 near 부등식의 거리 차이를1mm 더 요구하며 최종 검사 기준은 바꾸지 않는다. 기존 near50/mix226의 샘플링은 유지한다.

독립 행동 비율·보상·모델·loco·4명/1024환경/8토큰/1200step·기본21k 전이는 유지한다. 기존21k 파일이 없는 checkout에서는 `STAGE1_CHECKPOINT`로 존재하는 Stage1 PUSH 파일을 지정한다. 전용 train/test/VNC/eval·출력과 별도 checkpoint 계약을 사용하며 기존 near50/mix226 저장본과 resume/추론을 혼용하지 않는다. [실행 명령](../config.md).


## Goal-Aligned Virtual Object

`stage2_before_push_near50_pair50_source5_virtual`은 source5에서 `relationGraph.before_virtual_object: at_goal_xyz_v1`만 활성화한 별도 환경 설정이다. 기존 source5와 최근 배치 보완은 유지한다. 기본 전이 파일은 현재 연결된26k다. 비교 원본 실행도 같은26k를 지정하며 학습 YAML은 출력 이름만 다르다. 전용 train/test/VNC/eval 진입점과 출력·checkpoint 계약을 사용한다. 기존 저장본은 기존 wrapper로 계속 사용하고 서로 resume/추론을 혼용하지 않는다.

Actor와 Critic의 Stage1 Encoder 직전, 원본 scene을 복사하고 BEFORE 후행 SIT/CLIMB/ON_TOP Alias의 GTA XYZ를 선행 AT Goal token XYZ로 덮어쓴다. pose 위치는 intrinsic 뒤 `[Human|Object|Goal]`의 entity당7차원이며 task/group suffix와 구분한다. packet의 valid/task/actor/payload/target과 기존 `alias_layout()`으로 연결하므로 task record 순서나 Encoder 내부 token 셔플에 의존하지 않는다. 복사하는 값은 기존 env-local Goal XYZ 전체이며 높이를 재계산하거나 Goal quaternion을 복사하지 않는다. Alias quaternion은 매 스텝 실제 상자 회전을 유지하고 크기·BPS·속도·intrinsic도 유지한다.

원본 obs_buf·물리 상자·원본 Object/Goal token·Task graph·Human coordination packet을 수정하지 않는다. 기존 Alias target 리다이렉션도 유지한다. 독립 과제와 COUPLED PUSH에는 위치 변경이 없으며 네트워크 구조·가중치 전이·동결 설정도 유지한다. 기존처럼 actor encoder/입력 RMS는 동결하고 critic encoder는 학습한다.

후행 progress는 이미 선행 AT 목표 XY를 사용하므로 가상 상자 접근과 같은 목적지이며 보상 코드는 바꾸지 않는다. SIT/CLIMB는 root XY, ON_TOP은 payload XY에서 접근 거리를 계산하고 기존 실제 받침 크기/buffer·성공 포화를 유지한다. Progress는 gate 이전에도 지급하고 state/success는 실제 상자와 현재 AT 성공 gate를 따른다. HOLDING·PUSH·AMP 등 기존 추가 항도 유지하며 Soft Gate/Ready Reward를 추가하지 않는다.

실제 상자가 도착하기 전의 조기 상호작용과 전체 Entity Self-Attention을 통한 선행 AT 성능 변화를 비교해야 한다. 가상 Alias의 velocity/회전은 실제 상자를 따르므로 Stage1 전체 관측 분포와 완전히 같다는 뜻은 아니다. 검증 결과는 [changelog](../../changelog.md)에 기록한다.


## Asymmetric Progress

`stage2_before_push_near50_pair50_source5_virtual_asym`은 위 Virtual Object 실험에서 `before.asymmetric_progress.lambda_inner: 10.0`만 추가한 별도 환경이다. 기본26k 전이·모델·PPO/AMP·Virtual GTA·배치와 reward 가중치0.2는 유지하며 전용 train/test/VNC/eval·출력·checkpoint 계약을 사용한다. 기존 source5/virtual의 YAML과 저장본 계약은 유지한다.

후행 SIT/CLIMB/ON_TOP 수행 Edge의 progress만 다음과 같이 바꾼다.

```text
outer = max(0, d-r)
inner = max(0, r-d)
p = 1 / (1 + (outer + lambda_inner*(1-current_AT_success)*inner)/sigma)
```

`d`는 가상 상자(선행 AT Goal)까지 XY 거리이며 SIT/CLIMB는 root, ON_TOP은 payload 기준이다. 기존 `r=norm(size_xy)/2+buffer`, SIT/CLIMB buffer0.3·ON_TOP buffer0.1, sigma1.0을 그대로 사용한다. AT 미완료 시 반경 바깥은 기존 progress와 같고 경계에서1, 안쪽으로 들어갈수록 감소한다. 현재 AT 성공 시 기존 식으로 복원하고 성공을 잃으면 다시 안쪽 감쇠를 적용한다. lambda0은 기존 식과 같고 유한한0 이상 값을 허용한다.

`humanoid_ma_before.py`가 같은 스텝의 phi/오차와 기존 `interaction_own_success()`를 사용해 선행 성공을 `predecessor_at_progress()`에 전달한다. Runtime의 AT raw 성공과 동일한 판정 함수를 사용하며 이전 스텝 own_success나 achieved 이력은 읽지 않는다. AT 판정에는 기존 높이 조건도 포함하며 새 성공 기준을 만들지 않는다. PUSH 후처리는 CONTACT 항만 변경하므로 선행 AT 판정은 동일하다.

State/Success의 current_AT_success gate와 성공 saturation, ON_TOP HOLDING·독립 과제·PUSH·COUPLED 보상은 유지한다. 추가 Ready Reward·Soft Gate·직접 Penalty는 없다. 반경 주변 선호를 유도할 뿐 정지 대기나 AT 성공 직후 Skill 수행을 보장하지 않으며, ON_TOP 반경은 사람 대신 payload 기준이다. 검증 기록은 [changelog](../../changelog.md)를 따른다.


## 상자→목표 최대5m 변형

`stage2_before_push_near50_pair50_source5_goal5`, `stage2_before_push_near50_pair50_source5_virtual_goal5`, `stage2_before_push_near50_pair50_source5_virtual_asym_goal5`는 각각 기존 source5/virtual/virtual_asym에서 목표 배치만 바꾼 별도 실험이다. `before.goal_distance: [1.0, 5.0]`을 활성화하고 `jointPush.scene.goal_distance: [2.0, 5.0]`을 사용한다. 기본 전이는 세 실험 모두26k이며 기존 Stage2 저장본에서 resume하는 설정이 아니다.

단독 AT와 BEFORE 선행 AT는 현재 운반 상자 중심에서 반경1–5m, 단독 ON_TOP은 현재 들 상자 중심에서 반경1–5m에 목표 받침을 놓는다. 거리와 방향을 균등하게 뽑고 사람과 목표 사이 최소1m·기존 물리 충돌 검사를 유지한다. 해당 목표에는 환경 원점±4.5m 정사각형 제한을 적용하지 않는다. AT 목표 Z와 ON_TOP 받침 높이는 기존 계산을 유지한다. PUSH는 기존 접근2–5m와 방향/경로/충돌 규칙을 유지하고 이동 거리만2–5m로 줄인다.

BEFORE 후행의 현재 공유 상자/들 상자 배치와 near50 조건은 유지한다. 후행 SIT/CLIMB의 사람→선행 목표, ON_TOP의 들 상자→선행 목표에는 별도 거리 상한을 추가하지 않는다. Virtual GTA와 비대칭 progress(lambda10)는 각각 기존 실험대로 유지하며 보상·성공·Gate·PPO/AMP·모델을 바꾸지 않는다.

새 설정의 후보 검사에서만 상자 간격·AT 목표와 다른 물체의 간격에1mm 여유를 더 요구하고, PUSH 후보 cone은89.98°로 뽑는다. 최종 충돌 기준과90° cone은 유지한다. 기존 설정의 샘플링/RNG 경로와 checkpoint 계약은 유지하고 새 거리 설정은 별도 계약으로 저장한다. 전용 train/test/VNC/eval과 출력 이름을 사용한다. [실행 명령](../config.md), [변경·검증](../../changelog.md).

## goal5 기반 목표 중심 후행 배치

세 goal5 실험에 `_goalroute`를 붙인 별도 설정이다. 단독 과제의 goal5 거리, pair 생성 확률0.5, alias/Virtual, lambda10, reward/gate/success, Stage1 전이·동결과 AMP/PPO는 각 원본을 유지한다. 모든 거리는 reset 시 XY 기준이다.

- 선행 A→공유 상자 O와 O→기존 AT 목표 G는 각각1–5m다. 목표 Z는 기존 상자 높이 절반을 유지한다.
- 후행 SIT/CLIMB는 G를 중심으로 B를1–10m 반경에 배치한다.
- 후행 ON_TOP은 G→들 상자 P를1–5m, P→사람 B를1–5m로 뽑는다. B→G≥1m를 만족하는 각도를 선택한다. B→P+P→G는2–10m이고 B→G는1–10m 범위지만 균등분포가 아니다.
- 기존 AT 목표가 생성된 뒤 후행 root와 kinematic body의 XY를 같은 양만큼 이동하고 ON_TOP 들 상자 XY만 배치한다. 원본 상자·목표·회전·초기 자세와 속도는 이 배치 연산에서 유지한다. 기존 전체 장면 충돌/경로 검사와64회 유한 재시도를 사용한다.

near50은 생성된 쌍의 절반에 `D_B < D_A`를 강제한다. `D_A=|A−O|+|O−G|`, SIT/CLIMB의 `D_B=|B−G|`, ON_TOP의 `D_B=|B−P|+|P−G|`다. 나머지 절반에는 대소 조건을 강제하지 않으므로 실제 후행 짧음 비율은50%보다 클 수 있다. 거리 대소는 완료 시간 대소를 뜻하지 않는다.

설정은 `before.distance_mixture`의 `goal_route_sum`/`goal_centered` 계약으로 구분한다. near SIT/CLIMB는 허용 반경을 직접 제한하고 near ON_TOP은 두 U[1,5] 합의 조건부 CDF로 거리 합을 뽑은 뒤 해당 합의 유효 구간에서 두 거리를 분할한다. 이는 거리 합 조건의 rejection 분포와 같고, 짧은 선행 경로에서 수많은 후보를 버리는 루프를 피한다. 이후 G–B 최소 거리는 각도를 조건부 샘플링한다. 후보 near 비교에는 수치 여유를 두고 최종 strict 대소와 실제 PhysX 반영을 검증한다.

기존 설정의 배치 함수/RNG 경로는 그대로 두며 새 설정의 난수 소비 순서는 다르다. 새 before 계약과 출력·실행 스크립트를 분리했으므로 기존 goal5 checkpoint로 새 실험을 resume하지 않는다. 기본 전이는 동일한 Stage1 26k다.

신규 goalroute의 reset에서는 AT 목표·후행 배치를 PUSH 후보 검사보다 먼저 확정한다. 이후 고정된 과제 물체와 사람의 충돌을 선검사해 이미 불가능한 장면은 기존 외부 재시도로 넘긴다. 이 선검사는 앞으로 배치할 PUSH 상자·distractor·단독 ON_TOP 받침을 제외하며, 최종 전체 장면 검사는 그대로 수행한다. 따라서 물리 판정을 완화하지 않고 불가능한 장면의 PUSH 순차 fallback 반복을 줄인다.
