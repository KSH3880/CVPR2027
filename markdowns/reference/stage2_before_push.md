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

공통 Stage2 encoder는 config와 무관하게 packet의 BEFORE 의존 관계를 보고 alias를 변환한다. 역할별 복제 토큰의 feature/pose와 원본 packet은 보존한다. 새 coordination의 Linear bias는 원본처럼0으로 초기화하며, 기존 PTH 로드는 저장된 bias를 그대로 복원한다.

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

새 거리 설정에서 PUSH 경로 후보가 소진되면 성공한 환경의 배치를 보존하고 실패 환경만 기존 장면 재샘플링 루프에서 다시 배치한다. 거리·경로·충돌 조건은 완화하지 않으며 기존 실험의 후보 횟수와 실패 동작은 유지한다.

독립 행동 비율·보상·모델·loco·4명/1024환경/8토큰/1200step·기본21k 전이는 유지한다. 기존21k 파일이 없는 checkout에서는 `STAGE1_CHECKPOINT`로 존재하는 Stage1 PUSH 파일을 지정한다. 전용 train/test/VNC/eval·출력과 별도 checkpoint 계약을 사용하며 기존 near50/mix226 저장본과 resume/추론을 혼용하지 않는다. [실행 명령](../config.md).
