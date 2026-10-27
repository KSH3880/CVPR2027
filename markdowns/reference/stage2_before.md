# Stage2 BEFORE

`stage2_shared_full_success`의 fixed Stage1 전이·공통 장면 encoder·PPO·전원 loco 초기화를 기반으로 한다. 큰 공유 받침, 구역 토큰, COUPLED, 공동운반은 사용하지 않는다. 실행은 [config](../config.md), 추가 평가 옵션은 [USAGE](../USAGE.md)에 있다.

## 과제와 물체

기본값은 4명·1024환경·8개 실제 물체 토큰이다. 학습·평가 모두 인원 N≥1, 물체 M≥N을 지원한다. 사람 순서를 섞고 남은 사람이 둘 이상이면 BEFORE 80%/단독 20%를 순차 추첨한다. BEFORE는 정확히 2명을 소비하고 단독은 1명이다. 마지막 1명은 단독이다. 4명·8물체에서는 두 쌍/한 쌍+단독2명/단독4명이 가능하며 장면 비율은 각각 64/35.2/0.8%다. 필요한 고유 물체 수가 M을 넘으면 해당 환경의 그룹·역할·과제를 다시 샘플링한다. 따라서 예산이 작은 경우 최종 그룹·과제 비율은 조건부 분포가 된다. 재시도는 1024회로 제한하며 실패 시 물체 수를 늘리라는 오류를 낸다. 전원 ON_TOP 고정 과제는 M≥2N이 필수이고, pair preset은 N≥2여야 한다. `before_*`는 최대 쌍 수, `one_pair_*`는 한 쌍이며 남은 사람은 단독이다.

선행 A는 AT 운반, 후행 B는 A의 payload에 SIT/CLIMB 또는 자기 payload를 ON_TOP한다. 두 역할은 가능한 상자 크기에서 무작위 배정한다. SIT/CLIMB/ON_TOP의 가중치는 같으며 두 후보 중 하나라도 만족하는 크기 조건에 한해 과제를 뽑는다. 단독은 SIT/CLIMB/AT/ON_TOP=10/25/32.5/32.5%다.

물리 bank는 max(4N, M)개로, 사람별 운반/SIT/CLIMB/일반 받침 크기를 기존 Stage2 범위로 샘플링하고 4N개를 넘는 추가 물체에는 일반 받침 크기를 쓴다. 과제용 물체를 중복 없이 모으고 남는 토큰은 미할당 일반 bank 상자로 채운다. M개 모두 실제 관측·충돌 물체이며 비활성 bank만 장면 밖으로 둔다. 과제·역할은 reset마다, 실제 크기는 환경 생성 시 추첨한다. 고정 BEFORE SIT/CLIMB preset은 선행 후보 운반 bank를 해당 과제와 운반 크기 범위의 교집합에서 만든다.

Viewer에서는 유효 task graph의 source와 target 전체를 기준으로 물체 사용자를 집계한다. 미할당 distractor는 회색, 한 사람의 과제 물체(개인 ON_TOP 받침 포함)는 그 사람 색, 여러 사람이 사용하는 물체는 노란색이다. BEFORE의 선행 payload는 후행도 사용하는 공유 물체이므로 노란색이고, 색상은 reset마다 갱신한다.

## 모델·보상

Actor/critic의 human SA4는 SELF/NONE/BEFORE 세 관계를 사용한다. B query→A key만 BEFORE이고 역방향·다른 사람은 NONE이다. NONE은 attention 차단이 아니다. 관계는 task packet의 선행 payload와 후행 target 연결로 계산한다. GraspView·물리 받침 ID metadata·구역 분할은 사용하지 않는다.

선행의 현재 AT 성공(3D거리≤20cm, 높이 차이≤1mm)이 후행 목표의 state/success/성공 포화를 허용한다. HOLDING·progress는 유지한다. AT가 깨지면 gate도 닫히며 완료 latch나 조기 수행/대기 페널티는 없다. 성공 전 행동 자체를 금지하지 않는다. 단독 보상·CPA·파워·상자 속도 페널티는 기반 설정과 동일하다.

## 초기화와 실행 계약

학습·평가 모두 loco이며 Stage1 과제별 AMP를 사용한다. 공동 reference나 크기별 RSI 생성은 하지 않는다. 기본 거리·상자 크기·고정 선반·AT 목적지 높이 분포는 현재 기반 설정을 따른다. 후행은 선행 상자 주변에서 SIT 1~5m, CLIMB/ON_TOP 1~10m로 배치하고, ON_TOP의 자기 payload는 자기 사람 기준 기존 loco 규칙으로 배치한다. 장면 물체와 distractor 겹침을 검사하며 과제/역할을 유지한 채 위치를 재추첨한다.

기본 전이는 `stage1/Stage1V2Fixed_00030500.pth`다. actor encoder·관측 RMS는 동결하고 새 coordination은 항등 초기화한다. 학습률·AMP/PPO 설정은 공유 full-success의 값을 유지한다. PPO는 `scene_minibatch_size=4096`을 기준으로 `minibatch_size=4096×N`을 계산하므로 홀수 인원도 학습할 수 있다. 기본 4명에서는 기존 16384와 같다. BEFORE 전용 checkpoint 계약을 저장해 다른 실험과 재개/평가를 구분한다.

`BEFORE_FOLLOWER_NEAR=1`은 loco 평가에만 적용한다. 선행 상자 중심까지의 두 사람 root XY 거리를 1~5m에서 추첨해 후행에 작은 값을 주고 각도도 추첨한다. ON_TOP 후행의 자기 payload는 후행 사람의 최종 root XY 기준 1~3m·무작위 각도로 별도 배치한다. 선행 상자는 유지하고 기존 높이·자세 기울기와 겹침 검사를 보존한다. 초기 배치에만 적용하며 단독 인원·옵션을 끈 평가·학습 배치·보상 gate는 유지한다.

검증 결과는 [changelog](../../changelog.md), 산출물은 `output_etc/check/stage2_before/`에 둔다.

## 바닥 전용 비교: stage2_before_ground

`stage2_before_ground`는 기존 BEFORE와 별도 env/train YAML·train/test/VNC·output을 쓴다. `randomHeight=false`, `randomHeightProb=0`, `platformMode=ground_only`로 모든 초기 상자를 바닥에 두고, BEFORE 선행과 단독 AT의 목표 중심 높이를 운반 상자 높이의 절반으로 둔다. 시작 받침과 AT 도착 받침은 모두 비활성화한다. 물리 actor 인덱스 구조를 유지하기 위해 고정 받침 슬롯은 지하 Z=-20m/-19.5m에 격리한다. ON_TOP의 쌓기 대상 상자는 실제 과제 물체로 유지한다.

그룹 추첨·물체 예산 재샘플링·loco 초기화·현재 AT gate·보상·색상·학습 설정은 기존 BEFORE와 동일하다. 상자 yaw·XY·크기는 계속 무작위다. Stage1 fixed 가중치 전이로 새 학습을 시작하며 기존 BEFORE의 학습 프로세스와 설정은 유지한다. 저장된 reset 계약이 달라 기존 BEFORE checkpoint의 직접 resume/환경 변경 평가는 거부한다. ground test/VNC는 ground 저장본 옆 환경 설정을 사용한다.

## 바닥 BEFORE 거리 혼합: stage2_before_ground_mix50

`stage2_before_ground`에서 후행 접근 거리를 줄인 샘플을 더 자주 제공하는 별도 실험이다. 모든 사람은 loco로 시작하며 동작 중간 자세 RSI를 추가하지 않는다. BEFORE 쌍마다 episode graph를 만든 뒤 가까운/기존 배치를 각각 50%로 독립 추첨한다. 물리 겹침 재시도에서는 선택한 모드를 유지한다. 유한한 샘플 수에서 정확히 절반을 강제하지 않는다.

| 초기 root XY/상자 중심 XY 거리 | 가까운 배치 50% | 기존 배치 50% |
| --- | --- | --- |
| 선행 사람 → 선행 AT 상자 | 1~10m | 1~10m |
| 선행 상자 → 후행 SIT 사람 | 1~5m | 1~5m |
| 선행 상자 → 후행 CLIMB/ON_TOP 사람 | 1~5m | 1~10m |
| ON_TOP 후행 사람 → 자기 운반 상자 | 1~5m | 1~10m |

각 반경과 각도는 독립 추첨하고 거리 대소 관계를 강제하지 않는다. ON_TOP은 선행 상자→후행 사람→후행 자기 상자 순서다. 사람/물체 간 겹침 검사는 그대로이며 최종 거리 분포는 이 조건에 영향을 받는다. 단독 과제·선행 AT 목표 추첨·과제/그룹 비율·모델·보상·AMP/PPO·바닥 높이는 기존 ground와 같다.

학습과 기본 test/VNC 모두 저장된 혼합 배치를 사용한다. `BEFORE_FOLLOWER_NEAR=1`은 평가에서만 모든 BEFORE 쌍에 가까운 모드를 선택한다. 선행 사람→상자 1~10m, 선행 상자→후행 사람 1~5m, ON_TOP 후행→자기 상자 1~5m이며 거리 순서를 강제하지 않는다. 기준은 선행 상자의 초기 위치이며 AT 도착 위치 주변 배치는 아니다. 저장 config/계약·단독 과제·학습 비율은 바꾸지 않는다. `relation/sampling/before_near_fraction`은 기록 구간에서 새로 추첨한 BEFORE 쌍 중 가까운 모드 비율이며, `before_pairs`는 분모인 쌍 수다. 배치 재시도는 중복 집계하지 않는다.

`before.distance_mixture`가 checkpoint 계약에 포함되어 기존 ground와 직접 resume/혼용하지 않는다. 기본 Stage1 source는 형제 checkout의 fixed73000 파일이며 `STAGE1_CHECKPOINT`로 바꿀 수 있다. 기존 checkpoint는 읽기만 하고 새 결과는 `output/stage2_before_ground_mix50/`에 저장한다. 실행법은 [config](../config.md), 추가 옵션은 [USAGE](../USAGE.md#stage2-before-5050-혼합-배치)를 따른다.

## 바닥 BEFORE 거리 혼합: stage2_before_ground_mix334_1200

별도 config·checkpoint 계약·output을 쓰며 에피소드 최대 길이는 장면 전체에 1200 제어 스텝(약40초)이다. BEFORE 쌍마다 아래 모드를 독립 추첨하고, 물리 배치 재시도 동안 모드를 유지한다. 단독 과제의 배치·AT 도착 목표 샘플러·과제/그룹 비율·loco 초기화·Stage1 동결·보상·AMP/PPO는 ground와 같다.

| 초기 수평 거리 | 기존 30% | 가까움 30% | 더 가까움 40% |
| --- | --- | --- | --- |
| 선행 AT 사람 → 자기 상자 | 1~10m | 1~10m | 1~5m |
| 후행 SIT 사람 → 선행 상자 | 1~5m | 1~5m | 1~2.5m |
| 후행 CLIMB 사람 → 선행 상자 | 1~10m | 1~5m | 1~2.5m |
| 후행 ON_TOP 사람 → 선행 상자 | 1~10m | 1~5m | 1~2.5m |
| ON_TOP 자기 상자 → 후행 사람 | 1~10m | 1~5m | 1~2.5m |

최소 거리는1m를 유지한다. 더 가까운 모드에서 선행 상자는 고정하고 선행 사람의 기존 방사 방향·반경을1~10m에서1~5m로 선형 변환한다. 이후 후행과 ON_TOP 자기 상자를 각자 독립 각도·반경으로 배치한다. AT 목표는 기존 샘플러로 최종 사람/상자 배치 후 생성한다. 후행이 반드시 선행보다 가깝게 나오도록 정렬하지 않으며, 겹침 검사에 따른 거부 추첨은 그대로다. 거리 기준은 선행 상자의 초기 위치이고, AT 도착 위치까지의 운반 거리는 별도로 축소하지 않는다.

`relation/sampling/before_original_fraction`, `before_medium_fraction`, `before_short_fraction`은 새 BEFORE 쌍의 세 모드 비율이다. `before_near_fraction`은 두 가까운 모드를 합친 비율(기대70%), `before_pairs`는 분모이며 배치 재시도는 중복 집계하지 않는다. 기본 test/VNC도30/30/40을 사용한다. `BEFORE_FOLLOWER_NEAR=1`은 기존 mix50와 같은 의미로 가운데 모드(선행1~10m·후행/자기 상자1~5m)만 선택한다. mix334 전용 test/VNC의 `BEFORE_FOLLOWER_NEAR=2`는 더 가까운 모드(선행1~5m·후행/자기 상자1~2.5m)만 선택한다. 평가에서만 BEFORE 쌍의 모드를 고정하며 저장 config와 학습 분포는 유지한다.

기본 source는 형제 checkout의 fixed73000이다. 기존 ground/mix50 checkpoint와 직접 resume하지 않으며 새 실험의 저장본으로만 재개한다. 결과는 `output/stage2_before_ground_mix334_1200/`에 저장한다. 학습은1200스텝을 검사하며, 평가는 `EPISODE_LENGTH`로 재정의할 수 있다. 실행은 [config](../config.md), 추가 옵션은 [USAGE](../USAGE.md#stage2-before-mix334-1200)를 따른다.

## 바닥 BEFORE 거리 혼합: stage2_before_ground_mix226_1200

BEFORE 쌍마다 기존20%·가까움20%·더 가까움60%를 추첨한다. 아래 거리는 초기 사람 root와 상자 중심, AT 목표의 수평 XY 거리다. 모드는 물리 배치 재시도 중 유지한다.

| 역할/과제 | 기존20% | 가까움20% | 더 가까움60% |
| --- | --- | --- | --- |
| 선행 AT 사람 → 자기 상자 | 1~10m | 1~10m | 1~5m |
| 선행 AT 상자 → AT 목표 | 1~10m | 1~10m | 1~5m |
| 후행 SIT → 선행 상자 | 1~5m | 1~5m | 1~2.5m |
| 후행 CLIMB → 선행 상자 | 1~10m | 1~5m | 1~2.5m |
| 후행 ON_TOP → 선행 상자 | 1~5m | 1~2.5m | 1~1.25m |
| ON_TOP 자기 상자 → 후행 사람 | 1~5m | 1~2.5m | 1~1.25m |

기존 선행 상자 위치를 유지하고 더 가까운 모드에서 선행 사람을 당기는 방식은 mix334와 같다. 각 반지름은 구간 균등, 방향은 독립 무작위이며 선행·후행 거리 순서를 강제하지 않는다. ON_TOP 자기 상자는 최종 후행 사람 위치가 기준이며 선행 상자까지의 운반 거리 자체를 제한하는 것은 아니다.

BEFORE 선행 AT 목표는 초기 상자 중심에서 위 거리로 뽑으며 환경 중심 ±4.5m 사각형으로 제한하지 않는다. 선행 사람과 목표 사이 최소1m 제외 조건 및 기존 물리 겹침 재시도는 유지하므로 승인된 최종 배치의 분포는 조건부다. 목표 Z는 상자 반높이인 바닥 목표를 유지한다. 후행의 대상은 선행이 옮기는 같은 상자다.

단독 과제는 기존 초기화와 환경 중심 AT/ON_TOP 목표 샘플러를 유지한다. BEFORE/단독 그룹 추첨·전원 loco·보상/gate·Stage1 동결/전이·PPO는 변경하지 않으며 전체 장면의 에피소드 상한은1200스텝이다. `before_original_fraction`, `before_medium_fraction`, `before_short_fraction`은20/20/60, `before_near_fraction`은 두 가까운 모드 합계인80%를 기대한다.

기본 source는 형제 checkout의 fixed73000이며 결과는 `output/stage2_before_ground_mix226_1200/`에 저장한다. 기존 ground/mix50/mix334 checkpoint와 직접 resume하지 않고 이 실험의 저장본으로만 재개한다. test/VNC의 `BEFORE_FOLLOWER_NEAR=0/1/2`는 기본 혼합/가까움/더 가까움을 선택하며 AT 운반 거리도 해당 모드를 따른다. 실행은 [config](../config.md), 추가 옵션은 [USAGE](../USAGE.md#stage2-before-mix226-1200)를 따른다.


## 선행 AT 목표 progress: stage2_before_ground_mix226_atprogress_1200

mix226의 과제/역할 추첨·크기 조건·20/20/60 거리 혼합·loco 초기화·1200스텝·PPO·모델/관측을 유지한다. 현재 mix226 본학습과 같은 fixed78000에서 새로 전이하며 actor encoder와 관측 RMS를 동결한다. 기존 실험과 config/train/test/VNC/output/checkpoint 계약을 분리한다.

`before.progress_target: predecessor_at_goal`을 설정한다. BEFORE 후행 SIT/CLIMB의 사람 root XY 또는 ON_TOP의 자기 payload XY에서 선행 AT 목표 XY까지 거리로 progress를 계산한다. 원래 대상 상자의 XY bbox 반경과 기존 task별 buffer(SIT/CLIMB0.3m, ON_TOP0.1m), sigma1, progress 가중치0.2를 유지한다. 따라서 `0.2/(1+max(distance-radius,0))`이며 정지해 있어도 현재 근접도에 따른 보상을 받는다. 별도 보상항이나 대기 영역은 추가하지 않는다.

선행 AT gate가 열려도 progress는 AT 목표 기준을 유지한다. state/success는 실제 대상 상자에 대한 기존 기하와 현재 AT gate를 따르며, 실제 성공 시 기존 progress 포화도 유지한다. ON_TOP의 HOLDING 전체와 선행/단독 edge는 변경하지 않는다. 목표점 중앙을 비우도록 강제하는 보상은 없다. 선행 목표는 기존 전체 장면 관측에 포함되며 별도 후행 입력은 추가하지 않는다.

`before_reward.py`의 `predecessor_at_progress`가 graph의 prerequisite AT edge에서 목표를 찾는다. `humanoid_ma_before.py`는 공통 기하 계산 후 progress만 교체하며 논리 물체 순열과 부분 reset에도 같은 경로를 사용한다. 기존 `distance_xy`, `target`, state/success 기하는 실제 상자 기준을 유지한다. 새 실험의 `relation/before/{sit,climb,ontop}/{gate,raw_success,success}`는 BEFORE 후행만 집계한 스텝 평균이며 에피소드 성공률이 아니다. 기존 합산 지표도 유지한다.

결과는 `output/stage2_before_ground_mix226_atprogress_1200/`에 저장한다. 기존 mix226 checkpoint와 직접 resume하지 않는다. 실행은 [config](../config.md), 옵션은 [USAGE](../USAGE.md#stage2-before-mix226-at-progress-1200)를 따른다.


## Actor SELF/NONE bias 0

`stage2_before_ground_mix226_actorbias0_1200`과 `stage2_before_ground_mix226_atprogress_actorbias0_1200`은 각각 mix226 및 AT progress를 기반으로 `relationGraph.actor_human_bias: before_only`만 추가한다. 두 실험 모두 현재 비교와 같은 fixed78000에서 Stage2를 새로 시작한다. actor scene encoder·관측 RMS 동결, 4명·1024환경·8물체, 20/20/60 거리 혼합·1200스텝·보상·PPO는 각 기반 설정을 유지한다.

actor의 human coordination 4 layer × 2 head에서 NONE(0)·SELF(1) bias는 상수 0이며 BEFORE(2)만 학습한다. `CoordinationBlock.relation_bias`는 이 모드에서 `[heads, 1]` BEFORE 파라미터만 저장한다. `effective_relation_bias()`가 `[0, 0, BEFORE]` 테이블을 만들어 attention에 사용하므로 SELF/NONE을 위한 optimizer 상태·학습 파라미터가 없다. 분석 도구는 raw 파라미터의 3열을 가정하지 않고 이 메서드로 실제 bias를 읽는다. critic은 기존 `[heads, 3]` SELF/NONE/BEFORE 모두 학습한다. QK/V·action head·critic의 기존 학습 경로는 유지한다.

기존 옵션 미지정은 `all`로 처리하며 계산·파라미터 형태·기존 checkpoint 계약을 보존한다. 새 mode는 BEFORE checkpoint 계약에 기록하고 train/test 설정 불일치를 거부한다. 두 새 실험끼리도 progress 계약이 다르므로 교차 resume하지 않는다. 결과는 각각 동명의 `output/<컨피그>/`에 저장한다. 실행은 [config](../config.md), 옵션은 [USAGE](../USAGE.md#stage2-before-actor-selfnone-bias-0)를 따른다.


## AT progress 후행 carryWith RSI 50/50

`stage2_before_ground_mix226_atprogress_carry50_1200`은 AT progress 226의 거리20/20/60·1200스텝·보상·모델·전체 관계 bias 학습을 유지한다. 같은 fixed78000에서 전이한다. `before.follower_carry_rsi={probability: 0.5, skill: carryWith}`를 checkpoint 계약에 포함해 기존 실험과 혼용을 막는다.

graph마다 BEFORE ON_TOP 후행만 50% carryWith를 요청하며 물리 배치 재시도 중 선택은 유지한다. 나머지 사람과 loco 후행의 배치는 기존과 같다. carryWith는 프레임의 사람·자기 상자 상대 자세를 보존해 후행 위치로 평행이동하고 함께 yaw 회전한다. 후행→선행 상자 거리와 선행 접근/운반 거리만 기존 mode를 따르며, RSI 자기 상자는 손에 들린 위치를 사용해 loco의 1~5/1~2.5/1~1.25m 배치를 생략한다. 클립 끝 목표나 선행 클립 시간 동기화는 사용하지 않는다. 시작 후에는 정책이 행동한다.

`BeforeCarryRsiCache`는 carryWith만 실제 payload 크기별 기존 단독 물리 screening(.1초)과 기하 검사를 수행한다. 공중 payload(바닥8cm 초과)·기존 HOLDING 조건·BEFORE 몸 관통 조건을 만족한 프레임만 균등 샘플링하며 클립을 리타게팅하지 않는다. 유효 프레임 없는 크기는 loco로 fallback하고 횟수와 실현 비율을 기록한다. 50%는 요청 확률이며 사용 가능률이 낮으면 실현 비율은 낮아질 수 있다. 초기 손-물체 상대 자세와 장기 정책 파지 성공은 별개다.

기본 평가에는 RSI를 넣지 않으며 별도 `--before_carry_rsi mix|carry`로만 활성화한다. 학습 gate·단독 행동·Stage1 encoder/RMS 동결은 유지한다. 실행 옵션은 [USAGE](../USAGE.md#stage2-before-mix226-at-progress-carry50)에 있다.

## 단독·BEFORE 전체 Stage1 RSI

`stage2_before_ground_mix226_atprogress_fullrsi_1200`은 AT progress mix226 기반의 별도 실험이다. Stage1 fixed78000에서 새로 전이하며, 기존4명/1024환경/8토큰·1200스텝·20/20/60·ground-only·보상/gate·전체 관계 bias·동결 정책을 유지한다. 단독의 높이/선반까지 원본 Stage1로 돌리는 실험은 아니다.

모든 역할에서 Stage1의 과제별 templateRsi를 사용한다: HOLDING=50%loco/50%pickUp, SIT=50%loco/50%sit, CLIMB=50%loco/50%climb, AT/ON_TOP=50%loco/10%pickUp/30%carryWith/10%putDown. RSI 프레임은70% 확률로 유효 행동 구간의 마지막30%,30% 확률로 전체 유효 프레임에서 선택한다. 후반 후보가 없으면 전체 유효 프레임으로 대체한다. loco에는 행동 후반 우선 추첨을 적용하지 않는다.

순서는 graph·역할·실제 물체 binding→쌍별 거리 모드→사람별 시작 행동→크기에 맞는 프레임→배치다. 크기 fallback은 Stage1의 `resolve_skills`를 그대로 사용한다. 이미 가능한 요청은 유지하며 불가능한 요청만 원래 가중치×사용 가능 mask로 재추첨한다. 물리 재시도 중 graph·거리 모드·처음 요청한 행동은 다시 추첨하지 않는다. 합성 배치가8회 실패한 경우 불가능한 후행 자세가 명확하면 해당 후행만, 그 외 장면 기하 충돌이면 해당 장면의 선택된 비-loco 행동을 제외하고 동일 가중치 규칙으로 재추첨한다. 재시도는 최대64회이며 모두 실패하면 오류로 중단한다. 이 합성 fallback은 크기 fallback과 별도로 기록하며, 최종 적용 분포는 요청 비율과 다를 수 있다.

단독 AT와 ON_TOP의 자유 목표 XY는 Stage1의 환경 중심±4.5m/사람·payload1m 제외를 사용한다. 단독 putDown 목표/받침은 클립 종료 위치다. ground-only를 유지하며 RSI source 자체의 클립 높이는 보존한다.

BEFORE 선행 loco의 접근 거리와 loco/pickUp/carryWith의 상자→AT 목표 거리는 기존226 모드다. 선행 RSI의 사람↔상자 상대 자세는 보존하고, putDown 목표는 클립 종료 위치를 사용한다. 후행 loco는 기존 거리표, ON_TOP pickUp/carryWith는 사람+자기 payload 묶음을 선행 상자 주변으로 이동한다. 후행 sit/climb/putDown은 선행 상자를 대상으로 XY/yaw만 정렬한다. 사람을 높이 이동/기울여 들린 받침에 억지로 맞추지 않으며, 받침이 바닥에서 벗어나거나 기울어진 조합은 재시도/fallback한다. 공유 상자의 실제 상태는 선행만 결정한다. ON_TOP putDown payload가 이미 선행 AT 목표 위에 있는 것은 빈 목표 공간 heuristic의 예외로 허용하되 상자/사람 관통 검사는 유지한다.

`before_full_rsi_cache.py`는 모든 단독 bank 및 가능한 같은 환경의 선행·후행 상자 크기 조합을 사전 검사한다. Stage1과 같은 native0.1초 PD 유지·속도/변위·body-point/발 높이/fall 검사를 사용하며, 캐시 유효성이 장기 물리 안정성이나 두 사람의 합성 장면 안정성을 보장하지는 않는다. 장면 reset에서는 별도의 공유 상자/사람 기하 검사를 수행한다. AMP reference history에는 실제 선택된 클립/시점을 사용하며, XY/yaw 변환은 heading-local AMP 특징을 보존한다.

평가 전용 `both_carry`는 BEFORE ON_TOP 쌍의 두 사람에게 carryWith를 요청한다. 실제 크기/배치 검사를 유지하며 적용 불가한 사람만 loco로 fallback한다. 다른 과제/단독은 loco다. `mix`와 학습의 Stage1 확률·후반 프레임 선택은 유지한다. 사용법은 [USAGE](../USAGE.md#stage2-before-mix226-at-progress-full-rsi)에 있다.

`rsi/{solo,leader,follower}/{holding,sit,climb,at,ontop}/{skill}_{requested,executed}`와 `size_fallback`, `placement_fallback`, `rsi/late_branch_fraction`으로 실제 분포를 확인한다. 후반 branch 비율은 재시도 중 프레임 선택도 포함한다. 평가 기본은 전원loco, `BEFORE_FULL_RSI=mix`만 학습 혼합을 사용한다. 전원loco에서의 BEFORE 성공과 RSI 시작 성공은 분리해 비교한다.

## AT progress 공유 action head adapter

`stage2_before_ground_mix226_atprogress_adapter_1200`은 AT progress 226 설정에서 `relationGraph.actor_head_adapter: tokenhsi_internal_v1`만 추가한다. 거리20/20/60·loco 시작·1200step·후행 AT progress·gate·학습률2e-5·전체 SELF/NONE/BEFORE bias를 유지한다. Stage1 V2 Fixed78000에서 새로 전이하며 기존 Stage2 학습 가중치를 이어받지 않는다.

actor의 Stage1 encoder/RMS와 action head 세 Linear를 동결한다. 사람 SA4와 공유 adapter 두 개를 학습하고 critic/AMP의 학습 정책은 유지한다. TokenHSI `amp_network_builder_transformer_adapt.py`의 내부 adapter처럼 hidden Linear 출력에 병렬 선형 보정을 더한 뒤 기존 ReLU를 적용한다.

```text
x  = human SA4 output (64)
h1 = ReLU(frozen Linear1(x)  + Adapter1(x))   # 64 → 1024
h2 = ReLU(frozen Linear2(h1) + Adapter2(h1))  # 1024 → 512
mu = frozen Linear3(h2)                     # 512 → 32
```

Adapter1·2는 bias를 포함한 full-size Linear이고 weight/bias 모두0에서 시작한다. 별도 activation·출력층 adapter·관계별 분기는 없다. 단독/선행/후행 및 모든 스킬/관계에 같은 두 adapter를 사용한다. `requires_grad=False`인 head를 통과해서도 SA에 gradient를 전달한다. 초기 adapter 기여0은 초기 정책을 보존하지만 학습 이후 출력 변화나 기존 능력 보존을 보장하는 제약은 아니다.

기존 action head의 checkpoint key를 유지해 Stage1 tensor 전체를 복사한다. 새 `action_adapters.*`만 SA와 함께 새로 초기화하며 체크포인트 계약에 adapter 종류를 포함해 기존 head 학습형과 혼용을 거부한다. resume 시 head 동결·adapter optimizer 구성을 다시 적용한다.

## Adapter actor·critic SELF/NONE bias 0

`stage2_before_ground_mix226_atprogress_adapter_bias0_1200`은 공유 adapter 설정에서 `relationGraph.actor_human_bias`와 `critic_human_bias`를 모두 `before_only`로 지정한다. actor/critic 사람 SA4의 각 layer/head에서 NONE·SELF는 상수0이고 BEFORE만 학습한다. 각 block의 bias 파라미터는 `[heads, 1]`이며 forward에서 앞에0 두 열을 붙이므로 optimizer 업데이트·resume 후에도 SELF/NONE이 바뀌지 않는다. Actor와 critic의 BEFORE bias·QKV 가중치는 서로 별개다.

Stage1의 actor/critic 모두 SELF/NONE 관계 bias0 규칙을 Stage2 사람 SA에도 적용하는 비교다. QK 점수·attention 자체를0으로 만들지 않으며, adapter Linear의 bias와 일반 Linear bias는 이 옵션의 대상이 아니다. 기존 Stage1 encoder의 관계 bias 계산은 유지한다.

기반 adapter 실험의 encoder/RMS·head3층 동결, 공유 Adapter1·2의0 초기화·학습, critic/AMP 학습, AT progress·거리20/20/60·loco·1200step·학습률2e-5는 그대로다. 같은 fixed78000에서 새로 시작한다. 기존 actor-only bias0 실험의 critic은 계속 전체 관계 bias를 학습한다. 새 설정은 actor/critic bias 모드를 checkpoint 계약에 포함하므로 기존 adapter·actor-only 체크포인트와 혼용할 수 없다.
