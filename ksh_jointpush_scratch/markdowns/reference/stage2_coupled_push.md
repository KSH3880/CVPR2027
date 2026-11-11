# 공통 Stage2 PUSH와 coupled

[환경 YAML](../../tokenhsi/data/cfg/multi_agent/stage2_coupled_push.yaml)과 [학습 YAML](../../tokenhsi/data/cfg/train/rlg/stage2_coupled_push.yaml)을 사용한다. 실행 명령은 [config](../config.md), 전이·resume 옵션은 [USAGE](../USAGE.md)에 있다.

## 같은 모델로 따로 학습

`stage2_before_push`와 `stage2_coupled_push`는 **같은 `Stage2PushBuilder`**를 사용한다. 둘 다4명·1024환경·8개 관측 물체 토큰·1200step·loco 시작이다. 각 실험은 Stage1 PUSH21k에서 별도로 전이하고 출력·optimizer·checkpoint를 분리한다. 기존4과제 BEFORE 설정은 기존 모델을 유지하며 시간만1200step으로 맞춘다.

| 공통 항목 | 설정 |
| --- | --- |
| task embedding | SIT/CLIMB/AT/ON_TOP/PUSH 5종,64차원 |
| 기본 정책 | 기존 GTA·task bias·Action MLP·action head |
| 관측 | task packet5N + 그룹 ID N, 기본1272차원 |
| 사람 coordination | NONE/SELF/BEFORE/COUPLED 4종 relation, actor/critic 각4층 |
| AMP | carry/sit/climb/push 4family,1330차원 |
| 전이 | 기존 Stage1 PUSH tensor165개 복사, coordination의 Linear bias0·출력 identity 초기화 |
| 동결 | actor encoder와 입력 RMS |
| 학습 | coordination·action head·critic·AMP 등 기존 Stage2 범위 |

BEFORE 의존 관계는 후행→선행 방향이고, COUPLED는 같은 공동 PUSH 그룹의 다른 사람끼리 대칭이다. 그룹 ID는 다른 사람과 같은 그룹인지 판단하는 데만 쓰며 숫자 자체를 신경망에 embedding하지 않는다. BEFORE의 object alias·shared-edge fusion은 공통 encoder 경로에서 처리한다. alias 변환은 config 플래그와 무관하게 원본 task packet의 BEFORE 의존 관계를 보고 적용한다. coupled·독립 packet은 그대로 통과한다. 복제 물체 토큰의 feature/pose와 물리 매핑·보상/coordination에 쓰는 원본 packet은 유지하며 encoder에 전달하는 복사본만 변환한다. 모델 tensor 이름·크기는 두 실험에서 동일하고, relation/장면/보상이 다르다.

기존 Stage1 PUSH PTH는 전이 원본이다. 새 Stage2 PTH와 같은 방식으로 직접 resume하지 않는다. 기존5N suffix/3종 사람 relation BEFORE PUSH 검사 저장본도 새 모델에 resume하지 않는다. 앞으로 두 환경을 섞는 학습을 위한 모델 구조는 같지만, 서로 다른 환경 checkpoint의 resume 계약은 계속 구분한다.

## 샘플링

남은 사람을 순차적으로 선택하고 0.8 확률로2명부터 남은 인원까지 균등한 크기의 공동 그룹을 만든다. 공동 그룹은 전원 PUSH다. 독립 역할은 다음 비율로 고르며, YAML과 실제 샘플러·AMP expert 혼합이 같은 설정을 읽는다.

| 독립 행동 | 비율 |
| --- | --- |
| AT | 25% |
| ON_TOP | 20% |
| SIT | 10% |
| CLIMB | 15% |
| PUSH | 30% |

이 확률은 독립 역할 기준이다. 그룹 인원은 모두 PUSH이므로 전체 사람의 PUSH 비율은 더 높다.4명·0.8 그룹 확률에서 독립 인원의 기대 비율은 약17.73%, 전체 PUSH 기대 비율은 약87.59%다. BEFORE 쌍 샘플링 확률과 혼동하지 않는다.

## 상자·초기 배치

기본 PUSH는 Stage1 PUSH의167종·밀도3·마찰0.4다. 공동 상자는 기본 상자의 local X 폭을 참여 인원만큼 늘리고 Y/Z는 유지한다. 같은 밀도이므로 질량도 같은 배수로 증가한다. 사람별 관측에는 공동 상자를 폭 방향의 동일한 구역으로 나눠 보여주고, 실제 물리 actor와 목표는 그룹에서 공유한다. 구역 순서는 reset마다 섞는다.

24개 물리 후보(일반 AT/SIT/CLIMB/받침16개, 독립 PUSH4개,2인용2개·3인용1개·4인용1개)를 만들고8개 논리 토큰에 배정한다. 공동 상자의 논리 토큰은 사람별 구역이지만 물리 actor는 하나다. 사용하지 않는 후보는1000m 이상 멀리 둔다.

인원 상한 없이 N≥1·논리 물체 M≥N을 지원한다. 기본 학습은4명·8토큰이며 기존 PTH로 다른 N/M의 추론을 할 수 있다. 물리 후보는 일반4N개·k인용 상자별 floor(N/k)개(k=1…N)·추가 여분 max(M−2N,0)개다. 전원 공동일 때 상자 local X 폭·질량을 기본 상자의 N배로 늘리고 깊이·높이·밀도·인당 구역 크기·개인 목표 규칙을 유지한다. 모델/RMS tensor를 변환하지 않으며, 추론 계약 비교에서 양쪽 인원·논리 토큰·물리 후보 수만 정규화한다. 나머지 저장 설정과 resume의 N/M은 일치해야 한다. [임의 인원 관찰 명령](../config.md#추론-옵션). 공간 채점의 인원별 집계도 N명까지 기록한다.

앞 N개 논리 토큰은 사람별 주 물체/공동 구역이고, 나머지 M−N개는 ON_TOP 받침과 distractor다. ON_TOP 인원이 여분 수를 넘는 장면만 그룹·과제를 다시 뽑으며 이미 유효한 장면은 보존한다. M<2N이면 필요한 받침을 앞 여분 슬롯부터 압축 배정하고 관측/과제 그래프/물리 actor를 같은 매핑으로 연결한다. 받침으로 쓰지 않는 물리 후보는 멀리 둔다. 전원 ON_TOP 고정 preset은 M<2N에서 가능한 그래프가 없으므로 명시적으로 오류를 낸다. 기존4명·8토큰 샘플링은 유지한다.

기본 `stage2_coupled_push`의 PUSH 출발 상자는 모든 참여자에서2–10m, 목표는 실제 상자 중심에서2–10m다. 두 거리는 수평 XY 기준이다. 거리 설정이 없는 과거 coupled 저장본도 기존2–10m로 복원한다.

거리 변형 `stage2_coupled_push_goal5`는 공동 PUSH의 사람→상자2–10m를 유지하고 상자→목표만2–5m로 줄인다. `stage2JointPush.source_distance`·`goal_distance`로 기록하며 독립 PUSH는 두 거리 모두2–10m다. 기본 전이는 `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`이고, 나머지 환경·학습 설정은 기본 coupled와 같다. 전용 YAML·train/test/VNC/eval과 별도 출력 `output/stage2_coupled_push_goal5/`를 사용하며 기본 coupled 저장본과 resume/추론을 혼용하지 않는다.

공동 참여자는 상자 뒤쪽에 두고, 공동 목표는 local +Y 정면 직선으로 고정한다. 독립 PUSH는 local +Y ±45도다. 사람·다른 상자·이동 경로와 충돌하지 않도록 후보를 거부한다. 나머지 독립 행동 크기·배치는 Stage1의 기존 규칙을 따른다. 시작은 전부 loco이며 PUSH는 `push_loco` 모션을 쓴다. 접촉 RSI bank는 사용하지 않는다.

충돌 경로는 관측용 가상 구역으로 변환하지 않고 실제 물리 상자에서 직접 구성한다. distractor 하나의 재시도 동안 고정된 다른 상자 경로·크기·사람 위치를 재사용하고, 다음 distractor와 reset에서는 새로 구성한다. 단독 PUSH 전용 접근/이동 경로 검사는 실제 단독 PUSH 행만 계산하며, 다른 사람과 비PUSH 상자도 장애물로 유지한다. 후보 생성 순서·거리·충돌 여유·재시도 한도는 유지한다.

## 보상과 평가

CONTACT+AT 기본 보상과 가중치는 Stage1 PUSH를 따른다. 공동 CONTACT는 각자 담당 구역의 뒤쪽 면을 기준으로 계산한다. 개인 접촉 성공은 양손/상자 힘 조건도 사용한다.

공동 AT gate는 전원이 상자 뒤쪽 면0.18m 이내에 양손을 둔 현재 상태에서 열린다. `goal_saturation: true`는 실제 상자 중심이 목표에 도달했을 때 이 gate를 열어 접촉이 풀려도 AT 성공/포화 보상을 유지한다. 성공 이력으로 gate를 영구 고정하는 방식은 아니다.

일반 평가는 학습 장면과 인원을 유지하는1200step·512환경×3seed다. PUSH 성공은 **구역 중심이 아니라 실제 전체 상자 중심**과 공유 목표의3D 거리≤0.20m를 사용한다. 점/영역 리포트에서 PUSH 값은 같다. 개인 낙상 후 나머지는 계속 수행한다. 학습에서는 기존처럼 한 명 낙상 시 환경 전체를 reset한다. [평가 규약](evaluation.md).

원본 이식 대상은 GitHub `stage1_2_jointpush_st`의 `stage2_jointpush_goal_saturation_density3_cone90_coupled_straight`다.1200step·직선 목표·그룹 gate·goal saturation은 유지하고, 독립 행동 비율·AMP 비율을 현재 Stage1 PUSH 설정에 연결했다. 원격의 별도 Stage2 모델/추가 adapter/RSI 스택은 가져오지 않았다.


## 개인 목표·공통 4면 실험

`stage2_coupled_push_individual_goal`은 goal5의 공동 거리·26k 전이·4명/1024환경/8토큰을 유지한다. `stage2JointPush`의 `goal_mode: individual`, `contact_face: group_four`, `gate_alignment: false`를 함께 사용하는 별도 계약이다. 기존 coupled/goal5의 동작은 유지한다.

이 실험의 `coupled_probability`는0.5다. 그룹 생성 시 공동/단독 선택 확률을50/50으로 설정하며,4명 기준 기대 인원 비율은 공동63.54%·단독36.46%다. AMP expert 혼합도 같은 확률을 따른다. 기존0.8 저장본과 새 학습 설정의 resume 계약은 다르다.

원본 개인 목표 컨피그의 단독 사람→상자는 SIT1–5m·CLIMB/CARRY1–10m·PUSH2–10m이며 단독 PUSH 목표 거리는2–10m다. 단독 거리 축소는 아래 별도 solo5 컨피그에서만 적용한다.

- 그룹당 2–5m를 한 번 뽑고 초기 상자 로컬 +Y 이동 벡터를 각 분할 토큰의 초기 중심에 더한다. 목표는 고정되며 AT 연결·관측·보상은 각 토큰과 개인 목표를 사용한다. 공유 중심 목표는 별도 보관하여 물리 경로·충돌 검사에 사용한다.
- 접촉 면은 실제 공유 중심→공유 중심 목표 방향과 안쪽 밀기 방향이60° 이내인 실제 외곽 네 면에서 선택한다. 각 후보 면을 기존 구역 번호 순서로 균등 분할하고, 담당 구역까지 양손 중 최대 거리의 그룹 최댓값이 가장 작은 면을 공통 선택한다. ±Y 면은 로컬 +X 순서, ±X 면은 로컬 +Y 순서로 나눈다. 가상의 내부 경계면은 후보가 아니다.
- CONTACT는 선택 면의 개인 담당 구역까지 XY 거리·접촉력을 사용한다. 전원 gate는 같은 선택 면 전체에 양손이0.18m 이내이고 상자 중심 기준 바깥쪽 반공간에 있는지 검사한다. 기존처럼 gate 자체는 구역별 배정·접촉력을 요구하지 않는다. 별도의 `aligned` 검사는 없다. 네 면 선택은 단독과 같은 yaw 기준이며 gate는 면의 가로 범위와 상자 높이도 검사한다.
- 개인 토큰→개인 목표의3D 거리≤0.2m·Z 오차≤0.001m면 그 사람의 gate를 해제하고 기존 AT→CONTACT 포화 로직을 적용한다. 이탈하면 현재 조건으로 재판정한다. 목표 관측 토큰/분할 축은 접촉 면이 바뀌어도 바꾸지 않는다.
- 정식 평가는 개인 토큰→목표 거리≤0.2m를 사용한다. `coupled_scores`는2/3/4명별 개인 ever/final과 전원 동시 성공의 ever/final을 기록한다. 서로 다른 시점의 개인 성공을 합쳐 공동 성공으로 세지 않는다. 이는 접촉 gate와 독립적인 공간 평가다.
- 개인 목표 원본과 solo5의 eval은 `shared_center_scores`도 기록한다. 실제 공유 상자 중심→고정 공유 중심 목표의3D 거리≤0.2m를2/3/4명 그룹별 ever/final로 집계하며 구성원 중 최소1명이 유효한 frame만 인정한다. 전원 개인 목표 동시 성공과 별도 지표이며 회전·손 접촉·보상 gate 조건은 추가하지 않는다. 실제 물리 중심/공유 목표와 성공 mask를 trace에 저장하고 CPU 재채점으로 검증한다. 이전 trace는 같은 물리 상자의 균등 분할 토큰·개인 목표 평균으로 중심/목표를 복원한다.
- 단독 행동의 보상·접촉 규칙은 유지한다. 개인 목표 변환은 공동 구성원에게만 적용하여 단독 CARRY AT 목표 높이와 받침을 보존한다. 출력은 `output/stage2_coupled_push_individual_goal/`이며 학습·viewer·VNC·eval 전용 wrapper와 저장본을 사용한다.


## 개인 목표 solo5 거리 변형

`stage2_coupled_push_individual_goal_solo5`는 개인 목표 원본의 공동 확률0.5·공동 접근2–10m/이동2–5m·26k 전이를 유지한다.

단독 거리는 `stage2JointPush.solo_source_max_distance: 5.0`·`solo_goal_max_distance: 5.0`으로 제한한다. 이 값은 공통 Stage1 기준값보다 우선하며 다른 실험에는 적용하지 않는다. 단독 SIT/CLIMB/CARRY AT/ON_TOP의 사람→대상 상자는1–5m, 단독 PUSH는 사람→상자와 상자→목표 모두2–5m다. CARRY AT 목표·ON_TOP 받침의 XY 샘플링은 환경 원점 기준±4.5m 정사각형과 기존 제외 조건을 유지한다. 공동은 사람→실제 상자 중심2–10m·각 분할 토큰→개인 목표2–5m다. 거리는 XY 기준이다.

전용 환경/학습 YAML과 train/test/VNC/eval wrapper를 사용한다. 출력은 `output/stage2_coupled_push_individual_goal_solo5/`이며 원본 개인 목표 저장본과 계약을 혼용하지 않는다. AT 목표 높이 보존 버그 수정은 두 컨피그에 공통 적용한다.

## 개인 목표 solo5 carry5 거리 변형

`stage2_coupled_push_individual_goal_solo5_carry5`는 solo5의 단독 접근1–5m(PUSH2–5m)·단독 PUSH 이동2–5m·공동 접근2–10m/이동2–5m·확률0.5·26k 전이·모델/보상을 유지한다. `stage2JointPush.solo_carry_goal_distance: [1.0, 5.0]`만 추가하여 단독 AT의 목표점과 ON_TOP 받침 XY를 운반 상자 기준 거리 U(1,5)m·방향 U(0,2π)로 뽑는다. 기존 환경 원점±4.5m 정사각형 제한은 이 설정의 두 목표에 적용하지 않는다. 사람↔목표 최소1m와 높이·충돌·경로·장면 재시도 조건은 유지하며, 유효성 검사 때문에 최종 채택 분포는 후보의 균등 분포와 달라질 수 있다. 기존 설정은 해당 키가 없어 기존 샘플러/RNG를 그대로 사용한다. 전용 출력은 `output/stage2_coupled_push_individual_goal_solo5_carry5/`이고 기존 PTH 계약과 구분한다. [실행 명령](../config.md).

## carry5 지면 운반 변형

`stage2_coupled_push_individual_goal_solo5_carry5_ground`는 carry5의 `box.reset.platformMode: ground_only`, `randomHeight: false`, `randomHeightProb: 0.0`만 변경한 별도 학습 설정이다. 단독 AT·ON_TOP의 운반 상자는 지면에서 시작하고 AT 목표도 상자 바닥이 지면에 닿는 높이다. 원본/목표 선반 actor는 각각 Z−20m/−19.5m에 두어 장면에서 제외하며 ON_TOP의 실제 받침 상자는 유지한다. 기존 BEFORE의 ground_only 경로를 coupled에서도 사용하므로 elevated carry5의 원본5cm clearance도 추가하지 않는다.

운반 거리1–5m·사람 접근 거리·공동 확률/거리/물성·보상·정책·1024환경·Stage1 PUSH26k 전이를 유지한다. 새 이름의 환경/학습 YAML·train/test/VNC/eval과 출력 폴더를 사용한다. 저장된 높이 계약이 달라 기존 carry5/solo5 PTH와 resume/추론 설정을 혼용하지 않는다. [학습·viewer·VNC 명령](../config.md).

## 사람 attention 관계별 개수 보정

solo5 기반 `stage2_coupled_push_individual_goal_solo5_countnorm_coupled`·`stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none`은 기존 샘플링·보상·물성·Stage1 PUSH26k 전이·학습 설정을 유지하고 사람 attention의 관계별 개수 보정만 추가한다. 환경 `relationGraph.human_attention_count_normalization`에 각각 `coupled`·`coupled_none`을 저장한다. GTA/object attention에는 적용하지 않는다.

각 query i의 관계 r인 유효 key 개수를 K(i,r)로 세고 선택한 관계의 점수만 `QK/√d + bias(r) − log K(i,r)`로 만든다. 전자는 COUPLED만, 후자는 COUPLED/NONE을 따로 보정한다. SELF/BEFORE는 유지하며 COUPLED 수에는 자기 자신을 포함하지 않는다. masked key는 개수에서 제외하고 존재하지 않는 관계에는 보정할 항목이 없다. 학습 범위 내2/3/4명부터 동일한 규칙을 적용하고 actor/critic의4개 사람 attention 층과 추론/resume에서 유지한다.

동일 관계 안의 점수 차이와 선호 비율은 유지된다. 관계별 지수 가중치의 합을 개수 대신 평균으로 비교하므로 같은 상태/점수 분포에서 key 복제로 관계 비중이 커지는 효과를 보정한다. 상태가 다른 상대를 추가했을 때도 관계 비중이나 액션이 같음을 보장하지 않으며, 마지막 미준비 상대 신호의 희석이나 전원 준비 판단을 별도로 해결하는 것은 아니다.

기본값 `none`은 기존 계산이며 기존 설정/YAML에는 보정 키를 넣지 않는다. 새 학습 파라미터는 없고 tensor 이름/형상도 동일하지만, 동작 모드는 저장 env와 `coupled_push_contract.graph`로 구분하여 기존 solo5 및 다른 보정 모드의 PTH 혼용을 거부한다. [실행 명령](../config.md), [CPU 검증](../../tokenhsi/tests/test_human_attention_count_normalization.py).

## 최대1명 COUPLED 선택 학습

solo5의 `keep1`·`shrink1` 설정은 `relationGraph.coupled_peer_selection`으로 구분한다. 공동2명부터 임의 인원까지 각 query의 미접촉 상대 중1명을 선택하고, 선택한 상대가 접촉하면 다음 미접촉 상대로 교체한다. 유지 가능한 선택은 그대로 두며 접촉을 잃은 상대도 후보에 포함한다. keep1은 미접촉 상대가 없으면 접촉한 상대1명을 유지하고, shrink1은 선택을 비운다. 단독 그룹은 두 모드 모두 상대가 없다. 양손 기하 `on_face`와 별도 RNG를 사용하며 부분 reset은 해당 환경의 선택만 초기화한다.

선택되지 않은 COUPLED는 NONE으로 바꾼다. NONE/SELF/BEFORE와 실제 공동 그룹·공유 상자·개인 목표·native gate/보상은 유지한다. NONE 상대도 attention에 참여하므로 COUPLED가0명이어도 SELF만 보는 것은 아니다. 학습과 추론의 actor/critic4층에 같은 선택을 사용하며 관계별 개수 보정은 추가하지 않는다.

관측 배치는 `[scene | task packet 5N | selected peer IDs N | group IDs N]`이다. 선택 ID는0부터N−1, 상대 없음은−1이며 정규화를 통과시킨다. 모델은 ID를 추가 특징으로 임베딩하지 않고 관계 분류에만 사용한다. PPO rollout/미니배치/가치 bootstrap은 저장된 ID를 사용하며 모델 forward에서 재선택하지 않는다. 기존6N suffix는 기본 설정에서 유지하고 두 설정만7N이다.4명/8토큰 관측은1272→1276차원이며 모델 파라미터 이름/형상과 Stage1 PUSH26k 전이는 유지한다. 모드는 저장 checkpoint 계약에 포함해 기존 solo5 및 다른 선택 모드와의 혼용을 거부한다.

기존 `COUPLED_PEERS=keep3|shrink3`는 viewer의 actor 전용 선택이며 5명 이상 그룹에 적용한다. 현재는 이 경우도 미선택 상대를 NONE으로 바꾼다. 과거 마스킹 평가는 changelog의 당시 기록으로 구분한다. 두 학습 설정에는 이 viewer override를 중첩하지 않는다.

## ground 관계 fusion A/B

`stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg`(A)와 `_relfusion_zerobg`(B)는 ground의 환경·보상·거리·물성·4명/1024환경/8물체를 유지하고 Stage1 PUSH26k에서 새로 전이한다. actor encoder는 동결하고 coordination·action head 및 critic은 학습한다. actor/critic의 새 파라미터는 독립이다.

- `relationGraph.human_relation_fusion`은 A=`learnedbg`, B=`zerobg`다. 기존 기본값 `none`은 기존 계산·state_dict를 유지한다. countnorm 및 peer selection과의 조합은 지원하지 않는다.
- 각 coordination은 4층·2헤드·Pre-LN·일반 FFN64→128→64다. COUPLED 임베딩 하나(64차원)와 fusion MLP128→64→64를 네 층이 공유한다. 각 층/헤드의 투영과 임베딩 내적으로 COUPLED bias를 만든다.
- LayerNorm→QKV→softmax(QK/√32+bias)→V집계→헤드 결합→COUPLED residual fusion→output projection→residual→Pre-LN FFN 순서다. 단독은 fusion 추가분만 0이고 나머지 coordination을 통과한다. 인원별 임베딩은 없고 전원 접촉 후에도 같은 관계를 유지한다.
- A/B의 유일한 구조 차이는 SELF/NONE 배경 bias다. A는 레이어·헤드별 lookup, B는 상수0이다. 어느 쪽도 SELF/NONE을 마스킹하지 않는다. BEFORE는 이번 학습에 없고 새 경로에서 bias/fusion을 추가하지 않는다. 이후 BEFORE 지원은 별도 확장이다.
- bias 투영·배경 lookup·fusion 마지막 층·attention output projection·FFN 마지막 층은0 초기화한다. 초기 coordination은 identity이며 output projection 학습 후 fusion gradient가 열린다.
- 구조 선택은 저장된 relationGraph 및 coupled checkpoint 계약에 포함된다. 기존 ground 또는 A/B 사이 resume은 거부하며 새 구조의 저장/복원만 지원한다.

임베딩은 공동 소속을 전달하지만 contact 준비 상태는 attention 피처에서 배워야 한다. 단독 성능 보존과 인원 확장을 보장하지 않는다. GPU 검증 및 결과는 changelog와 `output_etc/check/relfusion_ab_20261103/`를 따른다.
