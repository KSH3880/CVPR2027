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

PUSH 출발 상자는 각 참여자에서2–10m, 목표는 상자에서2–10m다. 공동 참여자는 상자 뒤쪽에 두고, 공동 목표는 local +Y 정면 직선으로 고정한다. 독립 PUSH는 local +Y ±45도다. 사람·다른 상자·이동 경로와 충돌하지 않도록 후보를 거부한다. 나머지 독립 행동 크기·배치는 Stage1의 기존 규칙을 따른다. 시작은 전부 loco이며 PUSH는 `push_loco` 모션을 쓴다. 접촉 RSI bank는 사용하지 않는다.

## 보상과 평가

CONTACT+AT 기본 보상과 가중치는 Stage1 PUSH를 따른다. 공동 CONTACT는 각자 담당 구역의 뒤쪽 면을 기준으로 계산한다. 개인 접촉 성공은 양손/상자 힘 조건도 사용한다.

공동 AT gate는 전원이 상자 뒤쪽 면0.18m 이내에 양손을 둔 현재 상태에서 열린다. `goal_saturation: true`는 실제 상자 중심이 목표에 도달했을 때 이 gate를 열어 접촉이 풀려도 AT 성공/포화 보상을 유지한다. 성공 이력으로 gate를 영구 고정하는 방식은 아니다.

일반 평가는 학습 장면과 인원을 유지하는1200step·512환경×3seed다. PUSH 성공은 **구역 중심이 아니라 실제 전체 상자 중심**과 공유 목표의3D 거리≤0.20m를 사용한다. 점/영역 리포트에서 PUSH 값은 같다. 개인 낙상 후 나머지는 계속 수행한다. 학습에서는 기존처럼 한 명 낙상 시 환경 전체를 reset한다. [평가 규약](evaluation.md).

원본 이식 대상은 GitHub `stage1_2_jointpush_st`의 `stage2_jointpush_goal_saturation_density3_cone90_coupled_straight`다.1200step·직선 목표·그룹 gate·goal saturation은 유지하고, 독립 행동 비율·AMP 비율을 현재 Stage1 PUSH 설정에 연결했다. 원격의 별도 Stage2 모델/추가 adapter/RSI 스택은 가져오지 않았다.

## 구역별 고정 목표 변형

`stage2_coupled_push_region_goals`는 Stage1 PUSH21k·4명/1024환경·1200step·물성·과제/AMP/PPO·초기 배치·밀기 거리2~10m를 기존 coupled와 같게 유지한다. `stage2JointPush.goal_representation: initial_region_offset_v1`만 추가하며 전용 train/test/VNC/eval·출력·checkpoint 계약으로 분리한다. 원래 coupled checkpoint를 직접 resume하지 않는다.

reset에서 물리 상자 초기 중심을 c₀, 사람 i의 구역 초기 중심을 cᵢ₀, 기존 공통 중심 목표를 g라 할 때 목표 gᵢ = g + (cᵢ₀ − c₀)를 고정한다. 따라서 gᵢ − cᵢ₀ = g − c₀이며, 모든 구역의 밀기 방향과 거리는 원래와 같다. 목표 offset은 초기 yaw와 구역 순서로 계산하고 현재 상자의 회전에는 따라가지 않는다. 물리 상자는 하나를 유지한다.

PUSH task packet은 각자 goal token을 가리키고 사람 간 COUPLED 관계는 그룹으로 유지한다. GTA goal pose·AT state/progress/성공·goal saturation·marker는 각자의 고정 목표와 현재 구역 중심을 사용한다. 전원 양손18cm 접촉 gate와 CONTACT 계산은 기존과 같다. 경로/충돌 검사는 전체 물리 상자와 원래 중심 목표를 사용해 기존 sampler의 난수 소비와 밀기 경로를 유지한다. 단독 PUSH와 다른 독립 과제는 그대로다.

이 변형의 정식 PUSH 평가는 각 구역 중심과 대응 목표의 거리20cm를 채점한다. 기존 전체 상자 중심 평가와 구분하며, 목표가 고정되어 상자가 목표 중심에서 회전하면 구역 목표 오차도 증가한다. 학습의 높이 허용오차1mm와 정식 평가의 기존 점 거리 기준은 각각 유지한다.
