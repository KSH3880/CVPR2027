# 4명 랜덤 공동/단독 운반 구현 인수인계

작성일: 2026-10-16. 작업 저장소: `/home/hwanhee/ksh/stage2_coupled`.

이 문서는 **구현 요청과 설계 명세**다. 아래 4명 설정·스크립트는 아직 구현하거나 검증하지 않았다. 이 문서를 작성하면서 기존 실행 코드와 설정은 변경하지 않았다.

## 1. 다음 Codex가 할 일

이 저장소는 `/home/hwanhee/ksh/stage2_jointcarry`를 정리한 같은 코드 계열이다. 과거 실험을 다시 가져오는 작업이 아니다. 현재 저장소의 두 설정을 기준으로 **4명 랜덤 팀 구성 학습 경로 두 개**를 만들고 연결·검증한다.

| 기준 설정 | 유지할 차이 | 권장 신규 이름 |
| --- | --- | --- |
| `stage2_at20` | 실제 payload 중심을 관측하는 Human SA4 | `stage2_at20_4agent` |
| `stage2_grasp_at20` | 공동일 때 사람별 파지 앵커를 관측하는 SA4 GraspView | `stage2_grasp_at20_4agent` |

신규 이름은 제안이다. 기존 2명 설정을 보존하고 4명 비교 설정을 별도로 추가하는 방식을 권장한다. 두 실험에서 **GraspView 유무 외의 샘플러·보상·데이터·학습 조건은 같게** 한다.

중요한 용어 정정: **AT20은 성공 보상 가중치 20이 아니라 AT 성공반경 20cm**다. 실제 YAML은 `success.at_distance_tolerance: 0.2`, `at_z_tolerance: 0.001`이다. 박스 중심과 목표점의 **3차원 거리 ≤0.2m AND 높이 차이 절댓값 ≤0.001m**를 공동·단독 AT에 적용한다. `state_reward_weight`, `progress_reward_weight`, `success_reward_weight`는 각각 **0.2**다. ON_TOP 성공조건은 변경하지 않는다.

두 실험 모두 같은 Stage1 checkpoint에서 새 Stage2 학습을 시작한다. 여기서 스크래치는 Stage1 전이를 하지 않는다는 뜻이 아니다. Stage1 encoder/head 초기화는 유지하고 기존 Stage2 학습 상태를 이어받지 않는다는 뜻이다. 현재 가이드는 CarryDistill30000을 사용하며, 실제 경로는 사용자가 지정한 `STAGE1_CHECKPOINT`를 확인한다.

본학습 자동 실행·기존 학습 종료는 요청 범위로 추정하지 않는다. 전용 명령과 짧은 검증 결과를 제공한다.

## 2. 목적과 범위

기존 2명 공동 장면에는 SELF/COUPLED만 있고 NONE은 없다. 2명 단독 장면에는 SELF/NONE만 있다. 4명 장면에서는 공동 에이전트도 SELF·COUPLED·NONE을 동시에 경험하도록 한다.

예: A/B 공동, C/D 공동이면 A의 관계는 `[SELF, COUPLED, NONE, NONE]`이다. A/B 공동, C와 D 단독이어도 A의 관계는 같다. C가 단독이면 C 자신만 SELF이고 나머지는 NONE이다.

이것은 관련 상대를 구분할 경험을 늘리는 실험이다. QK가 반드시 SELF/COUPLED에 집중하거나 공동집기가 해결된다고 보장하지 않는다. 높은 masked 이전 NONE QK만으로 오류라고 판단하지 않는다.

이번에는 다음을 추가하지 않는다.

- BEFORE 관계, 3명 이상 공동 운반, 한 사람의 여러 동시 과제.
- NONE hard mask, QK identity 초기화, 별도 attention 지도학습.
- Move/heading 보상, TeamHold 보상, 새 AMP 가중치나 RSI focus 스케줄.
- GraspView를 이유로 실제 상자 형상이나 기존 반대 앵커 보상 변경.

미래 BEFORE 확장을 막는 고정 사람 ID 규칙은 피하되, 이번 과제에 필요 없는 일반 task planner는 만들지 않는다.

## 3. 팀 샘플링: 고정 장면 비율이 아닌 순차 랜덤 분할

사람은 4명, 공동팀 크기는 2명이다. 추천·논의 기준 확률은 **p=0.8**이다.

1. episode reset에서 4명의 ID 순서를 무작위로 섞는다.
2. 아직 배정되지 않은 첫 사람을 꺼낸다.
3. 다른 미배정 사람이 남았으면 확률 p로 공동팀을 만든다. 상대는 남은 사람 중 균등하게 선택한다.
4. 공동팀을 만들지 않으면 그 사람은 단독이다. 마지막 한 명도 단독이다.
5. 전원이 배정될 때까지 반복한다.
6. 그룹마다 AT/ON_TOP을 독립적으로 50/50 선택한다. 공동팀의 두 사람은 동일 과제·동일 payload·동일 target을 공유한다. 단독은 각자 독립 과제를 가진다.

사람들을 물리 슬롯에 고정한 뒤 H0/H1만 공동으로 만드는 방식이 아니다. 관계와 과제를 먼저 정하고, 적합한 물체 슬롯에 배정한 뒤 초기 배치와 RSI를 만든다. 관계 배정은 episode 동안 유지한다.

의사코드:

```text
remaining = random_permutation([0, 1, 2, 3])
groups = []
while remaining:
    a = pop_first(remaining)
    if remaining and uniform(0, 1) < 0.8:
        b = remove_uniform_random_member(remaining)
        groups.append([a, b])
    else:
        groups.append([a])
for group in groups:
    assign_one_carry_task(group, AT_or_ON_TOP_50_50)
```

위 의사코드는 동작 명세이며, 실제 구현은 GPU 배치 처리에 적합한 형태로 작성한다.

### 실제 구성 비율

위 순차 방식에서 공동 pair 수를 K라고 하면:

\[
P(K=0)=(1-p)^3,\quad P(K=2)=p^2,\quad P(K=1)=1-(1-p)^3-p^2.
\]

| 장면 구성 | p=0.8일 때 reset 샘플 비율 | 공동/단독 사람 수 |
| --- | ---: | --- |
| 공동 2팀 | 64% | 4 / 0 |
| 공동 1팀 + 단독 2명 | 35.2% | 2 / 2 |
| 단독 4명 | 0.8% | 0 / 4 |

사람 기준 공동 비율은 `64% + 35.2%/2 = 81.6%`, 단독은 18.4%다. **p=80%가 공동 환경 80%라는 뜻은 아니다.** episode 길이가 다르면 실제 timestep 기준 점유율은 달라질 수 있으므로 reset 구성 비율과 timestep 비율을 구분해서 기록한다.

## 4. 현재 코드에서 확인한 변경 지점

아래는 2026-10-16 이 저장소의 실제 코드 기준이다. 구현 전 AGENTS.md와 최신 changelog를 먼저 확인하고, 변경된 부분은 다시 대조한다.

| 파일 | 확인한 제약 / 필요한 작업 |
| --- | --- |
| `tokenhsi/utils/mixed_carry_spec.py` | `validate_sampler`가 2명/4물체·40/40/20을 강제. `joint_scenes`가 owner 0/1 payload만 비교. 새 그룹 샘플러와 사람별 소속/상대/물체 매핑 필요 |
| `tokenhsi/utils/joint_carry_spec.py` | 기존 환경·단독20% 계약 검증. 신규 샘플러 계약을 명시적으로 구분 |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py` | `_joint_env_mask`로 생성 때 환경 전체의 공동/단독·상자 크기를 나눔. 물체 생성/논리 슬롯/reset/관측/actor index 확장 필요 |
| `tokenhsi/env/tasks/multi_agent/edge_ontop_task.py` | graph 생성에 환경 단위 joint mask 전달, 장면 단위 진단. 그룹 단위로 확장 |
| `tokenhsi/env/tasks/multi_agent/joint_carry_reset.py` | 환경당 공동 pair 1개, `source_frame`/`skill_index`도 환경당 1개, payload0/support2 및 두 사람 직접 대입. 그룹별 gather/scatter 필요 |
| `tokenhsi/env/tasks/multi_agent/joint_carry_reward.py` | payload0의 두 사람 앵커 점수, 환경당 `coupled_holding` 1개, packet reshape 2명. 그룹별 geometry/gate/CPA 필요 |
| `tokenhsi/env/tasks/multi_agent/joint_carry_platform.py` | 환경당 마지막 actor 하나가 받침. 최대 공동 2팀 각각의 받침 상태/index 필요 |
| `tokenhsi/env/tasks/multi_agent/edge_context_task.py` | reset 뒤 grasp side 계산·CPA release 적용 경로. 새 pair 단위 의미와 연결 |
| `tokenhsi/learning/multi_agent/grasp_view.py` | 2명/4물체/634 관측의 상수 slice와 2개 view. 가변 M/O 및 그룹별 view로 확장 |
| `tokenhsi/learning/multi_agent/amp_network_builder_ma.py` | Stage1 호환 encoder와 SA/head 입력. 관측 길이/사람 수/GraspView 호출 연결 확인 |
| `tokenhsi/learning/multi_agent/human_coordination.py` | 관계 행렬·human 순열. 4명 관계 및 action 복원 검증 |
| `tokenhsi/learning/multi_agent/ma_agent.py`, `ma_players.py` | GraspView `num_views=2` 고정. PPO packing·AMP flatten·RMS·checkpoint 계약 확인 |
| `tokenhsi/learning/multi_agent/scene_normalizer.py`, `stage2_transfer.py` | 물체 수/view 수 변화의 정규화·Stage1 초기화 계약 검증 |
| `tokenhsi/utils/task_role_spec.py`, `relation_task_spec.py` | packet/entity ID·모델 저장 계약과 신규 설정 일치 확인 |
| `tokenhsi/utils/size_rsi.py`, `tokenhsi/env/tasks/multi_agent/size_rsi_cache.py` | 단독용 asset/skill profile 및 캐시를 새 물체 배정에 맞게 사용 |

**CLI 숫자만 `4 2048 ...`로 바꾸면 완료되는 작업이 아니다.** 다른 저장소의 6명 viewer 지원도 이 정리 저장소의 4명 학습/reset 지원을 증명하지 않는다. 예전 코드를 통째로 복사하지 말고 필요한 동작만 검증해서 반영한다.

## 5. 물체·장면 배정: 구현 전에 확정할 핵심

현재 단독은 Stage1 크기·size-aware RSI, 공동은 고정 긴 payload를 쓴다. 이 구분을 유지한다.

- 공동 payload: `[0.52, 0.80, 0.40]m`.
- 공동 ON_TOP 받침 상자: `[0.52, 0.80, 0.30]m`.
- 단독 payload/support: 현재 Stage1 size 샘플러·캐시 계약 유지. 편의상 공동 상자 크기를 단독에 쓰지 않는다.
- 각 그룹이 payload와 목표를 소유하고 다른 그룹과 의도치 않게 공유하지 않는다.
- 전원 단독 ON_TOP이면 payload 4개와 support 4개가 필요하다. 두 공동 ON_TOP이면 긴 payload 2개와 그 support 2개가 필요하다.

Isaac Gym asset 크기는 생성 때 정해지므로 reset마다 물체 크기를 바꾸는 것으로 구현하지 않는다. episode마다 랜덤 구성을 바꾸려면 필요한 물리 asset을 미리 만들고 논리 역할에 배정하는 방식을 검토한다.

**물리 actor 개수와 관측에 노출할 object 슬롯 개수는 아직 확정하지 않았다.** 위 최대 동시 사용량 8개가 곧 서로 다른 형상까지 포함한 전체 사전 생성 actor 수라는 뜻은 아니다. 적합한 asset bank, 비활성 actor 위치, 논리 객체 매핑, 관측 포함/제외 방식을 기존 코드에 맞게 설계해 설명한 뒤 구현한다. 비활성 객체를 먼 곳에 두는 것만으로 관측·GTA·RMS 문제가 모두 해결됐다고 가정하지 않는다.

현재 `envSpacing: 5`와 2명 배치 그대로 4명/여러 payload를 넣으면 초기 관통 또는 다른 환경과 간섭할 수 있다. 그룹 간 초기 배치·목표 clearance·환경 간 여유를 시뮬레이션으로 확인한다. 조정이 필요하면 이유와 값을 명시한다. 반대로 너무 멀리 분리해 NONE 상대가 항상 거리만으로 구별되는 배치로 고정하지 않는다.

이 asset/관측 설계처럼 구현 방향을 크게 바꾸는 미정 사항은 사용자에게 짧게 설명하고 확정한다. 이미 정한 p=0.8, 두 모델 비교, AMP 비율을 다시 결정 질문으로 돌리지 않는다.

## 6. RSI·AMP·Goal360 보존

### 공동 RSI

각 pair에 같은 clip/frame을 적용하고, 두 사람·공유 payload·속도·몸체 상태에 같은 강체변환을 적용한다. 두 팀 사이의 clip/frame/회전/위치는 독립적으로 샘플링할 수 있다. pair의 데이터 속 사람 순서를 실제 사람 ID에 연결할 때 두 사람의 모든 상태·관측·AMP 이력 매핑을 함께 바꾼다. 물리 ID가 특정 전진/후진 역할에 영구 고정되지 않게 한다.

현재 FullPickup 100 / carryWith 129 / putDown 135 프레임 데이터와 해시 검증을 유지한다. 공동 데이터에 단독 size RSI screening을 새로 적용해서 pickup 프레임을 제외하지 않는다.

현재 학습 skill 비율:

| 공동 과제 | loco | pickUp | carryWith | putDown |
| --- | ---: | ---: | ---: | ---: |
| AT | 0.4 | 0.1 | 0.4 | 0.1 |
| ON_TOP | 0.5 | 0.1 | 0.4 | 0 |

공동 ON_TOP에 바닥 putDown RSI를 넣지 않는다. `skillInitProb`만 보고 전부 loco라고 해석하지 말고 실제 template 경로를 따른다. 환경 전체에 ON_TOP이 하나 있다는 이유로 다른 AT팀의 putDown을 금지하지 않는다.

### 단독 RSI

각 singleton은 기존 Stage1 size-aware 초기화와 캐시를 사용한다. 현재 `independentCarryRsi`는 loco/pickUp/carryWith/putDown = 0.4/0.1/0.4/0.1이다. 실제 과제별 허용 경로를 유지하며 공동의 초기화 결과를 덮어쓰지 않는다. 2026-10-16 changelog의 단독 skill 선택 `else` 복구를 되돌리지 않는다.

### 목표 샘플링

현재 공동 Goal360 설정은 `uniform_angle_radius`, 거리 `[2.2, 2.8]m`, clearance 0.1m다. **putDown은 `reference_terminal_xy` 예외**로 reference 마지막 위치를 사용한다. 무조건 모든 skill을 새 360도 목표로 덮어쓰지 않는다. 그룹별 상대 목표 방향을 샘플링하고 세계 회전만 랜덤하게 해서 상대 목표 방향이 고정되는 오류를 피한다.

### AMP

현재 원본80% / 후진10% / 옆걸음10%, `uniform_clip_then_uniform_time`, task conditioning을 유지한다. 보정 reference는 `joint_carry/teamhoi_retarget_posture_all/amp_motions.yaml` 및 현재 해시 계약을 사용한다.

AMP는 사람별 판별이지 공동 pair 합동 판별이 아니다. 특정 ID에 후진 reference를 고정 배정하지 않는다. 실제 reset의 AMP history와 discriminator expert reference 샘플링은 구분한다. 공동은 `ampHistory: repeat_reset_state`를 유지한다. 4명으로 바뀐 flatten 순서에서 다른 사람/환경의 이력이 섞이지 않아야 한다.

## 7. 보상·CPA·받침을 팀별로 적용

### 보상

기존 개인 HOLDING state/progress/success를 유지한다. 각 pair가 반대 앵커를 동시에 만족할 때 **그 pair의 AT/ON_TOP state·progress·success**를 연다. 한 명이 놓치면 해당 pair의 placement gate만 다시 닫힌다. 다른 공동팀/단독의 gate는 독립적이다. 단독은 본인 holding gate를 유지한다.

현재 앵커는 payload local Y의 `±(size_y/2 − 0.15)`이고, 손 위치는 두 손 평균으로 계산한다. 점수는 기존 `exp(-10*d²)`, 만족 임계값 0.9다. 두 사람이 같은 앵커만 만족하면 공동 gate가 열리지 않는다. GraspView 관측 배정 때문에 보상을 특정 ID→특정 앵커로 새로 고정하지 않는다.

AT20·ON_TOP·성공 비종료·보상 비포화 설정을 유지하고 보상을 전원 평균/전원 최솟값으로 바꾸지 않는다. task/AMP 최종 가중치는 기존 각각0.5를 유지한다.

### CPA release

현재 collision coeff=0.5, CPA ttc discount=0.99, release buffer=1.0m를 유지한다. 해당 coupled pair의 두 사람이 모두 payload 반대각 길이/2+1m 안에 들어오면 **그 pair 사이**의 CPA를 해제한다.

4명에서는 어느 한 팀이 가까워졌다고 그 사람과 다른 팀/단독 사이의 CPA까지 없애면 안 된다. 현재 CPA 함수는 사람쌍 violation을 구한 뒤 `amax`하므로 pair별 release mask를 reduction 전에 반영하는 경로를 검토한다. 2명 기준 결과는 동일해야 한다.

### LocoPlatform

공동 loco인 그룹마다 확률0.5로 payload 아래에 받침을 둔다. 동시에 두 팀이 loco면 둘 다 받침을 가질 수 있다. 단독이나 pickup/carryWith/putDown에는 이 새 공동 받침을 적용하지 않는다.

현재 `heightSampleMax=1.0`, `maxTopSurfaceHeight=1.2`, size `[0.4,0.4,0.02]`, initialGap=0.05와 높이 계산을 유지한다. payload/support/받침 actor를 혼동하지 않는다. reset에서 이전 episode의 받침이 남거나 다른 팀 payload 높이를 바꾸지 않게 한다. 받침은 정책 task의 목표 객체가 아니다.

## 8. 모델·GraspView·순서 셔플

두 모델 모두 frozen Stage1 actor encoder → human64 → Human SA4(2head, FFN128, residual/LN) → 64입력 action head다. Critic은 기존 actor와 대응되는 구조로 전체 학습한다. `head_input: ca_only`, `critic_ca: true`는 현재 코드의 호환 키 이름이며 CA 모듈을 추가하라는 의미가 아니다.

Actor encoder와 scene RMS 동결, Stage1 head 초기화, SA와 action head 학습을 유지한다. SELF/COUPLED/NONE bias를 모두 사용한다. **4명 전체가 한 SA 안에 함께 들어가야 한다.** 두 개의 독립 2명 정책으로 쪼개면 NONE 경험을 늘리는 목적을 달성하지 못한다.

### GraspView의 4명 확장

1. 실제 reset 상태를 바탕으로 각 coupled pair의 반대 앵커를 배정한다. 기존의 payload local Y 비교 방식으로 상대적으로 가까운 반대쪽을 배정하며 두 사람이 같은 쪽을 받지 않는다.
2. 해당 배정은 episode 동안 고정한다. 매 프레임 가까운 앵커를 재선택하지 않는다.
3. 사람 i의 view에서는 **자기 팀 payload** 중심만 해당 파지 앵커 위치로 이동한다. 다른 팀 payload/목표는 바꾸지 않는다.
4. 회전 R·실제 size는 유지한다. 위치 T와 앵커점의 속도 `v + omega × delta`를 현재 좌표계에 맞게 일관되게 바꾼다. 시뮬레이터의 물체 상태 자체는 바꾸지 않는다.
5. 단독 사람은 원래 관측 view를 사용한다. 동일한 단독 view는 encoder 계산을 공유할 수 있다.
6. 각 view에서 **담당 사람의 human feature만** 선택해 원래 사람 순서로 모은 뒤 4명 SA에 입력한다. task packet 행 번호를 사람 ID라고 가정하지 않는다.

현재 `grasp_view.py`의 446/566/568/624와 634, packet 마지막10, view2 같은 고정 상수를 재사용하지 않는다. 관측 생성부의 실제 M/O 레이아웃을 기준으로 slice를 계산하고 actor/critic/player/RMS 모두 같은 계약을 사용한다. 원본 2명 raw634, GraspView1268 결과를 회귀 비교한다.

### 순열의 구분

- episode의 사람↔공동 역할 배정 랜덤화.
- entity/task token 순서 셔플과 payload/target/owner ID 연결.
- SA 내부 human 순서 셔플과 relation matrix의 행/열 동시 순열.
- SA 출력·action을 실제 사람 순서로 복원.
- PPO/AMP 배치의 환경×사람 packing.

위 다섯 가지는 서로 다른 과정이다. 하나를 했다고 나머지를 대신했다고 생각하지 않는다. 그룹 ID/packet 위치/논리 object ID/물리 actor ID를 혼용하지 않는다.

## 9. 구현 순서와 산출물

1. 기존 두 YAML 쌍·wrapper와 CPU 테스트를 확인해 기준값을 확보한다.
2. 그룹 샘플러/매핑 및 물리 asset·관측 설계를 정한다. 미정인 asset bank/물체 수는 변경안을 설명한다.
3. graph/RSI/목표/보상/CPA/받침을 그룹별로 연결한다. 물리 commit과 history reset 순서를 보존한다.
4. GraspView·RMS·Stage1 전이·순열·action packing을 확장한다.
5. 신규 env/train YAML 두 쌍과 각 `_train.sh`, `_test.sh`, `_vnc.sh`를 만든다. output도 분리한다.
6. CPU 검증, 2048환경 짧은 학습, checkpoint 저장/재로드, headless 평가 및 viewer 경로를 확인한다.
7. config.md의 목록/학습/로컬 추론/VNC 명령과 structure.md, changelog를 실제 결과에 맞게 갱신한다.

체크포인트에는 sampler/p·역할/물체/관측 계약·AT20·GraspView·AMP/RSI 계약을 식별할 수 있게 한다. 호환되지 않는 구성을 조용히 재개하지 않는다. 단, 모델의 가변 인원 추론 지원 여부와 동일 학습환경 resume 계약은 구분한다.

권장 명령 형식은 아래와 같다. **아직 실행 가능한 명령이 아니다.** `<O>`는 확정한 정책 object 수다. 추가 bank actor를 포함한 전체 simulator actor 수와 혼동하지 않는다.

```bash
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS \
TOKENHSI_GPU=<사용할_GPU> STAGE1_CHECKPOINT=<동일_Stage1_PTH> \
bash tokenhsi/scripts/multi_agent/stage2_at20_4agent_train.sh 4 2048 <O>

env -u RESUME_CHECKPOINT -u MAX_ITERATIONS \
TOKENHSI_GPU=<사용할_GPU> STAGE1_CHECKPOINT=<동일_Stage1_PTH> \
bash tokenhsi/scripts/multi_agent/stage2_grasp_at20_4agent_train.sh 4 2048 <O>
```

## 10. 필수 검증과 보고 기준

### CPU / 수치 검증

- 샘플 수를 충분히 늘려 K=0/1/2 분포가 0.8/35.2/64% 근처인지 확인. 사람 ID별 공동 확률·partner 빈도 편향도 확인.
- 모든 사람이 정확히 한 그룹·한 carry interaction을 가짐. 그룹 밖 payload/target 공유 없음. 두 공동팀과 공동+단독 AT/ON_TOP 혼합 모두 검사.
- joint query의 SELF1/COUPLED1/NONE2, solo query의 SELF1/NONE3 확인. task와 같은 목표 위치라는 이유만으로 잘못 coupled 처리하지 않음.
- 팀1 holding을 끊어도 팀2/solo placement 보상이 그대로인지, 같은 앵커 동시 만족은 실패인지 확인.
- AT20 거리/높이 경계와 ON_TOP 보상 회귀, CPA pair 해제 후 cross-team 페널티 유지 확인.
- graph/사람/entity 순열 후 출력/action을 역순열했을 때 수치 허용오차 내 동일. RNG를 통제한 결정론적 forward로 확인.
- GraspView solo raw 동일, 각 coupled view는 자기 payload만 변경, sides episode 고정, rigid-point velocity·frame 변환 확인.
- 기존 2명 경로의 관측/보상/Stage1 전이 회귀. CPU 테스트가 통과했다는 이유만으로 물리 검증 완료라고 쓰지 않음.

### 시뮬레이션 / 짧은 학습

- GPU 점유 확인 후 지정 GPU만 사용. runtime_env.sh·tokenhsi 환경 사용, 직접 import 시 Isaac Gym을 torch보다 먼저 import.
- 학습 검증도 **2048환경**. `MAX_ITERATIONS`만 줄이고 `_check` output을 별도로 사용한다. 4명으로 메모리가 부족하면 수치와 원인을 보고하고 임의로 환경 수를 줄이지 않는다.
- 두 joint, joint+solo, allsolo 및 AT/ON_TOP 혼합, loco/pickUp/carryWith/허용 putDown 강제 reset을 평가 환경에서 각각 확인.
- 두 팀이 다른 RSI 프레임/skill일 때, 두 loco 받침 활성일 때, subset env만 reset할 때 상태 누출·관통·폭발·비정상 속도·NaN 확인.
- 초기 접촉 프레임 수 회귀와 실제 사용 프레임 분포 확인. 기존 FullPickup 데이터를 임의 필터링하지 않음.
- train/save/reload/eval 경로에서 관측/action/RMS shape와 finite 확인. Actor encoder/RMS 동결, Critic/SA/head 갱신도 확인.
- test와 VNC가 새 설정·인원·물체 수를 전달하는지 확인. headless만 실행했다면 실제 VNC 화면 확인과 구분해 보고.

### 학습 지표

최소한 단독/공동 × AT/ON_TOP 성공률, 공동 동시holding, 개인holding, 시작skill별 공동 성과(loco/carryWith 등), AMP를 구분한다. 장면 전체4명 성공과 pair 성공을 섞지 않는다. loco는 받침 유/무도 구분한다.

공동 성공의 분모는 사람 수인지 pair 수인지 명시하고, 동일 팀을 두 번 세는 지표는 그렇게 표시한다. RSI에서 이미 잡은 상태로 시작한 것을 loco 공동집기 성공에 포함하지 않는다. 경험 중 한 번 만족/현재 만족/일정 시간 유지도 구분한다.

두 새 실험은 같은 epoch뿐 아니라 환경 timestep·사람 transition 수·wall time도 함께 기록한다. 기존 2명과 비교할 때 4명은 사람 transition 수가 다름을 밝힌다. 초기 성공률만으로 GraspView 우열을 단정하지 않는다.

attention 검사는 같은 장면/시작skill에서 layer/head별 원래 scaled QK와 bias 포함 softmax를 구분한다. NONE 총합과 NONE 하나당 평균을 함께 표시한다. 4명 공동 query에는 NONE이 2개라서 단일 SELF와 총합을 직접 비교하면 해석이 달라진다. 필요 시 residual 대비 attention 출력 크기도 확인하되 이 검사 때문에 본 구현을 불필요하게 확대하지 않는다.

## 11. 사용자 의도 요약

"Stage1이 중점 파지를 배웠으니 GraspView도 놓치기 아쉽고, 기존 SA4도 비교하고 싶다. 두 설정을 각각 4명 랜덤 그룹으로 학습해서 공동/단독이 섞인 관계를 경험하게 하자. AMP·RSI·순서 셔플·AT20은 정확히 유지하고, 한 팀의 초기화/보상이 다른 팀과 꼬이지 않게 구현해 달라."

이 문서에서 확정하지 않은 asset bank/관측 물체 수는 실제 코드와 시뮬레이터 제약을 보고 제안할 항목이다. 나머지는 위 요구를 기준으로 구현·검증하고, 수행한 결과와 남은 한계를 구분해 전달한다.
