# 통합 Stage2

`stage2`는 BEFORE(enable)·COUPLED·solo를 하나의 정책으로 RL 학습한다. 학습 기본값은 4명·1024환경·8물체 토큰·1200step이다. N≥4·M≥N을 지원하며 5명 이상 추론은 전용 그룹 스폰 옵션을 사용한다. 고정된 두 teacher를 모방하는 별도 학습은 [stage2_distill](stage2_distill.md)이다. 명령은 [config](../config.md), 추가 실행법은 [USAGE](../USAGE.md#통합-stage2)를 따른다.

`stage2_900`은 같은 RL 설정의 900step 버전이다. 환경의 `episodeLength`와 실험/저장 이름만 다르며 전용 train/test/VNC/eval 스크립트를 사용한다. Viewer와 정식 평가도 기본900step이다. 새900 버전 wrapper의 GPU는2/6/7만 허용하며 기본6이다.

## 관계와 샘플링

- 매 episode에 BEFORE용·coupled용·혼합용을 1:1:1로 뽑는다. BEFORE용은 기존 쌍 생성 확률 .5와 크기 호환 역할 선택, coupled용은 기존 공동 생성 확률 .5와 2~4명 그룹을 유지하며 두 유형 모두 solo를 포함할 수 있다.
- 혼합용은 무작위 BEFORE 2명과 COUPLED 2명이다. 한 사람은 한 그룹에만 속하고 그룹 사이 작업 물체는 공유하지 않는다.
- solo 과제 확률은 SIT/CLIMB/CARRY_AT/ON_TOP/PUSH=10/15/25/20/30%다. 접근은 1~5m, carry 목표는 1~5m, solo PUSH 접근/이동은 2~5m다. 공동 PUSH 접근은 2~10m, 이동은 2~5m다.
- BEFORE 선행자는 CARRY_AT, 후행자는 SIT/CLIMB/ON_TOP이다. 선행 목표를 중심으로 후행을 배치하며, 쌍의 절반에 후행 경로 합이 선행 경로 합보다 짧은 near 조건을 적용한다. 나머지 절반에는 이 부등식을 강제하지 않는다. 배치 재시도는 관계·과제·near 여부를 유지한다. BEFORE가 포함된 장면의 실패 시에는 원래 BEFORE처럼 loco 참조를 다시 뽑고, coupled 전용 장면은 기존처럼 사람 참조를 유지한다.
- loco 초기화·기존 지면/물성·PUSH 후보 배치와 경로 검사·distractor 검사를 유지한다. 혼합은 BEFORE 배치 후 PUSH 후보를 뽑고 전체 충돌·near 조건을 검사한다. BEFORE를 포함한 장면에는 기존 BEFORE의 사람 간 .7m 검사를 적용하며, 원래 coupled 장면의 고정 loco 배치에는 이 추가 검사를 적용하지 않는다. 4명에는 별도 그룹 중심 간격을 추가하지 않는다.

## 물체와 관측

4명 기준 coupled의 24개 물리 후보를 사용한다: CARRY/SIT/CLIMB/support/solo PUSH 각4개, 공동 2명용2개·3명용1개·4명용1개. 인원이 늘면 기존 coupled bank 계산을 따르며 필요한 후보만 장면에 사용하고 나머지는 멀리 둔다. 관측 토큰8개와 실제 상자 개수는 다르다.

작업 물체와 공동 구역 토큰의 ID를 정한 뒤 BEFORE alias는 실제 미사용 토큰에 배정한다. 원본 task packet은 물리 공유 관계와 보상에 유지하고 encoder 입력에서만 후행 target을 alias로 돌린다. Alias의 XYZ만 선행 AT 목표 XYZ로 바꾸며 회전·물성은 공유 물체를 따른다. 공동 구역은 기존 개인별 분할 토큰·개인 목표를 유지한다. 남는 토큰은 기존 support 후보의 distractor로 채우되 필수 토큰·alias를 재배치하지 않는다.

M≥N을 허용한다. 관계/행동을 먼저 샘플링하고 실제 task packet에서 쓰는 물체 슬롯과 BEFORE alias 수를 합산하여 M 이내인 장면만 채택한다. 부족한 환경 행만 다시 샘플링하며 M≥2N에서는 기존 샘플과 RNG 소비를 유지한다. 현재 과제 집합의 수요는 `N + ON_TOP 사람 수`와 일치한다. BEFORE SIT/CLIMB는 공유로 비워진 owner 슬롯을 alias로 사용하고, BEFORE ON_TOP은 추가 alias 하나를 사용한다. M=N이면 ON_TOP을 포함한 구성이 채택되지 않으므로 장면/행동 분포는 예산에 조건부다.

고정 `holding_ontop`은2N, 4명의 `before_stack`은6토큰이 필요하다. 고정 `mixed`를 M=N으로 요청하면 적어도 한 명의 carry 후보가 SIT/CLIMB 크기와 호환되어야 한다. 정해진 물체 후보에서 이 조건이 불가능하면 명시적인 오류를 내며 M을 늘려야 한다. 임의 혼합 장면은 관계/과제를 다시 뽑아 예산에 맞춘다.

관측은 기존 5N task packet+N group suffix를 유지한다. 4명·8토큰은 1272차원이며 인원/토큰 증가에 따라 관측 길이가 늘어난다. 물리 충돌 검사는 실제 전체 상자 기준이며 alias/구역을 별도 물리 상자로 중복 취급하지 않는다.

## 5명 이상 추론

`GROUP_SPAWN_OVER4=1` / `--group_spawn_over4`로 명시적으로 선택하며 학습에는 적용하지 않는다. 사람 순서를 무작위로 섞고 남은 사람에서 solo/enable/coupled를 동일 확률로 선택한다. solo는 1명, enable은 2명, coupled는 2~min(4, 남은 사람)명을 할당한다. 마지막 한 명은 solo이며 행동·역할·크기 호환은 기존 sampler를 따른다. 이는 그룹 선택 확률이며 사람 수나 장면 유형의 1:1:1 비율이 아니다. `mixed`는 두 관계가 모두 있는 구성만 채택한다. `before`/`coupled`는 해당 관계와 solo만, `all_coupled`는 최대4명 그룹과 남는 solo PUSH를 생성한다.

사람을 기존 loco 자세/위치로 만든 후 그룹별 XY 평행 이동으로 초기 그룹 중심을 원 위에 최소5m 간격으로 둔다. 그룹 내부 상대 위치·높이·자세·속도와 기존 작업 거리/물성은 보존한다. BEFORE 후행의 goalroute 배치가 끝나면 해당 쌍의 사람·작업 상자·목표를 같은 XY만큼 옮겨 중심을 복원한다. 이후 공동 PUSH 후보와 전체 충돌/경로/near 조건을 검사한다. 5m는 초기 그룹 중심 간격이며 작업 영역 분리나 rollout 중 간격을 보장하지 않는다.

실제 작업 슬롯·BEFORE alias·공동 구역 토큰·ON_TOP 필수 받침을 보존하고 불필요한 물리 후보는 멀리 주차한다. 해당 관측의 물체 특징/pose를 비우고 actor/critic encoder의 `object_valid`로 제외한다. 카메라도 필수 물체·사람·목표만 포함한다. wrapper의 공동 구역 배정 기본값은 ordered다. 기존 coupled 전용 [5m 배치 명세](group_spawn_over4.md)를 통합 관계에 확장한 경로다.

## 두 teacher와 distillation

`stage2_teachers_{test,vnc,eval}.sh`는 두 PTH의 원래 모델·전처리·RMS를 각각 복원하고 고정한다. 두 teacher 모두 전체 장면을 관측하며 실제 공동 PUSH 그룹원만 coupled teacher의 평균 행동을 선택한다. BEFORE 선행/후행과 모든 solo는 enable teacher가 담당한다. 선택 후 기존 action clipping/scaling을 적용한다. 가중치 평균·재학습·단일 PTH 병합은 없다. 실행에 사용한 경로/epoch/frame/SHA256·담당 규칙을 출력 폴더의 `teachers.json`에 기록한다.

같은 관측에 대한 위 사람별 teacher 평균 행동을 정답으로 삼는 distillation은 별도 [stage2_distill](stage2_distill.md) 컨피그/학습 루프에서 실행한다. 학습한 student의 viewer/eval은 두 teacher 없이 단일 student PTH를 사용한다.

## 정책과 보상

동일 Stage1 PUSH 26k에서 actor/critic·action/value head·AMP를 전이하고 actor encoder와 입력 RMS를 동결한다. 사람 attention은 공통 64차원·2헤드·4층이다. BEFORE/COUPLED 각각 64차원 임베딩을 두고 attention 투영·fusion MLP·action head를 공유한다. NONE/SELF bias는 0이다. actor/critic 모두 동일한 관계 처리 구조를 사용한다.

| 대상 | 학습 내부 AT 조건 |
|---|---|
| BEFORE 그룹 | 3D 거리≤20cm |
| COUPLED·solo | 3D 거리≤20cm 및 높이 오차≤1mm |

BEFORE는 선행 AT의 current 성공에 따른 후행 state gate·기존 progress 목표와 대기 속도 감점을 유지한다. 감점 반경은 받침 XY 대각선 반경+2.5m, 계수1이며 ON_TOP도 사람 위치로 진입을 판정한다. 감점은 진입 후 선행 AT 성공 시 해제되고 대기 위반으로 조기 종료하지 않는다. COUPLED는 기존 그룹 접촉/개인 도착 gate와 CONTACT+AT 보상 포화를 유지한다. AMP 참조 family는 현재 episode 과제 구성에 맞춰 샘플링한다.

## 평가와 저장

정식 평가는 기존 독립 float64 공간 채점으로 AT/PUSH·공유 중심3D20cm 및 기존 SIT/CLIMB/ON_TOP 점/영역 조건을 유지한다. BEFORE pair 지표는 prerequisite에서, coupled 그룹 지표는 실제 PUSH 그룹에서 계산하여 혼합 장면의 BEFORE 쌍을 공동 PUSH 분모에 넣지 않는다. rollout과 CPU 재채점은 같은 평가용 그룹을 기록·사용한다.

`stage2`·`stage2_distill`과 두 teacher의 viewer/eval은4명도 ordered가 기본이다. 상자의 로컬 X 좌표로 사람을 좌우 정렬하여 담당 구역을 reset 때 배정한다. 학습의 랜덤 구역 배정과 별도이며 실제4면 접촉 선택은 기존 코드가 계산한다. `COUPLED_ASSIGNMENT` / eval의 `--coupled_assignment`로 명시적으로 바꿀 수 있고 결과 JSON과 eval manifest에 배정 방식을 기록한다.

새 설정과 BEFORE 배치·보상은 checkpoint 계약에 포함한다. 단일 통합 모델 wrapper에는 `stage2` 체크포인트를 사용하며 두 teacher는 전용 wrapper의 두 인자로 지정한다. 5명 이상과 두 teacher도 동일 공간 채점·CPU 재채점을 사용하고 결과에 teacher/그룹 스폰 설정을 기록한다. 기본 4명 학습 분포 평가와 구분한다. 확인 산출물은 `output_etc/check/`에만 둔다.
