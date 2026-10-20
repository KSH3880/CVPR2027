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

`BEFORE_FOLLOWER_NEAR=1`은 평가에만 적용한다. 선행 상자 중심까지의 두 root XY 거리를 1~2m에서 추첨해 후행을 더 가깝게 두고 각도도 추첨한다. 다른 초기화 규칙과 보상 gate는 유지한다.

검증 결과는 [changelog](../../changelog.md), 산출물은 `output_etc/check/stage2_before/`에 둔다.

## 바닥 전용 비교: stage2_before_ground

`stage2_before_ground`는 기존 BEFORE와 별도 env/train YAML·train/test/VNC·output을 쓴다. `randomHeight=false`, `randomHeightProb=0`, `platformMode=ground_only`로 모든 초기 상자를 바닥에 두고, BEFORE 선행과 단독 AT의 목표 중심 높이를 운반 상자 높이의 절반으로 둔다. 시작 받침과 AT 도착 받침은 모두 비활성화한다. 물리 actor 인덱스 구조를 유지하기 위해 고정 받침 슬롯은 지하 Z=-20m/-19.5m에 격리한다. ON_TOP의 쌓기 대상 상자는 실제 과제 물체로 유지한다.

그룹 추첨·물체 예산 재샘플링·loco 초기화·현재 AT gate·보상·색상·학습 설정은 기존 BEFORE와 동일하다. 상자 yaw·XY·크기는 계속 무작위다. Stage1 fixed 가중치 전이로 새 학습을 시작하며 기존 BEFORE의 학습 프로세스와 설정은 유지한다. 저장된 reset 계약이 달라 기존 BEFORE checkpoint의 직접 resume/환경 변경 평가는 거부한다. ground test/VNC는 ground 저장본 옆 환경 설정을 사용한다.
