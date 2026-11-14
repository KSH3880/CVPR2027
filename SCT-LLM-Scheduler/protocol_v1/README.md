# CVPR2027 공용 평가 프로토콜

**기준 버전: `cvpr2027-evaluation-v1.0.0` · 2026-11-11 · 공동 PUSH 출발 변위 20cm 확정.**

평가를 구현하거나 결과를 비교할 때 이 문서를 기준으로 삼는다. 단위·샘플링·성공 조건·집계·예외 처리는 모든 방법과 레포에 동일하게 적용한다. 학습 reward/gate 및 viewer 진단은 평가 기준과 별개다.

- 전체 규칙: 이 문서.
- 레포별 진입점·실행 명령·checkpoint·현재 지원 상태: [implementations.md](implementations.md).
- 시뮬레이터 없이 확인하는 계산 예제: [checks/README.md](checks/README.md).
- 다른 레포에서는 아래 데이터 계약으로 상태를 내보내고 같은 계산을 적용한다. 같은 seed만으로 같은 장면이 보장되지는 않으므로 초기 snapshot을 공유한다.

순서: **공통 규칙 → Enabling 차이 → Coupled 차이와 추가 지표 → 테이블 구성 → 기록·구현 검증 → 미정 항목**.

## 1. 공통 규칙

### 1.1 실행·공정성

| 항목 | 규칙 |
| --- | --- |
| 규모 | 설정당 512장면 × 3seeds |
| 시간 | 최대 900step = 30초, 제어 주기 1/30초 |
| 초기화 | loco 100% |
| 정책 | 평균 행동, 저장 정규화 통계 고정 |
| Episode | 첫 episode만 평가; 성공해도 reset하지 않음 |
| Checkpoint | 방법별 하나를 모든 인원수에 사용 |
| 비교 조건 | 방법 간 동일한 초기 상태·과제·관계·배정·물성·성공 기준 |
| 관측 물체 토큰 | N명에 M=2N; 작업 물체·alias·구역·distractor 포함 |
| 단독 행동 확률 | SIT / CLIMB / AT / ON_TOP / PUSH 각각 20% |

한 사람은 한 그룹에 속하며 다른 그룹과 작업 물체를 공유하지 않는다. Alias·구역 토큰은 물리 상자를 복제하지 않는다. 행동을 먼저 선택하고 크기를 생성하여 크기 호환성에 의한 행동 비율 변화를 막는다.

### 1.2 상자 크기·물성

전체 길이 X/Y/Z, 단위 m. 일반 상자는 각 축의 0.05m 격자에서 균등 추출한다. Enable·coupled의 변경은 2·3절을 따른다.

| 용도 | X | Y | Z |
| --- | ---: | ---: | ---: |
| SIT | 0.25–0.80 | 0.25–0.80 | 0.35–0.50 |
| CLIMB | 0.40–0.80 | 0.40–0.80 | 0.20–0.70 |
| AT 물체 | 0.20–0.60 | 0.20–0.60 | 0.20–0.60 |
| ON_TOP payload | 0.20–0.60 | 0.20–0.60 | 0.20–0.60 |
| ON_TOP 받침·distractor 후보 | 0.40–0.80 | 0.40–0.80 | 0.25–0.50 |

PUSH는 아래 조합 중 `min(X,Y) ≥ 1.1×Z`를 만족하는 **167개 profile에서 균등 선택**한다.

| 축 | 후보 |
| --- | --- |
| X / Y | 1.60–2.20m, 0.10m 간격 |
| Z | 1.45–1.70m, 0.05m 간격 |

| 물성·초기 상태 | 값 |
| --- | --- |
| 비-PUSH 밀도 | 100kg/m³ |
| PUSH 밀도 / 마찰계수 | 3kg/m³ / 0.4 |
| 질량 | 밀도×실제 부피 |
| 높이·회전 | ground, 직립 중심 Z=높이/2, yaw 랜덤 |

### 1.3 거리·방향

초기 XY 직선거리 기준. 사람은 root, 물체는 물리 상자 중심을 사용한다. 경로합은 아래 구간들의 합이다.

H=사람, O=상자, P=payload, S=받침, G=목표.

| 행동 | 접근 | 이동 | 경로합 |
| --- | --- | --- | --- |
| SIT | H→O 1–5m | 상자 자체가 목표 | 1–5m |
| CLIMB | H→O 1–5m | 상자 자체가 목표 | 1–5m |
| AT | H→O 1–5m | O→G 1–5m | 2–10m |
| ON_TOP | H→P 1–5m | P→S 1–5m | 2–10m |
| PUSH | H→O 2–5m | O→G 2–5m | 4–10m |

- 목표·받침과 초기 사람 사이 최소 1m. 관계별 변경은 2·3절 적용.
- 일반 접근·AT/ON_TOP 목표 후보는 반지름·방향각을 균등 추출한다. 단독 PUSH 목표는 상자 로컬 +Y 주변 ±45°.
- 충돌·최소 거리·경로 검사 후 채택하므로 최종 분포는 조건부다.

### 1.4 배치·distractor

- 사람·상자 겹침, 작업 경로·목표 간섭을 검사하고 부적합 배치를 재추출한다. 정책 실행 후 실패한 장면은 제외하지 않는다.
- **5명 이상:** 그룹 중심(root XY 평균) 간 최소 5m. 그룹 내부 배치를 보존해 원 위로 평행 이동한다. 그룹 수 G>1일 때 반경 `R=5/(2 sin(π/G))`.
- 후속 배치로 그룹 중심이 변하면 사람·관련 물체·목표를 함께 이동해 복원한다. 5m는 초기 중심 간격에만 적용한다.
- **모든 방법·인원수에서 distractor는 관측에 유지하고 물리에서 제거한다.** 작업 물체·ON_TOP 받침·enable alias·coupled 구역은 제거 대상에서 제외한다.
- Distractor는 정책 encoder의 attention에서도 제외하지 않는다. Stage1·Stage2 모두 전체 관측 토큰을 사용한다.
- 장면 생성·배치 검사 후 distractor만 물리 장면 밖으로 이동한다. 관측 토큰·형상/BPS는 유지하고 위치·회전·선속도·각속도는 초기 샘플 값으로 고정한다. 보상·성공 판정은 실제 작업 물체 상태를 사용한다.
- 4명 관측용 distractor 배치: 환경 중심 반경 4.5m 원 안에서 면적 기준 균등 추출 후 간섭 시 재추출.
- 5명 이상은 k명 그룹마다 2k 토큰을 배정한다. 작업 물체·받침·alias·구역에 필요한 토큰을 제외한 나머지는 해당 그룹 중심 반경 4.5m 원 안에 관측용 distractor로 배치한다. 면적 기준 균등 추출 후 간섭 시 재추출하며, 물리는 제거한다.

### 1.5 공간 성공 조건

단독·enable·coupled에 공통 적용한다. 주지표는 Region, Point는 보조 기준이다.

| 행동 | 검사점 | Region 조건 | 목표 높이 |
| --- | --- | --- | --- |
| AT | 물체 중심 | 지정 목표까지 3D 거리≤20cm | 지정 목표 Z |
| PUSH | 물리 상자 중심 | 지정 목표까지 3D 거리≤20cm | 지정 목표 Z |
| SIT | pelvis/root | 받침 XY 전체 100% + 높이 오차≤20cm | 상판 최고 Z+0.10m |
| CLIMB | pelvis/root | 받침 XY 전체 100% + 높이 오차≤20cm | 상판 최고 Z+0.94m |
| ON_TOP | payload 중심 | 받침 XY 전체 100% + 높이 오차≤20cm | 상판 최고 Z+payload 직립 높이/2 |

- XY100%: 받침 yaw 로컬 좌표의 `|x|≤0.5Lx, |y|≤0.5Ly`. 검사점 기준이며 사람·payload 전체가 받침 안에 들어갈 필요는 없다.
- 모든 비교 방법의 성공 판정과 Error에 XY100%를 적용한다. 레포의 일반 평가 기본값을 그대로 쓰지 말고 실제 적용값을 기록한다.
- Point: 목표점까지 3D 거리≤20cm. AT/PUSH는 Region과 동일.
- 실제 물리 상태를 float64로 채점한다. Virtual alias·학습 reward/gate·추가 AT 높이 1mm 조건은 사용하지 않는다.
- 받침이 기울어진 경우 목표 상판 높이는 중심Z에 회전 직육면체의 수직 반범위를 더한다. 단위 쿼터니언 회전행렬 R에서 수직 반범위는 `sum_j(abs(R_zj) × size_j/2)`다. XY 영역은 받침 yaw만 사용한다. ON_TOP payload의 목표 높이 offset은 **직립 높이 size_z/2**이며 payload 기울기별 반범위로 바꾸지 않는다.
- 추가 접촉·속도·손 놓기·발 높이·유지시간 조건 없음.

### 1.6 성공 기록·낙상

- 첫 action 이후부터 채점. `ever`는 유효 frame에서 한 번 이상 성공.
- 낙상 frame까지 채점하고 해당 사람의 제어·관절 motor를 해제한다. 수동 물리 actor로 남기며 다른 사람은 계속 수행한다.
- 낙상 전 성공은 보존한다. 한 명의 낙상으로 그룹 전체를 실패 처리하지 않는다.
- 전원 낙상 시 조기 종료 가능. 실제 기록 길이 T를 보존한다.
- 낙상: 발(left_foot/right_foot)을 제외한 몸체 중 접촉력의 축별 절댓값 합이0.1보다 큰 몸체가 있고, 발을 제외한 몸체 중 높이 문턱 미만인 몸체가 있으면 낙상이다. 두 조건의 몸체가 같을 필요는 없다. 높이 문턱은 일반 몸체0.15m·head0.30m, `progress>1` 이후 적용한다.

### 1.7 지표·통계

| 지표 | 성공 단위·정의 |
| --- | --- |
| Independent Interaction Success Rate | 단독 개인의 ever 성공 |
| Enabling Interaction Group Success Rate | 같은 스텝에 선행 AT·후행 행동이 모두 성공한 적이 있는 쌍 |
| Coupled Interaction Group Success Rate | 공유 상자 중심이 공동 목표에 ever 도달한 그룹 |
| Mixed Interaction Group Success Rate | 같은 seed의 위 세 SR을 1/3씩 평균 |

- Seed별 SR·Mixed를 먼저 계산한 뒤 3seeds 평균±표본표준편차(ddof=1) 보고.
- 평가 seed 값은 사용자가 탐색 후 확정한다. 확정한 seed는 모든 비교 방법에 동일하게 적용한다.
- 유형별 성공/시도 수와 C2/C3/C4 분포 저장. Coupled 내부는 실제 그룹 하나당 시도 하나.
- 유형 분모가 0이면 해당 SR과 Mixed는 null. 관계 전용 평가는 해당 SR만 보고.
- 초기 상태·per-frame pose·생존·낙상·성공·실행 버전을 저장하고 CPU 재채점으로 검증.

### 1.8 공통 Error — 메인 eval

매 평가에서 **Point Error·Region Error**를 함께 출력한다. 단위는 cm이며 두 지표 모두 같은 Region 성공 trial만 사용한다. 본 문서의 기준은 1.5절의 XY100%이며, 기존80%로 저장한 Error는 재집계한 결과와 구분한다.

| 행동 | 검사점 | Point 목표 | Region 목표 |
| --- | --- | --- | --- |
| SIT | pelvis/root | 받침 상판 중심+0.10m | XY 전체 100%의 같은 높이 평면 |
| CLIMB | pelvis/root | 받침 상판 중심+0.94m | XY 전체 100%의 같은 높이 평면 |
| AT | 운반 상자 중심 | 지정 3D 목표 | Point와 동일 |
| ON_TOP | payload 중심 | 받침 상판 중심+payload 직립 높이/2 | XY 전체 100%의 같은 높이 평면 |
| PUSH | 실제 단독·공유 상자 중심 | 지정 3D 목표 | Point와 동일 |

- Point는 목표점까지, Region은 목표 평면의 가장 가까운 점까지 3D 거리를 사용한다. Region Error에 성공 판정의 높이 ±20cm 허용오차를 빼지 않는다.
- Region 거리: 받침 yaw 로컬 좌표에서 `sqrt(max(|x|−0.5Lx,0)² + max(|y|−0.5Ly,0)² + (z−z*)²)`.
- Trial Error는 `100 × min(유효 frame의 거리)`. 첫 action 이후부터 낙상 frame까지 포함하고 낙상 이후는 제외한다. 성공 이전·이후 모두 탐색한다. Coupled는 구성원 최소 한 명이 유효한 frame을 사용한다.
- Independent는 개인 Region 성공, Enable은 **그룹 동시 성공한 쌍의 선행·후행 둘 다**, Coupled는 공유 상자 중심 성공한 그룹만 포함한다. Coupled는 사람 수와 무관하게 그룹당 한 번 집계한다.
- 행동별 성공 표본의 산술평균과 표본 수를 출력하며, PUSH는 단독 한 명·공동 그룹 하나를 각각 표본 하나로 센다. Coupled 2/3/4명 결과도 별도 출력한다. 별도의 행동 간 전체 macro는 정의하지 않는다.
- Seed별 평균 후 seed 평균±표본표준편차(ddof=1)를 보고한다. 성공 표본이 없으면 N/A(null)이며, seed 평균에 N/A가 있으면 전체 평균·표준편차도 N/A로 둔다.

### 1.9 Stage1 비교 입력

- 같은 물리 장면·성공 기준에서 원본 Stage1 checkpoint의 가중치·정규화 통계를 고정하여 평가한다.
- Enable: 후행에게 목표 위치에 정렬한 가상 받침을 입력한다. 실제 공유 상자의 위치·물리는 바꾸지 않는다.
- Coupled: 같은 물리 공유 상자의 개인별 PUSH 구역을 각 참여자의 과제로 입력한다. 물리 상자를 복제하지 않는다.
- Stage2 coordination은 사용하지 않고 외부 대기·동기화·freeze 제어를 추가하지 않는다. 공통 낙상 처리는 유지한다.
- 실제 정책에 들어가는 첫 관측과 원본 정책의 행동 평균·표준편차·정규화 일치, 가중치 불변을 검증한다.
- **SCT + FSM:** 같은 SCT 입력·정책에 FSM 대기·해제를 추가한다. Enable은 선행 AT 성공까지 후행을 고정하고, Coupled는 자기 그룹의 실제 상자에 양손12cm 미만으로 접근한 도착 이력을 누적한다. 먼저 도착한 사람은 고정하고 마지막 참여자가 도착하면 그 사람은 고정하지 않은 채 전원 한 번 해제한다. 사람의 root·DOF 상태를 서브스텝마다 복원하는 방식이며 대기 중 충돌체·반력은 유지한다. 공통 SR은 변경하지 않는다.

## 2. Enable 변경 규칙

### 2.1 구성·크기

- 선행 AT 1명 + 후행 1명. 후행 SIT / CLIMB / ON_TOP 각각 1/3.
- 선행 운반 상자 O를 후행 대상·받침으로 공유한다. 공유 상자만 **전체 공통 교집합** 적용.

| 대상 | X | Y | Z |
| --- | ---: | ---: | ---: |
| 공유 상자 O | 0.40–0.60m | 0.40–0.60m | 0.35–0.50m |

Alias는 같은 크기·물성을 사용한다. ON_TOP의 별도 payload는 공통 기본값을 유지한다.

### 2.2 거리

A=선행, B=후행, G=선행 목표, P=후행 payload.

| 구간 | 범위 |
| --- | --- |
| A→O / O→G | 각각 1–5m, 합 2–10m |
| SIT·CLIMB B→G | 2–10m |
| ON_TOP B→P / P→G | 각각 1–5m, 합 2–10m |
| ON_TOP B→G | 최소 2m, 자연 최대 10m |

**Near50 해제.** 선행·후행 경로합의 실제 값·분포·대소관계는 맞추지 않는다. ON_TOP B→G에 5m 상한을 두지 않는다.

### 2.3 그룹 성공·완료 시간

- `ever(선행 AT AND 후행 행동)`. 같은 유효 스텝에 두 조건을 모두 만족해야 한다. 별도 시간 순서·유지시간 조건은 없다.
- 한 쌍을 시도 하나로 집계한다.
- **Completion Time ↓ (s):** 두 조건을 처음 동시에 만족한 시각. 개인 완료 시간의 합·최댓값을 사용하지 않는다.
- 성공한 쌍만 평균하고 실패는 제외한다. 성공이 없으면 N/A. AT→SIT/CLIMB/ON_TOP별 평균 후 세 행동 동일 가중 평균, seed별 산출 후 평균±표본표준편차.
- 제어 step t의 완료 시각은 t/30초다. t=0 초기 상태는 성공 채점에 포함하지 않는다.

## 3. Coupled 변경 규칙

### 3.1 구성·상자 확장

2–4명이 같은 물리 상자를 공동 PUSH한다. 공통 base profile `(X,Y,Z)`를 참여 인원 k에 따라 `(kX,Y,Z)`로 확장한다. 밀도 유지, 질량 k배.

| 인원 | 실제 X | Y / Z | Base 대비 질량 |
| --- | ---: | --- | ---: |
| 2명 | 3.20–4.40m | 공통 base와 동일 | 2배 |
| 3명 | 4.80–6.60m | 공통 base와 동일 | 3배 |
| 4명 | 6.40–8.80m | 공통 base와 동일 | 4배 |

### 3.2 거리·배정

| 항목 | 조건 |
| --- | --- |
| 각 참여자→공동 상자 | 전원 2–10m |
| 공동 상자→공동 목표 | 2–5m |
| 초기 참여자 위치 | 상자 뒤쪽 조건 유지 |
| 목표 방향 | 상자 로컬 +Y |
| 구역 배정 | reset 시 로컬 X 순서의 ordered |

### 3.3 그룹 성공·추가 지표

- 실제 공유 상자 중심의 공통 PUSH 조건을 ever 만족하면 그룹 성공. 인원수와 관계없이 시도 하나.
- 구성원 최소 한 명이 유효한 frame에서 새 성공을 기록한다. 개인 구역 전원 도착·동시 접촉·전원 생존은 SR 조건에 추가하지 않는다.
- 메인 eval의 Error는 1.8절의 성공 trial 최소 거리다. Coupled에서는 Point·Region Error가 같으며 final error를 사용하지 않는다. Table 2에는 Error 열을 두지 않는다.

### 3.4 두 지표의 공통 시작·종료 — Table 2 전용

성공·실패를 모두 포함한다. 실제 공유 상자의 reset 직후 중심을 c_init, 첫 action 후부터 기록한 중심을 c_t(t=1,…,T), 공동 목표를 g라 한다.

- **시작 s:** 초기 중심 대비 수평 직선 변위 `||c_t.xy−c_init.xy|| ≥ 0.20m`를 처음 만족한 frame. 성공 trial에서는 최초 성공까지, 실패 trial에서는 기록 끝까지 찾는다.
- 누적 경로 길이·목표 방향·손 접근·속력·높이·접촉력 조건은 추가하지 않는다. 한 번 시작하면 취소하지 않는다.
- **성공 종료 e:** `||c_t−g||_3D ≤ 0.20m`이며 원래 참여자 중 최소 한 명이 해당 frame에서 유효한 최초 시점. 실패는 기록 끝 T를 사용한다.
- 원본 counter: 시작하면 `a=s−1`, 미출발이면 `a=T`. 성공하면 `b=e−1`, 실패면 `b=T`.
- t_coop 분모는 **b−a**. 성공은 e−s, 실패는 T−s+1이다. 시작·종료 frame을 포함한 개수로 임의 변경하지 않는다.
- 50cm 출발 조건은 과거 민감도 비교로만 보존한다. 현재 주 결과에 섞지 않는다.

### 3.5 Cooperative Time Ratio (t_coop) ↑

공동 frame C_t는 **원래 참여자 전원의 양손 중심이 실제 회전 상자의 전체 경계 표면에서 각각 0.12m 미만**일 때 1이다.

- 옆면·윗면·아랫면 전체를 사용한다. 내부 점도 경계까지의 절대 거리를 사용한다. Jerk용 둘레64점까지의 거리가 아니다.
- 손 반지름·접촉력·미는 방향 조건 없음. 추가 생존 mask를 붙이거나 낙상자를 참여 그룹에서 제외하지 않는다. 낙상 후 수동 손 근접도 이 지표에서는 그대로 계산한다.
- 상자의 pose·크기·손 위치는 float32로 계산한다. box 로컬 좌표 p, half-size h에서 `q=abs(p)−h`, 표면 거리는 `abs(||max(q,0)||₂ + min(max(q),0))`다.
- 분자 `N_coop = sum(C_t, t=1…T)`는 **시작 전·성공 후까지 포함한 기록 전체**에서 누적한다.
- trial 값은 `100 × clip(float32(N_coop)/float32(b−a), 0, 1)` (%).
- `0/0 → NaN`은 평균에서 제외. `양수/0 → +Inf → 1`, 음수 비율은 0으로 clip. 미출발을 일괄0으로 바꾸지 않는다.
- 전체 trial 수·유효 수·미출발 수·분모≤0 수·NaN 수·상한/하한 clip 수를 보존한다.

원본 누적·counter·clip 방식을 유지한 **공동 근접 참여 지표**다. 시작부터 완료까지의 순수 접촉 비율이나 힘 동기화를 뜻하지 않는다. 성공 후 근접이나 분모 축소로 100%에 포화될 수 있으므로 상한 clip 비율을 함께 확인한다.

### 3.6 Mean Absolute Jerk ↓

실제 상자의 로컬 중간 높이 z=0에서 XY 직사각형 **전체 둘레에 등간격64점**을 둔다. 시작점 (−L/2,−W/2,0), +Z에서 반시계 방향이며 점 번호는 시간에 따라 고정한다. 변마다16개를 주거나 꼭짓점·전체 표면점으로 바꾸지 않는다.

실제 위치·쿼터니언으로 월드 좌표를 구한다. 물체의 회전·기울기도 반영한다. float32와 **dt=0.0333초**를 사용하고 다음 순차 연산을 유지한다.

```text
v[t] = (p[t+1] − p[t]) / dt
A[t] = (v[t+1] − v[t]) / dt
j[t] = (A[t+1] − A[t]) / dt
```

- 원본 `quat_rotate_dimflex`와 `compute_rim_velocity_variance_norm`의 연산·마스크를 사용한다. 함수 이름과 달리 최종 값은 분산이 아닌 절대 Jerk 평균이다. 보존한 함수는 [checks/teamhoi_reference.py](checks/teamhoi_reference.py)에 있다.
- 0-based 속도 인덱스 t에서 유효 mask는 `a≤t<b`. 가속도는 이웃 속도 mask의 AND, Jerk는 이웃 가속도 mask의 AND다. 따라서 Jerk t는 `a≤t`이고 `t+2<b`인 경우 유효하다.
- 유효 시점·64점·XYZ축의 `abs(j)`를 모두 합하고 `유효 시점 수×64×3`으로 나눈다. 단위 m/s³.
- 벡터 L2·RMS·분산·최종 속도가 아니다. 필터링·평활화·단일 결합 3차 차분으로 교체하지 않는다.
- 유효 Jerk가 없으면 원본 함수는0을 반환한다. **최종 trial 값 J=0은 평균에서 제외**한다. 실제0 Jerk도 제외되며 성공 여부로는 제외하지 않는다. 전체·유효·0 제외 trial 수를 함께 기록한다.

### 3.7 t_coop·Jerk 집계와 출처

- 2/3/4명 각각 성공·실패 전체 trial을 사용한다. t_coop은 NaN, Jerk는0 trial 제외 규칙을 적용한다. 유효 trial이 없으면 N/A.
- 주 요약: **3seed trial 합산 → 인원별 trial 산술평균 → 2/3/4명 동일 가중 평균**.
- 별도 통계: **seed별 인원 평균 → 해당 seed의 2/3/4명 동일 가중 평균 → seed 평균±표본표준편차(ddof=1)**. 인원별 seed 통계도 저장한다.
- 유효 표본 수가 다르면 합산 trial 평균과 seed 평균이 다를 수 있다. 합산 평균에 seed 표준편차를 붙여 `평균±표준편차`로 혼용하지 않는다.
- Group Success Rate는 1.7절의 seed 평균±표본표준편차를 유지한다. t_coop·Jerk와 함께 제시한다.
- TeamHOI 공개 구현을 공동 PUSH에 맞게 조정한 지표다. 시작·종료·물체 형상·t_coop 거리 대상·접촉력 제외를 변경했고 원본 누적·차분·counter·clip·0/NaN 평균 규칙은 보존했다. 인원 동일 가중 요약은 추가 통계다.
- 낮은 Jerk만으로 성공이나 전원의 힘 동기화가 좋아졌다고 해석하지 않는다.

## 4. 장면·테이블 구성

### 4.1 Independent / Mixed

- **Independent:** 전원 독립 과제. 이 개인 SR을 Table 1의 Ind.에 사용한다.
- **Mixed:** I/E/C 랜덤 구성. 한 장면에 모든 관계를 강제하지 않는다. Mixed 안의 solo SR은 Ind. 전용 결과와 별도다.
- 모든 인원수에 같은 Mixed sampler 적용: 사람 순서를 섞고 가능한 I/E/C를 균등 선택 → 1명/2명/2–min(4, 남은 인원)명 배정 → 전원 할당까지 반복. C 크기는 가능한 값에서 균등 선택, 마지막 한 명은 solo.

### 4.2 Table 1 — Versatility & Scalability

| 항목 | 구성 |
| --- | --- |
| 방법 | TokenHSI (+ PUSH), SCT, SCT + Coop. FT, Ours |
| 인원 | 2 / 4 / 8 / 16명 |
| Ind. | Independent 전용 장면의 개인 SR |
| Mixed | 랜덤 장면의 Mixed Interaction Group Success Rate |

### 4.3 Table 2 — Cooperative Performance

각 설정은 **해당 쌍·그룹만 배치**하며 다른 solo·관계 그룹을 추가하지 않는다. 공통 distractor 규칙은 유지한다.

| 표 | 설정 | 실제 인원 N | 토큰 M | 지표 |
| --- | --- | ---: | ---: | --- |
| 2(a) | AT→SIT | 2 | 4 | Enabling Interaction Group Success Rate |
| 2(a) | AT→CLIMB | 2 | 4 | Enabling Interaction Group Success Rate |
| 2(a) | AT→ON_TOP | 2 | 4 | Enabling Interaction Group Success Rate |
| 2(b) | 공동 PUSH 2명 | 2 | 4 | Coupled Interaction Group Success Rate |
| 2(b) | 공동 PUSH 3명 | 3 | 6 | Coupled Interaction Group Success Rate |
| 2(b) | 공동 PUSH 4명 | 4 | 8 | Coupled Interaction Group Success Rate |

- 비교군 후보: TokenHSI (+ PUSH), SCT, SCT + FSM, SCT + Coop. FT, Ours. 실제 포함 행·checkpoint는 실행 manifest로 고정한다.
- 각 행에 512장면×3seeds 적용.
- Enable Avg.: seed별 세 행동 Group SR의 동일 가중 평균. Time은 2.3절을 따른다.
- Coupled 열: 2/3/4명 Group Success Rate·Avg.·t_coop(%)·Mean Absolute Jerk(m/s³). Error는 메인 eval의 두 공통 출력으로 유지한다.
- SR은 1.5절의 XY100%를 적용한다. Time·t_coop·Jerk의 구현·재집계 경로는 [레포별 안내](implementations.md)를 따른다. 기존 구간 내6cm t_coop은 이 버전의 지표가 아니다.

### 4.4 Table 3 — Ablation

| 표 | 모드 | 인원 | 비교 |
| --- | --- | --- | --- |
| 3(a) | Independent | 1 / 2 / 4 / 8 / 16명 | w/o Task-Relation Bias, w/o Task-Action Fusion, SCT |
| 3(b) | Mixed | 2 / 4 / 8 / 16명 | w/o Dependency-Relation Bias, w/o Dependency Feature Fusion, Ours |

- **Bias 두 비교는 재학습 없이 같은 checkpoint의 추론에서 bias 출력만0으로 바꾼다.** 관계·과제 할당·공유 임베딩·fusion·그 외 연산은 보존한다. Actor/critic에 적용하며 가중치·정규화는 바꾸지 않는다.
- Stage1은 Task-Relation Bias, Full은 Dependency-Relation Bias를 제거한다. 각 실제 정책 frame에서 제거 적용을 검증한다.
- Fusion 제거는 대상 연산·학습/추론 조건이 아직 미정이다. Bias 추론 제거 결과와 구분한다.
- 같은 인원·모드의 초기 장면과 실제 첫 정책 입력을 비교한다. 옵션 off의 행동 일치·모델/정규화 hash 불변도 확인한다.

## 5. 기록·이식·검증 계약

### 5.1 최소 기록

인원 N, 물체 토큰 M, 환경 B, 실제 기록 길이 T를 함께 저장한다. 배열의 frame0은 첫 action 후다. 시간값은 제어 step 번호와 함께 기록한다.

| 데이터 | 권장 형태 | 의미 |
| --- | --- | --- |
| 초기 상태 | [B,N,…], [B,physical_boxes,…] | reset 직후 root·DOF·실제 상자 pose/속도·크기·질량·목표·역할·배정·distractor 관측 상태 |
| 역할·소속 | [B,N] | 행동 이름/ID 매핑·Independent/Enabling/Coupled·그룹 ID·선행 ID |
| Root·payload·받침 | [T,B,N 또는 M,3/7] | 실제 물리 상태와 목표, yaw·기울기 포함 |
| 유효 mask | [T,B,N] | 해당 frame의 성공 채점 자격; 낙상 frame은 true, 다음 frame부터 false |
| 손 위치 | [T,B,N,2,3] | 실제 양손 중심, 좌/우 순서 명시 |
| 공동 상자 | [T,B,G,7] | 실제 공유 상자 중심 xyz·쿼터니언 xyzw. 가상 구역 pose로 대체하지 않음 |
| 공동 상자 초기 중심·크기·목표 | [B,G,3] | 초기 중심은 첫 action 전 reset 상태 |
| Group membership | [B,G,N] 또는 동등 ID | 원래 배정된 참여자 집합; 낙상 후에도 유지 |
| 채점 결과 | trial별 | 최초 성공 step·ever·Time·Point/Region 최소 Error·start/end counter·전체 공동 frame 수·Jerk·유효/제외 사유 |

포맷은 NPZ/PT/JSON 등 레포에 맞게 선택하되 의미·정밀도·시간축을 유지한다. Group size가 달라도 Coupled trial 하나를 참가자 수만큼 복제하지 않는다. 포즈는 원래 float32 원소를 보존하고 공간 SR/Error를 계산할 때 float64로 올린다.

### 5.2 실행 manifest와 결과 재사용

- 프로토콜 ID·checkpoint 경로/SHA256·정책 유형·seed 목록·512환경·900step·dt·인원·모드·실제 그룹/행동 건수·코드 버전·물성/자산 설정·초기 snapshot SHA를 기록한다.
- 현재 비교 실행은 하나의 공통 초기 snapshot에서 분기한다. 같은 seed라도 환경 수·RNG 호출 순서·asset 생성이 다르면 동일 장면으로 취급하지 않는다.
- 크기·거리·그룹 배치·경로 검사를 초기 상태에서 확인한다. 배치 실패 재시도는 정책 실행 전에 수행하고 재시도 상한·미해결 배치 수를 기록한다. 실패 trial을 성능을 보고 제거하지 않는다.
- 모든 방법에 같은 시뮬레이터·humanoid/상자 자산·DOF·PD·접촉/마찰·물리 시간 설정을 사용한다. 같은 자산·초기 상태를 제공할 수 없으면 별도 비교로 표시한다.
- 초기 물리 상태뿐 아니라 **첫 행동에 실제 사용한 관측**을 확인한다. reset 직후 관측 재계산으로 다른 상태를 읽지 않도록 한다.
- 결과 재사용은 checkpoint·프로토콜·초기 상태·실제 실행 설정·필수 raw 기록·검증 일치가 확인된 경우만 허용한다. 정의 변경 후 raw 궤적 재집계는 기존 결과를 덮어쓰지 않고 새 버전 결과로 저장한다.
- 64환경 탐색은 정식512환경 결과와 구분한다. 임의로 첫64 trial이512환경의 같은 seed 첫64개라고 가정하지 않는다.

### 5.3 레포 구현 수락 기준

1. [공통 계산 예제](checks/README.md)를 통과한다. 후보 구현을 지정해 같은 입력·기대값과 비교할 수 있다.
2. 원본 정책의 가중치·정규화·평균 행동을 보존한다. Stage1 입력 변환 및 bias 제거 외의 모듈 개입을 명시한다.
3. 초기 snapshot·실제 첫 정책 입력·필수 토큰·group membership·거리·크기·N>4의 중심 간격을 확인한다.
4. Distractor 물리 제거와 관측·attention 보존을 frame마다 확인한다.
5. 저장 궤적을 CPU로 재채점해 최초 성공·SR·건수·Error가 원래 채점과 일치하는지 검증한다. t_coop·Jerk는 trial별 numerator/counter/제외 수까지 확인한다.
6. seed별 수치·합산 수치·표본표준편차·인원 동일 가중 평균을 각각 검산한다. 원본 checkpoint·코드·기존 결과 hash 보존을 확인한다.

공통 계산 예제 통과만으로 시뮬레이션·정책 입력·샘플링까지 검증됐다고 표시하지 않는다. 레포별 지원/미지원·알려진 차이는 [implementations.md](implementations.md)에 기록한다.

## 6. 확정·미정 사항

| 항목 | 상태 |
| --- | --- |
| 공간 SR·Error·Enable 동시 성공·distractor 물리 제거·M=2N·5명 이상 배치 | 본 문서 기준 적용 |
| 공동 PUSH 주 시작 변위 | **20cm 확정**; 50cm는 과거 민감도 비교 |
| Bias ablation | 동일 checkpoint 추론 제거 확정 |
| 평가 seed | 최종 논문 seed는 별도 확정; 현재 실행 seed는 manifest에 기록 |
| 비교군·checkpoint·학습 예산 | 최종 논문 포함 행과 checkpoint 선정 규칙 별도 확정 |
| Fusion ablation | 제거 대상·학습/추론 조건 미정 |
| 배치 구현 이식 | 레포별 재시도 상한·미해결 배치 처리·자산/시뮬레이터 버전 기록 필요 |

규칙 변경 시 버전을 올리고 변경 항목·영향 지표·재집계/재실행 필요 여부를 기록한다. 예전 정의의 숫자와 새 정의의 숫자를 같은 열에 섞지 않는다.
