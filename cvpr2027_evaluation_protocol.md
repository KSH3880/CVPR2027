# CVPR2027 평가 프로토콜

현재 평가 조건을 정리한다. SR·Error 집계는 메인 eval을 재사용하며, 장면 생성·Stage1 입력 변환·XY100% 적용은 `eval/`의 보조 경로를 사용한다. 일반 메인 eval의 XY 기본값80%와 학습 조건은 별개다. 추가 결정 항목은 5절에 둔다.

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
- 보조 평가에서 `eval/stage2_xy100.py`로 적용한다. 일반 메인 eval 기본값80%를 자동 변경하지 않는다.
- Point: 목표점까지 3D 거리≤20cm. AT/PUSH는 Region과 동일.
- 실제 물리 상태를 float64로 채점한다. Virtual alias·학습 reward/gate·추가 AT 높이 1mm 조건은 사용하지 않는다.
- 추가 접촉·속도·손 놓기·발 높이·유지시간 조건 없음.

### 1.6 성공 기록·낙상

- 첫 action 이후부터 채점. `ever`는 유효 frame에서 한 번 이상 성공.
- 낙상 frame까지 채점하고 해당 사람의 제어·관절 motor를 해제한다. 수동 물리 actor로 남기며 다른 사람은 계속 수행한다.
- 낙상 전 성공은 보존한다. 한 명의 낙상으로 그룹 전체를 실패 처리하지 않는다.
- 전원 낙상 시 조기 종료 가능. 낙상 판정은 공통 접촉·높이·grace 규칙 적용.

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
- `eval/stage1_protocol_policy.py`에서 원본 Stage1 행동·정규화 일치와 가중치 불변을 검증한다.
- **SCT + FSM:** 같은 SCT 입력·정책에 [FSM 대기·해제](FSM/README.md)를 추가한다. Enable은 선행 AT 성공까지 후행을 고정하고, Coupled는 자기 그룹의 실제 상자에 양손12cm 미만으로 접근한 도착 이력을 누적한다. 먼저 도착한 사람은 고정하고 마지막 참여자가 도착하면 그 사람은 고정하지 않은 채 전원 한 번 해제한다. 사람의 root·DOF 상태를 서브스텝마다 복원하는 방식이며 대기 중 충돌체·반력은 유지한다. 공통 SR은 변경하지 않는다.

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
- Time은 `eval/table2_metrics.py`의 Table 2 보조 평가에서 기록한다.

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

### 3.4 Cooperative Time Ratio — Table 2 전용

`t_coop = 100 × (밀기 구간에서 전원이 동시에 양손 접촉한 frame 수) / (밀기 구간 frame 수)`.

[TeamHOI의 협력 시간 개념](https://github.com/sail-sg/TeamHOI/blob/6fdc9885f8a9c82854be99adca6f2570e67ac63a/teamhoi/env/tasks/humanoid_multi_amp_teamhoi_task.py#L690-L715)을 PUSH에 맞게 수정한다. 원본의64개 후보 점·12cm·추가 접촉력 조건 대신 실제 상자 표면·6cm·거리 조건만 사용한다.

| 항목 | 규칙 |
| --- | --- |
| 시작 | 상자 중심이 초기 위치에서 수평 20cm 이상 이동한 최초 프레임; 손 접근 조건 없음 |
| 종료 | 첫 목표 도달 또는 episode 종료 |
| 중간 정지 | 밀기 구간에 포함 |
| 접촉 | 전원이 같은 프레임에 양손 조건 만족; 실제 공유 상자 표면까지3D 거리, 힘 조건 없음 |
| 접촉 문턱 | 손 중심→상자 표면6cm 미만. 손 반지름을 추가하지 않음; 8/12cm 재집계도 보존 |
| 실패 trial | 실제 접촉 비율을 집계하며 목표 도달 실패만으로 0% 처리하지 않음 |
| 밀기 시작 실패 | 0% |
| 낙상 | 최초 참여자 집합 유지; motor 해제 후 수동 접촉은 인정하지 않음 |
| 프레임 | 30Hz 제어 프레임; 시작·최초 성공 프레임 모두 포함. 원시 기록은 에피소드 끝까지 저장 |
| 집계 | Trial별 비율 평균 → 2/3/4명 동일 가중 평균 → seed 평균±표본표준편차 |

성공 여부와 무관하게 전체 trial을 포함한다. `eval/table2_metrics.py`에서 기록하며 `cooperation.npz`의 양손·실제 상자 원시 좌표로 GPU 재실행 없이 거리 문턱을 재집계한다.

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

- 비교 방법: TokenHSI (+ PUSH), SCT, SCT + FSM, Ours.
- 각 행에 512장면×3seeds 적용.
- Enable Avg.: seed별 세 행동 Group SR의 동일 가중 평균. Time은 2.3절을 따른다.
- Coupled 열: 2/3/4명 Group SR·Avg.·t_coop(%). Error는 제외하며 메인 eval의 두 Error 출력은 유지한다.
- Table 2 전용 실행·Time·t_coop·동일 가중 요약은 `eval/run_table2.py`·`summarize_table2.py`를 사용한다. SR은 1.5절의 XY100%를 적용한다.

### 4.4 Table 3 — Ablation

| 표 | 모드 | 인원 | 비교 |
| --- | --- | --- | --- |
| 3(a) | Independent | 1 / 2 / 4 / 8 / 16명 | w/o Task-Relation Bias, w/o Task-Action Fusion, SCT |
| 3(b) | Mixed | 2 / 4 / 8 / 16명 | w/o Dependency-Relation Bias, w/o Dependency Feature Fusion, Ours |

† 추론 시 제거, ‡ 해당 요소 없이 재학습. 같은 인원·모드의 공통 장면으로 평가한다.

## 5. 추가 결정 사항

| 항목 | 미정 내용 |
| --- | --- |
| 공통 기본값 | 크기·거리·ground 높이 및 목표 높이·기울기 처리의 최종 채택 |
| 모델·checkpoint | 비교군 학습 범위·예산, Ours RL/distillation, checkpoint 선정 |
| 평가 seed | 사용자 탐색 후 확정 |
| FSM·ablation | FSM 대기·해제의 구현은 `FSM/README.md`; ablation의 제거 모듈·재학습 조건은 미정 |
| 재현성 | 초기 배치 재시도 상한, 공통 snapshot·실행 버전·결과 재사용 규칙 |
