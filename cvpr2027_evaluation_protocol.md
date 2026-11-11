# CVPR2027 평가 프로토콜

기본 크기·거리·공간 판정은 기존 통합 eval 기준안이다. 추가 결정 항목은 5절에 둔다. 새 프로토콜의 코드 반영은 별도다.

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
- 장면 생성·배치 검사 후 distractor만 물리 장면 밖으로 이동한다. 관측 토큰·형상/BPS는 유지하고 위치·회전·선속도·각속도는 초기 샘플 값으로 고정한다. 보상·성공 판정은 실제 작업 물체 상태를 사용한다.
- 4명 관측용 distractor 배치: 환경 중심 반경 4.5m 원 안에서 면적 기준 균등 추출 후 간섭 시 재추출. 5명 이상 배치 영역은 미정.

### 1.5 공간 성공 조건

단독·enable·coupled에 공통 적용한다. 주지표는 Region, Point는 보조 기준안이다.

| 행동 | 검사점 | Region 조건 | 목표 높이 |
| --- | --- | --- | --- |
| AT | 물체 중심 | 지정 목표까지 3D 거리≤20cm | 지정 목표 Z |
| PUSH | 물리 상자 중심 | 지정 목표까지 3D 거리≤20cm | 지정 목표 Z |
| SIT | pelvis/root | 받침 각 축 중앙 80% + 높이 오차≤20cm | 상판 최고 Z+0.10m |
| CLIMB | pelvis/root | 받침 각 축 중앙 80% + 높이 오차≤20cm | 상판 최고 Z+0.94m |
| ON_TOP | payload 중심 | 받침 각 축 중앙 80% + 높이 오차≤20cm | 상판 최고 Z+payload 직립 높이/2 |

- 중앙 80%: 받침 yaw 로컬 좌표의 `|x|≤0.4Lx, |y|≤0.4Ly`. 검사점 기준.
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
| Enabling Interaction Group Success Rate | 같은 유효 frame에서 선행·후행이 동시에 성공한 적이 있는 쌍 |
| Coupled Interaction Group Success Rate | 공유 상자 중심이 공동 목표에 ever 도달한 그룹 |
| Mixed Interaction Group Success Rate | 같은 seed의 위 세 SR을 1/3씩 평균 |

- Seed별 SR·Mixed를 먼저 계산한 뒤 3seeds 평균±표본표준편차(ddof=1) 보고.
- 유형별 성공/시도 수와 C2/C3/C4 분포 저장. Coupled 내부는 실제 그룹 하나당 시도 하나.
- 유형 분모가 0이면 해당 SR과 Mixed는 null. 관계 전용 평가는 해당 SR만 보고.
- 초기 상태·per-frame pose·생존·낙상·성공·실행 버전을 저장하고 CPU 재채점으로 검증.

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

- `ever(선행 AT AND 후행 행동)`: 같은 frame에서 두 행동의 성공 조건을 모두 만족해야 한다. 서로 다른 frame의 개별 ever 성공을 AND하지 않는다. 추가 유지시간 조건은 없다.
- 두 사람 모두 채점 유효한 frame에서 판정한다(각자의 낙상 frame 포함, 이후 제외). 이미 달성한 동시 성공은 보존한다.
- 한 쌍을 시도 하나로 집계한다.
- 완료 시간 기준안: 두 행동이 처음 동시에 성공한 frame의 시간, 실패는 30초. 평균 방식은 미정.

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
- 최종 오차 기준안: 모든 시도의 공유 상자 중심–목표 3D 거리(cm). 측정 시점·전원 낙상 처리·C2/C3/C4 평균 방식은 미정.
- 공동 접촉 시간 비율 `t_coop`: 추후 결정.

## 4. 장면·테이블 구성

### 4.1 Independent / Mixed

- **Independent:** 전원 독립 과제. 이 개인 SR을 Table 1의 Ind.에 사용한다.
- **Mixed:** I/E/C 랜덤 구성. 한 장면에 모든 관계를 강제하지 않는다. Mixed 안의 solo SR은 Ind. 전용 결과와 별도다.
- Mixed sampler 기준안: 사람 순서를 섞고 가능한 I/E/C를 균등 선택 → 1명/2명/2–min(4, 남은 인원)명 배정 → 전원 할당까지 반복. C 크기는 가능한 값에서 균등 선택, 마지막 한 명은 solo.

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
- Enable Avg. 기준안: seed별 세 행동 SR의 동일 가중 평균.
- 추가 열: Enable Time, Coupled Error·t_coop. 세부 산식은 미정.

### 4.4 Table 3 — Ablation

| 표 | 모드 | 인원 | 비교 |
| --- | --- | --- | --- |
| 3(a) | Independent | 1 / 2 / 4 / 8 / 16명 | w/o Task-Relation Bias, w/o Task-Action Fusion, SCT |
| 3(b) | Mixed | 2 / 4 / 8 / 16명 | w/o Dependency-Relation Bias, w/o Dependency Feature Fusion, Ours |

† 추론 시 제거, ‡ 해당 요소 없이 재학습. 같은 인원·모드의 공통 장면으로 평가한다.

## 5. 추가 결정 사항

| 항목 | 미정 내용 |
| --- | --- |
| Distractor | 5명 이상 관측용 distractor 배치 영역 |
| t_coop | 밀기 구간, 접촉 판정, 속력·힘 문턱, 분모 0·낙상 처리·그룹 평균 |
| 공통 기본값 | 크기·거리·ground 높이, Region 주지표·Point 보조 및 목표 높이·기울기 처리의 최종 채택 |
| Mixed sampler | 순차 균등 선택을 모든 인원수에 적용할지 |
| Time / Error / Avg. | 실패·전원 낙상 처리, 측정 시점, 행동별·그룹 크기별 평균 |
| 모델·checkpoint | 비교군 학습 범위·예산, Ours RL/distillation, checkpoint 선정·validation/test seed 분리 |
| FSM·ablation | 대기·해제·동기화·freeze 방식, 사용 정보, 제거 모듈·재학습 조건 |
| 재현성 | 초기 배치 재시도 상한, 공통 snapshot·실행 버전·결과 재사용 규칙 |
