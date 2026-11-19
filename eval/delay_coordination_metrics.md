# 지연 대응 평가 지표와 집계 기준

2026년 11월 15일 완료한 Enabling 및 Coupled 지연 실험의 현재 지표를 정리한다. **Group SR은 전체 trial 기준이고, Enabling 준비율과 사전점유율은 AT가 성공한 trial만 대상으로 한다.** 후행 실패나 낙상을 이유로 준비율과 점유율을 제외하지 않는다.

## 평가 조건

- Enabling: AT→SIT, AT→CLIMB, AT→ONTOP, 각 2명. 선행자를 시작부터 0/2/4초 정지시킨다.
- Coupled: 공동 PUSH 2/3/4명. 장면 ID modulo 인원수로 고른 한 명을 시작부터 0/2/4초 정지시킨다.
- 모델: TokenHSI40.5k, SCT26k, SCT+FSM26k, SCT+CoopFT7k, DCT V4 2k. 기존 checkpoint로 평가하며 관계 구조는 유지한다.
- 모델과 지연 간 같은 초기 장면을 사용한다. 초기 속도는 0이며, 정지 중에도 충돌은 유지한다. 정지 시간이나 해제 신호를 정책 입력에 추가하지 않는다.
- 조건마다 64환경 × seed 1/6/7, 30초(900step). 각 장면·모델·지연의 Group SR 분모는 192 trial이다.
- 총 270조건, 17,280 trial. 신규 240조건과 기존 35초 기록의 앞 30초를 재집계한 30조건으로 구성된다.

화살표는 항상 **지연 0→2→4초** 순서다. 기존 35초 단일 seed 파일럿과 현재 결과를 섞지 않는다. Post-AT Completion Delay는 현재 보고 지표에서 제외한다.

## 지표와 표본 요약

| 평가 | 지표 | 계산 대상 | 단위와 해석 |
|---|---|---|---|
| 공통 | Group SR | 전체 trial, 실패 포함 | %, 높을수록 좋음 |
| Enabling | AT SR | 전체 trial, AT 실패 포함 | %, 준비율·점유율의 표본 수를 설명하는 보조 지표 |
| Enabling | Preparation Progress 준비율 | 30초 내 AT 성공 trial | %, 목표까지의 순거리 감소율. 점유율과 함께 해석 |
| Enabling | Premature Occupancy 사전점유율 | 30초 내 AT 성공 trial | %, AT 전 배치 영역 점유 프레임 비율. 낮을수록 좋음 |
| Coupled | t_coop | 성공·실패 모두 계산 후 최종 비율이 유한한 trial | %, 전원이 동시에 양손 근접한 프레임의 기존 정규화 지표 |
| Coupled | Jerk | 성공·실패 모두 계산 후 Jerk가 0이 아닌 trial | m/s³, 물체 둘레 점의 3차 차분 크기. SR과 함께 해석 |

## Enabling

### Group SR과 AT SR

trial별로 평가 시간 안에 한 번이라도 Group 성공 조건을 만족하면 성공이다.

\[
\mathrm{Group\ SR}=100\frac{\#\{\text{Group 성공 trial}\}}{\#\{\text{전체 trial}\}}
\]

Enabling Group 성공은 **동일 프레임에서 선행 AT와 후행 과제의 성공 조건이 모두 만족**되는 경우다. 두 사람이 서로 다른 시점에 각각 성공한 것만으로는 Group 성공이 아니다. 판정은 공용 `spatial_success`의 region 기준과 해당 agent의 `alive` 조건을 사용한다.

AT는 대상 물체 중심과 예정 목표의 3차원 거리가 0.20m 이하인 조건이다. 후행 SIT/CLIMB/ONTOP은 기존 과제별 목표 높이와 물체의 전체 XY 영역을 사용하는 XY100% region 기준이며, 높이 오차 허용값은 0.20m다. reward의 성공 판정을 대신 사용하지 않는다.

`t_AT`는 선행 AT 조건이 처음 만족된 시점이다. 이후 AT 조건을 잃어도 최초 시점은 바뀌지 않는다. AT SR은 30초 내 `t_AT`가 존재하는 trial의 비율이다.

### Preparation Progress 준비율

후행자의 root XY 위치와 **선행 AT 물체의 예정 배치 목표 XY** 사이 거리를 `d(t)`로 둔다. 목표는 고정된 예정 위치이며, 현재 움직이는 물체를 따라가지 않는다.

\[
P_{\mathrm{prep}}=100\frac{d(0)-d(t_{\mathrm{AT}}^-)}{\max(d(0),10^{-6})}
\]

- `d(0)`: 초기 snapshot에서의 거리.
- `d(t_AT^-)`: 최초 AT 성공 직전 프레임의 거리. 첫 평가 step에서 AT가 성공하면 초기 snapshot을 사용한다.
- 100%는 목표까지의 초기 거리를 모두 줄인 경우, 0%는 초기와 같은 거리, 음수는 초기보다 멀어진 경우다. 음수를 0으로 자르지 않는다.
- 이동 경로의 총길이나 이동 방향을 측정하지 않는다. 목표를 지나치거나 옆으로 멀어지면 준비율이 낮아질 수 있다.

**AT 실패 trial은 NaN으로 두고 평균에서 제외한다. AT가 성공했다면 후행 실패·Group 실패·낙상 trial도 포함한다.** 낙상 시 준비율을 0으로 바꾸거나 낙상 직전 값으로 고정하지 않으며, 기록된 AT 직전 root 위치를 그대로 쓴다.

준비율이 높다는 것은 목표에 가까워졌다는 뜻이다. 적절한 위치에서 정지해 기다렸다는 직접적인 측정은 아니므로 사전점유율 및 SR과 함께 제시한다.

### Premature Occupancy 사전점유율

\[
P_{\mathrm{occ}}=\frac{100}{N_{\mathrm{pre}}}
\sum_{t<t_{\mathrm{AT}}}\mathbf{1}[\text{후행자 근사 footprint와 배치 영역이 겹침}]
\]

현재 형상 근사는 다음과 같다. 모든 모델에 같은 값을 적용한다.

- 배치 영역 중심: 선행 AT 물체의 예정 목표 XY.
- 배치 영역 모양: 물체 XY 크기의 직사각형, **초기 yaw**로 회전. 각 반크기에 안전 margin 0.10m를 추가한다.
- 후행자 footprint: root XY를 중심으로 한 반경 0.35m 원. 실제 몸체 전체의 투영 형상을 쓰는 것은 아니다.
- 겹침 판정: 후행 root에서 위 직사각형까지의 XY 최단거리가 0.35m 이하이면 점유다.
- 시간 구간: 초기 snapshot부터 최초 AT 성공 직전까지. AT 성공 프레임 자체는 제외한다.

각 trial에서 먼저 점유 프레임 비율을 계산하고, AT 성공 trial 사이에서 평균한다. AT까지 오래 걸린 trial에 프레임 수만큼 더 큰 가중치를 주지 않는다. AT 실패 trial은 제외하며, AT 성공 trial의 후행 낙상 이후 프레임은 별도로 제외하지 않는다.

이 지표는 **예정 배치 영역의 사전점유**다. 실제 신체 충돌, 접촉력, 선행 작업 방해를 직접 판정한 값은 아니다. 표의 0.0%는 반올림일 수 있으므로 모든 모델에 대해 정확히 0이라고 해석하지 않는다.

### AT 성공 표본에 대한 해석

Group SR의 분모는 전체 64 trial/seed지만 준비율·점유율의 분모는 해당 seed에서 AT가 성공한 trial 수다. 따라서 모델·지연마다 준비율·점유율에 들어가는 장면이 달라질 수 있다. 현재 주 표는 모델 간 공통 AT 성공 장면만 교집합으로 고른 결과가 아니다.

AT SR 또는 AT 성공 건수를 함께 제시한다. 이 조건부 표본 차이 때문에 준비율·점유율 평균의 변화만을 동일 episode의 행동 변화로 해석하지 않는다.

## Coupled

### Group SR

평가 시간 안에 공동 물체 중심과 shared-center 목표의 **3차원 거리**가 0.20m 이하이며, 적어도 한 agent가 `alive`인 프레임이 있으면 성공이다. 전체 trial을 분모로 한다. 성공 시점에 모든 사람의 손이 물체에 근접해야 한다는 조건은 Group SR에 추가하지 않는다.

### t_coop

물체의 회전과 크기를 반영한 표면까지의 거리가 **모든 agent의 양손에서 각각 0.12m 미만**이면 그 프레임을 전원 근접으로 센다. 손 거리이며 접촉력이나 실제 미는 힘의 판정은 아니다.

현재 구현은 기존 20cm 규약을 그대로 사용한다. 기록된 step 후 프레임을 0부터 899까지 인덱싱할 때:

- `s`: 최초 성공 프레임까지의 범위에서 물체 중심의 초기 대비 XY 변위가 처음 0.20m 이상인 프레임. 없으면 900.
- `e`: 최초 Group 성공 프레임의 인덱스. 성공하지 못하면 900.
- `C_all`: **900개 기록 프레임 전체**에서 전원이 동시에 양손 근접한 프레임 수. 초기 snapshot은 포함하지 않는다.

\[
t_{\mathrm{coop}}=100\operatorname{clip}\left(\frac{C_{\mathrm{all}}}{e-s},0,1\right)
\]

**분자는 전체 기록 구간이고 분모는 이동 시작부터 성공/종료까지라서 시간 구간이 일치하지 않는다.** 출발 전이나 성공 후의 근접 프레임도 분자에 들어간다. 따라서 현재 값을 엄밀한 “운반 중 협동 시간 비율”이라고 표현하지 않는다. `t_coop`은 여기서 초 단위 시간이 아니라 % 지표다.

성공·실패를 모두 계산하며 별도의 낙상 프레임 mask를 손 근접에 적용하지 않는다. 비율을 [0,1]로 clip한 뒤 NaN 등 비유한 값을 평균에서 제외한다. 미출발 trial이 일괄 제외되는 것은 아니다. 예를 들어 분모가 0일 때 0/0은 NaN으로 제외되지만, 양수/0은 무한대에서 1로 clip되어 포함된다. 음수 비율은 0으로 clip된다. `tcoop_valid`와 `tcoop_clipped`에 유효 수와 원시 비율이 1을 넘은 건수를 저장한다.

### Jerk

물체 로컬 XY 둘레에 균등 배치한 **64개 점(z=0)**을 물체 quaternion과 중심 위치로 월드 좌표로 바꾼다. 중심 병진과 물체 회전을 모두 반영한 XYZ 궤적이며, agent 관절의 Jerk는 아니다.

점 좌표 `p`를 속도→가속도→Jerk 순서로 세 번 차분한다. 기존 구현과 동일하게 float32와 **delta_t=0.0333초**를 쓴다.

\[
v_t=\frac{p_{t+1}-p_t}{\Delta t},\qquad
a_t=\frac{v_{t+1}-v_t}{\Delta t},\qquad
j_t=\frac{a_{t+1}-a_t}{\Delta t}
\]

앞의 `s`, `e`에 대해 차분 시작 인덱스가 `t >= s`이고 `t+2 < e`인 항만 선택한다. 이 mask는 기존 구현 그대로이며, 각 Jerk 항은 원래 좌표 네 프레임을 사용한다.

\[
J=\frac{1}{|\mathcal T|\,64\,3}
\sum_{t\in\mathcal T}\sum_{k=1}^{64}\sum_{c\in\{x,y,z\}}|j_{t,k,c}|
\]

벡터 L2 norm, RMS, 분산이 아니라 **시간·점·XYZ 성분에 대한 절댓값 평균**이다. 원본 함수명 `compute_rim_velocity_variance_norm`과 실제 계산 내용이 다르므로 함수명으로 지표를 설명하지 않는다.

유효 mask가 없으면 구현상 0이 된다. 성공·실패를 모두 계산한 다음 **Jerk가 정확히 0인 trial을 제외**하고 평균한다. 따라서 미출발·구간 부족과 실제 0 Jerk가 동일하게 제외될 수 있다. 유효 수는 `jerk_valid`에 저장한다. 실패를 일괄 0 처리하는 규칙은 아니다.

낮은 Jerk가 반드시 더 나은 협동을 뜻하지는 않는다. 움직임이 적거나 실패한 trial도 낮게 나올 수 있으므로 Group SR 및 t_coop과 함께 해석한다.

## 평균과 표준편차

1. 각 장면·모델·지연·seed 안에서 trial별 지표를 먼저 평균한다. 지표별 포함·제외 규칙은 위 표를 따른다.
2. 개별 장면 표는 3seed 평균과 seed 간 표본표준편차(`ddof=1`)를 보고한다.
3. 종류별 평균은 각 seed 안에서 Enabling 3종 또는 Coupled 2/3/4명을 **동일 가중치 1/3**로 평균한 뒤, 다시 3seed 평균과 표본표준편차를 구한다.

Group SR은 trial 수가 모두 같으므로 위 평균이 전체 성공 건수/전체 trial과 같다. 준비율·점유율·t_coop·Jerk는 유효 trial 수가 다를 수 있어 전체 유효 trial을 합친 pooled 평균과 일반적으로 다르다. 종류별 표준편차는 장면 간 편차나 개별 표준편차의 평균이 아니다.

## 구현과 결과 위치

| 내용 | 파일 |
|---|---|
| 현재 30초 채점과 종류별 집계 | [analyze_suite.py](../FSM/dct/analyze_suite.py) |
| 준비율과 점유율 | [analyze_coordination_behavior.py](analyze_coordination_behavior.py)의 `behavior_arrays` |
| Coupled 시작·종료 인덱스와 t_coop | [coupled_cooperation_metrics.py](coupled_cooperation_metrics.py)의 `counters`, `ratios`, `surface_contact` |
| 손과 물체 표면 거리 | [table2_metrics.py](table2_metrics.py)의 `surface_distance` |
| 공용 성공 조건 | [evaluation_protocol.py](../tokenhsi/utils/evaluation_protocol.py), [stage2_xy100.py](stage2_xy100.py) |
| Jerk 원본 함수 | [task_util.py](../output_etc/check/coupled_cooperation_four_models_20261111/reference/task_util.py) |
| 전체 장면 평균과 표준편차 | [report.md](../output_etc/check/dct_delay_env64_seeds167_30s_20261115/report.md) |
| seed별 값과 유효 수 | [summary.json](../output_etc/check/dct_delay_env64_seeds167_30s_20261115/summary.json)의 `runs`, `macro_per_seed` |
| 완료 검증 | [completion_verification.json](../output_etc/check/dct_delay_env64_seeds167_30s_20261115/completion_verification.json) |

trial별 배열은 같은 출력 폴더의 `analysis/<setting>/<model>_d<delay>_seed<seed>.npz`에 있다. Enabling의 `first_at`, `prep_percent`, `occupancy_percent`와 Coupled의 `tcoop`(0~1), `jerk`, 공통 `first_group`을 확인할 수 있다. `first_at`과 `first_group`은 1부터 세는 step 번호이며 미성공은 -1이다.

## 최종 종류별 평균

아래 표는 3seed 평균만 표시한다. 각 셀은 지연 0→2→4초 순서이며, 표준편차와 개별 장면 값은 위 전체 결과표에 있다. Enabling은 AT 성공률을 함께 표시한다.

### Enabling

| 모델 | Group SR % | 준비율 % | 사전점유율 % | AT SR % |
|---|---|---|---|---|
| TokenHSI | 10.9→5.9→3.8 | 73.8→70.4→67.5 | 31.9→40.0→41.2 | 51.4→51.6→53.8 |
| SCT | 30.4→26.7→20.1 | 55.3→63.1→64.0 | 21.3→34.4→43.2 | 76.2→69.1→66.5 |
| SCT+FSM | 41.1→41.0→40.6 | 0.0→0.0→0.0 | 0.0→0.0→0.0 | 81.4→80.2→81.9 |
| SCT+CoopFT7k | 53.6→53.5→55.0 | 12.1→18.8→23.3 | 0.0→0.1→0.2 | 97.4→96.4→96.0 |
| DCT | 85.9→84.2→82.3 | 20.9→28.2→33.2 | 0.0→0.2→1.2 | 98.6→97.0→96.2 |

### Coupled

| 모델 | Group SR % | t_coop % | Jerk m/s³ |
|---|---|---|---|
| TokenHSI | 60.1→51.2→45.0 | 45.7→39.8→29.9 | 18.3→15.3→12.1 |
| SCT | 74.7→73.4→59.7 | 51.9→51.0→36.1 | 22.7→21.6→21.6 |
| SCT+FSM | 72.6→74.3→75.2 | 59.9→64.7→65.6 | 31.4→30.9→31.0 |
| SCT+CoopFT7k | 84.9→81.6→78.1 | 82.5→76.1→64.3 | 14.2→13.4→10.4 |
| DCT | 90.8→88.0→81.9 | 84.8→80.9→70.5 | 13.3→12.6→10.2 |
