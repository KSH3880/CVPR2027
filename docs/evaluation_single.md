# 단독 행동 평가 인수인계: TokenHSI식 점 ever / 우리 영역식 ever

작성·코드 확인: **2026-10-22**. 프로토콜 ID: **single_match_region_v1**.

이 문서는 평가의 공통점과 차이점을 먼저 요약하고, 뒤에서 사람·상자·목표 샘플링과 물리·종료·통계를 상세히 설명한다. 코드 링크는 저장소 기준 상대 경로다.

## 한눈에 보는 공통 조건과 차이점

공개 TokenHSI와 우리 모델 모두 **같은 궤적을 두 성공 기준으로 채점**한다. 아래 ‘TokenHSI식’과 ‘우리 영역식’은 모델 이름이 아니라 평가 기준이다.

**공통 조건**

| 항목 | 두 평가의 공통점 |
| --- | --- |
| 환경 | 환경당 사람 1명, 행동별 독립 평가 |
| 시간 | 최대 **30초**. 중간에 넘어져 종료되면 그 시점까지 평가 |
| 횟수 | **512환경 × 3seed**, 행동당 총 1,536시도 |
| 초기화 | 같은 locomotion 초기화와 사람·상자·목표 샘플링 |
| 물체 | 같은 상자 크기·위치·물리 조건 |
| 집계 | **ever**: 첫 episode에서 조건을 한 번이라도 만족하면 성공 |
| 통계 | seed별 성공률의 평균 ± 표본 표준편차 |
| 추가 조건 | 접촉·손 놓기·발 높이·방향·속도·유지시간을 성공 조건으로 추가하지 않음 |

예를 들어 10초에 성공했다가 15초에 떨어져도 **ever는 성공**이다. 종료 순간의 final은 현재 표에 넣지 않는다. 낙상 종료에는 접촉·신체 높이 검사가 있으며, 이는 성공 조건과 별개다.

**다른 것은 성공으로 인정하는 공간이다.**

- **TokenHSI식:** 목표 **점으로부터 3D 거리 20cm 이내**.
- **우리 영역식:** 목표 **상판 내부 XY 영역**에 들어오고, 목표 높이와의 **Z 오차가 20cm 이내**.
- **AT는 두 기준이 완전히 같다.**

검사점은 AT·ON_TOP에서는 **운반 상자의 중심**, SIT·CLIMB에서는 **골반/root**이다.

| 행동 | 목표 위치 | TokenHSI식 | 우리 영역식 |
| --- | --- | --- | --- |
| AT | 운반 목표점 | 상자 중심의 **3D 거리 ≤20cm** | **동일** |
| ON_TOP | 받침 상판 중심 + 운반 상자 직립 높이의 절반 | 상자 중심의 **3D 거리 ≤20cm** | 상자 중심 XY가 **내부80%** AND 목표 높이와 **Z 오차 ≤20cm** |
| SIT | 상판 중심 + **10cm** | 골반의 **3D 거리 ≤20cm** | 골반 XY가 **내부80%** AND 목표 높이와 **Z 오차 ≤20cm** |
| CLIMB | 상판 중심 + **94cm** | 골반의 **3D 거리 ≤20cm** | 골반 XY가 **내부80%** AND 목표 높이와 **Z 오차 ≤20cm** |

내부80%는 **가로·세로 각각 80%**다. 상자의 yaw 회전을 반영한 로컬 좌표에서:

$$
|x_{\mathrm{local}}|\le 0.4L_x,\qquad |y_{\mathrm{local}}|\le 0.4L_y
$$

각 가장자리에서 10%씩 제외하므로 실제 면적은 전체의 **64%**다. 몸이나 상자 전체가 들어오는지 검사하는 것은 아니고 **위 표의 중심점만 검사한다.**

**20cm를 쓰더라도 높이와 XY를 처리하는 방식이 다르다.**

TokenHSI식:

$$
\sqrt{\Delta x^2+\Delta y^2+\Delta z^2}\le0.20
$$

우리 영역식(ON_TOP·SIT·CLIMB):

$$
\text{XY가 내부80\%}\quad\land\quad|\Delta z|\le0.20
$$

점 기준은 높이 오차가 커질수록 허용 XY 거리가 줄어들고, 영역 기준은 XY와 Z를 독립적으로 검사한다. **단순히 XY만 바꾸면서 원래의 3D 거리 조건을 그대로 유지한 것은 아니다.**

또한 **영역식이 항상 더 높은 것은 아니다.** 작은 상자에서는 중심에서 15cm 떨어진 위치가 점 기준으로는 성공이어도 내부80% 밖이라 영역 기준으로 실패할 수 있다.

**우리 표의 ‘TokenHSI식’과 원본 TokenHSI 평가 전체도 구분해야 한다.** 원본의 점 성공 조건과 ever 집계를 따르지만, 두 모델 모두 **공통 상자·30초·우리 초기화**에서 평가한다. 원본 SIT 의자와 CLIMB 가구 혼합 환경을 그대로 재현한 것은 아니고, **ON_TOP은 Carry를 받침 위 배치로 확장한 과제**다.

현재 평가에는 예전의 **ON_TOP 높이 1mm, SIT +12cm±7cm, CLIMB 발 높이 조건이 없다.** 이 조건들은 기존 학습 보상/이전 `single` 평가와 구분해야 한다.

아래부터는 샘플링·수식·물리·통계의 상세 명세다.

---

## 1. 무엇을 비교하는가

**정책과 평가 기준은 서로 다른 축이다.** 공개 TokenHSI, 우리 Fixed, 우리 Point fine-tuning 각각의 행동 궤적을 두 기준으로 채점한다.

| 축 | 의미 |
| --- | --- |
| 정책 | 실제 행동을 생성하는 checkpoint. 공개 TokenHSI Unified Stage1 40k / 우리 Fixed 70~85k / 원본78k에서 Point fine-tuning한 +1k·+2k·+3k |
| TokenHSI식 점 ever | 과제별 검사점이 목표점에서 3D 거리20cm 이내에 한 번이라도 도달했는가 |
| 우리 영역식 ever | AT는 점 기준 그대로. 다른 과제는 검사점 XY가 상판 내부80%에 있고 목표 높이와 Z 오차20cm 이내에 동시에 한 번이라도 도달했는가 |

정책당 seed별 rollout을 한 번 생성하고 같은 궤적에 두 판정을 적용한다. 점 지표와 영역 지표를 위해 별도의 장면을 샘플링하지 않는다. 서로 다른 정책끼리는 같은 seed의 기록된 초기 장면을 대조하지만, 이후 행동·궤적은 당연히 달라진다.

여기서 ‘TokenHSI식’은 **성공 판정 방식**을 뜻한다. 공개 TokenHSI 논문 전체 benchmark를 그대로 재현했다는 뜻은 아니다. 이번 비교는 공통 상자·우리 환경·30초 조건으로 맞춘 평가이며 ON_TOP은 별도 확장 과제다.

## 2. 실행 단위와 공통 조건

| 항목 | 설정 |
| --- | --- |
| 사람 | 환경당1명. 다인·협동 평가 아님 |
| 행동 | AT, ON_TOP, SIT, CLIMB를 각각 독립 실행 |
| 상호작용 물체 | AT/SIT/CLIMB는1개, ON_TOP은 payload+받침2개. 바닥·운반 선반 등 보조 physics actor는 이 개수에 포함하지 않음 |
| 정책 실행 | checkpoint와 정규화를 고정한 deterministic 추론. 평가 중 학습 없음 |
| 시간 설정 | 모든 과제 `episode_length=900`, 제어 간격1/30초, 명목상30초 |
| 실제 종료 경계 | 코드의 `progress >= 900-1`로 일반 시간 종료 trace는899step(약29.97초). 초기 physics warmup은 평가 성공 frame에 포함하지 않음 |
| 표본 | seed0·1·2를 각각 별도 프로세스로 실행, 각512환경의 첫 episode1회 |
| 정책·행동당 표본 | 512×3 = **1,536시도** |
| 정책당 전체 | 4행동×3seed =12개 실행,6,144시도 |
| 시작 skill | `--eval_skills loco --eval_skill_probs 1.0`. 들고 있거나 앉아 있거나 올라간 상태로 시작하지 않음 |
| 성공 누적 | 첫 episode에서 종료 frame까지 ever. 종료 후 reset된 episode는 제외 |
| 성공 후 | 성공했다고 즉시 종료하지 않음. 이후 실패해도 이미 얻은 ever 성공은 유지 |
| 조기 종료 | 기존 낙상 판정 유지. 30초를 모두 보장하지 않음 |
| 통계 | seed별 성공률3개의 평균(%) ± 표본 표준편차(%p), `ddof=1` |

문서 작성 시 메인 표는20개 정책/checkpoint,240개 실행,122,880시도다. 앞으로 모델을 추가하면 이 전체 건수는 바뀌지만 정책별1,536/6,144시도 규칙은 같다.

평가에서는 네 행동을 같은 수로 별도 실행한다. 학습 때의 SIT:CLIMB:ON_TOP=4:10:13 같은 행동 샘플링 비율을 적용하지 않는다.

## 3. 사람 초기화와 사람–출발 상자 거리

### 3.1 사람의 자세·위치·방향

- 모션 설정은 `tokenhsi/data/dataset_loco_sit_carry_climb.yaml`, humanoid asset은 `tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml`이다.
- 평가 skill은 loco만 사용한다. 해당 motion library에서 설정된 motion 가중치로 motion을 고르고 `stateInit=Random` 경로의 `sample_time_rsi`로 허용 초기 시점을 고른다.
- 모션에서 root 위치·회전·선속도·각속도 및 관절 위치·속도를 가져오고, root 위치에 환경 origin과 사람 spawn offset을 더한다.
- 사람의 방향을 ‘항상 상자를 보도록’ 강제하지 않으며, 사람 위치를 목표 주변 사각형에서 별도로 균등 추출하지 않는다.
- 학습용 행동별 RSI 혼합비·size-aware RSI screening/cache는 이 평가의 초기 skill 분포가 아니다. 평가에서는 공통 loco 초기화를 사용한다.
- 공개 TokenHSI 정책도 이번 비교에서는 위와 같은 **우리 환경의 초기화 경로**에서 시작한다.

### 3.2 출발 상자 XY

사람 root XY를 `r0`, 출발 상자 중심 XY를 `b0`라 두면:

```text
v ~ N(0, I₂)
u = v / ||v||₂                 # 원주 방향은 균등
ρ ~ U(1, d_max)                # 반경 자체를 균등 추출
b0 = r0 + ρ u
```

| 과제 | 여기서 출발 상자는 | d_max | 사람 root–출발 상자 중심 XY 거리 |
| --- | --- | ---: | --- |
| AT | 운반할 payload | 10m | U(1,10)m |
| ON_TOP | 운반할 payload | 10m | U(1,10)m |
| SIT | 앉을 상자 | 5m | U(1,5)m |
| CLIMB | 올라갈 상자 | 10m | U(1,10)m |

**모든 거리는 수평 XY 중심 간 거리**다. 3D 거리, 사람 발끝에서 상자 면까지의 거리, 상자 표면 사이 간격이 아니다. 반경이 균등하므로 원환 면적에 대한 균등 분포도 아니다. 상자 크기에 따라 가장 가까운 표면까지의 거리는 달라진다.

상자 초기 yaw는 U(0,2π), 초기 roll/pitch는0이며 선속도·각속도는0으로 설정한다. 동적 상자는 이후 중력·접촉으로 움직일 수 있다.

구현: [stage1_sampling.py](../../tokenhsi/utils/stage1_sampling.py), [humanoid_ma_carry.py](../../tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py)의 `_reset_ref_state_init`, `_reset_boxes` 관련 경로.

## 4. 상자 크기: 일반 single_match

모든 수치는 **전체 변 길이(m)**이며 half extent가 아니다. X/Y는 상자 로컬 축이다. yaw로 회전된 월드축 bbox 폭과 혼동하지 않는다. 크기는 환경 생성 때 배정하고 해당 평가 episode 동안 재추출하지 않는다.

| 과제·대상 | X | Y | Z | 추출 방식 |
| --- | --- | --- | --- | --- |
| AT payload | 아래9종 목록 | 아래9종 목록 | 아래9종 목록 | 각 유형 최소1개 포함 후 나머지503개 균등 복원 추출,512개 순서를 shuffle |
| ON_TOP payload | 0.20~0.60 | 0.20~0.60 | 0.20~0.60 | 각 축 독립,0.05m 격자 균등 |
| ON_TOP 받침 | 0.40~0.80 | 0.40~0.80 | 0.25~0.50 | 각 축 독립,0.05m 격자 균등 |
| SIT 상자 | 0.25~0.80 | 0.25~0.80 | 0.35~0.50 | 각 축 독립,0.05m 격자 균등 |
| CLIMB 상자 | 0.40~0.80 | 0.40~0.80 | 0.20~0.70 | 각 축 독립,0.05m 격자 균등 |

범위 양 끝을 모두 후보에 포함한다. 예: ON_TOP 받침 Z는 `{0.25,0.30,0.35,0.40,0.45,0.50}`. X=Y가 되도록 묶거나 모든 축에 같은 배율을 적용하지 않는다.

AT의9종 전체 크기는 다음과 같다. 이 목록은 원본 Carry의 `testSizes`에서 가져온다.

```text
(0.22,0.22,0.22), (0.27,0.27,0.27), (0.32,0.32,0.32),
(0.37,0.37,0.37), (0.42,0.42,0.42), (0.47,0.47,0.47),
(0.52,0.52,0.52), (0.57,0.57,0.57), (0.30,0.30,0.40)
```

SIT/CLIMB는 위 규격의 **상자만** 사용한다. 의자·캐비닛·테이블 mesh를 섞지 않는다. 코드의 일반 `box.build.testSizes`만 보고 평가 크기를 판단하면 안 된다. 전용 evaluator의 AT override와 `taskSizeRanges`가 실제 크기를 결정한다.

구현: [평가 크기 override](evaluate.py), [크기 sampler](../../tokenhsi/utils/size_rsi.py), [원본 Carry 목록](../../tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml). 실행 당시 설정은 각 run의 `resolved_env.yaml`에 남는다.

## 5. 운반 도착 목표와 상자 높이

### 5.1 AT 목표 / ON_TOP 받침의 XY

환경 origin을 `o`, 사람 초기 root를 `r0`, 초기 payload 중심을 `p0`라 하면:

```text
q_xy ~ o_xy + U([-4.5,4.5] × [-4.5,4.5])
accept iff ||q_xy - r0_xy||₂ >= 1m
       and ||q_xy - p0_xy||₂ >= 1m
otherwise resample the rejected rows
```

재추출은 코드상 최대256회이며 실패하면 오류를 낸다. 이 방향을 payload 방향과 맞추거나 사람–payload–목표를 일렬 배치하지 않는다.

- AT: `q_xy`가 운반 목표 XY다.
- ON_TOP: `q_xy`에 받침 상자를 둔다. 평가 중 목표는 움직인 받침의 **현재** 위치·높이를 따른다.
- **사람→목표와 payload→목표는 각각 최소1m**지만, 이 두 거리를 U(1,10)m 또는 U(1,4.5)m로 직접 뽑는 것이 아니다.
- 목표의 사각형 범위는 **환경 origin 기준**이다. payload나 사람을 중심으로 한 ±4.5m 사각형이 아니다.
- 이 조건은 중심 간 거리 조건이며 상자 bbox 간 간격/비충돌 판정과 같지 않다.
- SIT/CLIMB는 별도 운반 도착점을 샘플링하지 않는다. 배치된 해당 상자가 곧 목표다.

### 5.2 AT/ON_TOP 출발 payload와 AT 도착 목표의 Z

payload 직립 전체 높이를 `H`, 높임 여부를 `B ~ Bernoulli(0.5)`, `u ~ U(0,1)m`라 하면:

```text
h = B*u
z_center_base = min(H/2 + h, 1.2 - H/2)
출발 payload 중심 Z = z_center_base + 0.05m
AT 도착 목표 중심 Z = z_center_base       # 목표용으로 별도 높이 추출
선반 상판 Z = z_center_base - H/2
```

- 50%는 바닥 선택,50%는0~1m의 지지 높이를 뽑은 뒤 payload 상단≤1.2m가 되도록 clamp한다.
- **잘린 구간 안에서 다시 균등 추출하지 않는다.** clamp 때문에 높이 상한에 확률 질량이 생길 수 있다.
- 출발에만5cm의 접촉 여유를 더한다. 따라서 여유를 더한 초기 payload 상단은1.25m까지 가능하다. AT 목표에는5cm를 더하지 않는다.
- 선반 자체는 payload 밑에 배치한다. 바닥 선택인 선반은 지면 바로 아래에 둔다.
- ON_TOP 도착 높이에는 위 AT 랜덤 목표 높이를 적용하지 않는다. 받침은 초기 바닥 배치이고 목표 Z는 **현재 받침 상판+payload 직립 반높이**다.
- SIT/CLIMB 상자의 초기 중심 Z는 자기 높이의 절반이며 바닥 위에 둔다. 별도 높은 선반을 붙이지 않는다.

구현: [stage1_platforms.py](../../tokenhsi/utils/stage1_platforms.py)의 `sample_payload_height`, `reset_stage1_platforms`.

## 6. 물리·정책 입력 조건

| 항목 | 현재 평가 조건 |
| --- | --- |
| 시뮬레이터 | Isaac Gym / GPU PhysX, checkpoint의 저장된 env+sim 설정 사용 |
| 시간 간격 | simulator dt=1/60초, substeps=2, controlFrequencyInv=2 → 제어 dt=1/30초 |
| 구동 | PD control |
| 바닥 | 정지/동마찰1.0, restitution0.0 |
| PhysX 주요 값 | solver_type1, position iterations4, velocity iterations0, contact_offset0.02m, rest_offset0.0m |
| SIT/CLIMB 상자 | evaluator에서 `fix_base_link=True`. 평가 중 pose가 바뀌지 않는지 검사 |
| AT/ON_TOP payload | 동적 상자 |
| ON_TOP 받침 | **동적 상자**. 고정 책상으로 취급하지 않음 |
| 상자 밀도 | 기본100kg/m³, randomDensity=false. 동적 물체의 크기가 바뀌면 질량/관성도 달라짐 |
| 운반 출발/AT 도착 선반 | `fixed_warmup_v1` 경로. 첫 reset 전에 물리 초기화1회 수행 |

선반 physics, 동적 받침 여부를 바꾸면 동일한 성공식이라도 다른 실험이 된다. 정확한 전체 sim/asset 옵션은 저장된 `resolved_env.yaml`과 실행 코드가 기준이다.

우리 정책은 기존 관계/상자 bbox 관측을 사용한다. 공개 TokenHSI는 frozen bridge가 원본354차원 관측·정규화·task token·32차원 행동 경로를 구성한다. **물리 장면은 맞추지만 서로 다른 정책의 관측 표현까지 동일하지는 않다.**

| 과제 | 공개 TokenHSI에 전달하는 task/목표 |
| --- | --- |
| AT | Carry token, 운반 목표점 |
| ON_TOP | Carry token, 현재 받침 상판 중심+payload 직립 반높이. 받침을 별도 Carry 관측 물체로 추가하지 않음 |
| SIT | SIT token, 상판+10cm, facing은 상자 로컬+X |
| CLIMB | CLIMB token, 상판+94cm |

SIT의 로컬+X facing은 앞면이 없는 상자에 적용한 입력 규약이다. 성공식에 방향 조건은 없다. 공개 bridge의 실제 가중치는 `/home/cvlab/Desktop/CVPR2027/TokenHSI/output/tokenhsi/ckpt_stage1.pth`다. 공개 평가 명령의 우리73k `--checkpoint`는 환경/player 구성용이며, 공개 정책 가중치를 대신하는 것이 아니다.

## 7. 두 성공식: 수식 전체

`p`는 payload 중심, `r`은 골반/root 검사점, `c`는 받침 중심, `L=(Lx,Ly,Lz)`는 받침 전체 크기, `R`은 받침 quaternion에서 계산한 회전행렬이다. 목표 상판 높이는 회전 bbox의 world-Z 최고점:

```text
z_top = c_z + Σ_j |R[z,j]| * L_j/2
```

이것은 기울어진 물체의 실제 접촉 가능한 경사면을 재구성하는 계산은 아니다.

### 7.1 목표점과 검사점

| 과제 | 검사점 x | 목표 g_xy | 목표 g_z |
| --- | --- | --- | --- |
| AT | payload 중심 p | 샘플링한 AT 목표 XY | 샘플링한 AT 목표 중심 높이 |
| ON_TOP | payload 중심 p | 현재 받침 중심 XY | z_top + payload 직립 전체 높이/2 |
| SIT | 골반 r | 상자 중심 XY | z_top + 0.10m |
| CLIMB | 골반 r | 상자 중심 XY | z_top + 0.94m |

ON_TOP의 payload 반높이는 **직립 크기**의 절반이다. payload가 회전했을 때의 world-Z 반높이로 바꾸지 않는다.

### 7.2 TokenHSI식 점 frame 성공

네 행동 모두 해당 검사점과 목표점에 대해:

```text
point_success(t) = [sqrt((x_x-g_x)² + (x_y-g_y)² + (x_z-g_z)²) <= 0.20m]
```

3D 구형 허용 영역이다. XY20cm AND Z20cm가 아니다. Z 오차가 커지면 허용 XY 거리는 `sqrt(0.20² - Δz²)`로 줄어든다.

### 7.3 우리 영역식 frame 성공

AT는 위 점 성공과 **완전히 동일**하다. ON_TOP/SIT/CLIMB에서는 받침 yaw를 θ라 두고:

```text
dx = x_x - c_x; dy = x_y - c_y
u_x = cos(θ)*dx + sin(θ)*dy
u_y = -sin(θ)*dx + cos(θ)*dy
inside = [abs(u_x) <= 0.4*Lx] AND [abs(u_y) <= 0.4*Ly]
region_success(t) = inside AND [abs(x_z - g_z) <= 0.20m]
```

- ‘내부80%’는 **각 축 길이의80%**. 각 가장자리에서10%를 제외하며 면적은전체의64%다.
- XY와 Z 조건은 **같은 frame에서 동시에** 만족해야 한다.
- payload/골반 **중심점**을 검사한다. payload 전체 footprint, 양발 전체, 몸 전체가 상판 안에 있어야 한다는 조건은 아니다.
- 유한 평면까지의3D 최단 거리20cm 판정과도 다르다. XY가 내부80% 밖이면 높이가 맞아도 실패다.
- 접촉, 손 놓기, 정지, 방향, 직립, CLIMB 발 높이, 일정 시간 유지 조건은 추가하지 않는다.

**점 성공집합과 영역 성공집합은 일반적으로 포함 관계가 아니다.** 예를 들어0.25m 상자의 내부80% 반폭은0.10m다. 목표 높이가 맞고 중심에서X방향0.15m 떨어진 검사점은 점 기준 성공이지만 영역 기준 실패다. 반대로1.6m 상자의 중심에서X방향0.50m 떨어져 있고 높이가 맞으면 영역 성공이지만 점 실패다.

계산 기준: [rescore.py](rescore.py)의 `classify`. 저장된 float32 궤적을 float64로 변환해 계산하고 float32로도 환경별 ever가 같은지 대조한다.

## 8. ever, 낙상, 통계

환경 i의 첫 episode 유효 frame 집합을 T_i라 하면 지표 m의 성공은:

```text
E_i,m = 1  iff  any(success_m(t) for t in T_i)
```

- 첫 step의 action 적용 뒤 기록되는 frame부터 검사하며 종료 frame도 포함한다.
- 예:10초 성공 후15초 낙상 → ever 성공. 성공 없이15초 낙상 → 실패.
- 초기 상태에서 성공한 것으로 미리 세지 않으며, 후속 reset episode 성공을 더하지 않는다.
- raw `results.json`에 남아 있는 `final`은 진단용이고 현재 보고표에는 넣지 않는다.

낙상 종료는 첫 두 step 이후에 다음 두 조건을 모두 만족할 때 발생한다.

1. 양발을 제외한 신체 중 어느 부위의 접촉력 절댓값 합이0.1을 초과.
2. 양발을 제외한 신체 중 어느 부위가 종료 높이 아래로 내려감. 기본0.15m, 머리0.30m.

접촉 부위와 낮은 부위가 동일한 부위일 필요는 없다. ‘상자에 손이 닿았다’만으로 종료되거나 ‘골반 높이가 낮다’만으로 종료되는 식이 아니다. [종료 구현](../../tokenhsi/env/tasks/multi_agent/humanoid_ma.py).

seed s의 성공 건수를 C_s라 하면:

```text
P_s = 100 * C_s / 512
mean = (P_0 + P_1 + P_2) / 3
std = sqrt(Σ_s (P_s - mean)² / (3-1))
```

std는 **평가 장면 seed 간 변동**이다. 학습 seed 표준편차, 표준오차, 신뢰구간이 아니다. 평균 행은 먼저 seed별 네 행동 성공률을 동일 가중 평균하고 그3개 값의 mean/std를 계산한다. 행동별 std를 평균하지 않는다. 모든 seed의 분모가512라 pooled 성공률과 mean은 같지만 std는3개 seed 기준으로 구한다.

굵은 수치는 같은 행동·평가기준의 최고 **평균**과 동률이다. 통계적 유의성을 뜻하지 않는다. 성공률 차이만으로 학습 방식의 인과효과를 주장하지 않는다.

## 9. 학습 보상 및 과거 평가와 구분

현재 평가의 영역식은 예전 엄격한 `single/` 조건과 다르다.

| 과제 | 과거 single / 기존 plane 학습 성공 조건에 있던 항 | 현재 single_match 평가 |
| --- | --- | --- |
| AT | 목표3D20cm AND 별도 높이1mm | 별도1mm 제거,3D20cm만 |
| ON_TOP | payload 바닥과 받침 상판 간격≤1mm | payload 중심과 상판+직립 반높이의 Z 오차≤20cm |
| SIT | 골반 상판+12cm±7cm | 골반 상판+10cm±20cm |
| CLIMB | 골반 상판+94cm±20cm AND 평균 발 높이±7cm | 골반 높이만 유지, 별도 발 높이 조건 제거 |

Point fine-tuning은 학습 성공항의 XY를 중앙 반경20cm로 바꾸되 기존 학습의 높이/발 조건을 유지했다. 따라서 **Point FT의 학습 성공식도 TokenHSI식3D 점 평가식과 동일하지 않다.** 학습 보상 명세는 [stage1_success_ft.md](../../markdowns/stage1_success_ft.md)를 참조한다.

`trace_0.npz`의 마지막 `ours` flag 및 raw `results.json`의 `ours`는 실행 당시 **학습용 own_success 진단값**이다. 특히 Point FT에서는 point 학습 조건을 나타낸다. **이를 우리 영역식 ever로 복사하면 안 된다.** 최종 보고 값은 `rescore.classify`로 위치/크기에서 재계산한 `region_ever`다.

## 10. 원본 TokenHSI benchmark와의 차이

| 항목 | 원본 TokenHSI 경로 | 이번 공통 비교 |
| --- | --- | --- |
| 성공 거리/집계 | 과제별 목표점3D20cm, episode 중 성공 누적 | 같은 점 지표와 추가 영역 지표를 같은 궤적에 적용 |
| 시간 | Carry 약20초, SIT/CLIMB 약10초 | 모든 과제 최대30초 설정 |
| SIT 물체 | test 의자, 저장된 좌석 목표 | 지정 범위 상자, bbox 상판+10cm |
| CLIMB 물체 | Box/Cabinet/Table_Square 등 | 지정 범위 상자, bbox 상판+94cm |
| 초기화 | 원본 task별 loco 모션 경로 | 우리 환경의 공통 loco 초기화 |
| 반복 | 원본 wrapper의512환경 및 같은 seed에서 연속3회 reset | seed0·1·2 각각512환경의 첫 episode |
| 과제 | Traj/Sit/Carry/Climb | Traj 제외, Carry를 AT로 표기, ON_TOP 확장 |
| 물리/정책 연결 | 원본 task 환경 | 공통 multi-agent 기반1인 환경, 공개 정책에는 관측 bridge |

원본 의자에서 저장된 좌석 높이는 등받이까지 포함한 bbox 최고점과 다르다. 상자에서 bbox 상판을 사용하는 이번 방식과 혼동하지 않는다. ON_TOP 점 지표도 Carry 목표를 받침 위에 둔 확장으로, 원본 논문 Table1의 별도 과제가 아니다.

공개 정책 연결 코드: [native_policy.py](../../output_etc/tokenhsi_unified_512env_30s_20261020/native_policy.py). 원본 성공 구현: [humanoid_traj_sit_carry_climb.py](../../tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py)의 `_compute_metrics_evaluation_v1/v3`.

## 11. 결과·원시 데이터와 검증 범위

| 파일 | 용도 |
| --- | --- |
| [메인 MD](eval_30s_512env_3seeds.md) / [메인 JSON](eval_30s_512env_3seeds.json) | 정책별 두 지표·mean/std·seed별 성공 건수·원본 위치 |
| [Fine-tuning 비교](point_ft_comparison.md) | 원본78k와추가1k/2k/3k 비교 |
| `batches/point_ft_<iteration>/summary.json` | 해당 모델의 최종 재채점 통계 |
| 각 run의 `manifest.json` / 배치 `run_manifest.json` | checkpoint·해시·seed·조건·종료 코드·실행 기록 |
| `resolved_env.yaml` | 실행 당시 env/sim 설정 |
| `initial_0.npz` | root, boxes, sizes, goals, packet, source, support, distance |
| `trace_0.npz` | 골반·양발 위치, 물체 상태, 목표, active/done/terminate 및 진단 성공 flag, 크기와binding, dt |
| `completed.json` | 완료 결과와 artifact/source/checkpoint 해시 |
| `validation.json` | 검사 결과와 검증 범위 |

이전70~78k·공개 정책 궤적은 `output_etc/`에 있고79~85k는 `batches/79000_85000/`에 있다. 정확한 경로는 메인 JSON의 각 record와run을 따른다.

검증은 다음을 포함한다: 실행exit0, sampling failure0, 입력 finite, 첫 episode와종료 경계, 고정 상자 pose, 목표 높이 재구성, 기록된 점 성공의 환경별 재현, float32/64 환경별 ever 일치, mean/std 독립 검산, checkpoint/trace/설정 artifact 해시. 프레임별 반올림 차이가 있어도 환경별 ever가 같은지 별도로 확인한다.

초기 root·상자·목표·크기·binding·거리 배열은 정책/checkpoint 간 같은 seed와 비교했다. **기존 single_match 초기 archive는 전체 관절 DOF를 저장하지 않아 전체 초기 관절 상태까지 직접 대조했다고 주장할 수 없다.** 동일 seed만으로 모든 하드웨어/드라이버에서 bitwise 동일한 시뮬레이션을 보장하지 않는다.

## 12. plane / plane_max / plane_range와 혼동하지 말 것

이 문서 본문은 **일반 크기 single_match**다. 아래 실험은 성공식은 공유하지만 크기뿐 아니라 **배치/거리 샘플링도 다르다.** 결과를 같은 분포의 단순 크기 ablation으로 설명하면 안 된다.

| 평가 | 크기 | 환경×seed | 거리·배치 |
| --- | --- | --- | --- |
| single_match | 본문 일반 범위 | 512×3 | 사람–출발 물체 중심 간 반경, 운반 도착점은사각형 rejection sampling |
| plane | 기존 랜덤 XY를1·1.5·2·3배 | 32×3 | 별도 표면 접근 배치. 상세는해당 protocol |
| plane_max | 받침X=Y=0.8·1.2·1.6·2.4m | 32×3 | 사람은상자 로컬−X 면 중앙 바깥. 표면 간 접근 거리 유지 |
| plane_range | 받침X/Y 각각0.80~1.60m,0.05m 격자 독립균등 | 32×3 | plane_max의표면 접근 배치 유지 |

plane_max/range는 SIT·CLIMB·ON_TOP만 평가하며 AT는 제외한다. SIT/CLIMB는 해당 상자, ON_TOP은 **받침만** XY를 바꾼다. payload XYZ와 각 과제의 Z 범위는 본문 규칙을 유지한다.

표면 접근 배치는 SIT 사람→상자 가까운 면 U(1,5)m, CLIMB U(1,10)m, ON_TOP 사람→payload 가까운 면 U(1,10)m 및 payload 먼 면→받침 가까운 면 U(1,4.5)m다. ON_TOP은 일렬 배치이고 환경 간격20m를 사용한다. 따라서 ‘single_match에서도 ON_TOP의 상자 간 거리가 U(1,4.5)m’라고 옮겨 쓰면 잘못이다.

[plane 프로토콜](../plane/evaluation_protocol.md) · [plane_max 프로토콜](../plane_max/evaluation_protocol.md) · [plane_range 프로토콜](../plane_range/evaluation_protocol.md).

## 13. 전달 시 함께 명시할 해석과 한계

1. 보고하는 것은 **위치 도달 ever 성공률**이다. 안정적인 착석/직립/적층, 손 놓기, 일정 시간 유지 성공률이라고 부르지 않는다.
2. plane은 표현상 명칭이고 현재 구현은 ‘yaw 정렬 XY 사각형 내부 + world-Z 높이 허용구간’이다. 물체의 임의 표면에서 정확한 접촉 가능 영역을 추정하는 것은 아니다.
3. 상자 확대 시 중앙점까지 불필요하게 이동할 필요를 줄인다는 동기는 설명할 수 있지만, 영역 성공률이 높다는 사실만으로 영역 학습의 인과효과가 입증되지는 않는다.
4. 공개 TokenHSI와 우리 모델은 구조·학습 데이터·관측·목표 표현이 다르다. 학습 방법만 비교하려면 다른 조건을 통제한 ablation이 필요하다.
5. 원본78k에서 시작한 Point FT 비교는 이미 학습된 정책의 성공항 변경 실험이다. 처음부터 점으로 학습한 정책과 영역으로 학습한 정책의 동일 예산 비교가 아니다.
6. 모델 선택에 이 평가 seed를 여러 번 사용했다. 이 표를 바탕으로 선택한 checkpoint의 독립적인 최종 일반화 성능을 주장하려면 별도 미사용 평가 seed/장면이 필요하다. 현재 표를 독립 holdout이라고 표기하지 않는다.

논문 설명 예시:

> We evaluate all policies in a common box-based, single-agent environment with a nominal 30-second horizon and 512 episodes per evaluation seed across three seeds. Each trajectory is scored using both a TokenHSI-style 3D point-goal criterion and a region-based criterion. AT retains the 0.2 m point-goal criterion. For SIT, CLIMB, and ON_TOP, region success requires the pelvis or payload center to lie within the inner 80% of each horizontal support dimension and within 0.2 m of the task-specific target height. Success is accumulated over the first episode, including its terminating frame. ON_TOP is a Carry-to-support extension; these results do not reproduce the original TokenHSI benchmark environment.


## 14. 현재 워크스페이스 구현

73000 물체 교체 평가에 적용한 독립 구현과 실행 방법은 [SINGLE_MATCH.md](../runs/repro_stage1_73000_20261022/SINGLE_MATCH.md)를 따른다. `eval_carry_objects.py`도 이 평가기로 연결된다. 박스 환경과 의자/테이블 payload 교체 변형은 구분하며, 과거 2인/final 결과는 이 프로토콜 결과로 사용하지 않는다.
