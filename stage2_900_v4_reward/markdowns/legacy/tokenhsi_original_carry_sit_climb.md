# TokenHSI 원본 Stage 1: Carry·Sit·Climb 설정 상세

확인일: **2026-10-14**.

이 문서는 TokenHSI의 **단일 사람 Stage 1 foundational skill learning** 중 Carry·Sit·Climb만 다룬다. 논문, 공식 공개 코드, 이 저장소의 원본 태스크 구현, 현재 연결된 데이터셋을 구분한다. 별도의 다중 에이전트 실험 설정이나 그 평가 결과는 포함하지 않는다.

원본 Stage 1의 과제 목록은 `traj`, `sit`, `carry`, `climb`이다. **별도 `AT` 또는 `ON_TOP` 과제는 없다.** Carry는 상자를 지정된 3D 위치로 옮기는 과제이며, 높은 선반으로 옮기는 경우도 이 과제 안에 포함된다. 다른 동적 상자를 받침으로 지정하는 별도 ON_TOP 관계 평가는 이 Stage 1에 없다.

## 1. 근거와 적용 범위

| 근거 | 위치 | 확인 내용 |
| --- | --- | --- |
| 논문 | [TokenHSI, §4.1 및 부록 B](https://arxiv.org/html/2503.19901v2) | 과제 정의, 평가 지표, 데이터 수, 보고 성능 |
| 공식 환경 설정 | [GitHub YAML](https://github.com/liangpan99/TokenHSI/blob/main/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml) | 기본 파라미터와 train/eval 초기화 분포 |
| 공식 태스크 | [GitHub Python](https://github.com/liangpan99/TokenHSI/blob/main/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py) | 물체·목표 생성, 성공·보상·종료 수식 |
| 공식 evaluator | [GitHub player](https://github.com/liangpan99/TokenHSI/blob/main/tokenhsi/learning/transformer/trans_players.py) | 첫 에피소드 수집, 3회 반복, 성공·오차 집계 |
| 로컬 태스크 | [humanoid_traj_sit_carry_climb.py](../../tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py) | 현재 읽은 원본 태스크 경로 |
| 로컬 환경 YAML | [amp_humanoid_traj_sit_carry_climb.yaml](../../tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml) | 로컬 값 대조 |
| 로컬 evaluator | [trans_players.py](../../tokenhsi/learning/transformer/trans_players.py) | `run_eval()` |
| 평가 진입점 | [stage1_eval.sh](../../tokenhsi/scripts/tokenhsi/stage1_eval.sh) | 512환경, seed 0, 평가 task 지정 |
| 모션 목록 | [dataset_loco_sit_carry_climb.yaml](../../tokenhsi/data/dataset_loco_sit_carry_climb.yaml) | reference motion, skill, RSI 제외 구간 |

GitHub 링크는 변경 가능한 `main`이다. 이 문서는 확인일의 조사 기록이며, 공개 코드 전체와 로컬 코드 전체가 byte 단위로 같다는 뜻은 아니다. 아래 수식은 해당 실행 경로에서 확인한 동작이다. 데이터셋 실측치는 **현재 로컬에 연결된 파일 기준**이다.

## 2. 공통 환경·평가 설정

| 항목 | 값 / 동작 |
| --- | --- |
| 사람 | 환경당 1명 |
| humanoid asset | `mjcf/phys_humanoid_v3.xml` |
| 기준 root 높이 | `_char_h = 0.94m` |
| 제어 | PD control, `powerScale=1.0`, `controlFrequencyInv=2`, 제어 30Hz |
| 환경 배치 | `envSpacing=5`; 목표 샘플링 범위와 관련되지만 물리적 벽을 뜻하지 않음 |
| 바닥 | 정적/동적 마찰 1.0, 반발계수 0.0 |
| self collision | 활성화 |
| state initialization | `stateInit=Random`; 선택한 reference motion의 유효 시점에서 초기화 |
| AMP history | 10개 관측 step |
| 평가할 task | `--eval_task carry`, `sit`, `climb` 중 하나를 지정하고 해당 task 확률을 1로 설정 |
| 평가 물체 split | `--eval`에서 내부 `_mode=test`로 전환 |
| 환경 수 | 평가 wrapper가 **512**로 override |
| 반복 수 | evaluator의 `num_repeat=3` |
| seed | 공식 평가 wrapper의 `--seed 0` |
| 한 반복의 수집 | 환경당 첫 에피소드 하나, 총 512시도 |
| 과제당 총 시도 | 512×3=1,536; 같은 checkpoint의 반복 평가이지 3개 모델 학습을 뜻하지 않음 |
| 평가 초기 skill | Sit=`loco_sit`, Climb=`loco_climb`, Carry=`loco_carry`, 각각 확률 1 |
| 성공 후 종료 | 성공 즉시 종료하지 않음; 제한 시간 또는 조기 종료까지 진행 |
| IET | 평가에서 비활성화 |
| 낙상 종료 | 평가에서도 활성화 |

평가 사람의 root 위치·회전·관절·속도는 locomotion reference에서 가져온다. 모든 사람이 `(0,0)`의 동일한 고정 서기 자세로 시작한다고 해석하면 안 된다. 물체의 초기 XY는 **샘플링된 사람 root XY를 기준**으로 생성하지만, Carry 목표 XY는 별도의 환경 좌표 범위에서 생성한다.

Sit·Climb에는 선택된 대상 물체 하나가 활성화되고, Carry에는 운반 상자와 출발/도착 선반이 관여한다. 코드 내부에는 다른 task용 actor와 시각화 marker도 만들어지지만 비활성 task 물체는 이동시켜 숨기므로, actor 총수와 활성 과제의 대상 수를 구분해야 한다.

기본 PhysX 설정은 substeps 2, TGS solver(`solver_type=1`), position iterations 4, velocity iterations 0, contact offset 0.02m, rest offset 0이다. 이는 접촉 시뮬레이션 파라미터이며 성공 거리 허용치가 아니다.

## 3. Carry

### 3.1 상자 종류·크기·물리 속성

Isaac Gym `create_box()`로 생성하는 직육면체다. 기본 크기는 `(0.4,0.4,0.4)m`, 학습에서는 세 축에 같은 scale을 적용한다.

| 번호 | 학습 크기 X×Y×Z (m) | 평가 크기 X×Y×Z (m) |
| ---: | --- | --- |
| 1 | 0.20×0.20×0.20 | 0.22×0.22×0.22 |
| 2 | 0.25×0.25×0.25 | 0.27×0.27×0.27 |
| 3 | 0.30×0.30×0.30 | 0.32×0.32×0.32 |
| 4 | 0.35×0.35×0.35 | 0.37×0.37×0.37 |
| 5 | 0.40×0.40×0.40 | 0.42×0.42×0.42 |
| 6 | 0.45×0.45×0.45 | 0.47×0.47×0.47 |
| 7 | 0.50×0.50×0.50 | 0.52×0.52×0.52 |
| 8 | 0.55×0.55×0.55 | 0.57×0.57×0.57 |
| 9 | 0.60×0.60×0.60 | 0.30×0.30×0.40 |

- 학습 scale: 0.5~1.5, 간격 0.125. 평가에는 위 `testSizes`로 교체한다.
- 밀도 100kg/m³. 질량은 `100*sx*sy*sz`kg이며, 0.4m 정육면체는 6.4kg이다.
- 선형·각 감쇠 각 0.01, 최대 각속도 100rad/s. 운반 상자는 고정 asset이 아니다.
- 크기는 환경 생성 시 배정되며 episode reset마다 바뀌지 않는다.
- 512환경에서는 우선 9개 크기를 한 번씩 배정하고, 나머지는 균등 확률로 샘플링한 뒤 섞는다. 크기별 정확히 같은 개수는 아니다.
- 평가 JSON은 크기별 성공률을 분리하지 않고 전체를 집계한다.

### 3.2 출발 상자 위치·방향

평가의 `loco_carry` 시작에서:

1. 2D 표준정규 벡터를 정규화해 방향 `u`를 만든다. 방향각은 원 둘레에서 균등하다.
2. 반경 `r ~ U(1,10)`m를 샘플링한다.
3. 상자 XY는 `human_root_xy + r*u`다.
4. yaw는 `U(0,2π)`이며 초기 roll/pitch는 없다.
5. 상자의 선속도·각속도는 0으로 설정한다.

반경을 균등하게 뽑으므로 원환의 **면적에 대해 균등한 분포는 아니다**. 이 초기화는 학습의 locomotion 시작에도 해당하며, pickUp/carryWith/putDown reference 시작은 모션의 상자 상태를 사용한다.

### 3.3 출발 높이·선반

상자 높이를 `sz`, 바닥보다 높은 지지면 높이를 `b`라 하면:

```text
확률 0.5: b = 0
확률 0.5: b = min(U(0,1), 1.2 - sz)

선반 윗면 z = b
선반 중심 z = b - 0.01
출발 상자 중심 z = b + sz/2 + 0.05
```

- 선반 크기: **0.4×0.4×0.02m**, `fix_base_link=True`.
- 상자 윗면 1.2m clamp는 **마지막 5cm 여유를 더하기 전** 적용된다. 따라서 reset 직후 상자 윗면은 최대 1.25m일 수 있다.
- `randomHeight=True` 경로에서는 바닥 추첨에도 선반 윗면을 z=0으로 놓고 출발 상자에 5cm를 더한다. 높은 배치에만 actor를 새로 생성하는 방식이 아니다.
- 선반 폭은 상자 크기에 맞춰 늘어나지 않는다.

### 3.4 목표 위치·높이

평가의 목표점은 상자 **중심**이 도달할 위치다.

- XY: `[-4.5,4.5]²`에서 균등 샘플링한다. `envSpacing=5`에서 각 축 반폭이 `5-0.5=4.5m`다.
- 담당 사람 root 및 출발 상자와의 XY 거리가 각각 **1m 이상**이 될 때까지 재샘플링한다.
- 출발 상자에서 목표로 가는 이동 거리 자체를 `U(1,10)`으로 뽑는 것은 아니다.
- 목표 높이도 독립된 50% 추첨과 같은 clamp 규칙을 사용한다.

```text
목표 지지면 높이 b_goal = 0 또는 min(U(0,1), 1.2 - sz)
목표 중심 z = b_goal + sz/2
도착 선반 중심 z = b_goal - 0.01
```

목표에는 출발 때의 5cm 여유를 더하지 않는다. 목표 yaw·roll·pitch는 성공 판정 대상이 아니다. 학습의 putDown reference 시작은 이 랜덤 목표 경로 대신 해당 모션 마지막 상자 상태를 사용하므로 평가 초기화와 구분한다.

근거: 태스크의 `register_task_carry_post_init()`, `_reset_task_carry()`, `_regulate_height()`, `_load_platform_asset()`.

### 3.5 성공과 시간

```text
d(t) = ||box_center(t) - target_center||₂
success = episode 중 d(t) ≤ 0.20m인 step이 하나 이상 존재
```

상자 중심의 높이도 3D 거리에 포함되지만 별도의 z 허용치는 없다. 손 접촉·실제 grasp·손 놓기·상자 방향·속도·선반 접촉·유지시간은 성공 판정에 포함하지 않는다. 최대 600 step 설정으로 약 20초다.

### 3.6 보상: 평가 성공과 다른 조건

기본 task reward는 `walk + carry + handheld + putdown`이다. 기본 `onlyVelReward=True`, `onlyHeightHandHeldReward=False`다.

| 항목 | 기본 구현 |
| --- | --- |
| walk | 사람의 상자 방향 속도를 1.5m/s로 유도: `0.2*exp(-5*(1.5-v_dir)^2)`; 역방향/정지 속도에는 0, XY 0.5m 안에서는 0.2로 포화 |
| carry | 목표 방향 상자 속도 보상 최대 0.2 + `0.2*exp(-10*3D_distance²)` |
| carry 높이 gate | 상자 중심 z가 `sz/2+0.2` 이하이면 먼 거리 속도 항을 0으로 만듦; 그 뒤 목표 XY 0.5m 안에서는 속도 항을 1로 덮어씀 |
| 상자 과속 | 상자 3D 속도 2.5m/s 초과분에 `-(1-exp(-2*excess²))` |
| handheld | 양손 **평균 위치**와 상자 중심 거리로 `0.2*exp(-5*distance²)`; 사람–상자 XY>0.7m이면 0 |
| putdown | **높이 오차 ≤1mm AND XY 거리 ≤10cm**일 때 0.2, 아니면 0 |

**1mm 높이 조건은 원본에도 있지만 `compute_putdown_reward()`의 보상 조건이다. 평가 성공은 여전히 3D 20cm 도달이다.** 따라서 내려놓기 보상을 받지 못하는 궤적도 evaluation에서 성공할 수 있다. handheld 보상 역시 실제 접촉 센서 기반 grasp 판정이 아니다.

## 4. Sit

### 4.1 물체와 샘플링

- 데이터: `tokenhsi/data/dataset_sit/objects/{train,test}/StraightChair_Normal`.
- 현재 로컬 목록: 학습 **49개**, 평가 **26개**. 논문 수와 일치한다.
- 고정 URDF asset(`fix_base_link=True`)을 사용하며 임의의 상자 크기 조합을 생성하지 않는다.
- 물체별 `config.json`의 `bbox`, `center`, `facing`, `tarSitPos`를 읽는다.
- 생성 때 환경별 물체를 샘플링하고 고정한다. reset에서는 배치와 사람 상태가 바뀐다.
- 한 카테고리이므로 카테고리 안의 asset을 균등 확률로 선택한다. 모든 평가 asset이 반드시 같은 횟수만큼 배정되는 것은 아니다.

### 4.2 초기 배치와 사람

- 평가 초기 skill은 `loco_sit` 100%다.
- 의자 XY는 사람 root 기준 임의 방향에 반경 **U(1,5)m**로 배치한다.
- yaw는 U(0,2π), 초기 roll/pitch는 없다.
- 의자 바닥이 지면에 오도록 전체 bbox 높이에 맞춰 z를 설정한다. Carry식 높이 랜덤화나 선반은 없다.
- 물체 선속도·각속도는 0이다.

### 4.3 목표점

목표는 사람 root/pelvis 위치다. 물체 로컬 좌표의 `tarSitPos`를 월드 좌표로 변환한다.

```text
goal_world = quat_rotate(object_rotation, tarSitPos_local) + object_position
```

데이터 생성 코드는 좌석 중앙 부근 height map 5개 표본의 평균에 **0.10m**를 더해 `tarSitPos`를 만든다. 등받이를 포함한 **전체 bbox 윗면+10cm가 아니다**. 실행 시 매번 좌석 표면을 측정하는 것도 아니고, 저장된 `tarSitPos`를 사용한다.

근거: [Sit 데이터 생성기](../../tokenhsi/data/dataset_sit/generate_object.py)의 목표점 생성과 태스크 `_reset_task_sit()`.

### 4.4 성공·시간·보상

- 성공: root–목표점 **3D 거리 ≤0.20m**가 episode 중 한 번 이상.
- 제한 시간: 300 step 설정, 약 10초.
- 의자 접촉, 앉은 자세, 몸 방향, 정지·유지시간은 평가 조건에 없다.
- 기본 task reward: `0.7*exp(-10*root_goal_distance²) + 0.3*approach_velocity_reward`.
- 접근 속도 목표는 1.5m/s, 속도 항은 `exp(-2*(1.5-v_dir)^2)`이고 물체 XY 0.5m 이내에서 1로 포화한다.
- 물체 XY 1.5m 이내에서 root 속도 1.3m/s 초과 및 yaw 속도 3rad/s 초과를 각각 계수 0.5로 벌점 처리한다. 이는 보상이며 성공 판정에 추가되는 조건은 아니다.

## 5. Climb

### 5.1 물체와 샘플링

- 데이터: `tokenhsi/data/dataset_amass_climb/objects/{train,test}`.
- 카테고리: `Box`, `Cabinet`, `Table_Square`.
- 고정 URDF asset을 사용하며, 물체별 bbox와 표면 목표 높이를 읽는다.
- 환경 생성 시 물체가 고정되고 episode마다 물체 크기를 재샘플링하지 않는다.
- **세 카테고리에 각각 1/3 확률**, 카테고리 내부 asset에는 균등 확률을 준다. 전체 asset을 동일 확률로 선택하는 방식과 다르다.

| split | Box | Cabinet | Table_Square | 합계 |
| --- | ---: | ---: | ---: | ---: |
| 현재 로컬 train | 18 | 10 | 10 | **38** |
| 현재 로컬 test | 18 | 9 | 8 | **35** |
| 논문 test | — | — | — | **26** |

논문은 train/test를 38/26개로 기술한다. **현재 로컬 loader가 읽는 test 목록은 35개**이며 모든 항목에 `asset.urdf`와 `config.json`이 있는 것을 확인했다. 데이터 배포 시점이나 추가 파일의 이력은 조사하지 않았으므로 차이의 원인을 단정하지 않는다.

### 5.2 초기 배치와 사람

- 평가 초기 skill은 `loco_climb` 100%다.
- 물체 XY는 사람 root 기준 임의 방향에 반경 **U(1,10)m**로 배치한다.
- yaw는 U(0,2π), 초기 roll/pitch는 없다.
- 물체 바닥을 지면에 놓으며 Carry식 선반 높이 랜덤화는 없다.
- 물체 선속도·각속도는 0이다.

### 5.3 목표점

```text
goal_xy = object_position_xy
goal_z = object_position_z + stored_tarSitPos_z + 0.94
```

Climb도 호환성 때문에 표면 좌표를 `tarSitPos`라는 키에 저장한다. 데이터 생성 시 **중앙 height map의 표면 높이**에서 bbox 원점 높이를 빼 저장하고, 런타임에 humanoid 기준 root 높이 0.94m를 더한다. 따라서 정확한 기준은 저장된 중앙 표면이며 임의 형상의 bbox 최고점과 항상 같다고 가정하면 안 된다.

근거: [Climb 데이터 생성기](../../tokenhsi/data/dataset_amass_climb/generate_object.py), 태스크 `_reset_task_climb()`, [humanoid.py](../../tokenhsi/env/tasks/humanoid.py)의 `_char_h`.

### 5.4 성공·시간·보상

- 성공: root–목표점 **3D 거리 ≤0.20m**가 episode 중 한 번 이상.
- 제한 시간: 300 step 설정, 약 10초.
- 발 접촉, 양발 XY 위치·높이, 상판 위 정지·유지시간은 평가 성공 조건에 없다.
- task reward는 접근 속도 항 0.2 + root 목표 근접 항 0.5 + **양발 평균 높이** 항 0.3이다.
- 접근 속도 목표는 1.5m/s. 속도 항은 `exp(-2*(1.5-v_dir)^2)`이고 `v_dir≤0`이면 0이다.
- `valid_radius = sqrt(bbox_x²+bbox_y²)/2 + 0.3m` 안에서는 접근 항을 포화한다.
- root 근접 항: `exp(-10*root_goal_distance²)`.
- 발 높이 항: `exp(-50*(mean_feet_z-surface_z)^2)`이며 valid radius 밖에서는 0이다.
- 가까운 영역의 root 과속 벌점 계수는 1.0이다. 반경 1.5m 안의 기본 임계는 1.5m/s, valid radius 안에서는 1.3m/s다.

**발 높이는 보상에 사용되지만 평가 성공 판정에는 사용되지 않는다.**

## 6. 학습 초기화와 평가 초기화의 차이

| 과제 | 학습 skill 초기화 확률 | 평가 skill 초기화 확률 |
| --- | --- | --- |
| Sit | loco_sit 0.5, sit 0.5 | loco_sit 1.0 |
| Climb | loco_climb 0.5, climb 0.5, climbNoRSI 0 | loco_climb 1.0 |
| Carry | loco_carry 0.5, pickUp 0.1, carryWith 0.3, putDown 0.1, omomo 0 | loco_carry 1.0 |

`sample_motions()`와 `sample_time_rsi()`로 모션·시점을 선택한다. 실제 모션 목록과 RSI 제외 구간은 motion YAML을 따른다. 평가가 locomotion 100%라는 뜻은 **interaction reference의 중간/완성 상태에서 시작하지 않는다**는 뜻이지, motion state sampling 자체가 없다는 뜻은 아니다.

공식 전체 모델 학습은 네 foundational task를 함께 사용한다. 위 세 과제는 각각 선택 확률 0.3이고 나머지 확률 0.1은 경로 과제에 할당된다. task 선택 확률, 위 초기 skill 확률, `skillDiscProb`의 discriminator reference 확률은 서로 다른 설정이다.

공통 task 보상에는 `-0.0005*sum(abs(joint_torque*joint_velocity))` power 항이 추가된다. 로컬 원본 train YAML의 task/AMP 보상 혼합 가중치는 각각 0.5/0.5다. 보상에 쓰는 거리·속도·발 높이 조건을 evaluation 성공 조건과 혼동하지 않는다. [Train YAML](../../tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml)

공식 env YAML의 기본 학습 환경 수는 4,096이고 현재 로컬 복사본은 2,048이다. 논문 §4.1은 4,096환경·50k PPO iteration을 보고한다. 평가 환경 수는 이 값과 무관하게 공식 eval wrapper에서 512로 지정한다. 이 문서는 새 학습 실행을 지시하지 않는다.

## 7. 성공·오차·종료의 정확한 집계

### 7.1 step별 판정

세 과제 모두 다음 수식을 사용한다. Carry의 `p`는 상자 중심, Sit·Climb의 `p`는 사람 root다.

```text
d_t = ||p_t - goal||₂
success_buf += 1[d_t ≤ 0.20]
precision_buf = min(precision_buf, d_t)  # d_t≤0.20인 step에서만 갱신
```

처음에는 `success_buf=0`, `precision_buf=∞`다. 성공 episode에 대해서는 통과한 step에서만 최소값을 갱신해도 전체 episode의 최소 거리와 같다.

### 7.2 episode 종료 시 집계

```text
S_i = 1[success_buf_i > 0]
SR = sum(S_i) / num_trials
Error = mean(min_distance_i for successful episodes)
```

- 원본 evaluator는 `success_rate`를 0~1 비율, `success_precision`을 **m 단위**로 저장한다. 논문 Error(cm) 표시는 100을 곱한 단위다.
- 성공 시도가 0이면 코드가 `success_precision=0`을 출력한다. 이 값은 완벽한 정확도를 뜻하지 않는다.
- 성공 후 멀어지거나 낙상해 종료돼도 성공 기록이 남는다.
- 실패하면서 `terminate=True`인 경우 `fail_trials_because_terminate`로 집계한다. 성공 후 낙상은 이 실패 수에 들어가지 않는다.
- 한 환경이 종료된 뒤 reset되어 계속 동작하더라도 `has_collected`로 두 번째 episode부터는 집계에서 제외한다.
- 반복마다 전체 reset을 하지만 환경별 물체 asset·크기는 유지된다. 매 반복마다 seed를 재설정하지 않는다.
- JSON에는 `ObjectSet_test_0/1/2`의 결과를 각각 저장한다. 평균·표준편차를 그 JSON 자체에서 자동 계산하는 것은 아니다.
- 성공 직후 simulation을 끊거나 마지막 frame만 검사하는 방식이 아니다.

### 7.3 낙상과 time limit

발 접촉을 제외한 뒤 다음 두 조건이 동시에 성립하면 낙상으로 판단한다.

1. 발 이외 신체 중 어느 하나에서 접촉력의 **어느 축 절댓값이 0.1 초과**.
2. 발 이외 신체 중 어느 하나가 해당 부위의 termination height 아래로 내려감.

두 조건을 만족하는 신체 부위가 반드시 같은 부위일 필요는 없다. 높이는 기본 0.15m, 머리는 0.30m이며 `progress_buf>1`일 때부터 판정한다. 낙상 또는 `progress_buf≥episodeLength-1`에서 reset하므로 실제 평가 trace 길이는 명목 300/600과 한 step 차이날 수 있다.

학습 IET는 활성화되어 Sit·Climb의 root가 20cm 이내에 들어온 step을 누적하고, 설정 `maxIETSteps=60`에 대해 counter≥59에서 종료를 유발한다. 코드에는 목표에서 벗어났을 때 counter를 0으로 되돌리는 처리가 없어 **연속 60프레임 유지 조건으로 해석하면 안 된다**. 평가에서는 IET를 꺼서 이 규칙이 적용되지 않는다.

## 8. 공식 평가 실행 인터페이스

저장소 루트에서 공식 wrapper의 인터페이스는 다음과 같다. 아래는 명령 기록이며, 이 문서 작성 중 실행하지 않았다.

```bash
bash tokenhsi/scripts/tokenhsi/stage1_eval.sh carry
bash tokenhsi/scripts/tokenhsi/stage1_eval.sh sit
bash tokenhsi/scripts/tokenhsi/stage1_eval.sh climb
```

wrapper는 `HumanoidTrajSitCarryClimb`, 원본 env/train YAML, `output/tokenhsi/ckpt_stage1.pth`, `--test --headless --eval --num_envs 512 --seed 0`을 사용한다. 이 경로의 원본 checkpoint와 runtime 준비가 필요하며, 문서 작성으로 실행 가능성이나 checkpoint 존재를 검증한 것은 아니다.

결과는 checkpoint의 `nn/` 여부와 무관하게 `dirname(checkpoint)/../metrics/<checkpoint stem>/<task>/metrics_<timestamp>.json`이라는 코드상 상대 경로로 저장된다. [공식 wrapper](https://github.com/liangpan99/TokenHSI/blob/main/tokenhsi/scripts/tokenhsi/stage1_eval.sh)

## 9. 논문·공개 코드·로컬 데이터 차이

| 항목 | 논문 / 공식 설정 | 공개 코드 / 현재 로컬 확인 |
| --- | --- | --- |
| Sit 초기 거리 | 1~5m | 1~5m |
| Climb 초기 거리 | 논문 1~5m | 코드 1~10m |
| Carry 초기 거리 | 논문 1~9m | 코드 1~10m |
| Carry 목표 XY | 논문 10×10m | envSpacing=5의 코드 범위 9×9m |
| Climb test asset 수 | 논문 26 | 현재 로컬 목록 35 |
| 기본 학습 환경 수 | 공식 YAML·논문 4,096 | 현재 로컬 env YAML 2,048 |

원본 벤치마크를 재현하려면 단지 20cm 성공 수식을 적용하는 것 외에도 어떤 코드 버전·데이터 목록·checkpoint·초기화 분포를 사용하는지 고정해야 한다. 논문 숫자와 현재 로컬 파일의 차이는 자동으로 어느 한쪽에 맞춰 수정하지 않는다.

## 10. 현재 로컬 asset 크기·목표 높이

아래 수치는 연결된 데이터의 `config.json`을 읽어 계산했다. bbox 범위는 축별 최소/최대이며, 이 범위의 모든 조합을 샘플링한다는 뜻이 아니다. Sit의 bbox Z에는 등받이 등이 포함되어 좌석 높이와 다르다.

| 과제·split | 개수 | bbox X (m) | bbox Y (m) | bbox Z (m) | 월드 목표 root Z (m, 바닥 배치) |
| --- | ---: | --- | --- | --- | --- |
| Sit train | 49 | 0.4091~0.6941 | 0.4334~0.6430 | 0.6902~1.0144 | 0.4764~0.6195 |
| Sit test | 26 | 0.4224~0.7112 | 0.4868~0.7275 | 0.7227~1.0121 | 0.4270~0.6249 |
| Climb train | 38 | 0.4000~0.6823 | 0.3412~0.6823 | 0.2000~0.7250 | 1.1400~1.6650 |
| Climb test | 35 | 0.4000~0.6823 | 0.3000~0.6823 | 0.2000~0.7000 | 1.1400~1.6400 |

Climb을 물체 종류별로 나누면 다음과 같다. 모두 물체 로컬 bbox의 전체 길이이며 half extent가 아니다.

| split·종류 | 개수 | bbox X (m) | bbox Y (m) | bbox Z (m) |
| --- | ---: | --- | --- | --- |
| train Box | 18 | 0.4000~0.6000 | 0.4000~0.6000 | 0.2000~0.7000 |
| train Cabinet | 10 | 0.4200~0.6604 | 0.3510~0.5588 | 0.4663~0.7250 |
| train Table_Square | 10 | 0.6803~0.6823 | 0.3412~0.6823 | 0.3060~0.7019 |
| test Box | 18 | 0.4000~0.6000 | 0.3000~0.5000 | 0.2000~0.7000 |
| test Cabinet | 9 | 0.4431~0.5922 | 0.3602~0.5813 | 0.4990~0.6950 |
| test Table_Square | 8 | 0.6714~0.6823 | 0.4502~0.6823 | 0.3685~0.6366 |

- **Climb 학습 Box:** 밑면은 `(0.4,0.4)`, `(0.5,0.5)`, `(0.6,0.6)`m이고 각각 높이 `0.2/0.3/0.4/0.5/0.6/0.7`m를 조합한 18개다.
- **Climb 평가 Box:** 밑면은 `(0.4,0.5)`, `(0.5,0.3)`, `(0.6,0.4)`m이고 같은 여섯 높이를 조합한 18개다.
- **Sit에는 Box 카테고리가 없다.** 학습 의자 49개·평가 의자 26개다. 생성기의 `좌면 높이 + 0.10m` 규칙을 역산한 좌면 대표 높이는 학습 **0.3764~0.5195m**, 평가 **0.3270~0.5249m**다. 좌면 전체의 최저/최고 높이를 뜻하지 않는다. 전체 bbox Z(등받이 포함)를 앉을 상판 높이로 사용하면 안 된다.
- 실제 mesh는 bbox를 꽉 채운 직육면체가 아니며, 특히 의자 좌석·등받이와 테이블 다리는 별도 형상이다. 위 축별 범위의 임의 조합을 생성하는 샘플러가 아니라 저장된 asset 목록에서 선택한다.

아래 부록은 현재 로컬 평가 asset 전체 목록이다. 치수·높이는 소수 넷째 자리로 반올림했으며, 실행은 JSON의 원래 정밀도를 사용한다.


### 10.1. Sit 평가 asset 전체 목록

| 카테고리 / asset | bbox X×Y×Z (m) | 목표 root Z (m) |
| --- | --- | ---: |
| [StraightChair_Normal/00883cfc-9ec8-4494-bc40-6c768d4ed9d9](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/00883cfc-9ec8-4494-bc40-6c768d4ed9d9/config.json) | 0.5241 × 0.5751 × 0.9675 | 0.5882 |
| [StraightChair_Normal/0cd722cb-4cf5-4026-9906-187e51d78c67](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/0cd722cb-4cf5-4026-9906-187e51d78c67/config.json) | 0.5616 × 0.5346 × 0.9323 | 0.5399 |
| [StraightChair_Normal/0e874d17-37e5-372a-9be9-5ca5a3b1aa00](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/0e874d17-37e5-372a-9be9-5ca5a3b1aa00/config.json) | 0.4224 × 0.4979 × 0.7826 | 0.4963 |
| [StraightChair_Normal/108d44d9-1ce0-4dcd-86f3-5f6e69c4c879](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/108d44d9-1ce0-4dcd-86f3-5f6e69c4c879/config.json) | 0.4915 × 0.5365 × 0.8215 | 0.5260 |
| [StraightChair_Normal/11710d4f-b5a6-4f42-9149-e6d5c898bdf4](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/11710d4f-b5a6-4f42-9149-e6d5c898bdf4/config.json) | 0.5007 × 0.5093 × 0.8776 | 0.5520 |
| [StraightChair_Normal/1460b4ef-2317-347a-82d9-ae24ea0083be](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/1460b4ef-2317-347a-82d9-ae24ea0083be/config.json) | 0.4703 × 0.4980 × 0.7444 | 0.5284 |
| [StraightChair_Normal/24a49906-1627-4639-ae26-32d91d02871b](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/24a49906-1627-4639-ae26-32d91d02871b/config.json) | 0.5022 × 0.5922 × 1.0121 | 0.5727 |
| [StraightChair_Normal/252c2950-6bad-4bf7-a557-76b1a08f7394](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/252c2950-6bad-4bf7-a557-76b1a08f7394/config.json) | 0.4500 × 0.5526 × 0.9000 | 0.5743 |
| [StraightChair_Normal/28d53c9e-4f9f-48c4-9b88-3238876f7ad8](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/28d53c9e-4f9f-48c4-9b88-3238876f7ad8/config.json) | 0.4671 × 0.5289 × 0.8520 | 0.5420 |
| [StraightChair_Normal/299d19d1-7ea9-4293-9111-a177c61b8545](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/299d19d1-7ea9-4293-9111-a177c61b8545/config.json) | 0.6069 × 0.5578 × 0.8352 | 0.5640 |
| [StraightChair_Normal/3fb272ad-8d57-3632-b82b-4da71a5350f6](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/3fb272ad-8d57-3632-b82b-4da71a5350f6/config.json) | 0.4499 × 0.4957 × 0.7227 | 0.5068 |
| [StraightChair_Normal/409ae7cf-1cbe-4d88-8936-80d192c168b5](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/409ae7cf-1cbe-4d88-8936-80d192c168b5/config.json) | 0.5510 × 0.5356 × 0.8529 | 0.5701 |
| [StraightChair_Normal/496400ff-4540-4911-a1c8-a5ace630cf89](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/496400ff-4540-4911-a1c8-a5ace630cf89/config.json) | 0.4965 × 0.5360 × 0.8091 | 0.5701 |
| [StraightChair_Normal/549eba4e-2479-34ae-9f50-1f8a09e64d7c](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/549eba4e-2479-34ae-9f50-1f8a09e64d7c/config.json) | 0.4700 × 0.4868 × 0.9201 | 0.5541 |
| [StraightChair_Normal/56441da5-3a13-47b8-ad95-50df8613d6dd](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/56441da5-3a13-47b8-ad95-50df8613d6dd/config.json) | 0.4617 × 0.6265 × 0.8796 | 0.5508 |
| [StraightChair_Normal/64daf139-bf01-3270-8280-81f9a4c63b97](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/64daf139-bf01-3270-8280-81f9a4c63b97/config.json) | 0.4456 × 0.5448 × 0.9446 | 0.5252 |
| [StraightChair_Normal/8105b670-faf1-33e0-b9e4-16c199cc11ca](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/8105b670-faf1-33e0-b9e4-16c199cc11ca/config.json) | 0.5312 × 0.5007 × 0.7525 | 0.4270 |
| [StraightChair_Normal/8b348c4a-7c31-4196-8b26-5e083f2cd2be](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/8b348c4a-7c31-4196-8b26-5e083f2cd2be/config.json) | 0.7112 × 0.7275 × 0.9544 | 0.6249 |
| [StraightChair_Normal/9b42d6ed-1749-4fc6-b1a5-9086a9cea6b4](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/9b42d6ed-1749-4fc6-b1a5-9086a9cea6b4/config.json) | 0.4929 × 0.5209 × 0.9283 | 0.5631 |
| [StraightChair_Normal/9d40d9be-42d0-4dbb-9fed-0a57143f4e34](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/9d40d9be-42d0-4dbb-9fed-0a57143f4e34/config.json) | 0.5483 × 0.5088 × 0.7968 | 0.5299 |
| [StraightChair_Normal/a333e75d-cf32-4d75-810b-882701bec2d1](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/a333e75d-cf32-4d75-810b-882701bec2d1/config.json) | 0.5160 × 0.5632 × 0.8819 | 0.5394 |
| [StraightChair_Normal/cebf5ef0-da78-4a8a-ad63-718b4188558a](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/cebf5ef0-da78-4a8a-ad63-718b4188558a/config.json) | 0.5651 × 0.4984 × 0.7963 | 0.5387 |
| [StraightChair_Normal/d3902c3d-06e1-46d4-8f86-d9475dedf1c9](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/d3902c3d-06e1-46d4-8f86-d9475dedf1c9/config.json) | 0.4803 × 0.5041 × 0.7973 | 0.5310 |
| [StraightChair_Normal/e4b0dc19-5f8b-4ffb-8ab3-4b88847a710c](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/e4b0dc19-5f8b-4ffb-8ab3-4b88847a710c/config.json) | 0.6133 × 0.5996 × 0.8659 | 0.6114 |
| [StraightChair_Normal/e6f0fc32-aa55-427f-a014-f265af3fbcd6](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/e6f0fc32-aa55-427f-a014-f265af3fbcd6/config.json) | 0.5422 × 0.5085 × 0.8039 | 0.5386 |
| [StraightChair_Normal/efe1857b-d46a-47bd-b92f-0dc81a4eb620](../../../TokenHSI/tokenhsi/data/dataset_sit/objects/test/StraightChair_Normal/efe1857b-d46a-47bd-b92f-0dc81a4eb620/config.json) | 0.4744 × 0.5560 × 0.7567 | 0.5138 |


### 10.2. Climb 평가 asset 전체 목록

| 카테고리 / asset | bbox X×Y×Z (m) | 목표 root Z (m) |
| --- | --- | ---: |
| [Box/box_length_0-40_0-50_0-20](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-40_0-50_0-20/config.json) | 0.4000 × 0.5000 × 0.2000 | 1.1400 |
| [Box/box_length_0-40_0-50_0-30](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-40_0-50_0-30/config.json) | 0.4000 × 0.5000 × 0.3000 | 1.2400 |
| [Box/box_length_0-40_0-50_0-40](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-40_0-50_0-40/config.json) | 0.4000 × 0.5000 × 0.4000 | 1.3400 |
| [Box/box_length_0-40_0-50_0-50](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-40_0-50_0-50/config.json) | 0.4000 × 0.5000 × 0.5000 | 1.4400 |
| [Box/box_length_0-40_0-50_0-60](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-40_0-50_0-60/config.json) | 0.4000 × 0.5000 × 0.6000 | 1.5400 |
| [Box/box_length_0-40_0-50_0-70](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-40_0-50_0-70/config.json) | 0.4000 × 0.5000 × 0.7000 | 1.6400 |
| [Box/box_length_0-50_0-30_0-20](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-50_0-30_0-20/config.json) | 0.5000 × 0.3000 × 0.2000 | 1.1400 |
| [Box/box_length_0-50_0-30_0-30](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-50_0-30_0-30/config.json) | 0.5000 × 0.3000 × 0.3000 | 1.2400 |
| [Box/box_length_0-50_0-30_0-40](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-50_0-30_0-40/config.json) | 0.5000 × 0.3000 × 0.4000 | 1.3400 |
| [Box/box_length_0-50_0-30_0-50](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-50_0-30_0-50/config.json) | 0.5000 × 0.3000 × 0.5000 | 1.4400 |
| [Box/box_length_0-50_0-30_0-60](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-50_0-30_0-60/config.json) | 0.5000 × 0.3000 × 0.6000 | 1.5400 |
| [Box/box_length_0-50_0-30_0-70](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-50_0-30_0-70/config.json) | 0.5000 × 0.3000 × 0.7000 | 1.6400 |
| [Box/box_length_0-60_0-40_0-20](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-60_0-40_0-20/config.json) | 0.6000 × 0.4000 × 0.2000 | 1.1400 |
| [Box/box_length_0-60_0-40_0-30](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-60_0-40_0-30/config.json) | 0.6000 × 0.4000 × 0.3000 | 1.2400 |
| [Box/box_length_0-60_0-40_0-40](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-60_0-40_0-40/config.json) | 0.6000 × 0.4000 × 0.4000 | 1.3400 |
| [Box/box_length_0-60_0-40_0-50](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-60_0-40_0-50/config.json) | 0.6000 × 0.4000 × 0.5000 | 1.4400 |
| [Box/box_length_0-60_0-40_0-60](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-60_0-40_0-60/config.json) | 0.6000 × 0.4000 × 0.6000 | 1.5400 |
| [Box/box_length_0-60_0-40_0-70](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Box/box_length_0-60_0-40_0-70/config.json) | 0.6000 × 0.4000 × 0.7000 | 1.6400 |
| [Cabinet/0ed0b5e6-ffce-4401-92f7-2821b244a872](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Cabinet/0ed0b5e6-ffce-4401-92f7-2821b244a872/config.json) | 0.4431 × 0.4368 × 0.4990 | 1.4390 |
| [Cabinet/115678d8-847c-46c8-9fd4-f8801c65082a](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Cabinet/115678d8-847c-46c8-9fd4-f8801c65082a/config.json) | 0.4505 × 0.3602 × 0.5502 | 1.4902 |
| [Cabinet/2ad51dfc-9e9e-492b-b053-6e6373a9a2aa](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Cabinet/2ad51dfc-9e9e-492b-b053-6e6373a9a2aa/config.json) | 0.4900 × 0.4163 × 0.5774 | 1.5173 |
| [Cabinet/2aee7bfa-1696-44be-bf4f-36b2872e95b8](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Cabinet/2aee7bfa-1696-44be-bf4f-36b2872e95b8/config.json) | 0.4888 × 0.3806 × 0.6431 | 1.5831 |
| [Cabinet/2be2d0ea-23f7-4955-a433-e3d2c6a4c55b](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Cabinet/2be2d0ea-23f7-4955-a433-e3d2c6a4c55b/config.json) | 0.5500 × 0.4139 × 0.5500 | 1.4870 |
| [Cabinet/62319141-32fa-4353-b2ff-ec31ca232e3e](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Cabinet/62319141-32fa-4353-b2ff-ec31ca232e3e/config.json) | 0.4863 × 0.4846 × 0.6072 | 1.5472 |
| [Cabinet/63312500-4559-4afb-ac52-53e06ca52b57](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Cabinet/63312500-4559-4afb-ac52-53e06ca52b57/config.json) | 0.5922 × 0.4978 × 0.5721 | 1.5121 |
| [Cabinet/74a23289-5427-4455-9189-679c0534f616](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Cabinet/74a23289-5427-4455-9189-679c0534f616/config.json) | 0.5198 × 0.5813 × 0.6950 | 1.6350 |
| [Cabinet/d14e06fc-0908-4ab4-9efb-675155fdbedd](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Cabinet/d14e06fc-0908-4ab4-9efb-675155fdbedd/config.json) | 0.5500 × 0.4096 × 0.5000 | 1.4400 |
| [Table_Square/08dcb9d0-8a66-3edc-874b-3f437cc45742](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Table_Square/08dcb9d0-8a66-3edc-874b-3f437cc45742/config.json) | 0.6823 × 0.6821 × 0.5354 | 1.4731 |
| [Table_Square/15f76e07-8716-40a8-8d34-01f2158b95bc](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Table_Square/15f76e07-8716-40a8-8d34-01f2158b95bc/config.json) | 0.6823 × 0.6823 × 0.3685 | 1.3085 |
| [Table_Square/38b27b79-f6e6-4ef7-b525-e3c6b6de2619](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Table_Square/38b27b79-f6e6-4ef7-b525-e3c6b6de2619/config.json) | 0.6823 × 0.6823 × 0.5761 | 1.5160 |
| [Table_Square/411a9753-038b-43c9-96fe-d580ee23e23c](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Table_Square/411a9753-038b-43c9-96fe-d580ee23e23c/config.json) | 0.6823 × 0.6807 × 0.5054 | 1.4251 |
| [Table_Square/4d36ad1a-1766-333d-b930-190a3ae5ea67](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Table_Square/4d36ad1a-1766-333d-b930-190a3ae5ea67/config.json) | 0.6823 × 0.6823 × 0.3870 | 1.3270 |
| [Table_Square/59f44afe-59ca-4281-9a2e-753322f2542e](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Table_Square/59f44afe-59ca-4281-9a2e-753322f2542e/config.json) | 0.6714 × 0.6823 × 0.6366 | 1.5749 |
| [Table_Square/c13078c1-d6e9-4ddb-ba2c-6d3bf3919d3e](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Table_Square/c13078c1-d6e9-4ddb-ba2c-6d3bf3919d3e/config.json) | 0.6823 × 0.6823 × 0.5610 | 1.5010 |
| [Table_Square/de084b34-6bb6-461f-8b7a-a9df715aa0a4](../../../TokenHSI/tokenhsi/data/dataset_amass_climb/objects/test/Table_Square/de084b34-6bb6-461f-8b7a-a9df715aa0a4/config.json) | 0.6823 × 0.4502 × 0.4186 | 1.3493 |

각 행의 링크는 해당 asset의 `config.json`이다. 목표 root Z는 `bbox_z/2 + tarSitPos_z`이며 Climb만 추가로 0.94m를 더했다. 데이터 원본은 수정하지 않았다.
