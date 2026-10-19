# Joint Push 랜덤 형상·질량 설정

기준: 2026-10-15. 별도 `HumanoidMAJointPushRandomShape` 실험이다. 기존 GPU6의 고정 크기·60kg 학습과 구분한다.

## 범위

| 항목 | 범위 | 격자 간격 |
|---|---|---:|
| X: 미는 면의 가로 폭 | **2.0–3.0m** | 0.2m |
| Y: 미는 진행 방향 깊이 | **1.0–2.6m** | 0.2m |
| Z: 높이 | **1.0–1.5m** | 0.25m |
| 질량 | **40–60kg** | 5kg |

사용자가 지정한 하한은 폭2m/깊이1m/높이1m이며, 질량 상한은60kg이다. 폭·깊이 상한, 높이 상한1.5m, 질량 하한40kg은 우선 적용한 기본값이다. 축과 질량을 독립적으로 조합하며 총 **162개 형상 × 5개 질량 = 810조합**이다.

## 샘플링과 물리

환경 생성 때 각 환경에 한 조합을 균등 배정한다. 2048환경 안에 여러 형상과 질량이 함께 존재하며, 같은 환경의 형상과 질량은 reset 후에도 유지한다. episode마다 actor를 재생성하거나 GPU 준비 후 질량을 setter로 바꾸지 않는다.

실제 asset을 해당 X/Y/Z와 `density=mass/(X×Y×Z)`로 생성한다. 관성도 같은 실제 형상·질량에 맞춰 생성한다. 현재 범위에서 가능한 밀도 범위는 약3.42–30kg/m³다. 크기에 따라 질량이 자동 증가하는 고정 밀도 방식이 아니다.

`pushBox.size/mass`의2.6m/60kg은미사용슬롯과원본호환용기준값으로남으며, 실제공유상자값은`jointPush.shape_randomization`과환경별`_box_size[:,2]`/`_payload_masses`가결정한다.

미사용 object 슬롯은 기존 호환 구조를 유지한다. 공유 payload인 object2만 랜덤 형상/질량 asset을 사용한다.

## 크기에 따라 함께 바뀌는 항목

- 실제 collision/visual 형상과 관성.
- 관측에 들어가는 local bbox corners.
- reset 시 상자 중심 높이 `Z/2`와 목표 중심 높이.
- 고정된 미는 면의 중심·폭 및 손 목표 높이 계산.
- agent 시작 반경: 해당 형상의 XY 외접원 반경 +1–2m.
- 손 중점 최소 가로 간격: **폭X × 0.25**, 현재0.5–0.75m.
- PUSH 접근 보상에 쓰는 XY 외접원 반경.

손 목표 높이는 상자 바닥 기준0.8m를 유지한다. 질량을 명시적인 추가 observation으로 넣지는 않는다. 형상은 기존 bbox 관측에 반영되고, 질량 차이는 물리 반응에 나타난다.

## 유지하는 조건

기존 softgate 실험과 같은 모델·보상·AMP·loco-only RSI·PPO를 사용한다. 상세 공통값은 [JOINT_PUSH_SETTINGS.md](JOINT_PUSH_SETTINGS.md)를 따른다.

- 목표 방향은 초기 box local+Y로 정렬하고 거리는2.2–5m 랜덤.
- 미는 면은 local−Y로 시작하고 episode 동안 같은 물리면 유지.
- agent 위치와 heading은 각각360° 랜덤.
- 두 root의 최소 수평 간격0.8m.
- 마찰: box0.6, plane static/dynamic 각각1.0.
- 2048환경, 2agent, horizon32, minibatch16384, mini_epochs6, LR2e-5.
- Stage1 epoch29000 초기값에서 새 Stage2 시작. 기존 고정 크기 Stage2를 자동 resume하지 않음.

최소 깊이1m와 높이1–1.5m로 기본 형상을 제한하지만, 회전을 물리적으로 잠그는 것은 아니므로 어떤 힘에서도 절대 넘어지지 않도록 강제한 구성은 아니다.

## GPU와 실행

새 실험 train/test/view wrapper는 **physical GPU0**을 사용한다. GPU1은 초기 짧은 비학습 검증에만 사용했고, 현재 새 학습 대상으로 사용하지 않는다. 기존 GPU6 학습과 MPS 서버는 변경하지 않는다.

```bash
# 신규 학습
bash stage2_jointpush/scripts/joint_push_random_shape_train.sh

# 같은 랜덤 형상·질량 모델 재개
RESUME_CHECKPOINT=/absolute/path/checkpoint.pth \
  bash stage2_jointpush/scripts/joint_push_random_shape_train.sh

# checkpoint의 형상/질량 분포 복원 평가
bash stage2_jointpush/scripts/joint_push_random_shape_test.sh /absolute/path/checkpoint.pth

# 로컬 viewer
HEADLESS=0 bash stage2_jointpush/scripts/joint_push_random_shape_test.sh /absolute/path/checkpoint.pth

# VNC viewer
PORT=6090 bash stage2_jointpush/scripts/joint_push_random_shape_view.sh /absolute/path/checkpoint.pth
```

기본 output: `runs/jointpush/random_shape_mass_softgate_training/`. 고정 크기 모델, 3agent preview, 단일40kg mass preview와 별도 계약이다. 기존 고정 크기 checkpoint를 이 profile의 resume/view에 넣지 않는다.

- Env: `ports/koo_push/tokenhsi/data/cfg/multi_agent/joint_push_random_shape_mass_softgate_ca4.yaml`
- Train: `ports/koo_push/tokenhsi/data/cfg/train/rlg/joint_push_random_shape_mass_softgate_ca4.yaml`
- Task: `ports/koo_push/tokenhsi/env/tasks/multi_agent/humanoid_ma_joint_push_random_shape.py`
- 격자: `ports/koo_push/tokenhsi/env/tasks/multi_agent/joint_push_shape.py`

## 실행 기록

본학습은 tmux `jointpush_randomshape_gpu0`에서 GPU0/2048환경/seed6397로 시작했다. 기존 Stage2 checkpoint를 resume하지 않고 Stage1 초기값 전이를 사용한다.

CPU shape/spawn/reward13개, 초기GPU1 64환경120step, GPU0 2048환경2iter학습 및 저장/16환경60step복원을 확인했다. GPU0검증에서는810조합중740조합을배정했고 실제질량/관성오차가각각0.001이내였다. 초기2048환경검증의전체reset전부분reset검사순서오류는전체reset후부분reset검사로수정했으며실패기록은보존한다. 성능결과는아니다.

## 원본 Stage1 설명

전이한OPEN/PUSH Stage1의환경·관계그래프·RSI·contact/AT보상·embedding·AMP·Stage2차이는 [JOINT_PUSH_SETTINGS.md의15절](JOINT_PUSH_SETTINGS.md#15-원본-push-stage1-openpush-contactat-warmstart)을참조한다.
