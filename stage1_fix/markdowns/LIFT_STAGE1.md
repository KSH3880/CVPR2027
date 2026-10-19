# Stage 1 — 두 agent 독립 SUPPORT + LIFT

실행 가능한 별도 태스크 `HumanoidMALift`다. **현재 권장 AMP는 [TeamHOI full/masked 별도 설정](LIFT_TEAMHOI_AMP.md)**이며 아래의 단일 masked AMP는 초기 비교용 설정이다. 각 agent가 자기 상자에 접근하여 밑에서 받치고, 제자리의 목표 높이까지 들어 유지한다. 공동으로 한 상자를 드는 태스크나 운반은 포함하지 않는다. 현재 구현·짧은 학습·평가 경로를 검증했으며 lift 학습 성능은 아직 미검증이다.

## 장면과 관계

- 2 agents / 2 boxes / 2 goals, 학습 2048 environments, envSpacing 5.
- agent별 lane 간격 3m. 상자 전방 1.3–2m 거리에서 상자를 향해 서서 시작한다. 물체를 이미 잡은 RSI는 사용하지 않는다.
- 상자 0.55×0.65×0.35m, 질량 5kg, 마찰 1.0. 바닥에서 0.45m 높이의 양쪽 좁은 고정 rail 위에 놓아 손·전완을 넣을 공간을 확보한다. 초기 중심 높이는 약0.627m다. goal XY는 초기 상자 중심, goal Z는 1.6m다.
- H0↔O0, H1↔O1은 SUPPORT(ID6), O0↔G0, O1↔G1은 LIFT(ID7). H↔자기 G는 일반 OWN_GOAL(ID4), H↔H는 TEAMMATE(ID2). 다른 agent의 상자/goal로의 task edge는 없다.
- 6개 entity token의 clean-scene GTA actor/critic을 사용하며 관측은550-D다. 별도 semantic edge MLP가 SUPPORT/LIFT bias를 학습한다. 기존 carry/Push/Door의 관계 taxonomy와 checkpoint는 새 template·interaction metadata로 구분한다.

## 받치기 보상

`p_rh, p_lh, p_lf, p_rf` 네 접촉 proxy를 상자 local 좌표에서 계산한다. 손은 구형 손 중심, 전완은 lower-arm capsule의 손 쪽 축 끝(elbow→hand의72.4086%, 손 중심과7.1447cm 거리)을 사용하며, 상자 local +Z 방향의 반지름4cm만큼 보정해 바닥면과 맞닿을 표면점을 만든다.

raw state S는 다음 점수의 곱이다.

| 항목 | 계산/목적 |
|---|---|
| 바닥면 근접 | 네 점의 바닥면/XY 외곽 오차 중 최대 d에 exp(-100d²) |
| 사각형 중심 정렬 | 네 XY점 평균과 상자 중심 차이에 exp(-50‖c‖²) |
| 무게중심 포함 | 상자 중심이 일관된 convex winding의 사각형 내부에 있고 각 변에서2.5cm 이상 떨어지면1 |
| 면적 | 사각형 면적/0.015m²를 [0,1]로 제한하여 퇴화/교차 사각형 편법 차단 |
| 접촉 | 각 점의 표면 거리≤4cm 및 해당 손/전완 body의 net contact force≥0.5N이면1. 네 점 중 최소값 사용 |

접촉은 **표면 근접 + body net force의 proxy**이며 실제 contact-pair의 위치/힘을 읽은 것은 아니다. 상자-손 쌍 접촉을 엄밀하게 보증하지 않는다. 물리 안정성/학습 난이도는 본학습에서 확인해야 한다.

받치기 progress는 기존 HOLDING과 같은 root→상자 XY 거리 potential이다.

`P_support = 1 / (1 + max(d_root_xy - 0.5, 0) / 1.0)`

`S > 0.9`이면 받치기 성공 보상을 주며 현재 state·progress를1로 포화시킨다. 이 포화는 성공 이력에 latch하지 않는다. 접촉이 풀리면 즉시 현재 raw state로 돌아간다. lift에 곱하는 손 배치 점수도 이 현재 포화 state를 사용한다.

## Lift 보상

초기 상자 orientation을 기준으로 quaternion geodesic angle θ(yaw 포함)와 현재 각속도 ω를 계산한다.

`A = exp(-(θ / 10°)² - (‖ω‖ / 0.5rad/s)²)`

현재 포화 받치기 state를 G, 상자 중심과 고정 goal의3D 거리를 d라 하면:

- progress: `G × A / (1 + d / 1m)`
- state: `G × A × exp(-80d²)`
- success: d≤6cm, 초기 orientation 오차≤5°, 각속도≤0.2rad/s, 선속도≤0.1m/s, raw 받치기 state>0.9를 **연속0.5초** 유지. 하나라도 깨지면 유지 시간을0으로 초기화한다. 성공 bonus는 agent별 episode 최초1회 지급한다.

받치기 state/progress/success, lift state/progress/success의 weight는 각각0.2다. 받치기 success는 현재 성공 상태에서 매step 지급한다(기존 edge reward 방식). 기존 power/collision penalty를 함께 사용하고 agent 간 task reward 공유는 하지 않는다. 한 agent가 넘어지면 기존 multi-agent 환경 계약에 따라 해당 scene 전체가 reset된다. lift 성공만으로 episode를 조기 종료하지 않는다.

progress는 temporal delta/최고기록 갱신이 아닌 거리 기반 potential이며, lift 목표로 가면서 매step 받는 연속 보상이다.

## 초기 단일 OMOMO masked AMP (비교용)

- reference: 기존 loco12개 + 변환된 OMOMO **train** lift2개. sampling 확률은 loco0.33 / lift0.67. 원본 test의 largebox는 학습 목록에서 제외한다.
- 어깨/팔꿈치6D 자세24차원, 해당 관절 속도8차원, 양손 위치6차원을 마스킹한다. native129차원 중38개를0으로 만들어91차원만 prior가 평가한다. 모델에는 원래 손목 DOF가 없다.
- policy의 실제 관측/행동과 원본 모션은 유지한다. discriminator용 live/demo/history를 모두 같은 방식으로 가린다. AMP129×10=1290 입력 차원은 유지한다.
- 이번 전용 태스크는 하나의 masked discriminator를 사용한다. 접근 시 별도 full-body discriminator로 전환하는 TeamHOI dual-discriminator 구성은 구현하지 않았다. masked locomotion prior도 동일 마스크를 쓴다.

## 실행

`stage1_fix/`에서 실행한다. GPU6만 허용하며 기존 runtime이 GPU6 MPS를 검증해 연결한다. MPS daemon을 시작/중단/재시작하거나 설정을 바꾸지 않는다.

```bash
# 새 본학습: scratch, 2048env (현재 자동 시작하지 않음)
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/lift_stage1_train.sh

# 분리된 짧은 학습 경로 점검
TOKENHSI_GPU=6 MAX_ITERATIONS=1 OUTPUT_PATH=output/lift_stage1_check \
  bash tokenhsi/scripts/multi_agent/lift_stage1_train.sh

# checkpoint 평가 / 로컬 창 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/lift_stage1_test.sh CHECKPOINT.pth 16 3
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/lift_stage1_test.sh CHECKPOINT.pth 1 1
TOKENHSI_GPU=6 PORT=6081 bash tokenhsi/scripts/multi_agent/lift_stage1_vnc.sh CHECKPOINT.pth 1 1
```

VNC는 `http://localhost:6081/vnc.html?autoconnect=1&resize=remote` (원격이면6081 포트 포워딩). 기존 OMOMO reference 뷰어의6080 포트는 유지한다. GUI는 지정된 evaluation episode 횟수가 끝나면 종료한다. VNC wrapper가 GPU UUID를 전달하는 경우에도 physical GPU6인지 검증한다.

재개는 `RESUME_CHECKPOINT=...`, seed는 `SEED=...`, 평가의 저장 설정 재사용은 `CFG_ENV=<run>/relation_config.yaml`를 사용한다. 기존 task의 checkpoint는 사용할 수 없다.

## 확인된 범위

- CPU 신규11 + 기존 관계/속도보상 회귀12 =23 tests 통과. 접촉 손실, 사각형 퇴화/중심 이탈/윗면 접촉, 현재 상태 포화, 거리/각도 gate, 연속 유지와 일회성 bonus,38차원/전history 마스킹, actor/critic relation bias gradient 및 데이터 split을 확인했다.
- GPU6·기존 MPS에서4개 평가 환경의 reset/8step/부분reset/AMP demo·live history 유한값 검증 통과.
- 학습2048env, `MAX_ITERATIONS=1`로 유한 점검했다. 기존 rl_games 종료 조건 때문에 실제2회 업데이트 후 checkpoint를 저장하고 정상 종료했다.
- 해당 checkpoint를4env×1회 다시 로드하는 유한 lift 평가 통과(8 agent trials,37step, 성공0). 이는 구현 연결 검사이며 성능평가로 해석하지 않는다.
- 점검 checkpoint: `output/lift_stage1_check/LiftStage1_16-16-29-53/nn/LiftStage1.pth`. 평가 결과: `output/lift_stage1_check_eval/lift_eval.json`.
- 장기 학습은 시작하지 않았으며 기존 학습·뷰어·MPS daemon을 종료하지 않았다.

## 구현 위치

- `tokenhsi/utils/lift_spec.py`: geometry/reward/AMP 마스크/관계 matrix.
- `tokenhsi/env/tasks/multi_agent/humanoid_ma_lift.py`: asset/접근 reset/goal/보상·AMP 통합.
- `tokenhsi/learning/multi_agent/lift_eval.py`: 유한 checkpoint 평가.
- `tokenhsi/data/cfg/multi_agent/lift_stage1.yaml`, `data/cfg/train/rlg/amp_ma_lift_stage1.yaml`, `data/dataset_lift_stage1.yaml` (앞에 `tokenhsi/`): 별도실험 설정·학습 reference 목록.
- `tokenhsi/tests/test_lift_spec.py`, `scripts/multi_agent/check_lift_stage1.py` (둘 다 앞에 `tokenhsi/`): CPU 및 물리 통합 검증.

## 정책 없는 RSI 방식 초기 장면 보기

```bash
TOKENHSI_GPU=6 PORT=6082 bash tokenhsi/scripts/multi_agent/lift_stage1_preview_vnc.sh
# DISPLAY가 있는 로컬 창
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/lift_stage1_view.sh
```

http://localhost:6082/vnc.html?autoconnect=1&resize=remote 에 접속한다(원격이면6082 포트 포워딩). 두 agent·자기 색 상자·고정 rail·1.6m 목표 wireframe 구를 함께 보여 준다. 실제 학습의 접근 초기 상태를 reset하고 물리1step 후 정지한다. 모션 reference RSI나 학습된 policy rollout은 아니다. R 배치 재추첨, N 전체/agent0/agent1 카메라, S 스크린샷. 기존6080/6081 뷰어는 유지한다.

`lift_stage1_preview.py` + `{lift_stage1_view,lift_stage1_preview_vnc}.sh`가 전용 진입점이다. GPU6·기존 MPS만 사용하며3frame GUI 점검과 스크린샷 확인을 마쳤다. 학습/reward/reset 로직은 바꾸지 않았다.

2026-10-16: 좌우(Y) 폭을0.55→0.65m로 확대했다. 질량5kg을 유지하도록 밀도는 자동 재계산하고 rail은 각쪽5cm씩 바깥으로 이동한다. 이전 폭의 점검 checkpoint는 당시 저장된 relation_config.yaml과 함께 사용하며 새 폭 설정에는 새 checkpoint를 사용한다.

## OMOMO RSI 접촉 점검 (2026-10-16)

현재 학습은 `stateInit: Default`이며 object RSI는 구현/활성화하지 않는다. 모션 preview는 원본 bbox×프레임별 scale을 사용하므로 현재 학습 상자0.55×0.65×0.35m와 다르다. 세 lift crop의 scaled 원본 크기 중앙값은 smallbox0.383×0.434×0.234m, largebox0.472×0.460×0.409m, plasticbox0.554×0.277×0.384m다.

현재 상자 크기로 바꾸되 reference object pose를 유지한 CPU FK 기하 검사에서, 네 점 중 최대 바닥면 오차의 프레임별 최솟값은 순서대로35.6/4.5/36.1cm이며 네 점 모두4cm 이내인 프레임은 없다. 이는 일반적인 grasp 접촉 오차가 아니라 **현재 underhand 지지면 기준 오차**다. 원본 크기에서도 해당 조건을 충족하지 않으므로 상자 크기만의 문제가 아니다. 접촉 force는1로 가정한 기하 검사로, 물리 RSI 안정성을 검증하지 않았다. 원시 결과/재현 스크립트는 root runs/lift_rsi_audit/{geometry.json,audit.py}.

손은 반지름4cm 구이며 손목 DOF가 없어 손 회전만 수정해 접촉을 만들 수 없다. Underhand RSI를 추가하려면 현재 상자 바닥에 손/전완 distal 목표점을 배치하고 양쪽 어깨3DOF+팔꿈치1DOF를 함께 IK 보정해야 한다. 이후 관절 제한·충돌·발/균형·물리 정착을 확인해야 하며 원본 grasp 자세를 그대로 RSI로 쓰지 않는다. 현재 AMP는 어깨/팔꿈치 회전·속도와 손 위치를 마스킹하므로 팔꿈치를 reference에 고정할 필요가 없다.

## 1.7m 받치기 자세 설계 뷰어

`lift_support_pose.py`는 별도 정책 없는 자세 검사 도구다. 학습 목표1.6m/모션/초기화는 바꾸지 않는다. 현재 asset의 관절 범위를 지키는 양팔 IK로 상자 중심1.7m, 손/전완 끝 중심1.485m, 수평 전완을 설계한다. 팔꿈치는 양옆으로 벌리고 양팔을35도 안쪽으로 향하게 하며 상자 X위치도 함께 최적화한다. 손·전완 끝의7.1447cm 간격과 반지름4cm를 사용한다. 현재 상자 크기0.55×0.65×0.35m/질량5kg/마찰1.0.

동일 자세를 몸통 고정(파랑)/자유 루트(초록) 두 장면에 둔다. 상자는 두 경우 모두 자유 물체이며 외부 joint/고정/반복 위치 재설정 없이 중력을 적용한다. 시작은1step 후 정지이며 Space 물리재생/정지, R 자세복원, N 장면카메라 전환, S 스크린샷이다. 빨간 wire 점은 설계한 초기 접촉점이며 정지 화면에 표시된다.

2026-10-16 검사: IK 최대오차0.0043mm, 관절 범위 내, 실제 PhysX 초기 FK 차이0.513mm 이내. CPU PhysX 120Hz/2substeps, 원래 asset PD stiffness/damping 유지. 3초 뒤 몸통고정 상자는 중심1.6705m/기울기3.904도, 손·전완 총 수직 반력48.81N으로 하중49.05N을 지지했다. 자유 루트는 앞으로 넘어져 root높이0.251m로 하락했다. **관절/접촉상 받치기 가능하나 전신 균형 또는 실제 lift 정책의 성공을 보인 것은 아니다.** GPU6 렌더링/기존 MPS 연결만 사용하며 물리는CPU다.

원시 design.json/report.json/pose.png는 root `runs/lift_support_pose_170/`에 있다. 초기 구현의 fixed-root setter/PD target 임시버퍼 수명 문제를 indexed setter/영속 target buffer로 보완했으며 headless와GUI 실행 결과 일치를 확인했다.
