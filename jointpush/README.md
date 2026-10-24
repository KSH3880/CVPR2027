# Partial Joint Push — RawCuboid baseline

현재 기준은 `/home/hwanhee/jhh/TokenHSI_stage1_stage2_base`의 **stage1_v2_fixed_push_360_raw_cuboid**다. 외부 폴더는 읽기 전용이며 코드 복사본은 `ports/jhh_raw_cuboid/`, 새 partial 구현은 `partial/`에 둔다. 생성 설정·검증·학습 산출물은 모두 프로젝트의 `runs/jointpush/` 아래에 저장한다.

## 준비된 baseline

- 로컬 checkpoint: `jointpush/checkpoints/jhh_raw_cuboid/Stage1V2FixedPush360RawCuboid_00022500.pth`.
- 준비된 workspace: `runs/jointpush/raw_cuboid_partial_rear180_20261023/workspace/`.
- Stage1 actor encoder와 scene RMS를 전이·동결하고, 새4층 coordination을 identity 초기화한다. action head·critic·discriminator는 원래 Stage2 방식대로 학습하며 Stage1 optimizer/epoch는 이어받지 않는다.
- 새 학습은 PUSH 전용이다. baseline의 SIT/CLIMB/carry 혼합은 제외하되 5개 task embedding/4family AMP 규격은 유지한다. AMP expert는 baseline과 같이 push_loco33%/PUSH67%, RSI는 partial 기존 방침대로 loco-only다.

## 환경과 보상

사람4명, object token8개(담당 분할4+여분4), 물리 bank24개다. coupled 확률0.8로1~4명 그룹을 정하고 같은 그룹은 실제 상자 하나를 함께 민다. 분할은 관측/손 목표용이며 보상에서 실제 상자 위치를 바꾸지 않는다. attention에서 다른 agent나 상자를 강제로 가리는 mask는 없다.

1인분 크기는 baseline의 X1.2~1.7/Y1.45~1.7/Z1.45~1.7m이고 Y≥Z를 만족한다. n인 그룹은 X와 질량만 n배: 질량 n×U(20,30)kg, 마찰0.3. 접촉 기준 높이는 바닥에서1.04m다. 목표는 실제 상자 중심의 초기 로컬+Y 방향2~10m. 출발 위치는 상자 중심에서 goal 반대쪽을 중심으로 ±90°(후방180°), 몸 heading은360도 랜덤이다. 중심 반경1.65~6m 중 상자 외접원+0.45m clearance를 만족하는 구간에서 샘플링한다. 에피소드1200step, 학습2048환경이다. Stage1의 epoch별 시작방향 curriculum 대신 후방180° 랜덤 출발을 사용한다. 그룹과 물리 상자 위치는 거리와 무관하게 랜덤 배정한다. 그룹 내 분할 구역도 랜덤 순열로 중복 없이 배정하고 에피소드 동안 고정한다. 최근접 상자 검사나 거리 기반 재배정은 사용하지 않는다. 공동 구성원의 실제 상자 공유는 의도된 것이며 서로 다른 그룹이 같은 상자를 맡지는 않는다.

보상은 RawCuboid의 contact/AT state·progress·success 각각0.2를 사용한다. contact의 midpoint-only 연속 점수·1.04m 높이/±0.0325m 허용·유한 면 거리/양손 반력/상자 반력 조건과 physics tick별 접촉 OR 누적을 유지한다. 담당 분할 구역 중앙이 각 agent의 접촉 기준점이다. 접근 progress는 실제 전체 상자 장애물을 우회하여 담당 뒷면 구역으로 가는 baseline navigation 거리다.

**공동 gate:** 같은 물리 상자를 맡은 모든 agent의 양손이 실제 뒷면에서0.18m 이내에 있고, 그 면의 안쪽 법선이 goal 쪽을 향할 때 AT state/progress/success를 지급한다. 목표 도달 위치에서는 방향의0벡터 판정을 예외 처리한다. 접근·contact 보상은 gate로 끄지 않아 손을 붙이러 갈 수 있게 한다. AT 성공은 baseline의 exp(-10·3D거리²)≥0.9 및 높이오차≤1mm에 gate를 추가한 것이다.

**AT progress:** `1 / (1 + max(||실제 공동 상자 중심.xy - 공통 goal.xy|| - 0.1, 0))`. 분할 구역 중심이나 agent root로 계산하지 않는다. 이전 step 대비 이동량이 아닌 baseline의 현재 거리 점수이며, 같은 상자 구성원은 같은 값을 받는다.

Task reward 공유는 같은 물리 그룹 안에서 본인90%/다른 구성원 평균10%, solo는본인100%다. Power와 충돌 페널티는 baseline 설정을 유지하며4인에서는 가장 가까운 다른 사람과의 root 거리로 선형 충돌 페널티를 계산한다. 이전 CPA 방식은 사용하지 않는다. 전체 task/AMP 가중치는0.5/0.5다.

## 명령

프로젝트 루트에서 실행한다. 기본 conda는 `tokenhsi_juan`, GPU는 반드시 명시한다.

```bash
# 로컬 코드/읽기 전용 데이터 링크 무결성 검사: GPU 미사용
bash -c 'source jointpush/scripts/partial_common.sh; "$PARTIAL_PYTHON" jointpush/scripts/partial_prepare.py --verify'

# 새 workspace 준비: 생략 시 지정 RawCuboid run의 최신 numbered checkpoint 사용
python jointpush/scripts/partial_prepare.py --workspace runs/jointpush/my_partial/workspace

# 준비된22500회 baseline을 사용해 새 Stage2 학습
TOKENHSI_GPU=6 \
PARTIAL_WORK=/home/hwanhee/juan/CVPR2027/runs/jointpush/raw_cuboid_partial_rear180_20261023/workspace \
bash jointpush/scripts/partial_train.sh

# 저장된 Stage2 checkpoint headless 평가
TOKENHSI_GPU=6 HEADLESS=1 PARTIAL_WORK=/absolute/path/to/training/workspace \
bash jointpush/scripts/partial_test.sh /absolute/path/to/Stage2.pth

# 로컬 viewer
TOKENHSI_GPU=6 HEADLESS=0 PARTIAL_WORK=/absolute/path/to/training/workspace \
bash jointpush/scripts/partial_test.sh /absolute/path/to/Stage2.pth

# 서버 noVNC viewer
TOKENHSI_GPU=6 PORT=6086 PARTIAL_WORK=/absolute/path/to/training/workspace \
bash jointpush/scripts/partial_vnc.sh /absolute/path/to/Stage2.pth
```

GPU/MPS runtime은 로컬 snapshot을 사용하며 daemon을 시작·종료하지 않는다. 본학습은 GPU 여유50GiB 이상·기존 Python client 최대1개 조건에서만 시작한다. Viewer는 native 경로를 사용한다. 새 GUI/장치 조합은 기존 학습과 공유하는 MPS에서 시험하지 않는다. GPU6 native GUI/strict 복원/rollout은 확인했다. 학습은 선택GPU 한 장을 CUDA0으로 노출하고 검증된 MPS pipe를 사용하며 viewer는 unfiltered CUDA의 물리index와 Vulkan을 맞추고 MPS pipe를 제거한다. GPU6에서2048환경 PPO 본학습을 시작했고 실제 coordination gradient·2epoch 업데이트/동결/저장을 확인했다. 최신 로그·weight·tmux 경로는 `runs/jointpush/latest_partial_training.json`을 따른다.

## 검증과 정리

CPU13개 회귀(후방180°/비중첩/랜덤 배정4개 추가): 한 손 오배치 시 해당 그룹만 gate 닫힘, 실제 상자 중심/region 불변성, 성공/반력, 회전·agent 순열, baseline navigation/solo 보상 일치. CPU32환경24step과 strict Stage1 tensor165개 전이·새104개, coordination gradient/업데이트·actor/RMS 불변을 확인했다. 상세 기록은 `runs/jointpush/raw_cuboid_partial_20261023/`에 있다. validation_transfer.pth는 CPU 검사 fixture이며 학습 resume용이 아니다.

이전 koo_partial/koo_1021 전용 코드17개 항목은 정리했고 `runs/jointpush/retired_koo_partial_20261023.tar.gz`에 보관했다. 기존 checkpoint·실험 결과와 외부 원본은 보존한다. 과거 Native Y2m attention 진단에 쓰는 `ports/koo_native_y2m`과 해당 진단 도구는 남겨뒀다. 새 partial은 `stage2_jointpush` 코드에 의존하지 않는다.

2026-10-23 출발/배정 갱신 검증: 2,048개 샘플의 후방 반원·clearance·사람 최소0.9m 간격, 4,096개 그룹의 랜덤 배정/그룹 간 물리 상자 비공유, CPU32환경24step 및 에피소드 내 구역 배정 불변을 확인했다. 새 reset 계약은 partitionPush.version=4이며 준비된 workspace는 rear180 경로다. 기록: `runs/jointpush/raw_cuboid_partial_rear180_20261023/cpu_check.log`.

## 현재 학습 TensorBoard

현재 GPU6 Partial 학습 전용 주소는 **http://localhost:6016**이다. VS Code PORTS에서6016을 포워딩한다. tmux는 `tensorboard_partial_comparison`, 서버 설정/로그는 `runs/jointpush/tensorboard_raw_cuboid/`다. 기존 TensorBoard와 학습은 변경하지 않으며 GPU를 사용하지 않는다.

Scalars에서 `relation/partial_push/goal_distance`(작을수록 좋음), `relation/partial_push/contact`, `relation/partial_push/group_gate`, `relation/partial_push/at_success`와 `reward_terms/`를 확인한다. at_success는 agent-step 비율이며 episode 성공률이 아니다. 대부분 scalar의 Step축은 누적frame이고 `episode_lengths/iter`의 Step축은iteration이다.10초마다 새 로그를 읽는다.

현재6016 TensorBoard는 `current_rawcuboid_rear180`와 `previous_koo_partial` 두run을 함께 읽는다. Runs에서 둘 다 선택하면 공통tag의곡선이겹친다. 원본event는수정하지않고비교링크만사용한다. 이전/현재의episode상한600/1200과보상식차이로같은수치의성능비교보다추이를확인한다.
