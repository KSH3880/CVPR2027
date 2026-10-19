# 4-agent 가변 그룹 직선 Joint Push

현재 프로필은 `partitionPush.version: 2`다. 로컬 실행용 설정이며 GPU·학습·시뮬레이션은 서버에서 실행하지 않았다.

## 관계 샘플링

승현 `stage2_shared_full_success.yaml`의 `shared_support.sample_groups`와 같은 순차 규칙을 사용한다. 전체 사람은 항상4명이다.

1. 사람 순서를 무작위로 섞는다.
2. 남은 사람이 둘 이상이면 현재 사람에 대해 coupled80% / NONE20%를 추첨한다.
3. coupled면 그룹 크기를2명~남은 인원수에서 균등 추첨하고 해당 사람들을 함께 배정한다.
4. NONE은 한 명의 독립 Push이며, 마지막 한 명도 독립이다.

따라서4명공동/3+1/2+2/2+1+1/전원독립이 모두 가능하다. 각 사람을 독립적인 Bernoulli(.8)로 뽑는 방식이나 환경80%를4명공동으로 만드는 방식은 아니다. 그룹은 reset 때 다시 추첨하고 episode 동안 유지한다. 디버깅용 `group_preset: all_shared`와 `all_solo`도 가능하다.

## 물리 상자와8개 관측 토큰

초기 일반 상자8개를1인 크기·질량 분포에서 생성한다. 과제 그룹마다 물리 상자 하나를 사용하고, 그룹 인원이n명이면 X축을n등분한다. 모든 그룹의 구역 수를 합하면 항상4개다. 따라서 초기 일반 상자 중 무작위4개를 제외하고, 남은 일반4개와 과제 구역4개로 항상8개 물체 토큰을 구성한다.

| 구성 | 과제 물리 상자 | 과제 구역 토큰 | 남는 일반 상자 | 활성 물리 상자 합계 |
|---|---:|---:|---:|---:|
| 4명공동 | 1 | 4 | 4 | 5 |
| 3+1 / 2+2 | 2 | 4 | 4 | 6 |
| 2+1+1 | 3 | 4 | 4 | 7 |
| 1+1+1+1 | 4 | 4 | 4 | 8 |

Isaac Gym 생성 후 실제 형상·질량 setter를 바꾸지 않도록 일반8개+후보leader4명×가능그룹크기4종의24-actor bank를 사전 생성한다. reset마다 해당 그룹 크기의 후보를 사용하고, 제외한 일반4개 및 미사용 후보는 장면 밖으로 격리한다. 이들은 관측·할당에 들어가지 않는다. bank actor 수와 활성 장면 상자 수를 구분한다.

과제 그룹은 서로 떨어진 네 station 중 그룹 leader의 station에 배치하며 장면 yaw는 무작위다. 일반8개 후보 위치는 반경24~30m의 여덟 부채꼴 영역에서 샘플링한 뒤4개를 남긴다. 네 사람은 해당 과제 상자 뒷면에서 loco로 시작하고, 남은 활성 물체 중 실제 최근접 상자를 계산해 물리 할당을 확인한다. 이후에는 자기 그룹 상자를 유지하고, 그 상자의 구역 중 root XY에서 가장 가까운 구역 중심을 매 시점 선택한다. 구역 독점 소유권은 없어서 두 사람이 같은 구역을 선택할 수도 있다.

## 크기·질량 분포

설정의 `size_min/max`는 **1인 기준 X와 공통 Y/Z**다. 남는 일반 상자도 이1인 분포에서 축별 독립 균등 샘플링한다. 일반 상자끼리 크기·질량을 공유하지 않는다.

| 그룹 인원n | 물리 X | Y | Z | 총질량 | 분할 |
|---:|---:|---:|---:|---:|---:|
| 1 / 일반 상자 | 1~1.3m | 1.8~2.2m | 1.2~1.5m | 20~30kg | 1 |
| 2 | 2~2.6m | 동일 | 동일 | 40~60kg | 2 |
| 3 | 3~3.9m | 동일 | 동일 | 60~90kg | 3 |
| 4 | 4~5.2m | 동일 | 동일 | 80~120kg | 4 |

상자마다 X기준폭과 질량기준값을 추첨한 뒤n배 한다. 실제asset 밀도는 총질량/실제부피다. 크기·질량은 환경 생성 시 정하고, reset에서는 적합한 bank 후보를 선택한다. 마찰은 box0.6 / plane static·dynamic1.0이다. 물리 회전을 잠그지 않는다.

손point는 선택 구역 뒷면 중심에 양손간격0.3m로 배치하며 바닥 기준Z×0.5 높이다. 상자 pose를 따라 회전한다. 목표는 reset box local+Y로2~6m 떨어진 위치이며 episode 동안 고정한다. 같은 그룹은 같은 goal을 쓰고 다른 그룹은 독립 goal을 쓴다.

## 보상·모델·AMP

Contact는 양손 중 큰3D오차의 `exp(-10*d²)`이고 AT는 실제 물리 상자 중심과 고정goal 거리의 `exp(-10*D²)`다. state/progress/success는edge당0.2다. AT dense 항에는 **같은 그룹 안의** Contact phi 최솟값만 곱한다. AT 성공은 위치phi≥0.9·높이오차≤0.001m·같은 그룹 Contact phi≥0.9를 요구한다. 독립agent가 다른 그룹의 접촉을 기다리는 조건은 없다. 자기edge합산에 power와 사람CPA penalty를 추가한다.

물체 수가 다른 Stage1의 entity tokenizer/Transformer 및 entity별 RMS를 재사용한다. 현재 저장소의 koo Push CA4에서 actor encoder·관측 RMS를 동결하고 action head/CA4/critic/AMP를 학습한다. 서로 다른 구역 토큰도 같은 물리그룹의 canonical goal이면 coupled로 처리하며 다른 그룹은NONE이다. 승현 코드의 관측 분할·그룹 샘플링 방식을 적용하되, 네트워크는 koo Push Stage1 호환 CA4다.

초기화는loco-only이며 interaction RSI는 없다. AMP history는reset 상태를 반복한다. `partition_push_motions.yaml`의 raw Push180/Push와loco를 PUSH조건부0.67/0.33으로 사용한다. door expert 확률은0이며 unused doorOpen manifest자리도loco를 가리켜door데이터를 요구하지 않는다.

## Stage1 경로와 로컬 준비

기본 source는 `/home/hwanhee/koo_cvpr/stage1_fix/output/open_push_push180_contact_at_warmstart`다. 디렉터리를 주면 가장 높은 번호의contact-at warmstart를 고르고, 같은 번호의run이 여럿이면 정확한파일을 요구한다. 확인 당시 최신은 `OpenPushPush180ContactAtWarmstart_09-17-51-36/nn/OpenPushPush180ContactAtWarmstart_00029000.pth`다.

로컬에는 `stage2_jointpush/ports/koo_push` 코드·assets, 위Stage1파일과 다음 데이터를 준비한다. 서버 절대경로의symlink는로컬 위치로 연결해야 한다. 외부 원본은 수정하지 않는다.

- `tokenhsi/data/dataset_amass_loco`
- `tokenhsi/data/dataset_bones_push180`
- `tokenhsi/data/dataset_bones_push`

Isaac Gym/PyTorch/rl_games가 준비된 conda환경을 활성화한다. 스크립트는현재 활성환경을 사용하며로컬 기본GPU는0이다. 기존 프로세스를 시작·종료하지 않는다.

```bash
# 신규 학습: 로컬 Stage1 파일
TOKENHSI_GPU=0 STAGE1_CHECKPOINT=/local/path/OpenPushPush180ContactAtWarmstart_00029000.pth \
  bash stage2_jointpush/scripts/partition_push_train.sh

# 환경 수 변경이 필요할 때 명시 (기본1024)
NUM_ENVS=512 STAGE1_CHECKPOINT=/local/path/stage1.pth \
  bash stage2_jointpush/scripts/partition_push_train.sh

# 같은 version2 Stage2 재개: 저장 env 복원
RESUME_CHECKPOINT=/local/path/partition_stage2.pth \
  bash stage2_jointpush/scripts/partition_push_train.sh

# headless 평가 / 로컬 viewer
bash stage2_jointpush/scripts/partition_push_test.sh /local/path/partition_stage2.pth
HEADLESS=0 NUM_ENVS=1 bash stage2_jointpush/scripts/partition_push_test.sh /local/path/partition_stage2.pth

# 기존 VNC desktop 안에서 viewer (VNC 서버를 만들지는 않음)
NUM_ENVS=1 bash stage2_jointpush/scripts/partition_push_vnc.sh /local/path/partition_stage2.pth

# 추론 ablation: X와 질량 모두 1인 기준. 3명 그룹이면 X3.6m/75kg.
ABLATION_SIZE='1.2 2.0 1.35' ABLATION_MASS=25 \
  OUTPUT_PATH=/local/path/ablation_1p2_25kg \
  bash stage2_jointpush/scripts/partition_push_test.sh /local/path/partition_stage2.pth
```

기본 산출물은 `runs/jointpush/partition_push_train`, `partition_push_test`다. `OUTPUT_PATH`로실험결과를 분리한다. 신규 학습의환경설정은 `PARTITION_PUSH_ENV_CONFIG`, 모션manifest는 `PARTITION_PUSH_MOTION_FILE`로지정한다. 재개·평가는checkpoint 환경을 복원한다. version1의4인고정checkpoint는새프로필에서직접재개하지않으며Stage1부터새로시작한다.

## 확인 범위

CPU로분할기하·그룹gate·5종그룹실제reset메서드·4개제외/8토큰·부분reset·인원비례형상/질량·랜덤일반상자·서로다른구역의같은그룹coupled/다른그룹NONE·샘플링통계를검사했다. 기존shape회귀포함19개통과. source29000network설정일치는이전CPU검사에서확인했다. Isaac Gym생성·실제전이forward·GPU메모리·학습·추론성능은실행하지않았다.
