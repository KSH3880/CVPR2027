# 4-agent 가변 그룹 직선 Joint Push

현재 프로필은 `partitionPush.version: 2`다. 로컬 GPU 실행을 지원하며 현재 검증 기록은 문서 하단과 changelog를 따른다.

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

과제 그룹은 서로 떨어진 네 station 중 그룹 leader의 station에 배치하며 장면 yaw는 무작위다. 일반8개 후보 위치는 반경24~30m의 여덟 부채꼴 영역에서 샘플링한 뒤4개를 남긴다. 신규 학습의 네 사람은 각자 배정 상자 주변 360° 임의 위치에서 loco로 시작하고, 남은 활성 물체 중 실제 최근접 상자를 계산해 물리 할당을 확인한다. 이후에는 자기 그룹 상자를 유지한다. reset에서 같은 그룹 내 사람–구역 중심의 XY 거리 합이 최소가 되는 일대일 배정을 선택하며, 한 구역을 여러 사람이 배정받을 수 없다. 배정은 에피소드 동안 고정되고 관측 graph와 손 목표·보상이 같은 배정을 사용한다. 부분 reset은 해당 환경만 재배정한다.

신규 기본 `spawn_mode: surround_independent`는 각 상자의 XY 외접원 반경에 U(1,2)m를 더한 거리에서 사람별 방위각을 독립 U(0,2π)로 샘플링한다. 초기 몸 heading도 위치·상자 방향과 독립 U(0,2π)이며, 사람 root 간 거리는 최소 1.2m다. 상자와 초기 겹침을 피하면서 앞·뒤·양옆 어느 쪽에서도 접근하도록 학습한다. 사람 4명, 그룹 샘플링, 상자 분포, 목표, 손 contact point와 보상은 기존 설정을 유지한다.

`spawn_mode`가 없는 기존 checkpoint는 `rear_lanes`로 해석하여 예전 뒷면 정렬 시작을 재현한다. 재개·평가는 저장 환경을 복원하므로 기존 checkpoint를 resume하는 것만으로 새 spawn 설정이 적용되지 않는다. 현재 실행 중인 학습은 기존 분포로 계속 실행되며, 변경된 기본 YAML은 새 학습을 시작할 때 적용된다.


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

초기화는loco-only이며 interaction RSI는 없다. AMP history는reset 상태를 반복한다. `partition_push_motions.yaml`의 변환된 Push180 19개와 loco를 PUSH조건부0.67/0.33으로 사용한다. door expert 확률은0이며 unused doorOpen manifest자리도loco를 가리켜door데이터를 요구하지 않는다.

## Stage1 경로와 로컬 준비

기본 source는 이 checkout의 `stage2_jointpush/output/OpenPushPush180ContactAtWarmstart_00029000.pth`다. `STAGE1_CHECKPOINT`로 다른 파일이나 디렉터리를 지정할 수 있다. 디렉터리에서는 가장 높은 번호를 선택하며 같은 번호의 run이 여러 개면 정확한 파일이 필요하다.

변환된 학습 데이터는 `stage2_jointpush/motiondata/dataset_bones_push180/`에 보관하고, 실행 포트의 `tokenhsi/data/dataset_bones_push180`에서 상대 심링크로 참조한다. `dataset_amass_loco`는 기존 로컬 Stage1 데이터에 연결한다. `output/raw_push_amp`의 raw NPZ는 사용하지 않는다.

`partition_push_motions.yaml`은 Push180 19개와 loco 4개를 사용한다. 이전 목록의 별도 `dataset_bones_push` 2개(`003__A342`, `101__A341`)는 제외했다. 따라서 이전 21개 혼합 목록과 AMP 분포가 다르다. doorOpen 자리에는 기존처럼 loco 하나를 사용하고 door expert 확률은 0이다.

Isaac Gym/PyTorch/rl_games가 설치된 conda 환경에서 실행한다(기본 `tokenhsi`, 다른 활성 환경은 유지). 로컬 기본 GPU는 0이며 RTX 5090에서는 구형 TorchScript fusion을 비활성화한다. conda 활성화 hook의 unset 변수와 compiler 선택도 처리한다.

```bash
# 신규 학습: 받은 기본 Stage1 파일
bash stage2_jointpush/scripts/partition_push_train.sh

# 다른 로컬 Stage1 파일
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

# 로컬 viewer 전용 진입점 (기본 1환경, GPU0)
TOKENHSI_GPU=0 bash stage2_jointpush/scripts/partition_push_view.sh /local/path/partition_stage2.pth

# 추론 ablation: X와 질량 모두 1인 기준. 3명 그룹이면 X3.6m/75kg.
ABLATION_SIZE='1.2 2.0 1.35' ABLATION_MASS=25 \
  OUTPUT_PATH=/local/path/ablation_1p2_25kg \
  bash stage2_jointpush/scripts/partition_push_test.sh /local/path/partition_stage2.pth
```

기본 산출물은 `runs/jointpush/partition_push_train`, `partition_push_test`다. `OUTPUT_PATH`로실험결과를 분리한다. 신규 학습의환경설정은 `PARTITION_PUSH_ENV_CONFIG`, 모션manifest는 `PARTITION_PUSH_MOTION_FILE`로지정한다. 재개·평가는checkpoint 환경을 복원한다. version1의4인고정checkpoint는새프로필에서직접재개하지않으며Stage1부터새로시작한다.

## 확인 범위

로컬 CPU 테스트 14개 통과. AMP manifest의 loco 4개/doorOpen 자리 1개/Push180 19개 파일 경로, NPY 로드와 rotation/root translation 유한값을 확인했다. 이동한 모션·보고서 57개는 이동 전후 SHA-256이 같다.

RTX 5090/GPU0에서 기본 1024환경·`MAX_ITERATIONS=2`로 받은 Stage1 29000 전이, 학습 업데이트, checkpoint 저장까지 rc=0을 확인했다. 저장 모델의 16환경·30step·2회 복원 평가도 rc=0이다. 성공 성능을 평가한 장기 학습은 아니다. 로그와 checkpoint는 `runs/jointpush/partition_local_check_20261019/`에 있다.

```bash
# 독립된 출력 경로로 짧은 실행 검증
TOKENHSI_GPU=0 MAX_ITERATIONS=2 OUTPUT_PATH="$PWD/runs/jointpush/partition_push_new_check" \
  bash stage2_jointpush/scripts/partition_push_train.sh
# 저장 checkpoint를 화면 없이 복원 검증
NUM_ENVS=16 EPISODE_LENGTH=30 REPEATS=2 \
  bash stage2_jointpush/scripts/partition_push_test.sh /absolute/path/PartitionPushCa4.pth
```

## 로컬 viewer 화면 표시

`partition_push_view.sh`는 Isaac Gym 기본 viewer를 사용한다. 별도의 센서 영상 창과 Tk/Pillow 확대 표시는 제거했다. reset마다 사람·배정 상자·목표의 범위로 카메라를 맞추며, 이후 Isaac Gym 기본 카메라 조작을 사용한다. 기본 1환경이고 Esc 또는 창 닫기로 종료한다.

이 로컬 호스트에서는 Vulkan ICD 자동 탐색 시 네이티브 framebuffer가 검게 출력된다. view wrapper는 기존 `VK_ICD_FILENAMES`/`VK_DRIVER_FILES` 지정이 없고 NVIDIA manifest가 있으면 NVIDIA ICD를 선택한다. 다른 Vulkan 드라이버를 명시한 경우에는 사용자 설정을 유지한다. 학습 환경과 드라이버 설치는 변경하지 않는다.

1000iter checkpoint의 1환경/30step 네이티브 실행 rc=0 및 1600×900 실제 프레임을 확인했다: `runs/jointpush/partition_view_check_20261019/native_partition_nvidia_icd.png`.

## 이전 뒷면 정렬 학습 이어가기 (2026-10-20)

기존 3,000iter checkpoint의 `RESUME_CHECKPOINT` 경로를 사용해 이어 학습한다. 저장 환경에서 spawn_mode가 없으면 코드의 rear_lanes 기본값으로 복원되어 상자 뒷면·구역별 정렬 시작을 유지한다. 신규 기본 YAML의 surround 설정은 이 재개 실행에 적용되지 않는다. 실제 실행 명령·원본 checkpoint는 `runs/jointpush/partition_push_rear_resume_20261020/launch.sh`와 `launch.json`에 기록했다. 결과는 같은 디렉터리의 새 run에 저장한다.

### 구역 배정 수정 (2026-10-20)

새 checkpoint는 `partition_region_assignment=exclusive_reset_min_distance_v1`을 저장한다. 기존 checkpoint도 로딩할 수 있으나 새 프로세스에서는 위 일대일 고정 배정을 적용하고 이를 로그에 알린다. 실행 중인 학습에는 코드 수정이 즉시 적용되지 않으며 재시작이 필요하다. 구역 목표의 중복을 막는 수정으로, 물리적 신체 겹침 자체를 보장하는 제약은 아니다.

### 숨김 문 형식 비교

`object_layout: hidden_doors_then_regions_v1`은 닫힌 문 관측 proxy4개 + 담당 구역4개로 기존 Stage1/Joint Push의 문/상자 토큰 형식을 근사한다. 물리 문 actor는 없고 기존 먼 일반 물체의 관측 슬롯만 문으로 바뀐다. 손잡이 위치·문 flag·bbox를 복원하고 region graph 연결을8~11로 이동한다. 기본 모드와 기존 실행은 유지한다. 동일 epoch0 가중치의64환경 랜덤 출발 평가에서 기존/비교 모두64/64 넘어졌으며 평균29.52→38.97step이었다. 형식 복원만으로 해결되지는 않았다.

### 독립 랜덤 문 좌표 수정

사용자 정정으로 최신 관측 비교 모드는 `hidden_doors_scattered_regions_v2`다. 문은 물리에 존재하지 않고 좌표/닫힌 문 특징만 관측에 들어간다. 일반 물체 위치를 재사용하지 않는다. 과제 영역 XY 경계+4m 범위에서 독립 XY/yaw를 배치하고 문간2m·사람1.5m·과제 상자 외접원+0.75m 간격을 둔다. reset에서 선택 환경만 재배치하고 이후 고정한다. v1 비교 결과는 이전 좌표 재사용 방식의 결과다.

### 2026-10-21 과제 전용 학습

`task_regions_only_v1`은4명의담당구역4개만관측하고문·비과제상자를제거한다. 그룹이공유하는실제과제상자는1~4개이며group-size별형상을위한16개과제asset후보는멀리주차/선택한다. ordinary asset은생성하지않는다. graph는사람0~3/구역4~7/목표8~11,관측1180차원이다. 새profile은Start초기화와랜덤출발외접원여유0.3~0.8m이며Stage1에서새학습한다. 기존모드/체크포인트는보존한다.
