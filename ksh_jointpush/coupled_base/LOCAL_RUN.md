# 구역별 goal 공동 PUSH 로컬 실행

Git 브랜치는 `juan`, 실험 이름은 `stage2_coupled_push_region_goals`다. 코드·환경/학습 YAML·모션 목록·실행기는 Git에 포함한다. 큰 모션 데이터·PTH·학습 결과·머신별 심링크는 포함하지 않는다. 현재 본학습 베이스는 PUSH가 포함된 Stage1 26k이며 원래 Stage1 78k 파일은 이 학습에 필요 없다. 아래 자료 준비 예시는 초기21k 배포 경로이며, 현재 학습은 `STAGE1_CHECKPOINT`로 로컬26k를 명시한다. YAML 기본21k와 구분한다.

## 코드와 외부 자료

```bash
git clone --branch juan https://github.com/KSH3880/CVPR2027.git
cd CVPR2027
```

기존 checkout은 `juan` 브랜치의 최신 변경을 받는다. Linux·NVIDIA GPU 및 해당 GPU에 맞는 PyTorch/CUDA·Isaac Gym 환경이 필요하다. 이 서버 검증 환경은 Python3.8/PyTorch2.4 CUDA12.8이며, 다른 환경 이름은 `TOKENHSI_CONDA_ENV`로 지정한다. Isaac Gym은 torch보다 먼저 import해야 한다. 추가 Python 패키지는 `requirements.txt`를 따른다.

이미 로컬에 데이터가 있으면 아래 준비 명령에 그 데이터의 `tokenhsi/data` 경로를 지정한다. 필요한 폴더는 `dataset_amass_loco`, `dataset_amass_climb`, `dataset_sit`, `dataset_carry`, `jointpush_motions`다. 모션 목록의 file/obj_file 경로와 PUSH bank의21개 모션·캐릭터/mesh 해시를 준비 명령이 검사한다.

서버에서 받을 경우 아래 `Pro1`을 본인 SSH 접속 별칭 또는 `사용자@서버`로 바꾼다. 코드/config를 덮어쓰지 않도록 외부 자료 폴더에 받는다. `scp -r`은 서버의 데이터 심링크를 따라 실제 파일을 복사한다.

```bash
mkdir -p local_assets/coupled_data
for folder in dataset_amass_loco dataset_amass_climb dataset_sit dataset_carry jointpush_motions; do
  scp -r "Pro1:/home/hwanhee/juan/CVPR2027/ksh_jointpush/coupled_base/tokenhsi/data/$folder" local_assets/coupled_data/
done
scp Pro1:/home/hwanhee/juan/CVPR2027/ksh_jointpush/coupled_base/checkpoints/Stage1PushFinetune_00021000.pth local_assets/
```

## 준비와 실행

현재 injesus1010 서버에서는 `TOKENHSI_CONDA_ENV=tokenhsi118`을 지정한다. 구역별 goal train의 루트 wrapper는 `scripts/coupled_base_run.py`를 통해 PyTorch2.0/CUDA11.8·RTX5090 호환을 적용한다. 모션235항목과 PUSH21clip은 기존 `runs/assets/ksh_jointpush/data` 자료로 연결·검증했다. 2026-10-30 Stage1 PUSH21000 수신 후 GPU0·4명×1024환경 본학습을 시작했고 실제2epoch 업데이트를 검증했다. 실행 명령/설정/현재검증은 CVPR2027 루트의 `runs/jointpush/launch_coupled_region_goals_20261030_200442/`, 현재 학습 metadata는 `runs/jointpush/ksh_jointpush_active_training.json`이다.

현재 실행은 사용자 승인 후 AT/CONTACT 로깅을 추가하고 초기2epoch 저장본에서 모델·optimizer를 복원한 것이다. 실행/검증은 `runs/jointpush/resume_coupled_reward_split_20261030_210932/`, 현재 PID/출력은 위 활성 metadata를 따른다. TensorBoard6017의 `coupled_region_goals_21000`에서 `reward_terms/at_*`, `reward_terms/contact_*`를 확인한다. 값은 각 관계의 활성 edge 평균이며 AT에는 독립 AT도 포함된다. 혼합 과제에서 두 항목 합은 전체 보상이 아니다. 이전 로그는 `coupled_region_goals_21000_before_split`로 보존한다.

프로젝트 루트 `CVPR2027`에서 실행한다. 준비 명령은 데이터 경로를 연결하고21k를 로컬 checkpoints로 복사한 뒤 실행 준비 기록을 만든다. 자료 원본은 수정하지 않는다.

```bash
conda activate tokenhsi
python ksh_jointpush/scripts/coupled_base_prepare.py \
  --data-root "$PWD/local_assets/coupled_data" \
  --checkpoint "$PWD/local_assets/Stage1PushFinetune_00021000.pth"

# stage2_coupled_push_region_goals
TOKENHSI_CONDA_ENV=tokenhsi118 TOKENHSI_GPU=0 \
  STAGE1_CHECKPOINT="$PWD/ksh_jointpush/coupled_base/checkpoints/Stage1JointPushDensity3Cone90Finetune_00026000.pth" \
  bash ksh_jointpush/scripts/coupled_region_goals_train.sh 4 1024 8
```

본학습 결과는 `ksh_jointpush/coupled_base/output/stage2_coupled_push_region_goals/`에 저장한다. Stage1 21k에서 새 Stage2를 시작하며, 구공통goal Stage2 저장본을 resume하지 않는다. 확인 학습이 필요하면 위 학습 명령 앞에 `CHECK=1`을 추가한다. 환경 수는4명·1024환경을 유지하고 확인 출력은 `output_etc/check/`로 분리한다.

새 학습의 Stage2 checkpoint로 로컬 viewer를 실행한다.

```bash
CKPT='/absolute/path/to/Stage2CoupledPushRegionGoals_00000500.pth'
TOKENHSI_CONDA_ENV=tokenhsi TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=all_coupled \
  bash ksh_jointpush/scripts/coupled_region_goals_test.sh "$CKPT" 4 1 8 10
```

각자의 상자 구역과 고정goal을1:1대응한다. 현재 native 설정에서 공동 PUSH의 agent–상자 중심은2~10m, 상자–goal은2~5m이며 단독 PUSH는 기존2~10m다. 관측·AT보상/성공·marker·평가는 구역별goal을 사용한다. 물리 상자는 그룹에서 공유한다. 실행 중인 학습의 저장 설정은 자동으로 바뀌지 않는다. [설정/설계](markdowns/reference/stage2_coupled_push.md), [모든 실행 명령](markdowns/config.md).

현재 공동 PUSH는 사람의 초기 배치·참조 자세를 먼저 정한 뒤, 그 위치를 고정한 채 조건에 맞는 상자와 goal을 찾는 기존 순서를 사용한다. native YAML에서 `placement_order: box_goal_people_v1`을 제거했다. 이 필드가 있는 과거 저장 설정에는 상자 우선 구현을 남겨 두며, 새 학습과 해당 checkpoint 계약은 구분한다. 현재는 Stage1 PUSH26000에서 새 optimizer로 Stage2를 처음부터 실행한다.

이전 zero-shot viewer는 공동 두 거리2~5m·상자 우선 배치의 관찰 기록이며 현재 설정과 다르다. 관련 검증 자료는 `output_etc/check/zeroshot_box_first_20261030_231952/`에 보존한다.

## 1명 단독+3명 공동 PUSH 보기

현재5000iter를1+3으로고정한관찰용명령이다. 학습은변경하지않으며기본카메라는초록agent의단독상자를따라간다. 이미실행중인viewer와중복실행하지않는다.

```bash
bash /home/injesus1010/repos/CVPR2027/ksh_jointpush/coupled_base/output_etc/check/view_1plus3_5000_20261031_124256/run.sh
```

1=단독, 2=3명공동, 3=전체, N=다음장면, F=카메라추적, L=이미지저장, Esc=종료다. 저장프레임은 `/home/injesus1010/repos/CVPR2027/ksh_jointpush/coupled_base/output_etc/check/view_1plus3_5000_20261031_124256/live/imgs` 아래에생성된다.
