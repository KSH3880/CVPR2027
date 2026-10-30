# 구역별 goal 공동 PUSH 로컬 실행

Git 브랜치는 `juan`, 실험 이름은 `stage2_coupled_push_region_goals`다. 코드·환경/학습 YAML·모션 목록·실행기는 Git에 포함한다. 큰 모션 데이터·PTH·학습 결과·머신별 심링크는 포함하지 않는다. 베이스는 PUSH가 포함된 Stage1 21k이며 원래 Stage1 78k 파일은 이 학습에 필요 없다.

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

프로젝트 루트 `CVPR2027`에서 실행한다. 준비 명령은 데이터 경로를 연결하고21k를 로컬 checkpoints로 복사한 뒤 실행 준비 기록을 만든다. 자료 원본은 수정하지 않는다.

```bash
conda activate tokenhsi
python ksh_jointpush/scripts/coupled_base_prepare.py \
  --data-root "$PWD/local_assets/coupled_data" \
  --checkpoint "$PWD/local_assets/Stage1PushFinetune_00021000.pth"

# stage2_coupled_push_region_goals
TOKENHSI_CONDA_ENV=tokenhsi TOKENHSI_GPU=0 \
  bash ksh_jointpush/scripts/coupled_region_goals_train.sh 4 1024 8
```

본학습 결과는 `ksh_jointpush/coupled_base/output/stage2_coupled_push_region_goals/`에 저장한다. Stage1 21k에서 새 Stage2를 시작하며, 구공통goal Stage2 저장본을 resume하지 않는다. 확인 학습이 필요하면 위 학습 명령 앞에 `CHECK=1`을 추가한다. 환경 수는4명·1024환경을 유지하고 확인 출력은 `output_etc/check/`로 분리한다.

새 학습의 Stage2 checkpoint로 로컬 viewer를 실행한다.

```bash
CKPT='/absolute/path/to/Stage2CoupledPushRegionGoals_00000500.pth'
TOKENHSI_CONDA_ENV=tokenhsi TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=all_coupled \
  bash ksh_jointpush/scripts/coupled_region_goals_test.sh "$CKPT" 4 1 8 10
```

각자의 상자 구역과 고정goal을1:1대응하고 밀기 방향·거리2~10m를 유지한다. 관측·AT보상/성공·marker·평가는 구역별goal을 사용한다. 물리 상자는 그룹에서 공유한다. [설정/설계](markdowns/reference/stage2_coupled_push.md), [모든 실행 명령](markdowns/config.md).
