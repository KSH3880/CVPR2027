# Stage2 BEFORE total AT1mm — no-bias 학습

`/home/hwanhee/ksh/stage_1_2_base_1031`의 `enable_nobias` 브랜치,
커밋 `16f8b2c664ee94df482f72e87a961086950bce74`에서 가져온 독립 복사본이다.
원본 코드·설정은 그대로 유지했다. 출처와 파일 SHA256은 `IMPORT_MANIFEST.json`에 있다.

사람 attention의 BEFORE bias projection을 제거하고 BEFORE embedding/후행 잔차 MLP
fusion은 유지한다. 기본값은 **Stage1 26k 신규 전이, 4명, 1024환경, 8토큰, 900step**이다.
기존 bias Stage2 checkpoint를 resume용으로 사용하지 않는다.

## 로컬 준비와 CPU 확인

이 서버에는 초기 가중치와 PUSH 모션을 실제 복사했고, SIT/CARRY/LOCO/CLIMB 모션은
`/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data`에 읽기 전용으로 연결했다.
`ksh`로 연결되는 실행 의존성은 없다. 코드는 Git으로 관리하고 가중치·데이터·출력은 제외한다.

```bash
cd /home/hwanhee/juan/CVPR2027/stage2_before_total_at1mm_nobias
conda activate tokenhsi
CUDA_VISIBLE_DEVICES='' python local.py check
CUDA_VISIBLE_DEVICES='' python -m pytest -q tokenhsi/tests/test_before_total_at1mm_nobias.py tokenhsi/tests/test_before_relation_fusion.py
```

다른 checkout에서는 PyTorch/Isaac Gym 및 `requirements.txt` 환경과 외부 데이터를 준비한 후
실제 데이터 경로로 다음을 실행한다. 원본 파일은 읽기만 하며 기존 로컬 파일은 덮어쓰지 않는다.

```bash
python local.py setup --data-root /path/to/TokenHSI/tokenhsi/data \
  --checkpoint /path/to/Stage1JointPushDensity3Cone90Finetune_00026000.pth \
  --jointpush-motions /path/to/jointpush_motions
CUDA_VISIBLE_DEVICES='' python local.py check
```

## 학습 시작 명령

아래 명령은 사용자가 GPU를 선택한 뒤 직접 실행한다. 이번 이식 작업에서는 GPU를 사용하거나
학습·평가·viewer를 실행하지 않았다. 실행기는 `tokenhsi` conda 환경을 자동으로 활성화한다.

```bash
cd /home/hwanhee/juan/CVPR2027/stage2_before_total_at1mm_nobias
TOKENHSI_GPU=<사용할_GPU> bash tokenhsi/scripts/multi_agent/stage2_before_total_at1mm_nobias_train.sh 4 1024 8
```

기본 출력은 `output/stage2_before_total_at1mm_nobias/`이다. 신규 학습 전에는 셸에 남은
`CHECK`, `MAX_ITERATIONS`, `RESUME_CHECKPOINT`, `STAGE1_CHECKPOINT`, `ENV_CONFIG`,
`OUTPUT_PATH` 값을 확인한다. 추가 실행 옵션과 추론은 [config](markdowns/config.md),
구현 설명은 [reference](markdowns/reference/stage2_before_push.md)를 따른다.
로컬 변경·검증은 [CHANGELOG](CHANGELOG.md), 가져온 원본 이력은 [changelog](changelog.md)에 있다.
