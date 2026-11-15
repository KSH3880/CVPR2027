# Stage1 PUSH no-bias 학습

GitHub `KSH3880/CVPR2027`의 `sct_nobias` 브랜치, 커밋
`c385e3159bb7abcfa84efc160308ee5d2b6e566a`에서 가져온 독립 복사본이다.
출처와 파일별 SHA256은 `IMPORT_MANIFEST.json`에 있다. 원본 코드·설정은 그대로 유지했다.

`stage1_push_no_bias`는 **초기 checkpoint 없이 처음부터 학습**한다.
기본값은 **2명·2048환경·4물체·900step·seed42**이며 actor/critic의 task attention bias를
제거한다. GTA·task embedding·Action MLP·보상·샘플링은 유지한다.
이전 `stage2_before_total_at1mm_nobias`의 사람 BEFORE attention bias 제거와는 다른 실험이다.

## 로컬 준비와 CPU 확인

이 서버에는 PUSH 모션을 실제 복사했고, SIT/CARRY/LOCO/CLIMB 모션은
`/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data`에 읽기 전용으로 연결했다.
`ksh`로 연결되는 실행 의존성은 없다. 코드·설정·모션 목록 YAML은 Git으로 관리하며,
모션 배열·학습 결과·checkpoint는 제외한다.

```bash
cd /home/hwanhee/juan/CVPR2027/stage1_push_no_bias
conda activate tokenhsi
CUDA_VISIBLE_DEVICES='' python local.py check
mkdir -p output_etc/check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest -q \
  tokenhsi/tests/test_stage1_push_no_bias.py tokenhsi/tests/test_stage1_push.py
```

다른 checkout에서는 PyTorch/Isaac Gym 및 `requirements.txt` 환경을 준비하고,
공통 데이터와 PUSH 모션의 실제 경로를 지정한다. 원본은 읽기만 하고 기존 파일은 덮어쓰지 않는다.

```bash
python local.py setup --data-root /path/to/TokenHSI/tokenhsi/data \
  --jointpush-motions /path/to/jointpush_motions
CUDA_VISIBLE_DEVICES='' python local.py check
```

## 학습

아래 GPU 번호 `6`은 예시이며 사용할 GPU로 변경한다. 이번 이식 작업에서는 GPU·학습·평가·viewer를
실행하지 않았다. 실행기는 기본 `tokenhsi` conda 환경을 자동 활성화한다.

```bash
cd /home/hwanhee/juan/CVPR2027/stage1_push_no_bias
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_no_bias_train.sh 2 2048 4
```

본학습 출력은 `output/stage1_push_no_bias/`다. 신규 학습 전 셸에 남은
`CHECK`, `MAX_ITERATIONS`, `OUTPUT_PATH`, `ENV_CONFIG`, `RESUME_CHECKPOINT` 값을 확인한다.
기존 bias 모델 checkpoint와 재개·추론·평가를 혼용하지 않는다.

## 학습 후 평가

학습된 no-bias checkpoint와 같은 run의 `relation_config.yaml`을 함께 유지한다.
각 평가는 512환경×seed0/1/2×900step이다. 일반 평가는 저장된 학습 인원·물체 수를 사용하고,
단독 평가는 AT/ON_TOP/SIT/CLIMB/PUSH를 각각 1명으로 평가한다. 서로 다른 출력 폴더를 사용한다.

```bash
CKPT="/path/to/trained_no_bias_checkpoint.pth"
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_no_bias_eval.sh \
  "$CKPT" --output output_etc/check/no_bias_general
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_no_bias_single_eval.sh \
  "$CKPT" --output output_etc/check/no_bias_single
```

통계는 각 결과 폴더의 `summary.json`에 저장된다.
상세 구현은 [reference](markdowns/reference/stage1_push_no_bias.md), 추가 옵션은
[USAGE](markdowns/USAGE.md), 로컬 검증은 [CHANGELOG](CHANGELOG.md)를 따른다.
가져온 원본 이력은 [changelog](changelog.md)에 보존했다.
