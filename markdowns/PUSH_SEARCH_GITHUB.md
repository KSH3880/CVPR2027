# 커플드 PUSH 선별: 브랜치 공유와 로컬 실행

이 문서는 사용자가 직접 실행할 명령이다. 작성 과정에서 브랜치 생성·stage·commit·push는 수행하지 않았다. 대상은 ENABLING이 아니라 **실제 2인 공동 PUSH, 상자3.20×1.60×1.45m, seed20부터 선별**이다.

현재 서버 소스에는 GPU7·외부 정책 경로·기존 초기 snapshot 의존성이 있다. 아래는 Git 공유와 별도 파일 전송, **로컬 checkout에서만 적용할 경로 변경**까지 포함한다. 로컬 장비의 실제 시뮬레이션 실행은 아직 검증하지 않았다. 기존 서버 탐색/다른 실험은 변경하지 않는다.

## 1. 서버에서 새 브랜치 만들고 코드만 푸시

다음 명령은 이 저장소 루트에서 사용자가 직접 실행한다. 기존 다른 작업의 변경은 stage하지 않는다. 현재 기본 커밋은 `fce8f77`, 원격은 `https://github.com/KSH3880/CVPR2027.git`이다.

```bash
cd /home/hwanhee/koo_cvpr/stage1_2_base_teacher
git status --short
git diff --cached --stat
# 이미 stage된 다른 변경이 있으면 여기서 먼저 확인한다.

git switch -c coupled-push-search-local-20261113

git add -- \
  markdowns/PUSH_SEARCH_GITHUB.md \
  tokenhsi/scripts/analysis/push_fixed_layout.py \
  tokenhsi/scripts/analysis/push_fixed_rollout.py \
  tokenhsi/scripts/analysis/push_fixed_search.sh \
  tokenhsi/scripts/analysis/search_push_wait.py \
  tokenhsi/scripts/analysis/prefilter_push_initial.py \
  tokenhsi/scripts/analysis/natural_push_queue.py \
  tokenhsi/scripts/analysis/natural_push_stage_select.py \
  tokenhsi/scripts/analysis/push_comparison_rollout.py \
  tokenhsi/scripts/analysis/push_comparison_rollout.sh \
  tokenhsi/scripts/analysis/push_comparison_select.py \
  tokenhsi/scripts/analysis/search_genuine_push_cases.py \
  tokenhsi/scripts/analysis/push_path_metrics.py \
  tokenhsi/scripts/analysis/push_left_metrics.py \
  tokenhsi/scripts/analysis/build_push_review_html.py \
  tokenhsi/scripts/analysis/build_push_comparison_html.py \
  tokenhsi/scripts/analysis/push_comparison_viewer.html \
  tokenhsi/scripts/analysis/enabling_reference

git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "Add coupled push scene search and local setup guide"
git push -u origin coupled-push-search-local-20261113
```

`enabling_reference`라는 이름이지만 공통 FSM/Stage1 어댑터·TokenHSI 정책 소스 스냅샷이 들어 있어 PUSH에서도 사용한다. 원본 출처/해시는 그 안의 `manifest.json`에 있다.

`git add .`는 사용하지 않는다. 현재 수정된 공용 runtime·환경·ENABLING 코드·요약문서 전체, `output*`, PTH/PT/NPZ/ZIP, 모션 데이터 및 서버 절대경로 심링크는 위 공유에 포함하지 않는다. 특히 서버의 수정된 runtime은GPU6/7만허용하므로 위 명령은 기존 커밋의 범용 runtime을 유지한다.

## 2. 필요한 PTH 세 개

이 checkout의 `pth/`에도 아래 세 파일을 **실제 파일로 복사 완료**했다. 원본과 SHA256 일치를 확인했으며 복사 기록은 `output_etc/check/push_pth_copy_20261113/manifest.json`에 있다. 폴더에서 세 파일을 한 번에 내려받아 아래 로컬 배치 경로로 나눠 넣어도 된다. PTH는 Git 추적에서 제외된다.

| 용도 | 파일 | 서버 원본 |
| --- | --- | --- |
| Ours | `Stage2Distill900V4_00002000.pth` | `/home/hwanhee/ksh/stage1_2_base_teacher/output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/nn/Stage2Distill900V4_00002000.pth` |
| FSM의 Stage1 정책 | `Stage1JointPushDensity3Cone90Finetune_00026000.pth` | `/home/hwanhee/ksh/stage1_2_base_teacher/pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth` |
| TokenHSI PUSH 정책 | `TokenHSIPushKD_00040500.pth` | `/home/hwanhee/ksh/tokenhsi_push/checkpoints/TokenHSIPushKD_00040500.pth` |

FSM에는 별도 FSM 학습 가중치가 없다. Stage1 PTH와 `StateFreeze` 제어기를 함께 쓴다. Ours PTH는 초기 환경/공통 player 생성에도 사용하므로 TokenHSI부터 검사하더라도 필요하다.

로컬 Linux/NVIDIA CUDA·Isaac Gym·Python3.8 계열 `tokenhsi` conda 환경과 이 저장소 의존성을 먼저 준비한다. PTH만으로 실행되지는 않으며 기존 모션 데이터도 필요하다.

## 3. 로컬 clone과 가중치/초기 snapshot 전송

아래부터는 **로컬 PC**에서 실행한다. `PRO1` 값은 접속 가능한 SSH 주소/별칭으로 바꾼다. 서버 브랜치를 위에서 실제 푸시한 다음 clone한다.

```bash
git clone --branch coupled-push-search-local-20261113 \
  https://github.com/KSH3880/CVPR2027.git coupled-push-local
cd coupled-push-local
export PRO1='hwanhee@Pro1접속주소'

mkdir -p checkpoints/push_compare/ours
mkdir -p tokenhsi/scripts/analysis/enabling_reference/common/pth
mkdir -p tokenhsi/scripts/analysis/enabling_reference/tokenhsi_push/checkpoints

scp "$PRO1:/home/hwanhee/ksh/stage1_2_base_teacher/output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/nn/Stage2Distill900V4_00002000.pth" \
  checkpoints/push_compare/ours/
scp "$PRO1:/home/hwanhee/ksh/stage1_2_base_teacher/pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth" \
  tokenhsi/scripts/analysis/enabling_reference/common/pth/
scp "$PRO1:/home/hwanhee/ksh/tokenhsi_push/checkpoints/TokenHSIPushKD_00040500.pth" \
  tokenhsi/scripts/analysis/enabling_reference/tokenhsi_push/checkpoints/

mkdir -p output_etc/check/push_natural_prefilter_20261112/source/seed34
scp "$PRO1:/home/hwanhee/koo_cvpr/stage1_2_base_teacher/output_etc/check/push_natural_prefilter_20261112/source/seed34/initial.pt" \
  output_etc/check/push_natural_prefilter_20261112/source/seed34/
```

마지막 `initial.pt`는 고정 배치 기준 seed34/env367의 약4.1MB snapshot이다. 탐색 시작 seed는 별도로20을 지정한다. `push_fixed_layout.py`가 이 파일에서 전체 물리 상자/중심 목표 기준을 읽고 공동 상자 크기를 평가 최소값으로 바꾼다. 이 파일도 Git에 넣지 않고 별도로 복사한다.

Ours 환경 설정은 PTH의 `relation_experiment_config` 또는 같은 디렉터리의 `relation_config.yaml`에서 읽는다. 원 실험 YAML이 필요하면 `Stage2Distill900V4_11-21-37-16/relation_config.yaml`을 `checkpoints/push_compare/ours/relation_config.yaml`로 복사한다. 다른 실험의 YAML을 사용하지 않는다.

## 4. 모션 데이터와 TokenHSI 소스

`tokenhsi/data/dataset_stage1_jointpush.yaml`에 열거된 모션/물체 데이터가 필요하다. 로컬에 이미 있으면 해당 실데이터를 다음4개 경로로 연결한다. 없으면 아래 명령으로 서버 원본을 읽기 복사한다. 데이터가 크므로 기존 로컬 데이터가 있으면 재전송하지 않는다.

```bash
# 로컬에 데이터가 없을 때만 실행.
for dataset in dataset_amass_loco dataset_amass_climb dataset_sit dataset_carry; do
  rsync -aL --info=progress2 \
    "$PRO1:/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data/$dataset/" \
    "tokenhsi/data/$dataset/"
done

# 참조 TokenHSI 네트워크2개를 로컬 package 위치로 복사.
mkdir -p tokenhsi/learning/transformer
cp tokenhsi/scripts/analysis/enabling_reference/tokenhsi_push/tokenhsi/learning/transformer/amp_network_builder_transformer.py \
  tokenhsi/learning/transformer/
cp tokenhsi/scripts/analysis/enabling_reference/tokenhsi_push/tokenhsi/learning/transformer/composer.py \
  tokenhsi/learning/transformer/
```

데이터 링크가 끊기지 않았는지 확인한다. Isaac Gym SDK는 별도 설치한다. TokenHSI 관측 함수·push_kd 설정·어댑터는 동봉 참조 트리에 있다. 다른 TokenHSI 모델로 교체하지 않는다.

## 5. 로컬 checkout의 경로/GPU 수정

**Pro1에서 실행 중인 checkout에는 이 블록을 실행하지 않는다.** clone한 로컬에서 한 번 실행한다. 기존 스크립트를 그대로 실행하면 GPU7과 `/home/hwanhee/ksh`를 사용하므로 아래 수정이 필요하다.

```bash
python - <<'PY'
from pathlib import Path
root = Path.cwd().resolve()
assert root != Path('/home/hwanhee/koo_cvpr/stage1_2_base_teacher')
assert (root/'markdowns/PUSH_SEARCH_GITHUB.md').exists()
ref = root/'tokenhsi/scripts/analysis/enabling_reference'
ours = root/'checkpoints/push_compare/ours/Stage2Distill900V4_00002000.pth'
replace = {
    '/home/hwanhee/ksh/stage1_2_base_teacher/output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/nn/Stage2Distill900V4_00002000.pth': str(ours),
    '/home/hwanhee/ksh/stage1_2_base_teacher': str(ref/'common'),
    '/home/hwanhee/ksh/tokenhsi_push': str(ref/'tokenhsi_push'),
}
for filename in ['push_fixed_search.sh', 'push_comparison_rollout.sh',
                 'push_comparison_rollout.py', 'search_push_wait.py']:
    path = root/'tokenhsi/scripts/analysis'/filename
    text = path.read_text()
    for old, new in replace.items():
        text = text.replace(old, new)
    path.write_text(text)
path = root/'tokenhsi/scripts/analysis/natural_push_queue.py'
text = path.read_text()
assert "env=dict(os.environ,TOKENHSI_GPU='7')" in text
text = text.replace("env=dict(os.environ,TOKENHSI_GPU='7')", 'env=dict(os.environ)')
text = text.replace('gpu=7,max_rollouts=', "gpu=os.environ.get('TOKENHSI_GPU','0'),max_rollouts=")
path.write_text(text)
print('Local paths configured:', root)
PY
```

추론 어댑터는 원래 코드로 사용하며 PTH 내용/판정식/상자 크기를 바꾸지 않는다. 위 선택 파일만 commit하면 공용 runtime은 기존 커밋의 임의 GPU 지원 버전이다. 로컬에서 `Use TOKENHSI_GPU=6 or 7`이 나오면 서버 전용 runtime 변경까지 stage했는지 확인한다.

## 6. 로컬에서 seed20부터 탐색

먼저 GPU 점유를 확인한다. 로컬이1GPU라면 `TOKENHSI_GPU=0`, 4개 프로세스를 감당할 VRAM이 있으면 `NATURAL_PUSH_WORKERS=4`로 실행한다. 작은 GPU는 동시 worker 수를 줄일 수 있다. 각 worker의1024환경은 동일 조건을 유지한다.

```bash
nvidia-smi
conda activate tokenhsi

# 먼저 정책 action 없이1024환경 초기 생성 확인.
TOKENHSI_GPU=0 PUSH_FIXED_PROFILE=eval_min \
NATURAL_PUSH_ROOT="$PWD/output_etc/check/push_fixed_evalmin_local_seed20" \
bash tokenhsi/scripts/analysis/push_fixed_search.sh initial 20

# 같은 출력 루트에서 탐색. 앞서 만든seed20초기도 재사용.
TOKENHSI_GPU=0 PUSH_FIXED_PROFILE=eval_min NATURAL_PUSH_WORKERS=4 \
NATURAL_PUSH_ROOT="$PWD/output_etc/check/push_fixed_evalmin_local_seed20" \
bash tokenhsi/scripts/analysis/push_fixed_search.sh queue 20
```

상자 전체 크기3.20×1.60×1.45m·밀도3kg/m³·마찰0.4·질량22.272kg·목표 거리5m다. A=index1/B=index0·면 거리0.9/9m·63heading/native RSI·1024환경/900step이다. 초기 확인→TokenHSI→FSM→Ours 순서로, 앞 단계 탈락 시 후속을 생략한다. TokenHSI는 실제 선행 밀기와 이후 좌우2m 이탈을 분리 검사한다. ENABLING의 새 유사 경로 필터는 PUSH에 적용하지 않는다.

`queue 20`은 기존 폴더가 있으면 재사용/이어 탐색한다. seed20부터 별도 시험하려면 새 출력 이름을 사용한다. 상태는 출력 직하 `status.json`, step은 `rollouts/seed*/<method>/progress.json`에 있다. 상태 파일과 실제 PID 생존을 함께 확인한다.

## 7. 저장 기록으로 비교 HTML 만들기

```bash
mkdir -p output_etc/check/push_comparison_html_20261112
scp "$PRO1:/home/hwanhee/koo_cvpr/stage1_2_base_teacher/output_etc/check/push_comparison_html_20261112/three.min.js" \
  output_etc/check/push_comparison_html_20261112/

TOKENHSI_GPU=0 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh && python tokenhsi/scripts/analysis/build_push_review_html.py --root output_etc/check/push_fixed_evalmin_local_seed20 --output output_etc/check/push_comparison_html_20261112/local_push_review.html --watch'
```

HTML 자동 생성은 FSM까지 완료된 기록을 기다린다. TokenHSI 탈락이 계속되면 HTML도 생성 대기한다. 특정 완료 기록만 보려면 `--watch` 대신 `--seed <seed> --env <env>`를 지정한다. 실제 미실행 방법은 미완료로 표시한다.
