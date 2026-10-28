# 현재 서버 실행 환경

RTX PRO 6000 Blackwell에서 stage1 학습은 **2048환경**, 4명 stage2는 **1024환경**을 사용한다. 기본 명령은 [config.md](config.md), 코드 위치는 [structure.md](structure.md)를 따른다.

## 환경과 데이터

- 작업 레포: `/home/user/jhh/Projects/TokenHSI_stage1_stage2_base`
- 원본 코드·결과 보관 레포: `/home/cvlab/Desktop/CVPR2027/stage1_message` (보존)
- conda: `tokenhsi_sm120`, Python3.8. `runtime_env.sh`로 활성화한다.
- 데이터: `tokenhsi/data/dataset_*/` 심링크를 통해 읽는다. 현재 링크 대상은 `/home/user/jhh/Projects/TokenHSI/tokenhsi/data/`이며 원본은 읽기 전용으로 취급한다.
- SMPL: `body_models/smpl/`에 별도 설치한다. Isaac Gym은 활성 conda 환경의 설치본을 사용한다.

```bash
cd /home/user/jhh/Projects/TokenHSI_stage1_stage2_base
nvidia-smi
TOKENHSI_GPU=0 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
python -c "import isaacgym, torch; print(torch.cuda.get_device_name(0))"'
```

짧은 학습도 각 실험의 환경 수(stage1 2048·stage2 1024)를 유지하고 `CHECK=1`(기본 `MAX_ITERATIONS=1`) 또는 `MAX_ITERATIONS`를 지정한다. 확인 결과는 `output_etc/check/`로 제한한다. 본학습 전 `CHECK`, `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 값을 확인한다. 요청된 학습 시각화만 4환경이며 평가·VNC는 기본 1환경이다.

## Viewer와 문제 확인

`train_vnc.sh <컨피그>`는 4환경 학습, 전용 `*_vnc.sh`는 1환경 테스트를 Xvfb·x11vnc·noVNC로 표시한다. 학습 시각화 명령과 컨피그별 인자는 [config](config.md#학습)를 따른다. 사용자 로컬 `$HOME/opt/vnc/usr/bin/{Xvfb,x11vnc}`와 `$HOME/opt/noVNC`, 활성 conda의 websockify를 자동 탐색한다. `TOKENHSI_GPU`는 CUDA와 viewer에 함께 전달한다. GPU 내부의 `cuda:0`은 선택한 물리 GPU의 로컬 번호다. 화면 번호를 고를 때 기존 VNC 포트도 건너뛴다. 서비스 로그는 기본 `output_etc/check/gui/`에 저장하며 `GUI_LOG_DIR`로 변경한다.

| 증상 | 확인 |
| --- | --- |
| NVML 불일치 | `nvidia-smi`의 드라이버/라이브러리 오류; headless와 GUI 검증을 구분 |
| CUDA 메모리 부족 | 선택 GPU의 기존 프로세스·실제 actor/object 수·batch; 실험별 지정 환경 수 유지 |
| 모션 파일 누락 | 데이터 심링크·YAML 상대 경로 |
| `libpython3.8.so.1.0` 로드 실패 | `runtime_env.sh`와 conda 환경 |
| VNC 실패·검은 화면 | `VNC_DIR`, `NOVNC_DIR`, 포트 점유·viewer 생성 로그 |

머신 이관은 [PORTING_GUIDE.md](reference/PORTING_GUIDE.md)를 따른다. 학습 결과는 `output/<실험>/<run>/`, 평가 지표는 `output_etc/check/<실험>/eval/metrics/`에 저장한다.

## 추가 실행 옵션

`EXP`는 `stage1_v2_fixed`, `stage1_v2`, `stage1` 중 하나다. 기본 학습·로컬 추론·VNC 명령은 config에만 나열한다.

```bash
EXP=stage1_v2_fixed
# 확인 학습: MAX_ITERATIONS=1, 2048환경 유지
TOKENHSI_GPU=0 CHECK=1 bash "tokenhsi/scripts/multi_agent/${EXP}_train.sh" 2 2048 4
# 저장 상태에서 별도 본학습 output으로 재개
TOKENHSI_GPU=0 RESUME_CHECKPOINT="$CKPT" OUTPUT_PATH="output/${EXP}_resume" bash "tokenhsi/scripts/multi_agent/${EXP}_train.sh" 2 2048 4
```

- 확인 학습은 `output_etc/check/<실험>/train/`, 평가·viewer는 `output_etc/check/<실험>/eval/`가 기본이다. 분리할 때는 `OUTPUT_PATH=output_etc/check/<검사명>`을 지정한다.
- `CHECK=1`, 양수 `MAX_ITERATIONS`, 평가 모드 또는 경로에 `check`/`smoke`/`validation` 표시가 있으면 확인 출력으로 분류한다. 구 `output/foo_check`는 `output_etc/check/foo_check`로 전환하며 확인 루트 밖 지정은 거부한다. 직접 `run.py --check`도 지원한다. 리다이렉션 로그·스크린샷·분석 결과도 확인 폴더 아래에 둔다.
- 과제·초기 skill·혼합 확률 값과 예시는 [config의 추론 옵션](config.md#추론-옵션)을 따른다. `EPISODE_LENGTH`, `SEED`도 지정할 수 있다. 평가 인원 M≥1·물체 O≥M이며 ON_TOP 받침 부족 시 다른 적격 과제로 재추첨한다.
- 학습 시각화는 6080, 테스트 시각화는 6081을 포워딩한다. 예: `ssh -L 6080:localhost:6080 -L 6081:localhost:6081 <서버>`. 브라우저 주소는 `http://localhost:<포트>/vnc.html?autoconnect=1&resize=remote`다. `PORT`, `XVFB`, `VNC_DIR`, `NOVNC_DIR`, `WEBSOCKIFY`로 재정의한다.
- 학습 시각화의 기본 카메라 `TASK_CAMERA=overview`는 첫 4환경을 한 화면에 담는다. `TASK_CAMERA=stack`은 첫 환경만 따른다. 학습 시각화는 horizon 32 기준 전체 rollout을 minibatch로 사용한다(2명·4환경=256, 4명·4환경=512). `MINIBATCH_SIZE`로 재정의할 수 있다. 기본 본학습 YAML의 minibatch는 유지한다. `CHECK=1`을 붙이면 짧게 실행하고 종료한다.
- 별도로 옮긴 fixed checkpoint도 내부 `relation_experiment_config`의 env/sim을 복원한다. 원래 run의 `relation_config.yaml`이 있으면 우선 사용한다.

## checkpoint와 캐시

- stage1과 v2 checkpoint는 혼용하지 않는다. v2 → fixed 평가·resume는 고정 선반 물리로 전환하는 후속 실험으로 명시적으로 지원한다.
- test/VNC는 checkpoint run의 `relation_config.yaml`을 기본 사용한다. 저장 config가 없거나 다른 환경을 지정할 때는 `STAGE1_ENV_CONFIG`, `STAGE1_V2_ENV_CONFIG`, `STAGE1_V2_FIXED_ENV_CONFIG`로 명시한다. 명령 인자·평가 옵션은 해당 항목을 덮어쓴다.
- v2/fixed의 운영 RSI 캐시는 `output/rsi_cache_stage1_v2`, stage1은 `output/rsi_cache`다. 파일명은 설정·모션·관련 코드 hash이며 `.summary.json`은 캐시 요약이다. 캐시 입력/코드가 바뀌면 새 파일이 생길 수 있다.
- 모델·환경 상세는 [정책 참고](reference/ma_clean_scene_gta.md), 지표는 [진단 참고](reference/relation_diagnostics.md)를 따른다. 검증 기록은 [changelog](../changelog.md)에 연결한다.

## CPU 검증

```bash
TOKENHSI_GPU=0 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi:tokenhsi/tests pytest -q tokenhsi/tests'
```

## Stage2 전이와 RSI 준비

기본 전이는 `output/stage1/Stage1V2Fixed_00060000.pth`다. 새 stage1 파일은 `STAGE1_CHECKPOINT`, 재개는 `RESUME_CHECKPOINT`, 저장 환경 재정의는 `STAGE2_ENV_CONFIG`로 지정한다. Stage2는4명·1024환경·8장면 물체다. 기본 실행 명령은 [config](config.md), 크기별 공동 RSI 생성·검증과 새 모델 계약은 [Stage2 참고](reference/stage2.md)에 있다. 검증용 전이 PTH는 학습한 모델이 아니다. 공동 RSI·GTA·리워드는 Y 끝면에서15cm를 사용한다. 기본 캐시는 `joint_carry/stage2_size_rsi/inset15_single`이며 누락·생성 조건 변경 시15cm로 자동 생성한다.

첫 학습은 PPO 업데이트 전에 크기별 RSI 물리 검사를 수행할 수 있다. `[size RSI] screening`과 공동 생성기의 `Completed x / y`는 초기 캐시 준비 로그다. 1024환경의 첫 준비는 오래 걸릴 수 있으며, 생성된 캐시는 이후 실행에서 재사용한다.

### Stage1 가중치로 미학습 Stage2 viewer 보기

Stage1 fixed60000을 전이하고 짧은 추론으로 검증용 Stage2 checkpoint를 만든다. 추가 협력 모듈은 항등 초기화 상태이며 optimizer 학습을 실행하지 않는다. 생성 파일은 추론 확인용으로 사용하고, 실제 Stage2 학습은 `STAGE1_CHECKPOINT`로 Stage1 원본을 지정해 새로 시작한다.

```bash
export TOKENHSI_GPU=0
. tokenhsi/scripts/multi_agent/runtime_env.sh
STAGE2_PRESETS=all_solo,one_pair_at,one_pair_ontop,two_pairs_at,two_pairs_ontop STAGE2_SKILL=loco STAGE2_STEPS=16 \
python tokenhsi/tests/check_stage2.py --task HumanoidMAStage2 \
  --cfg_env tokenhsi/data/cfg/multi_agent/stage2.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage2.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --transfer_checkpoint output/stage1/Stage1V2Fixed_00060000.pth \
  --num_envs 2 --num_agents 4 --num_objects 8 --headless --no_video \
  --output_path output_etc/check/stage2_from_stage1_60000/transfer
```

4명·1환경 viewer에서 무작위 과제를 반복한다. 6081 포트를 포워딩한 뒤 `http://localhost:6081/vnc.html`로 접속한다. `F`는 카메라 추적, `K`는 보조선, `Esc`는 종료다.

```bash
TOKENHSI_GPU=0 PORT=6081 TASK_GRAPH=random_scenario EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 EPISODE_LENGTH=600 \
GUI_LOG_DIR=output_etc/check/stage2_from_stage1_60000/services \
OUTPUT_PATH=output_etc/check/stage2_from_stage1_60000/viewer \
bash tokenhsi/scripts/multi_agent/stage2_vnc.sh \
  output_etc/check/stage2_from_stage1_60000/transfer/nn/Stage2Transfer.pth 4 1 8 1000000
```

## Stage1 PUSH fine-tuning

`stage1_v2_fixed_push`는 기본 `output/stage1/Stage1V2Fixed_00060000.pth`를 가져와 새 optimizer/epoch으로 시작한다. 실행 명령은 [config](config.md), 구현 메모는 [Stage1 PUSH](reference/stage1_push.md)에 있다.

- `STAGE1_CHECKPOINT`: 최초 전이와 동결 teacher의 원본 fixed checkpoint.
- `OLD_TASK_FRACTION=0.5`: 전체 transition 중 기존 과제 비율. 기존 과제 내부 비율은 유지한다.
- `TEACHER_KL_COEF=0.001`: 기존 과제 행동 보존 KL 계수. `0`이면 teacher 생성·전용 checkpoint 로드·KL 관측 처리/forward를 생략한다. 원본 fixed에서 학생 모델을 처음 전이할 때 필요한 checkpoint 로드는 유지한다.
- `STAGE1_PUSH_ENV_CONFIG`: 최초 실행용 환경 YAML. `stage1Push.enabled`와 `goal_mode: face_normal`, 거리·RSI 옵션은 이 YAML에 둔다.
- `RESUME_CHECKPOINT`: PUSH checkpoint의 모델·optimizer·epoch을 재개한다. 저장 환경 설정을 복원하고 비율/KL override만 허용하며, 원본 fixed checkpoint는 최초 전이 인자로 사용한다.
- test/VNC 기본은 `TASK_GRAPH=push`, `EVAL_SKILLS=loco`다. 기존 과제별 평가는 `TASK_GRAPH=sit`, `climb`, `carry_at`, `carry_ontop`을 사용한다. PUSH RSI 평가는 `EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0`을 사용한다.
- 1환경 평가는 특정 과제를 지정한다. `TASK_GRAPH=random_scenario`의 혼합 평가는 최소 2환경이 필요하다.
- 평가에는 teacher 실행이 필요하지 않다. 학습 재개 시 원본 teacher checkpoint가 필요하다.

테스트/VNC는 `TASK_GRAPH`와 양수 `EVAL_SKILL_PROBS`에 필요한 시작 모션만 로드한다. PUSH 보행 시작은 4개, PUSH RSI 시작은 선별 bank의 7개 clip을 사용하며, 두 시작 방식을 혼합하면 두 묶음을 로드한다. 기존 task도 지정한 시작 skill만 읽고 Stage2 테스트는 backward/sideways AMP expert를 로드하지 않는다. 명령은 동일하다.

### PUSH 360도 접근 전이

`stage1_v2_fixed_push_360`의 최초 학습에는 `PUSH_CHECKPOINT=<기존 PUSH.pth>`를 지정한다. 모델·정책/AMP/가치 정규화는 그대로 가져오고 optimizer·epoch은 새로 시작한다. 동결 teacher는 원본 fixed60000을 사용한다. 물성·AMP·RSI bank가 다른 checkpoint는 거부한다. `PUSH_CHECKPOINT`를 생략하면 원본 fixed에서 시작한다.

새 360 checkpoint의 재개에는 `PUSH_CHECKPOINT`를 해제하고 `RESUME_CHECKPOINT=<360.pth>`를 지정한다. optimizer·epoch·해당 epoch의 커리큘럼 단계를 복원한다. 기존 PUSH checkpoint를 360 wrapper의 `RESUME_CHECKPOINT`나 test 인자로 직접 넣으면 거부한다. 전이는 `PUSH_CHECKPOINT`로 실행한다.

현재 360 YAML은 `contact_midpoint_only: true`로 접촉 상태 점수의 개별 손-면 거리 항을 제외한다. 이전 보상 checkpoint에서 새 보상으로 시작할 때는 `PUSH_CHECKPOINT`를 사용한다. `RESUME_CHECKPOINT`는 저장된 보상 설정을 그대로 복원한다.

학습 중 범위는 0/1000/3000 epoch에 확대되며 평가는 항상 최종 범위다. `EPISODE_LENGTH`는 테스트의 기존/PUSH 제한 시간을 함께 덮어쓴다. 기본 제한 시간과 배치·보상은 [PUSH 360 구현 메모](reference/stage1_push_360.md)를 따른다. 명령은 [config](config.md)에 있다.

### PUSH 추론 에피소드 길이

PUSH/PUSH 360의 test/VNC 인자는 `<checkpoint.pth> [agents] [envs] [objects] [repeats] [episode_steps]`다. 마지막 값은 에피소드당 최대 제어 step 수이며, 낙상 시 더 일찍 종료된다. 지정 우선순위는 마지막 인자 → `EPISODE_LENGTH` 환경변수 → 저장된 설정이다. 생략 시 PUSH 360은 PUSH1800step·기존 과제600step을 유지하며, 명시한 값은 두 과제군에 함께 적용한다.

```bash
TOKENHSI_GPU=0 PORT=6081 TASK_GRAPH=push EVAL_SKILLS=push bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_vnc.sh "$CKPT" 2 1 4 10 900
```

위 예시는 RSI 시작·10회 반복·에피소드당 최대900step(시뮬레이션 약30초)이다. `CKPT`는 [config](config.md#서버-vnc-추론)의 PUSH 360 checkpoint 경로로 지정한다.

### PUSH 360 추론 배치

test/VNC 명령의 `bash` 앞에 `PUSH_SPAWN_MODE`를 지정한다.

| 값 | 보행 시작 배치 |
| --- | --- |
| `random` | 기본값. 박스 주변 360도에서 생성 |
| `straight` | 미는 면 뒤쪽의 에이전트→박스→goal 직선에서 생성 |

두 모드 모두 root→박스 중심 거리는 1.65~6m에서 랜덤이다. `EVAL_SKILLS=loco`에 적용하며 `push` RSI 자세·목표 거리·학습 설정은 유지한다. 원래 PUSH 설정에는 `straight`를 사용할 수 없다. 직접 실행 인자는 `--push_spawn_mode random|straight`다.

`CKPT`는 [config](config.md#서버-vnc-추론)처럼 사용할 PUSH 360 checkpoint 경로로 지정한다.

```bash
PUSH_SPAWN_MODE=straight TOKENHSI_GPU=0 PORT=6081 TASK_GRAPH=push EVAL_SKILLS=loco bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_vnc.sh "$CKPT" 2 1 4 10
```

### PUSH 추론 RSI 자세

PUSH/PUSH 360 test/VNC에서 `PUSH_RSI_MODE`로 접촉 RSI 자세를 선택한다. 직접 실행 인자는 `--push_rsi_mode corrected|raw`다.

| 값 | 접촉 RSI 초기화 |
| --- | --- |
| `corrected` | 기존 PUSH/PUSH360의 기본값. 손 높이0.6175m 목표의 팔 보정을 사용 |
| `raw` | 새 PUSH360 raw의 기본값. 원본 관절·손 높이·참조 속도와 초기 AMP 이력을 사용 |

두 모드는 같은 선별7클립·11개 후보와 상자 진행률을 사용한다. 상자·goal·보상·정책은 유지하며, 원본 모드에서도 상자에 맞춘 전신 회전과 수평 이동, 손 중점의 면 밖6cm 정렬은 적용한다. reset 이후 행동은 동일한 학습 정책이 생성한다. checkpoint나 RSI cache를 새로 만들 필요는 없다.

환경변수 override는 추론 전용이다. 학습 RSI 기본값은 환경 YAML의 `stage1Push.rsi.mode`를 따르며, 기존 설정은 보정·새 raw 설정은 원본이다. `PUSH_SPAWN_MODE`는 보행 시작 배치 옵션이므로 접촉 RSI 자세 선택에는 영향을 주지 않는다.

```bash
PUSH_RSI_MODE=raw PUSH_SPAWN_MODE=straight TOKENHSI_GPU=0 PORT=6081 \
TASK_GRAPH=push EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 \
bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_vnc.sh \
  "output/stage1_v2_fixed_push_360_midpoint/Stage1V2FixedPush360_19-19-54-06/nn/Stage1V2FixedPush360.pth" \
  2 1 4 100 300
```

접속: `http://localhost:6081/vnc.html?autoconnect=1&resize=remote`. 기존 보정과 비교하려면 `PUSH_RSI_MODE=corrected`로 실행한다. 확률 변수의 정확한 이름은 `EVAL_SKILL_PROBS`다.

### PUSH 원본 RSI·가변 정육면체

`stage1_v2_fixed_push_360_raw`는 원본 RSI·고정 면 XY 접촉 보상·정육면체 한 변1.2~1.7m·질량12~20kg을 사용한다. 크기는 환경 생성 시 상자마다 균등 샘플링하고, 질량은 한 변 길이에 선형 대응한다(1.2m→12kg, 1.45m→16kg, 1.7m→20kg). 크기·질량은 actor 수명 동안 고정한다. 기존/PUSH50:50·teacher KL·AMP·목표/보행 커리큘럼은 유지한다.

- 기본 학습은 `output/stage1/Stage1V2Fixed_00078000.pth`에서 모델/정규화를 가져오고 optimizer/epoch을 새로 시작한다. `STAGE1_CHECKPOINT`로 원본 fixed와 teacher를 변경할 수 있다. `PUSH_CHECKPOINT` 전이는 같은 원본 teacher·본체 마찰을 사용하는 PUSH 모델에만 허용한다.
- 재개·test/VNC는 새 raw checkpoint를 사용한다. 기존 고정 크기 checkpoint를 새 wrapper의 resume/test에 넣으면 거부한다. 기존 wrapper와 checkpoint는 기존 물성·기본 보정을 유지한다.
- 이전 raw checkpoint의 재개/평가는 저장된 물성·보상을 복원한다. 본체0.9·밑판0.3 물성은 원본 fixed에서 새로 시작한다. 같은 원본 teacher·본체 마찰에서 보상만 변경할 때는 `PUSH_CHECKPOINT`로 전이한다.
- `stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform`의 기본 새 학습은 fixed78000에서 시작한다. 기존 cuboid 모델·정규화에서 새 optimizer/epoch으로 시작하려면 `PUSH_CHECKPOINT=<cuboid.pth>`를 지정한다. 이후 재개는 자체 checkpoint를 `RESUME_CHECKPOINT`로 지정한다. 기존 cuboid checkpoint의 일반 resume로 균등 거리 설정을 적용할 수는 없다. [설계](reference/stage1_push_rsi_uniform.md), 실행은 [config](config.md)를 따른다.
- RSI 원본7clip·11시각을 재사용한다. 관절·높이·속도는 바꾸지 않고, 초기 사람-상자 침투가 예상될 때만 전신을 미는 면 바깥쪽으로 최소 이동해5mm 여유를 확보한다. 기존 ref NPY·공용 RSI cache는 재생성하지 않는다.
- 실행 명령은 [config](config.md), 크기별 검증 범위와 구현 내용은 [구현 메모](reference/stage1_push_raw.md)를 따른다.

### PUSH 15kg 비교 시각화

질량 비교 실험의 별도 실행기로 기존 raw checkpoint를 15kg에서 볼 수 있다. 원본 RSI·크기·마찰·보상은 checkpoint 설정을 사용하고, 상자 생성 시 질량만 15kg으로 적용한다. 아래 명령은 비교에 사용한11000epoch 저장본·1환경·100회·최대300step이다.

```bash
PUSH_RSI_MODE=raw PUSH_SPAWN_MODE=straight \
EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 \
TOKENHSI_GPU=0 PORT=6081 \
bash output_etc/check/push_mass_sweep_20261021/view_15kg.sh \
  "output/stage1_v2_fixed_push_360_raw/Stage1V2FixedPush360Raw_20-20-27-10/nn/Stage1V2FixedPush360Raw_00011000.pth" \
  2 1 4 100 300
```

접속: `http://localhost:6081/vnc.html?autoconnect=1&resize=remote`. 시작은 기본 `EVAL_SKILLS=push`, `PUSH_RSI_MODE=raw`이며 보행 시작은 `EVAL_SKILLS=loco`로 선택한다. 결과는 `output_etc/check/push_mass_sweep_20261021/viewer_15kg/`에 저장한다.

### 6명·5과제 attention 진단

정육면체 epoch9000과 cuboid epoch8000을 각각 실제1환경·6명(SIT/CLIMB/carryAT/carryONTOP/PUSH-loco/PUSH-RSI)으로 실행한다. 학습 코드는 유지하고 평가 전용 장면에서 actor/critic의4개 레이어·2개 head, H6/O7/G6 전체19×19 행렬을 저장한다. [결과·방법](../output_etc/check/push_attention_six_20261022/report.md).

```bash
TOKENHSI_GPU=0 bash output_etc/check/push_attention_six_20261022/run.sh
```

현재 결과 뷰어는 서버8767 포트를 포워딩한 뒤 `http://localhost:8767/`에서 연다. 레이어·head·시간·actor/critic·run을 선택하며 QK, softmax(QK), bias, QK+bias, softmax(QK+bias)를 비교한다. 실행기는 현재 머신의 `DISPLAY=:2`를 기본으로 쓰며 환경변수로 변경할 수 있다.

## Stage1 jointpush 체크포인트와 환경

- `stage1_jointpush_train.sh`는 기본 스크래치 학습이며 PTH를 자동 전이하지 않는다. `RESUME_CHECKPOINT`는 동일 jointpush 계약으로 저장한 모델만 받는다.
- `stage1_jointpush_density3_cone90_finetune_train.sh`는 `STAGE1_CHECKPOINT`(기본 `pth/Stage1V2Fixed_00078000.pth`)에서 가중치/정규화를 가져오고 optimizer/epoch은 새로 시작한다. `RESUME_CHECKPOINT` 지정 시 전이를 건너뛰고 저장된 파인튜닝을 재개한다. 전이 증거는 실행 폴더의 `push_transfer_report.json`이다.
- 평가 기본 PTH는 `pth/Stage1PushFinetune_00010500.pth`다. 모델과 입력/AMP 정규화 통계만 읽고, 환경은 현재 `stage1_jointpush.yaml`을 사용한다. 따라서 저장된 옛 목표·과제 비율을 복원하지 않는다.
- 모든 과제의 평가는 loco100%다. 학습의 과제별 RSI 비율은 유지하고, 과제 비율·물성·loco 배치/목표 샘플링은 평가에서도 동일하게 사용한다.
- 현재 서버의 검증 환경은 `TOKENHSI_CONDA_ENV=tokenhsi_sm120`이다. 다른 머신에서는 설치된 환경 이름을 지정한다.
- `SIM_DEVICE=cpu`는 평가 스크립트에서 CPU 물리/정책을 선택한다. `EPISODE_LENGTH`는 짧은 평가용 horizon override다.
- PUSH 모션은 `tokenhsi/data/jointpush_motions/`의 로컬 심링크로 연결한다. 실제 NPY는 Git에 포함하지 않으며 RSI bank가 모션·캐릭터·mesh 해시와 상자 물성을 검사한다. 기존 과제 RSI 캐시는 최초 실행 시 필요한 크기별 후보를 생성할 수 있다.
- 현재 서버의 PUSH 두 데이터 폴더는 `../TokenHSI_stage1_push/tokenhsi/data/dataset_bones_push{,180}`에서 읽기 전용으로 연결한다. `push_loco`는 이 checkout의 `dataset_amass_loco` 연결을 사용하며, VNC 명령은 현재 서버 환경 `tokenhsi_sm120`과 제공된 14000 checkpoint를 명시한다.

평가의 loco100%·실제 물성·AMP 연결 검사는 아래처럼 실행한다. `--task_graph`를 `sit`, `climb`, `carry_at`, `carry_ontop`, `random_scenario`로 바꾸면 해당 과제도 확인할 수 있다. 새 모델의 actor/critic/discriminator gradient 검사도 포함하지만 PPO 학습 반복을 대신하지는 않는다.

```bash
TOKENHSI_CONDA_ENV=tokenhsi_jhh PYTHONDONTWRITEBYTECODE=1 bash -c '
. tokenhsi/scripts/multi_agent/runtime_env.sh
python tokenhsi/tests/check_stage1_jointpush.py \
  --task HumanoidMAStage1JointPush \
  --cfg_env tokenhsi/data/cfg/multi_agent/stage1_jointpush.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage1_jointpush.yaml \
  --motion_file tokenhsi/data/dataset_stage1_jointpush.yaml \
  --checkpoint pth/Stage1PushFinetune_00010500.pth \
  --task_graph push --num_envs 1 --num_agents 2 --num_objects 4 \
  --sim_device cpu --pipeline cpu --rl_device cpu \
  --test --eval --headless --no_video --seed 42 \
  --output_path output_etc/check/stage1_jointpush/eval_loco_push
'
```


78k·밀도3·cone90 파인튜닝의 현재 지속 실행은 사용자 서비스 `tokenhsi-stage1-jointpush-d3-cone90-ft78k-20261027.service`다. 상태는 `systemctl --user status tokenhsi-stage1-jointpush-d3-cone90-ft78k-20261027.service`, 로그는 `output/stage1_jointpush_density3_cone90_finetune/train.log`에서 확인한다. 이 실행만 중단하려면 `systemctl --user stop tokenhsi-stage1-jointpush-d3-cone90-ft78k-20261027.service`를 사용하며 MPS 서버를 종료하지 않는다.
