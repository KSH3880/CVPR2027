# 현재 서버 실행 환경

RTX PRO 6000 Blackwell에서 stage1 학습은 **2048환경**, 4명 stage2는 **1024환경**을 사용한다. 기본 명령은 [config.md](config.md), 코드 위치는 [structure.md](structure.md)를 따른다.

## 환경과 데이터

- 작업 레포: `/home/hwanhee/ksh/stage1_stage2`
- 원본 코드·결과 보관 레포: `/home/cvlab/Desktop/CVPR2027/stage1_message` (보존)
- conda: `tokenhsi`, Python3.8. `runtime_env.sh`로 활성화한다.
- 데이터: `tokenhsi/data/dataset_*/` 심링크를 통해 읽는다. 이 checkout의 실제 링크 대상은 `/home/cvlab/Desktop/CVPR2027/TokenHSI` 아래이며, 다른 서버의 `/home/hwanhee/CVPR2027/TokenHSI`를 포함한 공유 원본도 읽기 전용으로 취급한다.
- SMPL: `body_models/smpl/`에 별도 설치한다. Isaac Gym은 활성 conda 환경의 설치본을 사용한다.

```bash
cd /home/hwanhee/ksh/stage1_stage2
nvidia-smi
TOKENHSI_GPU=0 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
python -c "import isaacgym, torch; print(torch.cuda.get_device_name(0))"'
```

짧은 학습도 각 실험의 환경 수(stage1 2048·stage2 1024)를 유지하고 `CHECK=1`(기본 `MAX_ITERATIONS=1`) 또는 `MAX_ITERATIONS`를 지정한다. 확인 결과는 `output_etc/check/`로 제한한다. 본학습 전 `CHECK`, `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 값을 확인한다. 평가·VNC 환경 수는 별도다.

## Viewer와 문제 확인

전용 `*_vnc.sh`는 Xvfb·x11vnc·noVNC를 준비한다. `TOKENHSI_GPU`는 CUDA와 viewer에 함께 전달한다. GPU 내부의 `cuda:0`은 선택한 물리 GPU의 로컬 번호다. 화면 번호를 고를 때 기존 VNC 포트도 건너뛴다. 서비스 로그는 기본 `output_etc/check/gui/`에 저장하며 `GUI_LOG_DIR`로 변경한다.

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
- VNC는 로컬에서 `ssh -L 6080:localhost:6080 <서버>` 후 `http://localhost:6080/vnc.html`로 접속한다. `PORT`, `VNC_DIR`, `NOVNC_DIR`, `WEBSOCKIFY`로 재정의한다.

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

기본 전이는 `stage1/Stage1V2Fixed_00030500.pth`다. 새 stage1 파일은 `STAGE1_CHECKPOINT`, 재개는 `RESUME_CHECKPOINT`, 저장 환경 재정의는 `STAGE2_ENV_CONFIG`로 지정한다. Stage2는4명·1024환경·8장면 물체다. 기본 실행 명령은 [config](config.md), 크기별 공동 RSI 생성·검증과 새 모델 계약은 [Stage2 참고](reference/stage2.md)에 있다. 검증용 전이 PTH는 학습한 모델이 아니다. 공동 RSI·GTA·리워드는 Y 끝면에서15cm를 사용한다. 기본 캐시는 `joint_carry/stage2_size_rsi/inset15_single`이며 누락·생성 조건 변경 시15cm로 자동 생성한다.

### Stage1 가중치로 미학습 Stage2 viewer 보기

Stage1 fixed45000을 전이하고 짧은 추론으로 검증용 Stage2 checkpoint를 만든다. 추가 협력 모듈은 항등 초기화 상태이며 optimizer 학습을 실행하지 않는다. 생성 파일은 추론 확인용으로 사용하고, 실제 Stage2 학습은 `STAGE1_CHECKPOINT`로 Stage1 원본을 지정해 새로 시작한다.

```bash
export TOKENHSI_GPU=5
. tokenhsi/scripts/multi_agent/runtime_env.sh
STAGE2_PRESETS=all_solo,one_pair_at,one_pair_ontop,two_pairs_at,two_pairs_ontop STAGE2_SKILL=loco STAGE2_STEPS=16 \
python tokenhsi/tests/check_stage2.py --task HumanoidMAStage2 \
  --cfg_env tokenhsi/data/cfg/multi_agent/stage2.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage2.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --transfer_checkpoint stage1/Stage1V2Fixed_00045000.pth \
  --num_envs 2 --num_agents 4 --num_objects 8 --headless --no_video \
  --output_path output_etc/check/stage2_from_stage1_45000/transfer
```

4명·1환경 viewer에서 무작위 과제를 반복한다. 6081 포트를 포워딩한 뒤 `http://localhost:6081/vnc.html`로 접속한다. `F`는 카메라 추적, `K`는 보조선, `Esc`는 종료다.

```bash
TOKENHSI_GPU=5 PORT=6081 TASK_GRAPH=random_scenario EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 EPISODE_LENGTH=600 \
GUI_LOG_DIR=output_etc/check/stage2_from_stage1_45000/services \
OUTPUT_PATH=output_etc/check/stage2_from_stage1_45000/viewer \
bash tokenhsi/scripts/multi_agent/stage2_vnc.sh \
  output_etc/check/stage2_from_stage1_45000/transfer/nn/Stage2Transfer.pth 4 1 8 1000000
```

## 직육면체 PUSH 기울기 분석

학습 설정을 변경하지 않는 GPU6 전용 물리 검사다. 양손 중점을 지정점으로 두고 XYZ와 질량을 독립 비교한다. [조건·결과·한계](reference/push_rect_stability.md)를 먼저 확인한다.

```bash
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/analyze_push_rect_stability.sh
```

`OUTPUT_PATH=output_etc/check/<새이름>`으로 출력 경로를 지정할 수 있다. 기존 경로에는 덮어쓰지 않는다. 접촉체/AMP 선택자세 기반 humanoid의 결과를 분리하며, 학습2048환경과 별도의 평가환경 수를 사용한다. `REPORT.md`·`summary.json`·그래프·CSV·trace를 저장한다. 이 분석은 학습 정책 성공률이나 1~5m 장거리에서의 무기울기를 보장하지 않는다.


## PUSH 크기 제로샷

5과제 Stage1 PUSH의 기존 모델을 보행 시작으로 평가한다. `PUSH_EVAL_SIZE_X/Y/Z`는 실제 상자 메시/충돌체와 관측 bbox를 모두 바꾼다. 학습 YAML·checkpoint 계약·보상·20kg 질량·상자/바닥 마찰을 유지한다. 밀도는 새 부피에 맞춰 계산하며 크기에 따라 관성은 변한다. 두 PUSH상자만 바꾸고 입력의 나머지 실제 상자들은 유지한다. `size_zero_shot.json`에 실제 sampled크기·지정범위·기존 물성을 기록한다.

각 축은 `최소,최대`(0.05m격자) 또는 단일 고정값을 받는다. 미지정 축은 기존 분포를 따른다. 크기는 환경 생성 시 상자별·축별 독립 샘플링하고 같은 실행에서는 고정한다. 반복마다 크기를 바꾸지 않으며 `SEED=43`처럼 seed를 바꾸어 재실행하면 다른 샘플을 볼 수 있다. 학습·다른 과제·PUSH RSI와 함께 사용하면 오류를 낸다. 원래RSI는 새크기에 검증되지 않았으므로 보행 시작만 허용한다. 위치/목표의 중간 전환 옵션도 함께 사용하지 않는다.

Y2m 모델용 범위는 현재 X1.2~1.5/Y2.0/Z1.3m가 가운데 오도록 다음과 같이 확장한다.

| 축 | 평가 범위(m) | 중심(m) |
| --- | --- | --- |
| X | 0.9~1.8 | 1.35 |
| Y | 1.7~2.3 | 2.0 |
| Z | 1.0~1.6 | 1.3 |

사본 루트에서 실행한다. Y1.8m 모델은 해당 test/VNC와 checkpoint로 바꾸고 `PUSH_EVAL_SIZE_Y=1.5,2.1`로 지정하면 학습 Y1.8m 중심이 유지된다.

```bash
CKPT=output/stage1_depth200_20kg_friction04_push100_from1000/Stage1FiveTaskDepth20020kgFriction04_21-20-10-04/nn/Stage1FiveTaskDepth20020kgFriction04_00017500.pth

# 서버 VNC: 6082 포워딩 후 http://localhost:6082/vnc.html
TOKENHSI_GPU=6 PORT=6082 TASK_GRAPH=push_at EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 \
TASK_CAMERA=agent PUSH_GOAL_SWITCH_ANGLES= PUSH_BOX_SWITCH_MODES= \
PUSH_EVAL_SIZE_X=0.9,1.8 PUSH_EVAL_SIZE_Y=1.7,2.3 PUSH_EVAL_SIZE_Z=1.0,1.6 \
OUTPUT_PATH=output_etc/check/push_size_zero_shot_depth200_view \
bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth200_20kg_friction04_vnc.sh "$CKPT" 2 1 4 10

# 로컬 화면
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=push_at EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 \
TASK_CAMERA=agent PUSH_GOAL_SWITCH_ANGLES= PUSH_BOX_SWITCH_MODES= \
PUSH_EVAL_SIZE_X=0.9,1.8 PUSH_EVAL_SIZE_Y=1.7,2.3 PUSH_EVAL_SIZE_Z=1.0,1.6 \
OUTPUT_PATH=output_etc/check/push_size_zero_shot_local \
bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth200_20kg_friction04_test.sh "$CKPT" 2 1 4 10

# 화면 없는 평가:64환경·128개 독립 크기
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=push_at EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 \
PUSH_GOAL_SWITCH_ANGLES= PUSH_BOX_SWITCH_MODES= \
PUSH_EVAL_SIZE_X=0.9,1.8 PUSH_EVAL_SIZE_Y=1.7,2.3 PUSH_EVAL_SIZE_Z=1.0,1.6 \
OUTPUT_PATH=output_etc/check/push_size_zero_shot_eval \
bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth200_20kg_friction04_test.sh "$CKPT" 2 64 4 10
```

## PUSH 중간 상자 이동 제로샷

5과제 Stage1 `TASK_GRAPH=push_at`의 기존 모델로 보행 시작한다. 목표 좌표/높이를 고정하고 기본60step(2초) 뒤 PUSH 상자만 한 번 이동한다. `PUSH_BOX_SWITCH_MODES=across_goal,behind_agent`는 반복마다 두 조건을 교대하며 단일값도 가능하다. 목표 전환 옵션 `PUSH_GOAL_SWITCH_ANGLES`와 동시 지정하면 오류를 낸다.

- `across_goal`: 이동 직전 상자→목표 방향을 따라 **목표를 넘어 반대쪽**으로 보낸다. 목표에서 상자 중심까지 기본2m다. 상자가 이미 목표 중심이면 episode 초기 방향을 사용한다.
- `behind_agent`: 현재 pelvis의 전방(+X)을 XY로 투영한 방향의 뒤쪽에 놓는다. 사람 root에서 상자 중심까지 기본2m다.

`PUSH_BOX_SWITCH_STEP`과 `PUSH_BOX_SWITCH_DISTANCE`로 시점·기준거리를 바꾼다. 사람의 현재 콜라이더와 모든 상자를 검사하고 겹치면 같은 방향으로0.5m씩 더 멀리 놓는다. 안전한 위치를 찾지 못하면 오류를 내며 시뮬레이터를 변경하지 않는다. 실제 거리는 JSON에 남긴다. 상자 높이·회전·크기·질량·마찰은 유지하고 순간이동 시 상자의 선속도/각속도는0으로 만든다. 사람 상태·AMP·episode 시간·목표는 reset하지 않는다. 상자 위치에 맞춰 접촉 면과 현재 관측/관계 이력을 재계산한다. 이동은 물리적 외력이 아니며 teleport 길이를 실제 밀기 거리로 세지 않는다.

결과는 `metrics/push_box_switch_*.json`의 `box_switch`에 이동 전후 상자·고정 목표·사람 위치/방향·실제 배치거리·이동 후 성공률을 기록한다. 이전 상태 성공은 제거하며 이동 후 성공률은 실제 이동한 장면 기준이다. 기존학습과 일반 고정배치 평가 결과는 별도로 유지한다.

사본 루트에서 아래 중 원하는 모델 명령을 실행한다. VNC포트를 포워딩하고 해당 포트의 `/vnc.html`로 접속한다.

```bash
# Y2m/20kg/마찰0.4: 2초 후 상자 이동 두 조건을 교대로10회
TOKENHSI_GPU=6 PORT=6082 TASK_GRAPH=push_at EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 \
TASK_CAMERA=agent PUSH_GOAL_SWITCH_ANGLES= PUSH_BOX_SWITCH_MODES=across_goal,behind_agent \
PUSH_BOX_SWITCH_STEP=60 PUSH_BOX_SWITCH_DISTANCE=2 \
OUTPUT_PATH=output_etc/check/push_box_switch_depth200_view \
bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth200_20kg_friction04_vnc.sh \
output/stage1_depth200_20kg_friction04_push100_from1000/Stage1FiveTaskDepth20020kgFriction04_21-20-10-04/nn/Stage1FiveTaskDepth20020kgFriction04_00017500.pth 2 1 4 10

# Y1.8m/20kg/마찰0.6
TOKENHSI_GPU=6 PORT=6083 TASK_GRAPH=push_at EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 \
TASK_CAMERA=agent PUSH_GOAL_SWITCH_ANGLES= PUSH_BOX_SWITCH_MODES=across_goal,behind_agent \
PUSH_BOX_SWITCH_STEP=60 PUSH_BOX_SWITCH_DISTANCE=2 \
OUTPUT_PATH=output_etc/check/push_box_switch_depth180_view \
bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth180_20kg_vnc.sh \
output/stage1_depth180_20kg_push100_from1000/Stage1FiveTaskDepth18020kg_21-20-09-05/nn/Stage1FiveTaskDepth18020kg_00017500.pth 2 1 4 10

# 로컬 화면: 위 Y2m checkpoint 경로를 CKPT에 지정
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=push_at EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 \
TASK_CAMERA=agent PUSH_GOAL_SWITCH_ANGLES= PUSH_BOX_SWITCH_MODES=across_goal,behind_agent \
PUSH_BOX_SWITCH_STEP=60 OUTPUT_PATH=output_etc/check/push_box_switch_local \
bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth200_20kg_friction04_test.sh "$CKPT" 2 1 4 10

# 화면 없는 평가
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=push_at EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 \
PUSH_GOAL_SWITCH_ANGLES= PUSH_BOX_SWITCH_MODES=across_goal,behind_agent \
PUSH_BOX_SWITCH_STEP=60 OUTPUT_PATH=output_etc/check/push_box_switch_eval \
bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth200_20kg_friction04_test.sh "$CKPT" 2 64 4 10
```

## PUSH 중간 목표 전환 제로샷

5과제 Stage1의 PUSH100% checkpoint를 그대로 사용한다. 보행 시작이며 재학습은 없다. `PUSH_GOAL_SWITCH_ANGLES=90,180`은 반복별로90°/180°를 번갈아 적용하고, 단일값90 또는180은 해당 각도만 반복한다. 양수90°는 위에서 보았을 때 반시계 방향이다. **에피소드 시작의 상자→목표 방향**을 기준으로 회전하고, 전환 시점의 상자 중심에서 새 방향으로2m 떨어진 곳에 목표를 놓는다. 180°는 처음 밀던 방향의 반대쪽이다. 사람/상자의 자세·속도·AMP 이력·episode 시간은 재설정하지 않는다. 목표 높이도 유지한다.

기본 `PUSH_GOAL_SWITCH_STEP=150`(control dt1/30초에서5초), `PUSH_GOAL_SWITCH_DISTANCE=2`m다. 각 episode에서 한 번 전환한다. 전환 전 도달 여부와 관계없이 지정step에 변경하며, 미리 종료된 장면은 전환하지 않는다. 일반 성공률은 이전 목표의 성공 이력을 지운 뒤 전체 시도를 분모로 집계하고, JSON `goal_switch`의 성공률은 실제 전환된 장면만 분모로 사용한다. 전환 전 종료된 장면은 후자 분모에서 제외한다. 결과는 별도 `metrics/push_goal_switch_*.json`에 전환 좌표·각도와 함께 기록하며, 일반 고정목표 평가와 구분한다.

아래 명령은 이 사본 루트에서 실행한다. 두 모델 중 하나를 선택하고, VNC6082를 포워딩해 `http://localhost:6082/vnc.html`로 접속한다. 각20초 episode의5초 지점에 변경하며 남은15초 동안 새 목표를 따른다. `TASK_CAMERA=agent`를 사용한다.

```bash
# Y2m/20kg/마찰0.4: 서버 VNC, 90도/180도 교대로10회
TOKENHSI_GPU=6 PORT=6082 TASK_GRAPH=push_at EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 \
TASK_CAMERA=agent PUSH_GOAL_SWITCH_ANGLES=90,180 \
OUTPUT_PATH=output_etc/check/push_goal_switch_depth200_view \
bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth200_20kg_friction04_vnc.sh \
output/stage1_depth200_20kg_friction04_push100_from1000/Stage1FiveTaskDepth20020kgFriction04_21-20-10-04/nn/Stage1FiveTaskDepth20020kgFriction04_00017500.pth 2 1 4 10

# Y1.8m/20kg/마찰0.6: 서버 VNC
TOKENHSI_GPU=6 PORT=6083 TASK_GRAPH=push_at EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 \
TASK_CAMERA=agent PUSH_GOAL_SWITCH_ANGLES=90,180 \
OUTPUT_PATH=output_etc/check/push_goal_switch_depth180_view \
bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth180_20kg_vnc.sh \
output/stage1_depth180_20kg_push100_from1000/Stage1FiveTaskDepth18020kg_21-20-09-05/nn/Stage1FiveTaskDepth18020kg_00017500.pth 2 1 4 10
```

로컬 화면과 headless 평가에서는 같은 인자로 해당 `_test.sh`를 사용한다.

```bash
# 로컬 화면: 위 Y2m checkpoint 경로를 CKPT에 지정
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=push_at EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 \
TASK_CAMERA=agent PUSH_GOAL_SWITCH_ANGLES=90,180 OUTPUT_PATH=output_etc/check/push_goal_switch_local \
bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth200_20kg_friction04_test.sh "$CKPT" 2 1 4 10

# 화면 없는 평가: 64환경·각도별5회
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=push_at EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 \
PUSH_GOAL_SWITCH_ANGLES=90,180 OUTPUT_PATH=output_etc/check/push_goal_switch_eval \
bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth200_20kg_friction04_test.sh "$CKPT" 2 64 4 10
```

## Y1.8m 높은 손 위치 전도 검사

PUSH 작업 사본 루트에서 GPU6으로 실행한다. X1.2~1.5m/5cm·Y1.8m·Z1.3m와 X=Y 대조를 같은 질량으로 검사한다. 접촉체 높이0.65/0.95/1.0/1.1/1.12/1.15/1.2m·유한하중1/2mg·3양손배치, 실제humanoid21AMP선택자세·고정높이/원본높이 PD/IK 검사다. 학습/정책 평가는 아니며 기존 학습·설정·checkpoint를 변경하지 않는다. 출력은 존재하지 않는 `output_etc/check/` 하위 경로만 허용한다.

```bash
TOKENHSI_GPU=6 PUSH_DEPTH_MASS=20 bash tokenhsi/scripts/multi_agent/analyze_push_depth180_high.sh
TOKENHSI_GPU=6 PUSH_DEPTH_MASS=25 bash tokenhsi/scripts/multi_agent/analyze_push_depth180_high.sh
```

각 출력의 `REPORT.md`, `summary.json`, 접촉체/실제humanoid별 `audit.json`·`trace*.npz`를 확인한다. `PUSH_DEPTH_MASS` 기본은20kg이며25kg 비교도 같은 하중/mg 비율을 사용한다. 따라서 절대 힘을 고정한 질량 단독 대조는 아니다. `python tokenhsi/scripts/multi_agent/report_push_depth180_high.py <검사폴더>`로 trace를 재검증·재집계한다.
