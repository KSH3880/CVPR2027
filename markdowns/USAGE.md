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

### 재부팅 전 NVML 불일치 우회

현재 커널 드라이버가 `580.173.02`, 설치된 NVML이 `580.178.04`인 서버에서는 아래 명령으로 `nvitop`만 구버전 NVML을 사용한다. 저장소 루트에서 실행하며, 시스템 라이브러리와 학습 프로세스는 변경하지 않는다. 재부팅으로 드라이버가 갱신되면 이 우회 대신 일반 `nvitop`을 사용한다. `LD_LIBRARY_PATH`를 전역 export하지 않는다.

```bash
LD_LIBRARY_PATH="$PWD/output_etc/check/nvml_compat/lib/580.173.02${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" nvitop
```

구버전 라이브러리는 [Ubuntu 공식 snapshot 패키지](https://snapshot.ubuntu.com/ubuntu/20261001T000000Z/pool/restricted/n/nvidia-graphics-drivers-580/libnvidia-compute-580_580.173.02-0ubuntu0.24.04.1_amd64.deb)에서 `libnvidia-ml.so.580.173.02`만 추출하고 같은 폴더에 `libnvidia-ml.so.1` 심링크를 만들었다. 파일은 이 checkout의 Git 제외 산출물이므로 다른 checkout에는 자동 제공되지 않는다.

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

## Stage2 BEFORE 평가 옵션

`stage2_before`는 현재 레포의 fixed Stage1 전이 기반으로 새 학습을 시작한다. 기본 source는 `stage1/Stage1V2Fixed_00030500.pth`이며 `STAGE1_CHECKPOINT`로 지정한다. 기존 공동운반·공유 받침·다른 레포의 BEFORE checkpoint는 직접 재개/평가하지 않는다. train/test 환경 YAML 재정의는 `BEFORE_ENV_CONFIG`이고, test/VNC는 기본적으로 checkpoint 옆 `relation_config.yaml`을 사용한다.

인원 N≥1, 물체 M≥N을 train/test/VNC 위치 인자로 지정한다. 예: 아래 명령의 `4 1 8 10`을 `6 1 6 10`으로 바꾸면 6명·1환경·6물체다. BEFORE는 항상 2인 그룹이며 물체 예산을 넘는 환경은 task graph 전체를 재샘플링한다. 예산이 작으면 최종 그룹·과제 비율이 달라질 수 있다. 전원 `carry_ontop`은 M≥2N을 요구한다. 기존 4명·8물체 BEFORE 저장본도 가변 인원·물체 평가에 사용할 수 있다.

`BEFORE_FOLLOWER_NEAR=1`은 loco 평가 전용이다. 각 BEFORE 쌍의 선행 상자 중심까지 두 사람의 root XY 거리를 1~5m에서 추첨하고 후행에 작은 값을 준다. ON_TOP(STACK) 후행이 들고 올 자기 상자는 후행 사람의 최종 root XY 기준 1~3m로 별도 재배치한다. 각도도 매 reset 추첨하며 기존 높이와 자세 기울기를 보존하고 사람·상자 겹침을 검사한다. 거리는 초기 배치에만 적용하며 단독 인원, 옵션을 끈 평가, 기본 학습 배치와 보상은 바꾸지 않는다. 같은 seed는 같은 추첨을 재현한다.

```bash
# stage2_before
BEFORE_FOLLOWER_NEAR=1 TOKENHSI_GPU=5 PORT=6083 SEED=5125 \
TASK_GRAPH=random_scenario EVAL_SKILLS=loco \
bash tokenhsi/scripts/multi_agent/stage2_before_vnc.sh "$CKPT" 4 1 8 10
```

## Stage2 BEFORE 바닥 전용

`stage2_before_ground`는 인공 시작/도착 받침을 비활성화한 별도 비교 실험이다. 기본 GPU5·4명·1024환경·8물체이며, 가변 인원·물체와 `BEFORE_FOLLOWER_NEAR=1` 평가 옵션은 동일하다. `STAGE1_CHECKPOINT`로 fixed Stage1 가중치를 지정하고 `RESUME_CHECKPOINT`에는 ground 실험 저장본을 쓴다. 환경 YAML 재정의는 `BEFORE_GROUND_ENV_CONFIG`이며 test/VNC 기본값은 저장본 옆 `relation_config.yaml`이다. 기존 `stage2_before` 저장본은 ground 재개/평가에 사용하지 않는다. 기본 명령은 [config](config.md), 배치 계약은 [BEFORE 참고](reference/stage2_before.md#바닥-전용-비교-stage2_before_ground)를 따른다.

```bash
# stage2_before_ground
BEFORE_FOLLOWER_NEAR=1 TOKENHSI_GPU=5 SEED=5125 TASK_GRAPH=random_scenario EVAL_SKILLS=loco \
OUTPUT_PATH=output_etc/check/stage2_before_ground/follower_near_viewer \
bash tokenhsi/scripts/multi_agent/stage2_before_ground_vnc.sh "$CKPT" 4 1 8 10
```

## Stage2 BEFORE 50/50 혼합 배치

`stage2_before_ground_mix50`는 기본 GPU0·4명·1024환경·8물체다. train/test/VNC는 전용 wrapper를 사용한다. 기본 평가는 저장 config의 50/50 혼합 배치를 사용하며, test/VNC에 `BEFORE_FOLLOWER_NEAR=1`을 붙이면 BEFORE 쌍에 가까운 모드만 적용한다. 선행은 초기 상자에서 1~10m, 후행은 선행 초기 상자에서 1~5m, ON_TOP 자기 상자는 후행에서 1~5m이며 후행이 더 가깝도록 강제하지 않는다. 단독 인원·checkpoint 계약·학습 비율은 유지한다. 배치·비율·TensorBoard 지표는 [혼합 배치 계약](reference/stage2_before.md#바닥-before-거리-혼합-stage2_before_ground_mix50)을 따른다.

기본 Stage1 source는 저장소 루트 기준 `../stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00073000.pth`다. 다른 source는 `STAGE1_CHECKPOINT`로 지정한다. 재개는 이 실험의 checkpoint를 `RESUME_CHECKPOINT`로 지정하고, test/VNC도 이 실험의 checkpoint를 사용한다. 기존 ground checkpoint는 혼합 배치 계약이 달라 직접 resume하지 않는다.

환경 YAML 재정의는 `BEFORE_GROUND_MIX50_ENV_CONFIG`이며 test/VNC 기본값은 저장본 옆 `relation_config.yaml`이다. 기본 명령은 [config](config.md)에 있다. 본학습 전 `CHECK`, `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT`, `BEFORE_FOLLOWER_NEAR`의 잔여값을 해제한다. 짧은 확인 학습은 `CHECK=1`로 실행하고 환경 수는 1024를 유지한다.

```bash
# stage2_before_ground_mix50 — 가까운 BEFORE 배치 viewer
HEADLESS=0 BEFORE_FOLLOWER_NEAR=1 TOKENHSI_GPU=0 \
SEED=100 TASK_GRAPH=random_scenario EVAL_SKILLS=loco \
OUTPUT_PATH=output_etc/check/stage2_before_ground_mix50/follower_near_viewer \
bash tokenhsi/scripts/multi_agent/stage2_before_ground_mix50_test.sh \
"output/stage2_before_ground_mix50/Stage2BeforeGroundMix50_20-14-08-16/nn/Stage2BeforeGroundMix50.pth" 4 1 8 10
```

## Stage2 BEFORE mix334 1200

`stage2_before_ground_mix334_1200`은 GPU0·4명·1024환경·8물체, 최대1200스텝의 별도 실험이다. [배치 표](reference/stage2_before.md#바닥-before-거리-혼합-stage2_before_ground_mix334_1200)와 [학습·로컬·VNC 명령](config.md)을 따른다. 세 전용 wrapper를 사용하고 환경 재정의는 `BEFORE_GROUND_MIX334_1200_ENV_CONFIG`로 지정한다. test/VNC는 저장본 옆 `relation_config.yaml`을 기본으로 읽는다.

Stage1 source는 mix50와 같은 형제 checkout의 fixed73000이며 `STAGE1_CHECKPOINT`로 지정할 수 있다. `RESUME_CHECKPOINT`에는 이 새 실험의 저장본만 사용한다. 기존 ground/mix50를 직접 재개하는 설정이 아니다. 기본 평가는30/30/40이고, `BEFORE_FOLLOWER_NEAR=1`은 가운데 가까운 모드만 선택한다(더 가까운 모드 강제가 아님). `EPISODE_LENGTH`는 평가에만 적용하며 기본은1200이다. 단독 배치·학습 비율·저장 계약은 평가 옵션으로 변경하지 않는다.


## Bridge PUSH4 본학습 관리

기존 `tokenhsi-bridge-push4-20261023.service`는 사용자 요청으로 중단했다. 결과는 `output/stage2_bridge_push4/`에 보존한다. 현재 본학습은 근접·직선 혼합 실험 `tokenhsi-bridge-push4-mix50-20261024.service`이며, Stage1 PUSH raw13500에서 새로 시작한다. 4명·1024환경·8물체로 대화·SSH 세션 종료 후에도 유지된다. 일반 실행·평가는 [config](config.md), 배치·과제 계약은 [Bridge 참고](reference/stage2_bridge.md#stage2_bridge_push4_mix50)를 따른다.

```bash
systemctl --user status tokenhsi-bridge-push4-mix50-20261024.service
tail -f output/stage2_bridge_push4_mix50/train.log
```

장면 확인용 원본10장·모아보기·장면별 역할/좌표는 `output_etc/check/stage2_bridge_push4/scenes/`에 있다. 초기 배치 확인용이며 학습된 건너기 성공을 보여주는 자료는 아니다.
