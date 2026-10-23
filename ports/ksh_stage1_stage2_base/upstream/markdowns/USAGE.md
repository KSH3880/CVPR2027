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
