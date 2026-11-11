# TokenHSI PUSH + KD · standalone branch

이 브랜치는 `CVPR2027/tokenhsi_push_kd`의 코드만 저장소 루트에 둔 독립 snapshot이다.
다른 CVPR2027 연구 폴더와 부모 저장소의 과거 이력을 포함하지 않는다.

- `tokenhsi/`: 정책·학습·평가 코드와 설정.
- `push_reference/`: PUSH reference 물리 환경 의존성.
- `lpanlib/`: 실행에 필요한 유틸리티.
- `scripts/`, `tests/`, `docs/`, `handoff_support/`: 실행기·검증·문서·데이터 경로 템플릿.
- 가중치, 실제 모션/메시 데이터, 로컬 심링크, output/output_etc, 과거 제출 묶음은 제외했다.

환경·데이터·teacher/초기student 체크포인트는 별도로 준비해야 한다.
`handoff_support/manifests`와 `scripts/prepare_paths.py`를 참고한다.
`output_etc`에 보관된 다인 Table1/2 평가 실행기·결과는 이 코드 snapshot에 포함하지 않는다.
기존 머신의 경로·학습상태를 서술하는 아래 기록은 과거 실행 기록이며 새 clone의 상태가 아니다.

---

# TokenHSI PUSH + KD 이관본

## 현재 학습 (2026-11-06 16:09)

2026-11-07 06:31:33(KST)에5758 ITER 완료 후 CUDA launch timeout으로 종료됐다. 같은 시각 커널에 NVIDIA watchdog GPU locked/Xid8(PID444518)이 기록됐다. 직접적인 OOM 메시지는 없으며 GPU 정지의 근본 원인은 미확정이다. 최신5500 ITER 체크포인트는 CPU 로드·모델 finite·optimizer state 존재를 확인했다. 현재 학습은 중단 상태이며 `output/active_training.json`과 실행 폴더의 `failure_report_20261107.json`을 참조한다.

사용자 요청으로 수정된 shared simulator(version2)의 PUSH4096/native KD2048, KD0.1 학습을 seed42 초기 모델에서 새로 시작했다. PID444518, 최대26000 ITER, horizon32/minibatch16384/mini_epochs6/학습률2e-5를 사용한다. 실행 폴더는 `output/push4096_kd0p1_sharedv2_fresh_seed42_20261106_160919/`, 로그는 그 안의 `train.log`다. 초기 ITER 진행과 환경 계약 및 TensorBoard KD0.1을 확인했다. 이어서 학습하는 실행이 아니며, 최신 메타데이터는 `output/active_training.json`을 참조한다.

## 현재 검증 결과 (2026-11-06)

단일 simulator 통합의 reset 전달 버그를 수정했다. 같은 physics step 사이에 reference/native가 전역 state setter를 반복 호출하여 reference reset이 실제 physics에 반영되지 않았다. 두 풀의 상태·인덱스를 모아 simulate 직전에 setter별 한 번 제출하고, refresh도 step별 한 번으로 제한한다. Native의 Python/NumPy/Torch RNG도 기존 worker처럼 분리한다. PPO/KD loss 식과 optimizer 업데이트는 변경하지 않았다.

같은 reset과 액션을 사용한 8+8 GPU 환경 검사에서 수정 전 첫 step의 위치 차이는 최대34.7597m, 보상 차이는0.9910이었다. 수정 후12 step 및 중간 부분 reset에서 PUSH의 물리 상태·관측·보상이 worker와 정확히 일치했다. Native도 첫 reset 전 warmup과 환경의 월드 배치 좌표를 맞추면12 step 동안 물리 상태·관측이 정확히 일치한다. 배치 좌표와 초기 warmup이 다른 실제 두 backend의 전체 학습 trajectory가 bitwise 동일하다는 뜻은 아니다.

PUSH2048/native KD2048, KD0.001, seed42, 같은 초기 체크포인트에서 각각60 ITER를 완료했다. horizon32/minibatch16384/mini_epochs6/학습률2e-5를 유지했으며, 두 실행의 환경 계약은 backend 및 `sharedSimVersion: 2` 외에 동일하다.

| 방식 | PUSH reward 평균21–40 ITER | 평균41–60 ITER | 60 ITER 값 |
| --- | ---: | ---: | ---: |
| 기존 worker 재실행 | −1.48 | 30.38 | 31.64 |
| 수정한 shared 재실행 | 1.57 | 35.70 | 39.12 |

이 단기 비교에서는 보상 상승이 재현됐다. 장기 수렴·PUSH4096 성능·기존 스킬 보존까지 검증한 결과는 아니다. 수정 전 shared의 KD0.1/0.01/0.001 실행에는 이 버그가 있으므로 그 결과만으로 KD 계수나 환경 수의 영향을 판단할 수 없다. 기존20 ITER VRAM 검사는 물리 동등성을 검증하지 못했다.

회귀 테스트23개 통과. 두60 ITER 검증과 진단 프로세스 모두 종료했다. 결과·원본 trace·로그·입력 해시는 `output_etc/check/backend_parity/`에 있으며 `backend_comparison.png`, `training_comparison.json`, `physics_summary.json`, `report.txt`를 참조한다. 수정 전 shared 체크포인트는 새 shared 계약과 strict resume이 일치하지 않는다. 비교에는 별도 `env2048.yaml`을 사용했으며, 이후 위의 새 본학습 요청에 따라 기본값을 PUSH4096/native2048·KD0.1로 설정했다.

## 단일 simulator 비교 기록 (2026-11-06)

`env.simBackend: shared`로 reference PUSH 2048환경과 native KD 2048환경을 하나의 Isaac Gym simulator/학습 프로세스에 생성한다. 각 풀의 actor/DOF tensor와 reset 인덱스를 분리하고, 두 풀의 action을 넣은 뒤 physics를 한 번의 공통 루프로 진행한다. Reference의 PUSH 접촉 누적과 기존 PPO/KD loss 경로를 유지한다. Reference PUSH100%, KD0.01, horizon32/minibatch16384/mini_epochs6 설정이다.

RTX5090/tokenhsi118에서 각 방식으로 초기 상태부터 20 ITER를 실행한 학습 GPU 프로세스 합산 peak(1초 샘플, 데스크톱 제외):

| 방식 | 프로세스 수 | VRAM peak | 평균 ITER 시간 |
| --- | ---: | ---: | ---: |
| 기존 worker | 2 | 24.68 GiB | 7.74초 |
| shared simulator | 1 | 20.87 GiB | 7.75초 |

약 3.81 GiB(15.4%) 절감했다. 장기 학습 peak 보장은 아니며, 생성 순서가 달라 같은 seed도 rollout이 완전히 같지는 않다. 비교 자료는 `output_etc/check/shared_sim_vram/comparison.json` 및 각 하위 폴더에 있다. 검증 실행은 종료했으며 본학습은 중단 상태다.

현재 shared는 headless 학습 및 두 풀의 동일한 physics/ground/control 주기를 지원한다. Domain randomization은 지원하지 않는다. 평가/view는 기존 경로를 사용한다. 기존 worker가 필요하면 `TOKENHSI_SIM_BACKEND=worker`를 설정한다. 실제 선택한 backend를 체크포인트 환경 계약에 기록하므로 이전 backend 체크포인트의 strict resume 계약은 달라진다.

학습 환경 설정에 `env.referenceNumEnvs`를 지정하면 reference 환경 수만 별도로 설정한다. 생략하면 기존처럼 `numEnvs`와 같고, native KD는 계속 `numEnvs`개다. 학습 버퍼와 collision group offset도 두 풀의 실제 크기를 따른다. 평가에서는 기존 `--num_envs`를 사용한다. 수정 전 PUSH4096/KD2048 검증 기록은 `output_etc/check/shared_sim_push4096/`에 있다. 기본 설정은 이 옵션이4096이다.

## 새 학습 기본값 (2026-11-05)

현재 베이스 학습은 reference 슬롯의 PUSH 비율을 **100%**로 설정한다.
Reference의 기존 과제 비율은 모두 0%이며, 기존 스킬 distillation은 별도 native KD 풀에서 수행한다.
Reference는 장면당 1명·상자 2개(작업/보조 각 1개)다. Reference 2048명 + native KD 2048명으로 전체 4096개 정책 행 중 PUSH는 2048개(50%)다.
PPO minibatch16384/horizon32/mini_epochs6은 유지한다. 인원 일반화로 RSI 캐시 코드가 변경되어 기존 캐시와 별도로 재생성한다.
이전 제출 실험의50% 설정은 제출 압축파일에 보존되어 있다.

`push_kd.yaml`의 초기 체크포인트는 `weights/push_init_random_seed42.pth`다.
PUSH tokenizer(42→256→128→64)만 TokenHSI 기본 방식의 랜덤 weight/zero bias로 초기화한다.
기존 모델 가중치와 CARRY에서 가져온 PUSH 관측 정규화 통계는 유지한다.

```bash
/home/injesus1010/anaconda3/envs/tokenhsi118/bin/python scripts/expand_checkpoint.py \
  --src weights/teacher_stage1.pth --dst weights/push_init_random_seed42.pth --seed 42
```

새 초기화를 사용하려면 `RESUME_CHECKPOINT` 없이 새 학습을 시작한다. 기존 학습을 resume하면 저장된 tokenizer가 복원되며, 변경된 환경 계약도 기존 체크포인트와 다르다.
아래 이관 설명의50%/CARRY 복사 관련 내용은 변경 전 실험을 설명한다.

이전 2인 설정 VRAM 실측: RTX5090/tokenhsi118에서3 epoch가 정상 완료됐으나 본학습은18 epoch 후 AMP replay sample 할당 중 OOM 종료됐다.
학습 두 프로세스의1초 간격 샘플 peak 합계는29592MiB(28.90GiB)였다(다른 앱 제외).
main PyTorch peak는 allocated17.20GiB/reserved19.19GiB이며 simulator와 KD worker를 포함한 총량과 구분해야 한다.
측정 로그는 `output_etc/check/vram_push70/`에 있다. 장기 학습의 최대 사용량 보장은 아니다.

로컬 본학습 실행(tokenhsi118/RTX5090 호환, 현재 베이스의 고정 KD0.01):

```bash
TOKENHSI_GPU=0 SEED=42 OUTPUT_PATH=output/push100_random42_seed42 bash scripts/train_local.sh
```

2026-11-05 시작한 2인 실행은 OOM으로 종료됐다. 실행 기록은 `output/active_training.json`에 있다. PUSH80%·1인 본학습을 `output/push80_solo_random42_seed42_20261105_202741/`에서 실행했으며 사용자 요청으로 중단했다(202 iteration 로그). 실행 정보는 `output/active_training.json`을 참조한다.

## 제출 모델 로컬 추론 (2026-11-05)

이 PC의 `tokenhsi118` 환경과 RTX 5090에서 제출 11,600 epoch 모델의 PUSH 추론을 확인했다.

```bash
# 저장소 루트에서 실행. 기본 4장면/8명, seed 0, 1200-step horizon.
TOKENHSI_GPU=0 bash scripts/infer_submission.sh

# 화면 표시: 종료된 장면을 자동 reset하며 반복 (Escape 또는 창 닫기로 종료)
TOKENHSI_GPU=0 HEADLESS=0 NUM_ENVS=1 bash scripts/infer_submission.sh
```

기본 결과는 `output_etc/check/submission_inference/push_eval.json`에 저장된다.
`NUM_ENVS`, `SEED`, `CHECKPOINT`, `PUSH_EVAL_OUT`, `TOKENHSI_RUNTIME`을 환경변수로 변경할 수 있다.
GUI는 기본 연속 재생이며 `VIEW_LOOP=0`을 지정하면 한 에피소드 평가 후 종료한다. 연속 재생은 평가 JSON을 기록하지 않는다.
저장소를 옮기면 해당 Python으로 `scripts/prepare_paths.py`를 실행해 데이터 경로를 재생성한다.
4장면 실행은 동작 확인용이며 제출 성능 재현 평가가 아니다.

실험 목표는 **distillation 범위 안에서 기존 SIT/CARRY/CLIMB 성능 하락 5% 이내와 우리 목표 영역 기준 PUSH final 성공률 >85%를 동시에 만족하는 단일 student `.pth`**다. 현재 나이브 KD 구현의 점검도 포함한다. 평가 정의·탐색 범위·논문용 재현 및 제출 조건은 [실험 요청서](EXPERIMENTS.md), 전달용 문안은 [메시지](MESSAGE.md)를 따른다. 이는 목표이며 현재 달성한 성능이 아니다.

이 폴더 전체 또는 함께 제공한 압축파일을 인턴에게 전달하면 된다. GitHub 계정·clone은 필요 없다. 출처는 `baseline/tokenhsi-push-kd`, commit `0d6d488639429a97611c6bb379603a315e2d2867`이다. 실행 코드·물성·reward·AMP·RSI·KD 설정은 해당 커밋을 유지하며, 이관용 경로 설정과 출력 경로 검사만 추가했다.

포함: 소스, YAML, teacher/초기 student 가중치, 모션·물체·mesh 실제 파일, PUSH RSI 949후보 bank, CPU 테스트. 외부 데이터 심링크와 `.git`은 없다. **Isaac Gym SDK·conda 환경·학습 완료 PUSH 모델·기존 학습 결과는 포함하지 않는다.** Stage1 26k 정책 가중치는 필요 없다. 26k는 환경의 기준이고 teacher/student 초기값은 원본 TokenHSI Stage1이다.

## 1. 환경 준비

Linux/NVIDIA 환경에서 사용한다. 출발 머신의 실제 버전은 Python 3.8.20, torch 2.0.0, numpy 1.21.6, rl-games 1.1.4, gym 0.26.2, Isaac Gym 1.0rc4다. 전체 패키지 참고 목록은 `handoff_support/environment_versions.txt`다. GPU 학습 호환성을 보증하는 lockfile은 아니다. GPU 세대에 맞는 torch/CUDA/Isaac Gym 조합을 별도로 준비한다.

```bash
conda create -n tokenhsi_push python=3.8 -y
conda activate tokenhsi_push
# 서버 GPU/드라이버에 맞는 torch를 먼저 설치한다.
# 별도 제공받은 Isaac Gym Preview 4 SDK의 python 폴더를 지정한다.
pip install -e /absolute/path/to/isaacgym/python
# 이후 이 README가 있는 폴더로 이동한다.
pip install -r requirements.txt
pip install PyYAML tensorboard tensorboardX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
```

Isaac Gym SDK는 수신자가 별도로 준비한다. Python에서 직접 확인할 때는 `import isaacgym`을 `import torch`보다 먼저 실행한다. 설치 확인:

```bash
python -c 'import isaacgym; import torch; print(torch.__version__, torch.cuda.is_available())'
```

## 2. 폴더 검증과 경로 설정

아래 명령은 항상 이 README가 있는 폴더에서 실행한다. 폴더를 다시 옮기면 `prepare_paths.py`를 다시 실행한다. 모션 manifest 세 개의 경로만 현재 위치로 갱신하며 모션·보상·분포는 바꾸지 않는다. 수동으로 과거 `/home/inkyu/...` 경로를 만들 필요 없다.

```bash
python scripts/verify_bundle.py
python scripts/prepare_paths.py
python -m pytest -q tests --basetemp=output_etc/check/pytest_tmp -o cache_dir=output_etc/check/pytest_cache
```

`prepare_paths.py`는 원본 preflight도 실행하여 PUSH 모션 hash·mesh·949후보 계약·가중치 존재를 확인한다. 무결성 목록은 경로가 바뀌는 runtime manifest 세 개와 검증 결과 파일을 제외하며, 원본 데이터와 경로 template은 포함한다.

## 3. 짧은 통합 확인 → 본학습

먼저 `nvidia-smi`로 GPU 점유를 확인한다. 기존 프로세스는 종료하지 않는다. 아래 GPU 번호는 사용할 수 있는 번호로 바꾼다. CPU 검사를 통과한 뒤 GPU PPO 통합을 확인한다. 기본은 참조 2048scene(1명/2상자) + native KD 2048환경이며 4096 policy row다.

```bash
mkdir -p output_etc/check
TOKENHSI_GPU=0 MAX_ITERATIONS=1 OUTPUT_PATH=output_etc/check/smoke bash scripts/train.sh > output_etc/check/smoke.log 2>&1

# 통합 검사 성공 후 본학습. 확인용 제한 변수를 제거한다.
unset CHECK MAX_ITERATIONS RESUME_CHECKPOINT OUTPUT_PATH
TOKENHSI_GPU=0 OUTPUT_PATH=output/push_kd_seed42 SEED=42 bash scripts/train.sh

# 동일 설정 checkpoint에서 재개할 때만 사용한다.
TOKENHSI_GPU=0 RESUME_CHECKPOINT=/absolute/student.pth OUTPUT_PATH=output/push_kd_seed42 SEED=42 bash scripts/train.sh
```

`CHECK=1`은 1 epoch 확인 실행으로 처리한다. `CHECK` 또는 `MAX_ITERATIONS` 실행은 출력이 `output_etc/check/` 아래인지 검사한다. 본학습은 기본 26000 epoch, KD 계수는 `tokenhsi/data/cfg/train/rlg/push_kd.yaml`의 `kd_coef: 0.001`이다. teacher/RMS는 고정, PUSH만 PPO+AMP, native SIT/CARRY/CLIMB만 forward KL, TRAJ는 제외한다.

선택 CPU 물리 연결 확인(학습 아님, GPU 본학습 검증을 대신하지 않음):

```bash
TORCH_EXTENSIONS_DIR="$PWD/output_etc/check/torch_extensions" python scripts/check_reference_cpu.py mixed > output_etc/check/cpu_mixed.log 2>&1
TORCH_EXTENSIONS_DIR="$PWD/output_etc/check/torch_extensions" python scripts/check_reference_cpu.py policy > output_etc/check/cpu_policy.log 2>&1
```

## 4. 평가와 전달 결과

현재 단독 논문 평가는 [CVPR2027 프로토콜](cvpr2027_evaluation_protocol_1109.md)의 확정된 단독 규칙을 적용한다. 실행 방법과 기존 평가와의 차이·제한은 [단독 평가 구현 안내](docs/CVPR2027_SINGLE_EVAL.md)를 참고한다. 기존 SIT/CARRY/CLIMB 보존 평가 결과는 새 프로토콜 결과와 구분한다.

```bash
# SIT, CLIMB, AT, ON_TOP, PUSH 각각 512장면 × seed 0/1/2
EVAL_SEEDS="0 1 2" TOKENHSI_GPU=0 CHECKPOINT=/absolute/student.pth bash scripts/eval.sh
# PUSH만 새 프로토콜로 평가
EVAL_SEEDS="0 1 2" TOKENHSI_GPU=0 CHECKPOINT=/absolute/student.pth EVAL_TASK=push bash scripts/eval.sh
# 과거 보존 평가 재현 (새 프로토콜 결과가 아님)
EVAL_PROTOCOL=legacy TOKENHSI_GPU=0 CHECKPOINT=/absolute/student.pth EVAL_TASK=sit bash scripts/eval.sh
```

이관 확인 결과는 `HANDOFF_VALIDATION.md`, 실험 요청·기록 양식은 [EXPERIMENTS.md](EXPERIMENTS.md), 상세 방법은 [METHOD.md](METHOD.md), 원래 실행 안내는 [UPSTREAM_README.md](docs/UPSTREAM_README.md)를 참고한다. 원본 문서의 머신 절대경로는 출처 기록이다. GPU PPO 통합·장기 학습·성공률은 이관 작업에서 검증하지 않았다.

현재 실행: `output/push100_solo_random42_seed42_20261105_210745/` (PUSH100%, KD0.001, 새 초기화). 콘솔은 ITER 현재/최대와 FPS를 출력한다.

2026-11-06: KD 계수를0.01로 변경하고6500 ITER에서 재개했다. 현재 실행은 `output/push100_kd001_resume6500_20261106_114949/`이며, 의도적으로 KD 계수를 바꿔 resume할 때만 `PUSH_KD_ALLOW_COEF_CHANGE=1`을 사용한다.

현재 실행 정정: 사용자 의도에 따라 재개 실행을 중단하고 `output/push100_kd0p01_fresh_seed42_20261106_115849/`에서 KD0.01로 처음부터 시작했다(seed42 초기 체크포인트, resume 없음).
