# TokenHSI + PUSH, fine-tuning with policy distillation

원본 TokenHSI Stage1을 teacher/student 초기값으로 사용한다. PUSH는 PPO+AMP, 기존 SIT/CARRY/CLIMB는 고정 teacher의 Gaussian forward KL로 보존한다. 추론에는 student만 사용한다.

방법·비교표·수식·차이와 제한은 **[METHOD.md](METHOD.md)**, 코드 위치는 [STRUCTURE.md](STRUCTURE.md), 검증 결과는 [CHANGELOG.md](CHANGELOG.md)를 참고한다.

상자 크기와 student 입력의 관계는 [PUSH_INPUT.md](docs/PUSH_INPUT.md)에 정리했다. PUSH 큰 상자의 실제8개꼭짓점을 입력하고, 기존과제 KD 물체는 원본TokenHSI 설정을 유지한다.

## 준비와 CPU 검사

로컬 conda 환경 `tokenhsi_koo`의 Isaac Gym/rl_games/PyTorch를 사용한다. 모션·물체 데이터는 이 머신의 기존 프로젝트를 읽기 전용 링크/절대경로로 참조한다. 원본 소스는 수정하지 않는다. 다른 머신에서는 manifest 경로와 데이터 링크를 해당 머신의 데이터 위치로 바꿔야 한다.

```bash
cd /home/inkyu/projects/tokenhsi_push
python scripts/preflight.py
python -m pytest -q tests
```

원본 teacher와 확장 초기 student는 `weights/teacher_stage1.pth`, `weights/push_init.pth`에 준비돼 있다. 초기 student 재생성:

```bash
python scripts/expand_checkpoint.py --src weights/teacher_stage1.pth --dst weights/push_init.pth
```

## 실행

아래는 실행 방법이며 이번 작업에서 학습을 시작하지 않았다. GPU 점유/MPS 격리 확인 후 사용한다. 기본 학습은참조2048scene(2인·4상자)+native KD2048환경·26,000epoch이고 GPU를 자동 선택하지 않는다.

```bash
# 학습
TOKENHSI_GPU=0 bash scripts/train.sh

# 별도 출력의 짧은 통합 확인: 같은두pool, 1 PPO epoch
TOKENHSI_GPU=0 MAX_ITERATIONS=1 OUTPUT_PATH=output_etc/check/smoke bash scripts/train.sh

# 동일 계약 저장본 재개
TOKENHSI_GPU=0 RESUME_CHECKPOINT=/absolute/student.pth bash scripts/train.sh

# teacher를 로드하지 않는 student PUSH 평가, 기본 loco/512환경
TOKENHSI_GPU=0 CHECKPOINT=/absolute/student.pth EVAL_TASK=push bash scripts/eval.sh

# 기존 과제 평가
TOKENHSI_GPU=0 CHECKPOINT=/absolute/student.pth EVAL_TASK=carry bash scripts/eval.sh
```

`kd_coef`는 `tokenhsi/data/cfg/train/rlg/push_kd.yaml`에 있다. 기본0.001, 검증 후보0.0001/0.001/0.01이다. 비교 실험은 YAML/출력 경로를 분리하고 test 성능으로 선택하지 않는다. PUSH 평가 JSON은 기본 `output_etc/check/push_eval.json`에 저장하므로 seed별 실행 시 이 파일을 별도 보존한다. 기존3과제는 native TokenHSI 평가 규칙을 사용하며 우리 관계 과제와 동일한 항목이라고 주장하지 않는다.

현재 실행 경로는 `env/mixed_reference_env.py`다. 초기1인PUSH adapter는 비교 이력으로만 보존하며 학습 runner에서 사용하지 않는다. PUSH는 참조26k의 원본mixed scene, 기존3과제KD는 원본TokenHSI환경이다. 상세pool규모와masked slot은 [METHOD.md](METHOD.md)를 따른다.

CPU 물리 연결 검사(학습 아님):

```bash
TORCH_EXTENSIONS_DIR=/tmp/tokenhsi_push_extensions python scripts/check_reference_cpu.py push
TORCH_EXTENSIONS_DIR=/tmp/tokenhsi_push_extensions python scripts/check_reference_cpu.py mixed
```

TRAJ는 학습/KD/평가에서 제외한다. GPU 본학습은 미실행이며 전체GPU PPO·성공률은 미검증이다.

**검증 상태:** native pool을 spawn worker로 격리한 혼합 환경의 CPU reset·물리3step과 실제 student 행동3step을 통과했다. CPU 테스트12개도 통과했다. GPU PPO 통합·장기 성공률은 미검증이며 CPU 배열 IPC의 전송 비용도 측정 대상이다. 확정한 토큰·AMP·RSI 상세는 [METHOD.md](METHOD.md)에 있다.
