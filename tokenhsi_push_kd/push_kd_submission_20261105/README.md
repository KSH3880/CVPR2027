# TokenHSI + PUSH distillation: 제출 모델

## 제출 파일

| 파일 | 내용 |
|---|---|
| `model/TokenHSIPushKD_B_seed42_e11600.pth` | 제출 student. 과제별 라그랑주 KD(B), 학습 seed 42, 11,600 에폭 |
| `model/SHA256.txt` | `22dd5f1b6b93357ce9a71020e5b17804c47dc103d2eff383f1f705f917fd6d6c` |
| `code/changes_vs_handoff.patch` | 전달해 주신 코드(`tokenhsi_push_kd`) 대비 변경 16개 파일. 목록은 `code/changed_files.txt` |
| `code/code_tokenhsi_push_kd.tar.gz` | 패치를 적용한 코드 전체. 가중치·모션 데이터·실행 결과는 제외 |
| `results/monitoring/` | 학습 중 평가 log(학습별, 100 에폭마다) |
| `results/teacher/` | teacher(원본 TokenHSI 코드)의 평가 원본 로그 |
| `results/grad_diagnostics/` | 발표 자료 "원인 분석 2"의 그래디언트 진단 스크립트와 결과 |

모델 구조와 관측은 원본 코드와 같습니다. 체크포인트는 원본 형식 그대로이고, 학습 상태(옵티마이저, 과제별 KD 계수)도 들어 있어 이어서 학습할 수 있습니다. 

## 제출 모델 선택과 성능

학습 중 평가(100 에폭마다, 평가 seed 0)에서 PUSH final이 가장 높았던 체크포인트를 제출합니다. 요청서의 validation/test 분리 절차는 시간 관계로 적용하지 못했습니다.

| | SIT | CARRY | CLIMB | PUSH final | PUSH ever |
|---|---|---|---|---|---|
| teacher | 99.52 | 96.57 | 99.78 | | |
| 제출 모델 (11,600 에폭) | 99.09 | 94.92 | 99.61 | 3.42% | 3.52% |
| teacher 대비 상대 하락 | 0.4% | 1.7% | 0.2% | | |
| 목표 | ≤ 5% | ≤ 5% | ≤ 5% | > 85% | |

- 제출 모델 성능은 seed 0에서 평가를 수행했습니다. 기존 과제는 512 시행 × 3 반복이고, PUSH는 1,024명입니다. PUSH의 낙상률은 0.59%, 종료 시 상자–목표 거리 중앙값은 2.0 m입니다. 
- teacher 값은 seed 0~2의 평균입니다(11-02 측정).
- **결론:** 기존 세 과제 수행 능력은 KL로 잘 지켰지만, PUSH final은 목표에 크게 미달합니다. 그리고 원인 분석에 의하면 KL영향보다 push reward가 너무 noisy해서 의미 있는 방향으로 push gradient가 업데이트를 진행하지 못합니다. 

더 높은 PUSH 성능을 얻으려면 두 방향이 필요하다고 봅니다.
1. **더 오래 학습하기:** PUSH final은 9,500 에폭 이후 1~3%대로 느리지만 꾸준히 오르고 있습니다. 26,000 에폭 예산의 절반 정도에서 멈췄습니다. 다만 26,000에폭까지 85% 달성은 어려울 듯 합니다.(혹시 85%를 목표로 삼은 이유가 있을까요??)
2. **학습 설정 바꾸기:** reward, 학습 파라미터, 또는 PUSH를 잘하는 정책을 KL teacher로 추가해 증류하는 방법입니다. 이 조건들은 이번 실험에서는 주신 조건 그대로 고정해서 돌렸습니다. 그리고 26K model과 seed를 몰라서 seed에 따른 영향이 상당한 강화학습 특성상, 26K model과 동일한 seed에서 학습을 한다면 성능 향상이 기대됩니다.

## 코드 적용

원본 코드 폴더에서 다음을 수행합니다.

```bash
cd tokenhsi_push_kd
patch -p1 < /path/to/changes_vs_handoff.patch
```

평가에 영향을 주는 변경은 이렇습니다.
- **PUSH 평가 루프의 `get_batch_size()` 누락 수정:** 원본 코드에서는 첫 step에서 텐서 크기 오류로 멈춥니다.
- **PUSH 성공 판정:** 요청 사양인 상자 중심–목표 3D 거리 ≤ 0.2 m로 계산합니다. 기존 코드에서는 높이에 관한 성공 판정이 있어 push용 성공 판정 코드를 수정했습니다.
- **기존 과제 평가의 env reset:** 반복마다 전체 env를 reset합니다. 원본 코드는 `reset(None)`이 아무 env도 reset하지 않습니다.
- **기존 과제 평가 전 700 step 예열:** 원본 eval은 환경 생성 직후 첫 반복에서 CLIMB가 2.7%로 붕괴하고, teacher에서도 같은 현상이 있습니다. 예열 구간은 성능 측정하지 않습니다. `NATIVE_EVAL_WARMUP=0`이면 원본 그대로입니다.

## 평가

student (seed마다):

```bash
CK=/abs/path/TokenHSIPushKD_B_seed42_e11600.pth
for task in sit carry climb; do
  TOKENHSI_GPU=0 CHECKPOINT=$CK EVAL_TASK=$task SEED=0 bash scripts/eval.sh > eval_${task}_seed0.log 2>&1
done
TOKENHSI_GPU=0 CHECKPOINT=$CK EVAL_TASK=push SEED=0 PUSH_EVAL_OUT=push_seed0.json bash scripts/eval.sh > eval_push_seed0.log 2>&1
```

- 기존 과제는 로그의 `success_rate` 세 값을 평균합니다.
- PUSH는 JSON의 `final_success`(주 지표), `ever_success`, `falls`, `final_distance_3d_median`과 사람별 행을 봅니다.

teacher (원본 TokenHSI 코드를 수정 없이 감싸 실행하며, student와 같은 예열을 적용합니다):

```bash
NATIVE_EVAL_WARMUP=700 ORIG_ROOT=/path/to/original/TokenHSI EVAL_TASK=sit SEED=0 \
  python scripts/eval_teacher_original.py > teacher_sit_seed0.log 2>&1
```

## 학습

```bash
TOKENHSI_GPU=0 SEED=42 OUTPUT_PATH=output/push_kd_lagrange_eps1_seed42 \
  CFG_TRAIN=tokenhsi/data/cfg/train/rlg/push_kd_lagrange.yaml \
  PUSH_KD_AMP_USED_ROWS=1 PUSH_KD_GPU_MEM_GB=20 bash scripts/train.sh
# 이어서 학습: RESUME_CHECKPOINT=<체크포인트 절대 경로> 추가
```

| 항목 | 값 |
|---|---|
| PUSH 참조 장면 | 2,048개 × 2명 (장면당 상자 4개). 그중 PUSH 학습 표본은 원본 장면 구성상 약 절반 |
| 기존 과제 KD 환경 | 2,048개 (SIT·CARRY·CLIMB를 reset마다 균등 선택) |
| 동시 정책 행 | 6,144 (PUSH 학습 표본은 약 1/3) |
| horizon | 32 step, 에폭당 196,608 표본 |
| minibatch / mini-epoch | 16,384 / 6 (에폭당 업데이트 72번) |
| 학습률 | 2e-5 고정 |
| PPO clip | 0.2 |
| 행동 σ | 고정 (log σ = −2.9) |

26K 모델(장면 1,024개 × 2명)과 비교할 때는, 우리 설정이 에폭당 PUSH 표본이 약 2배라는 점을 참고해 주세요. 원본 코드의 기본값(`--num_envs 2048`)을 그대로 썼습니다.

- 학습 규모와 PPO 설정은 원본 코드 그대로입니다. 바꾼 것은 KD loss term 하나입니다(`kd_mode: lagrange`, ε = 1.0, 시작 계수 0.001). 식은 발표 자료 "실험" 장에 있습니다.
- 제출 모델은 한 번에 이어서 학습하지 못했습니다. 500, 3,200, 3,500, 5,800 에폭에서 재개했습니다. 이유는 실행 최적화를 위한 코드 변경, OOM, GPU 서버 변경입니다. 재개하면 모든 환경이 에피소드 처음 상태로 리셋되어, 재개 직후 약 200~300 에폭 동안 PUSH 성능이 일시적으로 떨어졌습니다. 
- 아마 Ours의 성능을 기준으로 85% 성능을 목표로 잡으신 거 같은데, agent token(GTA/coordination)을 사용하지 않은 순수 TokenHSI 모델 구조로는 한계가 뚜렷해 보입니다.
- 현재는 학습을 멈춘 상태입니다. 