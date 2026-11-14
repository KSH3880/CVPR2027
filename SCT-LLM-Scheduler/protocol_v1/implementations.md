# 레포별 구현·실행 안내

평가 규칙은 [README.md](README.md)의 `cvpr2027-evaluation-v1.0.0`을 따른다. 이 파일은 경로·실행 방법·현재 구현 상태만 기록한다. 아래 seed·checkpoint는 현재 실험 이력이며 최종 논문 선정 규칙이 아니다.

## 1. 현재 구현 상태

| 레포 | 역할·확인된 경로 | 적용 시 주의 |
| --- | --- | --- |
| `/home/hwanhee/ksh/stage1_2_base_teacher` | 공통 장면·Independent/Mixed·Table 2·SCT/Full 입력 변환·bias ablation·저장 궤적 CPU 채점 | 논문 경로는 `eval/` 보조 entry. 일반 `*_eval.sh`·viewer에 XY100%·논문 샘플러가 자동 적용되지 않음 |
| `/home/hwanhee/ksh/sct_coopft` | PPO4k의 Table 1/2 42회 완료. 공통 reference 장면과 CPU 재채점 확인 | 기존 suite의 checkpoint를 변경하지 않고 재사용. 다른 checkpoint는 새 manifest·출력 경로로 실행 필요 |
| `/home/hwanhee/ksh/stage1_2_base_jointpush` | Coupled teacher·공동 PUSH t_coop/Jerk 정의와 원본 함수 보존 | 기존 별도 실험은 seed0/1/2·1200step도 포함. 이 공용 규약의900step·공통 snapshot과 같다고 간주하지 않음 |
| `/home/hwanhee/ksh/stage_1_2_base_1031` | Enabling teacher·기존 거리/크기 샘플러 참고 | 학습 near50·reward/gate·기존 평가 기본값을 그대로 쓰지 말고 공통 크기/거리·동시 성공으로 연결 필요 |

다른 레포에서는 공통 데이터 계약에 맞는 adapter를 작성한다. 새 레포의 자동 실행까지 구현된 공용 launcher는 없으며, 계산 예제 통과와 실제 시뮬레이션 검증은 구분한다.

## 2. Teacher checkout의 진입점

아래 경로는 `/home/hwanhee/ksh/stage1_2_base_teacher` 기준이다.

| 작업 | 진입점 |
| --- | --- |
| Table 1 Independent/Mixed·2/4/8/16명 | `eval/run_stage2_scalability.py`, `stage2_scalability_eval.py` |
| 논문 그룹/행동 샘플러·5명 이상 배치·토큰 예산 | `eval/stage2_protocol_sampling.py`, `stage2_scalability_layout.py` |
| XY100%·distractor 물리 제거/관측 유지 | `eval/stage2_xy100.py`, `stage2_distractor_physics.py`, `stage2_full_observation.py` |
| SCT 원본 정책 입력 변환·native 행동 검증 | `eval/stage1_protocol_policy.py` |
| Table 2 단일 Enable 쌍·Coupled 그룹·원시 기록 | `eval/run_table2.py`, `table2_eval.py`, `table2_sampling.py` |
| SR·낙상·Point/Region Error | `tokenhsi/learning/multi_agent/formal_eval.py`, `tokenhsi/utils/evaluation_protocol.py`, `evaluation_errors.py` |
| 최신 t_coop·Jerk CPU 재집계 | `eval/reaggregate_coupled_cooperation.py`, `coupled_cooperation_metrics.py` |
| Bias 추론 제거 | `eval/run_bias_ablation.py`, `bias_ablation_eval.py`, `bias_ablation.py` |
| 전체 실행 안내 | `eval/README.md` |

`run_table2.py`의 기존 기본 checkpoint는 V2이며 `table2_metrics.py`가 실시간 출력하는6cm/구간 내 접촉률은 과거 지표다. 원시 기록·SR·Time은 재사용하고, **현재 t_coop/Jerk는 별도 재집계의20cm 조건**을 사용한다. 기존 summary 숫자를 최신 지표로 바꿔 부르지 않는다.

## 3. 확인된 실행 명령

Conda 환경은 `tokenhsi`. 학습·viewer·기존 프로세스를 종료하지 않는다. GPU가 필요한 새 실행은 `nvidia-smi`로 먼저 점유를 확인하고 이 실험에서는6/7만 사용한다. 결과·로그는 각 레포의 `output_etc/check/`에 둔다.

### CPU 계산 예제 — 어느 레포에서든 실행

```bash
CUDA_VISIBLE_DEVICES='' /home/hwanhee/anaconda3/envs/tokenhsi/bin/python \
  /home/hwanhee/ksh/evaluation_protocol/checks/verify.py
```

패키지를 다른 위치에 복사해도 `checks/verify.py`는 패키지 내부 파일만 참조한다. NumPy·PyTorch가 필요하며 Isaac Gym·checkpoint는 필요하지 않다.

### 기존 네 모델의 공동 PUSH 기록 재집계

```bash
cd /home/hwanhee/ksh/stage1_2_base_teacher
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python \
  eval/reaggregate_coupled_cooperation.py --workers 3
```

현재 helper는 고정된 네 모델·36개 기록을 읽고20/50cm 두 조건을 산출한다. **공용 규약 주 결과는20cm**이며50cm는 과거 민감도 비교다. 이 명령은 새 checkpoint 평가가 아니다. 원본 raw 기록은 읽기만 하고 별도 결과 폴더에 산출한다.

### 기존 Table 1/2 suite 이어 실행

```bash
cd /home/hwanhee/ksh/stage1_2_base_teacher
/home/hwanhee/anaconda3/envs/tokenhsi/bin/python \
  output_etc/check/v3_2000_tables12_seeds167_20261110/run_suite.py

cd /home/hwanhee/ksh/sct_coopft
/home/hwanhee/anaconda3/envs/tokenhsi/bin/python \
  output_etc/check/sct_coopft_4000_tables12_seeds167_20261111/run_suite.py
```

완료된 검증 결과를 재사용하는 기존 고정 suite다. 같은 큐가 실행 중일 때 중복 실행하지 않는다. 새 정책은 checkpoint·출력·reference를 명시한 별도 실행으로 분리한다. 일반 `eval/run_table2.py`의 기본값을 Ours V3라고 가정하지 않는다.

### 기존 bias 추론 제거 suite

```bash
cd /home/hwanhee/ksh/stage1_2_base_teacher
/home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/run_bias_ablation.py \
  --output output_etc/check/bias_ablation_20261110 --seeds 1 6 7 --workers-per-gpu 5
```

GPU6/7에서 실행하며 기존 suite는 완료됐다. 같은 checkpoint의 bias 출력만 제거하고 fusion·관계 할당을 유지한다.

## 4. 현재 실험 checkpoint·결과

현재 cohort: **512환경·seed1/6/7·900step·XY100%·M=2N**. Seed1/6/7은 실행 이력으로만 기록한다.

| 모델 | Checkpoint |
| --- | --- |
| SCT / SCT + FSM | `/home/hwanhee/ksh/stage1_2_base_teacher/pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth` |
| Ours V3 2k | `/home/hwanhee/ksh/stage1_2_base_teacher/output/stage2_distill_900_v3/Stage2Distill900V3_09-20-41-11/nn/Stage2Distill900V3_00002000.pth` |
| SCT + Coop. FT PPO4k | `/home/hwanhee/ksh/sct_coopft/output/sct_coopft/SCTCoopFT_10-22-24-41/nn/SCTCoopFT_00004000.pth` |

| 산출물 | 경로 |
| --- | --- |
| SCT/FSM/Ours Table 1/2·원시 궤적 | [원래 suite](../stage1_2_base_teacher/output_etc/check/v3_2000_tables12_seeds167_20261110/report.md) |
| Coop. FT Table 1/2·원시 궤적 | [Coop. FT suite](../sct_coopft/output_etc/check/sct_coopft_4000_tables12_seeds167_20261111/report.md) |
| 네 모델 최신 t_coop/Jerk·20cm 주 조건 | [재집계·인원/seed별 표](../stage1_2_base_teacher/output_etc/check/coupled_cooperation_four_models_20261111/report.md) |
| 최신 지표의 계산·기록 보존 검증 | [verification.json](../stage1_2_base_teacher/output_etc/check/coupled_cooperation_four_models_20261111/verification.json) |
| Bias 추론 제거 | [비교표](../stage1_2_base_teacher/output_etc/check/bias_ablation_20261110/report.md) |

**FSM 기존 기록의 알려진 문제:** reset 관측과 첫 정책 입력 사이에 관측을 재계산해 다른 rigid-body 상태를 읽는 불일치가 확인됐다. Independent에서도 freeze/release가0인데 행동이 달라졌다. 기존 FSM 수치는 이 문제가 남은 참고값이며 후처리로 수정되지 않는다. 새 공식 비교에는 첫 실제 정책 입력 검증을 포함한 수정·재평가가 필요하다. SCT/Full·Coop. FT 전체가 이 문제를 공유한다고 일반화하지 않는다.

## 5. 다른 레포 연결 절차

1. [README](README.md)의 공통 초기 장면·물성·자산을 복원하거나 동일 snapshot을 불러온다.
2. 레포의 task/group/token ID를 행동 이름·역할·원래 참여자 집합·실제 물리 상자로 변환한다.
3. 실제 첫 정책 입력을 확인하고 평균 행동·고정 정규화로 첫 episode를 기록한다.
4. 원시 상태를 공통 데이터 계약으로 내보낸다. 성공·Error는float64, 접촉·Jerk는float32 규칙을 지킨다.
5. [계산 예제](checks/README.md)와 CPU 재채점·초기 상태·attention·가중치 검증을 수행한다.
6. 프로토콜 ID·snapshot/checkpoint SHA·seed·실행 버전·지원/미지원 항목을 결과에 저장한다.

이 안내의 서버 경로는 위치 정보다. 프로토콜 자체의 계산 정의는 이 경로에 의존하지 않는다.
