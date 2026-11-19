# 지연 대응 파일럿

GPU6, 64환경, seed1, 35초(1050step). SCT26k / SCT+FSM26k / Coop. FT7k / Ours V4 2k를 비교한다. AT→SIT 2명은 선행자, 공동 PUSH3명은 장면 ID modulo3으로 고른 한 명을0/2/4초 정지시킨다. 총24회이며 추가 정식 반복은 없다.

`freeze.py`는 상위 `controller.py`의 root/DOF/몸체 상태 투영을 재사용한다. 강제 지연과 FSM 고유 대기의 합집합만 고정하고, FSM의 기존 predecessor 성공/전원 근접 경험에 따른 일회 해제는 유지한다. 외부 지연을 해제하더라도 FSM 대기가 남으면 계속 고정한다. 실제 kinematic actor는 아니며 충돌은 유지하고 상자 상태는 직접 보정하지 않는다.

네 모델은 같은64환경 reference를 사용한다. 모든 조건에서 모든 사람의 초기 root/DOF/몸체 속도를0으로 통일하고, 첫 몸체 관측을 reset의 kinematic pose에 맞춰 FSM 및 외부 정지가 첫 관측을 다르게 만드는 문제를 방지한다. 원래 샘플의 위치·자세·물체·관계·목표는 유지한다. 따라서 이전 논문 평가의 raw rollout과 같은 실험으로 합치지 않는다. 정지 시간·해제 신호는 정책 관측에 추가하지 않는다.

새로운64환경 reference는 기존 Table2 샘플러와 seed1로 생성한다. 기존512환경 기록의 앞64개를 잘라 쓴 것은 아니다. 모델/지연 간에는 같은 실제 초기 snapshot과 관측임을 검증한다. SCT/FSM은 기존 goal-aligned alias·Stage1 native 입력 변환을 사용한다. Coop. FT는 외부7k checkpoint의 native builder/MeanCooperation 구현을 읽기 전용으로 로드하고, 공통 물리 환경에서 strict load한다.

## 실행

저장소 루트에서 conda `tokenhsi`의 Python을 사용한다. Supervisor는 CPU에서 실행하고 자식이 `runtime_env.sh`로 GPU6을 설정한다. GPU별 다른 작업은 보존하며 작업당6GB 시작 예약과12GB 여유를 둔다. 사용자 요청에 따라 여러 작업을 병렬 실행한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python FSM/dct/run.py --phase reference --workers 2

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python FSM/dct/run.py --phase development --workers 8

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python FSM/dct/analyze.py --development

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python FSM/dct/run.py --phase formal --workers 12

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python FSM/dct/analyze.py
```

Development는4초 정지/해제를 확인하는135step8회이며 성능 비교에 포함하지 않는다. Supervisor 실패 후 재개는 관련 자식이 종료된 것을 확인한 뒤 수행한다. 완료 작업은 재사용하고 실패 폴더는 `failed_setup/`에 보존한다.

## 기록과 해석

출력: `output_etc/check/dct_delay_env64_seed1_20261115/`. `status.json`은 진행 상태, `manifest.json`은 출처, `rollouts/`는 원시 기록, `analysis/`는 episode별 통계와 시간별 곡선이다. 성공 후 reset 없이 기록하며 전원 낙상만 조기 종료한다.

- 모든 rollout에서30초/35초/30+지연초 창을 CPU 채점한다. 실패는 성공 시간 평균에서 제외하고 성공률·건수를 병기한다. 단일 seed이므로 seed 표준편차를 계산하지 않는다.
- Enabling은 AT/Group SR, 초기 대비 AT 직전 root→예정 목표 거리 감소율, root 원 반경0.35m/초기 yaw 물체 영역+margin0.10m 점유율, Post-AT 시간을 사용한다.
- Coupled는 Group SR, 기존20cm 시작·손 표면거리12cm·기록 전체 누적 t_coop과 NaN/clip 건수, 지연 대상의 최초 유효 양손 근접 시점, 그 이전 상자의 최대 XY 변위를 사용한다. 미근접 episode는 근접 시점/변위 평균에서 제외하고 건수를 병기한다. 먼저 미는 것이 반드시 실패를 의미하지 않는다.
- 몸체 위치·접촉력도 저장한다. Root 점유율/손 근접을 실제 충돌·유효 미는 힘으로 해석하지 않는다.

[결과](../../output_etc/check/dct_delay_env64_seed1_20261115/report.md) · [해석](../../output_etc/check/dct_delay_env64_seed1_20261115/interpretation.md) · [CSV](../../output_etc/check/dct_delay_env64_seed1_20261115/summary.csv) · [검증](../../output_etc/check/dct_delay_env64_seed1_20261115/final_audit.json) · [진행](../../output_etc/check/dct_delay_env64_seed1_20261115/status.json)

## TokenHSI40.5k 추가

`tokenhsi_eval.py`는 외부 `tokenhsi_push/scripts/tokenhsi_table_policy.py`의 원본397D 변환과 strict checkpoint 로드를 읽기 전용으로 재사용한다. 공통 환경의 초기 상태·속도·정지·성공 기준은 같고 실제 행동은 TokenHSI가 생성한다. 첫 공통 관측은 `first_policy_input.pt`, 실제397D 입력/정규화/행동은 `tokenhsi_first_input.pt`에 기록한다. TokenHSI의 자기 상태/담당 과제 입력에는 타인·distractor token이 들어가지 않는 구조 차이를 유지한다.

기존24조건은 보존하고 같은64환경 reference에서 TokenHSI의6조건만 추가한다. 출력은 `output_etc/check/dct_delay_tokenhsi_40500_env64_seed1_20261115/`이며 통합 표/CSV에는5모델30조건을 포함한다. 개발2회 후 정식6회를 GPU6에서 최대6개 병렬 실행한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python FSM/dct/run_tokenhsi.py --phase development

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python FSM/dct/analyze_tokenhsi.py --development

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python FSM/dct/run_tokenhsi.py --phase formal

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python FSM/dct/analyze_tokenhsi.py
```

[5모델 통합 결과](../../output_etc/check/dct_delay_tokenhsi_40500_env64_seed1_20261115/report.md) · [검증](../../output_etc/check/dct_delay_tokenhsi_40500_env64_seed1_20261115/completion_verification.json) · [진행](../../output_etc/check/dct_delay_tokenhsi_40500_env64_seed1_20261115/status.json)
